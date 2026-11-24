#!/usr/bin/env python3
"""Run Prompt Diffusion on frozen VisualCloze P=2 episodes using TPU/XLA."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import socket
import time
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image
from safetensors import safe_open

from prompt_diffusion_loader import (
    MODEL_REVISION,
    NEGATIVE_PROMPT,
    PROMPT,
    SOURCE_COMMIT,
    load_runtime_components,
)

PACKS = {
    "48e062452574ff0ded61c72664445d687ecd1c02": {
        "name": "test_s64_px256_v5", "records": 960, "episodes": 3840,
        "tasks": 15, "per_task": 64, "supervision": "v5",
    },
    "9e77a92f82d0e08c12f540d5c15c2b15e6c503a2": {
        "name": "val_s32_px256_lineageA", "records": 128, "episodes": 512,
        "tasks": 4, "per_task": 32,
        "supervision": "lineage-A native; no v5 recoding",
    },
}
CONDITIONS = {"real", "scrambled", "grey", "oracle"}
IMAGE_SIZE = 256
LATENT_SIZE = 32
NATIVE_PER_EPISODE = 2
EPISODE_BATCH = 4
INFERENCE_STEPS = 50
GUIDANCE_SCALE = 7.5


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def json_dump(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def validate_manifest(manifest: dict, run_dir: Path) -> dict:
    pack_id = str(manifest.get("pack_id"))
    if pack_id not in PACKS:
        raise SystemExit(f"unapproved pack_id: {pack_id!r}")
    contract = PACKS[pack_id]
    episodes = list(manifest.get("episodes", []))
    if manifest.get("records") != contract["records"] or len(episodes) != contract["episodes"]:
        raise SystemExit("pack record/episode count mismatch")
    if int(manifest.get("seed", -1)) != 42 or int(manifest.get("p", -1)) != 2:
        raise SystemExit("seed/P contract mismatch")
    if manifest.get("seed_rule") != "[seed, task_seed(task), q, P]" or manifest.get("demo_source_policy") != "same":
        raise SystemExit("episode construction contract mismatch")
    if {row.get("condition") for row in episodes} != CONDITIONS:
        raise SystemExit("condition set changed")
    if len({row.get("episode_id") for row in episodes}) != len(episodes):
        raise SystemExit("duplicate episode ids")
    task_counts = Counter(row["task"] for row in episodes)
    if len(task_counts) != contract["tasks"] or set(task_counts.values()) != {contract["per_task"] * 4}:
        raise SystemExit("task count changed")
    if len(episodes) % EPISODE_BATCH:
        raise SystemExit("episode count is not divisible by fixed batch")
    missing = [row["file"] for row in episodes if not (run_dir / row["file"]).is_file()]
    if missing:
        raise SystemExit(f"missing {len(missing)} episode files; first={missing[0]}")
    return contract


def to_chw(value: np.ndarray) -> np.ndarray:
    array = np.asarray(value, np.uint8)
    if array.shape != (IMAGE_SIZE, IMAGE_SIZE, 3):
        raise SystemExit(f"unexpected image shape: {array.shape}")
    return array.astype(np.float32).transpose(2, 0, 1) / 255.0


def noise_seed(episode: dict) -> int:
    # Model noise is deliberately independent of condition so all four arms
    # of a task/query share the same stochastic input.
    key = f"42|{episode['task']}|{int(episode['q'])}|2".encode()
    return int.from_bytes(hashlib.sha256(key).digest()[:8], "little") & 0x7FFF_FFFF_FFFF_FFFF


def load_episode_batch(run_dir: Path, episodes: list[dict]):
    demo_inputs, demo_targets, queries, noises, noise_rows = [], [], [], [], []
    import torch

    for episode in episodes:
        path = run_dir / episode["file"]
        if sha256(path) != episode["sha256"]:
            raise SystemExit(f"episode SHA-256 mismatch: {episode['episode_id']}")
        with np.load(path, allow_pickle=False) as packed:
            inputs = np.asarray(packed["demo_inputs"], np.uint8)
            targets = np.asarray(packed["demo_targets"], np.uint8)
            query = np.asarray(packed["query_input"], np.uint8)
        if inputs.shape != (2, 256, 256, 3) or targets.shape != inputs.shape:
            raise SystemExit(f"P=2 array shape changed: {episode['episode_id']}")
        for index in range(2):
            demo_inputs.append(to_chw(inputs[index]))
            demo_targets.append(to_chw(targets[index]))
            queries.append(to_chw(query))
        seed = noise_seed(episode)
        generator = torch.Generator(device="cpu").manual_seed(seed)
        noise = torch.randn((1, 4, LATENT_SIZE, LATENT_SIZE), generator=generator, dtype=torch.float32)
        noises.append(noise.repeat(2, 1, 1, 1))
        noise_rows.append({
            "seed": seed,
            "sha256": hashlib.sha256(noise.numpy().tobytes()).hexdigest(),
        })
    return (
        np.stack(demo_inputs), np.stack(demo_targets), np.stack(queries),
        torch.cat(noises, dim=0).numpy(), noise_rows,
    )


def ddim_coefficients(scheduler, steps: int):
    scheduler.set_timesteps(steps, device="cpu")
    timestep_values = [int(value) for value in scheduler.timesteps.tolist()]
    alpha_values = scheduler.alphas_cumprod.to(dtype=None).cpu()
    final_alpha = float(scheduler.final_alpha_cumprod)
    stride = scheduler.config.num_train_timesteps // steps
    rows = []
    for timestep in timestep_values:
        previous = timestep - stride
        alpha = float(alpha_values[timestep])
        alpha_previous = float(alpha_values[previous]) if previous >= 0 else final_alpha
        rows.append((timestep, alpha, alpha_previous))
    return rows


def predict_batch(components, arrays, embeddings, device, mesh):
    import torch
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.spmd as xs

    controlnet, unet, vae, scheduler = components
    demo_input_np, demo_target_np, query_np, noise_np, noise_rows = arrays
    rows = EPISODE_BATCH * NATIVE_PER_EPISODE
    if demo_input_np.shape != (rows, 3, 256, 256):
        raise SystemExit(f"unexpected native batch: {demo_input_np.shape}")
    dtype = torch.bfloat16
    demo_input = torch.from_numpy(demo_input_np).to(device=device, dtype=dtype)
    demo_target = torch.from_numpy(demo_target_np).to(device=device, dtype=dtype)
    query = torch.from_numpy(query_np).to(device=device, dtype=dtype)
    latents = torch.from_numpy(noise_np).to(device=device, dtype=dtype)
    positive, negative = embeddings
    positive = positive.repeat(rows, 1, 1).to(device=device, dtype=dtype)
    negative = negative.repeat(rows, 1, 1).to(device=device, dtype=dtype)
    image_pair = torch.cat([demo_input, demo_target], dim=1)

    for tensor in (demo_input, demo_target, query, latents, positive, negative, image_pair):
        xs.mark_sharding(tensor, mesh, ("data",) + (None,) * (tensor.ndim - 1))
    prompt_embeds = torch.cat([negative, positive], dim=0)
    query_guided = torch.cat([query, query], dim=0)
    pair_guided = torch.cat([image_pair, image_pair], dim=0)
    xs.mark_sharding(prompt_embeds, mesh, ("data", None, None))
    xs.mark_sharding(query_guided, mesh, ("data", None, None, None))
    xs.mark_sharding(pair_guided, mesh, ("data", None, None, None))

    with torch.no_grad():
        for timestep, alpha, alpha_previous in ddim_coefficients(scheduler, INFERENCE_STEPS):
            latent_input = torch.cat([latents, latents], dim=0)
            xs.mark_sharding(latent_input, mesh, ("data", None, None, None))
            timestep_tensor = torch.tensor(timestep, device=device, dtype=torch.int64)
            down, mid = controlnet(
                latent_input,
                timestep_tensor,
                encoder_hidden_states=prompt_embeds,
                controlnet_query_cond=query_guided,
                controlnet_cond=pair_guided,
                conditioning_scale=1.0,
                guess_mode=False,
                return_dict=False,
            )
            noise_guided = unet(
                latent_input,
                timestep_tensor,
                encoder_hidden_states=prompt_embeds,
                down_block_additional_residuals=down,
                mid_block_additional_residual=mid,
                return_dict=False,
            )[0]
            unconditioned, conditioned = noise_guided.chunk(2)
            noise_prediction = unconditioned + GUIDANCE_SCALE * (conditioned - unconditioned)
            beta = 1.0 - alpha
            original = (latents - math.sqrt(beta) * noise_prediction) / math.sqrt(alpha)
            direction = math.sqrt(1.0 - alpha_previous) * noise_prediction
            latents = math.sqrt(alpha_previous) * original + direction
            xs.mark_sharding(latents, mesh, ("data", None, None, None))
            xm.mark_step()
        decoded = vae.decode(latents / float(vae.config.scaling_factor), return_dict=False)[0]
        decoded = (decoded / 2.0 + 0.5).clamp(0.0, 1.0)
        xm.mark_step()
        decoded = decoded.float().cpu().reshape(EPISODE_BATCH, 2, 3, 256, 256).mean(dim=1)
        images = (decoded * 255.0).round().clamp(0, 255).to(torch.uint8)
    return images.permute(0, 2, 3, 1).numpy(), noise_rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--prompt-embeddings", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--expected-devices", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--log-every", type=int, default=8)
    args = parser.parse_args()
    if args.expected_devices != 8:
        raise SystemExit("pinned target requires exactly 8 TPU devices")

    import torch
    import torch_xla
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.spmd as xs
    import torch_xla.runtime as xr

    args.cache_dir.mkdir(parents=True, exist_ok=True)
    xr.initialize_cache(str(args.cache_dir.resolve()), readonly=False)
    xr.use_spmd()
    runtime_devices = int(xr.global_runtime_device_count())
    addressable = int(xr.addressable_runtime_device_count())
    backend = str(xr.device_type())
    if backend.upper() != "TPU" or runtime_devices != 8 or addressable != 8:
        raise SystemExit(f"refusing fallback: backend={backend!r}, runtime={runtime_devices}, addressable={addressable}")
    mesh = xs.Mesh(np.arange(8), (8,), ("data",))
    xs.set_global_mesh(mesh)
    device = xm.xla_device()
    evidence = {
        "backend": backend, "device": str(device),
        "runtime_device_count": runtime_devices,
        "addressable_runtime_device_count": addressable,
        "spmd_mesh_device_ids": list(range(8)),
        "torch": torch.__version__, "torch_xla": getattr(torch_xla, "__version__", "unknown"),
        "host": socket.gethostname(), "tpu_name": os.environ.get("EXPECTED_TPU_NAME"),
        "tpu_zone": os.environ.get("EXPECTED_TPU_ZONE"),
    }
    print("TPU_BACKEND_EVIDENCE " + json.dumps(evidence, sort_keys=True), flush=True)

    run_dir = args.run_dir.resolve()
    manifest = json.loads((run_dir / "manifest.json").read_text())
    contract = validate_manifest(manifest, run_dir)
    episodes = list(manifest["episodes"])
    if args.limit:
        if args.limit % EPISODE_BATCH:
            raise SystemExit("--limit must be divisible by 4")
        episodes = episodes[:args.limit]

    with safe_open(str(args.prompt_embeddings), framework="pt", device="cpu") as handle:
        metadata = handle.metadata()
        if metadata.get("prompt") != PROMPT or metadata.get("negative_prompt") != NEGATIVE_PROMPT:
            raise SystemExit("fixed prompt metadata mismatch")
        embeddings = (handle.get_tensor("positive"), handle.get_tensor("negative"))
    if any(tuple(value.shape) != (1, 77, 768) for value in embeddings):
        raise SystemExit("fixed prompt embedding shape mismatch")
    embedding_sha = sha256(args.prompt_embeddings)

    runtime = load_runtime_components(args.source_dir, args.model_dir, dtype=torch.bfloat16)
    controlnet, unet, vae, scheduler = runtime
    runtime = (
        controlnet.eval().to(device), unet.eval().to(device), vae.eval().to(device), scheduler,
    )
    xm.mark_step()

    predictions_dir = run_dir / "predictions"
    predictions_dir.mkdir(parents=True, exist_ok=True)
    partial_path = run_dir / "inference.partial.json"
    final_path = run_dir / "inference.json"
    previous = partial_path if partial_path.exists() else final_path
    expected_resume = {
        "pack_id": manifest["pack_id"], "source_commit": SOURCE_COMMIT,
        "model_revision": MODEL_REVISION, "prompt_embeddings_sha256": embedding_sha,
        "adapter": "p2_mean_of_two_native_prompt_diffusion_p1_v1",
        "backend": "TPU", "runtime_device_count": 8,
        "episode_batch": EPISODE_BATCH, "inference_steps": INFERENCE_STEPS,
        "guidance_scale": GUIDANCE_SCALE,
    }
    if args.resume and previous.exists():
        log = json.loads(previous.read_text())
        mismatch = {key: (log.get(key), value) for key, value in expected_resume.items() if log.get(key) != value}
        if mismatch:
            raise SystemExit(f"resume metadata mismatch: {mismatch}")
        seen = set()
        for row in log.get("results", []):
            episode_id = row.get("episode_id")
            path = run_dir / str(row.get("prediction"))
            if episode_id in seen or not path.is_file() or sha256(path) != row.get("prediction_sha256"):
                raise SystemExit(f"invalid resume row: {episode_id}")
            seen.add(episode_id)
        log.pop("finished_utc", None)
    else:
        log = {
            "schema": "prompt_diffusion_visualcloze_xla_inference_v1",
            "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "pack_name": contract["name"], "pack_id": manifest["pack_id"],
            "supervision": contract["supervision"], "model": "Prompt Diffusion",
            "source_commit": SOURCE_COMMIT, "model_revision": MODEL_REVISION,
            "prompt": PROMPT, "negative_prompt": NEGATIVE_PROMPT,
            "prompt_policy": "constant task-neutral prompt; no task or class names",
            "prompt_embeddings_sha256": embedding_sha,
            "adapter": "p2_mean_of_two_native_prompt_diffusion_p1_v1",
            "adapter_detail": "Run the two frozen demonstrations as two official native P=1 Prompt Diffusion forwards with identical per-query initial noise, then mean decoded float outputs before uint8.",
            "noise_policy": "CPU torch generator seeded by SHA256('42|task|q|2'); independent of condition and duplicated for both demonstrations",
            "episode_batch": EPISODE_BATCH, "native_batch": 8,
            "inference_steps": INFERENCE_STEPS, "scheduler": "DDIMScheduler",
            "guidance_scale": GUIDANCE_SCALE, **evidence, "results": [],
        }
    complete = {row["episode_id"] for row in log["results"]}
    json_dump(partial_path, log)

    batches = [episodes[index:index + EPISODE_BATCH] for index in range(0, len(episodes), EPISODE_BATCH)]
    for batch_index, batch in enumerate(batches, 1):
        already = [row["episode_id"] in complete for row in batch]
        if all(already):
            continue
        if any(already):
            raise SystemExit(f"partial resume batch: {[row['episode_id'] for row in batch]}")
        arrays = load_episode_batch(run_dir, batch)
        started = time.perf_counter()
        predictions, noise_rows = predict_batch(runtime, arrays, embeddings, device, mesh)
        elapsed = time.perf_counter() - started
        for episode, prediction, noise in zip(batch, predictions, noise_rows):
            output = predictions_dir / f"{episode['episode_id']}__pred256.png"
            temporary = output.with_suffix(output.suffix + ".tmp")
            Image.fromarray(prediction, "RGB").save(temporary, format="PNG")
            os.replace(temporary, output)
            log["results"].append({
                "episode_id": episode["episode_id"], "prediction": str(output.relative_to(run_dir)),
                "prediction_sha256": sha256(output), "batch_seconds": elapsed,
                "episode_batch": EPISODE_BATCH, "native_prompt_count": 2,
                "noise_seed": noise["seed"], "noise_sha256": noise["sha256"],
            })
            complete.add(episode["episode_id"])
        json_dump(partial_path, log)
        if batch_index == 1 or batch_index % args.log_every == 0 or len(log["results"]) == len(episodes):
            print(f"{contract['name']}: {len(log['results'])}/{len(episodes)} batch_seconds={elapsed:.3f} last={batch[-1]['episode_id']}", flush=True)
    log["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    json_dump(final_path, log)
    partial_path.unlink(missing_ok=True)
    print(f"COMPLETE {contract['name']} {len(log['results'])}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
