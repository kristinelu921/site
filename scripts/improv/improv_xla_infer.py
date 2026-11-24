#!/usr/bin/env python3
"""Run pinned IMProv on frozen VisualCloze P=2 episodes with PyTorch/XLA."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import time
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image
from safetensors import safe_open

from improv_loader import PROMPT, load_components

SOURCE_COMMIT = "41a8fce0723a4d8969b5cc23cf867912f799c68a"
MODEL_REVISION = "bc9d733ae2fc403600f87f8374b2eb544e6bd6dd"
VQ_REVISION = "f61ca369a6ac0fc3f45f335395320dee65b5a296"
PACKS = {
    "48e062452574ff0ded61c72664445d687ecd1c02": {"name": "test_s64_px256_v5", "records": 960, "episodes": 3840, "tasks": 15, "per_task": 64, "supervision": "v5"},
    "9e77a92f82d0e08c12f540d5c15c2b15e6c503a2": {"name": "val_s32_px256_lineageA", "records": 128, "episodes": 512, "tasks": 4, "per_task": 32, "supervision": "lineage-A native; no v5 recoding"},
}
CONDITIONS = {"real", "scrambled", "grey", "oracle"}
NATIVE_TILE = 111
CANVAS = 224


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


def validate_manifest(manifest: dict, run_dir: Path, episode_batch: int) -> dict:
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
    if len(episodes) % episode_batch:
        raise SystemExit("episode count must divide evenly by --episode-batch")
    missing = [row["file"] for row in episodes if not (run_dir / row["file"]).is_file()]
    if missing:
        raise SystemExit(f"missing {len(missing)} episode files; first={missing[0]}")
    return contract


def resize_tile(array: np.ndarray) -> np.ndarray:
    image = Image.fromarray(np.asarray(array, np.uint8), "RGB")
    image = image.resize((NATIVE_TILE, NATIVE_TILE), Image.Resampling.BICUBIC)
    return np.asarray(image, np.float32).transpose(2, 0, 1) / 255.0


def load_episode_batch(run_dir: Path, episodes: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    canvases = []
    masks = []
    for episode in episodes:
        path = run_dir / episode["file"]
        if sha256(path) != episode["sha256"]:
            raise SystemExit(f"episode SHA-256 mismatch: {episode['episode_id']}")
        with np.load(path, allow_pickle=False) as packed:
            demo_inputs = np.asarray(packed["demo_inputs"], np.uint8)
            demo_targets = np.asarray(packed["demo_targets"], np.uint8)
            query_input = np.asarray(packed["query_input"], np.uint8)
        if demo_inputs.shape != (2, 256, 256, 3) or demo_targets.shape != demo_inputs.shape or query_input.shape != (256, 256, 3):
            raise SystemExit(f"P=2 array shape changed: {episode['episode_id']}")
        query = resize_tile(query_input)
        for demo_input, demo_target in zip(demo_inputs, demo_targets):
            canvas = np.ones((3, CANVAS, CANVAS), np.float32)
            canvas[:, :NATIVE_TILE, :NATIVE_TILE] = resize_tile(demo_input)
            canvas[:, :NATIVE_TILE, -NATIVE_TILE:] = resize_tile(demo_target)
            canvas[:, -NATIVE_TILE:, :NATIVE_TILE] = query
            canvas[:, -NATIVE_TILE:, -NATIVE_TILE:] = 0.0
            mask = np.zeros((1, CANVAS, CANVAS), np.float32)
            mask[:, -NATIVE_TILE:, -NATIVE_TILE:] = 1.0
            canvases.append(canvas)
            masks.append(mask)
    return np.stack(canvases), np.stack(masks)


def one_step_predict(components, canvas_np, mask_np, prompt_embedding, device, mesh, episode_batch):
    import torch
    import torch.nn.functional as functional
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.spmd as xs

    image_encoder, image_decoder, mask_model, _scheduler = components
    rows = episode_batch * 2
    if canvas_np.shape != (rows, 3, CANVAS, CANVAS) or mask_np.shape != (rows, 1, CANVAS, CANVAS):
        raise SystemExit(f"unexpected native batch shapes: {canvas_np.shape}, {mask_np.shape}")
    canvas = torch.from_numpy(canvas_np).to(device)
    input_mask = torch.from_numpy(mask_np).to(device)
    prompt = prompt_embedding.repeat(rows, 1, 1).to(device)
    xs.mark_sharding(canvas, mesh, ("data", None, None, None))
    xs.mark_sharding(input_mask, mesh, ("data", None, None, None))
    xs.mark_sharding(prompt, mesh, ("data", None, None))

    # These indices are determined solely by the official fixed 224px grid.
    # Compute them on CPU to avoid graph-breaking scalar extraction in the
    # upstream scheduler while preserving its exact argsort/gather contract.
    mask_small_cpu = functional.interpolate(torch.from_numpy(mask_np[:1]), size=(14, 14), mode="bilinear")
    mask_flat_cpu = (mask_small_cpu > 0).to(torch.float32).reshape(1, 196)
    ids_shuffle_cpu = torch.argsort(mask_flat_cpu, dim=1)
    ids_restore = torch.argsort(ids_shuffle_cpu, dim=1).repeat(rows, 1).to(device)
    len_keep = int(196 * (1.0 - float(mask_flat_cpu.mean())))
    ids_keep = ids_shuffle_cpu[:, :len_keep].repeat(rows, 1).to(device)
    xs.mark_sharding(ids_restore, mesh, ("data", None))
    xs.mark_sharding(ids_keep, mesh, ("data", None))

    with torch.no_grad():
        masked_canvas = canvas * (input_mask < 0.5)
        image_latents = image_encoder(masked_canvas)
        image_latents = image_latents.permute(0, 2, 3, 1).reshape(rows, 196, 1024)
        masked_latents = torch.gather(
            image_latents, 1, ids_keep.unsqueeze(-1).repeat(1, 1, 1024)
        )
        logits = mask_model(masked_latents, ids_restore=ids_restore, encoder_hidden_states=prompt)
        image_tokens = logits.argmax(dim=-1).reshape(rows, 14, 14)
        decoded = image_decoder(image_tokens)
        query = decoded[:, :, -NATIVE_TILE:, -NATIVE_TILE:]
        query = functional.interpolate(query, size=(256, 256), mode="bicubic", align_corners=False)
        xm.mark_step()
        query = query.cpu().reshape(episode_batch, 2, 3, 256, 256).mean(dim=1)
        query = torch.clamp(query * 255.0, 0.0, 255.0).to(torch.uint8)
    return query.permute(0, 2, 3, 1).numpy()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--vq-dir", type=Path, required=True)
    parser.add_argument("--prompt-embedding", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--episode-batch", type=int, default=4)
    parser.add_argument("--expected-devices", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--log-every", type=int, default=8)
    args = parser.parse_args()
    if args.episode_batch != 4 or args.expected_devices != 8:
        raise SystemExit("pinned v5e-8 contract requires 4 episodes / 8 native P=1 rows")

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
    evidence = {"backend": backend, "device": str(device), "runtime_device_count": runtime_devices, "addressable_runtime_device_count": addressable, "spmd_mesh_device_ids": list(range(8)), "torch": torch.__version__, "torch_xla": getattr(torch_xla, "__version__", "unknown"), "host": socket.gethostname(), "tpu_name": os.environ.get("EXPECTED_TPU_NAME"), "tpu_zone": os.environ.get("EXPECTED_TPU_ZONE")}
    print("TPU_BACKEND_EVIDENCE " + json.dumps(evidence, sort_keys=True), flush=True)

    run_dir = args.run_dir.resolve()
    manifest = json.loads((run_dir / "manifest.json").read_text())
    contract = validate_manifest(manifest, run_dir, args.episode_batch)
    episodes = list(manifest["episodes"])
    if args.limit:
        if args.limit % args.episode_batch:
            raise SystemExit("--limit must be divisible by --episode-batch")
        episodes = episodes[:args.limit]

    with safe_open(str(args.prompt_embedding), framework="pt", device="cpu") as handle:
        if handle.metadata().get("prompt") != PROMPT:
            raise SystemExit("fixed prompt metadata mismatch")
        fixed_embedding = handle.get_tensor("embedding")
    if tuple(fixed_embedding.shape) != (1, 77, 1024):
        raise SystemExit("fixed prompt embedding identity mismatch")
    prompt_embedding_sha = sha256(args.prompt_embedding)
    components = load_components(args.source_dir, args.model_dir, args.vq_dir, load_text=False)
    image_encoder, image_decoder, mask_model, scheduler, _, _ = components
    runtime_components = [image_encoder.eval().to(device), image_decoder.eval().to(device), mask_model.eval().to(device), scheduler]
    xm.mark_step()

    predictions_dir = run_dir / "predictions"
    predictions_dir.mkdir(parents=True, exist_ok=True)
    partial_path = run_dir / "inference.partial.json"
    final_path = run_dir / "inference.json"
    previous = partial_path if partial_path.exists() else final_path
    expected_resume = {"pack_id": manifest["pack_id"], "source_commit": SOURCE_COMMIT, "model_revision": MODEL_REVISION, "vq_revision": VQ_REVISION, "prompt_embedding_sha256": prompt_embedding_sha, "adapter": "p2_mean_of_two_native_improv_p1_grids_v1", "backend": "TPU", "runtime_device_count": 8, "episode_batch": 4}
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
        log = {"schema": "improv_visualcloze_xla_inference_v1", "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "pack_name": contract["name"], "pack_id": manifest["pack_id"], "supervision": contract["supervision"], "model": "IMProv-v1-0", "source_commit": SOURCE_COMMIT, "model_revision": MODEL_REVISION, "vq_revision": VQ_REVISION, "prompt": PROMPT, "prompt_policy": "constant generic layout prompt; no task or class names", "prompt_embedding_sha256": prompt_embedding_sha, "adapter": "p2_mean_of_two_native_improv_p1_grids_v1", "adapter_detail": "Two official 224px P=1 grids per frozen P=2 episode; one-step argmax generation; crop each bottom-right query result; bicubic to 256; mean the two raw outputs before uint8.", "placement": "4 P=2 episodes = 8 native grids data-sharded over 8 TPU chips", "episode_batch": 4, **evidence, "results": []}
    complete = {row["episode_id"] for row in log["results"]}
    json_dump(partial_path, log)

    batches = [episodes[index:index + 4] for index in range(0, len(episodes), 4)]
    for batch_index, batch in enumerate(batches, 1):
        already = [row["episode_id"] in complete for row in batch]
        if all(already):
            continue
        if any(already):
            raise SystemExit(f"partial resume batch: {[row['episode_id'] for row in batch]}")
        arrays = load_episode_batch(run_dir, batch)
        started = time.perf_counter()
        predictions = one_step_predict(runtime_components, *arrays, fixed_embedding, device, mesh, 4)
        elapsed = time.perf_counter() - started
        for episode, prediction in zip(batch, predictions):
            output = predictions_dir / f"{episode['episode_id']}__pred256.png"
            temporary = output.with_suffix(output.suffix + ".tmp")
            Image.fromarray(prediction, "RGB").save(temporary, format="PNG")
            os.replace(temporary, output)
            log["results"].append({"episode_id": episode["episode_id"], "prediction": str(output.relative_to(run_dir)), "prediction_sha256": sha256(output), "batch_seconds": elapsed, "episode_batch": 4, "native_prompt_count": 2})
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
