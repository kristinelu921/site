#!/usr/bin/env python3
"""Painter native-P=1 ensemble inference for frozen P=4/8/16 TEST episodes."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

from painter_compat import install as install_import_shims


SOURCE_COMMIT = "cfec84587da2a2cc4e0b9acebe6fba1666e66477"
MODEL_REVISION = "97200aa029bab2e61ea5b4ce627534e35f152b23"
CHECKPOINT_SHA256 = "1bdd60ce013e82253e2a031726f36d07587947438b0e561d4338d6f0c1113f44"
CHECKPOINT_BYTES = 1_483_011_937
ARCH = "painter_vit_large_patch16_input896x448_win_dec64_8glb_sl1"
PACKS = {
    "48e062452574ff0ded61c72664445d687ecd1c02": {
        "name": "test_s64_px256_v5",
        "records": 960,
        "episodes": 3840,
        "tasks": 15,
        "per_task": 64,
        "supervision": "v5",
    },
    "9e77a92f82d0e08c12f540d5c15c2b15e6c503a2": {
        "name": "val_s32_px256_lineageA",
        "records": 128,
        "episodes": 512,
        "tasks": 4,
        "per_task": 32,
        "supervision": "lineage-A native; no v5 recoding",
    },
}
CONDITIONS = {"real", "scrambled", "grey", "oracle"}
IMAGENET_MEAN = np.asarray([0.485, 0.456, 0.406], np.float32)
IMAGENET_STD = np.asarray([0.229, 0.224, 0.225], np.float32)


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


def validate_manifest(manifest: dict, run_dir: Path, episode_batch: int) -> tuple[dict, int]:
    pack_id = str(manifest.get("pack_id"))
    if pack_id not in PACKS:
        raise SystemExit(f"unapproved pack_id: {pack_id!r}")
    contract = PACKS[pack_id]
    episodes = list(manifest.get("episodes", []))
    if manifest.get("records") != contract["records"] or len(episodes) != contract["episodes"]:
        raise SystemExit("pack record/episode count mismatch")
    prompt_count = int(manifest.get("p", -1))
    if int(manifest.get("seed", -1)) != 42 or prompt_count not in {4, 8, 16}:
        raise SystemExit("seed/P contract mismatch")
    if manifest.get("seed_rule") != "[seed, task_seed(task), q, P]":
        raise SystemExit("seed rule changed")
    if manifest.get("demo_source_policy") != "same":
        raise SystemExit("demonstrations are not same-source")
    if {row.get("condition") for row in episodes} != CONDITIONS:
        raise SystemExit("condition set changed")
    if len({row.get("episode_id") for row in episodes}) != len(episodes):
        raise SystemExit("duplicate episode ids")
    if len({row.get("task") for row in episodes}) != contract["tasks"]:
        raise SystemExit("task count changed")
    task_counts = Counter(row["task"] for row in episodes)
    if set(task_counts.values()) != {contract["per_task"] * len(CONDITIONS)}:
        raise SystemExit("per-task episode count changed")
    if len(episodes) % episode_batch:
        raise SystemExit("episode count must divide evenly by --episode-batch")
    missing = [row["file"] for row in episodes if not (run_dir / row["file"]).is_file()]
    if missing:
        raise SystemExit(f"missing {len(missing)} episode files; first={missing[0]}")
    return contract, prompt_count


def resize_rgb(array: np.ndarray, *, target: bool = False) -> np.ndarray:
    image = Image.fromarray(np.asarray(array, np.uint8), "RGB")
    if target:
        image = image.resize((448, 448), Image.Resampling.NEAREST)
    else:
        # Official Painter demo uses Pillow's default (nearest) for inputs.
        image = image.resize((448, 448))
    return np.asarray(image, np.float32) / 255.0


def load_episode_batch(run_dir: Path, episodes: list[dict], prompt_count: int) -> tuple[np.ndarray, np.ndarray]:
    input_rows: list[np.ndarray] = []
    target_rows: list[np.ndarray] = []
    for episode in episodes:
        episode_path = run_dir / episode["file"]
        if sha256(episode_path) != episode["sha256"]:
            raise SystemExit(f"episode SHA-256 mismatch: {episode['episode_id']}")
        with np.load(episode_path) as packed:
            demo_inputs = np.asarray(packed["demo_inputs"], np.uint8)
            demo_targets = np.asarray(packed["demo_targets"], np.uint8)
            query_input = np.asarray(packed["query_input"], np.uint8)
        if demo_inputs.shape != (prompt_count, 256, 256, 3) or demo_targets.shape != demo_inputs.shape:
            raise SystemExit(f"P={prompt_count} array shape changed: {episode['episode_id']}")
        query = resize_rgb(query_input)
        for prompt_input_u8, prompt_target_u8 in zip(demo_inputs, demo_targets):
            prompt_input = resize_rgb(prompt_input_u8)
            prompt_target = resize_rgb(prompt_target_u8, target=True)
            input_rows.append(np.concatenate((prompt_input, query), axis=0))
            # The query-target half is masked before the encoder consumes it;
            # the official demo fills it with the prompt target.
            target_rows.append(np.concatenate((prompt_target, prompt_target), axis=0))
    inputs = (np.stack(input_rows) - IMAGENET_MEAN) / IMAGENET_STD
    targets = (np.stack(target_rows) - IMAGENET_MEAN) / IMAGENET_STD
    return inputs, targets


def load_model(source_dir: Path, checkpoint_path: Path, device):
    import torch
    import torch_xla.core.xla_model as xm

    if checkpoint_path.stat().st_size != CHECKPOINT_BYTES:
        raise SystemExit("checkpoint byte-size mismatch")
    checkpoint_sha = sha256(checkpoint_path)
    if checkpoint_sha != CHECKPOINT_SHA256:
        raise SystemExit(f"checkpoint SHA-256 mismatch: {checkpoint_sha}")
    install_import_shims()
    sys.path.insert(0, str(source_dir / "Painter"))
    import models_painter  # noqa: PLC0415

    model = getattr(models_painter, ARCH)()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    message = model.load_state_dict(checkpoint["model"], strict=False)
    if message.missing_keys or message.unexpected_keys:
        raise SystemExit(
            f"checkpoint/model mismatch: missing={message.missing_keys}, "
            f"unexpected={message.unexpected_keys}"
        )
    del checkpoint
    model = model.eval().to(device)
    xm.mark_step()
    return model, checkpoint_sha


def predict_batch(model, inputs_np, targets_np, device, mesh, episode_batch: int, prompt_count: int) -> np.ndarray:
    import torch
    import torch.nn.functional as functional
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.spmd as xs

    expected_prompt_rows = episode_batch * prompt_count
    if inputs_np.shape != (expected_prompt_rows, 896, 448, 3):
        raise SystemExit(f"unexpected input shape: {inputs_np.shape}")
    x = torch.from_numpy(inputs_np).permute(0, 3, 1, 2).contiguous().to(device)
    y = torch.from_numpy(targets_np).permute(0, 3, 1, 2).contiguous().to(device)
    mask = torch.zeros(
        (expected_prompt_rows, model.patch_embed.num_patches), dtype=torch.bool, device=device
    )
    mask[:, model.patch_embed.num_patches // 2 :] = True
    xs.mark_sharding(x, mesh, ("data", None, None, None))
    xs.mark_sharding(y, mesh, ("data", None, None, None))
    xs.mark_sharding(mask, mesh, ("data", None))
    # torch.inference_mode creates version-counter-free inference tensors;
    # PyTorch/XLA 2.7 cannot expand Painter's mask token in that mode. no_grad
    # preserves the same model math and gradient-disabled evaluation contract.
    with torch.no_grad():
        latent = model.forward_encoder(x, y, mask)
        prediction = model.forward_decoder(latent)[:, :, 448:, :]
        xm.mark_step()
        # Official Painter post-processing begins after moving model output to
        # CPU. Average raw normalized native-P=1 outputs before all post-processing.
        prediction = prediction.cpu().reshape(episode_batch, prompt_count, 3, 448, 448).mean(dim=1)
        mean = torch.as_tensor(IMAGENET_MEAN)[None, :, None, None]
        std = torch.as_tensor(IMAGENET_STD)[None, :, None, None]
        prediction = torch.clamp((prediction * std + mean) * 255.0, 0.0, 255.0)
        prediction = functional.interpolate(prediction, (256, 256), mode="nearest")
    return prediction.permute(0, 2, 3, 1).to(torch.uint8).numpy()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--episode-batch", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--log-every", type=int, default=32)
    args = parser.parse_args()
    if args.episode_batch != 4:
        raise SystemExit("the pinned v4-8 SPMD contract requires --episode-batch 4")

    # Importing torch_xla and acquiring its device are intentionally inside
    # main: occupancy must be checked by the launch wrapper first.
    import torch
    import torch_xla
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.spmd as xs
    import torch_xla.runtime as xr

    args.cache_dir.mkdir(parents=True, exist_ok=True)
    xr.initialize_cache(str(args.cache_dir.resolve()), readonly=False)
    xr.use_spmd()
    runtime_devices = int(xr.global_runtime_device_count())
    addressable_runtime_devices = int(xr.addressable_runtime_device_count())
    backend = str(xr.device_type())
    if (
        backend.upper() != "TPU"
        or runtime_devices != 4
        or addressable_runtime_devices != 4
    ):
        raise SystemExit(
            "refusing fallback: "
            f"backend={backend!r}, runtime_devices={runtime_devices}, "
            f"addressable_runtime_devices={addressable_runtime_devices}"
        )
    mesh = xs.Mesh(np.arange(runtime_devices), (runtime_devices,), ("data",))
    xs.set_global_mesh(mesh)
    device = xm.xla_device()
    # Under XLA SPMD a four-device runtime is intentionally surfaced as one
    # logical xla:0 device; the physical count is established by the two
    # runtime-device checks above and by the four-element Mesh.
    supported = [str(value) for value in xm.get_xla_supported_devices("TPU")]
    if supported != ["xla:0"] or str(device) != "xla:0":
        raise SystemExit(f"invalid TPU devices: device={device}, supported={supported}")
    evidence = {
        "backend": backend,
        "device": str(device),
        "runtime_device_count": runtime_devices,
        "addressable_runtime_device_count": addressable_runtime_devices,
        "spmd_mesh_device_ids": list(range(runtime_devices)),
        "supported_tpu_devices": supported,
        "torch": torch.__version__,
        "torch_xla": getattr(torch_xla, "__version__", "unknown"),
        "host": socket.gethostname(),
        "pjrt_device": os.environ.get("PJRT_DEVICE"),
        "tpu_name": os.environ.get("EXPECTED_TPU_NAME"),
        "tpu_zone": os.environ.get("EXPECTED_TPU_ZONE"),
    }
    print("TPU_BACKEND_EVIDENCE " + json.dumps(evidence, sort_keys=True), flush=True)

    run_dir = args.run_dir.resolve()
    manifest = json.loads((run_dir / "manifest.json").read_text())
    contract, prompt_count = validate_manifest(manifest, run_dir, args.episode_batch)
    adapter = f"p{prompt_count}_mean_of_native_painter_p1_prompts_v1"
    episodes = list(manifest["episodes"])
    if args.limit:
        if args.limit % args.episode_batch:
            raise SystemExit("--limit must be divisible by --episode-batch")
        episodes = episodes[: args.limit]
    if not episodes:
        raise SystemExit("no episodes selected")

    model, checkpoint_sha = load_model(args.source_dir.resolve(), args.checkpoint.resolve(), device)
    predictions_dir = run_dir / "predictions"
    predictions_dir.mkdir(parents=True, exist_ok=True)
    partial_path = run_dir / "inference.partial.json"
    final_path = run_dir / "inference.json"
    previous = partial_path if partial_path.exists() else final_path
    if args.resume and previous.exists():
        log = json.loads(previous.read_text())
        expected = {
            "pack_id": manifest["pack_id"],
            "checkpoint_sha256": checkpoint_sha,
            "source_commit": SOURCE_COMMIT,
            "adapter": adapter,
            "backend": "TPU",
            "runtime_device_count": 4,
            "addressable_runtime_device_count": 4,
            "episode_batch": args.episode_batch,
            "native_prompt_count": prompt_count,
        }
        mismatch = {key: (log.get(key), value) for key, value in expected.items() if log.get(key) != value}
        if mismatch:
            raise SystemExit(f"resume metadata mismatch: {mismatch}")
        log.pop("finished_utc", None)
    else:
        log = {
            "schema": "painter_visualcloze_xla_inference_v1",
            "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "pack_name": contract["name"],
            "pack_id": manifest["pack_id"],
            "supervision": contract["supervision"],
            "model": "BAAI/Painter painter_vit_large",
            "source_commit": SOURCE_COMMIT,
            "model_revision": MODEL_REVISION,
            "checkpoint_sha256": checkpoint_sha,
            "checkpoint_bytes": CHECKPOINT_BYTES,
            "adapter": adapter,
            "adapter_detail": (
                f"Each frozen P={prompt_count} episode is {prompt_count} independent native one-prompt "
                "Painter forwards; raw normalized predictions are averaged before de-normalization, "
                "clipping, and resize. This is an ensemble, not native multi-prompt inference."
            ),
            "placement": f"{args.episode_batch} P={prompt_count} episodes per batch, data-sharded over v4-8 with XLA SPMD",
            "episode_batch": args.episode_batch,
            "native_prompt_count": prompt_count,
            **evidence,
            "results": [],
        }
    complete = {row["episode_id"] for row in log["results"]}
    json_dump(partial_path, log)

    batches = [episodes[index : index + args.episode_batch] for index in range(0, len(episodes), args.episode_batch)]
    for batch_index, batch in enumerate(batches, 1):
        already = [row["episode_id"] in complete for row in batch]
        if all(already):
            continue
        if any(already):
            raise SystemExit(f"partial resume batch: {[row['episode_id'] for row in batch]}")
        inputs_np, targets_np = load_episode_batch(run_dir, batch, prompt_count)
        started = time.perf_counter()
        predictions = predict_batch(model, inputs_np, targets_np, device, mesh, args.episode_batch, prompt_count)
        elapsed = time.perf_counter() - started
        for episode, prediction in zip(batch, predictions):
            output = predictions_dir / f"{episode['episode_id']}__pred256.png"
            Image.fromarray(prediction, "RGB").save(output)
            log["results"].append(
                {
                    "episode_id": episode["episode_id"],
                    "prediction": str(output.relative_to(run_dir)),
                    "prediction_sha256": sha256(output),
                    "batch_seconds": elapsed,
                    "episode_batch": args.episode_batch,
                    "native_prompt_count": prompt_count,
                }
            )
            complete.add(episode["episode_id"])
        json_dump(partial_path, log)
        completed = len(log["results"])
        if batch_index == 1 or batch_index % args.log_every == 0 or completed == len(episodes):
            print(
                f"{contract['name']}: {completed}/{len(episodes)} "
                f"batch_seconds={elapsed:.3f} last={batch[-1]['episode_id']}",
                flush=True,
            )
    log["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    json_dump(final_path, log)
    partial_path.unlink(missing_ok=True)
    print(f"COMPLETE {contract['name']} {len(log['results'])}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
