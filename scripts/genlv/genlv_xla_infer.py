#!/usr/bin/env python3
"""Run pinned GenLV-Huge on frozen VisualCloze P=2 episodes with PyTorch/XLA."""
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

from genlv_compat import install as install_import_shims


SOURCE_COMMIT = "e8922a52a705fded1580da9d9ebb803af6127637"
MODEL_REVISION = "686c5c21c2aedb9768fa18f3d5148212b039d2d2"
CHECKPOINT_SHA256 = "09e110d8c0781ba40848b49bf40c759daaa5eb1a6d4a9c959f5a75877d467ccc"
CHECKPOINT_BYTES = 2_457_130_637
ARCH = "xrestormer_prompt_crossattn_huge"
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
    if manifest.get("seed_rule") != "[seed, task_seed(task), q, P]":
        raise SystemExit("seed rule changed")
    if manifest.get("demo_source_policy") != "same":
        raise SystemExit("demonstrations are not same-source")
    if {row.get("condition") for row in episodes} != CONDITIONS:
        raise SystemExit("condition set changed")
    if len({row.get("episode_id") for row in episodes}) != len(episodes):
        raise SystemExit("duplicate episode ids")
    task_counts = Counter(row["task"] for row in episodes)
    if len(task_counts) != contract["tasks"]:
        raise SystemExit("task count changed")
    if set(task_counts.values()) != {contract["per_task"] * len(CONDITIONS)}:
        raise SystemExit("per-task episode count changed")
    if len(episodes) % episode_batch:
        raise SystemExit("episode count must divide evenly by --episode-batch")
    missing = [row["file"] for row in episodes if not (run_dir / row["file"]).is_file()]
    if missing:
        raise SystemExit(f"missing {len(missing)} episode files; first={missing[0]}")
    return contract


def load_episode_batch(run_dir: Path, episodes: list[dict]) -> tuple[np.ndarray, ...]:
    prompt_inputs: list[np.ndarray] = []
    prompt_targets: list[np.ndarray] = []
    queries: list[np.ndarray] = []
    for episode in episodes:
        episode_path = run_dir / episode["file"]
        if sha256(episode_path) != episode["sha256"]:
            raise SystemExit(f"episode SHA-256 mismatch: {episode['episode_id']}")
        with np.load(episode_path, allow_pickle=False) as packed:
            demo_inputs = np.asarray(packed["demo_inputs"], np.uint8)
            demo_targets = np.asarray(packed["demo_targets"], np.uint8)
            query_input = np.asarray(packed["query_input"], np.uint8)
        if demo_inputs.shape != (2, 256, 256, 3) or demo_targets.shape != demo_inputs.shape:
            raise SystemExit(f"P=2 array shape changed: {episode['episode_id']}")
        if query_input.shape != (256, 256, 3):
            raise SystemExit(f"query shape changed: {episode['episode_id']}")
        # Official GenLV reads with OpenCV and therefore consumes BGR [0,1].
        query_bgr = query_input[..., ::-1].astype(np.float32) / 255.0
        for demo_input, demo_target in zip(demo_inputs, demo_targets):
            prompt_inputs.append(demo_input[..., ::-1].astype(np.float32) / 255.0)
            prompt_targets.append(demo_target[..., ::-1].astype(np.float32) / 255.0)
            queries.append(query_bgr)
    arrays = []
    for rows in (prompt_inputs, prompt_targets, queries):
        arrays.append(np.ascontiguousarray(np.stack(rows).transpose(0, 3, 1, 2)))
    return tuple(arrays)


def load_model(source_dir: Path, checkpoint_path: Path, device):
    import torch
    import torch_xla.core.xla_model as xm

    if checkpoint_path.stat().st_size != CHECKPOINT_BYTES:
        raise SystemExit("checkpoint byte-size mismatch")
    checkpoint_sha = sha256(checkpoint_path)
    if checkpoint_sha != CHECKPOINT_SHA256:
        raise SystemExit(f"checkpoint SHA-256 mismatch: {checkpoint_sha}")
    install_import_shims()
    sys.path.insert(0, str(source_dir / "extended_version"))
    from models import xrestormer_prompt_crossattn_wores as models  # noqa: PLC0415

    model = getattr(models, ARCH)()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    message = model.load_state_dict(checkpoint["model"], strict=True)
    if message.missing_keys or message.unexpected_keys:
        raise SystemExit(
            f"checkpoint/model mismatch: missing={message.missing_keys}, "
            f"unexpected={message.unexpected_keys}"
        )
    del checkpoint
    model = model.eval().to(device)
    xm.mark_step()
    return model, checkpoint_sha


def predict_batch(model, arrays: tuple[np.ndarray, ...], device, mesh, episode_batch: int) -> np.ndarray:
    import torch
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.spmd as xs

    expected_rows = episode_batch * 2
    tensors = []
    for value in arrays:
        if value.shape != (expected_rows, 3, 256, 256):
            raise SystemExit(f"unexpected input shape: {value.shape}")
        tensor = torch.from_numpy(value).to(device)
        xs.mark_sharding(tensor, mesh, ("data", None, None, None))
        tensors.append(tensor)
    prompt_input, prompt_target, query = tensors
    with torch.no_grad():
        _, prediction, _ = model(
            imgs=[prompt_input, prompt_target, query, query], input_is_list=True
        )
        xm.mark_step()
        # GenLV is natively P=1. Preserve the frozen P=2 episode by running
        # both demonstrations, then mean the two raw BGR predictions.
        prediction = prediction.cpu().reshape(episode_batch, 2, 3, 256, 256).mean(dim=1)
        prediction = torch.clamp(prediction * 255.0, 0.0, 255.0).to(torch.uint8)
    # Return the public pack's RGB convention.
    return prediction.permute(0, 2, 3, 1).numpy()[..., ::-1].copy()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--episode-batch", type=int, required=True)
    parser.add_argument("--expected-devices", type=int, required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--log-every", type=int, default=8)
    args = parser.parse_args()
    if args.episode_batch != args.expected_devices:
        raise SystemExit("episode batch must equal physical TPU device count")

    import torch
    import torch_xla
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.spmd as xs
    import torch_xla.runtime as xr

    args.cache_dir.mkdir(parents=True, exist_ok=True)
    xr.initialize_cache(str(args.cache_dir.resolve()), readonly=False)
    xr.use_spmd()
    runtime_devices = int(xr.global_runtime_device_count())
    addressable_devices = int(xr.addressable_runtime_device_count())
    backend = str(xr.device_type())
    if backend.upper() != "TPU" or runtime_devices != args.expected_devices or addressable_devices != args.expected_devices:
        raise SystemExit(
            f"refusing fallback: backend={backend!r}, runtime={runtime_devices}, "
            f"addressable={addressable_devices}, expected={args.expected_devices}"
        )
    mesh = xs.Mesh(np.arange(runtime_devices), (runtime_devices,), ("data",))
    xs.set_global_mesh(mesh)
    device = xm.xla_device()
    evidence = {
        "backend": backend,
        "device": str(device),
        "runtime_device_count": runtime_devices,
        "addressable_runtime_device_count": addressable_devices,
        "spmd_mesh_device_ids": list(range(runtime_devices)),
        "torch": torch.__version__,
        "torch_xla": getattr(torch_xla, "__version__", "unknown"),
        "host": socket.gethostname(),
        "tpu_name": os.environ.get("EXPECTED_TPU_NAME"),
        "tpu_zone": os.environ.get("EXPECTED_TPU_ZONE"),
    }
    print("TPU_BACKEND_EVIDENCE " + json.dumps(evidence, sort_keys=True), flush=True)

    run_dir = args.run_dir.resolve()
    manifest = json.loads((run_dir / "manifest.json").read_text())
    contract = validate_manifest(manifest, run_dir, args.episode_batch)
    episodes = list(manifest["episodes"])
    if args.limit:
        if args.limit % args.episode_batch:
            raise SystemExit("--limit must be divisible by --episode-batch")
        episodes = episodes[: args.limit]
    model, checkpoint_sha = load_model(args.source_dir.resolve(), args.checkpoint.resolve(), device)

    predictions_dir = run_dir / "predictions"
    predictions_dir.mkdir(parents=True, exist_ok=True)
    partial_path = run_dir / "inference.partial.json"
    final_path = run_dir / "inference.json"
    previous = partial_path if partial_path.exists() else final_path
    expected_resume = {
        "pack_id": manifest["pack_id"], "checkpoint_sha256": checkpoint_sha,
        "source_commit": SOURCE_COMMIT,
        "adapter": "p2_mean_of_two_native_genlv_prompts_v1",
        "backend": "TPU", "runtime_device_count": runtime_devices,
        "episode_batch": args.episode_batch,
    }
    if args.resume and previous.exists():
        log = json.loads(previous.read_text())
        mismatch = {key: (log.get(key), value) for key, value in expected_resume.items() if log.get(key) != value}
        if mismatch:
            raise SystemExit(f"resume metadata mismatch: {mismatch}")
        seen_resume: set[str] = set()
        for row in log.get("results", []):
            episode_id = row.get("episode_id")
            prediction_rel = row.get("prediction")
            if not isinstance(episode_id, str) or not isinstance(prediction_rel, str):
                raise SystemExit("resume row is missing episode_id/prediction")
            if episode_id in seen_resume:
                raise SystemExit(f"duplicate resume row: {episode_id}")
            prediction_path = run_dir / prediction_rel
            if not prediction_path.is_file():
                raise SystemExit(f"resume prediction is missing: {prediction_path}")
            actual_sha = sha256(prediction_path)
            if actual_sha != row.get("prediction_sha256"):
                raise SystemExit(
                    f"resume prediction SHA-256 mismatch: {episode_id} "
                    f"expected={row.get('prediction_sha256')} actual={actual_sha}"
                )
            seen_resume.add(episode_id)
        log.pop("finished_utc", None)
    else:
        log = {
            "schema": "genlv_visualcloze_xla_inference_v1",
            "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "pack_name": contract["name"], "pack_id": manifest["pack_id"],
            "supervision": contract["supervision"], "model": "GenLV-101 Huge",
            "architecture": ARCH, "source_commit": SOURCE_COMMIT,
            "model_revision": MODEL_REVISION, "checkpoint_sha256": checkpoint_sha,
            "checkpoint_bytes": CHECKPOINT_BYTES,
            "adapter": "p2_mean_of_two_native_genlv_prompts_v1",
            "adapter_detail": "Two native P=1 forwards per frozen P=2 episode; mean raw BGR outputs before clipping and RGB conversion.",
            "placement": f"{args.episode_batch} P=2 episodes per batch, data-sharded over TPU with XLA SPMD",
            "episode_batch": args.episode_batch, **evidence, "results": [],
        }
    complete = {row["episode_id"] for row in log["results"]}
    json_dump(partial_path, log)

    batches = [episodes[index:index + args.episode_batch] for index in range(0, len(episodes), args.episode_batch)]
    for batch_index, batch in enumerate(batches, 1):
        already = [row["episode_id"] in complete for row in batch]
        if all(already):
            continue
        if any(already):
            raise SystemExit(f"partial resume batch: {[row['episode_id'] for row in batch]}")
        arrays = load_episode_batch(run_dir, batch)
        started = time.perf_counter()
        predictions = predict_batch(model, arrays, device, mesh, args.episode_batch)
        elapsed = time.perf_counter() - started
        for episode, prediction in zip(batch, predictions):
            # The directory may be removed by an external host cleanup.  Keep
            # every individual commit fail-closed and atomic even though the
            # recovery runner now uses a persistent node-local directory.
            predictions_dir.mkdir(parents=True, exist_ok=True)
            output = predictions_dir / f"{episode['episode_id']}__pred256.png"
            temporary = output.with_suffix(output.suffix + ".tmp")
            Image.fromarray(prediction, "RGB").save(temporary, format="PNG")
            os.replace(temporary, output)
            log["results"].append({
                "episode_id": episode["episode_id"],
                "prediction": str(output.relative_to(run_dir)),
                "prediction_sha256": sha256(output), "batch_seconds": elapsed,
                "episode_batch": args.episode_batch, "native_prompt_count": 2,
            })
            complete.add(episode["episode_id"])
        json_dump(partial_path, log)
        completed = len(log["results"])
        if batch_index == 1 or batch_index % args.log_every == 0 or completed == len(episodes):
            print(f"{contract['name']}: {completed}/{len(episodes)} batch_seconds={elapsed:.3f} last={batch[-1]['episode_id']}", flush=True)
    log["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    json_dump(final_path, log)
    partial_path.unlink(missing_ok=True)
    print(f"COMPLETE {contract['name']} {len(log['results'])}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
