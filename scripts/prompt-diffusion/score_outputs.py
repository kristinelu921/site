#!/usr/bin/env python3
"""Score completed Prompt Diffusion predictions with pinned pack-native metrics."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image


JFLUX = Path("/kmh-nfs-ssd-us-mount/code/xtiange/jflux2")
CONDITIONS = ("real", "scrambled", "grey", "oracle")
SOURCE_COMMIT = "8ecd2d126d9f9e17b80d9655bf8ed2df2e93f660"
MODEL_REVISION = "60ea7b6cc48d08487f95d06ba6ce59e15b619efd"
ADAPTER = "p2_mean_of_two_native_prompt_diffusion_p1_v1"
PROMPT_EMBEDDINGS_SHA256 = "bd7e07a5a0d51e6586928f913c78cdae665936abb12fa81b0bdcb65fdbc32219"
CONTRACTS = {
    "test": {
        "pack_id": "48e062452574ff0ded61c72664445d687ecd1c02",
        "episodes": 3840,
        "scorer": JFLUX / "jflux2/fasteval/metrics_v3.py",
        "scorer_sha256": "a82740a2aefc59062210dc0fadf3199c7fc1a624629f6c0154020f05dd8b4787",
    },
    "val": {
        "pack_id": "9e77a92f82d0e08c12f540d5c15c2b15e6c503a2",
        "episodes": 512,
        "scorer": JFLUX / "jflux2/fasteval/metrics.py",
        "scorer_sha256": "0f901201dbd18fec48aa6cb2a4102f27191fa4c89eed151c5113c3675cac0096",
    },
}


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


def verify_inference(inference: dict, contract: dict) -> None:
    expected = {
        "pack_id": contract["pack_id"],
        "model": "Prompt Diffusion",
        "source_commit": SOURCE_COMMIT,
        "model_revision": MODEL_REVISION,
        "adapter": ADAPTER,
        "prompt_embeddings_sha256": PROMPT_EMBEDDINGS_SHA256,
        "backend": "TPU",
    }
    for field, value in expected.items():
        if inference.get(field) != value:
            raise SystemExit(
                f"Prompt Diffusion identity mismatch for {field}: "
                f"{inference.get(field)!r} != {value!r}"
            )
    if len(inference.get("results", [])) != contract["episodes"]:
        raise SystemExit("inference is incomplete")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=sorted(CONTRACTS), required=True)
    parser.add_argument("--episode-run", type=Path, required=True)
    parser.add_argument("--inference-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=20_000)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument(
        "--baseline-report",
        type=Path,
        help="Completed pinned-score report supplying model-independent copy/perfect baselines",
    )
    args = parser.parse_args()
    if args.shard_count < 1 or not 0 <= args.shard_index < args.shard_count:
        raise SystemExit("invalid scoring shard")
    contract = CONTRACTS[args.split]
    scorer_sha = sha256(contract["scorer"])
    if scorer_sha != contract["scorer_sha256"]:
        raise SystemExit(f"refusing score: {args.split} scorer SHA-256 changed to {scorer_sha}")

    manifest = json.loads((args.episode_run / "manifest.json").read_text())
    inference = json.loads((args.inference_dir / "inference.json").read_text())
    if manifest.get("pack_id") != contract["pack_id"]:
        raise SystemExit("episode manifest pack identity mismatch")
    if len(manifest.get("episodes", [])) != contract["episodes"]:
        raise SystemExit("manifest episode-count mismatch")
    verify_inference(inference, contract)

    sys.path.insert(0, str(JFLUX))
    if args.split == "test":
        from jflux2.fasteval import metrics_v3 as metrics  # noqa: PLC0415
    else:
        from jflux2.fasteval import metrics  # type: ignore[no-redef]  # noqa: PLC0415

    episodes = {row["episode_id"]: row for row in manifest["episodes"]}
    generated_ids = [row["episode_id"] for row in inference["results"]]
    if len(set(generated_ids)) != contract["episodes"] or set(generated_ids) != set(episodes):
        raise SystemExit("inference episode IDs do not exactly match the frozen manifest")
    generated_subset = inference["results"][args.shard_index :: args.shard_count]
    shard_episodes = len(generated_subset)
    baseline_rows = None
    if args.baseline_report:
        baseline = json.loads(args.baseline_report.read_text())
        if baseline.get("pack_id") != contract["pack_id"]:
            raise SystemExit("baseline report pack identity mismatch")
        if baseline.get("scorer_sha256") != scorer_sha:
            raise SystemExit("baseline report scorer identity mismatch")
        if len(baseline.get("results", [])) != contract["episodes"]:
            raise SystemExit("baseline report is incomplete")
        baseline_rows = {row["episode_id"]: row for row in baseline["results"]}
        if len(baseline_rows) != contract["episodes"] or set(baseline_rows) != set(episodes):
            raise SystemExit("baseline report episode IDs do not match the frozen manifest")
    partial = args.out.with_suffix(".partial.json")
    results = []
    if args.resume and partial.exists():
        progress = json.loads(partial.read_text())
        if progress.get("pack_id") != contract["pack_id"]:
            raise SystemExit("partial score pack identity mismatch")
        if progress.get("shard_index", 0) != args.shard_index:
            raise SystemExit("partial score shard-index mismatch")
        if progress.get("shard_count", 1) != args.shard_count:
            raise SystemExit("partial score shard-count mismatch")
        results = list(progress.get("results", []))
    complete = {row["episode_id"] for row in results}

    for generated in generated_subset:
        episode_id = generated["episode_id"]
        if episode_id in complete:
            continue
        episode = episodes[episode_id]
        episode_path = args.episode_run / episode["file"]
        if sha256(episode_path) != episode["sha256"]:
            raise SystemExit(f"episode SHA-256 mismatch: {episode_id}")
        with np.load(episode_path) as packed:
            target = np.asarray(packed["query_target"], np.uint8)
            query_input = np.asarray(packed["query_input"], np.uint8)
        prediction_path = args.inference_dir / generated["prediction"]
        if sha256(prediction_path) != generated["prediction_sha256"]:
            raise SystemExit(f"prediction SHA-256 mismatch: {episode_id}")
        prediction = np.asarray(Image.open(prediction_path).convert("RGB"), np.uint8)
        if args.split == "test":
            kwargs = {"source": episode["source"], "perceptual": False}
            scored = metrics.score_pair_v3(episode["task"], prediction, target, **kwargs)
            if baseline_rows is None:
                copied = metrics.score_pair_v3(episode["task"], query_input, target, **kwargs)
                perfect = metrics.score_pair_v3(episode["task"], target, target, **kwargs)
        else:
            scored = metrics.score_pair(episode["task"], prediction, target, strict=True)
            if baseline_rows is None:
                copied = metrics.score_pair(episode["task"], query_input, target, strict=True)
                perfect = metrics.score_pair(episode["task"], target, target, strict=True)
        if baseline_rows is None:
            copy_input_floor = float(copied["primary_value"])
            perfect_target_ceiling = float(perfect["primary_value"])
        else:
            baseline_row = baseline_rows[episode_id]
            if baseline_row["primary_metric"] != scored["primary_metric"]:
                raise SystemExit(f"baseline primary metric mismatch: {episode_id}")
            copy_input_floor = float(baseline_row["copy_input_floor"])
            perfect_target_ceiling = float(baseline_row["perfect_target_ceiling"])
        results.append(
            {
                **episode,
                "prediction": generated["prediction"],
                "seconds": generated.get("batch_seconds"),
                "primary_metric": scored["primary_metric"],
                "primary_lower_is_better": bool(scored["primary_lower_is_better"]),
                "primary_value": float(scored["primary_value"]),
                "copy_input_floor": copy_input_floor,
                "perfect_target_ceiling": perfect_target_ceiling,
                "l1_01": float(scored["l1_01"]),
            }
        )
        complete.add(episode_id)
        if len(results) % 32 == 0:
            json_dump(
                partial,
                {
                    "schema": "prompt_diffusion_score_progress_v1",
                    "pack_id": contract["pack_id"],
                    "scorer_sha256": scorer_sha,
                    "shard_index": args.shard_index,
                    "shard_count": args.shard_count,
                    "results": results,
                },
            )
            print(
                f"{args.split} shard {args.shard_index + 1}/{args.shard_count}: "
                f"scored {len(results)}/{shard_episodes}",
                flush=True,
            )

    cells = {}
    for task in sorted({row["task"] for row in results}):
        for condition in CONDITIONS:
            rows = [row for row in results if row["task"] == task and row["condition"] == condition]
            if not rows:
                continue
            values = np.asarray([row["primary_value"] for row in rows], np.float64)
            finite = values[np.isfinite(values)]
            if finite.size:
                rng = np.random.default_rng(0)
                boots = finite[
                    rng.integers(0, finite.size, size=(int(args.bootstrap), finite.size))
                ].mean(axis=1)
                mean = float(finite.mean())
                ci95 = [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]
            else:
                mean = float("nan")
                ci95 = [float("nan"), float("nan")]
            cells[f"{task}@{condition}"] = {
                "task": task,
                "condition": condition,
                "primary_metric": rows[0]["primary_metric"],
                "primary_lower_is_better": rows[0]["primary_lower_is_better"],
                "mean": mean,
                "ci95": ci95,
                "n": int(finite.size),
                "n_nan": int(values.size - finite.size),
                "copy_input_mean": float(np.nanmean([row["copy_input_floor"] for row in rows])),
                "perfect_target_mean": float(
                    np.nanmean([row["perfect_target_ceiling"] for row in rows])
                ),
            }
    report = {
        "schema": "prompt_diffusion_visualcloze_scores_v1",
        "scored_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "split": args.split,
        "pack_id": contract["pack_id"],
        "model": inference["model"],
        "source_commit": inference["source_commit"],
        "model_revision": inference["model_revision"],
        "adapter": inference["adapter"],
        "prompt_embeddings_sha256": inference["prompt_embeddings_sha256"],
        "scorer_sha256": scorer_sha,
        "bootstrap_samples": int(args.bootstrap),
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "cells": cells,
        "results": results,
    }
    json_dump(args.out, report)
    partial.unlink(missing_ok=True)
    print(f"wrote {args.out} ({len(results)} scored episodes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
