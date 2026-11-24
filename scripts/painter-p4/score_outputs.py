#!/usr/bin/env python3
"""Score completed Painter native-P1 ensemble predictions with pinned metrics."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image


JFLUX = Path("/kmh-nfs-ssd-us-mount/code/xtiange/jflux2")
SOURCE_COMMIT = "cfec84587da2a2cc4e0b9acebe6fba1666e66477"
MODEL_REVISION = "97200aa029bab2e61ea5b4ce627534e35f152b23"
CONDITIONS = ("real", "scrambled", "grey", "oracle")
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
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def score_one(job: tuple[str, str, str, dict, dict]) -> dict:
    split, episode_root_s, inference_root_s, episode, generated = job
    sys.path.insert(0, str(JFLUX))
    if split == "test":
        from jflux2.fasteval import metrics_v3 as metrics
        metrics._LPIPS = {"net": None, "tried": True}
    else:
        from jflux2.fasteval import metrics

    episode_root = Path(episode_root_s)
    inference_root = Path(inference_root_s)
    episode_path = episode_root / episode["file"]
    if sha256(episode_path) != episode["sha256"]:
        raise RuntimeError(f"episode SHA-256 mismatch: {episode['episode_id']}")
    with np.load(episode_path) as packed:
        target = np.asarray(packed["query_target"], np.uint8)
        query_input = np.asarray(packed["query_input"], np.uint8)
    prediction_path = inference_root / generated["prediction"]
    if sha256(prediction_path) != generated["prediction_sha256"]:
        raise RuntimeError(f"prediction SHA-256 mismatch: {episode['episode_id']}")
    prediction = np.asarray(Image.open(prediction_path).convert("RGB"), np.uint8)
    if split == "test":
        kwargs = {"source": episode["source"], "perceptual": False}
        scored = metrics.score_pair_v3(episode["task"], prediction, target, **kwargs)
        copied = metrics.score_pair_v3(episode["task"], query_input, target, **kwargs)
        perfect = metrics.score_pair_v3(episode["task"], target, target, **kwargs)
    else:
        scored = metrics.score_pair(episode["task"], prediction, target, strict=True)
        copied = metrics.score_pair(episode["task"], query_input, target, strict=True)
        perfect = metrics.score_pair(episode["task"], target, target, strict=True)
    return {
        **episode,
        "prediction": generated["prediction"],
        "seconds": generated.get("seconds"),
        "primary_metric": scored["primary_metric"],
        "primary_lower_is_better": bool(scored["primary_lower_is_better"]),
        "primary_value": float(scored["primary_value"]),
        "copy_input_floor": float(copied["primary_value"]),
        "perfect_target_ceiling": float(perfect["primary_value"]),
        "l1_01": float(scored["l1_01"]),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=sorted(CONTRACTS), required=True)
    parser.add_argument("--episode-run", type=Path, required=True)
    parser.add_argument("--inference-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=20_000)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    contract = CONTRACTS[args.split]
    if sha256(contract["scorer"]) != contract["scorer_sha256"]:
        raise SystemExit("pinned scorer hash changed")

    manifest = json.loads((args.episode_run / "manifest.json").read_text())
    inference = json.loads((args.inference_dir / "inference.json").read_text())
    prompt_count = int(manifest.get("p", -1))
    adapter = f"p{prompt_count}_mean_of_native_painter_p1_prompts_v1"
    if prompt_count not in {4, 8, 16}:
        raise SystemExit("unapproved ensemble prompt count")
    if manifest.get("pack_id") != contract["pack_id"] or inference.get("pack_id") != contract["pack_id"]:
        raise SystemExit("pack identity mismatch")
    if len(manifest.get("episodes", [])) != contract["episodes"]:
        raise SystemExit("manifest episode-count mismatch")
    if len(inference.get("results", [])) != contract["episodes"]:
        raise SystemExit("inference result-count mismatch")
    if (
        inference.get("source_commit") != SOURCE_COMMIT
        or inference.get("model_revision") != MODEL_REVISION
        or inference.get("adapter") != adapter
        or inference.get("native_prompt_count") != prompt_count
    ):
        raise SystemExit("Painter identity mismatch")

    expected_cells = len({row["task"] for row in manifest["episodes"]}) * len(CONDITIONS)
    if args.out.exists():
        existing = json.loads(args.out.read_text())
        if (
            existing.get("pack_id") == contract["pack_id"]
            and existing.get("scorer_sha256") == contract["scorer_sha256"]
            and len(existing.get("results", [])) == contract["episodes"]
            and len(existing.get("cells", {})) == expected_cells
        ):
            print(f"{args.split}: existing complete score report verified; skipping")
            return 0
        raise SystemExit(f"refusing to overwrite invalid existing report: {args.out}")

    partial = args.out.with_suffix(".partial.json")
    progress = json.loads(partial.read_text()) if args.resume and partial.exists() else {
        "schema": "painter_score_progress_v1",
        "pack_id": contract["pack_id"],
        "scorer_sha256": contract["scorer_sha256"],
        "results": [],
    }
    if progress.get("pack_id") != contract["pack_id"]:
        raise SystemExit("partial score pack identity mismatch")
    results = list(progress["results"])
    complete = {row["episode_id"] for row in results}
    episodes = {row["episode_id"]: row for row in manifest["episodes"]}
    jobs = [
        (
            args.split,
            str(args.episode_run.resolve()),
            str(args.inference_dir.resolve()),
            episodes[row["episode_id"]],
            row,
        )
        for row in inference["results"]
        if row["episode_id"] not in complete
    ]
    print(f"{args.split}: parallel fill starts at {len(results)}/{contract['episodes']} with {args.workers} workers", flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for row in pool.map(score_one, jobs, chunksize=2):
            results.append(row)
            if len(results) % 32 == 0:
                progress["results"] = results
                json_dump(partial, progress)
                print(f"{args.split}: scored {len(results)}/{contract['episodes']}", flush=True)
    progress["results"] = results
    json_dump(partial, progress)
    print(f"{args.split}: parallel fill complete {len(results)}/{contract['episodes']}", flush=True)

    cells = {}
    for task in sorted({row["task"] for row in results}):
        for condition in CONDITIONS:
            rows = [row for row in results if row["task"] == task and row["condition"] == condition]
            if not rows:
                raise SystemExit(f"missing score cell: {task}@{condition}")
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
        "schema": "painter_visualcloze_scores_v1",
        "scored_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "split": args.split,
        "pack_id": contract["pack_id"],
        "model": inference.get("model"),
        "source_commit": SOURCE_COMMIT,
        "model_revision": MODEL_REVISION,
        "adapter": adapter,
        "native_prompt_count": prompt_count,
        "scorer_sha256": contract["scorer_sha256"],
        "optional_lpips_addon": "disabled_not_reported_primary_and_l1_unchanged",
        "bootstrap_samples": int(args.bootstrap),
        "cells": cells,
        "results": results,
    }
    json_dump(args.out, report)
    partial.unlink(missing_ok=True)
    print(f"wrote {args.out} ({len(results)} scored episodes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
