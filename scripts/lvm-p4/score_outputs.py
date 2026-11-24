#!/usr/bin/env python3
"""Score terminal LVM predictions with the pinned pack-native metrics."""
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


JFLUX = Path(__file__).resolve().parent / "scoring_snapshot"
SOURCE_COMMIT = "b6de939ef0eb1ee6593445a7f5268145f338749b"
MODEL_REVISION = "aa5a9039273bd4221a0f69f5c943de1c56c39d89"
ADAPTER = "native_p4_nine_frame_context_visual_tokens_v1"
CONDITIONS = ("real", "scrambled", "grey", "oracle")
CONTRACTS = {
    "test": {
        "pack_id": "48e062452574ff0ded61c72664445d687ecd1c02",
        "episodes": 3840,
        "scorer": JFLUX / "jflux2/fasteval/metrics_v3.py",
        "scorer_sha256": "a82740a2aefc59062210dc0fadf3199c7fc1a624629f6c0154020f05dd8b4787",
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=sorted(CONTRACTS), required=True)
    parser.add_argument("--episode-run", type=Path, required=True)
    parser.add_argument("--inference-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--baseline-report", type=Path)
    parser.add_argument("--bootstrap", type=int, default=20_000)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    contract = CONTRACTS[args.split]
    scorer_sha = sha256(contract["scorer"])
    if scorer_sha != contract["scorer_sha256"]:
        raise SystemExit(f"refusing score: scorer SHA changed to {scorer_sha}")
    manifest = json.loads((args.episode_run / "manifest.json").read_text())
    inference = json.loads((args.inference_dir / "inference.json").read_text())
    if manifest.get("pack_id") != contract["pack_id"] or inference.get("pack_id") != contract["pack_id"]:
        raise SystemExit("pack identity mismatch")
    if len(manifest.get("episodes", [])) != contract["episodes"] or len(inference.get("results", [])) != contract["episodes"]:
        raise SystemExit("manifest or inference is incomplete")
    expected_identity = {
        "model": "LVM 7B", "source_commit": SOURCE_COMMIT,
        "model_revision": MODEL_REVISION, "adapter": ADAPTER,
        "backend": "TPU", "runtime_device_count": 4,
    }
    for key, value in expected_identity.items():
        if inference.get(key) != value:
            raise SystemExit(f"LVM identity mismatch for {key}")

    sys.path.insert(0, str(JFLUX))
    if args.split == "test":
        from jflux2.fasteval import metrics_v3 as metrics  # noqa: PLC0415
    episodes = {row["episode_id"]: row for row in manifest["episodes"]}
    generated = {row["episode_id"]: row for row in inference["results"]}
    if set(episodes) != set(generated):
        raise SystemExit("inference IDs do not exactly match the frozen manifest")

    baselines = None
    if args.baseline_report:
        baseline = json.loads(args.baseline_report.read_text())
        if baseline.get("pack_id") != contract["pack_id"] or baseline.get("scorer_sha256") != scorer_sha:
            raise SystemExit("baseline report identity mismatch")
        baselines = {row["episode_id"]: row for row in baseline["results"]}
        if set(baselines) != set(episodes):
            raise SystemExit("baseline report episode IDs mismatch")

    partial = args.out.with_suffix(".partial.json")
    results = []
    if args.resume and partial.exists():
        progress = json.loads(partial.read_text())
        if progress.get("pack_id") != contract["pack_id"]:
            raise SystemExit("partial score pack identity mismatch")
        results = list(progress.get("results", []))
    complete = {row["episode_id"] for row in results}
    for episode_id, episode in episodes.items():
        if episode_id in complete:
            continue
        generated_row = generated[episode_id]
        episode_path = args.episode_run / episode["file"]
        if sha256(episode_path) != episode["sha256"]:
            raise SystemExit(f"episode SHA mismatch: {episode_id}")
        with np.load(episode_path, allow_pickle=False) as packed:
            target = np.asarray(packed["query_target"], np.uint8)
            query_input = np.asarray(packed["query_input"], np.uint8)
        prediction_path = args.inference_dir / generated_row["prediction"]
        if sha256(prediction_path) != generated_row["prediction_sha256"]:
            raise SystemExit(f"prediction SHA mismatch: {episode_id}")
        prediction = np.asarray(Image.open(prediction_path).convert("RGB"), np.uint8)
        kwargs = {"source": episode["source"], "perceptual": False}
        scored = metrics.score_pair_v3(episode["task"], prediction, target, **kwargs)
        if baselines is None:
            copied = metrics.score_pair_v3(episode["task"], query_input, target, **kwargs)
            perfect = metrics.score_pair_v3(episode["task"], target, target, **kwargs)
        if baselines is None:
            copy_floor = float(copied["primary_value"])
            perfect_ceiling = float(perfect["primary_value"])
        else:
            copy_floor = float(baselines[episode_id]["copy_input_floor"])
            perfect_ceiling = float(baselines[episode_id]["perfect_target_ceiling"])
        results.append({
            **episode,
            "prediction": generated_row["prediction"],
            "total_seconds": generated_row.get("total_seconds"),
            "primary_metric": scored["primary_metric"],
            "primary_lower_is_better": bool(scored["primary_lower_is_better"]),
            "primary_value": float(scored["primary_value"]),
            "copy_input_floor": copy_floor,
            "perfect_target_ceiling": perfect_ceiling,
            "l1_01": float(scored["l1_01"]),
        })
        complete.add(episode_id)
        if len(results) % 32 == 0:
            json_dump(partial, {"schema": "lvm_score_progress_v1", "pack_id": contract["pack_id"], "scorer_sha256": scorer_sha, "results": results})
            print(f"{args.split}: scored {len(results)}/{contract['episodes']}", flush=True)

    cells = {}
    for task in sorted({row["task"] for row in results}):
        for condition in CONDITIONS:
            rows = [row for row in results if row["task"] == task and row["condition"] == condition]
            values = np.asarray([row["primary_value"] for row in rows], np.float64)
            finite = values[np.isfinite(values)]
            if finite.size:
                rng = np.random.default_rng(0)
                boots = finite[rng.integers(0, finite.size, size=(args.bootstrap, finite.size))].mean(axis=1)
                mean = float(finite.mean())
                ci95 = [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]
            else:
                mean = float("nan")
                ci95 = [float("nan"), float("nan")]
            cells[f"{task}@{condition}"] = {
                "task": task, "condition": condition,
                "primary_metric": rows[0]["primary_metric"],
                "primary_lower_is_better": rows[0]["primary_lower_is_better"],
                "mean": mean, "ci95": ci95, "n": int(finite.size),
                "n_nan": int(values.size - finite.size),
                "copy_input_mean": float(np.nanmean([row["copy_input_floor"] for row in rows])),
                "perfect_target_mean": float(np.nanmean([row["perfect_target_ceiling"] for row in rows])),
            }
    report = {
        "schema": "lvm_visualcloze_scores_v1",
        "scored_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "split": args.split, "pack_id": contract["pack_id"],
        **expected_identity, "scorer_sha256": scorer_sha,
        "bootstrap_samples": args.bootstrap, "cells": cells, "results": results,
    }
    json_dump(args.out, report)
    partial.unlink(missing_ok=True)
    print(f"wrote {args.out} ({len(results)} scored episodes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
