#!/usr/bin/env python3
"""Score completed Painter predictions with the pack-native jflux metrics."""
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=sorted(CONTRACTS), required=True)
    parser.add_argument("--episode-run", type=Path, required=True)
    parser.add_argument("--inference-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=20_000)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    contract = CONTRACTS[args.split]
    scorer_sha = sha256(contract["scorer"])
    if scorer_sha != contract["scorer_sha256"]:
        raise SystemExit(f"refusing score: {args.split} scorer SHA-256 changed to {scorer_sha}")

    manifest = json.loads((args.episode_run / "manifest.json").read_text())
    inference = json.loads((args.inference_dir / "inference.json").read_text())
    if manifest.get("pack_id") != contract["pack_id"] or inference.get("pack_id") != contract["pack_id"]:
        raise SystemExit("pack identity mismatch")
    if len(manifest.get("episodes", [])) != contract["episodes"]:
        raise SystemExit("manifest episode-count mismatch")
    if len(inference.get("results", [])) != contract["episodes"]:
        raise SystemExit("inference is incomplete")
    if inference.get("checkpoint_sha256") != (
        "1bdd60ce013e82253e2a031726f36d07587947438b0e561d4338d6f0c1113f44"
    ):
        raise SystemExit("Painter checkpoint identity mismatch")
    if inference.get("adapter") != "p2_mean_of_two_native_painter_prompts_v1":
        raise SystemExit("Painter adapter identity mismatch")

    sys.path.insert(0, str(JFLUX))
    if args.split == "test":
        from jflux2.fasteval import metrics_v3 as metrics  # noqa: PLC0415
    else:
        from jflux2.fasteval import metrics  # type: ignore[no-redef]  # noqa: PLC0415

    episodes = {row["episode_id"]: row for row in manifest["episodes"]}
    partial = args.out.with_suffix(".partial.json")
    results = []
    if args.resume and partial.exists():
        progress = json.loads(partial.read_text())
        if progress.get("pack_id") != contract["pack_id"]:
            raise SystemExit("partial score pack identity mismatch")
        results = list(progress.get("results", []))
    complete = {row["episode_id"] for row in results}
    for generated in inference["results"]:
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
            copied = metrics.score_pair_v3(episode["task"], query_input, target, **kwargs)
            perfect = metrics.score_pair_v3(episode["task"], target, target, **kwargs)
        else:
            scored = metrics.score_pair(episode["task"], prediction, target, strict=True)
            copied = metrics.score_pair(episode["task"], query_input, target, strict=True)
            perfect = metrics.score_pair(episode["task"], target, target, strict=True)
        results.append(
            {
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
        )
        complete.add(episode_id)
        if len(results) % 32 == 0:
            json_dump(
                partial,
                {
                    "schema": "painter_score_progress_v1",
                    "pack_id": contract["pack_id"],
                    "scorer_sha256": scorer_sha,
                    "results": results,
                },
            )
            print(f"{args.split}: scored {len(results)}/{contract['episodes']}", flush=True)

    cells = {}
    for task in sorted({row["task"] for row in results}):
        for condition in CONDITIONS:
            rows = [row for row in results if row["task"] == task and row["condition"] == condition]
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
        "source_commit": inference.get("source_commit"),
        "model_revision": inference.get("model_revision"),
        "checkpoint_sha256": inference.get("checkpoint_sha256"),
        "adapter": inference.get("adapter"),
        "scorer_sha256": scorer_sha,
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

