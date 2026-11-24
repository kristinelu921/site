#!/usr/bin/env python3
"""Replace the five saved restoration primaries with batched LPIPS-Alex scores.

The output keeps every non-restoration row and cell byte-for-value equivalent to
the source score report. Targets and query inputs come from the frozen episode
files; predictions come from the already-completed model run.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import sys
import tarfile
from pathlib import Path

import numpy as np
from PIL import Image


JFLUX = Path("/kmh-nfs-ssd-us-mount/code/xtiange/jflux2")
SCORER = JFLUX / "jflux2/fasteval/metrics_v3.py"
SCORER_SHA256 = "a82740a2aefc59062210dc0fadf3199c7fc1a624629f6c0154020f05dd8b4787"
RESTORATION = {"colorization", "denoising", "deblurring", "inpainting", "super_resolution"}
CONDITIONS = ("real", "scrambled", "grey", "oracle")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


class EpisodeStore:
    def __init__(self, directory: Path | None, archive: Path | None):
        self.directory = directory
        self.tar = tarfile.open(archive, "r") if archive else None

    def load(self, relative: str) -> tuple[np.ndarray, np.ndarray]:
        if self.directory:
            context = np.load(self.directory / relative, allow_pickle=False)
        else:
            member = self.tar.getmember("./" + relative.lstrip("./"))
            extracted = self.tar.extractfile(member)
            if extracted is None:
                raise FileNotFoundError(relative)
            context = np.load(io.BytesIO(extracted.read()), allow_pickle=False)
        with context as packed:
            return np.asarray(packed["query_input"], np.uint8), np.asarray(packed["query_target"], np.uint8)

    def close(self) -> None:
        if self.tar:
            self.tar.close()


def lpips_batches(pairs: list[tuple[np.ndarray, np.ndarray]], batch_size: int) -> list[float]:
    sys.path.insert(0, str(JFLUX))
    from jflux2.fasteval import metrics_v3 as metrics  # noqa: PLC0415
    import torch  # noqa: PLC0415

    net = metrics._lpips_net()  # noqa: SLF001 - exact pinned scorer implementation
    if net is None:
        raise RuntimeError("pinned LPIPS-Alex dependency is unavailable")
    values: list[float] = []
    for start in range(0, len(pairs), batch_size):
        group = pairs[start : start + batch_size]
        pred = np.stack([item[0] for item in group]).astype(np.float32) / 127.5 - 1.0
        targ = np.stack([item[1] for item in group]).astype(np.float32) / 127.5 - 1.0
        pred_t = torch.from_numpy(np.ascontiguousarray(pred.transpose(0, 3, 1, 2)))
        targ_t = torch.from_numpy(np.ascontiguousarray(targ.transpose(0, 3, 1, 2)))
        with torch.no_grad():
            batch = net(pred_t, targ_t).detach().cpu().numpy().reshape(-1)
        values.extend(float(value) for value in batch)
        print(f"LPIPS {min(start + len(group), len(pairs))}/{len(pairs)}", flush=True)
    return values


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--prediction-root", type=Path, required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--episode-dir", type=Path)
    source.add_argument("--episode-tar", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--bootstrap", type=int, default=20_000)
    args = parser.parse_args()

    if sha256(SCORER) != SCORER_SHA256:
        raise SystemExit("refusing rescore: pinned metrics_v3 SHA changed")
    report = json.loads(args.scores.read_text())
    selected = [row for row in report["results"] if row["task"] in RESTORATION]
    if len(selected) != 5 * 64 * 4:
        raise SystemExit(f"expected 1,280 restoration rows, found {len(selected)}")
    if len({row["episode_id"] for row in selected}) != len(selected):
        raise SystemExit("duplicate restoration episode IDs")

    store = EpisodeStore(args.episode_dir, args.episode_tar)
    try:
        pred_pairs: list[tuple[np.ndarray, np.ndarray]] = []
        episode_arrays: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for row in selected:
            query_input, target = store.load(row["file"])
            episode_arrays[row["episode_id"]] = (query_input, target)
            prediction_path = args.prediction_root / row["prediction"]
            if not prediction_path.is_file():
                raise FileNotFoundError(prediction_path)
            prediction = np.asarray(Image.open(prediction_path).convert("RGB"), np.uint8)
            if prediction.shape != target.shape:
                raise RuntimeError(f"shape mismatch for {row['episode_id']}: {prediction.shape} vs {target.shape}")
            pred_pairs.append((prediction, target))
        pred_values = lpips_batches(pred_pairs, args.batch_size)

        baseline_rows = [row for row in selected if row["condition"] == "real"]
        baseline_pairs = [episode_arrays[row["episode_id"]] for row in baseline_rows]
        baseline_values = lpips_batches(baseline_pairs, args.batch_size)
    finally:
        store.close()

    baseline = {
        (row["task"], int(row.get("q", row.get("query_index", int(row["episode_id"].split("__q")[1].split("__")[0]))))): value
        for row, value in zip(baseline_rows, baseline_values, strict=True)
    }
    replacements = {}
    for row, value in zip(selected, pred_values, strict=True):
        q = int(row.get("q", row.get("query_index", int(row["episode_id"].split("__q")[1].split("__")[0]))))
        replacement = dict(row)
        replacement.update({
            "primary_metric": "lpips_alex",
            "primary_lower_is_better": True,
            "primary_value": value,
            "copy_input_floor": baseline[(row["task"], q)],
            "perfect_target_ceiling": 0.0,
        })
        replacements[row["episode_id"]] = replacement
    report["results"] = [replacements.get(row["episode_id"], row) for row in report["results"]]

    for task in sorted(RESTORATION):
        for condition in CONDITIONS:
            rows = [row for row in report["results"] if row["task"] == task and row["condition"] == condition]
            values = np.asarray([row["primary_value"] for row in rows], np.float64)
            rng = np.random.default_rng(0)
            boots = values[rng.integers(0, values.size, size=(args.bootstrap, values.size))].mean(axis=1)
            report["cells"][f"{task}@{condition}"] = {
                "task": task,
                "condition": condition,
                "primary_metric": "lpips_alex",
                "primary_lower_is_better": True,
                "mean": float(values.mean()),
                "ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
                "n": int(values.size),
                "n_nan": 0,
                "copy_input_mean": float(np.mean([row["copy_input_floor"] for row in rows])),
                "perfect_target_mean": 0.0,
            }

    report["perceptual_primary"] = True
    report["scorer"] = "jflux2.fasteval.metrics_v3 LPIPS-Alex batched equivalent"
    report["scorer_sha256"] = SCORER_SHA256
    report["perceptual_rescore"] = {
        "source_scores": str(args.scores),
        "source_scores_sha256": sha256(args.scores),
        "prediction_root": str(args.prediction_root),
        "episode_source": str(args.episode_dir or args.episode_tar),
        "batch_size": args.batch_size,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_suffix(args.out.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, args.out)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
