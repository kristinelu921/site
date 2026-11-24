#!/usr/bin/env python3
"""Full hash and episode-condition audit for the TPU Painter inputs."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np


CONTRACTS = {
    "test": {
        "pack_id": "48e062452574ff0ded61c72664445d687ecd1c02",
        "records": 960,
        "episodes": 3840,
        "tasks": 15,
        "per_task": 64,
    },
    "val": {
        "pack_id": "9e77a92f82d0e08c12f540d5c15c2b15e6c503a2",
        "records": 128,
        "episodes": 512,
        "tasks": 4,
        "per_task": 32,
    },
}
CONDITIONS = ("real", "scrambled", "grey", "oracle")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def audit(split: str, run_dir: Path) -> dict:
    contract = CONTRACTS[split]
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["pack_id"] == contract["pack_id"]
    assert manifest["records"] == contract["records"]
    assert manifest["seed"] == 42 and manifest["p"] == 2
    assert manifest["seed_rule"] == "[seed, task_seed(task), q, P]"
    assert manifest["demo_source_policy"] == "same"
    episodes = manifest["episodes"]
    assert len(episodes) == contract["episodes"]
    assert len({row["episode_id"] for row in episodes}) == len(episodes)
    task_counts = Counter(row["task"] for row in episodes)
    condition_counts = Counter(row["condition"] for row in episodes)
    assert len(task_counts) == contract["tasks"]
    assert set(task_counts.values()) == {contract["per_task"] * 4}
    assert set(condition_counts) == set(CONDITIONS)
    assert set(condition_counts.values()) == {contract["records"]}
    rows = {(row["task"], int(row["q"]), row["condition"]): row for row in episodes}

    checked = 0
    for task in sorted(task_counts):
        for query in range(contract["per_task"]):
            group = [rows[(task, query, condition)] for condition in CONDITIONS]
            assert len({tuple(row["demo_indices"]) for row in group}) == 1
            assert len({tuple(row["demo_record_ids"]) for row in group}) == 1
            assert len({row["record_id"] for row in group}) == 1
            assert len({row["source"] for row in group}) == 1
            arrays = {}
            for row in group:
                path = run_dir / row["file"]
                assert sha256(path) == row["sha256"], row["episode_id"]
                with np.load(path) as packed:
                    assert packed["demo_inputs"].shape == (2, 256, 256, 3)
                    assert packed["demo_targets"].shape == (2, 256, 256, 3)
                    assert packed["query_input"].shape == (256, 256, 3)
                    assert packed["query_target"].shape == (256, 256, 3)
                    assert all(packed[key].dtype == np.uint8 for key in packed.files)
                    arrays[row["condition"]] = {key: packed[key].copy() for key in packed.files}
                checked += 1
            for condition in CONDITIONS[1:]:
                assert np.array_equal(arrays["real"]["query_input"], arrays[condition]["query_input"])
                assert np.array_equal(arrays["real"]["query_target"], arrays[condition]["query_target"])
            assert np.array_equal(arrays["real"]["demo_inputs"], arrays["scrambled"]["demo_inputs"])
            assert np.array_equal(
                arrays["real"]["demo_targets"][::-1], arrays["scrambled"]["demo_targets"]
            )
            assert np.all(arrays["grey"]["demo_inputs"] == 128)
            assert np.all(arrays["grey"]["demo_targets"] == 128)
            assert np.array_equal(
                arrays["oracle"]["demo_inputs"],
                np.repeat(arrays["oracle"]["query_input"][None], 2, axis=0),
            )
            assert np.array_equal(
                arrays["oracle"]["demo_targets"],
                np.repeat(arrays["oracle"]["query_target"][None], 2, axis=0),
            )
    return {
        "split": split,
        "pack_id": manifest["pack_id"],
        "records": manifest["records"],
        "episodes": len(episodes),
        "tasks": sorted(task_counts),
        "conditions": dict(sorted(condition_counts.items())),
        "episode_files_checked": checked,
        "sha256_checked": True,
        "scramble_contract": "P=2 swap of already-coded real demo targets",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-dir", type=Path, required=True)
    parser.add_argument("--val-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = {
        "schema": "painter_tpu_input_audit_v1",
        "test": audit("test", args.test_dir),
        "val": audit("val", args.val_dir),
    }
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(rendered)
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

