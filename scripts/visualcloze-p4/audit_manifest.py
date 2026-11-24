#!/usr/bin/env python3
"""Fail-closed audit for one frozen P-ladder manifest."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
from pathlib import Path


PACK_ID = "48e062452574ff0ded61c72664445d687ecd1c02"
CONDITIONS = {"real", "scrambled", "grey", "oracle"}
SOURCE_HASHES = {
    "episode_builder": "7b7eb18e98afb288e69e6394008d5a058729189592c26637b452e2e8c289a0c8",
    "pack_loader": "30a7630291e16a9a111309a775aa3e23e89bff2cd94a01f54127a456e9259a65",
    "conventions": "d4f9bce12ff179c0f85c86b312ebf2d85e2bc55a25bf5095956705a263e675f9",
    "scorer": "a82740a2aefc59062210dc0fadf3199c7fc1a624629f6c0154020f05dd8b4787",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--p", type=int, choices=(4, 8, 16), required=True)
    parser.add_argument("--verify-files", action="store_true")
    args = parser.parse_args()
    manifest_path = args.run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    episodes = manifest.get("episodes", [])
    assert manifest.get("pack_id") == PACK_ID
    assert manifest.get("records") == 960
    assert manifest.get("p") == args.p
    assert manifest.get("seed") == 42
    assert manifest.get("seed_rule") == "[seed, task_seed(task), q, P]"
    assert manifest.get("demo_source_policy") == "same"
    assert manifest.get("conventions") == "jflux2.dataloaders.multi_task_loader_v4"
    assert manifest.get("source_files_sha256") == SOURCE_HASHES
    assert len(episodes) == 3840
    assert len({row["episode_id"] for row in episodes}) == 3840
    assert {row["condition"] for row in episodes} == CONDITIONS
    assert {row["p"] for row in episodes} == {args.p}
    task_counts = collections.Counter(row["task"] for row in episodes)
    condition_counts = collections.Counter(row["condition"] for row in episodes)
    assert len(task_counts) == 15 and set(task_counts.values()) == {256}
    assert condition_counts == collections.Counter({c: 960 for c in CONDITIONS})
    # Prove no same-source draw fell back to the task-wide pool.
    assert all(len(row["demo_indices"]) == args.p for row in episodes)
    assert all(row["source"] for row in episodes)
    pack = json.loads((args.run_dir / "pack_samples.json").read_text())
    assert pack.get("pack_id") == PACK_ID
    by_record_id = {str(row["record_id"]): str(row.get("source")) for row in pack["records"]}
    assert all(
        all(by_record_id[str(record_id)] == row["source"] for record_id in row["demo_record_ids"])
        for row in episodes
    )
    if args.verify_files:
        for row in episodes:
            path = args.run_dir / row["file"]
            if sha256(path) != row["sha256"]:
                raise SystemExit(f"episode hash mismatch: {path}")
    report = {
        "status": "PASS",
        "manifest_sha256": sha256(manifest_path),
        "pack_id": PACK_ID,
        "p": args.p,
        "extrapolation": args.p in (8, 16),
        "episodes": len(episodes),
        "task_counts": dict(sorted(task_counts.items())),
        "condition_counts": dict(sorted(condition_counts.items())),
        "all_demos_same_source_as_query": True,
        "verified_episode_files": bool(args.verify_files),
    }
    output = args.run_dir / "manifest_audit.json"
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
