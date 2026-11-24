#!/usr/bin/env python3
"""Turn the append-only LVM journal into a strict terminal inference manifest."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episode-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((args.episode_run / "manifest.json").read_text())
    expected = {row["episode_id"]: row for row in manifest["episodes"]}
    journal_rows = [json.loads(line) for line in (args.output_dir / "inference.jsonl").read_text().splitlines() if line.strip()]
    latest = {row["episode_id"]: row for row in journal_rows}
    if set(latest) != set(expected) or len(latest) != 960:
        raise SystemExit(f"inference incomplete: {len(latest)}/960")
    results = []
    for episode in manifest["episodes"]:
        row = latest[episode["episode_id"]]
        prediction = args.output_dir / row["prediction"]
        if not prediction.is_file() or sha256(prediction) != row["prediction_sha256"]:
            raise SystemExit(f"prediction mismatch: {episode['episode_id']}")
        results.append(row)
    report = {
        "schema": "lvm_visualcloze_p1_real_inference_v1",
        "status": "terminal_all_images_and_metadata_committed",
        "pack_id": manifest["pack_id"], "p": 1, "condition": "real",
        "model": "LVM 7B", "source_commit": manifest["model_source_commit"],
        "model_revision": manifest["model_revision"], "vq_revision": manifest["vq_revision"],
        "adapter": "native_p1_three_frame_context_visual_tokens_v1",
        "preprocess": manifest["preprocess"], "postprocess": manifest["postprocess"],
        "results": results,
    }
    temporary = args.output_dir / "inference.tmp.json"
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, args.output_dir / "inference.json")
    print("finalized 960/960")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
