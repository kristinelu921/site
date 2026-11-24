#!/usr/bin/env python3
"""Convert the complete LVM JSONL journal into a strict terminal manifest."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

from PIL import Image


SOURCE_COMMIT = "b6de939ef0eb1ee6593445a7f5268145f338749b"
MODEL_REVISION = "aa5a9039273bd4221a0f69f5c943de1c56c39d89"
VQ_REVISION = "e4b5f5f83d9588ed52b6d1b65b91262f9f2d804b"
VQ_SHA256 = "fed960b9b88968dcaab28fd0cb28b786693b83b21fa93b1a1ad1177b3f7e9fd1"
ADAPTER = "native_p4_nine_frame_context_visual_tokens_v1"
SHARD_SHA256 = {
    "model-00001-of-00003.safetensors": "408ea2352575c13b1fde56c1046c5874ce6ad090dfc73bd20a40239666d9fb92",
    "model-00002-of-00003.safetensors": "fb5b969ad83107edb225755375eb1d2de20bc059dd6c0d78a2be60565a12cdd7",
    "model-00003-of-00003.safetensors": "cf66c81d2c97de9e10b64e3d2f6693e57f55e78b5ae50779e4f39f5b4bc2f00",
}
CONTRACTS = {"test": ("48e062452574ff0ded61c72664445d687ecd1c02", 3840)}


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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=sorted(CONTRACTS), required=True)
    parser.add_argument("--episode-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    pack_id, expected_count = CONTRACTS[args.split]

    manifest = json.loads((args.episode_run / "manifest.json").read_text())
    if manifest.get("pack_id") != pack_id or len(manifest.get("episodes", [])) != expected_count:
        raise SystemExit("frozen episode manifest contract mismatch")
    if manifest.get("seed") != 42 or manifest.get("p") != 4:
        raise SystemExit("frozen episode seed/P contract mismatch")
    episodes = {row["episode_id"]: row for row in manifest["episodes"]}
    if len(episodes) != expected_count:
        raise SystemExit("duplicate frozen episode IDs")

    journal_rows = [json.loads(line) for line in (args.output_dir / "inference.jsonl").read_text().splitlines() if line.strip()]
    journal = {row["episode_id"]: row for row in journal_rows}
    if len(journal_rows) != expected_count or len(journal) != expected_count or set(journal) != set(episodes):
        raise SystemExit("journal IDs/count do not exactly match the frozen manifest")
    backend = json.loads((args.output_dir / "backend.json").read_text())
    if backend.get("backend") != "tpu" or backend.get("device_count") != 4:
        raise SystemExit("LVM backend evidence is not the assigned four-device v4-8 TPU")
    seed_manifest = json.loads((args.output_dir / "resume_seed_manifest.json").read_text())
    seed_ids = set(seed_manifest["committed_episode_ids"])
    if len(seed_ids) != 14 or not seed_ids.issubset(episodes):
        raise SystemExit("resume-seed identity/count mismatch")

    results = []
    for index, episode in enumerate(manifest["episodes"], 1):
        row = journal[episode["episode_id"]]
        for field in ("task", "condition", "q"):
            if row.get(field) != episode.get(field):
                raise SystemExit(f"journal/episode mismatch for {episode['episode_id']}:{field}")
        if row.get("backend") != "tpu" or int(row.get("model_seed", -1)) != 0:
            raise SystemExit(f"backend/seed mismatch for {episode['episode_id']}")
        relative = Path(episode["task"]) / episode["condition"] / f"q{int(episode['q']):02d}.png"
        prediction = args.output_dir / relative
        if not prediction.is_file():
            raise SystemExit(f"missing prediction: {prediction}")
        with Image.open(prediction) as image:
            if image.convert("RGB").size != (256, 256):
                raise SystemExit(f"bad prediction shape: {prediction}")
        results.append({
            "episode_id": episode["episode_id"],
            "prediction": str(relative),
            "prediction_sha256": sha256(prediction),
            "encode_seconds": row.get("encode_seconds"),
            "generation_seconds": row.get("generation_seconds"),
            "decode_seconds": row.get("decode_seconds"),
            "total_seconds": row.get("total_seconds"),
            "completed_utc": row.get("completed_utc"),
        })
        if index % 256 == 0:
            print(f"{args.split}: audited {index}/{expected_count}", flush=True)

    report = {
        "schema": "lvm_visualcloze_terminal_inference_v1",
        "status": "terminal_all_images_and_metadata_committed",
        "finished_audit_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "split": args.split,
        "pack_id": pack_id,
        "model": "LVM 7B",
        "source_commit": SOURCE_COMMIT,
        "model_revision": MODEL_REVISION,
        "checkpoint_shards_sha256": SHARD_SHA256,
        "vq_revision": VQ_REVISION,
        "vq_sha256": VQ_SHA256,
        "adapter": ADAPTER,
        "backend": "TPU",
        "runtime_device_count": 4,
        "resume_seed": {
            "episodes": 14,
            "origin_accelerator": "v6e-8",
            "origin_device_count": 8,
            "resume_accelerator": "v4-8",
            "resume_device_count": 4,
            "manifest_sha256": sha256(args.output_dir / "resume_seed_manifest.json"),
        },
        "model_seed": 0,
        "results": results,
    }
    json_dump(args.output_dir / "inference.json", report)
    print(f"wrote {args.output_dir / 'inference.json'} ({len(results)} results)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
