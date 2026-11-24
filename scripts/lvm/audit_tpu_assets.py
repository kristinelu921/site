#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def audit_pack(path: Path, pack_id: str, count: int, json_hash: str, npz_hash: str) -> dict:
    assert sha256(path / "samples.json") == json_hash
    assert sha256(path / "samples.npz") == npz_hash
    meta = json.loads((path / "samples.json").read_text())
    assert meta["pack_id"] == pack_id
    assert len(meta["records"]) == count
    return {"pack_id": pack_id, "records": count}


def audit_manifest(path: Path, pack_id: str, episodes: int, lineage_a: bool) -> dict:
    data = json.loads(path.read_text())
    assert data["pack_id"] == pack_id
    assert len(data["episodes"]) == episodes
    assert data["p"] == 2 and data["seed"] == 42
    assert data["demo_source_policy"] == "same"
    counts = Counter(x["condition"] for x in data["episodes"])
    assert set(counts) == {"real", "scrambled", "grey", "oracle"}
    assert len(set(counts.values())) == 1
    if lineage_a:
        assert all(x.get("native_lineage_a_target") is True for x in data["episodes"])
    return {"episodes": episodes, "conditions": dict(counts)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root
    result = {
        "test_pack": audit_pack(root / "packs/test", "48e062452574ff0ded61c72664445d687ecd1c02", 960,
                                "97d0a0f0f17f2901a457782226c4103b5e7d503ff22e66dd199fcb0463f1d1e4",
                                "adf91884ed38e7612113fdd3f61b17745cab95e1e49d733e8d654e1c7e440508"),
        "val_pack": audit_pack(root / "packs/validation", "9e77a92f82d0e08c12f540d5c15c2b15e6c503a2", 128,
                               "20dd02977a77bf19157c99bb3d6f462d0f12e0652fef5d6ee4b41343aec896cb",
                               "a6d6c8c343a4cfd642e9f576df5af5441a9270e48a08da595d19db3e5aa6f08e"),
        "test_manifest": audit_manifest(root / "episodes/test/manifest.json", "48e062452574ff0ded61c72664445d687ecd1c02", 3840, False),
        "val_manifest": audit_manifest(root / "episodes/validation/manifest.json", "9e77a92f82d0e08c12f540d5c15c2b15e6c503a2", 512, True),
    }
    conversion = json.loads((root / "model/conversion_manifest.json").read_text())
    assert conversion["tensor_count"] == 291
    assert conversion["all_roundtrips_exact"] is True
    result["conversion"] = {"tensors": 291, "parameters": conversion["parameter_count"], "roundtrip": "exact"}
    (root / "logs/input_audit.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": "PASS", "episodes": 4352, "tensors": 291}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
