#!/usr/bin/env python3
"""Freeze, audit, score, and visualize the clean LVM P=1/real test run.

The model context for every query is exactly three 256px RGB frames:
    demo input, demo target, query input
The fourth frame is generated and compared with the episode-coded query target.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


JFLUX = Path("/kmh-nfs-ssd-us-mount/code/xtiange/jflux2")
PACK_DIR = Path("/kmh-nfs-ssd-us-mount/logs/xtiange/mt_eval_packs/test_s64_px256_v5")
PACK_ID = "48e062452574ff0ded61c72664445d687ecd1c02"
PACK_SHA256 = {
    "samples.json": "97d0a0f0f17f2901a457782226c4103b5e7d503ff22e66dd199fcb0463f1d1e4",
    "samples.npz": "adf91884ed38e7612113fdd3f61b17745cab95e1e49d733e8d654e1c7e440508",
}
SOURCE_FILES = {
    "episode_builder": JFLUX / "jflux2/analysis/t16_offline_eval_v1.py",
    "pack_loader": JFLUX / "jflux2/analysis/ttt_ladder_v1.py",
    "conventions": JFLUX / "jflux2/dataloaders/multi_task_loader_v4.py",
    "scorer": JFLUX / "jflux2/fasteval/metrics_v3.py",
}
SOURCE_SHA256 = {
    "episode_builder": "7b7eb18e98afb288e69e6394008d5a058729189592c26637b452e2e8c289a0c8",
    "pack_loader": "30a7630291e16a9a111309a775aa3e23e89bff2cd94a01f54127a456e9259a65",
    "conventions": "d4f9bce12ff179c0f85c86b312ebf2d85e2bc55a25bf5095956705a263e675f9",
    "scorer": "a82740a2aefc59062210dc0fadf3199c7fc1a624629f6c0154020f05dd8b4787",
}
TASKS = (
    "relative_depth", "surface_normals", "semantic_segmentation",
    "instance_segmentation", "panoptic_segmentation", "edge_detection",
    "saliency_detection", "human_pose_2d", "face_landmarks",
    "object_keypoints", "colorization", "denoising", "deblurring",
    "inpainting", "super_resolution",
)
MODEL_SOURCE_COMMIT = "b6de939ef0eb1ee6593445a7f5268145f338749b"
MODEL_REVISION = "aa5a9039273bd4221a0f69f5c943de1c56c39d89"
VQ_REVISION = "e4b5f5f83d9588ed52b6d1b65b91262f9f2d804b"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def dump_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def git_sha(path: Path) -> str:
    return subprocess.check_output(
        ["git", "-c", f"safe.directory={path}", "-C", str(path), "rev-parse", "HEAD"],
        text=True,
    ).strip()


def source_hashes() -> dict[str, str]:
    actual = {name: sha256(path) for name, path in SOURCE_FILES.items()}
    if actual != SOURCE_SHA256:
        raise SystemExit(f"refusing changed episode/scoring source: {actual}")
    return actual


def official_preprocess(frame: np.ndarray) -> np.ndarray:
    """Exact released LVM image loader: RGB, PIL bicubic 256-square, float32 / 255."""
    image = Image.fromarray(np.asarray(frame, np.uint8)).convert("RGB")
    image = image.resize((256, 256), Image.Resampling.BICUBIC)
    return np.asarray(image, dtype=np.float32) / 255.0


def prepare(args: argparse.Namespace) -> int:
    sys.path.insert(0, str(JFLUX))
    from jflux2.analysis import t16_offline_eval_v1 as T16  # noqa: PLC0415
    from jflux2.analysis.ttt_ladder_v1 import load_pack  # noqa: PLC0415
    from jflux2.dataloaders import multi_task_loader_v4 as CV4  # noqa: PLC0415

    T16.CV = CV4
    for name, expected in PACK_SHA256.items():
        actual = sha256(args.pack / name)
        if actual != expected:
            raise SystemExit(f"pack hash mismatch for {name}: {actual}")
    pack = load_pack(str(args.pack))
    if pack.get("pack_id") != PACK_ID or len(pack["records"]) != 960:
        raise SystemExit("not the pinned supervision-v5 960-record test pack")

    out = args.out.resolve()
    episode_dir = out / "episodes"
    episode_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, object] = {
        "schema": "lvm_visualcloze_p1_real_test_v1",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "pack_id": PACK_ID,
        "pack_sha256": PACK_SHA256,
        "records": 960,
        "split": "test",
        "tasks": list(TASKS),
        "p": 1,
        "condition": "real",
        "conditions": ["real"],
        "seed": 42,
        "seed_rule": "[seed, task_seed(task), q, P]",
        "demo_source_policy": "same",
        "episode_order": ["demo_input", "demo_target", "query_input", "generated_query_target"],
        "context_frames": 3,
        "context_tokens": 768,
        "generated_frames": 1,
        "generated_tokens": 256,
        "source_files_sha256": source_hashes(),
        "jflux_git_sha": git_sha(JFLUX),
        "model_source_commit": MODEL_SOURCE_COMMIT,
        "model_revision": MODEL_REVISION,
        "vq_revision": VQ_REVISION,
        "preprocess": {
            "source": "official LVM evaluation/vqlm_demo/utils.py::read_image_to_tensor",
            "color": "PIL RGB",
            "resize": [256, 256],
            "resize_resample": "Pillow default for RGB = BICUBIC",
            "range": "float32 [0,1] via uint8/255",
            "pack_is_already_256_square": True,
        },
        "postprocess": {
            "source": "official LVM evaluation/vqlm_demo/generate_videos.py and batch_generation.py",
            "decode": "VQGAN decode_code, clamp [0,1]",
            "quantize": "(clipped_float * 255).astype(uint8); truncation, no rounding",
            "resize": None,
            "native_output": [256, 256],
        },
        "episodes": [],
    }
    palette_audit: list[dict[str, object]] = []

    for task in TASKS:
        global_indices = [i for i, row in enumerate(pack["records"]) if row["task"] == task]
        if len(global_indices) != 64:
            raise SystemExit(f"expected 64 records for {task}, got {len(global_indices)}")
        images = np.asarray(pack["images"][global_indices], np.uint8)
        masks = np.asarray(pack["masks"][global_indices], np.uint8)
        sources = [str(pack["records"][i].get("source")) for i in global_indices]
        candidates = {s: [j for j, x in enumerate(sources) if x == s] for s in set(sources)}
        palette = CV4.family_of(task) == "palette"

        for q in range(64):
            rng = np.random.default_rng([42, T16.task_seed(task), q, 1])
            convention_id = 0
            if palette:
                pool = CV4.convention_pools(task)[1]
                convention_id = int(pool[int(rng.integers(0, len(pool)))])
            built, demo_indices, query_target = T16.build_conditions(
                images, masks, task, q, 1, rng, cid=convention_id,
                which=("real",), candidates=candidates[sources[q]],
            )
            if len(demo_indices) != 1 or int(demo_indices[0]) == q:
                raise SystemExit(f"invalid singleton demonstration for {task} q{q}")
            demo_input = np.rint(np.transpose(built["real"][0][0], (1, 2, 0)) * 255).astype(np.uint8)
            demo_target = np.rint(np.transpose(built["real"][0][1], (1, 2, 0)) * 255).astype(np.uint8)
            query_target = np.asarray(query_target, np.uint8)

            scored_positions = None
            palette_map = None
            if palette:
                raw_tiles = np.stack([masks[int(demo_indices[0])], masks[q]]).astype(np.float32) / 255.0
                original, assigned = CV4.palette_mapping_np(raw_tiles, convention_id)
                demo_positions = sorted(
                    int(x) for x in np.unique(CV4._posmap(CV4._packed(demo_target.astype(np.int64)))) if x > 0
                )
                query_positions = sorted(
                    int(x) for x in np.unique(CV4._posmap(CV4._packed(query_target.astype(np.int64)))) if x > 0
                )
                scored_positions = demo_positions
                palette_map = {
                    "original_positions": [int(x) for x in original],
                    "assigned_positions": [int(x) for x in assigned],
                    "demo_scored_positions": demo_positions,
                    "query_positions": query_positions,
                    "uncovered_query_positions": sorted(set(query_positions) - set(demo_positions)),
                }

            episode_id = f"{task}__q{q:02d}__P1__real"
            path = episode_dir / f"{episode_id}.npz"
            np.savez_compressed(
                path,
                demo_inputs=demo_input[None],
                demo_targets=demo_target[None],
                query_input=images[q],
                query_target=query_target,
            )
            row = {
                "episode_id": episode_id,
                "file": str(path.relative_to(out)),
                "sha256": sha256(path),
                "task": task,
                "q": q,
                "p": 1,
                "condition": "real",
                "source": sources[q],
                "record_id": str(pack["records"][global_indices[q]].get("record_id")),
                "demo_indices": [int(demo_indices[0])],
                "demo_record_ids": [str(pack["records"][global_indices[int(demo_indices[0])]].get("record_id"))],
                "convention_id": convention_id,
                "palette_task": palette,
                "scored_positions": scored_positions,
                "palette_map": palette_map,
            }
            manifest["episodes"].append(row)
            if palette and q < 3:
                palette_audit.append({key: row[key] for key in (
                    "episode_id", "task", "q", "source", "demo_indices", "convention_id",
                    "scored_positions", "palette_map",
                )})
        print(f"prepared {task}: 64/64", flush=True)

    # q-major order gives the user one example from every task immediately
    # after the first compile, without changing any episode draw or content.
    manifest["episodes"].sort(key=lambda row: (int(row["q"]), TASKS.index(row["task"])))
    dump_json(out / "manifest.json", manifest)
    dump_json(out / "palette_audit_examples.json", {"schema": "p1_palette_audit_v1", "examples": palette_audit})
    print(f"prepared {len(manifest['episodes'])} P=1 real episodes in {out}", flush=True)
    return 0


def audit(args: argparse.Namespace) -> int:
    manifest = json.loads((args.run / "manifest.json").read_text())
    if manifest.get("schema") != "lvm_visualcloze_p1_real_test_v1":
        raise SystemExit("manifest schema mismatch")
    if manifest.get("pack_id") != PACK_ID or manifest.get("p") != 1 or manifest.get("conditions") != ["real"]:
        raise SystemExit("P=1/real identity mismatch")
    if manifest.get("source_files_sha256") != source_hashes():
        raise SystemExit("source contract changed")
    episodes = manifest.get("episodes", [])
    if len(episodes) != 960 or len({row["episode_id"] for row in episodes}) != 960:
        raise SystemExit("expected 960 unique episodes")
    counts = Counter(row["task"] for row in episodes)
    if counts != Counter({task: 64 for task in TASKS}):
        raise SystemExit(f"task counts mismatch: {counts}")

    pack = json.loads((args.pack / "samples.json").read_text())
    records = defaultdict(list)
    for row in pack["records"]:
        records[row["task"]].append(row)
    byte_count = 0
    palette_count = 0
    official_resize_max_abs = 0.0
    for index, row in enumerate(episodes, 1):
        if row["condition"] != "real" or row["p"] != 1:
            raise SystemExit(f"non-P1 real episode: {row['episode_id']}")
        if len(row["demo_indices"]) != 1 or row["q"] in row["demo_indices"]:
            raise SystemExit(f"bad demo set: {row['episode_id']}")
        query_record = records[row["task"]][row["q"]]
        demo_record = records[row["task"]][row["demo_indices"][0]]
        if str(query_record["source"]) != row["source"] or str(demo_record["source"]) != row["source"]:
            raise SystemExit(f"cross-source episode: {row['episode_id']}")
        path = args.run / row["file"]
        if sha256(path) != row["sha256"]:
            raise SystemExit(f"episode hash mismatch: {row['episode_id']}")
        byte_count += path.stat().st_size
        with np.load(path, allow_pickle=False) as data:
            if set(data.files) != {"demo_inputs", "demo_targets", "query_input", "query_target"}:
                raise SystemExit(f"unexpected arrays: {row['episode_id']}")
            if data["demo_inputs"].shape != (1, 256, 256, 3) or data["demo_targets"].shape != (1, 256, 256, 3):
                raise SystemExit(f"bad demo shape: {row['episode_id']}")
            if data["query_input"].shape != (256, 256, 3) or data["query_target"].shape != (256, 256, 3):
                raise SystemExit(f"bad query shape: {row['episode_id']}")
            if any(data[name].dtype != np.uint8 for name in data.files):
                raise SystemExit(f"non-uint8 episode: {row['episode_id']}")
            for frame in (data["demo_inputs"][0], data["demo_targets"][0], data["query_input"]):
                official = official_preprocess(frame)
                direct = frame.astype(np.float32) / 255.0
                official_resize_max_abs = max(official_resize_max_abs, float(np.max(np.abs(official - direct))))
        if row["palette_task"]:
            palette_count += 1
            if row["scored_positions"] != row["palette_map"]["demo_scored_positions"]:
                raise SystemExit(f"palette coverage mismatch: {row['episode_id']}")
        elif row["scored_positions"] is not None or row["palette_map"] is not None:
            raise SystemExit(f"spurious palette metadata: {row['episode_id']}")
        if index % 128 == 0:
            print(f"audited {index}/960", flush=True)
    if official_resize_max_abs != 0.0:
        raise SystemExit(f"official same-size resize changed pack pixels: {official_resize_max_abs}")
    result = {
        "status": "PASS",
        "episodes": 960,
        "tasks": 15,
        "p": 1,
        "conditions": ["real"],
        "palette_episodes": palette_count,
        "audited_episode_bytes": byte_count,
        "official_preprocess_same_size_max_abs": official_resize_max_abs,
        "official_postprocess": "clip [0,1], multiply 255, truncate uint8, no output resize",
    }
    dump_json(args.output, result)
    print(json.dumps(result, sort_keys=True))
    return 0


def contact_sheet(args: argparse.Namespace) -> int:
    manifest = json.loads((args.run / "manifest.json").read_text())
    wanted = set(args.tasks.split(",")) if args.tasks else set(TASKS)
    rows = [row for row in manifest["episodes"] if row["task"] in wanted and row["q"] == args.q]
    if not rows:
        raise SystemExit("no selected examples")
    tile, label_h, gap = 256, 34, 10
    columns = ("Demo input", "Demo target", "Query input", "LVM output", "Query target")
    sheet = Image.new("RGB", (len(columns) * tile + 2 * gap, len(rows) * (tile + label_h) + label_h), "white")
    draw = ImageDraw.Draw(sheet)
    # Pillow 10.0 (the official run's pinned image stack) does not yet accept
    # the optional size argument. Tile pixels are never resized here.
    font = ImageFont.load_default()
    for c, label in enumerate(columns):
        draw.text((gap + c * tile + 6, 8), label, fill="black", font=font)
    for r, row in enumerate(rows):
        with np.load(args.run / row["file"], allow_pickle=False) as data:
            pred_path = args.predictions / row["task"] / "real" / f"q{row['q']:02d}.png"
            if not pred_path.exists():
                continue
            panels = [
                Image.fromarray(data["demo_inputs"][0]),
                Image.fromarray(data["demo_targets"][0]),
                Image.fromarray(data["query_input"]),
                Image.open(pred_path).convert("RGB"),
                Image.fromarray(data["query_target"]),
            ]
        y = label_h + r * (tile + label_h)
        for c, panel in enumerate(panels):
            if panel.size != (256, 256):
                raise SystemExit(f"non-native panel size for {row['episode_id']}: {panel.size}")
            sheet.paste(panel, (gap + c * tile, y))
        map_note = ""
        if row["palette_task"]:
            map_note = f"  demo-visible palette positions: {row['scored_positions']}"
        draw.text((gap + 4, y + tile + 6), f"{row['task']} q{row['q']:02d}{map_note}", fill="black", font=font)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(args.output)
    print(args.output)
    return 0


def score(args: argparse.Namespace) -> int:
    sys.path.insert(0, str(JFLUX))
    from jflux2.fasteval import metrics_v3 as metrics  # noqa: PLC0415

    manifest = json.loads((args.run / "manifest.json").read_text())
    episodes = manifest["episodes"]
    if len(episodes) != 960:
        raise SystemExit("refusing incomplete manifest")
    results = []
    for index, row in enumerate(episodes, 1):
        prediction_path = args.predictions / row["task"] / "real" / f"q{row['q']:02d}.png"
        if not prediction_path.exists():
            raise SystemExit(f"missing prediction: {prediction_path}")
        with np.load(args.run / row["file"], allow_pickle=False) as data:
            query_input = np.asarray(data["query_input"], np.uint8)
            target = np.asarray(data["query_target"], np.uint8)
        prediction = np.asarray(Image.open(prediction_path).convert("RGB"), np.uint8)
        kwargs = {
            "source": row["source"],
            "scored_positions": row["scored_positions"],
            "perceptual": bool(args.perceptual),
        }
        scored = metrics.score_pair_v3(row["task"], prediction, target, **kwargs)
        copied = metrics.score_pair_v3(row["task"], query_input, target, **kwargs)
        perfect = metrics.score_pair_v3(row["task"], target, target, **kwargs)
        results.append({
            **row,
            "prediction": str(prediction_path.relative_to(args.predictions)),
            "prediction_sha256": sha256(prediction_path),
            "primary_metric": scored["primary_metric"],
            "primary_lower_is_better": bool(scored["primary_lower_is_better"]),
            "primary_value": float(scored["primary_value"]),
            "copy_input_floor": float(copied["primary_value"]),
            "perfect_target_ceiling": float(perfect["primary_value"]),
            "l1_01": float(scored["l1_01"]),
        })
        if index % 64 == 0:
            print(f"scored {index}/960", flush=True)
    cells = {}
    for task in TASKS:
        task_rows = [row for row in results if row["task"] == task]
        values = np.asarray([row["primary_value"] for row in task_rows], np.float64)
        finite = values[np.isfinite(values)]
        cells[task] = {
            "task": task,
            "condition": "real",
            "p": 1,
            "primary_metric": task_rows[0]["primary_metric"],
            "primary_lower_is_better": task_rows[0]["primary_lower_is_better"],
            "mean": float(finite.mean()) if finite.size else float("nan"),
            "n": int(finite.size),
            "n_nan": int(values.size - finite.size),
            "copy_input_mean": float(np.nanmean([row["copy_input_floor"] for row in task_rows])),
            "perfect_target_mean": float(np.nanmean([row["perfect_target_ceiling"] for row in task_rows])),
        }
    report = {
        "schema": "lvm_visualcloze_p1_real_scores_v1",
        "pack_id": PACK_ID,
        "p": 1,
        "condition": "real",
        "scorer_sha256": SOURCE_SHA256["scorer"],
        "coverage_aware_palette_scoring": True,
        "perceptual": bool(args.perceptual),
        "cells": cells,
        "results": results,
    }
    dump_json(args.output, report)
    print(args.output)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("prepare")
    p.add_argument("--pack", type=Path, default=PACK_DIR)
    p.add_argument("--out", type=Path, required=True)
    p.set_defaults(func=prepare)
    p = commands.add_parser("audit")
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--pack", type=Path, default=PACK_DIR)
    p.add_argument("--output", type=Path, required=True)
    p.set_defaults(func=audit)
    p = commands.add_parser("contact-sheet")
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--predictions", type=Path, required=True)
    p.add_argument("--q", type=int, default=0)
    p.add_argument("--tasks", default="")
    p.add_argument("--output", type=Path, required=True)
    p.set_defaults(func=contact_sheet)
    p = commands.add_parser("score")
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--predictions", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--perceptual", action="store_true")
    p.set_defaults(func=score)
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
