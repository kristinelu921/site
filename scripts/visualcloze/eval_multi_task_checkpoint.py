#!/usr/bin/env python3
"""Standalone TPU evaluator for `multi_task_trainer_optimize_text_token_only` checkpoints.

WHY THIS EXISTS (three hard blockers in the trainer's own validation path)
-------------------------------------------------------------------------
1. `validate_task_list()` RAISES when validation is enabled and the task list holds a
   TRAIN_ONLY task. The 24-task arms hold four of them (illumination,
   intrinsic_decomposition, reflectance, shadow_detection), so the trainer cannot be
   asked to validate those checkpoints at all.
2. Restoring a bank through `restore_checkpoint` runs `assert_manifest_compatible`,
   which forces the CONFIG task list to match the checkpoint manifest ORDER exactly.
   That makes "just hand it a 20-task val list" impossible.
3. `_select_validation_samples_for_artifacts` RAISES if any index in `range(num_tasks)`
   never appears in the val stream -- again the four train-only rows.

This evaluator sidesteps all three by building the bank with `TaskEmbeddings.set_tokens`
(numerically identical to restore -- there is exactly ONE `TaskParam` leaf in the bundle,
`bundle.task_embeddings.tokens`, and restore writes `jnp.asarray(host, dtype=fp32)` into
it, which is what `set_tokens` does) and by driving the forward itself.

WHAT IS REUSED VERBATIM FROM THE TRAINER (never reimplemented)
--------------------------------------------------------------
  `_normalize_validation_latents`  RGB[0,1] -> [-1,1] -> ONE `ae.encode` over concat(img,mask)
  `_run_validation_forward`        mesh placement + prc_img + 50-step Euler + scatter + decode
  `_validation_timesteps`          get_schedule(max(50, n), image_seq_len)
  `_put_validation_value_to_mesh`  the batch-1 mesh guard (commit 2c6a69c)
  `_save_rgb_image` semantics      clip[-1,1] -> 127.5*(x+1) -> uint8
  the metric                       jnp.mean(|decoded - ref_decoded|, axis=(1,2,3))

`ref_decoded` is `ae.decode(mask_latent)` -- the VAE ROUND-TRIP of the target, not the raw
target. There is therefore a nonzero VAE floor baked into the headline number; `l1_floor`
below measures it.

DETERMINISM
-----------
Inference has NO RNG: the ODE starts from the input-image latent (`batched_prc_img(
image_latent)`), `ae.encode` uses the posterior mean only, and the Euler loop is plain
arithmetic. So given identical inputs, cross-arm deltas are 100% attributable to the bank.

The LOADER, however, is NOT order-deterministic: `vgi_tfds.py` hardcodes
`interleave(..., deterministic=False)` and `_common.apply_threading_options` sets
`options.deterministic = False`. Worse, the per-record renderer is seeded POSITIONALLY
(`counter + seed`), so a reordering re-renders `denoising` / `inpainting` with different
randomness. Therefore this tool FREEZES a sample pack to disk once (records chosen by
min-hash over `record_id`, which depends only on the SET of ids, never on arrival order)
and evaluates every arm against the byte-identical frozen tensors. That is the only way
the arms become comparable.

USAGE
-----
  # one checkpoint
  python tools/eval_multi_task_checkpoint.py \
      --ckpt-dir gs://kmh-gcp-us-east5/xtiange/mt_both20_lr1e4_sample/checkpoints \
      --out-dir /kmh-nfs-ssd-us-mount/logs/xtiange/mt_eval_20k_v1 --samples-per-task 8

  # several checkpoints, ONE base-model load (what the queued job does)
  python tools/eval_multi_task_checkpoint.py --ckpt-dir A --ckpt-dir B ... --out-dir DIR

The daemon runs `<python> <script> --workdir=<dir> <args>`, so an injected `--workdir` is
stripped (logged only) before the strict argparse -- same contract as
`compute/uinfra_wiring_entry.py`.
"""
from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import os
import re
import sys
import time
import traceback
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import numpy as np  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

import jflux2.trainers.multi_task_trainer_optimize_text_token_only as T  # noqa: E402

# --------------------------------------------------------------------------- #
# Task-set constants
# --------------------------------------------------------------------------- #
# The both-split intersection, in the `mt_both20_lr1e4_sample` manifest order (verified
# offline against the real sidecar). Used as the CANONICAL row order for every arm: each
# arm contributes its 20 named rows into this order via `task_index_for_name`, so a
# 24-row arm and a 20-row arm end up byte-comparable without any manifest-order restore
# constraint.
CANONICAL_EVALUABLE_TASKS: tuple[str, ...] = (
    "metric_depth",
    "relative_depth",
    "surface_normals",
    "scene_coordinates",
    "semantic_segmentation",
    "instance_segmentation",
    "panoptic_segmentation",
    "edge_detection",
    "saliency_detection",
    "human_pose_2d",
    "face_landmarks",
    "object_keypoints",
    "optical_flow",
    "scene_flow",
    "stereo_disparity",
    "colorization",
    "denoising",
    "deblurring",
    "inpainting",
    "super_resolution",
)

# Held by the 24-task arms, zero records in the vgi test split (fullpass v5 section [3]).
NOT_EVALUABLE_TRAIN_ONLY_REASON = (
    "train-only task: 0 records in the vgi test split; the train sources have no test "
    "counterpart, so the trained row cannot be scored (reported, never silently skipped)"
)
# In the test split but in NO checkpoint bank.
NOT_EVALUABLE_TEST_ONLY_REASON = (
    "test-only task: no embedding row exists in ANY of the four checkpoints (it was "
    "absent from every training task list), so it is structurally unreachable"
)

DEFAULT_DATA_DIR = "data/kristine/vgi_datasets/vgi_rgb"
DEFAULT_FLUX_WEIGHTS = "pretrained-ckpts/xtiange/jax_weights/4B_base/jflux2_transformer.safetensors"
DEFAULT_FLUX_CONFIG = "pretrained-ckpts/xtiange/jax_weights/4B/transformer_config.json"
DEFAULT_VAE_WEIGHTS = "pretrained-ckpts/xtiange/jax_weights/4B/jflux2_vae.safetensors"
DEFAULT_VAE_CONFIG = "pretrained-ckpts/xtiange/jax_weights/4B/vae_config.json"

PACK_SCHEMA = "jflux2.multi_task_eval_pack.v1"
SUMMARY_SCHEMA = "jflux2.multi_task_eval_summary.v1"

# `runtime.seed` for all four arms; `TaskEmbeddings` is constructed with it, so this
# reproduces the exact step-0 bank (`0.01 * default_rng(seed).standard_normal(...)`).
TRAINER_INIT_SEED = 42

# Strip chrome geometry. Recorded in every summary.json as `viz_geometry` so
# summarize_multi_task_eval.py can crop the panels back out EXACTLY instead of guessing
# (guessing is an off-by-N crop that silently shifts every cross-arm tile).
STRIP_HEADER_H = 24
STRIP_CAPTION_H = 16
STRIP_PANELS = 4   # input | prediction | target(VAE round-trip) | raw target


def log(msg: str) -> None:
    print(f"[eval] {msg}", flush=True)


# --------------------------------------------------------------------------- #
# Frozen sample pack
# --------------------------------------------------------------------------- #
def _digest_key(record_id: str) -> int:
    """Order-independent selection key. Depends only on the record id, never arrival order."""
    return int(hashlib.sha1(record_id.encode("utf-8")).hexdigest()[:16], 16)


def _as_uint8_tile(arr: np.ndarray, *, name: str) -> np.ndarray:
    """float32 [0,1] HWC -> uint8 HWC. vgi emits uint-cast tiles, so this is lossless."""
    a = np.asarray(arr, dtype=np.float32)
    if a.min() < -1e-6 or a.max() > 1.0 + 1e-6:
        raise ValueError(f"{name}: expected [0,1] tiles, got min={a.min()} max={a.max()}")
    scaled = np.clip(a, 0.0, 1.0) * 255.0
    err = float(np.max(np.abs(scaled - np.rint(scaled)))) if scaled.size else 0.0
    if err > 1e-2:
        # Not fatal (we still store uint8), but the user must know the pack is lossy.
        log(f"WARNING {name}: tiles are not uint-cast (max quantization residual {err:.4f}); "
            "the frozen pack rounds them, so pack pixels differ slightly from the loader's")
    return np.rint(scaled).astype(np.uint8)


def _part_contract(*, split: str, data_dir: str, image_size: int,
                   samples_per_task: int, max_records_scanned: int = 0,
                   supervision_version: int = 1) -> dict:
    """What a cached per-task part must agree with before it can be reused.

    data_dir is stored bucket-RELATIVE for the same reason pack_id hashes it that way: the
    zone mirrors are byte-equal at identical relative paths, so a part frozen in one zone is
    valid in the other.
    """
    # supervision_version IS part of the contract. It changes the rendered TARGET for
    # metric_depth / instance_segmentation / panoptic_segmentation (v3) and the rendered INPUT for
    # scale-dependent degradations like super_resolution (v2). Omitting it would let a v3 rebuild
    # silently reuse v1 parts and produce a pack that claims to be v3 while holding v1 pixels --
    # the exact failure the rebuild is meant to correct.
    return {"split": split, "data_dir_rel": _bucket_relative(data_dir),
            "image_size": int(image_size), "samples_per_task": int(samples_per_task),
            "max_records_scanned": int(max_records_scanned),
            "supervision_version": int(supervision_version)}


def _save_pack_part(part_dir: Path, task: str, *, images, masks, records, pool: int,
                    split: str, data_dir: str, image_size: int, samples_per_task: int,
                    max_records_scanned: int = 0, truncated: bool = False,
                    supervision_version: int = 1) -> None:
    """Persist one task's frozen samples. Written atomically: a torn part on preemption would
    otherwise be indistinguishable from a complete one and silently poison the pack."""
    safe = safe_component(task)
    # NOTE the temp name MUST end in .npz: np.savez_compressed silently APPENDS ".npz" to a
    # filename that lacks it, so ".<task>.npz.tmp" was written as ".<task>.npz.tmp.npz" and the
    # os.replace below failed with FileNotFoundError -- a bug that would only ever have fired
    # on a real preemption resume.
    tmp_npz = part_dir / f".{safe}.tmp.npz"
    tmp_json = part_dir / f".{safe}.tmp.json"
    np.savez_compressed(tmp_npz, images=np.stack(images, 0), masks=np.stack(masks, 0))
    # supervision_version MUST be forwarded. Accepting it in the signature but not passing it here
    # stamped every part with version 1 while the .npz held v3 pixels -- so a v3 build rejected its
    # own parts and re-scanned forever, and (far worse) a v1 build would ACCEPT those v3 pixels as
    # v1. Caught by a unit check that saved at v3 and asserted the reload at v1 fails.
    meta = dict(_part_contract(split=split, data_dir=data_dir, image_size=image_size,
                               samples_per_task=samples_per_task,
                               max_records_scanned=max_records_scanned,
                               supervision_version=supervision_version))
    meta.update({"task": task, "pool": int(pool), "records": records,
                 "truncated": bool(truncated), "schema": PACK_SCHEMA + ".part"})
    with tmp_json.open("w", encoding="utf-8") as f:
        json.dump(meta, f)
    os.replace(tmp_npz, part_dir / f"{safe}.npz")
    os.replace(tmp_json, part_dir / f"{safe}.json")


def _load_pack_part(part_dir: Path, task: str, *, split: str, data_dir: str,
                    image_size: int, samples_per_task: int,
                    max_records_scanned: int = 0,
                    supervision_version: int = 1) -> dict | None:
    """-> {images, masks, records, pool} if a VALID part for this contract exists, else None."""
    safe = safe_component(task)
    p_json, p_npz = part_dir / f"{safe}.json", part_dir / f"{safe}.npz"
    if not (p_json.is_file() and p_npz.is_file()):
        return None
    try:
        with p_json.open("r", encoding="utf-8") as f:
            meta = json.load(f)
        want = _part_contract(split=split, data_dir=data_dir, image_size=image_size,
                              samples_per_task=samples_per_task,
                              max_records_scanned=max_records_scanned,
                              supervision_version=supervision_version)
        if meta.get("schema") != PACK_SCHEMA + ".part" or meta.get("task") != task:
            return None
        if any(meta.get(k) != v for k, v in want.items()):
            log(f"pack part {task}: contract mismatch (have "
                f"{ {k: meta.get(k) for k in want} } vs {want}); ignoring and re-scanning")
            return None
        with np.load(p_npz) as z:
            images, masks = z["images"], z["masks"]
        recs = meta["records"]
        if len(recs) != images.shape[0] or len(recs) != masks.shape[0]:
            log(f"pack part {task}: {len(recs)} records vs images{images.shape}; re-scanning")
            return None
        return {"images": list(images), "masks": list(masks), "records": recs,
                "pool": meta.get("pool", 0), "truncated": meta.get("truncated", False)}
    except Exception as exc:  # noqa: BLE001 - a corrupt part must never be fatal
        log(f"pack part {task}: unreadable ({type(exc).__name__}: {exc}); re-scanning")
        return None


def build_sample_pack(
    *,
    tasks: list[str],
    data_dir: str,
    samples_per_task: int,
    image_size: int,
    split: str,
    seed: int,
    pack_dir: str,
    max_records_scanned: int = 0,
    supervision_version: int = 1,
) -> dict:
    """Stream each task's split ONCE and keep the `samples_per_task` smallest-hash records.

    Min-hash, not "take the first N": the loader's interleave is `deterministic=False`, so
    "first N" is not reproducible. Selection by `sha1(record_id)` depends only on the SET
    of ids in the split, so it is reproducible AND nested in N (the N=32 set is a strict
    subset of the N=64 set, so escalating a task pools with what you already computed).
    """
    from jflux2.dataloaders.multi_task_loader import load_data

    Path(pack_dir).mkdir(parents=True, exist_ok=True)
    part_dir = Path(pack_dir) / "parts"
    part_dir.mkdir(parents=True, exist_ok=True)
    images: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    records: list[dict] = []
    per_task_pool: dict[str, int] = {}
    truncated_tasks: dict[str, bool] = {}

    for task in tasks:
        t0 = time.perf_counter()
        # RESUME: each task's scan is a full pass over the split (35 GiB test / 67 GiB train),
        # so on a preemptible v5p a mid-build teardown used to discard every completed task
        # and start the whole freeze over. Each task is now persisted the moment it finishes,
        # keyed by the render contract, so a resumed attempt re-pays at most ONE task.
        cached = _load_pack_part(part_dir, task, split=split, data_dir=data_dir,
                                 image_size=image_size, samples_per_task=samples_per_task,
                                 max_records_scanned=max_records_scanned,
                                 supervision_version=supervision_version)
        if cached is not None:
            images.extend(cached["images"])
            masks.extend(cached["masks"])
            records.extend(cached["records"])
            per_task_pool[task] = int(cached["pool"])
            truncated_tasks[task] = bool(cached.get("truncated", False))
            log(f"pack task={task} RESUMED from {part_dir} "
                f"(kept={len(cached['records'])}, no re-scan)")
            continue
        heap: list[tuple[int, int, np.ndarray, np.ndarray, str, str]] = []  # max-heap by -key
        seen = 0
        truncated = False
        ds = load_data(
            datasets=[{"name": task, "weight": 1.0, "data_dir": data_dir}],
            split=split,
            batch_size=32,
            max_num_pair=1,
            repeat=False,          # MANDATORY: a repeating stream never terminates here
            shuffle=False,
            shuffle_buffer_size=256,
            cache=False,
            combine_mode="concat",
            default_raw_image_crop_size=image_size,
            direction_randomization=False,
            seed=seed,
            supervision_version=supervision_version,
        )
        for img_b, msk_b, meta in ds:
            img_np = np.asarray(img_b, dtype=np.float32)
            msk_np = np.asarray(msk_b, dtype=np.float32)
            n = int(img_np.shape[0])
            ids = T._decode_string_batch(meta["filename"], expected_len=n)
            srcs = T._decode_string_batch(meta["source"], expected_len=n)
            for i in range(n):
                seen += 1
                key = _digest_key(ids[i])
                item = (
                    -key,
                    seen,
                    _as_uint8_tile(img_np[i], name=f"{task} input"),
                    _as_uint8_tile(msk_np[i], name=f"{task} target"),
                    ids[i],
                    srcs[i],
                )
                if len(heap) < samples_per_task:
                    heapq.heappush(heap, item)
                elif -key > heap[0][0]:  # this record's hash is smaller than the worst kept
                    heapq.heapreplace(heap, item)
            if max_records_scanned and seen >= max_records_scanned:
                truncated = True
                break
        if not heap:
            raise RuntimeError(
                f"Task {task!r} yielded ZERO records from split={split!r} at {data_dir!r}. "
                "A task with no test data must be declared NOT EVALUABLE, not packed."
            )
        chosen = sorted(heap, key=lambda it: (-it[0], it[4]))  # ascending hash, id tiebreak
        per_task_pool[task] = seen
        t_images, t_masks, t_records = [], [], []
        for _, _, img_u8, msk_u8, rid, src in chosen:
            t_images.append(img_u8)
            t_masks.append(msk_u8)
            t_records.append({"task": task, "record_id": rid, "source": src,
                              "sha1": hashlib.sha1(rid.encode("utf-8")).hexdigest()})
        images.extend(t_images)
        masks.extend(t_masks)
        records.extend(t_records)
        _save_pack_part(part_dir, task, images=t_images, masks=t_masks, records=t_records,
                        pool=seen, split=split, data_dir=data_dir, image_size=image_size,
                        samples_per_task=samples_per_task,
                        max_records_scanned=max_records_scanned, truncated=truncated,
                        supervision_version=supervision_version)
        scan_note = " SCAN TRUNCATED" if truncated else ""
        log(f"pack task={task} pool={seen} kept={len(chosen)}{scan_note} "
            f"({time.perf_counter() - t0:.1f}s)")
        truncated_tasks[task] = truncated
        if len(chosen) < samples_per_task:
            log(f"WARNING task={task} pool ({seen}) < samples_per_task ({samples_per_task}); "
                f"kept {len(chosen)}")

    pack = {
        "schema": PACK_SCHEMA,
        "split": split,
        "data_dir": data_dir,
        "image_size": image_size,
        "samples_per_task": samples_per_task,
        "seed": seed,
        "tasks": list(tasks),
        "per_task_pool_size": per_task_pool,
        "max_records_scanned": int(max_records_scanned),
        "supervision_version": int(supervision_version),
        "scan_truncated": {t: bool(v) for t, v in truncated_tasks.items()},
        "records": records,
        "images": np.stack(images, axis=0),
        "masks": np.stack(masks, axis=0),
    }
    pack["pack_id"] = _compute_pack_id(pack)
    pack["identical_target_groups"] = identical_target_groups(pack)
    if pack["identical_target_groups"]:
        log(f"pack identical-target groups (measured): {pack['identical_target_groups']}")
    save_sample_pack(pack, pack_dir)
    return pack


def identical_target_groups(pack: dict) -> list[list[str]]:
    """Tasks whose SELECTED (record_id -> target pixels) are byte-identical IN THIS PACK.

    Which task families are byte-identical is SPLIT-DEPENDENT -- verdict.txt records
    `train instance_segmentation==panoptic_segmentation|coco2017` but
    `test semantic_segmentation==instance_segmentation|nyu_depth_v2`, i.e. a different pair on
    each split, and `intrinsic_decomposition==reflectance` only on train. A hardcoded family
    table is therefore wrong on at least one split however it is written: collapsing all three
    segmentation tasks over-collapses on BOTH (it folds in a genuinely distinct supervision
    each time). So measure it instead of inheriting it -- this is ground truth for the samples
    actually scored, not a claim carried over from a report.
    """
    by_task: dict[str, dict[str, bytes]] = {}
    masks = pack["masks"]
    for i, rec in enumerate(pack["records"]):
        by_task.setdefault(rec["task"], {})[rec["record_id"]] = (
            hashlib.sha1(np.ascontiguousarray(masks[i]).tobytes()).digest())
    tasks = list(by_task)
    groups: list[list[str]] = []
    claimed: set[str] = set()
    for i, a in enumerate(tasks):
        if a in claimed:
            continue
        grp = [a]
        for b in tasks[i + 1:]:
            if b in claimed:
                continue
            # identical only if they selected the SAME records AND the same target bytes
            if by_task[a] == by_task[b]:
                grp.append(b)
                claimed.add(b)
        claimed.add(a)
        if len(grp) > 1:
            groups.append(sorted(grp))
    return sorted(groups)


def _compute_pack_id(pack: dict) -> str:
    """sha1 over the sample set, the render config, AND the actual pixel bytes.

    Hashing only the sorted (task, record_id) list was NOT sufficient to make the
    summarizer's "same pack_id => comparable" check sound: the same record ids rendered at a
    different `image_size` (or out of a different `data_dir`/`split`) yield DIFFERENT pixels
    under an identical id list, so two arms scored on genuinely different inputs would have
    collided on one id and been reported as a valid paired contrast. Folding in the config
    fields and the raw uint8 tiles makes the id identify the exact bytes every arm saw.

    DO NOT add `supervision_version` to the hashed fields, even though it changes the render.
    This id is RECOMPUTED AND VERIFIED on every pack read (see load_sample_pack, which raises on
    mismatch), and packs frozen before that key existed have no value for it -- so folding it in
    would make every previously frozen pack unreadable and retro-invalidate every arm already
    scored against one. It is unnecessary anyway: the raw uint8 tiles are hashed below, and a
    different supervision version renders different pixels by definition, so a v1 and a v3 pack
    over the same records cannot collide. The version is recorded in the pack metadata for
    provenance instead, and enforced where it actually matters -- `_part_contract`, so a rebuild
    cannot reuse parts rendered under a different version.
    """
    h = hashlib.sha1()
    payload = "\n".join(sorted(f"{r['task']}\t{r['record_id']}" for r in pack["records"]))
    h.update(payload.encode("utf-8"))
    for k in ("split", "data_dir", "image_size", "max_records_scanned"):
        v = pack.get(k)
        if k == "data_dir" and v is not None:
            # The bucket-RELATIVE path, for the same reason the reuse guard compares that:
            # the zone mirrors hold identical bytes at identical relative paths, so hashing
            # the full gs:// URL would give two ids to one set of pixels and make a pack
            # gratuitously zone-locked. The pixels themselves are hashed below, so a genuine
            # content difference is still caught.
            v = _bucket_relative(v)
        h.update(f"\x00{k}={v!r}".encode("utf-8"))
    for key in ("images", "masks"):
        arr = pack.get(key)
        if arr is None:
            continue
        a = np.ascontiguousarray(arr)
        h.update(f"\x00{key}:{a.dtype.str}:{tuple(a.shape)}".encode("utf-8"))
        h.update(a.tobytes())
    return h.hexdigest()


def save_sample_pack(pack: dict, pack_dir: str) -> None:
    Path(pack_dir).mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        Path(pack_dir) / "samples.npz", images=pack["images"], masks=pack["masks"]
    )
    meta = {k: v for k, v in pack.items() if k not in ("images", "masks")}
    with (Path(pack_dir) / "samples.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    log(f"pack written pack_id={pack['pack_id']} n={len(pack['records'])} -> {pack_dir}")


def load_sample_pack(pack_dir: str) -> dict:
    with (Path(pack_dir) / "samples.json").open("r", encoding="utf-8") as f:
        meta = json.load(f)
    if meta.get("schema") != PACK_SCHEMA:
        raise ValueError(f"{pack_dir}/samples.json is not a {PACK_SCHEMA} pack.")
    with np.load(Path(pack_dir) / "samples.npz") as z:
        meta["images"] = z["images"]
        meta["masks"] = z["masks"]
    n = len(meta["records"])
    if meta["images"].shape[0] != n or meta["masks"].shape[0] != n:
        raise ValueError(
            f"pack {pack_dir} is inconsistent: {n} records but images"
            f"{meta['images'].shape} masks{meta['masks'].shape}"
        )
    recomputed = _compute_pack_id(meta)
    if recomputed != meta.get("pack_id"):
        raise ValueError(
            f"pack {pack_dir} pack_id mismatch: stored {meta.get('pack_id')} vs recomputed "
            f"{recomputed}. The frozen sample set was edited; refusing to use it."
        )
    if "identical_target_groups" not in meta:
        # Compute on load for any pack that predates the field (or was written by a helper
        # that does not build one), so the dedup is ALWAYS measured rather than falling back
        # to the hardcoded family table that measurement disproved on both splits.
        meta["identical_target_groups"] = identical_target_groups(meta)
        log(f"pack carried no identical-target groups; measured on load: "
            f"{meta['identical_target_groups'] or 'NONE'}")
    log(f"pack loaded pack_id={meta['pack_id']} n={n} tasks={len(meta['tasks'])}")
    return meta


def pack_indices_for_task(pack: dict, task: str) -> list[int]:
    return [i for i, r in enumerate(pack["records"]) if r["task"] == task]


# --------------------------------------------------------------------------- #
# Base model / AE / mesh
# --------------------------------------------------------------------------- #
def build_mesh(device_type: str):
    """The TRAIN-style mesh: (data=N, fsdp=1). Params end up fully REPLICATED."""
    devices = T._resolve_mesh_devices(device_type)
    mesh = T.create_hsdp_mesh(
        device_type=device_type, data_axis_size=len(devices), fsdp_axis_size=1
    )
    log(f"mesh {dict(mesh.shape)} over {len(devices)} {device_type} device(s)")
    return mesh, devices[0]


def build_flux_and_ae(*, mesh, bucket_prefix: str, flux_weights: str, flux_config: str,
                      vae_weights: str, vae_config: str, seed: int,
                      gradient_checkpointing: bool = False):
    """Load the FROZEN 4B FLUX2 + the AE with EXACTLY the trainer's cfg overrides."""
    from jflux2.model import Flux2
    from jflux2.modules.autoencoder import AutoEncoder
    from port_safetensor_jax import (
        flux2_params_from_diffusers_config,
        load_converted_jax_safetensors as load_flux_weights,
        load_converted_weights_into_model,
    )
    from port_vae_safetensor_jax import (
        autoencoder_params_from_diffusers_config,
        load_converted_jax_safetensors as load_vae_weights,
        load_converted_weights_into_autoencoder,
    )

    R = lambda p: T._resolve_bucket_relative_path(p, bucket_prefix)  # noqa: E731

    t0 = time.perf_counter()
    cfg = flux2_params_from_diffusers_config(R(flux_config), rng_seed=seed,
                                             param_dtype=jnp.bfloat16)
    cfg.lora_rank = 0
    cfg.lora_alpha = 1.0
    cfg.double_stream_txt_lora_rank = 0
    cfg.double_stream_txt_lora_alpha = 0.0
    # Remat is a differentiation-only transform: the FORWARD is bit-exact with or without it, and
    # eager forward is ~2x faster without -- so eval keeps it OFF, which is why this was hardcoded.
    # But TRAINERS also call this builder, and for them the hardcode is a silent memory bug: every
    # activation in every double and single block is stored for the backward. That is what made
    # P>=2 of the demo ladder OOM (19.55G requested vs 16.62G free). Now opt-in, default unchanged.
    cfg.gradient_checkpointing = bool(gradient_checkpointing)
    model = Flux2(cfg)
    model = load_converted_weights_into_model(model, load_flux_weights(R(flux_weights)))
    T.shard_model_params_for_hsdp(model, mesh)  # fsdp=1 => full replication
    log(f"flux loaded ({time.perf_counter() - t0:.1f}s) context_in_dim={cfg.context_in_dim} "
        f"guidance_embed={getattr(cfg, 'use_guidance_embed', None)}")

    t0 = time.perf_counter()
    ae_params = autoencoder_params_from_diffusers_config(R(vae_config), rng_seed=seed,
                                                         param_dtype=jnp.bfloat16)
    ae = AutoEncoder(ae_params)
    ae = load_converted_weights_into_autoencoder(ae, load_vae_weights(R(vae_weights)))
    T._validate_autoencoder_compatibility(ae)
    # MUST be replicate, NOT shard_model_params_for_hsdp: fsdp-sharding lands the 1-D BN
    # buffers running_mean/running_var on device 0 and `decode` then dies in inv_normalize
    # with "Received incompatible devices for jitted computation" (real TPU failure).
    T._replicate_module_params(ae, mesh)
    log(f"ae loaded ({time.perf_counter() - t0:.1f}s)")
    return model, ae, cfg


# --------------------------------------------------------------------------- #
# Embedding bank
# --------------------------------------------------------------------------- #
def resolve_checkpoint_paths(ckpt_dir: str, step: int | None) -> tuple[str, str]:
    """-> (manifest_dir, checkpoint_path_or_dir). Accepts a dir OR a checkpoint file."""
    raw = str(ckpt_dir).rstrip("/")
    base = os.path.basename(raw)
    if base.startswith("checkpoint_"):
        return os.path.dirname(raw), raw
    if step is not None:
        return raw, f"{raw}/checkpoint_{int(step)}"
    return raw, raw  # load_task_embeddings resolves the latest


def _bucket_relative(path: str) -> str:
    """Strip any known zone bucket prefix -> the bucket-RELATIVE path.

    The whole point of the mirrored-bucket contract is that ONE relative path names the
    byte-identical dataset in either zone (`ZONE_TO_BUCKET_PREFIX[zone] + rel`; the us-east5
    mirror was verified object-count- and byte-equal to us-central1). So comparing raw
    `gs://` URLs would reject reusing a pack built in us-east5 from a run in us-central1 even
    though the pixels are provably the same bytes -- a false alarm that would force a
    needless ~700 GiB re-freeze. Compare the relative path instead.
    """
    p = str(path).rstrip("/")
    for prefix in set(T.ZONE_TO_BUCKET_PREFIX.values()):
        pre = prefix.rstrip("/") + "/"
        if p.startswith(pre):
            return p[len(pre):]
    return p


def _step_from_checkpoint_path(ckpt_path: str) -> int | None:
    """-> the step encoded in `.../checkpoint_<N>`, else None (an unpinned dir).

    `last_saved_step` in the manifest sidecar is the LATEST step the run ever wrote, which is
    not the step loaded when `--step` pins an earlier checkpoint. Labelling viz/summaries
    with the sidecar value would stamp `step=20000` on images decoded from checkpoint_10000.
    """
    m = re.search(r"checkpoint_(\d+)/?$", str(ckpt_path).rstrip("/"))
    return int(m.group(1)) if m else None


def load_bank_for_tasks(
    *, ckpt_dir: str, step: int | None, tasks: list[str], context_in_dim: int,
    split: str = "test",
) -> dict:
    """Build a `[len(tasks), 512, C]` bank in CANONICAL task order, row-selected by NAME.

    Returns the bank plus a full audit trail: for every task, the manifest ROW INDEX it
    came from and the sha1 of the exact row bytes, so a reader can prove task X used row X.
    """
    manifest_dir, ckpt_path = resolve_checkpoint_paths(ckpt_dir, step)
    manifest = T.read_task_manifest_sidecar(manifest_dir)
    tokens, payload_manifest = T.load_task_embeddings(ckpt_path)
    if manifest is None:
        manifest = payload_manifest
    if manifest is None:
        raise RuntimeError(
            f"{ckpt_dir} carries no task manifest (neither payload nor sidecar); its rows "
            "cannot be mapped to task names, so it cannot be evaluated."
        )
    # Structural checks the caller would otherwise have to trust.
    if str(manifest.get("schema")) != T.TASK_MANIFEST_SCHEMA:
        raise RuntimeError(f"{ckpt_dir}: unexpected manifest schema {manifest.get('schema')!r}")
    if int(manifest["num_tokens"]) != T.EXPECTED_TEXT_TOKEN_COUNT:
        raise RuntimeError(f"{ckpt_dir}: num_tokens={manifest['num_tokens']} != "
                           f"{T.EXPECTED_TEXT_TOKEN_COUNT}")
    if int(manifest["context_in_dim"]) != int(context_in_dim):
        raise RuntimeError(
            f"{ckpt_dir}: manifest context_in_dim={manifest['context_in_dim']} does not match "
            f"the loaded FLUX context_in_dim={context_in_dim}; the bank cannot drive this model."
        )
    expected = (int(manifest["num_tasks"]), T.EXPECTED_TEXT_TOKEN_COUNT, int(context_in_dim))
    if tuple(tokens.shape) != expected:
        raise RuntimeError(f"{ckpt_dir}: tokens{tuple(tokens.shape)} != expected {expected}")

    names = [str(n) for n in manifest.get("task_names", [])]
    # An arm only holds the tasks it was TRAINED on: mt_both20_lr1e4_sample has 20 rows while
    # the other three have 24, so a 24-task train pack asks this arm for rows that do not
    # exist. Report them as not-evaluable-for-this-arm rather than raising (which would fail
    # the whole job) or silently substituting some other row (which would be a lie).
    absent = [t for t in tasks if t not in names]
    present = [t for t in tasks if t in names]
    rows: list[np.ndarray] = []
    audit: list[dict] = []
    for task in present:
        row = T.task_index_for_name(manifest, task)
        vec = np.asarray(tokens[row], dtype=np.float32)
        rows.append(vec)
        audit.append({
            "task": task,
            "canonical_index": len(rows) - 1,
            "manifest_row": int(row),
            "manifest_name_at_row": names[row] if row < len(names) else None,
            "row_sha1": hashlib.sha1(vec.tobytes()).hexdigest(),
            "row_std": float(vec.std()),
        })
    bank = np.stack(rows, axis=0) if rows else np.zeros((0, 512, context_in_dim), np.float32)

    # Rows the checkpoint HOLDS but that have no test data. Collected here, while the full
    # tokens array is still in hand, so they cost nothing and carry the measurable evidence
    # (row_std, divergence from the step-0 init) instead of being silently dropped from the
    # report. NOTE what that evidence does and does not establish: a large
    # `max_abs_delta_vs_init` shows the row MOVED under the optimizer, which is necessary but
    # NOT sufficient for "the row learned the task" -- unscoreable is unscoreable, and no
    # held-out number exists for these four tasks. The field is deliberately named for the
    # measurement, not for the conclusion.
    # seed 42 is the trainer's runtime.seed for all four arms (TaskEmbeddings is constructed
    # with seed=runtime.seed); rows fill C-order so row i matches for 20- and 24-row draws.
    init_ref = init_bank_like(int(manifest["num_tasks"]), context_in_dim=context_in_dim,
                              seed=TRAINER_INIT_SEED)
    held: list[dict] = []
    requested = set(tasks)
    for name in names:
        if name not in T.TRAIN_ONLY_VGI_TASKS:
            continue
        # On the TRAIN split these tasks ARE scored -- declaring them "cannot be scored" in
        # the same summary that carries their L1 would tell the reader the opposite of what
        # the run produced.
        if split == "train" or name in requested:
            continue
        row = T.task_index_for_name(manifest, name)
        vec = np.asarray(tokens[row], dtype=np.float32)
        held.append({
            "task": name,
            "evaluable": False,
            "n": 0,
            "manifest_row": int(row),
            "row_std": float(vec.std()),
            "max_abs_delta_vs_init": float(np.max(np.abs(vec - init_ref[row]))),
            "status": "not_evaluable_train_only",
            "reason": NOT_EVALUABLE_TRAIN_ONLY_REASON,
        })

    for name in absent:
        held.append({
            "task": name, "evaluable": False, "n": 0, "manifest_row": None,
            "row_std": None, "max_abs_delta_vs_init": None,
            "status": "not_in_this_arms_bank",
            "reason": (f"this arm's bank holds {len(names)} task rows and {name!r} is not one "
                       "of them (it was absent from THIS arm's training task list), so there "
                       "is no embedding to evaluate -- the row is missing, not bad"),
        })
    return {
        "bank": bank,
        "manifest": manifest,
        "checkpoint_path": ckpt_path,
        "loaded_step": _step_from_checkpoint_path(ckpt_path),
        "tasks_present": present,
        "tasks_absent": absent,
        "row_audit": audit,
        "manifest_task_names": names,
        "held_train_only_rows": held,
        "last_saved_step": int(manifest.get("last_saved_step", -1)),
        "num_tasks_in_bank": int(manifest["num_tasks"]),
        "bank_std": float(np.asarray(tokens).std()),
    }


def make_task_embeddings(bank: np.ndarray, *, context_in_dim: int, seed: int, mesh):
    """`set_tokens` is numerically identical to `restore_checkpoint` for this trainer.

    `restore_checkpoint` reaches the weights via `_restore_module_state_from_named_flat(
    ..., TaskParam)` and `_assert_only_task_trainable` proves there is exactly ONE TaskParam
    leaf in the bundle. The value written is `jnp.asarray(host, dtype=var.value.dtype)` with
    `var.value.dtype == float32`, which is exactly what `set_tokens` does. The only thing
    restore adds is sharding preservation, which `shard_task_embeddings_for_hsdp` supplies.
    Using `set_tokens` also skips `assert_manifest_compatible`, which is the check that
    would otherwise force the config task list to match the manifest ORDER.
    """
    te = T.TaskEmbeddings(
        num_tasks=int(bank.shape[0]),
        num_tokens=T.EXPECTED_TEXT_TOKEN_COUNT,
        context_in_dim=int(context_in_dim),
        seed=seed,
        task_param_cls=T.TaskParam,
    )
    te.set_tokens(jnp.asarray(bank, dtype=jnp.float32))
    T.shard_task_embeddings_for_hsdp(te, mesh)
    return te


def init_bank_like(num_tasks: int, *, context_in_dim: int, seed: int) -> np.ndarray:
    """The exact step-0 state: `0.01 * rng.standard_normal(...)` with the trainer's seed.

    `standard_normal` fills C-order, so rows 0..k of a 20-row draw are bit-identical to
    rows 0..k of a 24-row draw at the same seed. This is a RANDOM-EMBEDDING control -- which
    row a task gets is immaterial, they are iid draws from one distribution.
    """
    rng = np.random.default_rng(seed)
    return (0.01 * rng.standard_normal(
        (num_tasks, T.EXPECTED_TEXT_TOKEN_COUNT, context_in_dim))).astype(np.float32)


# --------------------------------------------------------------------------- #
# Visualization
# --------------------------------------------------------------------------- #
def _to_uint8_hwc(chw: np.ndarray) -> np.ndarray:
    """Exactly `_save_rgb_image`'s pixel semantics: clip [-1,1] then 127.5*(x+1)."""
    img = np.asarray(chw, dtype=np.float32)
    img = np.clip(img, -1.0, 1.0)
    arr = np.transpose(img, (1, 2, 0))
    return np.clip(127.5 * (arr + 1.0), 0.0, 255.0).astype(np.uint8)


def save_comparison_strip(
    *, input_chw: np.ndarray, pred_chw: np.ndarray, target_chw: np.ndarray,
    path: Path, header: str, captions: tuple[str, ...],
    raw_target_chw: np.ndarray | None = None,
    header_h: int = STRIP_HEADER_H, caption_h: int = STRIP_CAPTION_H,
) -> tuple[int, int]:
    """`input | prediction | target(VAE round-trip) [| raw target]`, plus labels.

    The first three panels are deliberately in `_write_validation_artifacts` (:2501) order so
    the in-run PNGs and these offline PNGs are visually interchangeable. The optional FOURTH
    panel is the RAW dataset target, i.e. the same tile before the VAE round-trip. It matters
    for manual inspection: the scored reference is `ae.decode(encode(target))`, so panel 3
    already carries the VAE's own error, and without panel 4 a reader cannot tell a genuine
    model failure from detail the autoencoder itself destroyed.
    """
    tiles = [input_chw, pred_chw, target_chw]
    if raw_target_chw is not None:
        tiles.append(raw_target_chw)
    panels = [_to_uint8_hwc(x) for x in tiles]
    ncol = len(panels)
    h, w = panels[0].shape[:2]
    canvas = Image.new("RGB", (w * ncol, header_h + h + caption_h), (0, 0, 0))
    for i, p in enumerate(panels):
        canvas.paste(Image.fromarray(p), (i * w, header_h))
    d = ImageDraw.Draw(canvas)
    d.text((4, (header_h - 11) // 2), _fit_text(d, header, w * ncol - 8), fill=(255, 255, 255))
    for i, cap in enumerate(captions):
        # Each caption is clamped to ITS OWN panel width so adjacent captions cannot collide.
        d.text((i * w + 4, header_h + h + (caption_h - 11) // 2),
               _fit_text(d, cap, w - 8), fill=(200, 200, 200))
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)
    return canvas.size


def _fit_text(draw: ImageDraw.ImageDraw, text: str, max_px: int) -> str:
    """Truncate to the available PIXEL width, not a guessed character count.

    A fixed character cap either clips a short header needlessly or lets a long record_id
    run past the panel and collide with the next caption. Measure instead.
    """
    if max_px <= 0 or not text:
        return ""
    try:
        if draw.textlength(text) <= max_px:
            return text
    except Exception:  # pragma: no cover - very old PIL without textlength
        return text[: max(1, max_px // 6)]
    ell = "..."
    lo, hi = 0, len(text)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if draw.textlength(text[:mid] + ell) <= max_px:
            lo = mid
        else:
            hi = mid - 1
    return (text[:lo] + ell) if lo else ""


def safe_component(value: str) -> str:
    """The trainer's own filename sanitizer (`:2496-2499`)."""
    return str(value).replace("/", "_")


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #
def _rgb01_to_pm1_chw(tile_u8: np.ndarray) -> np.ndarray:
    """uint8 HWC -> float32 CHW in [-1,1], the space the decoder output lives in."""
    a = np.asarray(tile_u8, dtype=np.float32) / 255.0
    return np.transpose(a * 2.0 - 1.0, (2, 0, 1))


def evaluate_task(
    *, model, ae, task_embeddings, mesh, val_device, pack: dict, task: str,
    canonical_index: int, batch_size: int, num_steps: int, guidance: float,
    viz_dir: Path | None, viz_per_task: int, arm: str, step_label: int,
    split: str = "test",
) -> dict:
    """Denoise every packed sample of one task and score it with the trainer's metric."""
    idxs = pack_indices_for_task(pack, task)
    if not idxs:
        raise RuntimeError(f"pack {pack['pack_id']} has no samples for task {task!r}")
    images_u8 = pack["images"]
    masks_u8 = pack["masks"]

    per_sample: list[dict] = []
    n_viz = 0
    for start in range(0, len(idxs), batch_size):
        chunk = idxs[start:start + batch_size]
        img_b = np.stack([images_u8[i] for i in chunk]).astype(np.float32) / 255.0
        msk_b = np.stack([masks_u8[i] for i in chunk]).astype(np.float32) / 255.0

        # Trainer path: [B,H,W,3] in [0,1] -> transpose -> *2-1 -> ONE ae.encode over 2B.
        image_latent, mask_latent = T._normalize_validation_latents(
            img_b, msk_b, ae=ae, val_device=val_device, val_mesh=mesh
        )
        # THE one semantic deviation from run_validation_on_split: the row index comes from
        # our canonical order, not metadata["source_dataset_index"] (which a single-task
        # loader always reports as 0). Safe: TaskEmbeddings.batched is a pure jnp.take.
        task_indices = jnp.asarray([canonical_index] * len(chunk), dtype=jnp.int32)

        decoded, ref_decoded, input_decoded = T._run_validation_forward(
            model=model, task_embeddings=task_embeddings, ae_decode=ae,
            image_latent=image_latent, mask_latent=mask_latent, task_indices=task_indices,
            guidance=guidance, num_steps=num_steps,
            val_device=val_device, val_mesh=mesh, decode_input=True,
        )
        # Byte-identical to run_validation_on_split:2603.
        batch_l1 = np.asarray(jnp.mean(jnp.abs(decoded - ref_decoded), axis=(1, 2, 3)))
        dec_np = np.asarray(decoded, dtype=np.float32)
        ref_np = np.asarray(ref_decoded, dtype=np.float32)
        inp_np = np.asarray(input_decoded, dtype=np.float32)

        for j, gidx in enumerate(chunk):
            rec = pack["records"][gidx]
            # Extra reference columns, computed on the SAME decoded tensors.
            raw_target = _rgb01_to_pm1_chw(masks_u8[gidx])
            row = {
                "record_id": rec["record_id"],
                "source": rec.get("source", ""),
                "pack_index": int(gidx),
                "l1_pm1": float(batch_l1[j]),
                "l1_01": float(batch_l1[j]) / 2.0,
                "l1_vs_raw_target_pm1": float(np.mean(np.abs(dec_np[j] - raw_target))),
                "psnr_01": _psnr_01(dec_np[j], ref_np[j]),
            }
            per_sample.append(row)
            if viz_dir is not None and n_viz < viz_per_task:
                safe = safe_component(rec["record_id"])
                path = (viz_dir / task /
                        f"{split}_step_{step_label:07d}_{safe_component(task)}_{safe}"
                        "_comparison.png")
                # The split goes in the HEADER and the FILENAME. A train-split panel is the
                # model reproducing something it was optimized on; if that label is only in a
                # sidecar json, a PNG pasted into a doc becomes an unfalsifiable claim.
                tag = "TRAIN (in-distribution, NOT held out)" if split == "train" else split
                save_comparison_strip(
                    input_chw=inp_np[j], pred_chw=dec_np[j], target_chw=ref_np[j], path=path,
                    raw_target_chw=raw_target,
                    header=(f"split={tag}  task={task}  arm={arm}  step={step_label}  "
                            f"file={rec['record_id']}"),
                    captions=(
                        "input",
                        f"prediction  L1[0,1]={row['l1_01']:.4f}",
                        "target (VAE round-trip) = SCORED ref",
                        f"raw target  (VAE floor {row['l1_vs_raw_target_pm1'] / 2.0:.4f})",
                    ),
                )
                n_viz += 1
    vals = np.asarray([r["l1_01"] for r in per_sample], dtype=np.float64)
    return {
        "task": task,
        "split": split,
        "evaluable": True,
        "n": int(vals.size),
        "l1_01_mean": float(vals.mean()),
        "l1_01_std": float(vals.std(ddof=1)) if vals.size > 1 else 0.0,
        "l1_01_sem": float(vals.std(ddof=1) / np.sqrt(vals.size)) if vals.size > 1 else 0.0,
        "l1_pm1_mean": float(vals.mean() * 2.0),
        "psnr_01_mean": float(np.mean([r["psnr_01"] for r in per_sample])),
        "l1_vs_raw_target_pm1_mean": float(np.mean([r["l1_vs_raw_target_pm1"] for r in per_sample])),
        "samples": per_sample,
    }


def _psnr_01(a_pm1: np.ndarray, b_pm1: np.ndarray) -> float:
    mse = float(np.mean(((a_pm1 - b_pm1) / 2.0) ** 2))
    if mse <= 1e-12:
        return 99.0
    return float(10.0 * np.log10(1.0 / mse))


def compute_reference_baselines(*, ae, mesh, val_device, pack: dict, tasks: list[str],
                                batch_size: int) -> dict:
    """Three zero-denoise references, on the frozen pack, in the metric's own space.

    l1_copy  : predict the input unchanged -> if an arm cannot beat this, it learned nothing.
    l1_black : predict all-black -> the degenerate-solution detector for the mostly-black
               targets (edge_detection, saliency_detection).
    l1_floor : decode(encode(target)) vs decode(encode(target)) is 0 by construction, so the
               floor here is the VAE round-trip error measured against the RAW target; it is
               the denominator anchor that stops VAE error being read as model error.

    All are measured against `ref_decoded = ae.decode(mask_latent)`, the SAME reference the
    headline metric uses, so they are directly subtractable.
    """
    out: dict[str, dict] = {}
    for task in tasks:
        idxs = pack_indices_for_task(pack, task)
        copy_v, black_v, floor_v = [], [], []
        for start in range(0, len(idxs), batch_size):
            chunk = idxs[start:start + batch_size]
            img_b = np.stack([pack["images"][i] for i in chunk]).astype(np.float32) / 255.0
            msk_b = np.stack([pack["masks"][i] for i in chunk]).astype(np.float32) / 255.0
            image_latent, mask_latent = T._normalize_validation_latents(
                img_b, msk_b, ae=ae, val_device=val_device, val_mesh=mesh
            )
            with mesh:
                ref = np.asarray(ae.decode(T._put_validation_value_to_mesh(mask_latent, mesh)),
                                 dtype=np.float32)
                inp = np.asarray(ae.decode(T._put_validation_value_to_mesh(image_latent, mesh)),
                                 dtype=np.float32)
            raw_t = np.stack([_rgb01_to_pm1_chw(pack["masks"][i]) for i in chunk])
            copy_v += list(np.mean(np.abs(inp - ref), axis=(1, 2, 3)) / 2.0)
            black_v += list(np.mean(np.abs(-1.0 - ref), axis=(1, 2, 3)) / 2.0)
            floor_v += list(np.mean(np.abs(ref - raw_t), axis=(1, 2, 3)) / 2.0)
        out[task] = {
            "l1_copy_01": float(np.mean(copy_v)),
            "l1_black_01": float(np.mean(black_v)),
            "l1_floor_01": float(np.mean(floor_v)),
            "n": len(idxs),
        }
        log(f"baseline task={task} copy={out[task]['l1_copy_01']:.4f} "
            f"black={out[task]['l1_black_01']:.4f} vae_floor={out[task]['l1_floor_01']:.4f}")
    return out


def evaluate_arm(
    *, arm: str, ckpt_dir: str, step: int | None, model, ae, mesh, val_device,
    pack: dict, tasks: list[str], context_in_dim: int, batch_size: int, num_steps: int,
    guidance: float, seed: int, out_dir: Path, viz_per_task: int,
    baselines: dict, init_bank: bool = False, stub_model: bool = False,
) -> dict:
    t_arm = time.perf_counter()
    if init_bank:
        bank = init_bank_like(len(tasks), context_in_dim=context_in_dim, seed=seed)
        info = {
            "bank": bank, "manifest": None, "checkpoint_path": None, "row_audit": [
                {"task": t, "canonical_index": i, "manifest_row": i,
                 "manifest_name_at_row": None,
                 "row_sha1": hashlib.sha1(bank[i].tobytes()).hexdigest(),
                 "row_std": float(bank[i].std())} for i, t in enumerate(tasks)],
            "manifest_task_names": [], "held_train_only_rows": [], "last_saved_step": 0,
            "num_tasks_in_bank": len(tasks), "bank_std": float(bank.std()),
            "tasks_present": list(tasks), "tasks_absent": [],
        }
        not_evaluable: list[dict] = []
        step_label = 0
        _ = init_bank  # the synthetic arm has every requested row by construction
    else:
        info = load_bank_for_tasks(ckpt_dir=ckpt_dir, step=step, tasks=tasks,
                                   context_in_dim=context_in_dim,
                                   split=str(pack.get("split", "test")))
        bank = info["bank"]
        # Pinned checkpoint wins; the sidecar's last_saved_step is only the fallback for an
        # unpinned dir (where load_task_embeddings did resolve the latest).
        step_label = info.get("loaded_step")
        if step_label is None:
            step_label = info["last_saved_step"]
        elif step_label != info["last_saved_step"]:
            log(f"arm={arm} loaded checkpoint step={step_label} (manifest last_saved_step="
                f"{info['last_saved_step']}); labelling with the LOADED step")
        step_label = int(step_label)
        # Reported explicitly, never silently skipped: rows the checkpoint HOLDS but that
        # have zero test records, plus the test-only task no bank contains at all.
        not_evaluable = list(info["held_train_only_rows"])  # incl. rows absent from this bank
        pack_split = str(pack.get("split", "test"))
        for n in sorted(T.TEST_ONLY_VGI_TASKS):
            if n not in info["manifest_task_names"]:
                reason = NOT_EVALUABLE_TEST_ONLY_REASON
                if pack_split == "train":
                    # On train it ALSO has zero records, so the test-split wording ("give it a
                    # row and it could be scored") is false here.
                    reason += (" -- and on the TRAIN split it additionally has zero records, "
                               "so no row could be scored here even if one existed")
                not_evaluable.append({
                    "task": n, "evaluable": False, "n": 0, "manifest_row": None,
                    "row_std": None, "max_abs_delta_vs_init": None,
                    "status": "not_evaluable_test_only", "reason": reason})

    log(f"arm={arm} bank{tuple(bank.shape)} std={info['bank_std']:.5f} "
        f"step={step_label} rows={[a['manifest_row'] for a in info['row_audit']]}")
    task_embeddings = make_task_embeddings(bank, context_in_dim=context_in_dim,
                                           seed=seed, mesh=mesh)
    viz_dir = out_dir / "arms" / arm / "viz"
    rows: list[dict] = []
    # `bank` holds ONE row per present task, in this order. Indexing it with a position from
    # the REQUESTED list would silently score a task with another task's embedding whenever
    # an arm is missing a row, which is the worst possible failure: plausible numbers, wrong
    # provenance. `evaluate_task(canonical_index=ci)` therefore indexes eval_tasks.
    eval_tasks = list(info.get("tasks_present") or tasks)
    if len(eval_tasks) != int(bank.shape[0]):
        raise RuntimeError(
            f"arm={arm}: bank has {bank.shape[0]} rows but {len(eval_tasks)} tasks to score; "
            "refusing to guess the row->task mapping.")
    skipped = [t for t in tasks if t not in eval_tasks]
    if skipped:
        log(f"arm={arm} SKIPPING {len(skipped)} task(s) absent from this arm's bank: {skipped}")
    for ci, task in enumerate(eval_tasks):
        t0 = time.perf_counter()
        row = evaluate_task(
            model=model, ae=ae, task_embeddings=task_embeddings, mesh=mesh,
            val_device=val_device, pack=pack, task=task, canonical_index=ci,
            batch_size=batch_size, num_steps=num_steps, guidance=guidance,
            viz_dir=viz_dir if viz_per_task > 0 else None, viz_per_task=viz_per_task,
            arm=arm, step_label=step_label, split=str(pack.get("split", "test")),
        )
        audit = info["row_audit"][ci]
        row.update({"manifest_row": audit["manifest_row"], "row_sha1": audit["row_sha1"],
                    "row_std": audit["row_std"], "status": "evaluated"})
        b = baselines.get(task, {})
        row.update({k: b.get(k) for k in ("l1_copy_01", "l1_black_01", "l1_floor_01")})
        row["norm_score"] = _norm_score(row["l1_01_mean"], b)
        row["beats_copy"] = (row["l1_01_mean"] < b["l1_copy_01"]) if b else None
        row["beats_black"] = (row["l1_01_mean"] < b["l1_black_01"]) if b else None
        rows.append(row)
        log(f"arm={arm} task={task} row={audit['manifest_row']} n={row['n']} "
            f"L1[0,1]={row['l1_01_mean']:.4f}+-{row['l1_01_sem']:.4f} "
            f"norm={row['norm_score'] if row['norm_score'] is None else round(row['norm_score'], 3)} "
            f"({time.perf_counter() - t0:.1f}s)")

    ev = [r for r in rows if r["n"] > 0]
    example_w = float(np.average([r["l1_01_mean"] for r in ev],
                                 weights=[r["n"] for r in ev])) if ev else None
    summary = {
        "schema": SUMMARY_SCHEMA,
        "arm": arm,
        "checkpoint_dir": ckpt_dir,
        "checkpoint_path": info["checkpoint_path"],
        "step": step_label,
        "pack_id": pack["pack_id"],
        "pack_split": pack["split"],
        "identical_target_groups": pack.get("identical_target_groups"),
        "in_distribution": pack["split"] == "train",
        "pack_scan_truncated": bool(any((pack.get("scan_truncated") or {}).values())),
        "pack_max_records_scanned": int(pack.get("max_records_scanned", 0) or 0),
        "samples_per_task": pack["samples_per_task"],
        "num_steps": max(num_steps, T._MIN_VALIDATION_NUM_STEPS),
        "num_steps_requested": num_steps,
        "guidance": guidance,
        "image_size": pack["image_size"],
        "seed": seed,
        # Carried per ARM (not only in eval_index.json) so the summarizer can stamp a
        # loud warning on the table: a stub run's numbers come from a random-init tiny
        # transformer and must never be read as a result.
        "stub_model": bool(stub_model),
        "random_bank_baseline": bool(init_bank),
        "bank_shape": list(bank.shape),
        "bank_std": info["bank_std"],
        "num_tasks_in_checkpoint": info["num_tasks_in_bank"],
        "manifest_task_names": info["manifest_task_names"],
        "canonical_task_order": list(eval_tasks),
        "requested_task_order": list(tasks),
        "tasks_absent_from_bank": list(info.get("tasks_absent") or []),
        "row_audit": info["row_audit"],
        "viz_geometry": {
            "header_h": STRIP_HEADER_H, "caption_h": STRIP_CAPTION_H,
            "panels": STRIP_PANELS,
            "panel_order": ["input", "prediction", "target_vae_roundtrip", "target_raw"],
        },
        "metric": {
            "name": "avg_l1",
            "definition": "mean(|ae.decode(denoise(x)) - ae.decode(mask_latent)|) over (C,H,W)",
            "reference": "VAE ROUND-TRIP of the target, not the raw target",
            "headline_scale": "[0,1] pixel units (l1_01 = l1_pm1 / 2)",
            "lower_is_better": True,
        },
        "avg_l1_01_examplewise": example_w,
        "avg_l1_01_macro": float(np.mean([r["l1_01_mean"] for r in ev])) if ev else None,
        "mean_norm_score": _safe_mean([r["norm_score"] for r in ev]),
        "rows": rows,
        "not_evaluable": not_evaluable,
        "baselines": baselines,
        "elapsed_s": round(time.perf_counter() - t_arm, 1),
    }
    dest = out_dir / "arms" / arm / "summary.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    log(f"arm={arm} DONE avg_l1[0,1]={example_w} mean_norm={summary['mean_norm_score']} "
        f"-> {dest} ({summary['elapsed_s']}s)")
    return summary


def _norm_score(l1: float, b: dict) -> float | None:
    """(L1 - floor) / (copy - floor): 0 = VAE-limited perfect, 1 = no better than identity."""
    if not b:
        return None
    denom = b["l1_copy_01"] - b["l1_floor_01"]
    if abs(denom) < 1e-9:
        return None
    return float((l1 - b["l1_floor_01"]) / denom)


def _safe_mean(vals) -> float | None:
    ok = [v for v in vals if v is not None]
    return float(np.mean(ok)) if ok else None


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def strip_workdir(argv: list[str]) -> tuple[list[str], str | None]:
    """The daemon runs `<python> <script> --workdir=<dir> <args>`; we write our own out-dir.

    Same contract handling as compute/uinfra_wiring_entry.py:49-58.
    """
    kept, workdir, i = [], None, 0
    while i < len(argv):
        a = argv[i]
        if a.startswith("--workdir="):
            workdir = a.split("=", 1)[1]
        elif a == "--workdir" and i + 1 < len(argv):
            workdir, i = argv[i + 1], i + 1
        else:
            kept.append(a)
        i += 1
    return kept, workdir


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt-dir", action="append", default=None, required=False,
                   help="checkpoint dir (or a checkpoint_N file). Repeat to evaluate several "
                        "arms in ONE process, sharing the base FLUX/AE load.")
    p.add_argument("--arm-name", action="append", default=None,
                   help="label per --ckpt-dir (default: the run-name path component).")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--tasks", default="", help="comma-separated; default: the 20 both-split tasks")
    p.add_argument("--samples-per-task", type=int, default=8)
    p.add_argument("--viz-per-task", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-steps", type=int, default=50,
                   help="floored at 50 by the trainer; anything lower is silently 50")
    p.add_argument("--guidance", type=float, default=4.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--step", type=int, default=None, help="checkpoint step (default: latest)")
    p.add_argument("--image-size", type=int, default=256)
    p.add_argument("--split", default="test", choices=["test", "val", "train"],
                   help="which split the frozen pack is drawn from. `train` is IN-DISTRIBUTION "
                        "(the arms were optimized on it) and is for INSPECTION, not for any "
                        "generalization claim.")
    p.add_argument("--max-records-scanned", type=int, default=0,
                   help="stop each task's freeze scan after N records (0 = full pass). The "
                        "train split is 67 GiB/268 shards and is streamed ONCE PER TASK, so an "
                        "unbounded 24-task train freeze reads ~1.6 TB. With a cap the sample "
                        "set is drawn from the records SEEN, so it is frozen in the pack but "
                        "NOT reproducible from scratch (the loader interleave is "
                        "deterministic=False). Every arm still scores the identical pack.")
    p.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    p.add_argument("--pack-dir", default=None,
                   help="frozen sample pack. Reused if present, built if absent. "
                        "Default: <out-dir>/pack")
    p.add_argument("--supervision-version", type=int, default=1, choices=[1, 2, 3, 4, 5],
                   help="target/input render version. 1 = historical default. 2 = scale-dependent "
                        "degradations at RENDER resolution (un-breaks super_resolution, which is "
                        "an identity map at v1). 3 = v2 plus fixed-scale metric_depth and a real "
                        "instance/panoptic split (stops 5 task pairs rendering byte-identical "
                        "targets). 4 = v3 plus SR pinned to x4 on both splits, motion-blur "
                        "deblurring, subject-aware crops for keypoints/saliency (2026-09-05). "
                        "A pack built at one version cannot reuse parts from another.")
    p.add_argument("--rebuild-pack", action="store_true")
    p.add_argument("--build-pack-only", action="store_true")
    p.add_argument("--include-init-baseline", action="store_true",
                   help="also score a random (step-0) bank at the same seed")
    p.add_argument("--device-type", default="auto", choices=["auto", "tpu", "gpu", "cpu"])
    p.add_argument("--gcp-zone", default=None)
    p.add_argument("--flux-weights", default=DEFAULT_FLUX_WEIGHTS)
    p.add_argument("--flux-config", default=DEFAULT_FLUX_CONFIG)
    p.add_argument("--vae-weights", default=DEFAULT_VAE_WEIGHTS)
    p.add_argument("--vae-config", default=DEFAULT_VAE_CONFIG)
    p.add_argument("--stub-model", action="store_true",
                   help="SELF-TEST ONLY: build a tiny random-init Flux2+AE instead of loading "
                        "the 4B weights, so the whole pipeline can be exercised on CPU.")
    p.add_argument("--stub-hidden", type=int, default=64)
    return p


def _arm_label(ckpt_dir: str, override: str | None) -> str:
    if override:
        return override
    parts = [p for p in str(ckpt_dir).rstrip("/").split("/") if p]
    for i, part in enumerate(parts):
        if part == "checkpoints" and i > 0:
            return parts[i - 1]
    return parts[-1] if parts else "arm"


def build_stub_flux_and_ae(*, mesh, hidden: int, seed: int):
    """Tiny random-init stack with the SAME structural contract as the 4B one.

    128 packed latent channels (z_channels=32 * ps 2x2) is a hard requirement of
    `_prepare_latents_from_batch` / `_validate_autoencoder_compatibility`, so the stub keeps
    z_channels=32 and shrinks only `ch` / `ch_mult` / `num_res_blocks`. `ch` must stay a
    multiple of 32 because `ResnetBlock` hardcodes `GroupNorm(num_groups=32)`.
    """
    from flax import nnx
    from jflux2.model import Flux2, Flux2Params
    from jflux2.modules.autoencoder import AutoEncoder, AutoEncoderParams

    ae = AutoEncoder(AutoEncoderParams(
        resolution=256, in_channels=3, ch=32, out_ch=3, ch_mult=[1, 2],
        num_res_blocks=1, z_channels=32, rngs=nnx.Rngs(default=seed),
        param_dtype=jnp.float32))
    T._validate_autoencoder_compatibility(ae)
    T._replicate_module_params(ae, mesh)

    cfg = Flux2Params(
        in_channels=T.EXPECTED_LATENT_CHANNELS, context_in_dim=hidden, hidden_size=hidden,
        num_heads=4, depth=1, depth_single_blocks=1, axes_dim=[hidden // 16] * 4,
        use_guidance_embed=False, rngs=nnx.Rngs(default=seed), param_dtype=jnp.float32,
        lora_rank=0, gradient_checkpointing=False)
    model = Flux2(cfg)
    T.shard_model_params_for_hsdp(model, mesh)
    log(f"STUB model hidden={hidden} context_in_dim={cfg.context_in_dim} (NOT the real 4B)")
    return model, ae, cfg


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    argv, workdir = strip_workdir(argv)
    if workdir is not None:
        log(f"daemon --workdir (informational, stripped): {workdir}")
    args = build_parser().parse_args(argv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    on_train = args.split == "train"
    default_tasks = (list(CANONICAL_EVALUABLE_TASKS) + sorted(T.TRAIN_ONLY_VGI_TASKS)
                     if on_train else list(CANONICAL_EVALUABLE_TASKS))
    tasks = ([t.strip() for t in args.tasks.split(",") if t.strip()]
             if args.tasks else default_tasks)
    # Which tasks are evaluable depends on the SPLIT. The four train-only tasks
    # (illumination, intrinsic_decomposition, reflectance, shadow_detection) have zero test
    # records but DO have train data and DO have a trained row in the 24-task arms -- on the
    # train split they are the only way to see what those rows learned at all.
    forbidden = T.TEST_ONLY_VGI_TASKS if on_train else (T.TRAIN_ONLY_VGI_TASKS |
                                                        T.TEST_ONLY_VGI_TASKS)
    bad = [t for t in tasks if t in forbidden]
    if bad:
        raise RuntimeError(
            f"--tasks contains task(s) {bad} that are not evaluable on split={args.split!r}. "
            f"Train-only tasks {sorted(T.TRAIN_ONLY_VGI_TASKS)} have no TEST data; test-only "
            f"{sorted(T.TEST_ONLY_VGI_TASKS)} has no embedding row in any checkpoint.")
    if args.num_steps < T._MIN_VALIDATION_NUM_STEPS:
        log(f"WARNING --num-steps {args.num_steps} is below the trainer's hard floor "
            f"{T._MIN_VALIDATION_NUM_STEPS}; it will be silently floored to 50.")

    zone = T._resolve_zone(args.gcp_zone)
    bucket = T.ZONE_TO_BUCKET_PREFIX[zone]
    data_dir = (args.data_dir if args.data_dir.startswith("gs://")
                else T._resolve_bucket_relative_path(args.data_dir, bucket))
    log(f"zone={zone} bucket={bucket} data_dir={data_dir}")
    log(f"tasks({len(tasks)})={tasks}")

    pack_dir = args.pack_dir or str(out_dir / "pack")
    have_pack = (Path(pack_dir) / "samples.json").exists() and not args.rebuild_pack
    if have_pack:
        pack = load_sample_pack(pack_dir)
        missing = [t for t in tasks if not pack_indices_for_task(pack, t)]
        if missing:
            raise RuntimeError(
                f"existing pack {pack['pack_id']} at {pack_dir} has no samples for {missing}. "
                "Pass --rebuild-pack or point --pack-dir at a matching pack.")
        # The pack is authoritative: every sample IN it is scored. Say so loudly, because
        # --samples-per-task looks like it should shrink the run and does not (it only ever
        # affects pack CONSTRUCTION), and getting this wrong silently multiplies the runtime.
        # A reused pack silently overrides --image-size/--split/--data-dir (its pixels are
        # already baked). Refuse rather than report numbers under the wrong contract.
        for key, want in (("split", args.split), ("data_dir", data_dir),
                          ("image_size", int(args.image_size))):
            got = pack.get(key)
            if key == "data_dir" and got is not None:
                # Zone-insensitive: the mirrors hold identical bytes at identical relative
                # paths, so only a different RELATIVE path is a real conflict.
                if _bucket_relative(got) == _bucket_relative(want):
                    if got != want:
                        log(f"pack data_dir={got} vs requested {want}: same bucket-relative "
                            f"path in a mirrored zone, so the pixels are identical; reusing")
                    continue
            if got is not None and got != want:
                raise RuntimeError(
                    f"existing pack at {pack_dir} was frozen with {key}={got!r} but this run "
                    f"asked for {key}={want!r}. The pack's pixels are authoritative, so the "
                    f"run would silently ignore your flag. Pass --rebuild-pack, point "
                    f"--pack-dir elsewhere, or drop the conflicting flag.")
        pack_n = int(pack.get("samples_per_task", 0))
        if pack_n and pack_n != int(args.samples_per_task):
            log(f"WARNING --samples-per-task {args.samples_per_task} is IGNORED: the existing "
                f"pack at {pack_dir} was frozen with {pack_n} samples/task and all of them are "
                f"scored ({len(pack['records'])} denoises per arm). Use --rebuild-pack (or a "
                f"different --pack-dir) to change the sample count.")
    else:
        log(f"building frozen sample pack (min-hash over record_id) -> {pack_dir}")
        pack = build_sample_pack(
            tasks=tasks, data_dir=data_dir, samples_per_task=args.samples_per_task,
            image_size=args.image_size, split=args.split, seed=args.seed, pack_dir=pack_dir,
            max_records_scanned=args.max_records_scanned,
            supervision_version=args.supervision_version)
    if args.build_pack_only:
        log("build-pack-only: done")
        return 0

    if not args.ckpt_dir and not args.include_init_baseline:
        raise RuntimeError(
            "--ckpt-dir is required unless --build-pack-only or --include-init-baseline is "
            "set. (--include-init-baseline alone scores ONLY the untrained step-0 bank, "
            "which is how the pipeline is self-tested end-to-end without a checkpoint.)")
    args.ckpt_dir = args.ckpt_dir or []
    names = args.arm_name or []
    if names and len(names) != len(args.ckpt_dir):
        raise RuntimeError(
            f"--arm-name given {len(names)} time(s) but --ckpt-dir {len(args.ckpt_dir)} time(s); "
            "they must match 1:1 or --arm-name must be omitted entirely.")
    arms = [(_arm_label(d, names[i] if names else None), d)
            for i, d in enumerate(args.ckpt_dir)]

    mesh, val_device = build_mesh(args.device_type)
    if args.stub_model:
        model, ae, cfg = build_stub_flux_and_ae(mesh=mesh, hidden=args.stub_hidden,
                                                seed=args.seed)
    else:
        model, ae, cfg = build_flux_and_ae(
            mesh=mesh, bucket_prefix=bucket, flux_weights=args.flux_weights,
            flux_config=args.flux_config, vae_weights=args.vae_weights,
            vae_config=args.vae_config, seed=args.seed)
    context_in_dim = int(cfg.context_in_dim)

    log("computing zero-denoise reference baselines (copy / black / vae-floor)")
    baselines = compute_reference_baselines(ae=ae, mesh=mesh, val_device=val_device,
                                            pack=pack, tasks=tasks,
                                            batch_size=args.batch_size)

    written: list[str] = []
    jobs = list(arms) + ([("init_random_seed%d" % args.seed, None)]
                         if args.include_init_baseline else [])
    for arm, ckpt in jobs:
        summary = evaluate_arm(
            arm=arm, ckpt_dir=ckpt or "", step=args.step, model=model, ae=ae, mesh=mesh,
            val_device=val_device, pack=pack, tasks=tasks, context_in_dim=context_in_dim,
            batch_size=args.batch_size, num_steps=args.num_steps, guidance=args.guidance,
            seed=args.seed, out_dir=out_dir, viz_per_task=args.viz_per_task,
            baselines=baselines, init_bank=(ckpt is None),
            stub_model=bool(args.stub_model))
        written.append(str(out_dir / "arms" / arm / "summary.json"))

    index = {
        "schema": "jflux2.multi_task_eval_index.v1",
        "out_dir": str(out_dir), "pack_dir": pack_dir, "pack_id": pack["pack_id"],
        "split": pack["split"], "in_distribution": pack["split"] == "train",
        "max_records_scanned": int(pack.get("max_records_scanned", 0) or 0),
        "scan_truncated": bool(any((pack.get("scan_truncated") or {}).values())),
        "arms": [a for a, _ in jobs], "summaries": written,
        "tasks": tasks, "num_steps": max(args.num_steps, T._MIN_VALIDATION_NUM_STEPS),
        "guidance": args.guidance, "samples_per_task": args.samples_per_task,
        "stub_model": bool(args.stub_model),
    }
    with (out_dir / "eval_index.json").open("w", encoding="utf-8") as f:
        json.dump(index, f, indent=2)
    log(f"ALL ARMS DONE -> {out_dir / 'eval_index.json'}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception as exc:  # noqa: BLE001
        # Raise from THIS staged file so the unified classifier sees a deepest frame under
        # /staging/ => code_bug => TERMINAL, instead of resuming a deterministic failure
        # forever (the 46-restart trap; see compute/uinfra_wiring_entry.py).
        traceback.print_exc()
        sys.stdout.flush()
        raise RuntimeError(f"eval_multi_task_checkpoint FAILED: {type(exc).__name__}: {exc}") from exc
