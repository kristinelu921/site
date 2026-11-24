#!/usr/bin/env python3
"""T3/T4 offline evaluation of a 16-task in-context LoRA checkpoint (plan docs/PLAN_t16_lora_random_p_2026-09-05.md).

Runs `inference/engine_v2.py` (the certified v51 inference path) on the supervision-v4 test pack
(`test_s32_px256_v4`, 16 tasks x 32 records, same record ids as the v3 pack the frozen P-ladder used) and scores
with `jflux2.fasteval.metrics.score_pair` (task-native primaries), with the controls the frozen programme used:

  --mode positive_control   (T3, run FIRST) demo-swap on the colour coding: for each query the P demos are rendered
                            in coding A, then in coding B (both train-pool ids); the prediction must sit closer (L1)
                            to the target rendered in the demos' coding than in the other one. cond-correct is the
                            fraction of (query, coding) cases that follow the demos; chance 0.5, bar 0.75 (plan T3).
  --mode ladder             (T4) per task: real demos at P in --p-list; at P = max(--p-list) also scrambled
                            (deranged targets), grey (no content), oracle (the query's own pair), targets_only;
                            P=0 is represented by grey at P=1. Records are scored with the task-native primary;
                            NaN (all-black keypoint targets) is dropped by nanmean; per-cell checkpointing.
  --held-out-id             ladder cells rendered in a held-out coding id per family (demos AND target), the
                            prediction mapped back with invert_convention_np before scoring (F7).

Demos for a query are drawn leave-one-out from the other pack records of the same task (seeded per
(task, query, P)). Every cell writes per-record values so paired bootstraps over records are possible offline.

MULTI-HOST SHARDING (2026-09-07). The engine runs on ONE device, so a v6e-16/32 slice can only help by splitting the
queries across its hosts: `--shard auto` calls `jax.distributed.initialize()` (required on every host of a multi-host
slice before any device use), takes (jax.process_index(), jax.process_count()) as (k, n), evaluates the queries
q with position % n == k for every task, uses this host's own device, and writes `<out>.p<k>of<n>.json` (+ its own
partial). `--shard k/n` does the same without JAX distributed (for manual splits). `--merge` reads the shard files
next to `--out`, concatenates per-case / per-record lists and recomputes every statistic, and writes `--out`.
Defaults (`--shard none`) are byte-identical to the single-host behaviour. Why: single-host v6e-8 cards are the
scarce resource in this fleet while v6e-16 is offered hundreds of times a day; T3 on 16 x 64 x 2 generations is
~55 min on one device and ~14 min over four hosts.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import zlib
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tools"))

from jflux2.dataloaders import multi_task_loader_v2 as CV2  # noqa: E402
from jflux2.dataloaders import multi_task_loader_v3 as CV3  # noqa: E402
from jflux2.dataloaders import multi_task_loader_v4 as CV4  # noqa: E402
from jflux2.task_sets import T16_V3  # noqa: E402

# Coding conventions MUST match the loader the checkpoint was trained with (2026-09-08, T3 on the v4c 16-task run):
#   v2 (multi_task_loader_v2): golden-palette shift/perm ids, colorization = rgb permutation, restoration = chroma coding.
#   v3 (multi_task_loader_v3): palette tasks = a seeded permutation over the DISTINCT_PALETTE prefix, applied ONCE per
#       episode over the union of the tiles' labels (demos and query share the map); appearance/restoration tasks are
#       FIXED (identity only, no coding variation exists to test).
# Under v2 ids the v4c targets (DISTINCT_PALETTE colours) are not in the golden lookup, so every palette "coding" was a
# no-op: T3 read cond-correct 0.000 with l1_demo == l1_other on all 768 palette cases, and the fixed families sat at 0.5.
CONVENTIONS = {"v2": CV2, "v3": CV3, "v4": CV4}
CV = CV3
PC_IDS = {"gray": (0, 1), "normals": (0, 1), "palette": (0, 6), "appearance": (0, 1), "restoration": (0, 1)}


def task_seed(task: str) -> int:
    """Process-STABLE 16-bit task hash for the per-query rng (2026-09-08). Earlier ladders used ``hash(task)``, which
    Python randomises per process (PYTHONHASHSEED), so two ladders drew DIFFERENT demo sets for the same query: the
    16-task and 15-task T4 ladders of 2026-09-08 are paired on query records and targets only, not on demos."""
    return zlib.crc32(task.encode("utf-8")) & 0xFFFF


def log(msg: str) -> None:
    print(f"[t16eval {time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ----------------------------------------------------------------- pure helpers (unit-tested)
PACK_CACHE_ROOT = os.environ.get("MT_EVAL_PACK_CACHE", "/tmp/mt_eval_packs")


def resolve_pack(pack: str, cache_root: str = PACK_CACHE_ROOT, copy=None) -> str:
    """-> a LOCAL pack dir. A `gs://` pack (the full test pack lives in GCS only, user directive 2026-09-05) is copied
    once into `<cache_root>/<pack name>/` (samples.json + samples.npz); a local path is returned unchanged."""
    if not str(pack).startswith("gs://"):
        return str(pack)
    name = str(pack).rstrip("/").split("/")[-1]
    local = Path(cache_root) / name
    local.mkdir(parents=True, exist_ok=True)
    if copy is None:
        import tensorflow as tf  # noqa: PLC0415 - only needed for the GCS path
        copy = lambda src, dst: tf.io.gfile.copy(src, dst, overwrite=True)  # noqa: E731
    for fn in ("samples.json", "samples.npz"):
        dst = local / fn
        if not dst.is_file():
            tmp = local / (fn + ".part")
            copy(f"{str(pack).rstrip('/')}/{fn}", str(tmp))
            os.replace(tmp, dst)
    return str(local)


def draw_demo_indices(n_records: int, query: int, p: int, rng: np.random.Generator, candidates=None) -> np.ndarray:
    """`candidates` (2026-09-13, decision D28-A): restrict the demo pool, e.g. to the query's own SOURCE. Training episodes
    are single-source for every task; the pool-wide draw handed 64 % of DIODE queries NYU demos."""
    pool = range(n_records) if candidates is None else candidates
    others = np.array([i for i in pool if i != query])
    if others.size < p:
        others = np.array([i for i in range(n_records) if i != query])   # fall back to the pool when a source is tiny
    return rng.choice(others, size=p, replace=False)


def derange(idx: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """A permutation of `idx` with no fixed point (P>=2); P==1 returns a DIFFERENT record's index is impossible
    inside the set, so the caller pairs the demo input with another record's target instead."""
    if len(idx) < 2:
        return idx.copy()
    for _ in range(1000):
        perm = rng.permutation(len(idx))
        if not np.any(perm == np.arange(len(idx))):
            return idx[perm]
    return np.roll(idx, 1)


def nanmean_ci(values: list[float], n_boot: int = 20000, seed: int = 0):
    v = np.array([x for x in values if x == x], np.float64)
    if v.size == 0:
        return float("nan"), (float("nan"), float("nan")), 0
    rng = np.random.default_rng(seed)
    boots = np.array([v[rng.integers(0, v.size, v.size)].mean() for _ in range(n_boot)])
    return float(v.mean()), (float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))), int(v.size)


# ----------------------------------------------------------------- multi-host sharding (pure; unit-tested)
def parse_shard(spec):
    """'none'/'' -> (0, 1); 'k/n' -> (k, n); 'auto' -> None (resolve from JAX process index/count later)."""
    if spec in (None, "", "none"):
        return 0, 1
    if spec == "auto":
        return None
    k, n = (int(x) for x in str(spec).split("/"))
    if n < 1 or not (0 <= k < n):
        raise ValueError(f"--shard {spec!r}: need 0 <= k < n")
    return k, n


def select_shard(qs, k: int, n: int) -> list:
    """Queries at positions congruent to k mod n: the n shards partition `qs` and preserve its order."""
    return [q for i, q in enumerate(qs) if i % n == k]


def shard_out_path(out: Path, k: int, n: int) -> Path:
    return out if n == 1 else out.with_name(f"{out.stem}.p{k}of{n}{out.suffix}")


def merge_results(results: list) -> dict:
    """Merge shard result dicts (same mode/run/checkpoint/pack) into one; every statistic is recomputed from the
    concatenated per-case / per-record lists, so a merged cell equals a single-host cell over the same queries."""
    base = {k: v for k, v in results[0].items() if k != "cells"}
    base.pop("shard", None); base["merged_from"] = len(results); base["cells"] = {}
    keys = []
    for r in results:
        for key in r["cells"]:
            if key not in keys:
                keys.append(key)
    for key in keys:
        parts = [r["cells"][key] for r in results if key in r["cells"]]
        c = dict(parts[0])
        if "per_case" in c:
            per = sorted((x for p in parts for x in p["per_case"]), key=lambda x: (x["q"], x["demo_id"]))
            follow = [float(x["l1_demo"] < x["l1_other"]) for x in per]
            m, ci, n = nanmean_ci(follow)
            c.update(per_case=per, cond_correct=m, ci=ci, n_cases=n, verdict="PASS" if m >= 0.75 else "FAIL")
        else:
            recs = sorted((x for p in parts for x in p["per_record"]), key=lambda x: x["q"])
            vals = [float(x["value"]) for x in recs]
            m, ci, n = nanmean_ci(vals)
            c.update(per_record=recs, mean=m, ci=ci, n=n, n_nan=int(sum(1 for v in vals if v != v)))
        base["cells"][key] = c
    return base


def preds_path(save_dir, task: str, p: int, cond: str, held: int, shard: tuple) -> Path:
    """One .npz per ladder cell (and per shard, so shards never clobber each other):
    <save_dir>/<task>__P<p>__<cond>[__id<held>][.p<k>of<n>].npz with uint8 preds [n,H,W,3] + q/record_id/demos."""
    k, n = shard
    stem = f"{task}__P{int(p)}__{cond}" + (f"__id{int(held)}" if held else "") + (f".p{k}of{n}" if n > 1 else "")
    return Path(save_dir) / f"{stem}.npz"


def resolve_checkpoint_arg(run_name: str, checkpoint, bucket: str, user: str = "xtiange"):
    """`InferenceEngine.from_run` takes None (latest) or a FULL checkpoint path; a bare step such as "19000" was passed
    through as a path and failed in restore_checkpoint (job 3ef7a524, 2026-09-07: 'Unexpected checkpoint payload type:
    NoneType'). A bare step resolves to the trainer's layout gs://<bucket>/<user>/<run_name>/checkpoints/checkpoint_<step>."""
    if checkpoint is None or str(checkpoint).strip() == "":
        return None
    c = str(checkpoint).strip()
    if c.isdigit():
        return f"gs://{bucket}/{user}/{run_name}/checkpoints/checkpoint_{int(c)}"
    return c


def to_chw01(hwc_u8: np.ndarray) -> np.ndarray:
    return np.transpose(np.asarray(hwc_u8, np.float32) / 255.0, (2, 0, 1))


def to_hwc_u8(chw01: np.ndarray) -> np.ndarray:
    return np.rint(np.clip(np.transpose(np.asarray(chw01, np.float32), (1, 2, 0)), 0, 1) * 255).astype(np.uint8)


def code(hwc_u8: np.ndarray, task: str, cid: int) -> np.ndarray:
    if cid == 0:
        return np.asarray(hwc_u8, np.uint8)
    return np.rint(CV.apply_convention_np(np.asarray(hwc_u8, np.float32) / 255.0, task, cid) * 255).astype(np.uint8)


def code_episode(tiles_u8, task: str, cid: int) -> list:
    """Code every target tile of ONE episode with coding `cid`, consistently across the tiles.

    v3 palette codings are a permutation over the union of labels present in the episode, so the tiles must be coded
    together (per-tile coding would give demos and query different label->colour maps, i.e. an unlearnable episode).
    For every other family the per-tile transform is label-free, so this equals per-tile `code`."""
    tiles = [np.asarray(t, np.uint8) for t in tiles_u8]
    if not tiles:
        return tiles
    try:
        fam = CV.family_of(task)
    except (KeyError, ValueError):         # 2026-09-21: procedural unseen tasks are not in any family -> label-free
        fam = "unknown"
    if cid == 0 and not (CV is CV4 and fam == "palette"):   # v4 palette: cid 0 = compact ranks, not raw ids
        return tiles
    if hasattr(CV, "apply_palette_np") and fam == "palette":
        stack = np.stack(tiles).astype(np.float32) / 255.0
        coded = CV.apply_palette_np(stack, int(cid))                # v3 and v4 code the episode's tiles TOGETHER
        return [np.rint(c * 255).astype(np.uint8) for c in coded]
    return [code(t, task, cid) for t in tiles]


def uncode(pred_hwc01: np.ndarray, task: str, cid: int) -> np.ndarray:
    if cid == 0:
        return np.rint(np.clip(pred_hwc01, 0, 1) * 255).astype(np.uint8)
    return np.rint(np.clip(CV.invert_convention_np(pred_hwc01, task, cid), 0, 1) * 255).astype(np.uint8)


def build_conditions(imgs, msks, task, q, p, rng, *, cid: int, which: tuple[str, ...], candidates=None):
    """-> ({condition: [(demo_in_chw01, demo_tg_chw01), ...]}, demo indices, query target coded with `cid` as part of
    the same episode) for one query. `candidates`: D28-A demo pool restriction."""
    n = len(imgs)
    di = draw_demo_indices(n, q, p, rng, candidates)
    grey = np.full_like(imgs[0], 128)
    conds = {}
    # One episode = the demo targets AND the query target: code them together so a v3 palette map is shared.
    ep = code_episode([msks[j] for j in di] + [msks[q]], task, cid)
    demo_t, query_t = ep[:-1], ep[-1]
    if "real" in which:
        conds["real"] = [(to_chw01(imgs[j]), to_chw01(t)) for j, t in zip(di, demo_t)]
    if "scrambled" in which:
        if p >= 2:
            dj = derange(di, rng)
            # Permute the already-coded targets. Coding again in scrambled order
            # changes v4's first-appearance palette ranks and confounds pairing
            # with a different class->colour map from the unchanged query target.
            coded_by_demo = {int(j): t for j, t in zip(di, demo_t)}
            scr = [coded_by_demo[int(j)] for j in dj]
        else:
            # A singleton has no within-episode permutation. Retain the legacy
            # outside-target control; it is not a palette-preserving pairing test.
            dj = draw_demo_indices(n, q, 1, np.random.default_rng(rng.integers(1 << 30)))
            scr = code_episode([msks[b] for b in dj] + [msks[q]], task, cid)[:-1]
        conds["scrambled"] = [(to_chw01(imgs[a]), to_chw01(t)) for a, t in zip(di, scr)]
    if "grey" in which:
        conds["grey"] = [(to_chw01(grey), to_chw01(grey))] * p
    if "oracle" in which:
        conds["oracle"] = [(to_chw01(imgs[q]), to_chw01(query_t))] * p
    if "targets_only" in which:
        conds["targets_only"] = [(to_chw01(t), to_chw01(t)) for t in demo_t]
    return conds, di, query_t


# ----------------------------------------------------------------- driver
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["ladder", "positive_control"], default="ladder")
    ap.add_argument("--run-name", default="v51-t16v3-randp-conv-r64-30k")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--config", default=None,
                    help="training yaml to run the checkpoint under (default: configs/**/<run_name>.yaml). "
                         "Use the frozen copy in the run's log dir when the run was launched from a differently "
                         "named yaml (the v4c 16-task run: <run_name>-mp.yaml); the engine's identity audit "
                         "compares its sha256 with the checkpoint manifest")
    ap.add_argument("--bucket", default="kmh-gcp-us-east5")
    ap.add_argument("--pack", default="/kmh-nfs-ssd-us-mount/logs/xtiange/mt_eval_packs/test_s64_px256_v4c",
                    help="validation pack (local) or the full TEST pack gs://kmh-gcp-us-east5/xtiange/mt_eval_packs/test_full_px256_v4c")
    ap.add_argument("--tasks", default=",".join(T16_V3))
    ap.add_argument("--queries", type=int, default=0, help="per task; 0 = all pack records")
    ap.add_argument("--p-list", default="1,2,4")
    ap.add_argument("--controls-at-max-p", default="scrambled,grey,oracle,targets_only")
    ap.add_argument("--held-out-id", action="store_true", help="render demos+target in the family's held-out id")
    ap.add_argument("--num-steps", type=int, default=25)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-pairs", type=int, default=5)
    ap.add_argument("--out", required=True)
    ap.add_argument("--gcp-zone", default=None)
    ap.add_argument("--shard", default="none", help="'none' | 'k/n' | 'auto' (multi-host: JAX process index/count)")
    ap.add_argument("--demo-source-policy", choices=("pool", "same"), default="pool",
                    help="D28-A (2026-09-13): 'same' draws demos from the query's own source, as training episodes are")
    ap.add_argument("--conventions", choices=sorted(CONVENTIONS), default="v3",
                    help="coding conventions of the loader the checkpoint was TRAINED with (v4c runs: v3)")
    ap.add_argument("--merge", action="store_true", help="merge <out>.p*of*.json shard files into --out (no model)")
    ap.add_argument("--save-preds", default=None,
                    help="directory: save every ladder cell's predictions as uint8 .npz (see preds_path) so any metric "
                         "(official variants, later fixes) can be scored offline on identical records without re-inference; "
                         "~12.6 MB per cell of 64 queries at 256px")
    args = ap.parse_args()
    if args.gcp_zone:
        os.environ["GCP_ZONE"] = args.gcp_zone

    if args.merge:
        out = Path(args.out)
        files = sorted(out.parent.glob(f"{out.stem}.p*of*{out.suffix}"))
        files = [f for f in files if ".partial" not in f.name]
        if not files:
            raise SystemExit(f"--merge: no shard files {out.stem}.p*of*{out.suffix} next to {out}")
        merged = merge_results([json.load(open(f)) for f in files])
        json.dump(merged, open(out, "w"), indent=1)
        log(f"merged {len(files)} shard files -> {out} ({len(merged['cells'])} cells)")
        return 0
    global CV
    CV = CONVENTIONS[args.conventions]
    log(f"coding conventions: {args.conventions} ({CV.__name__})")
    shard = parse_shard(args.shard)
    if shard is None:
        import jax  # noqa: PLC0415
        try:
            jax.distributed.initialize()
        except Exception as exc:  # noqa: BLE001 - single-process hosts raise here; that is the (0, 1) case
            log(f"jax.distributed.initialize() not applicable ({type(exc).__name__}: {str(exc)[:80]}); single process")
        shard = (int(jax.process_index()), int(jax.process_count()))
    shard_k, shard_n = shard
    log(f"shard {shard_k}/{shard_n}")

    import eval_multi_task_checkpoint as E  # noqa: E402
    pk = E.load_sample_pack(resolve_pack(args.pack))
    tasks = [t for t in args.tasks.split(",") if t]
    p_list = sorted({int(x) for x in args.p_list.split(",") if x})
    controls = tuple(c for c in args.controls_at_max_p.split(",") if c)
    from jflux2.fasteval.metrics import score_pair
    from inference.engine_v2 import InferenceEngine
    ckpt_path = resolve_checkpoint_arg(args.run_name, args.checkpoint, args.bucket, user="xtiange")
    log(f"checkpoint arg {args.checkpoint!r} -> {ckpt_path}")
    eng = InferenceEngine.from_run(args.run_name, ckpt_path, bucket=args.bucket, user="xtiange",
                                   verbose=True, audit_restore=True, max_pairs=max(args.max_pairs, max(p_list) + 1),
                                   config_path=args.config)

    if shard_n > 1:
        import jax  # noqa: PLC0415
        object.__setattr__(eng, "device", jax.local_devices()[0])   # jax.devices()[0] lives on host 0 only
        log(f"multi-host: using local device {eng.device} on process {shard_k}")
    out = shard_out_path(Path(args.out), shard_k, shard_n); partial = out.with_suffix(".partial.json")
    result = json.load(open(partial)) if partial.exists() else {"mode": args.mode, "run_name": args.run_name,
                                                                "checkpoint": args.checkpoint, "checkpoint_path": ckpt_path,
                                                                "config_path": args.config,
                                                                "conventions": args.conventions,
                                                                "save_preds": args.save_preds,
                                                                "pack_id": pk["pack_id"],
                                                                "num_steps": args.num_steps, "seed": args.seed,
                                                                "shard": [shard_k, shard_n], "cells": {}}
    cells = result["cells"]

    def save():
        tmp = partial.with_suffix(".tmp"); json.dump(result, open(tmp, "w"), indent=1); os.replace(tmp, partial)

    for task in tasks:
        idx = [i for i, r in enumerate(pk["records"]) if r["task"] == task]
        imgs, msks = pk["images"][idx], pk["masks"][idx]
        srcs = [str(pk["records"][i].get("source")) for i in idx]
        same_src = {s_: [k for k, x in enumerate(srcs) if x == s_] for s_ in set(srcs)}
        cand_of = (lambda q: same_src[srcs[q]]) if args.demo_source_policy == "same" else (lambda q: None)
        v4_palette = (CV is CV4 and CV.family_of(task) == "palette")
        qs = list(range(len(idx))) if args.queries <= 0 else list(range(min(args.queries, len(idx))))
        qs = select_shard(qs, shard_k, shard_n)
        fam = CV.family_of(task)
        held = CV.convention_pools(task)[1][0] if args.held_out_id else 0
        if args.mode == "positive_control":
            key = f"{task}@pc{max(p_list)}"
            if key in cells:
                log(f"skip {key} (done)"); continue
            if fam not in PC_IDS:
                log(f"{key}: family {fam!r} has no coding variation under {args.conventions}; positive control not applicable")
                cells[key] = {"family": fam, "ids": None, "cond_correct": None, "ci": None, "n_cases": 0, "per_case": [],
                              "verdict": "N/A"}; save(); continue
            a_id, b_id = PC_IDS[fam]
            follow = []; per = []
            for q in qs:
                rng = np.random.default_rng([args.seed, task_seed(task), q, max(p_list)])
                for demo_id, other_id in ((a_id, b_id), (b_id, a_id)):
                    conds, di, query_t = build_conditions(imgs, msks, task, q, max(p_list), rng, cid=demo_id, which=("real",))
                    pred = np.transpose(eng.predict(conds["real"], to_chw01(imgs[q]), num_steps=args.num_steps, noise_seed=args.seed), (1, 2, 0))
                    # the query target in the DEMO coding shares the episode's map; the OTHER coding is applied to the
                    # same episode's tiles so a palette permutation is again consistent with what those demos would carry
                    t_demo = np.asarray(query_t, np.float32) / 255.0
                    t_other = np.asarray(code_episode([msks[j] for j in di] + [msks[q]], task, other_id)[-1], np.float32) / 255.0
                    l_demo, l_other = float(np.abs(pred - t_demo).mean()), float(np.abs(pred - t_other).mean())
                    follow.append(float(l_demo < l_other)); per.append({"q": q, "demo_id": demo_id, "l1_demo": l_demo, "l1_other": l_other})
                log(f"{task} q{q}: running cond-correct {np.mean(follow):.3f} ({len(follow)} cases)")
            m, ci, n = nanmean_ci(follow)
            cells[key] = {"family": fam, "ids": [a_id, b_id], "cond_correct": m, "ci": ci, "n_cases": n, "per_case": per,
                          "verdict": "PASS" if m >= 0.75 else "FAIL"}
            log(f"{key}: cond-correct {m:.3f} [{ci[0]:.3f}, {ci[1]:.3f}] n={n} -> {cells[key]['verdict']}"); save()
            continue
        for p in p_list:
            which = ("real",) + (controls if p == max(p_list) else ()) + (("grey",) if p == min(p_list) and "grey" not in controls else ())
            for cond in which:
                key = f"{task}@P{p}" + ("" if cond == "real" else f"~{cond}") + (f"#id{held}" if held else "")
                if key in cells:
                    log(f"skip {key} (done)"); continue
                vals, recs, preds, coded_targets, covered_sets, cids = [], [], [], [], [], []
                for q in qs:
                    rng = np.random.default_rng([args.seed, task_seed(task), q, p])
                    # v4 palette (D6-B / D1-B): every query gets its own episode coding from the EVAL pool; the
                    # prediction is scored in that coding against the coded query target, and classes/joints that no
                    # demo carried are recorded so the offline scorer can exclude them (their colour is unknowable).
                    q_cid = int(held)
                    if v4_palette and not held:
                        pool_ids = CV.convention_pools(task)[1]
                        q_cid = int(pool_ids[int(rng.integers(0, len(pool_ids)))])
                    conds, di, query_t = build_conditions(imgs, msks, task, q, p, rng, cid=q_cid, which=(cond,), candidates=cand_of(q))
                    pred_chw = eng.predict(conds[cond], to_chw01(imgs[q]), num_steps=args.num_steps, noise_seed=args.seed)
                    if v4_palette:
                        pred_u8 = np.rint(np.clip(np.transpose(pred_chw, (1, 2, 0)), 0, 1) * 255).astype(np.uint8)
                        target_u8 = np.asarray(query_t, np.uint8)
                        demo_t = code_episode([msks[j] for j in di] + [msks[q]], task, q_cid)[:-1]
                        cov = sorted(set(int(x) for t in demo_t for x in np.unique(CV._posmap(CV._packed(np.asarray(t, np.int64)))) if x > 0))
                    else:
                        pred_u8 = uncode(np.transpose(pred_chw, (1, 2, 0)), task, held)
                        target_u8 = msks[q]; cov = None
                    coded_targets.append(target_u8); covered_sets.append(cov); cids.append(q_cid)
                    s = score_pair(task, pred_u8, target_u8)
                    v = float(s["primary_value"]); vals.append(v); preds.append(pred_u8)
                    recs.append({"q": q, "record_id": pk["records"][idx[q]]["record_id"], "demos": [int(x) for x in di],
                                 "primary": s["primary_metric"], "lower": bool(s["primary_lower_is_better"]), "value": v,
                                 "l1_01": float(s["l1_01"]),
                                 # every numeric score the family produced (2026-09-08), so secondary/official metrics
                                 # need no re-inference; strings (family, primary name) are dropped
                                 "scores": {k: float(x) for k, x in s.items() if isinstance(x, (int, float)) and not isinstance(x, bool)}})
                if args.save_preds and preds:
                    pp = preds_path(args.save_preds, task, p, cond, held, (shard_k, shard_n))
                    pp.parent.mkdir(parents=True, exist_ok=True)
                    tmpp = pp.with_suffix(".tmp.npz")
                    np.savez(tmpp, preds=np.stack(preds).astype(np.uint8), q=np.asarray(qs, np.int64),
                             record_id=np.asarray([r["record_id"] for r in recs]),
                             demos=np.asarray([r["demos"] for r in recs], np.int64), task=task, P=int(p), condition=cond,
                             coding_id=int(held), run_name=args.run_name, checkpoint_path=str(ckpt_path), pack_id=pk["pack_id"],
                             conventions=args.conventions, demo_source_policy=args.demo_source_policy,
                             coding_ids=np.asarray(cids, np.int64),
                             coded_targets=(np.stack(coded_targets).astype(np.uint8) if v4_palette else np.zeros((0,), np.uint8)),
                             covered=np.asarray([(c + [-1] * (1024 - len(c)))[:1024] if c is not None else [-2] * 1024 for c in covered_sets], np.int64))
                    os.replace(tmpp, pp)
                m, ci, n = nanmean_ci(vals)
                cells[key] = {"task": task, "P": p, "condition": cond, "coding_id": held, "mean": m, "ci": ci, "n": n,
                              "n_nan": int(sum(1 for v in vals if v != v)), "primary": recs[0]["primary"] if recs else None,
                              "lower_is_better": recs[0]["lower"] if recs else None, "per_record": recs}
                log(f"{key}: {recs[0]['primary'] if recs else '?'} {m:.4f} [{ci[0]:.4f}, {ci[1]:.4f}] n={n} nan={cells[key]['n_nan']}"); save()
    json.dump(result, open(out, "w"), indent=1)
    log(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
