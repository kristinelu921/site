#!/usr/bin/env python3
"""metrics_v3: corrections to the v2 official-metric scorers, found by the 2026-09-12 interface audit.

Version-forward from `jflux2.fasteval.metrics_v2` (untouched, so every number already reported can still be
reproduced exactly). This module imports v2 and overrides ONLY what the audit proved wrong. Two kinds of change:

**A. Plain defects** — v2 computes something other than what it says it computes.

  saliency adaptive_fbeta   v2 used `th = min(2*mean, 1.0)`. For a target whose mean luma >= 0.5 that is
                            `p >= 1.0`, which is EMPTY even for a pixel-perfect prediction, because
                            `_luma01` of pure white is 0.9999999999999999 (the BT.601 weights do not sum to
                            1 in binary floating point). MEASURED: a perfect prediction scored
                            adaptive_fbeta = 0.0 on 1 of 64 pack records. Fixed by clipping the threshold to
                            the image maximum instead of to 1.0.

  semantic miou_dataset     v2 pooled per-class intersection/union keyed on the DISTINCT_PALETTE index alone.
                            The index is assigned from each dataset's own label id, so ade20k position 5 and
                            nyu position 5 are DIFFERENT semantic classes. MEASURED: 54 of 220 (source, class)
                            pairs were merged; the pooled number moves t16 0.0434 -> 0.0414 and
                            t15 0.0344 -> 0.0359, i.e. the reported t16-t15 gap +0.0090 -> +0.0055.
                            Fixed by keying the pool on (source, class). The PRIMARY per-record `miou` was
                            never affected (it is computed within one record).

  aggregate() and PSNR      A bit-exact prediction gives PSNR = +inf, which v2's `isfinite` filter dropped
                            SILENTLY: a perfect appearance cell aggregated to n=0, mean=nan instead of to
                            "perfect". No published number moves (real outputs are never bit-exact), but the
                            verification protocol itself reported nan. Now counted (`n_inf`) and exposed as
                            `mean_capped` at PSNR_CAP_DB.

**B. Definition corrections** — v2 computes a well-defined quantity that is NOT the metric the community
reports. These MOVE HEADLINE NUMBERS, so both variants are returned and the legacy key is kept.

  panoptic / instance pq    v2 scored the void region (the encoder's black class-0 marker, 15.6% of nyu
                            pixels) as an ordinary segment. Kirillov et al. exclude void from PQ and drop
                            predicted segments that lie mostly inside it. MEASURED on t16_ck30000 P4/real:
                            panoptic 0.1284 -> 0.1774 (+0.0490), instance 0.1433 -> 0.1955 (+0.0522),
                            about +38% relative. The void-excluded variant still scores exactly 1.0 on
                            GT-vs-GT. New key `pq_novoid` (+ `sq_novoid`, `rq_novoid`); `pq` keeps the v2
                            value so old comparisons stay reproducible.

  relative_depth delta1     v2's MiDaS scale+shift fit is UNCONSTRAINED, so it absorbs a sign inversion: a
                            perfectly inverted prediction (near-dark instead of near-bright) scores
                            delta1_aligned = 1.0000 with align_scale = -1.0. MEASURED on t16_ck30000:
                            P4/real 0.4834 -> 0.4802 sign-constrained (12 of 64 P1 predictions actually fit a
                            NEGATIVE scale, vs 3 of 64 at P4), so the inflation is larger for weaker arms and
                            compresses the P1->P4 scaling gap. New key `delta1_aligned_ssc`.

WHAT THIS MODULE DOES NOT FIX. The audit also found defects in the RENDERED GROUND TRUTH itself, which no
scorer can repair — they need a re-render (and, for training, a re-run):
  * NYU metric_depth is decoded over [0.5, 10] m although the payload declares `meters_0_to_10_fixed_range`
    with `range_meters [0.0, 10.0]` and `resolution_mm 0.1526` (= 10/65535; the code's 9.5/65534 = 0.14496 mm
    does not match). Every NYU depth target therefore carries 0.4999 + 0.9500 * d_true: +45% at 1 m, +20% at
    2 m, +5% at 5 m, 0% at 10 m.  (jflux2/dataloaders/vgi_rgb/rgb_supervision.py:148)
  * metric_depth and surface_normals are resized BILINEARLY (`discrete=False`), which blends the reserved
    code-0 invalid marker with valid codes. A synthetic constant 5.00 m surface beside a hole produces ring
    pixels decoding to 27-48 m that the validity test accepts as ground truth. Ring fraction in the pack:
    surface_normals/diode 3.51% mean (16.06% max, 62/64 records), metric_depth/diode 1.98% (10.68% max);
    metric_depth/nyu is unaffected (no invalid pixels).
  * DIODE ships a per-pixel validity mask that the renderer selects away and never applies.
See `research_reports/metric_interface_audit_2026-09-12.md`.
"""
from __future__ import annotations

import os
import sys

import numpy as np

from jflux2.fasteval import metrics as M1
from jflux2.fasteval import metrics_v2 as M2
from jflux2.dataloaders.vgi_rgb import rgb_supervision_v3 as V3

EDGE_THRESHOLDS = M2.EDGE_THRESHOLDS
FBETA_SQ = M2.FBETA_SQ
PSNR_CAP_DB = 100.0

# primaries: unchanged from v2 except where a definition correction supersedes the legacy key
PRIMARY_V3 = dict(M2.PRIMARY_V2)
PRIMARY_V3["instance_segmentation"] = ("pq_novoid", False)
PRIMARY_V3["panoptic_segmentation"] = ("pq_novoid", False)
PRIMARY_V3["relative_depth"] = ("delta1_aligned_ssc", False)
PRIMARY_V3["colorization"] = ("psnr_chroma", False)

# 2026-09-16 (user decision, ledger 05:05): for the five APPEARANCE tasks the decision metric is LPIPS, not PSNR —
# copying the input scores 28.4 dB on deblurring (above the trained adapter's 24.9) because PSNR rewards the blurry
# conditional mean, while under LPIPS the trained adapter wins on all five. Opt-in so every earlier comparison keeps
# its meaning: pass `perceptual=True` to `score_pair_v3`, or read `PRIMARY_V3_PERCEPTUAL` / `primary_v3(task, True)`.
# NOTE `lpips_alex` needs torch; on a host without it the value is NaN, so TPU jobs keep the distortion primary and
# are re-scored offline from the saved predictions (`jflux2/analysis/lpips_rescore_v1.py`).
APPEARANCE_TASKS_V3 = ("colorization", "denoising", "deblurring", "inpainting", "super_resolution")
PRIMARY_V3_PERCEPTUAL = dict(PRIMARY_V3)
for _t in APPEARANCE_TASKS_V3:
    PRIMARY_V3_PERCEPTUAL[_t] = ("lpips_alex", True)


# Procedural UNSEEN tasks (2026-09-21, tools/make_procedural_pack_v1.py): an exact operator on a natural photo. Scored
# like the appearance tasks — PSNR by default, LPIPS as the perceptual primary — because their targets are images
# rather than labels. No library prompt exists for them by construction.
PROCEDURAL_TASKS_V1 = ("proc_canny", "proc_hue120", "proc_hflip", "proc_blur", "proc_invert", "proc_sobel")
for _t in PROCEDURAL_TASKS_V1:
    PRIMARY_V3[_t] = ("psnr", False)
    PRIMARY_V3_PERCEPTUAL[_t] = ("lpips_alex", True)


def primary_v3(task: str, perceptual: bool = False):
    """(metric name, lower_is_better) for `task`; `perceptual=True` swaps the appearance tasks to LPIPS."""
    return (PRIMARY_V3_PERCEPTUAL if perceptual else PRIMARY_V3)[task]
PRIMARY_V3["face_landmarks"] = ("nme_bbox_matched", True)
PRIMARY_V3["object_keypoints"] = ("pck_bbox_reject", False)
PRIMARY_V3["human_pose_2d"] = ("pck_bbox_reject", False)
POOLED_V3 = dict(M2.POOLED_V2)


# ---------------------------------------------------------------- A. defect fixes
def saliency_metrics(pred_u8, targ_u8) -> dict:
    """v2's saliency metrics with the adaptive threshold clipped to the image max rather than to 1.0."""
    out = M2.saliency_metrics(pred_u8, targ_u8)
    p = M2._luma01(pred_u8); t = M2._luma01(targ_u8)
    gt = t > 0.5
    if not gt.any():
        return out
    th = min(2.0 * float(p.mean()), float(p.max()))
    pb = p >= th
    tp = float((pb & gt).sum())
    pr = tp / max(float(pb.sum()), 1.0); rc = tp / max(float(gt.sum()), 1.0)
    out["adaptive_fbeta"] = float((1 + FBETA_SQ) * pr * rc / max(FBETA_SQ * pr + rc, M2._EPS))
    out["adaptive_thresh"] = float(th)
    return out


def semantic_metrics(pred_u8, targ_u8, *, source=None, scored_positions=None) -> dict:
    """v2's semantic metrics with the dataset-level pool keyed on (source, class), not class alone.

    `scored_positions` (2026-09-13, decision D6-B): under demo-defined coding a query class that appears in NO demo has
    an arbitrary colour the model cannot know; pass the set of palette positions present in the demos and only those
    classes are scored (`miou`, pool). The excluded share is reported as `uncovered_frac`."""
    out = M2.semantic_metrics(pred_u8, targ_u8)
    if scored_positions is not None:
        keep = set(int(p) for p in scored_positions)
        cols = M1._unique_colours(targ_u8)
        pl = M1.decode_to_target_vocabulary(pred_u8, targ_u8, cols)
        tl = M1.decode_to_target_vocabulary(targ_u8, targ_u8, cols)
        pos = M2._palette_positions(np.asarray(cols, np.uint8)[None])[0]
        pc = pos[pl]; tc = pos[tl]
        present = [int(c) for c in np.unique(tc) if c > 0]
        classes = [c for c in present if c in keep]
        inter, union, ious = {}, {}, []
        for c in classes:
            p_ = pc == c; t_ = tc == c
            i = int((p_ & t_).sum()); u = int((p_ | t_).sum())
            inter[c] = i; union[c] = u
            if u > 0:
                ious.append(i / u)
        out["miou"] = float(np.mean(ious)) if ious else float("nan")
        out["n_classes_in_target"] = float(len(classes))
        out["uncovered_frac"] = float(1.0 - len(classes) / len(present)) if present else 0.0
        out["_pool"] = {"inter": inter, "union": union}
    src = str(source) if source is not None else "_"
    pool = out.get("_pool") or {}
    out["_pool"] = {"inter": {(src, c): v for c, v in pool.get("inter", {}).items()},
                    "union": {(src, c): v for c, v in pool.get("union", {}).items()}}
    return out


# ---------------------------------------------------------------- A2. diagnostics that separate structure from calibration
def metric_depth_metrics(pred_u8, targ_u8, *, source=None) -> dict:
    """v2's metric-depth numbers plus SCALE-ALIGNED variants (`abs_rel_scale_aligned`, `delta1_scale_aligned`).

    The 2026-09-13 floor investigation found the baseline has real depth structure (per-record Pearson r 0.734 with
    the target) but a wrong ABSOLUTE scale, and on the exponential code->metres map that alone drops it below a
    flat-card null. Sign, stated in the encoding's own terms (near = bright): NYU predictions are 16 codes DARKER
    than target = too FAR (median 4.03 vs 2.82 m); DIODE 22 codes BRIGHTER = too NEAR (4.76 vs 6.11 m). Both point
    at the hypersim training median of 3.93 m -- a single-source scale prior, not a renderer bug (2026-09-13
    pre-retrain review; an earlier docstring here had the near/far wording inverted). The scale-aligned number applies one
    global multiplicative correction per record (median ratio, the standard zero-shot depth protocol) so that
    structure and calibration can be read separately. Measured on the baseline: abs_rel 0.5295 -> ~0.20.
    Reported ALONGSIDE the metric number, never instead of it.
    """
    out = dict(M2.metric_depth_metrics(pred_u8, targ_u8, source=source))
    gm = V3.depth_metres_from_rgb(np.asarray(targ_u8, np.uint8))
    p8 = np.asarray(pred_u8, np.uint8).copy(); p8[p8[..., 0] == 0, 0] = 1
    pm = V3.depth_metres_from_rgb(p8)
    cap = M2.NYU_MAX_DEPTH_M if (source and "nyu" in str(source)) else V3.METRIC_DEPTH_RANGE_M[1]
    valid = np.isfinite(gm) & (gm > 0) & (gm <= cap) & np.isfinite(pm) & (pm > 0)
    if not valid.any():
        out["abs_rel_scale_aligned"] = float("nan"); out["delta1_scale_aligned"] = float("nan"); out["align_scale_metres"] = float("nan")
        return out
    p = pm[valid]; g = gm[valid]
    s = float(np.median(g / p))                        # one global scale per record
    pa = np.clip(p * s, V3.METRIC_DEPTH_RANGE_M[0], cap)
    ratio = np.maximum(pa / g, g / pa)
    out["abs_rel_scale_aligned"] = float(np.mean(np.abs(pa - g) / g))
    out["delta1_scale_aligned"] = float(np.mean(ratio < 1.25))
    out["align_scale_metres"] = s
    return out


def semantic_metrics_large(pred_u8, targ_u8, *, source=None, min_frac: float = 0.05, scored_positions=None) -> dict:
    """v3 semantic metrics plus `miou_ge5pct`: mIoU over classes covering >= 5 % of the image.

    Classes under 1 % of the image are 34 % of all scored classes on the pack and each counts equally, so
    dozens of near-zero tiny classes dominate the plain mean. This view reports the coarse segmentation the
    model actually does. Reported ALONGSIDE `miou`, never instead of it.
    """
    out = semantic_metrics(pred_u8, targ_u8, source=source, scored_positions=scored_positions)
    cols = M1._unique_colours(targ_u8)
    pl = M1.decode_to_target_vocabulary(pred_u8, targ_u8, cols)
    tl = M1.decode_to_target_vocabulary(targ_u8, targ_u8, cols)
    pos = M2._palette_positions(np.asarray(cols, np.uint8)[None])[0]
    pc = pos[pl]; tc = pos[tl]
    n = tc.size; ious = []
    for c in np.unique(tc):
        if c <= 0:
            continue
        t = tc == c
        if t.sum() / n < min_frac:
            continue
        p = pc == c
        u = (p | t).sum()
        if u:
            ious.append((p & t).sum() / u)
    out["miou_ge5pct"] = float(np.mean(ious)) if ious else float("nan")
    out["n_classes_ge5pct"] = float(len(ious))
    return out


# ---------------------------------------------------------------- B. definition corrections
def partition_metrics(pred_u8, targ_u8) -> dict:
    """Class-agnostic PQ, both as v2 computed it (`pq`) and with VOID EXCLUDED per Kirillov (`pq_novoid`).

    The v4c palette encoder reserves DISTINCT_PALETTE position 0 (black) for void/unlabelled. panopticapi
    excludes void target regions from PQ and removes predicted segments whose area is mostly inside void;
    v2 scored void as an ordinary segment, which depressed pq by about 38% relative on nyu.
    """
    out = dict(M2.partition_metrics(pred_u8, targ_u8))
    cols = M1._unique_colours(targ_u8)
    pl = M1.decode_to_target_vocabulary(pred_u8, targ_u8, cols)
    tl = M1.decode_to_target_vocabulary(targ_u8, targ_u8, cols)
    pos = M2._palette_positions(np.asarray(cols, np.uint8)[None])[0]
    void_vocab = [i for i, p in enumerate(pos) if p == 0]
    if not void_vocab:
        out.update({"pq_novoid": out["pq"], "sq_novoid": out["sq"], "rq_novoid": out["rq"],
                    "void_frac": 0.0})
        return out
    void = np.isin(tl, void_vocab)
    keep = ~void
    if not keep.any():
        out.update({"pq_novoid": float("nan"), "sq_novoid": float("nan"), "rq_novoid": float("nan"),
                    "void_frac": 1.0})
        return out
    # drop predicted segments lying mostly inside void (panopticapi's VOID rule), then score on non-void pixels
    pl2 = pl.copy()
    for ps in np.unique(pl):
        m = pl == ps
        if m.any() and (m & void).sum() / m.sum() > 0.5:
            pl2[m] = -1
    pv, tv = pl2[keep], tl[keep]
    tsegs = [s for s in np.unique(tv)]
    psegs = [s for s in np.unique(pv) if s != -1]
    if not tsegs:
        out.update({"pq_novoid": float("nan"), "sq_novoid": float("nan"), "rq_novoid": float("nan"),
                    "void_frac": float(void.mean())})
        return out
    pairs = []
    for ti in tsegs:
        tm = tv == ti
        for pi in psegs:
            pm = pv == pi
            inter = int((tm & pm).sum())
            if inter:
                pairs.append((inter / int((tm | pm).sum()), int(ti), int(pi)))
    pairs.sort(reverse=True)
    used_t, used_p, ious = set(), set(), []
    for iou, ti, pi in pairs:
        if iou < 0.5 or ti in used_t or pi in used_p:
            continue
        used_t.add(ti); used_p.add(pi); ious.append(iou)
    tp = len(ious); fp = len(psegs) - tp; fn = len(tsegs) - tp
    sq = float(np.mean(ious)) if tp else 0.0
    rq = tp / max(tp + 0.5 * fp + 0.5 * fn, M2._EPS)
    out.update({"pq_novoid": float(sq * rq), "sq_novoid": sq, "rq_novoid": float(rq),
                "n_target_segs_novoid": float(len(tsegs)), "void_frac": float(void.mean())})
    return out


def relative_depth_metrics(pred_u8, targ_u8, *, floor: float = 0.05) -> dict:
    """v2's MiDaS-aligned relative depth, plus a SIGN-CONSTRAINED variant (`delta1_aligned_ssc`).

    v2's least-squares scale+shift is unconstrained, so a prediction with the depth convention inverted
    (near dark instead of near bright) is mapped onto the target by a NEGATIVE scale and scores a perfect
    delta1_aligned = 1.0. Constraining s >= 0 makes the metric sensitive to the convention the demos specify,
    which is the quantity this project is actually measuring.
    """
    out = dict(M2.relative_depth_metrics(pred_u8, targ_u8, floor=floor))
    ps, _ = M1.decode_scalar(pred_u8); ts, tv = M1.decode_scalar(targ_u8)
    if not tv.any():
        out["delta1_aligned_ssc"] = float("nan"); out["abs_rel_aligned_ssc"] = float("nan")
        return out
    x = ps[tv].astype(np.float64); y = ts[tv].astype(np.float64)
    n = x.size; sx, sy, sxx, sxy = x.sum(), y.sum(), (x * x).sum(), (x * y).sum()
    den = n * sxx - sx * sx
    if abs(den) < M2._EPS:
        s, t = 0.0, float(y.mean())
    else:
        s = (n * sxy - sx * sy) / den; t = (sy - s * sx) / n
    if s < 0:                                   # the correction: no sign flip allowed
        s, t = 0.0, float(y.mean())
    xa = s * x + t
    keep = y >= floor
    if keep.any():
        pa = np.clip(xa[keep], floor, None); g = y[keep]
        out["delta1_aligned_ssc"] = float(np.mean(np.maximum(pa / g, g / pa) < 1.25))
        out["abs_rel_aligned_ssc"] = float(np.mean(np.abs(pa - g) / g))
    else:
        out["delta1_aligned_ssc"] = float("nan"); out["abs_rel_aligned_ssc"] = float("nan")
    out["align_scale_clamped"] = float(s)
    return out


# ---------------------------------------------------------------- round-2 corrections (2026-09-12, gap-closure pass)
FACE_MISS_POLICY = "matched"     # 'matched' scores only landmarks the prediction actually produced
KP_REJECT_FRAC = 0.47            # rejection radius as a fraction of the TARGET VOCABULARY's own minimum separation
KP_REJECT_RADIUS = 40.0          # legacy fixed radius, kept so the first v3 numbers reproduce; see below

# WHY THE RADIUS IS NOW DERIVED RATHER THAN FIXED (2026-09-12, second review).
# The original constant carried the comment "under half the 85-unit minimum separation of the joint vocabulary".
# Measured against DISTINCT_PALETTE, which is what the v4c pack actually uses, the minimum separation is 85.00 for
# the 16-joint (human_pose_2d) and 20-joint (object_keypoints) prefixes but 24.04 for the 68-landmark
# (face_landmarks) prefix -- on 64 of 64 records. So the fixed 40 was 0.47x the separation on two tasks and
# 1.66x on the third, i.e. LARGER than the distance at which face landmark names become ambiguous (~12).
# The comment also cited draw_discs / LABEL_PALETTE, which is not this pack's coding: under LABEL_PALETTE the
# prefixes separate by 35.0 and 8.0, where a fixed 40 would be indefensible on every task.
# Deriving the radius from the target's own vocabulary reproduces 40 where 40 was right and ~11 where it was not.


def _ycbcr_from_rgb(a01):
    r, g, b = a01[..., 0], a01[..., 1], a01[..., 2]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    return y, 0.564 * (b - y), 0.713 * (r - y)


def _rgb_from_ycbcr(y, cb, cr):
    return np.stack([y + 1.403 * cr, y - 0.344 * cb - 0.714 * cr, y + 1.773 * cb], -1)


_LPIPS = {"net": None, "tried": False}
LPIPS_PKG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "third_party", "lpips_pkg")
LPIPS_TORCH_HOME = os.path.join(os.path.dirname(LPIPS_PKG_DIR), "torch_home")


def _lpips_net():
    """The OFFICIAL `lpips` package (Zhang et al. 2018, v0.1, AlexNet trunk), installed repo-locally under
    third_party/ so the shared conda env is untouched; the AlexNet backbone lives in third_party/torch_home.
    Lazy and optional: on a host without torch (the TPU eval jobs) `lpips_alex` is reported as NaN and nothing
    else changes. Returns None when unavailable."""
    if _LPIPS["tried"]:
        return _LPIPS["net"]
    _LPIPS["tried"] = True
    try:
        os.environ.setdefault("TORCH_HOME", LPIPS_TORCH_HOME)
        if LPIPS_PKG_DIR not in sys.path:
            sys.path.insert(0, LPIPS_PKG_DIR)
        import torch, lpips  # noqa: F401
        net = lpips.LPIPS(net="alex", verbose=False)
        net.eval()
        _LPIPS["net"] = net
    except Exception:  # noqa: BLE001 - perceptual metric is an add-on, never a hard dependency of the scorer
        _LPIPS["net"] = None
    return _LPIPS["net"]


LPIPS_MIN_SIDE = 32   # AlexNet's 11x11/4 conv + pools need this much; the pack is 256x256, tests use tiny images


def lpips_alex(pred_u8, targ_u8) -> float:
    """LPIPS distance (lower = more perceptually similar), inputs uint8 RGB HWC.
    NaN if the package is absent or the image is smaller than the AlexNet trunk can consume."""
    net = _lpips_net()
    if net is None or min(np.asarray(pred_u8).shape[:2]) < LPIPS_MIN_SIDE:
        return float("nan")
    try:
        import torch
        to = lambda a: torch.from_numpy(np.ascontiguousarray(np.transpose(np.asarray(a, np.float32) / 127.5 - 1.0, (2, 0, 1)))[None])
        with torch.no_grad():
            return float(net(to(pred_u8), to(targ_u8)))
    except Exception:  # noqa: BLE001 - an add-on metric must never take the scorer down
        return float("nan")


def appearance_metrics(pred_u8, targ_u8, *, task: str) -> dict:
    """v2's appearance metrics plus `psnr_chroma` for colorization and `lpips_alex` for all five.

    LPIPS is here because PSNR/SSIM are DISTORTION metrics and reward the blurry conditional mean: on the
    2026-09-13 floor study the baseline's super-resolution output restores 75-78 % of the target's
    high-frequency power and is closer to target sharpness on 63/64 records, yet loses PSNR and SSIM to plain
    bicubic. A perceptual metric is needed to report what the eye sees; it is reported ALONGSIDE psnr.

    Colorization models are GIVEN the luminance, so raw-RGB PSNR scores the model partly on an input it was
    handed. The literature composites the PREDICTED chroma onto the GROUND-TRUTH luma. Measured in the round-2
    pass: this flips one pre-registered Holm decision — `pairxyd-loo`'s colorization correspondence gain
    (+2.313 dB raw RGB) drops to +0.737 dB and stops being Holm-significant.
    """
    out = dict(M2.appearance_metrics(pred_u8, targ_u8, task=task))
    if task == "colorization":
        p01 = np.asarray(pred_u8, np.float64) / 255.0
        t01 = np.asarray(targ_u8, np.float64) / 255.0
        # Formulated as target + the CHROMA DIFFERENCE, not as a YCbCr round trip: the transform is linear, so
        # this is algebraically the chroma-substituted image AND exactly t01 when pred == targ. Composing
        # rgb_from_ycbcr(target_luma, pred_chroma) directly leaves ~1e-16 round-trip error, which makes a
        # pixel-perfect prediction score ~320 dB instead of inf and breaks the self-score contract.
        _, tcb, tcr = _ycbcr_from_rgb(t01)
        _, pcb, pcr = _ycbcr_from_rgb(p01)
        comp = np.clip(t01 + _rgb_from_ycbcr(0.0, pcb - tcb, pcr - tcr), 0.0, 1.0)
        mse = float(np.mean((comp - t01) ** 2))
        out["psnr_chroma"] = float("inf") if mse <= 0 else float(10.0 * np.log10(1.0 / mse))
    out["lpips_alex"] = lpips_alex(pred_u8, targ_u8)
    return out


def keypoint_metrics(pred_u8, targ_u8, *, task: str, scored_positions=None) -> dict:
    """v2's keypoint metrics, with two round-2 corrections.

    (1) REJECTION RADIUS. `metrics.keypoint_pck` assigns EVERY prediction foreground pixel to the nearest colour
    present in the target with no threshold, so a disc drawn for a joint the target does not annotate is absorbed
    into whichever annotated joint is nearest in RGB and drags its centroid. The joint index is globally decodable
    (draw_discs + DISTINCT_PALETTE), so the information to reject exists. We keep a predicted pixel only if it is
    within `KP_REJECT_RADIUS` of the colour it is assigned to.

    (2) FACE COVERAGE. A landmark missing from the prediction was charged the IMAGE DIAGONAL (362 px, 1.7-2.5x the
    record's own bbox side), mixing localisation with coverage on an arbitrary scale. We report `nme_bbox_matched`
    (localisation over the landmarks the prediction produced) and `kp_coverage` separately. Measured: this swaps
    the top two arms on face_landmarks (t16 0.3398/0.3217 vs t15 0.3431/0.3190) and shows `bridge-a1`'s entire
    P1->P4 face "gain" is coverage (0.8858->0.9221), with localisation flat (0.3721->0.3676).
    """
    out = dict(M2.keypoint_metrics(pred_u8, targ_u8, task=task))
    # v1's LUMINANCE floor, restored. The first v3 draft used sum(channels) > 0, which silently admitted
    # near-black decoder haze that v1 excluded -- an undocumented behaviour change, not part of the fix.
    tfg = np.asarray(targ_u8).max(axis=-1) > 24
    pfg = np.asarray(pred_u8).max(axis=-1) > 24
    cols = M1._unique_colours(targ_u8, tfg)
    if cols.size == 0:
        out["pck_bbox_reject"] = float("nan"); out["kp_coverage"] = 0.0
        out["nme_bbox_matched"] = float("nan")
        return out
    c64 = np.asarray(cols, np.int64)
    if len(c64) > 1:
        dd = ((c64[:, None, :] - c64[None, :, :]) ** 2).sum(-1).astype(np.float64)
        np.fill_diagonal(dd, np.inf)
        radius = KP_REJECT_FRAC * float(np.sqrt(dd.min()))
    else:
        radius = float("inf")
    out["kp_reject_radius"] = radius

    def centroids(img, fg, reject):
        a = np.asarray(img, np.int64)
        cen = {}
        if not fg.any():
            return cen
        px = a[fg].reshape(-1, 3)
        d = ((px[:, None, :] - c64[None, :, :]) ** 2).sum(-1)
        j = d.argmin(1)
        keep = np.sqrt(d[np.arange(d.shape[0]), j]) <= radius if reject else np.ones(j.shape, bool)
        ys, xs = np.nonzero(fg)
        for k in range(len(c64)):
            m = (j == k) & keep
            if m.any():
                cen[k] = (float(ys[m].mean()), float(xs[m].mean()))
        return cen

    tc = centroids(targ_u8, tfg, False)
    pc = centroids(pred_u8, pfg, True)
    if scored_positions is not None:                              # D1-B: score only joints some demo carried
        cpos = M2._palette_positions(np.asarray(cols, np.uint8)[None])[0]
        keep = set(int(x) for x in scored_positions)
        tc = {k: v for k, v in tc.items() if int(cpos[k]) in keep}
        pc = {k: v for k, v in pc.items() if int(cpos[k]) in keep}
    if not tc:
        out["pck_bbox_reject"] = float("nan"); out["kp_coverage"] = 0.0
        out["nme_bbox_matched"] = float("nan")
        return out
    ys = [v[0] for v in tc.values()]; xs = [v[1] for v in tc.values()]
    side = max(max(ys) - min(ys), max(xs) - min(xs))
    norm = side if side > 1.0 else float(np.hypot(*np.asarray(targ_u8).shape[:2]))
    errs, hits = [], 0
    for k, (y, x) in tc.items():
        if k in pc:
            errs.append(float(np.hypot(y - pc[k][0], x - pc[k][1]))); hits += 1
    out["kp_coverage"] = float(hits / len(tc))
    if errs:
        e = np.asarray(errs)
        out["pck_bbox_reject"] = float(np.mean(e <= 0.1 * norm))
        out["nme_bbox_matched"] = float(np.mean(e) / norm)
    else:
        out["pck_bbox_reject"] = 0.0
        out["nme_bbox_matched"] = float("nan")

    # FACTORISATION (2026-09-12). The fused number is a NAMING metric wearing a localisation metric's name:
    # colour-blind localisation measures 0.67 / 0.76 / 0.63 on the baseline where the fused score reads
    # 0.16 / 0.34 / 0.10, so 69-87% of the attainable score is lost at the naming step -- and a perfectly
    # localised, RANDOMLY renamed prediction scores 0.158 on human_pose_2d against the real model's 0.161,
    # i.e. the published pose number sits at the chance-naming floor. Report the two factors, never just
    # the product. `pck_blind` matches blobs to joints by position only (greedy nearest, one-to-one).
    tk = sorted(tc); pk = sorted(pc)
    if tk and pk:
        pairs = sorted((float(np.hypot(tc[a][0] - pc[b][0], tc[a][1] - pc[b][1])), a, b)
                       for a in tk for b in pk)
        ut, up, blind_hits, named = set(), set(), 0, 0
        for dist, a, b in pairs:
            if a in ut or b in up:
                continue
            ut.add(a); up.add(b)
            if dist <= 0.1 * norm:
                blind_hits += 1
                if a == b:
                    named += 1
        out["pck_blind"] = float(blind_hits / len(tc))
        out["naming_acc"] = float(named / blind_hits) if blind_hits else float("nan")
        out["naming_chance"] = float(1.0 / len(tc))
    else:
        out["pck_blind"] = 0.0 if tk else float("nan")
        out["naming_acc"] = float("nan"); out["naming_chance"] = float("nan")
    return out


# ---------------------------------------------------------------- dispatch + aggregation
def score_pair_v3(task: str, pred_u8: np.ndarray, targ_u8: np.ndarray, *, source: str | None = None,
                  scored_positions=None, perceptual: bool = False) -> dict:
    """Same contract as `metrics_v2.score_pair_v2`, with the audit's corrections applied."""
    pred_u8 = np.asarray(pred_u8, np.uint8); targ_u8 = np.asarray(targ_u8, np.uint8)
    if task not in PRIMARY_V3:
        raise KeyError(f"score_pair_v3: no official metric registered for task {task!r}")
    if task == "metric_depth":
        out = metric_depth_metrics(pred_u8, targ_u8, source=source)
    elif task == "relative_depth":
        out = relative_depth_metrics(pred_u8, targ_u8)
    elif task == "semantic_segmentation":
        out = semantic_metrics_large(pred_u8, targ_u8, source=source, scored_positions=scored_positions)
    elif task in ("instance_segmentation", "panoptic_segmentation"):
        out = partition_metrics(pred_u8, targ_u8)
    elif task == "saliency_detection":
        out = saliency_metrics(pred_u8, targ_u8)
    elif task in ("human_pose_2d", "object_keypoints", "face_landmarks"):
        out = keypoint_metrics(pred_u8, targ_u8, task=task, scored_positions=scored_positions)
    elif task in ("colorization", "denoising", "deblurring", "inpainting", "super_resolution") or task in PROCEDURAL_TASKS_V1:
        out = appearance_metrics(pred_u8, targ_u8, task=task)
    else:
        out = M2.score_pair_v2(task, pred_u8, targ_u8, source=source)
        out.pop("primary_metric", None); out.pop("primary_lower_is_better", None); out.pop("primary_value", None)
    name, lower = primary_v3(task, perceptual)
    out["_source"] = str(source) if source is not None else "_"
    out.setdefault("l1_01", float(np.mean(np.abs(pred_u8.astype(np.float64) - targ_u8.astype(np.float64)) / 255.0)))
    out["primary_metric"] = name; out["primary_lower_is_better"] = lower
    out["primary_value"] = float(out.get(name, float("nan")))
    return out


def _annotate_counts(agg: dict, per_record: list) -> dict:
    """Add n_nan / n_inf / mean_capped to every numeric key of an aggregate. NaN and +inf must be COUNTED, not
    silently dropped: a differing n across arms un-pairs a comparison, and a perfect PSNR cell used to
    aggregate to n=0."""
    keys = sorted({k for r in per_record for k, v in r.items() if not k.startswith("_") and isinstance(v, (int, float))})
    for k in keys:
        if k not in agg:
            continue
        vals = np.asarray([r.get(k, float("nan")) for r in per_record], np.float64)
        n_nan = int(np.isnan(vals).sum())
        if n_nan:
            agg[k]["n_nan"] = n_nan
        n_inf = int(np.isposinf(vals).sum())
        if n_inf:
            capped = np.where(np.isposinf(vals), PSNR_CAP_DB, vals)
            capped = capped[np.isfinite(capped)]
            agg[k]["n_inf"] = n_inf
            agg[k]["mean_capped"] = float(capped.mean()) if capped.size else float("nan")
    return agg


def aggregate(task: str, per_record: list) -> dict:
    """v2's aggregation, plus: perfect (+inf) PSNR records counted rather than dropped, NaN counted, and a
    PER-SOURCE breakdown under `by_source` whenever a cell mixes sources.

    The per-source split exists because a pooled cell can hide opposite signs: denoising on the baseline is
    +3.19 dB above copy-the-input on div2k and -2.97 dB below it on sidd, and the pooled mean reports neither.
    """
    out = _annotate_counts(M2.aggregate(task, per_record), per_record)
    srcs = sorted({r.get("_source", "_") for r in per_record})
    if len(srcs) > 1:
        out["by_source"] = {}
        for s_ in srcs:
            sub = [r for r in per_record if r.get("_source", "_") == s_]
            agg = _annotate_counts(M2.aggregate(task, sub), sub)
            out["by_source"][s_] = {k: v for k, v in agg.items() if isinstance(v, dict) and "mean" in v}
            out["by_source"][s_]["n_records"] = len(sub)
    return out
