#!/usr/bin/env python3
"""Task-appropriate metrics for all 20 tasks, by INVERTING the target encoding.

CORRECTION 2026-09-22: nearest-query-vocabulary segmentation metrics do NOT certify
raw RGB rendering. Six real-pack, target-assisted washed counterexamples retain
perfect mIoU/PQ while 99.85-100% of pixels lie more than 32 RGB units from the
target vocabulary. See tools/audit_metric_rendering_gap_v1.py and the E26 ledger.
The historical statement below that outside-vocabulary colours are always
penalised is too strong: colours within the same Voronoi cell decode identically.
Colour permutations are free for class-agnostic PQ, not generally semantic mIoU.
Scorer semantics are unchanged; report raw renders and RGB diagnostics alongside
structural metrics. The synthetic audit uses GT and is not a deployable method.

WHY L1 ALONE IS NOT ENOUGH
--------------------------
Every task's target is an RGB *rendering* of something else — a depth map, a flow field, a label
partition, a set of keypoints. Mean L1 on those renderings answers "how similar do the pictures
look", which is only the question we care about for the five appearance tasks. For the rest it
conflates things that should be separated and is blind to things that matter:

  * Depth/disparity/saliency renders encode INVALID pixels as pure black. L1 over the whole frame
    partly measures "did it paint black in the right places", and on sources like DIODE the invalid
    region is large. Those pixels must be masked out, not averaged in.
  * Surface normals: L1 on the RGB is not angular error. A normal off by 90 degrees in-plane can
    have a smaller RGB L1 than one off by 20 degrees out-of-plane.
  * Optical flow: hue is DIRECTION and value is SPEED. L1 mixes a direction error and a magnitude
    error into one number from which neither is recoverable.
  * Segmentation: the palette index is arbitrary. Two perfect segmentations with permuted colours
    have large L1 and identical quality. This is the case Stage 46 flagged — all three
    segmentation tasks sat at 0.35-0.42 for EVERY arm, which is what a saturated metric looks like.

HOW THIS IS MADE CORRECT RATHER THAN PLAUSIBLE
----------------------------------------------
Each decoder inverts a specific encoder in `jflux2/dataloaders/vgi_rgb/`, and
`tests/test_fasteval_metrics.py` verifies it by ROUND TRIP against the real encoder:
encode(known) -> decode -> compare. That is the strongest cheap check available, and it is why
these decoders are not guesses. Two encodings make it possible at all:

  * `_common.flow_to_rgb` is deliberately invertible — "Middlebury colour wheel with a *fixed*
    maximum, so it is invertible", `_FLOW_MAX = 64.0` px global. So EPE in real pixels is
    recoverable, not just a direction.
  * `_common.colorize_labels` uses a fixed 1024-entry `LABEL_PALETTE`, so a partition is
    recoverable by nearest-palette lookup.

WHAT REMAINS A PROXY, STATED NOT HIDDEN
---------------------------------------
  * Depth / disparity / scene-flow renders are normalised PER IMAGE (2nd-98th percentile for
    depth, 98th for disparity) and the scale is NOT stored. So absolute metric depth is
    unrecoverable BY CONSTRUCTION. Metrics are computed in the normalised space the target itself
    defines — correct for comparing predictions against that target, but they are not AbsRel in
    metres and must not be reported as such.
  * Keypoint tasks render discs/skeletons. Joint identity and ordering are not recoverable from a
    raster, so true PCK is impossible here. What is computed is a structure-overlap proxy, and it
    is named `*_proxy` so it cannot be mistaken for PCK.

Nothing here replaces `l1_01`. It is kept alongside, because Stage 46 and everything before it is
expressed in it, and dropping it would break continuity for a metric change.
"""

from __future__ import annotations

import numpy as np

FLOW_MAX_PX = 64.0          # must match _common._FLOW_MAX
_EPS = 1e-8

# Lower-is-better metric BASE names. Matching strips a trailing "_proxy", because the keypoint
# family renames its keys and `mae_proxy` was consequently reported as higher-is-better -- a sign
# error in the schema, not in the number.
LOWER_IS_BETTER = {"mae", "rmse", "l1_01", "l1_pm1", "mean_deg", "median_deg", "epe_px",
                   "dir_deg", "bad_3px", "chamfer_px", "pck_px_err"}

# NOT performance numbers. Counts, selected hyperparameters and coverage fractions must be labelled
# so a reader (or a script sorting a column) cannot treat them as quality. `n_*_segs` are
# diagnostics -- they are how the AE-reference fragmentation bug was caught -- and
# `f1_best_thresh` is the argmax of a sweep, not a score.
DIAGNOSTIC = {"n_pred_segs", "n_target_segs", "valid_frac", "n_joints", "n_matched_joints",
              "pred_fg_frac", "mean_saturation",
              "n_classes_in_target", "family", "scored_vs", "primary_metric",
              "primary_lower_is_better", "primary_value"}

# Suffix/prefix rules, because an explicit set kept missing new keys. The review caught
# `f1_best_thresh`; a later pass then found `boundary_f_thresh`, `px_f1_best_thresh` and
# `primary_value` leaking through the same gap. A pattern rule closes the class, not one instance.
_DIAGNOSTIC_SUFFIXES = ("_thresh",)
_DIAGNOSTIC_PREFIXES = ("n_", "primary_")

# DECORATORS that qualify a metric without changing WHAT it measures, so they must be stripped
# before any schema lookup. Getting this wrong has now bitten twice in the same way:
#   * `f1_best_thresh_proxy` ends in `_proxy`, not `_thresh`, so the diagnostic suffix rule missed
#     it and a swept argmax threshold was reported as a score for all three keypoint tasks.
#   * `px_mae` is not the literal string `mae`, so LOWER_IS_BETTER's exact-set lookup missed it and
#     an ERROR metric was labelled higher-is-better -- the identical bug the review flagged for
#     `mae_proxy`, which I had fixed by NAME rather than by rule, so the variant slipped through.
# Both were caught by the arm-exceeds-ceiling guard in tools/make_tokensweep_csv.py, not by reading.
_DECORATOR_SUFFIXES = ("_proxy",)
_DECORATOR_PREFIXES = ("px_",)


def _schema_base(name: str) -> str:
    """Strip decorators so schema lookups see the metric being measured, not its variant label."""
    changed = True
    while changed:                       # loop: `px_f1_best_thresh_proxy` needs both ends removed
        changed = False
        for s in _DECORATOR_SUFFIXES:
            if name.endswith(s) and len(name) > len(s):
                name, changed = name[: -len(s)], True
        for p in _DECORATOR_PREFIXES:
            if name.startswith(p) and len(name) > len(p):
                name, changed = name[len(p):], True
    return name


def metric_kind(name: str) -> str:
    """-> 'diagnostic' or 'metric'.

    Selected thresholds, counts and echoed bookkeeping are NOT performance numbers; labelling them
    as such invites a reader (or a script sorting a column) to treat a swept argmax as a score.
    """
    for n in (name, _schema_base(name)):
        if (n in DIAGNOSTIC or n.endswith(_DIAGNOSTIC_SUFFIXES)
                or n.startswith(_DIAGNOSTIC_PREFIXES)):
            return "diagnostic"
    return "metric"


def lower_is_better(name: str) -> bool:
    return _schema_base(name) in LOWER_IS_BETTER

# ---------------------------------------------------------------- decoders
def is_grayscale(rgb: np.ndarray, tol: int = 2) -> bool:
    """True if R==G==B everywhere (within uint8 tolerance).

    Used to pick the VARIANT within a task family: `edge_detection` and `scene_coordinates` render
    differently depending on the source dataset (a grayscale mask vs a colorised label map, an
    inverse-depth ramp vs an xyz RGB), so the encoding is detected from the data rather than
    assumed from the task name.
    """
    a = np.asarray(rgb, dtype=np.int16)
    return bool(np.abs(a[..., 0] - a[..., 1]).max() <= tol
                and np.abs(a[..., 1] - a[..., 2]).max() <= tol)


def decode_scalar(rgb_u8: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Grayscale render -> (scalar in [0,1], valid mask). Inverts `_common.grayscale_to_rgb`.

    Invalid pixels are encoded as pure black by `_inverse_depth_rgb` / `_signed_scalar_rgb`, so
    black is treated as invalid. That does conflate "invalid" with "a genuine zero", which is a
    property of the ENCODING and not of this decoder — and it is exactly why these pixels must be
    excluded rather than averaged into an L1.
    """
    g = np.asarray(rgb_u8, dtype=np.float32).mean(axis=-1) / 255.0
    return g, g > 0.0


def decode_normals(rgb_u8: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """RGB -> unit normals. Inverts `build_normals`: rgb = (clip(n,-1,1)+1)*0.5*255.

    Invalid is rgb=(0,0,0), which decodes to (-1,-1,-1) with norm sqrt(3); genuine normals are
    unit, so validity is a norm test rather than a colour test.
    """
    # float64 (2026-09-12 fix): in float32 the renormalised vector's self-dot lands 3e-7 from 1, and
    # arccos has an infinite derivative there, so GT-vs-GT scored mean_deg 0.0062 instead of 0. The
    # reported number is unchanged (40.5043 either way); this only removes the self-score artifact.
    n = np.asarray(rgb_u8, dtype=np.float64) / 255.0 * 2.0 - 1.0
    norm = np.linalg.norm(n, axis=-1)
    valid = np.abs(norm - 1.0) < 0.35
    unit = n / np.maximum(norm, _EPS)[..., None]
    return unit, valid


def decode_flow(rgb_u8: np.ndarray, max_magnitude: float = FLOW_MAX_PX):
    """Middlebury wheel -> (u, v) in PIXELS, valid mask. Inverts `_common.flow_to_rgb`.

    The forward transform is HSV at full saturation, so it inverts exactly on a CLEAN wheel image.

    BUT THE INPUT IS NEVER CLEAN. Both the AE round trip and a generated prediction perturb
    saturation, so an inversion that REQUIRES `min(rgb) == 0` breaks on precisely the data it is
    meant to score. An earlier version tested `rn >= 1 - 1e-6` and `bn <= 1e-6` to pick the hue
    sextant; on any real image none of those six conditions match, hue silently falls through to
    its initial 0, and every pixel decodes as pointing in one fixed direction. That would report a
    large EPE for a perfectly good flow field and would be indistinguishable from model failure.

    So this uses the STANDARD RGB->HSV hue formula, which is exact for s == 1 and degrades
    gracefully below it, and it reports mean saturation as a diagnostic so a caller can see how far
    from the clean wheel the input actually was.
    """
    a = np.asarray(rgb_u8, dtype=np.float64) / 255.0
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    mx = a.max(axis=-1)
    mn = a.min(axis=-1)
    d = mx - mn
    valid = mx > (1.0 / 255.0)               # black == invalid/occluded per the encoder
    dz = np.maximum(d, _EPS)

    # standard hue, in turns; robust to s < 1 rather than requiring s == 1 exactly
    hue = np.where(mx == r, ((g - b) / dz) % 6.0,
          np.where(mx == g, (b - r) / dz + 2.0,
                            (r - g) / dz + 4.0)) / 6.0
    hue = np.where(d <= _EPS, 0.0, hue)      # achromatic: direction undefined

    angle = (hue * 2.0 - 1.0) * np.pi
    # value carries speed. Using mx (not d) matches the encoder, which sets value = |flow|/max.
    mag = mx * max_magnitude
    u = -mag * np.cos(angle)
    v = -mag * np.sin(angle)
    return u.astype(np.float32), v.astype(np.float32), valid


def _unique_colours(img_u8: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    """Distinct RGB rows, via a PACKED int32 key.

    `np.unique(..., axis=0)` on a 65536x3 array measured **92 ms** -- it lexsorts rows -- and it was
    the single largest cost in the whole metric suite, called twice per segmentation record and once
    per keypoint record. Packing r<<16 | g<<8 | b into one int32 and taking `np.unique` on the 1-D
    result is the identical operation on uint8 data (the packing is a bijection for 0..255 channels)
    and measures **0.8 ms**: 115x faster.
    """
    a = np.asarray(img_u8, dtype=np.int32)
    if mask is not None:
        a = a[mask]
    a = a.reshape(-1, 3)
    packed = (a[:, 0] << 16) | (a[:, 1] << 8) | a[:, 2]
    keys = np.unique(packed)
    return np.stack([(keys >> 16) & 255, (keys >> 8) & 255, keys & 255], axis=1).astype(np.int16)


def decode_to_target_vocabulary(img_u8: np.ndarray, target_u8: np.ndarray,
                                cols: np.ndarray | None = None) -> np.ndarray:
    """Assign each pixel to the nearest colour PRESENT IN THE TARGET. Use this for PQ / mIoU.

    WHY NOT the full 1024-entry palette. The prediction is decoded through the AE, which turns a
    7-colour label map into a smooth image -- MEASURED: 11,163 distinct colours out of a 7-colour
    target. Nearest-of-1024 then scatters those into hundreds of spurious labels, and PQ collapses
    for ANY model. The ceiling proved it: the AE round trip of the target itself scored PQ 0.0043,
    BELOW the arms' 0.0054, and a measured value above a perfect reference means the metric is
    reading noise. Median filtering did not rescue it (15x15 reached only PQ 0.0091 with 205
    segments against 7).

    Restricting to the target's own colours lifts the same ceiling to PQ 0.36-0.85 / mIoU
    0.52-0.85. The diagnostic that pointed here was SQ ~ 0.98 with RQ ~ 0.004: the matched regions
    were already near-perfect, so the problem was spurious segments, not bad boundaries.

    WHAT THIS DOES AND DOES NOT ASSUME. It gives the prediction the target's LABEL VOCABULARY (the
    set of colours, not their spatial layout). That is standard for scoring a generated
    segmentation raster -- some finite label set is required to quantize into, and the target's set
    is the least arbitrary available. So these metrics measure REGION STRUCTURE GIVEN THE
    VOCABULARY.

    CORRECTION (2026-09-12 audit): the sentence that used to stand here — "they do not measure whether the
    model chose the right absolute colour" — is FALSE as stated. Measured counterexample: a perfect partition
    relabelled to unused palette indices scores pq 0.6351 / pq_novoid 0.6339, 0 of 64 records perfect. Colour
    choice is penalised whenever the prediction's colours fall outside the target's set; only permutations
    WITHIN that set are free. The absolute id is still unlearnable in principle, which is why this is recorded
    as a convention deviation rather than repaired.

    2026-09-22 clarification: "whenever" in that historical correction is too
    strong. Some wrong colours change assignments and are penalised; arbitrarily
    off-palette colours inside the same nearest-colour cell are not. A perfect
    structural score is therefore not proof of faithful RGB rendering. See the
    module-level correction and the target-assisted rendering-gap audit.
    """
    if cols is None:
        cols = _unique_colours(target_u8)
    # int64 arithmetic (2026-09-08 fix): the differences were squared in int16, and 255^2 = 65025 wraps to -511, so a
    # pixel could be assigned to a vocabulary colour it is FAR from (green -> blue). Both sides of a comparison went
    # through the same wrong map, which is why GT-vs-GT still scored 1.0 and hid it; on the v4c validation pack the
    # GT decoded through its own vocabulary differed from itself on 63/64 semantic and 64/64 instance/panoptic records.
    flat = np.asarray(img_u8, dtype=np.int64).reshape(-1, 3)
    cols64 = np.asarray(cols, dtype=np.int64)
    out = np.empty(flat.shape[0], dtype=np.int32)
    step = 16384
    for i in range(0, flat.shape[0], step):
        d = ((flat[i:i + step, None, :] - cols64[None, :, :]) ** 2).sum(-1)
        out[i:i + step] = d.argmin(1)
    return out.reshape(np.asarray(img_u8).shape[:2])


def decode_labels(rgb_u8: np.ndarray, palette: np.ndarray) -> np.ndarray:
    """Colorised label map -> a CANONICAL palette index per pixel, by nearest palette entry.

    NOT the original label, and it cannot be. `colorize_labels` is NOT INJECTIVE: 124 of the 1024
    palette entries are duplicate colours (each shared by exactly 2 indices), so two distinct
    labels can render to the same RGB. On top of that the encoder applies `% 1024` to ids that are
    packed panoptic values (r + 256g + 65536b), which aliases further. Both losses are in the
    ENCODER, i.e. in the target the model was trained to reproduce -- they are not introduced here.

    Measured, not assumed: an earlier version of the round-trip test asserted exact label recovery
    and failed on 32% of pixels, which is what surfaced the aliasing.

    Why this is still the right basis for the segmentation metric: prediction and target go through
    the SAME decode, so class_agnostic_pq compares like with like and answers "did it reproduce the
    rendered partition". What aliasing costs is that the rendered target may itself merge two truly
    distinct segments -- a ceiling on the data, which `n_target_segs` exposes.

    Nearest rather than exact-match because a generated prediction will not land on palette colours
    exactly; the target will.
    """
    flat = np.asarray(rgb_u8, dtype=np.int64).reshape(-1, 3)      # int64: int16 squares overflow (see decode_to_target_vocabulary)
    pal = np.asarray(palette, dtype=np.int64)
    # chunked to keep the (pixels x 1024) distance matrix bounded
    out = np.empty(flat.shape[0], dtype=np.int32)
    step = 8192
    for i in range(0, flat.shape[0], step):
        d = ((flat[i:i + step, None, :] - pal[None, :, :]) ** 2).sum(-1)
        out[i:i + step] = d.argmin(1)
    return out.reshape(rgb_u8.shape[:2])


# ---------------------------------------------------------------- metrics
def psnr(pred01: np.ndarray, targ01: np.ndarray) -> float:
    mse = float(np.mean((pred01 - targ01) ** 2))
    return float("inf") if mse <= 0 else float(10.0 * np.log10(1.0 / mse))


def ssim(pred01: np.ndarray, targ01: np.ndarray, win: int = 7) -> float:
    """Uniform-window SSIM on the mean channel. Deliberately the uniform (not Gaussian) variant —
    stated because SSIM values are only comparable under the same window definition."""
    x = pred01.mean(-1) if pred01.ndim == 3 else pred01
    y = targ01.mean(-1) if targ01.ndim == 3 else targ01
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    # separable box filter: O(pixels) per call rather than materialising a
    # (H-w+1, W-w+1, w, w) view and reducing it, which measured ~109 ms/record for the five
    # appearance tasks. Identical values -- a uniform filter IS the mean over the same window.
    from scipy.ndimage import uniform_filter

    def f(a):
        return uniform_filter(a.astype(np.float64), size=win, mode="reflect")

    mx, my = f(x), f(y)
    mxx, myy, mxy = f(x * x), f(y * y), f(x * y)
    vx, vy, vxy = mxx - mx * mx, myy - my * my, mxy - mx * my
    s = ((2 * mx * my + c1) * (2 * vxy + c2)) / ((mx * mx + my * my + c1) * (vx + vy + c2))
    return float(np.mean(s))


def masked_scalar_metrics(pred01, targ01, valid) -> dict:
    """MAE / RMSE / delta<1.25 over VALID target pixels only, in the target's normalised space."""
    if not valid.any():
        return {"mae": float("nan"), "rmse": float("nan"), "delta1": float("nan"),
                "valid_frac": 0.0}
    p, t = pred01[valid], targ01[valid]
    ratio = np.maximum(p / np.maximum(t, _EPS), t / np.maximum(p, _EPS))
    return {"mae": float(np.mean(np.abs(p - t))),
            "rmse": float(np.sqrt(np.mean((p - t) ** 2))),
            "delta1": float(np.mean(ratio < 1.25)),
            "valid_frac": float(valid.mean())}


def binary_mask_metrics(pred01, targ01, thresh: float = 0.5) -> dict:
    """F1 at a fixed threshold plus MAE, and best-F1 over a threshold sweep.

    Best-F1 is reported because a generated image's absolute intensity calibration is not the
    property under test; fixed-threshold F1 is kept so the two cannot be confused.
    """
    t = targ01 > thresh
    out = {"mae": float(np.mean(np.abs(pred01 - targ01)))}
    best, best_th = 0.0, thresh
    for th in np.linspace(0.05, 0.95, 19):
        p = pred01 > th
        tp = float((p & t).sum()); fp = float((p & ~t).sum()); fn = float((~p & t).sum())
        f1 = 0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn)
        if th == thresh or abs(th - thresh) < 1e-9:
            out["f1_at_0.5"] = f1
        if f1 > best:
            best, best_th = f1, float(th)
    out["f1_best"] = best
    out["f1_best_thresh"] = best_th
    out.setdefault("f1_at_0.5", 0.0)
    return out


def angular_error_deg(pred_unit, targ_unit, valid) -> dict:
    if not valid.any():
        return {"mean_deg": float("nan"), "median_deg": float("nan"), "within_11deg": float("nan"),
                "valid_frac": 0.0}
    dot = np.clip((pred_unit * targ_unit).sum(-1)[valid], -1.0, 1.0)
    ang = np.degrees(np.arccos(dot))
    return {"mean_deg": float(np.mean(ang)), "median_deg": float(np.median(ang)),
            "within_11deg": float(np.mean(ang < 11.25)), "valid_frac": float(valid.mean())}


def flow_metrics(pu, pv, tu, tv, valid) -> dict:
    """EPE in pixels, plus direction error separated from magnitude — the thing L1 cannot give."""
    if not valid.any():
        return {"epe_px": float("nan"), "bad_3px": float("nan"), "dir_deg": float("nan"),
                "valid_frac": 0.0}
    du, dv = pu[valid] - tu[valid], pv[valid] - tv[valid]
    epe = np.sqrt(du * du + dv * dv)
    pm = np.sqrt(pu[valid] ** 2 + pv[valid] ** 2)
    tm = np.sqrt(tu[valid] ** 2 + tv[valid] ** 2)
    cos = (pu[valid] * tu[valid] + pv[valid] * tv[valid]) / np.maximum(pm * tm, _EPS)
    return {"epe_px": float(np.mean(epe)), "bad_3px": float(np.mean(epe > 3.0)),
            "dir_deg": float(np.mean(np.degrees(np.arccos(np.clip(cos, -1, 1))))),
            "valid_frac": float(valid.mean())}


def class_agnostic_pq(pred_lab, targ_lab, iou_thresh: float = 0.5) -> dict:
    """Class-agnostic Panoptic Quality: greedy IoU matching between predicted and target segments.

    Permutation-invariant **over permutations WITHIN the target's own colour set** — the palette index is
    arbitrary, so two identical partitions whose colours are permuted among the target's colours score
    identically. Plain L1 (and per-pixel accuracy) both fail that, which is what made the segmentation
    numbers look saturated.

    CORRECTION (2026-09-12 audit): the invariance does NOT extend to colours outside that set, because the
    prediction reaches here through `decode_to_target_vocabulary`. The target's colours are a NON-CONTIGUOUS
    subset of DISTINCT_PALETTE, so a PERFECT partition relabelled with unused palette indices is quantised
    onto the wrong vocabulary entries and scores pq 0.6351 (0 of 64 records perfect), not 1.0. Within-
    vocabulary permutation does score exactly 1.0000 on 64/64. So these numbers are not colour-agnostic in
    the strong sense this docstring used to claim.
    """
    pl, tl = np.asarray(pred_lab), np.asarray(targ_lab)
    tsegs = np.unique(tl)
    psegs = np.unique(pl)
    if tsegs.size == 0:
        return {"pq": float("nan"), "sq": float("nan"), "rq": float("nan"), "n_target_segs": 0}

    # Pairwise IoU via a CONFUSION MATRIX built with one bincount pass. The obvious
    # implementation intersects every (target, pred) mask pair, which is O(T * P * pixels) -- with
    # ~10 segments each that measured 232 ms/record, the single largest metric cost. A confusion
    # matrix is O(pixels) and gives every pair at once; results are identical because
    # |A & B| and |A| + |B| are exactly what the matrix and its margins hold.
    ti_of = np.searchsorted(tsegs, tl.ravel())
    pi_of = np.searchsorted(psegs, pl.ravel())
    cm = np.bincount(ti_of * psegs.size + pi_of,
                     minlength=tsegs.size * psegs.size).reshape(tsegs.size, psegs.size)
    t_area = cm.sum(1, keepdims=True)
    p_area = cm.sum(0, keepdims=True)
    union = t_area + p_area - cm
    with np.errstate(divide="ignore", invalid="ignore"):
        iou = np.where(union > 0, cm / np.maximum(union, 1), 0.0)
    ii, jj = np.nonzero(cm > 0)
    pairs = [(float(iou[i, j]), int(tsegs[i]), int(psegs[j])) for i, j in zip(ii, jj)]
    tsegs = list(tsegs)
    psegs = list(psegs)
    pairs.sort(reverse=True)
    used_t, used_p, ious = set(), set(), []
    for iou, ti, pi in pairs:
        if iou < iou_thresh or ti in used_t or pi in used_p:
            continue
        used_t.add(ti); used_p.add(pi); ious.append(iou)
    tp = len(ious)
    fp, fn = len(psegs) - tp, len(tsegs) - tp
    sq = float(np.mean(ious)) if tp else 0.0
    rq = tp / max(tp + 0.5 * fp + 0.5 * fn, _EPS)
    return {"pq": float(sq * rq), "sq": sq, "rq": float(rq),
            "n_target_segs": len(tsegs), "n_pred_segs": len(psegs)}


def mean_iou(pred_lab, targ_lab) -> dict:
    """mIoU over the classes PRESENT in the target.

    Valid for `semantic_segmentation` specifically: its builder runs `_panoptic_to_semantic`
    before `colorize_labels`, so the palette index IS the semantic class and is shared across
    images. (It would be meaningless for instance/panoptic, whose ids are per-image and arbitrary
    -- which is why those keep PQ, a permutation-invariant partition metric, instead.)

    Averaged over target-present classes only: including absent classes would score a model for
    correctly not predicting things that were never there, and the average would be dominated by
    the 1024-entry palette rather than by the handful of classes in the image.
    """
    pl, tl = np.asarray(pred_lab), np.asarray(targ_lab)
    classes = np.unique(tl)
    ious = []
    for c in classes:
        p, t = (pl == c), (tl == c)
        union = float((p | t).sum())
        if union > 0:
            ious.append(float((p & t).sum()) / union)
    return {"miou": float(np.mean(ious)) if ious else float("nan"),
            "n_classes_in_target": int(len(classes)),
            "pixel_acc": float((pl == tl).mean())}


def boundary_f_score(pred01, targ01, tol_px: int = 2, thresh: float = 0.5) -> dict:
    """BSDS-style boundary precision/recall/F with a tolerance, for EDGE maps.

    Plain per-pixel F1 is the wrong metric for thin structures: a one-pixel misalignment turns a
    correct edge into both a false positive AND a false negative, so a good edge map can score near
    zero. Standard edge benchmarks therefore allow a small matching tolerance. Implemented with a
    max-filter dilation, which is the cheap equivalent of "is there a target edge within tol_px".
    """
    from scipy.ndimage import maximum_filter

    p = (pred01.mean(-1) if pred01.ndim == 3 else pred01)
    t = (targ01.mean(-1) if targ01.ndim == 3 else targ01)
    tb = t > thresh
    k = 2 * tol_px + 1
    best = {"boundary_f": 0.0, "boundary_precision": 0.0, "boundary_recall": 0.0,
            "boundary_f_thresh": thresh}
    t_dil = maximum_filter(tb.astype(np.uint8), size=k) > 0
    for th in np.linspace(0.05, 0.95, 19):
        pb = p > th
        if not pb.any() and not tb.any():
            continue
        p_dil = maximum_filter(pb.astype(np.uint8), size=k) > 0
        # precision: predicted edge pixels near SOME target edge; recall: target pixels near SOME pred
        prec = float((pb & t_dil).sum()) / max(float(pb.sum()), _EPS)
        rec = float((tb & p_dil).sum()) / max(float(tb.sum()), _EPS)
        f = 0.0 if (prec + rec) <= 0 else 2 * prec * rec / (prec + rec)
        if f > best["boundary_f"]:
            best = {"boundary_f": f, "boundary_precision": prec, "boundary_recall": rec,
                    "boundary_f_thresh": float(th)}
    return best


def keypoint_pck(pred_u8, targ_u8, palette, alpha: float = 0.05) -> dict:
    """TRUE PCK, using the fact that joint identity IS recoverable from colour.

    `_common.draw_discs` sets `colour = LABEL_PALETTE[(joint_index % 1023) + 1]` and documents it as
    "the fixed, decodable convention keypoint targets need". An earlier version of this module
    asserted PCK was impossible from a raster and shipped a structure proxy instead -- that was
    wrong, and the encoder had said so.

    Per joint colour: take the centroid of the target discs and of the predicted discs, and count
    the joint correct if the centroids are within `alpha * image_diagonal`. Multiple instances share
    a colour per joint, so with >1 instance the centroid is an approximation -- reported via
    `n_joints` so multi-person records can be filtered. Distances are also returned in pixels,
    which is threshold-free.

    BACKGROUND MUST BE EXCLUDED FIRST, and getting this wrong was a real bug. A keypoint render is
    ~99.5% black (measured: nonblack_frac 0.001-0.005). An earlier version assigned EVERY pixel to
    its nearest of the 1024 palette entries, so the vast black background was distributed across
    joint colours and each "predicted disc" became a scatter covering the whole frame. The centroid
    was then meaningless and PCK read 0.125-0.204 even for a PERFECT (AE round-trip) prediction --
    which I initially mis-attributed to the AE erasing small discs. Restricting to foreground first
    is what makes the centroid mean anything.

    Foreground is defined by luminance, not by colour distance, because the background is pure black
    by construction and a threshold on brightness cannot accidentally admit it.
    """
    pred_u8 = np.asarray(pred_u8)
    targ_u8 = np.asarray(targ_u8)
    # joint colours are exactly the non-black colours PRESENT IN THE TARGET
    tfg = targ_u8.max(axis=-1) > 24
    pfg = pred_u8.max(axis=-1) > 24
    if not tfg.any():
        return {"pck": float("nan"), "pck_px_err": float("nan"), "n_joints": 0,
                "n_matched_joints": 0, "pred_fg_frac": float(pfg.mean())}
    joint_cols = _unique_colours(targ_u8, tfg)

    def assign(img, fg):
        """-> label map over FOREGROUND only; background = -1."""
        lab = np.full(img.shape[:2], -1, dtype=np.int32)
        px = img[fg].reshape(-1, 3).astype(np.int64)                 # int64: int16 squares overflow (2026-09-08 fix)
        if px.size:
            dist = ((px[:, None, :] - joint_cols[None, :, :].astype(np.int64)) ** 2).sum(-1)
            lab[fg] = dist.argmin(1)
        return lab

    pl = assign(pred_u8, pfg)
    tl = assign(targ_u8, tfg)
    h, w = tl.shape
    diag = float(np.hypot(h, w))
    tol = alpha * diag
    joints = [c for c in np.unique(tl) if c >= 0]          # -1 is background
    if not joints:
        return {"pck": float("nan"), "pck_px_err": float("nan"), "n_joints": 0,
                "n_matched_joints": 0}
    # v4c (2026-09-05): official protocols normalise by the SUBJECT, not the image (Animal-Pose PCK@0.05 of the
    # longest bbox side, MPII PCKh by head size, 300W NME by inter-ocular distance, COCO/AP-10K OKS by object
    # scale). Report a bbox-normalised PCK@0.1 and an NME alongside the historical image-diagonal PCK.
    tys, txs = np.nonzero(tl >= 0)
    bbox_side = float(max(1, max(tys.max() - tys.min() + 1, txs.max() - txs.min() + 1)))
    tol_bbox = 0.1 * bbox_side
    hits, hits_bbox, errs, matched = 0, 0, [], 0
    for c in joints:
        ty, tx = np.nonzero(tl == c)
        py, px = np.nonzero(pl == c)
        if len(py) == 0:
            errs.append(diag)                              # missing joint: worst case, not skipped
            continue
        matched += 1
        d = float(np.hypot(ty.mean() - py.mean(), tx.mean() - px.mean()))
        errs.append(d)
        hits += int(d <= tol)
        hits_bbox += int(d <= tol_bbox)
    return {"pck": hits / len(joints), "pck_px_err": float(np.mean(errs)),
            "pck_bbox": hits_bbox / len(joints),                       # PCK@0.1 of the target's keypoint bbox side
            "nme_bbox": float(np.mean(errs) / bbox_side),              # mean error / bbox side (lower better)
            "n_joints": len(joints), "n_matched_joints": matched,
            "pred_fg_frac": float(pfg.mean())}


def keypoint_structure_proxy(pred01, targ01) -> dict:
    """PROXY, not PCK. Named so it cannot be mistaken for one.

    Keypoint targets are rendered discs/skeletons. Joint IDENTITY and ORDERING are not recoverable
    from a raster, so PCK — which requires matching joint k to joint k — is impossible here. What
    this measures is whether ink landed in the right places: F1 over the non-background mask, plus
    a symmetric Chamfer distance in pixels between the two ink sets. Useful for ranking arms,
    NOT a pose-estimation number.
    """
    # ink = deviation from the local background; the render draws bright marks
    pi = pred01.mean(-1) if pred01.ndim == 3 else pred01
    ti = targ01.mean(-1) if targ01.ndim == 3 else targ01
    tm = ti > (ti.mean() + 2 * ti.std())
    out = binary_mask_metrics(pi, tm.astype(np.float32), thresh=0.5)
    pm = pi > (pi.mean() + 2 * pi.std())
    ys, xs = np.nonzero(tm); pys, pxs = np.nonzero(pm)
    if len(ys) == 0 or len(pys) == 0:
        out["chamfer_px_proxy"] = float("nan")
        return {f"{k}_proxy" if not k.endswith("proxy") else k: v for k, v in out.items()}
    # subsample for cost; Chamfer is O(n*m)
    sel_t = np.linspace(0, len(ys) - 1, min(len(ys), 400)).astype(int)
    sel_p = np.linspace(0, len(pys) - 1, min(len(pys), 400)).astype(int)
    T = np.stack([ys[sel_t], xs[sel_t]], 1).astype(np.float32)
    P = np.stack([pys[sel_p], pxs[sel_p]], 1).astype(np.float32)
    d = np.sqrt(((T[:, None, :] - P[None, :, :]) ** 2).sum(-1))
    out["chamfer_px"] = float(0.5 * (d.min(1).mean() + d.min(0).mean()))
    return {f"{k}_proxy" if not k.endswith("proxy") else k: v for k, v in out.items()}


# ---------------------------------------------------------------- task registry
# Family per task, taken from `rgb_supervision.BUILDERS` -- NOT invented. Two families are
# variant-dependent (the same task renders differently by source dataset), so those detect the
# encoding from the target itself via `is_grayscale`.
FAMILY: dict[str, str] = {
    # appearance: the target IS an RGB photo, so RGB metrics are the native ones
    "colorization": "appearance", "denoising": "appearance", "deblurring": "appearance",
    "inpainting": "appearance", "super_resolution": "appearance",
    # per-image-normalised grayscale scalar fields
    "metric_depth": "scalar", "relative_depth": "scalar",
    "stereo_disparity": "scalar", "scene_flow": "scalar",
    # binary/soft masks
    "saliency_detection": "mask",
    # exactly invertible
    "surface_normals": "normals",
    "optical_flow": "flow",
    # permutation-invariant partitions
    # semantic gets mIoU (its palette index IS the class, shared across images) PLUS PQ;
    # instance/panoptic ids are per-image and arbitrary, so they get PQ only
    "semantic_segmentation": "semantic",
    "instance_segmentation": "labels", "panoptic_segmentation": "labels",
    # edges are THIN structures: boundary P/R/F with a tolerance, not per-pixel F1 and not PQ.
    # Measured on the pack: edge targets are GRAYSCALE with nonblack_frac 0.10-0.14.
    "edge_detection": "edges",
    # variant-dependent: inverse-depth ramp (grayscale) OR xyz RGB (Hypersim)
    "scene_coordinates": "scalar_or_appearance",
    # proxy only -- see keypoint_structure_proxy
    # PCK is possible after all: draw_discs encodes joint index in the palette colour
    "human_pose_2d": "keypoints", "face_landmarks": "keypoints", "object_keypoints": "keypoints",

    # ---- tasks that EXISTED IN PACKS but were never mapped, so they fell through to
    # "appearance"/PSNR silently. Measured from the packs 2026-08-29:
    #   illumination            RGB, ~3.7k colours          -> appearance (fallback was right)
    #   intrinsic_decomposition RGB, ~2.6k colours          -> appearance (fallback was right)
    #   reflectance             RGB, ~2.4k colours          -> appearance (fallback was right)
    #   shadow_detection        grayscale, 98.8% at 0/255   -> mask  <-- FALLBACK WAS WRONG,
    #     a binary shadow mask was being scored by PSNR. Now f1_best.
    "illumination": "appearance", "intrinsic_decomposition": "appearance",
    "reflectance": "appearance", "shadow_detection": "mask",

    # ---- LINEAGE-A task names (the lvm 4-task pool used by model_v51 + trainer_pair_v42).
    # These are DIFFERENT RENDERINGS of similar-sounding concepts and were previously ABSENT,
    # so `FAMILY.get(task, "appearance")` below scored them as PSNR with no error raised --
    # the bug that invalidated two DemoGrad runs. Families assigned by MEASURING the exported
    # targets (inference/our_demo_pairs_v3, the exact v51 val convention):
    #   depth        grayscale, 160 levels                  -> scalar  (mae)
    #   normal       RGB, ~9.8k colours                     -> normals (mean_deg)
    #   segmentation grayscale, 2 colours, 100% at 0/255    -> mask    (f1_best)
    #   panoptic     RGB, 15 colours                        -> labels  (pq)
    # NOTE the segmentation split: lineage A renders LEGACY WHITE/BLACK (binary -> mask), while
    # lineage B's "semantic_segmentation" is a class PALETTE (-> semantic/miou). Same word,
    # different family, different metric. Never map one onto the other.
    "depth": "scalar",
    "normal": "normals",
    "segmentation": "mask",
    "panoptic": "labels",
}

# The single number to rank arms by, per family, and whether lower is better.
PRIMARY: dict[str, tuple[str, bool]] = {
    "appearance": ("psnr_db", False),      # higher better
    "scalar": ("mae", True),
    "mask": ("f1_best", False),
    "normals": ("mean_deg", True),
    "flow": ("epe_px", True),
    "labels": ("pq", False),
    "semantic": ("miou", False),           # standard for semantic segmentation
    "edges": ("boundary_f", False),        # standard for edge detection
    "keypoints": ("pck", False),           # TRUE PCK, not the old proxy
}


def _palette():
    from jflux2.dataloaders.vgi_rgb._common import LABEL_PALETTE
    return LABEL_PALETTE


def score_pair(task: str, pred_u8: np.ndarray, targ_u8: np.ndarray,
               *, strict: bool = False) -> dict:
    """Task-appropriate metrics for ONE (prediction, target) uint8 RGB pair.

    Always also returns `l1_01` so nothing that depends on the historical metric breaks, and
    `family`/`primary` so a reader knows which number to rank by and in which direction.
    """
    p01 = np.asarray(pred_u8, np.float32) / 255.0
    t01 = np.asarray(targ_u8, np.float32) / 255.0
    if strict and task not in FAMILY:
        raise KeyError(
            f"score_pair: unknown task {task!r}. It would silently fall through to the "
            f"'appearance' family and be scored by PSNR. Add it to FAMILY (measure the target "
            f"encoding first) or pass strict=False deliberately.")
    fam = FAMILY.get(task, "appearance")
    if fam == "scalar_or_appearance":
        fam = "scalar" if is_grayscale(targ_u8) else "appearance"

    out: dict = {"family": fam, "l1_01": float(np.mean(np.abs(p01 - t01)))}
    if fam == "appearance":
        out["psnr_db"] = psnr(p01, t01)
        out["ssim"] = ssim(p01, t01)
    elif fam == "scalar":
        ps, _ = decode_scalar(pred_u8)
        ts, tv = decode_scalar(targ_u8)
        out.update(masked_scalar_metrics(ps, ts, tv))
    elif fam == "mask":
        ps, _ = decode_scalar(pred_u8)
        ts, _ = decode_scalar(targ_u8)
        out.update(binary_mask_metrics(ps, ts))
    elif fam == "normals":
        pn, _ = decode_normals(pred_u8)
        tn, tv = decode_normals(targ_u8)
        out.update(angular_error_deg(pn, tn, tv))
    elif fam == "flow":
        pu, pv, _ = decode_flow(pred_u8)
        tu, tv_, tvalid = decode_flow(targ_u8)
        out.update(flow_metrics(pu, pv, tu, tv_, tvalid))
    elif fam == "labels":
        cols = _unique_colours(targ_u8)          # computed ONCE, reused for both decodes
        pl = decode_to_target_vocabulary(pred_u8, targ_u8, cols)
        tl = decode_to_target_vocabulary(targ_u8, targ_u8, cols)
        out.update(class_agnostic_pq(pl, tl))
    elif fam == "semantic":
        cols = _unique_colours(targ_u8)
        pl = decode_to_target_vocabulary(pred_u8, targ_u8, cols)
        tl = decode_to_target_vocabulary(targ_u8, targ_u8, cols)
        out.update(mean_iou(pl, tl))
        out.update(class_agnostic_pq(pl, tl))      # keep PQ alongside: structure vs labelling
    elif fam == "edges":
        out.update(boundary_f_score(p01, t01))
        ps, _ = decode_scalar(pred_u8); ts, _ = decode_scalar(targ_u8)
        out.update({f"px_{k}": v for k, v in binary_mask_metrics(ps, ts).items()})
    elif fam == "keypoints":
        out.update(keypoint_pck(pred_u8, targ_u8, _palette()))
        out.update(keypoint_structure_proxy(p01, t01))   # retained for continuity
    name, lower = PRIMARY.get(fam, ("l1_01", True))
    out["primary_metric"] = name
    out["primary_value"] = out.get(name, float("nan"))
    out["primary_lower_is_better"] = lower
    if lower_is_better(name) != lower:      # guard the table against drifting from the helper
        raise AssertionError(f"PRIMARY direction for {name!r} disagrees with lower_is_better()")
    return out


