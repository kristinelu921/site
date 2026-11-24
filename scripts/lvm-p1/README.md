# LVM-7B clean P=1 / real-condition test

This run evaluates the released open-weight LVM-7B checkpoint on all 960
queries of `test_s64_px256_v5`, with exactly one same-source demonstration
input/target pair per query and only the `real` condition. It is not an
ensemble: the visual sentence is one native autoregressive call over
`[demo input, demo target, query input]`, followed by one generated frame.

The episode seed is `[42, task_seed(task), q, 1]`. Palette-task demonstration
and query targets are coded together once per episode. The manifest records the
complete color map, the demo-visible palette positions used by coverage-aware
scoring, and the query-only positions that cannot be inferred from a singleton
demonstration.

## Official released-code image contract

- Preprocess: convert to PIL RGB, call `resize((256, 256))` (Pillow RGB default
  is bicubic), convert to float32, divide by 255.
- The v5 pack is already 256x256 RGB, and the audit requires this official
  same-size resize to be pixel-identical.
- Decode: official VQGAN `decode_code`, clamp to `[0,1]`.
- Export: multiply by 255 and cast to `uint8` (truncate; do not round).
- Do not resize the native 256x256 generated frame for scoring. Contact-sheet
  layout is display-only and preserves each 256x256 tile exactly.

This uses the released 7B code/checkpoint, not the paper's unreleased 3B
checkpoint. Source commit, model revision, VQ revision, pack hashes, jflux
builder/convention/scorer hashes, and every episode hash are pinned in the run.

No GPU fallback is permitted. TPU launch scripts are added only after selecting
an exact `kristine` resource from the required reconciler's final AVAILABLE
table and will bind the matching regional GCS prefix.
