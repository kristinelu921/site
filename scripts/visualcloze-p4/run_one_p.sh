#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 || ! "$1" =~ ^(4|8|16)$ ]]; then
  echo "usage: $0 4|8|16" >&2
  exit 2
fi

readonly P="$1"
readonly WORK_ROOT=/dev/shm/visualcloze_p_ladder_20260930
readonly BUNDLE="$WORK_ROOT/bundle"
readonly RUN_DIR="$WORK_ROOT/runs/P${P}"
readonly VENV="$WORK_ROOT/venv"
readonly SCORE_PY=/kmh-nfs-ssd-us-mount/code/xtiange/anaconda3/envs/resnet/bin/python

"$SCORE_PY" "$BUNDLE/audit_manifest.py" "$RUN_DIR" --p "$P"

export PJRT_DEVICE=TPU
export HF_HOME="$WORK_ROOT/hf-home"
export HF_XET_HIGH_PERFORMANCE=1
export TOKENIZERS_PARALLELISM=false

set +e
"$VENV/bin/python" "$BUNDLE/visualcloze_v5_eval.py" infer \
  --run-dir "$RUN_DIR" \
  --cache-dir "$WORK_ROOT" \
  --steps 30 \
  --guidance 30 \
  --model-seed 0 \
  --batch-size 2 \
  --flash-block-size 1024 \
  --local-files-only \
  --resume &
infer_pid=$!
"$BUNDLE/sync_loop.sh" "$P" "$infer_pid" &
sync_pid=$!
wait "$infer_pid"
infer_rc=$?
wait "$sync_pid" || true
set -e

if [[ "$infer_rc" -ne 0 ]]; then
  "$BUNDLE/sync_one_p.sh" "$P" || true
  exit "$infer_rc"
fi

"$SCORE_PY" "$BUNDLE/visualcloze_v5_eval.py" score --run-dir "$RUN_DIR" --resume
"$SCORE_PY" "$BUNDLE/visualcloze_v5_eval.py" score --run-dir "$RUN_DIR" --perceptual --resume
"$BUNDLE/sync_one_p.sh" "$P"
date -u +%Y-%m-%dT%H:%M:%SZ > "$RUN_DIR/COMPLETE_UTC"
gcloud storage cp "$RUN_DIR/COMPLETE_UTC" \
  "gs://kmh-gcp-us-west4/kristine/eval_runs/2026-09-30/visualcloze_jflux/test_p4_p8_p16_v5e8/results/P${P}/COMPLETE_UTC"

