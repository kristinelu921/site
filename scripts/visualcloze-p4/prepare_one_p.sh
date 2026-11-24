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
readonly PREP_PY=/kmh-nfs-ssd-us-mount/code/xtiange/anaconda3/envs/resnet/bin/python
readonly GCS_ROOT=gs://kmh-gcp-us-west4/kristine/eval_runs/2026-09-30/visualcloze_jflux/test_p4_p8_p16_v5e8

mkdir -p "$RUN_DIR" "$WORK_ROOT/staging" "$WORK_ROOT/logs"

"$PREP_PY" "$BUNDLE/prepare_region_local.py" \
  --p "$P" \
  --out "$RUN_DIR" \
  --eval-script "$BUNDLE/visualcloze_v5_eval.py"

"$PREP_PY" "$BUNDLE/audit_manifest.py" "$RUN_DIR" --p "$P" --verify-files

tar -C "$RUN_DIR" -cf "$WORK_ROOT/staging/P${P}_episodes.tar" manifest.json manifest_audit.json regional_pack_audit.json pack_samples.json episodes
sha256sum "$WORK_ROOT/staging/P${P}_episodes.tar" > "$WORK_ROOT/staging/P${P}_episodes.tar.sha256"
gcloud storage cp "$WORK_ROOT/staging/P${P}_episodes.tar" "$WORK_ROOT/staging/P${P}_episodes.tar.sha256" \
  "$GCS_ROOT/inputs/P${P}/"
date -u +%Y-%m-%dT%H:%M:%SZ > "$WORK_ROOT/logs/P${P}_prepare_complete_utc.txt"
