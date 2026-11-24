#!/usr/bin/env bash
set -euo pipefail

root=/dev/shm/native-p1-ensembles-painter
base=gs://kmh-gcp-us-central2/kristine/eval_runs/2026-09-30/native_p1_ensembles_v4_queues
export PJRT_DEVICE=TPU
export TPU_STDERR_LOG_LEVEL=0
export TPU_MIN_LOG_LEVEL=0
export TPU_LOG_DIR="$root/logs/tpu"
export LIBTPU_INIT_ARGS="--xla_tpu_scoped_vmem_limit_kib=98304"
mkdir -p "$TPU_LOG_DIR"
sync_out() {
  if [[ -d "$root/work/p4/predictions" ]]; then
    gsutil -m rsync -r "$root/work/p4/predictions" "$base/outputs/painter/p4/predictions" || true
  fi
  for name in inference.partial.json inference.json; do
    [[ ! -f "$root/work/p4/$name" ]] || gsutil cp "$root/work/p4/$name" "$base/outputs/painter/p4/$name" || true
  done
  gsutil cp "$root/logs/runner_p4.log" "$base/logs/painter/runner_p4.log" || true
}
trap sync_out EXIT
( while true; do sleep 300; sync_out; done ) &
sync_pid=$!
/kmh-nfs-ssd-us-mount/code/kristine/miniconda3/envs/lvm310/bin/python "$root/bundle/painter_xla_infer.py" \
  --run-dir "$root/work/p4" --source-dir "$root/bundle/vendor/Painter" \
  --checkpoint "$root/inputs/painter_vit_large.pth" --cache-dir "$root/cache/xla" \
  --episode-batch 4 --resume --log-every 8 2>&1 | tee "$root/logs/runner_p4.log"
kill "$sync_pid" 2>/dev/null || true
sync_out
