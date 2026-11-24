#!/usr/bin/env bash
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
TPU=kmh-tpuvm-v4-8-spot-kristine-01-00-29-33-N-us-central2-b
ZONE=us-central2-b
PROJECT=he-vision-group
[[ "$TPU" == *kristine* ]]

ELF_TPU_INITIALIZE_MISSING=0 bash tpu_scripts/check_tpus.sh "$PROJECT" | tee "$HERE/logs_inventory_before_setup_a3_0100.txt"
awk -v t="$TPU" 'f && $1==t && $2=="'"$ZONE"'"{ok=1} /^AVAILABLE$/{f=1} END{exit !ok}' "$HERE/logs_inventory_before_setup_a3_0100.txt"
gcloud compute tpus tpu-vm describe "$TPU" --zone "$ZONE" --project "$PROJECT" --format=json > "$HERE/api_before_setup_a3_0100.json"
python - "$HERE/api_before_setup_a3_0100.json" <<'PY'
import json,sys
d=json.load(open(sys.argv[1])); assert d['state']=='READY' and d['health']=='HEALTHY'
assert d['acceleratorType']=='v4-8' and d['acceleratorConfig']['topology']=='2x2x1'
assert d['schedulingConfig']['spot'] is True
PY
bash .agents/skills/query-monitor-gcp-tpus/scripts/check_tpu_occupancy.sh "$PROJECT" "$TPU" "$ZONE" | tee "$HERE/occupancy_before_setup_a3_0100.log"
grep -q '^AVAILABLE ' "$HERE/occupancy_before_setup_a3_0100.log"
gcloud compute tpus tpu-vm ssh "$TPU" --zone "$ZONE" --project "$PROJECT" --worker=0 \
  --command='if fuser /dev/accel* >/tmp/jflux_ttt_accel_precheck 2>&1; then cat /tmp/jflux_ttt_accel_precheck; exit 41; fi'
gcloud compute tpus tpu-vm ssh "$TPU" --zone "$ZONE" --project "$PROJECT" --worker=0 --command='bash -s' < "$HERE/setup_wait_launch_remote.sh"
