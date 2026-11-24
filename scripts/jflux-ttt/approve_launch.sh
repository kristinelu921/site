#!/usr/bin/env bash
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
TPU=kmh-tpuvm-v4-8-spot-kristine-01-00-29-33-N-us-central2-b
ZONE=us-central2-b
PROJECT=he-vision-group
GCS=gs://kmh-gcp-us-central2/kristine/eval_runs/2026-10-01/jflux_ttt/t3ez_p16_q0_all_tasks_v4
NONCE=jflux-ttt-t3ez-p16-q0-v4-20261001-a3-0100
[[ "$TPU" == *kristine* ]]

gsutil cp "$GCS/logs/REMOTE_READY_a3_0100.json" "$HERE/REMOTE_READY_for_approval_a3_0100.json"
python - "$HERE/REMOTE_READY_for_approval_a3_0100.json" "$NONCE" <<'PY'
import json,sys
d=json.load(open(sys.argv[1])); assert d['status']=='REMOTE_READY_FOR_FINAL_GATE'
assert d['nonce']==sys.argv[2] and d['assets']=='sha256_pass' and d['cpu_preflight']=='pass'
PY

ELF_TPU_INITIALIZE_MISSING=0 bash tpu_scripts/check_tpus.sh "$PROJECT" | tee "$HERE/logs_inventory_final_gate_a3_0100.txt"
awk -v t="$TPU" 'f && $1==t && $2=="'"$ZONE"'"{ok=1} /^AVAILABLE$/{f=1} END{exit !ok}' "$HERE/logs_inventory_final_gate_a3_0100.txt"
gcloud compute tpus tpu-vm describe "$TPU" --zone "$ZONE" --project "$PROJECT" --format=json > "$HERE/api_final_gate_a3_0100.json"
python - "$HERE/api_final_gate_a3_0100.json" <<'PY'
import json,sys
d=json.load(open(sys.argv[1])); assert d['state']=='READY' and d['health']=='HEALTHY'
assert d['acceleratorType']=='v4-8' and d['acceleratorConfig']['topology']=='2x2x1'
assert d['schedulingConfig']['spot'] is True
PY
bash .agents/skills/query-monitor-gcp-tpus/scripts/check_tpu_occupancy.sh "$PROJECT" "$TPU" "$ZONE" | tee "$HERE/occupancy_final_gate_a3_0100.log"
grep -q '^AVAILABLE ' "$HERE/occupancy_final_gate_a3_0100.log"
gcloud compute tpus tpu-vm ssh "$TPU" --zone "$ZONE" --project "$PROJECT" --worker=0 \
  --command='if fuser /dev/accel* >/tmp/jflux_ttt_accel_final_gate 2>&1; then cat /tmp/jflux_ttt_accel_final_gate; exit 41; fi'

python - "$HERE/LAUNCH_APPROVED_a3_0100.json" "$NONCE" "$TPU" "$ZONE" <<'PY'
import json,sys,datetime
out,nonce,tpu,zone=sys.argv[1:]
json.dump({'status':'FINAL_GATE_PASS','nonce':nonce,'tpu_name':tpu,'zone':zone,
           'approved_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()},open(out,'w'),sort_keys=True)
PY
gsutil cp "$HERE/LAUNCH_APPROVED_a3_0100.json" "$GCS/control/LAUNCH_APPROVED_a3_0100.json"
echo "FINAL_GATE_PASS launch approval uploaded"
