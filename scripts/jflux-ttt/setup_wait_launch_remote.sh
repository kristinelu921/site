#!/usr/bin/env bash
set -euo pipefail
ROOT=/dev/shm/jflux_ttt_t3ez_p16_q0_all_tasks_v4_a3_0100
GCS=gs://kmh-gcp-us-central2/kristine/eval_runs/2026-10-01/jflux_ttt/t3ez_p16_q0_all_tasks_v4
STD="$GCS/inputs/standard"
PY=/kmh-nfs-ssd-us-mount/code/xtiange/anaconda3/envs/resnet/bin/python
NONCE=jflux-ttt-t3ez-p16-q0-v4-20261001-a3-0100
APPROVAL="$GCS/control/LAUNCH_APPROVED_a3_0100.json"
EPS='relative_depth:0,surface_normals:0,semantic_segmentation:0,instance_segmentation:0,panoptic_segmentation:0,edge_detection:0,saliency_detection:0,human_pose_2d:0,face_landmarks:0,object_keypoints:0,colorization:0,denoising:0,deblurring:0,inpainting:0,super_resolution:0'

umask 002
if test -e "$ROOT"; then
  echo "refusing to overwrite existing run root $ROOT" >&2
  exit 1
fi
mkdir -p "$ROOT"/{assets,logs,output,tmp}
export TMPDIR="$ROOT/tmp"
export JFLUX2_ASSET_CACHE="$ROOT/assets"

gsutil cp "$GCS/source/run_bundle.tar" "$ROOT/run_bundle.tar"
gsutil cp "$GCS/source/run_bundle.tar.sha256" "$ROOT/run_bundle.tar.sha256"
(cd "$ROOT" && sha256sum -c run_bundle.tar.sha256)
tar -C "$ROOT" -xf "$ROOT/run_bundle.tar"
gsutil cp "$GCS/source/SOURCE_SHA256SUMS" "$ROOT/SOURCE_SHA256SUMS"
(cd "$ROOT/source" && sha256sum -c "$ROOT/SOURCE_SHA256SUMS")

fetch() {
  local rel=$1
  mkdir -p "$ROOT/assets/$(dirname "$rel")"
  gsutil cp "$STD/$rel" "$ROOT/assets/$rel"
}
fetch pretrained-ckpts/xtiange/jax_weights/4B_base/jflux2_transformer.safetensors
fetch pretrained-ckpts/xtiange/jax_weights/4B/jflux2_vae.safetensors
fetch xtiange/ttt/ctx_qwen3_4b.npz
fetch xtiange/mt_eval_packs/test_s64_px256_v5/samples.json
fetch xtiange/mt_eval_packs/test_s64_px256_v5/samples.npz
gsutil cp "$GCS/source/INPUT_SHA256SUMS" "$ROOT/INPUT_SHA256SUMS"
(cd "$ROOT/assets" && sha256sum -c "$ROOT/INPUT_SHA256SUMS") | tee "$ROOT/logs/input_hashes.log"

PACK="$ROOT/assets/xtiange/mt_eval_packs/test_s64_px256_v5"
JAX_PLATFORMS=cpu PYTHONNOUSERSITE=1 PYTHONPATH="$ROOT/source" "$PY" - "$PACK" "$EPS" <<'PY' | tee "$ROOT/logs/cpu_preflight.json"
import json, sys
from pathlib import Path
from jflux2.analysis import capacity_ladder_v8 as M
from jflux2.analysis import ttt_ladder_v1 as L
from jflux2.analysis import capacity_ladder_v6 as C
pack=L.load_pack(sys.argv[1])
assert pack['pack_id']=='48e062452574ff0ded61c72664445d687ecd1c02'
arms=M.parse_arms('T3EZ=q3ez/f3')
assert M.canonical_arms(arms)=='T3EZ=q3ez/f3'
eps=[(x.split(':')[0],int(x.split(':')[1])) for x in sys.argv[2].split(',')]
tasks=[]
for t,q in eps:
    idx=[i for i,r in enumerate(pack['records']) if r['task']==t]
    src=[str(pack['records'][i].get('source')) for i in idx]
    assert q < len(idx) and len(C.same_source(src,q))-1 >= 16, (t,q,len(C.same_source(src,q))-1)
    tasks.append(t)
assert len(eps)==15 and len(set(tasks))==15
print(json.dumps({'status':'PASS','pack_id':pack['pack_id'],'episodes':len(eps),'tasks':tasks,
                  'arms':'T3EZ=q3ez/f3','p':16,'steps':200,'backend':'cpu'},sort_keys=True))
PY

"$PY" - "$ROOT/logs/REMOTE_READY_a3_0100.json" "$NONCE" <<'PY'
import json,sys
json.dump({'status':'REMOTE_READY_FOR_FINAL_GATE','nonce':sys.argv[2],
           'tpu_name':'kmh-tpuvm-v4-8-spot-kristine-01-00-29-33-N-us-central2-b',
           'zone':'us-central2-b','assets':'sha256_pass','cpu_preflight':'pass'},open(sys.argv[1],'w'),sort_keys=True)
PY
gsutil cp "$ROOT/logs/REMOTE_READY_a3_0100.json" "$GCS/logs/REMOTE_READY_a3_0100.json"
echo "REMOTE_READY_FOR_FINAL_GATE $APPROVAL"

while ! gsutil -q stat "$APPROVAL"; do sleep 10; done
gsutil cp "$APPROVAL" "$ROOT/LAUNCH_APPROVED.json"
"$PY" - "$ROOT/LAUNCH_APPROVED.json" "$NONCE" <<'PY'
import json,sys
d=json.load(open(sys.argv[1]))
assert d['status']=='FINAL_GATE_PASS' and d['nonce']==sys.argv[2]
assert d['tpu_name']=='kmh-tpuvm-v4-8-spot-kristine-01-00-29-33-N-us-central2-b'
PY

if gsutil -q stat "$GCS/output/results.partial.json"; then
  gsutil -m rsync -r "$GCS/output" "$ROOT/output"
fi

sync_loop() {
  while kill -0 "$1" 2>/dev/null; do
    gsutil -m rsync -r "$ROOT/output" "$GCS/output" >/dev/null 2>&1 || true
    gsutil -q cp "$ROOT/logs/runner_a3_0100.log" "$GCS/logs/runner_a3_0100.log" 2>/dev/null || true
    sleep 120
  done
}

export PYTHONNOUSERSITE=1
export PYTHONPATH="$ROOT/source"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export TPU_LOG_DIR="$ROOT/logs/tpu"
mkdir -p "$TPU_LOG_DIR"
set +e
"$PY" -m jflux2.analysis.capacity_ladder_v8 \
  --pack "$PACK" \
  --episodes "$EPS" --arms 'T3EZ=q3ez/f3' --p 16 --steps 200 --lr 0.01 --batch 4 --pool-t 8 \
  --noprompt-steps 50 --seed 42 --param-dtype bfloat16 --local-fsdp 1 --out "$ROOT/output" \
  > >(tee "$ROOT/logs/runner_a3_0100.log") 2>&1 &
RUN_PID=$!
sync_loop "$RUN_PID" & SYNC_PID=$!
wait "$RUN_PID"; RC=$?
wait "$SYNC_PID" 2>/dev/null || true
gsutil -m rsync -r "$ROOT/output" "$GCS/output" || true
gsutil -m rsync -r "$ROOT/logs" "$GCS/logs" || true
"$PY" - "$ROOT/logs/terminal_status_a3_0100.json" "$RC" <<'PY'
import json,sys,datetime
json.dump({'exit_code':int(sys.argv[2]),
           'finished_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()},open(sys.argv[1],'w'),sort_keys=True)
PY
gsutil cp "$ROOT/logs/terminal_status_a3_0100.json" "$GCS/logs/terminal_status_a3_0100.json" || true
exit "$RC"
