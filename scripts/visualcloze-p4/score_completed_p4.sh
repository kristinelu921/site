#!/usr/bin/env bash
set -euo pipefail

repo=/kmh-nfs-ssd-us-mount/code/kristine/lvm
bundle="$repo/eval_runs/2026-09-30/visualcloze_jflux/test_p4_p8_p16_v5e8"
recovery="$bundle/scoring_recovery"
run="$recovery/P4"
python=/kmh-nfs-ssd-us-mount/code/xtiange/anaconda3/envs/resnet/bin/python
dest=gs://kmh-gcp-us-west4/kristine/eval_runs/2026-09-30/visualcloze_jflux/test_p4_p8_p16_v5e8/results/P4

expected_archive_sha=$(awk '{print $1}' "$recovery/P4_episodes.tar.sha256")
observed_archive_sha=$(sha256sum "$recovery/P4_episodes.tar" | awk '{print $1}')
[[ "$observed_archive_sha" == "$expected_archive_sha" ]]
if [[ ! -d "$run/episodes" ]]; then
  tar -C "$run" -xf "$recovery/P4_episodes.tar"
fi
"$python" "$bundle/audit_manifest.py" "$run" --p 4 --verify-files \
  | tee "$recovery/manifest_audit.score_recovery.log"

run_score() {
  local stem=$1
  shift
  local attempt rc
  for attempt in 1 2 3 4 5 6 7 8; do
    [[ ! -f "$run/${stem}.json" ]] || return 0
    set +e
    "$python" "$bundle/visualcloze_v5_eval.py" score --run-dir "$run" --resume "$@" \
      2>&1 | tee -a "$recovery/${stem}.attempts.log"
    rc=${PIPESTATUS[0]}
    set -e
    [[ "$rc" -ne 0 ]] || break
    echo "${stem}: attempt ${attempt} exited ${rc}; resuming from durable partial" \
      | tee -a "$recovery/${stem}.attempts.log"
  done
  [[ -f "$run/${stem}.json" ]]
}

run_score scores
run_score scores_perceptual --perceptual

"$python" - "$run/scores.json" "$run/scores_perceptual.json" <<'PY'
import json,sys
for path in sys.argv[1:]:
    data=json.load(open(path))
    assert data["pack_id"]=="48e062452574ff0ded61c72664445d687ecd1c02"
    assert len(data["results"])==3840
    assert len(data["cells"])==60
    print(path, "PASS", len(data["results"]), len(data["cells"]))
PY

gcloud storage cp \
  "$run/scores.json" "$run/scores_perceptual.json" \
  "$recovery/manifest_audit.score_recovery.log" \
  "$recovery/scores.attempts.log" "$recovery/scores_perceptual.attempts.log" \
  "$dest/"
