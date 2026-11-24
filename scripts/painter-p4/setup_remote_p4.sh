#!/usr/bin/env bash
set -euo pipefail

root=/dev/shm/native-p1-ensembles-painter
base=gs://kmh-gcp-us-central2/kristine/eval_runs/2026-09-30/native_p1_ensembles_v4_queues
rm -rf "$root"
mkdir -p "$root"/{inputs,bundle,work/p4,logs,cache,tmp}
export TMPDIR="$root/tmp" XDG_CACHE_HOME="$root/cache" PIP_CACHE_DIR="$root/cache/pip"
python=/kmh-nfs-ssd-us-mount/code/kristine/miniconda3/envs/lvm310/bin/python
[[ -x "$python" ]]
gsutil -m cp "$base/assets/painter/run_bundle.tar" "$base/assets/painter/painter_vit_large.standard.pth" \
  "$base/assets/painter/overlay/painter_xla_infer.py" "$base/assets/painter/overlay/painter_compat.py" \
  "$base/episodes/test_p4_episodes.standard.tar" "$base/episodes/test_p4_episodes.tar.sha256" \
  "$base/episodes/audit_p4.json" "$root/inputs/"
mv "$root/inputs/painter_vit_large.standard.pth" "$root/inputs/painter_vit_large.pth"
mv "$root/inputs/test_p4_episodes.standard.tar" "$root/inputs/test_p4_episodes.tar"
expected=$(awk '{print $1}' "$root/inputs/test_p4_episodes.tar.sha256")
actual=$(sha256sum "$root/inputs/test_p4_episodes.tar" | awk '{print $1}')
[[ "$actual" == "$expected" ]]
tar -C "$root/bundle" -xf "$root/inputs/run_bundle.tar"
cp "$root/inputs/painter_xla_infer.py" "$root/inputs/painter_compat.py" "$root/bundle/"
tar -C "$root/work/p4" -xf "$root/inputs/test_p4_episodes.tar"
# Reuse the user-requested immutable NFS lvm310 environment. Do not import
# torch_xla here: setup must not acquire TPU devices before the final gate.
"$python" -m pip show torch torch-xla numpy pillow > "$root/logs/runtime_packages.txt"
date -u +'%Y-%m-%dT%H:%M:%SZ' > "$root/SETUP_COMPLETE"
gsutil -m cp "$root/SETUP_COMPLETE" "$root/logs/runtime_packages.txt" "$base/logs/painter/"
gsutil cp "$base/scripts/painter/run_remote_p4.sh" "$root/run_remote_p4.sh"
chmod +x "$root/run_remote_p4.sh"
echo "REMOTE_SETUP_READY model=painter root=$root" >&2
marker="$base/control/launch_painter_p4_attempt5"
until gsutil -q stat "$marker"; do sleep 5; done
export EXPECTED_TPU_NAME="${EXPECTED_TPU_NAME:?}" EXPECTED_TPU_ZONE="${EXPECTED_TPU_ZONE:?}"
exec "$root/run_remote_p4.sh"
