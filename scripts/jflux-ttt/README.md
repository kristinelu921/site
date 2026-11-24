# JFlux T3EZ test-time-training benchmark

This bundle runs the current JFlux method of record, `T3EZ=q3ez/f3`, as a
bounded benchmark on the current supervision-v5 test pack.

- Pack: `test_s64_px256_v5`, pack id
  `48e062452574ff0ded61c72664445d687ecd1c02`.
- Cohort: query `q0` from each of the 15 tasks (15 episodes total).
- Test-time learner: P=16 same-source demonstrations, 200 Adam steps, batch 4,
  eight practice-time samples per demonstration, 32 learned prompt rows.
- Conditions: paired `real` and `scrambled` fits. The harness also records copy,
  target-VAE, and frozen no-prompt controls.
- Readout: three Euler steps on FLUX.2's schedule (`f3`).
- Model and VAE weights remain frozen; only the per-episode 1.97M-number prompt
  residual/offset state is fitted.
- Target TPU: the exact Kristine-owned Spot v4-8 in `config.json`. No resource
  is provisioned, restarted, reset, or reimaged by these scripts.

All heavy inputs are copied byte-for-byte to the dedicated `us-central2`
prefix in `config.json` and downloaded into `/dev/shm` before the final TPU
occupancy gate. The source snapshot is the authoritative JFlux source closure
plus one deployment-only change: `inference/zone_assets_v1.py` recognizes the
dedicated central2 input prefix. This changes artifact routing only and is
included in `SOURCE_SHA256SUMS` and the harness's recorded source hashes.

Launch order:

1. `prepare_bundle.sh`
2. `stage_inputs.sh`
3. `start_attached.sh` (waits after CPU-only preflight)
4. `approve_launch.sh` (fresh full inventory, exact API and VFIO checks)

The attached remote script periodically syncs partial results and logs to GCS,
permitting a reviewed resume after Spot preemption.

Current operational status is recorded in `STATUS.md`. The initial candidate
failed TPU initialization twice before model load or output, so the run is
staged/blocked pending a different safe TPU; it is not being reported as live.
