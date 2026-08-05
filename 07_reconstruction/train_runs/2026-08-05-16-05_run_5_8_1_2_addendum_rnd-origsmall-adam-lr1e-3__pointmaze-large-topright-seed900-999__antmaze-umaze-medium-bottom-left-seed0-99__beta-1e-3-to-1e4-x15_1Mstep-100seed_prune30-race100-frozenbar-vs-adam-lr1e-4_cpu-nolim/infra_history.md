# Infrastructure history — Adam learning-rate 1e-3 addendum

## 2026-08-05 16:57 — launch

- Sweep `2026-08-05-16-05_lr1e3`: 4,500 runs (45 configurations x 100 seeds), 1,000,000 env steps
  each, cpu and nolim partitions only.
- 15 worker jobs submitted, ids 6533853–6533867, plus the 20-minute monitoring loop 6533868. All
  recorded in `slurm/submitted_jobids_2026-08-05-16-05_lr1e3_sl5nw.txt`.
- Every job started immediately and `AllocCPUS` equalled its `--ntasks` on all five shapes
  (32, 30, 24, 12, 8), so `--ntasks-per-core=2` held everywhere.
- 416 worker slots; 411 runs claimed within three minutes, 0 failed.
- Launch verification before submitting: two short runs (2,500 steps) built by the run's own
  `worker.build_cmd` from real queue markers — one PointMaze, one AntMaze UMaze — both exited 0 and
  wrote records with `completed: true`, the right `env_setup`, `rnd_lr` 0.001 and `mse_mean`
  readout.

## 2026-08-05 17:00 — the monitor's top-up submitted the fleet a second time; sizing fixed

- Symptom: three minutes after launch the id file held 36 ids instead of 16, and 23 extra cpu jobs
  sat `PENDING (QOSMaxCpuPerUserLimit)`.
- Cause: `plan_jobs.py` sized the pools from the partition QOS counter alone
  (`scontrol show assoc_mgr qos=cspartcpu` → `MaxTRESPU=cpu=400(<used>)`). On the cpu partition that
  counter read **`400(0)` while 384 of this user's CPUs were RUNNING there**, so the monitor's
  top-up saw the whole cap as free and planned the entire fleet again. Slurm's own admission control
  was correct throughout — it held every extra job at the cap — so nothing over-ran; the cost was 23
  useless queue entries that would have recurred every 20 minutes. The nolim counter was accurate at
  the same moment (`80(66)`), so this is specific to that counter, not to the reading code.
- Fix: `pool_room()` now takes the LARGER of the QOS counter and this user's own running-plus-pending
  CPUs in that partition, summed from `squeue -u $USER -t R,PD -p <partition> -o %C`. Pending jobs
  count, so a top-up can never submit the same work twice. Verified straight after: cpu reads 398 in
  use → 0 threads to fill, nolim 66 → 0, plan is 0 jobs.
- The 23 duplicates were cancelled by id, taken from this run's own id file and each re-checked as
  still PENDING at the moment of cancellation. No running work was touched.
