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

## 2026-08-05 18:30 — a slow tail of 24 runs on slurm4; processes healthy, no action taken

- Observation: at 88 minutes, 392 of the 416 in-flight runs had written their first checkpoint
  (step 50,000) and 16 had already reached 100,000, but 24 had written nothing. All 24 belong to one
  job, 6533866 (24 tasks, node slurm4, nolim partition).
- Checked and ruled out:
  - **Not hung.** `srun --jobid=6533866 --overlap` shows all its workers at 99.9% CPU, 1 h 28 m
    elapsed, resident memory about 715 MB each, spread across distinct hardware threads. Node load
    average 32.00 against `CPUAlloc=32` — fully busy, not oversubscribed.
  - **Not a thread-packing asymmetry.** Every one of the 32 worker processes on slurm4 sits on a
    hardware thread in the range 0–15 / 24–39, and the node's sibling map pairs *n* with *n*+24, so
    all 16 physical cores in use carry exactly two tasks. Job 6533866 (24 tasks) and job 6533867
    (8 tasks, same node) are packed identically.
  - **Not a queue or infrastructure failure.** `failed/` is empty, no orphan was detected, no
    collaborator problem report exists, and the invariant checker passes every tick.
- Not explained: job 6533867's 8 runs on the same node checkpointed at 46–49 minutes while job
  6533866's 24 have not at 88, under what appear to be identical conditions. The missing runs skew
  towards AntMaze Medium (the heaviest environment: 1000-step episodes) — about 14 Medium, 8 UMaze
  and 3 PointMaze — which accounts for part but not all of the gap.
- Measured pace from the records that reached 100,000 steps: 5,369 s for 100,000 steps, i.e. about
  **15 h per 1,000,000-step run**, slightly better than the 17–20 h projected from train runs 5
  and 1.2.
- Decision: no action. The runs are computing at full speed and nothing is lost. Threshold for
  looking again: if those 24 still have no first checkpoint at about 2 hours (roughly 19:00), that
  is a genuine anomaly rather than a slow tail.

## 2026-08-05 18:45 — the slow tail explained: one job filled one socket; re-placed across both

The 18:30 entry above left the split unexplained. It is now measured and fixed.

- **Cause.** slurm4 is a 2-socket Intel Xeon E5-2670 v3 (12 cores per socket; NUMA node0 = CPUs
  0–11 and 24–35, node1 = 12–23 and 36–47). Slurm packed job 6533866's 24 tasks onto **socket 0
  alone** — all 12 of that socket's cores double-loaded, 24 workers sharing one memory controller —
  while the 8 tasks of job 6533867 sat on socket 1 with a whole controller between them. Counted
  directly from each worker's `physical_package_id`: **socket 0 carried 26 worker processes,
  socket 1 carried 8.**
- **Why it costs so much.** This workload is memory-latency-bound, so bandwidth per task, not
  cycles, sets its speed. Both jobs showed 99.9% CPU and identical per-task memory, yet after
  1 h 43 m every one of 6533867's 8 runs had reached 100,000 steps while not one of 6533866's 24
  had reached 50,000 — a slowdown of more than 2x, projecting to over 35 h per run instead of
  about 17 h.
- **Fix in the sizing code.** `plan_jobs.py` now emits `--ntasks-per-socket = ceil(ntasks /
  sockets)` on every multi-socket node, so a job can no longer be packed onto one socket.
- **Fix on the cluster.** Job 6533866 was cancelled (its id taken from this run's own id file, and
  re-checked as RUNNING first) and the freed room resubmitted as job 6533925, 20 tasks with
  `--ntasks-per-socket=10`. Verified after it started: socket 0 now carries 12 worker processes and
  socket 1 carries 18, against 26/8 before.
- **Cost and benefit.** 24 runs lost about 1 h 45 m each (roughly 42 core-hours; no completed
  record was destroyed, since none of the 24 had reached its first checkpoint). Against that, those
  runs would each have taken over 35 h rather than about 17 h. `requeue_orphans.py` returned all 24
  markers to `pending/` — queue went pending 4084 -> 4088, running 416 -> 412 — and the kill and
  requeue are recorded in `slurm/killed_orphans_<sweep id>.txt`.
