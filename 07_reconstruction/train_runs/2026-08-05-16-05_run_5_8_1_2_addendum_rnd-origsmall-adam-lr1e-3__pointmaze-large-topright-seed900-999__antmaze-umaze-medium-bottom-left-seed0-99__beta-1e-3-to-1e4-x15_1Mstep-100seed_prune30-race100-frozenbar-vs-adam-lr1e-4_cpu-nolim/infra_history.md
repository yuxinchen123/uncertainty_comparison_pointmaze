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

## 2026-08-05 19:10 — two sizing bugs in plan_jobs.py, both found by the collaborator, both fixed

`yuxinchen` joined this sweep at about 19:00 (446 worker slots) and filed two problem reports. Both
were real bugs in `slurm/plan_jobs.py`, both are fixed, and both reports are in
`for_collaborator/problems/resolved/` with the verification appended.

1. **The pool size was read from the wrong user's row.** `scontrol show assoc_mgr qos=<qos>
   flags=qos` prints one `MaxTRESPU=` line per user of the QOS, all with the same cap, and
   `pool_room()` used a bare `re.search` — which returns whichever user is printed first. `sl5nw` is
   first in both records, so the owner always read the right row and never noticed; the collaborator
   read the owner's usage and was told her free nolim room was 0 while she in fact had the whole
   pool. The error direction is safe (under-submit), so nothing was ever over-run; the cost was an
   idle pool. Fixed by anchoring on the running user's own `<user>(<uid>)` heading. Verified live:
   the old search returned 0 (cpu) and 64 (nolim), both `sl5nw`'s; the anchored search returns
   398/64 for `sl5nw` and 382/64 for `yuxinchen`.
2. **`--ntasks-per-socket` made the second job on a half-used node pend forever.** The pin added at
   18:45 is right for a job that takes a whole node, but after a 24-task job takes 6 cores on each
   socket of a 12-core-per-socket node, a 22-task job pinned to 11 per socket needs 6 cores on each
   side again — one more than the 11 that remain — so Slurm holds it on `(Resources)`. She hit 8
   such jobs (154 idle slots) and cleared them by resubmitting the same shapes without the pin.
   Fixed by emitting the pin ONLY for the first job planned on a node that was completely idle. That
   keeps the protection it was added for (a single-socket job ran at under half speed here) without
   the over-constraint: once the first job has taken an even split, the free cores it leaves are
   already spread across both sockets. Verified live over the current candidate nodes — idle
   2-socket nodes get the pin on job 1 and not job 2; used and single-socket nodes get none.

The owner fleet was unaffected by either bug: `sl5nw`'s QOS row is first, and no owner job ever
pended from the socket pin (the pools were at cap, so no top-up job was planned in that window).

## 2026-08-06 12:10 — walltime guard added to the worker (the 2026-08-09 cliff)

- Trigger: the user asked why a job had ended and whether workers should not simply keep taking new
  runs. They should, and they do — job 6533853's log shows 32 workers, 64 claims, 32 finishes and
  0 exits, each worker claiming its next run in the same second it finished the last. The job that
  ended was `r812-s-nolim1` (6528782), the LAST run-8.1.2 worker job, whose 30 workers all logged
  `queue empty; exiting` because that sweep was stopped on 2026-08-05. It freed the 32 nolim threads
  the top-up then refilled with 6534341 and 6534342.
- The real exposure the question surfaced: the 13 cpu jobs carry `--time=4-00:00:00` (the cpu
  partition maximum) and started 2026-08-05 16:57, so they end **2026-08-09 16:57**. At a measured
  median of 15.8 h per run, workers would keep claiming right up to the deadline and roughly 384
  in-flight runs would be killed at once. This run writes no checkpoints, so each restarts from
  zero: about 3,000 core-hours burned and then repeated, plus 384 markers hitting requeue_orphans in
  one tick.
- Fix: `slurm/worker.py` now refuses to claim when the job has less than `WORKER_REQUIRED_HOURS`
  (default 20 h) of walltime left, logs why, and exits. The job then ends early, its pool room frees,
  and the monitor's top-up submits a fresh full-walltime job that picks the work up. Same idle time
  as being killed, but nothing is lost and nothing is redone.
- The guard FAILS OPEN: an unreadable job end time claims anyway. It is an optimization, not a
  correctness rule — failing closed on a scontrol hiccup would idle the whole fleet.
- Tests: `slurm/test_worker_walltime_guard.py`, 7 cases (plenty of time, too little, the exact
  boundary, configurability, fail-open, the default covering the measured run length, and a 20-day
  nolim job never being blocked). All pass; the 35-test launch-gate suite still passes.
- Scope: worker.py is read at process start, so the 924 runs already in flight are unaffected. The
  guard takes effect for every job submitted from the next top-up onward, which is well before the
  2026-08-09 deadline.

## 2026-08-06 20:55 — the sweep grew to 300 seeds and a second learning rate, without stopping

The user asked for two changes to a sweep that was already running with 926 workers: raise the seed
target per configuration from 100 to 300, and add Adam 1e-2 alongside Adam 1e-3 on the same three
environments and the same 15 bonus weights. Both were applied to the LIVE queue. No job was
cancelled, no job was resubmitted, and no completed run was repeated.

- **Why it works.** A worker calls `claim()` in a loop, and `claim()` re-lists
  `queue/<sweep_id>/pending/` every time. A marker dropped into `pending/` is therefore picked up by
  the workers already running — no restart, and no code change has to reach them. The worker code
  itself is untouched.
- **What `slurm/extend_queue.py` did:** created 22,500 markers for work the queue had never held
  (45 Adam 1e-2 configurations × 300 seeds = 13,500, plus seeds 100–299 of the 45 Adam 1e-3
  configurations = 9,000), and renamed the 2,696 markers still sitting in `pending/`. It touched
  nothing in `running/` (924), `done/` (880), `failed/` or `pruned/`. Result: 25,196 + 924 + 880 =
  27,000 units, each present exactly once, verified by reading every marker back.
- **Why the renames were needed.** `claim()` takes the 32 lexically smallest names, which is a
  seed-ordered frontier ONLY while every id is zero-padded to one width. The old queue padded to 4
  digits (run_total 4,500), the extended one to 5 (27,000), and `00000` sorts before `1676`. Left
  mixed, every new marker would have sorted ahead of every old one and starved the Adam 1e-3 arm.
- **Why the renames are safe.** Each marker is moved OUT of `pending/` into `staging/`, rewritten
  there, and moved back under its new name, one at a time. So a unit is never visible under two
  names at once (no double claim), and `pending/` never dips by more than one (no worker sees an
  empty queue and exits). A marker stranded in `staging/` by a crash is returned to `pending/` by the
  next invocation, so re-running is always the repair. The whole pass took 104 seconds and reported
  `claimed_mid_rename 0`.
- **Ordering after the change.** The lexically smallest pending marker is now `00045…lr0.01…seed900`
  — seed 0 of the new arm, because ids 0–44 (seed 0 of the old arm) are already done. The new arm
  therefore catches up to the running arm's frontier first and the two then advance together seed by
  seed, so both reach the 30-seed decision floor at about the same time.
- **Rate at which the new work starts.** All 924 slots were busy at the moment of the change, and a
  slot only frees when its run ends (~18 h median, ~52 completions/hour across the fleet), so the
  fleet turns over onto the new markers within about 18 hours rather than instantly. This is
  expected, not a fault.
- **Frozen bars unchanged.** Both learning rates race against the same per-environment bars in
  `slurm/FROZEN_BARS.json` (sha256 unchanged), so every decision already logged stays verifiable and
  the new arm is judged by exactly the same standard.
- **Monitor re-armed.** `monitor.sh` now passes `--n_target 300`; it is read fresh each tick, so that
  took effect immediately. `monitor_loop.slurm`'s completion check (`N_CONFIGS`) is baked into the
  spooled copy of a running job, so job **6533868** was replaced by **6534388** (submitted first,
  confirmed RUNNING on heartpiece, only then was 6533868 cancelled BY ITS OWN ID from this run's id
  file). Its first tick reports "0 decided configurations of 90" and all invariants holding.
- **Report readability.** `20_mins_monitoring/monitoring_report.py` gained a learning-rate column:
  with both rates in one sweep, an environment's table has 30 rows and the bonus weight alone no
  longer names a row.
- **The run folder keeps its name.** Renaming it would break the `RUN_DIR` of every running worker.
  The folder name records how the run started; `experiment_background.md` records what it became.
- Gates re-run at the new scale before applying: 53 unit tests pass (9 new in
  `slurm/test_extend_queue.py`), and `slurm/simulate_truncation.py --full` passes with 90
  configurations, 45 truncated at the 30-seed floor, 45 survivors at 300, 0 invariant violations.
  Code committed as `3891bdb` BEFORE the queue was touched.

## 2026-08-08 01:53 — sweep stopped (drain); pending archived, in-flight kept

The user stopped the sweep to free the CPU pools for train run 6. All 14,946 pending markers
moved to `queue/2026-08-05-16-05_lr1e3/stopped_unlaunched/` (0 lost to claim races); the 924
in-flight runs — at this point all Adam 1e-2 — finish and are kept; no job cancelled, ours and
the collaborator's drain on their own as workers find the queue empty (~21 h). State at the
stop, per arm and environment, in `SWEEP_STOPPED.md`. Monitor 6534388 stays up through the
drain (top-up submits nothing with pending empty) and is stopped BY ITS OWN ID when running/
reaches 0, followed by one final regeneration of the analysis tables. Collaborator notified via
`for_collaborator/SWEEP_STOPPING_NOTICE.md`.
