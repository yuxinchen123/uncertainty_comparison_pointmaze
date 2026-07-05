# Run 3.1.2 — arm-B reshape event (2026-07-02, ~14:3x)

## What happened
Arm B (16 tasks × 1 cpu) was launched without `--ntasks-per-core=2`. On every bucket except the
gpu-allowlist (`rope-scale`, which already had the flag from its hand-resubmission), Slurm (a) charged
each 1-cpu task a FULL core — `AllocCPUS=32` per job, 2× the intent — and (b) bound the 16 tasks
two-per-hardware-thread: every worker ran at ~49.8% CPU while half the allocated cores sat idle.
Confirmed by `ps` inside job 6270469 (arm A workers: 197% CPU; arm B: 49.8%) and reproduced at small
scale on puma01 (probe jobs 6272963/64/69: `-n4 -c1` charged 8 CPUs with two tasks sharing one thread;
`--ntasks-per-core=2` charged 4 with one task per thread; `-n4 -c2` charged 8 with one core per task —
so the arm-A shape was never affected). Rules updated: global `cluster-slurm.md` (new hyperthreading
section), project `slurm-submission.md` §7, `run-id-and-logging.md` (archive-before-requeue).

## The surgical fix (only arm-B ids from this run's own id file; arm A and rope-scale untouched)
1. Cancelled 17 jobs: 11 running fat (6270469/71/73/75 spec-decode, 6270481/83 lr-warmup,
   6270487/89/91/93/95 ffn-gate) + 6 never-started pending (6270477/79, 6270485, 6270497/99, 6270501).
2. Their 176 in-flight configs (identified from worker logs: claimed-but-not-finished; cross-check
   176 + 80 rope-scale in-flight = 256 running markers, exact) were:
   - partial JSONs (all 176 had checkpoints, `completed=false`) ARCHIVED to
     `data/<armB-sweep>/killed_attempts_2026-07-02-14-*/` — the lost-work evidence for the infra
     analysis survives the reruns, which overwrite the live path on their first checkpoint;
   - queue markers moved `running/` → `pending/` (pending 644→820).
3. Resubmitted 17 jobs with `--ntasks-per-core=2` (same SWEEP_ID, same buckets: 6 jaguar03 + 3 puma01
   via reservation, 8 cpu-partition): ids 6273002–6273011, 6273017–6273023, appended to
   `submitted_jobids_<armB-sweep>.txt`.
4. Verified: all 17 `AllocCPUS=16` and RUNNING; worker spot check on jaguar03 shows arm-B trainers at
   ~99% CPU each (one full thread per worker, sibling-paired cores).
5. `worker.py` log lines now carry timestamps (claim/finish times recomputable from logs alone; the
   pre-fix attempts' timing lives in sacct Start/End + archived checkpoint `runtime_seconds`/mtimes).

## Cost / benefit
Lost: ~176 in-flight attempts averaging ~5 trained hours each (≈880 run-hours; partial curves kept in
the archive). Gained: ~88 idle-but-allocated cores returned to use for the remaining ~3.5 days, arm-B
workers at full-thread speed (~2× the broken binding), and a uniformly correct 16×1 shape for the
infrastructure comparison.

## Timing attribution rules (for the infra analysis)
- Job level: sacct `Start`/`End` per job id (permanent).
- Attempt level: each JSON's `runtime_seconds` is that attempt's own clock; killed attempts live in the
  archive dir; reruns never mix clocks with killed attempts.
- Arm-B eras: BEFORE ~14:3x, non-rope-scale arm-B paces reflect the broken half-thread binding — the
  infra analysis must split arm-B data at this boundary (job id ≤ 6270522 fat era; ≥ 6273002 fixed era)
  and use only the fixed era for the headline 8×2-vs-16×1 comparison (rope-scale was correct throughout).

## Cancellation safety
Only ids from `slurm/submitted_jobids_<sweep>.txt` files are ever cancelled; never blanket cancels.

# Follow-up sweep (unit norm, ridge 1e-8) — WaitTime kill event + user stop (2026-07-04)

## Event 1: srun WaitTime=3600 killed 94 mid-flight runs (~09:20-15:25)
The sweep's queue drained in one wave (256 workers >= 250 configs), so each worker exits after its one
run — and this cluster's `slurm.conf` `WaitTime=3600` makes srun kill a step's remaining tasks 3600 s
after its FIRST task exits. 13 of 16 jobs died that way (3 at exactly 1 h — their spare workers exited
at start; 10 at ~1 h after their fastest worker finished); 1 job completed inside the window; job
6314802 (ai05) had no finisher and kept running. 140/250 runs completed; 0 model-side failures.
Recovery at ~16:31: 94 killed attempts' partials archived to
`data/<sweep_id>/killed_attempts_2026-07-04-16-34/`, markers requeued, 6 recovery jobs submitted
(6380501-06) with the worker scripts fixed to `srun --wait=0`.

## Verification of the fix (controlled probe, ~16:53)
Jobs 6382110/6382111 (4 tasks; task 0 exits at t=0, tasks 1-3 run 300 s), logs in `logs/wait_probe/`:
`srun --wait=30` killed tasks 1-3 exactly 30 s after task 0 exited (job FAILED 9:0, step CANCELLED —
the same signature as the event); `srun --wait=0` ran all tasks to natural completion (COMPLETED 0:0,
elapsed 5:00). So `--wait=0` = unlimited wait / never kill, confirming the man page.

## Event 2: user stop (~17:00) — enough signal at 140/250
All 7 live jobs (6380501-06 + 6314802) cancelled, ids taken ONLY from
`slurm/submitted_jobids_<sweep_id>.txt`. Final coverage 140/250 (beta=1e-4: 20, 1e-3: 31, 1e-2: 27,
1e-1: 31, 1e0: 31 seeds), 0 failed. The interim answer stands as final: unit norm at replica-level
rho does not recover the replica (best 22.99 +/- 5.75 vs 53.81 +/- 5.95); Hypothesis 2 revived.
Status after the stop: `queue/<sweep_id>/running/` still holds 110 markers (16 ai05 first-wave + 94
recovery attempts, all killed by the cancel); `local/` holds 140 completed + 16 ai05 partials; the 94
recovery attempts died before their first checkpoint (~20 min in) so they left no JSONs.

## Resume path (if more seeds are ever wanted)
Archive the 16 ai05 partials (completed=false) out of `local/` to a new
`killed_attempts_<ts>/`, move all 110 `running/` markers back to `pending/`, and submit jobs with the
(already fixed) `worker_16x1.slurm`; ids append to the same id file. Analyses need no change (they
read `completed=true` only).
