# Infrastructure history — train run 6

## 2026-08-08 02:30–02:40 — launch

- Queue built: sweep `2026-08-08-02-46_run6`, 36,000 markers (120 configurations x 300 seeds),
  all group-readable. Launch commit `5df4463` (code committed BEFORE submission).
- Launch gates passed first: 24 unit tests + the full-scale two-phase simulation (120 synthetic
  configurations -> 108 truncated at the 20-seed floor, 8 stopped at 100, 4 winners complete at
  300, 0 invariant violations at every wave).
- Frozen bar recomputed from train run 5's records at launch: 38.6412 (beta=1000, n=300) —
  reproduces the published value exactly; sha256 pinned in FROZEN_BARS.json.
- The open cpu/nolim pools were full with the DRAINING learning-rate-addendum sweep (stopped
  2026-08-08 01:53; its jobs exit as their ~18 h in-flight runs finish over ~21 h), so the launch
  went to the reservation `sl5nw_156` (jaguar03 + puma01, ends 2026-08-19):
  - canary `6534619` (4x1, puma01): claimed in id order, AllocCPUS=4, workers ~99% CPU,
    RSS ~745 MB inside the 1500M/cpu grant — passed before anything else was submitted;
  - fleet `6534620` (64x1), `6534621` (64x1), `6534622` (26x1) on puma01 — 158 threads,
    memory-capped at 1500M/cpu (158 x 1500M = 237,000 of the 248,000 MB allocatable);
  - `6534623` (14x1, jaguar03, --gpus-per-node=0) — jaguar03 had 30 free threads of 222 (192 held
    by the CleanRL session's reservation jobs); 14 taken, the owner's 16-CPU headroom left free.
  - monitor_loop `6534624` (nolim, 20-day walltime); all ids in
    `slurm/submitted_jobids_2026-08-08-02-46_run6_sl5nw.txt`.
- 172 worker slots in flight within 10 minutes, all four arms claiming (57 alg1 / 53 alg2.1 /
  35 alg2.2 / 27 alg2.3), failed/ empty.
- 20-minute session cron armed in the same turn as the submission; each tick also (a) watches the
  addendum drain and closes it out when its running/ empties, and (b) tops the open cpu/nolim
  pools up for run 6 (cap minus the 16-CPU owner headroom) as the old sweep's jobs free them,
  via plan_jobs.py --submit.
- Collaborator packet generated from the addendum's proven packet (paths, sweep id and job-name
  prefixes regenerated; smoke test rewritten to exercise the four run-6 arms on the PointMaze
  task). Handoff path: `for_collaborator/README.md`.

## 2026-08-08 02:50 — first tick: addendum collaborator cancel handled; no open-pool top-up yet

- The addendum collaborator cancelled her 22 jobs at 01:56 (details in that run's
  infra_history.md); the 478 orphan markers her cancel produced were re-archived before any
  worker reclaimed them.
- Top-up arithmetic: my cpu counter reads 542/400 (draining addendum ~384 + run-6's puma01
  reservation jobs 158, which count into the counter while riding above the cap), so
  plan_jobs correctly submits nothing. My open-cpu room appears as the addendum drains
  (up to 400 − 158 − 16 = 226 threads); nolim similarly (80 − 2 monitor − 16 = 62 as its 62
  free). The ~480 CPUs her cancel freed sit under HER caps — available immediately if she
  submits run-6 workers from for_collaborator/.

## 2026-08-08 04:15–04:35 — user-ordered restart: full cpu -> nolim -> reservation fleet

The user ordered: kill the draining addendum jobs AND run 6's reservation-only fleet, re-queue
run 6's claimed work, and resubmit at full width. Executed in order:

1. Addendum: monitor 6534388 cancelled FIRST (so its orphan recovery could not re-pend), then the
   17 worker jobs — every id verified against that run's own id file before its scancel. 440
   partial records archived to `killed_attempts_*_final-cancel/`; the 440 killed markers went to
   `stopped_unlaunched/` (final state: 2,303 done + 15,864 stopped + 8,833 pruned = 27,000).
2. Run 6: worker jobs 6534619–6534623 cancelled (monitor_loop 6534624 kept); 51 partial records
   archived; all 172 claimed markers returned to pending/ — the queue read 36,000 pending again.
3. VERIFIED UNTOUCHED before proceeding: the Atari CleanRL run-2 sweep's 14 jobs (own id file;
   all RUNNING, incl. the 192-CPU jaguar03 job) and the other session's fossil3.
4. Full resubmission, open partitions first: plan_jobs submitted 16 jobs / 444 slots (cpu 384 on
   affogato02+bigcat01-06, nolim 60 on slurm2/slurm3), then the reservation bucket last
   (puma01 64+64+30, jaguar03 14 with the 16-CPU headroom kept). Fleet: 21 jobs, 616 worker
   slots, all RUNNING, AllocCPUS == ntasks on every new shape.

Cost accepted by the user for the immediate width: the 440 addendum in-flight runs (up to ~18 h
each) and 172 run-6 runs (~2 h each) were killed; the run-6 ones re-run from scratch, the addendum
ones never run.
