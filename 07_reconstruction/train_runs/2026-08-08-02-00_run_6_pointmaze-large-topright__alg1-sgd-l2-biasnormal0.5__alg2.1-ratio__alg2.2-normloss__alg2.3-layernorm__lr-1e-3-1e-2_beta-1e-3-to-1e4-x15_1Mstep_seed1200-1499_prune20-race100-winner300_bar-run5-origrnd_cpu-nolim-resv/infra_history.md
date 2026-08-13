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

## 2026-08-13 02:36-02:55 — ext4m launch (the 4M extension sweep)

- Queue built: sweep `2026-08-13-02-36_run6ext4m`, 900 markers (3 configurations x 300 seeds x
  4M steps, seeds 1500-1799, resumable via 0.5M-step checkpoints; design slurm/EXT4M_DESIGN.md).
- Launch gates: 6 queue-convention tests + 4 checkpoint tests incl. the full suspend -> resume ->
  auto-delete -> complete lifecycle on real trainings; all passed. Launch commits a4a0da4+3476207.
- Canary 6536743 (8 workers, bigcat01): RUNNING, AllocCPUS==8, 8 claims across all three
  configurations, sstat showed CPU accumulating (trainer stdout is srun-buffered — not a stall).
- Full submission 6536744-6536757: 12 more cpu jobs (affogato02, bigcat02-06), 1 nolim
  (heartpiece 20), and the puma01 reservation job 6536757 at the NEW 2-CPUs-per-worker shape —
  AllocCPUS 158 = 79 workers x 2 verified. r6e4m12 (30 cpu) pended briefly on QOSMaxCpuPerUser.
  No jaguar03 (user's instruction). Monitor loop 6536758 (nolim); 20-min cron f1ad63c0 armed.
- Total own fleet at launch: ~449 worker slots running + 30 pending, vs 900 runs.

## 2026-08-13 13:03-13:20 — ext4m fleet swap to the final optimized design

- Trigger: checkpoint coverage 444/450 (98.7%) at 13:03, inside the 10-hour research budget.
- Cancelled the 39 first/second-wave job ids from the owner id file (monitor 6536758 kept);
  requeue re-pended all 450 markers (446 resume from their 0.5M checkpoints under the
  optimized trainer, bit-exact; ~6 checkpoint-less stragglers restart, ~58 CPU-hours lost).
- Resubmitted 22 jobs / 600 slots (ledger released the cancelled slots, budget exactly spent):
  FINAL design — loop workers for the full 96 h walltime (claim guard 12 h, trainer suspend at
  the wall), throughput-research switches applied (APPLIED_CHANGES.md: foreach polyak +
  torch reward combine + interop=1, −3.2/−3.3% wall), cpu + nolim only, NO puma01, NO
  reservation, 5 unpinned pending ladder jobs as replenishment.
- The 1M sweep reached SWEEP_COMPLETE earlier today (11:03 tick); final Table 60/Figure 18
  committed as e1bc1f0.

## 2026-08-13 13:4x-14:1x — ext4m retired, ext96h fresh start (user order)

- User rule: no checkpoint-resumed runs (the buffer-tail resume is not faithful); refresh runs
  only. Cancelled the whole ext4m fleet incl. its monitor (all ids from the sweep's own file),
  DELETED queue/data/checkpoints of 2026-08-13-02-36_run6ext4m (0 runs had completed), removed
  the ext4m scripts from the working tree (git history keeps them), marked the SWEEPS.md row.
- Built sweep 2026-08-13-14-00_run6ext96h: 900 fresh markers, 10M-step cap, 96-hour
  single-attempt runs (design slurm/EXT96H_DESIGN.md). Gates: 7 + 2 tests passed.
