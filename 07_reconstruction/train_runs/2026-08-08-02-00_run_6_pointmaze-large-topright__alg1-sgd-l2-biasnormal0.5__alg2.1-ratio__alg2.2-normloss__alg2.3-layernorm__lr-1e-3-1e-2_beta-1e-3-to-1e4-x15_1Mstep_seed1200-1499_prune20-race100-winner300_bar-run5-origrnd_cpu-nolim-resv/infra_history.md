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
