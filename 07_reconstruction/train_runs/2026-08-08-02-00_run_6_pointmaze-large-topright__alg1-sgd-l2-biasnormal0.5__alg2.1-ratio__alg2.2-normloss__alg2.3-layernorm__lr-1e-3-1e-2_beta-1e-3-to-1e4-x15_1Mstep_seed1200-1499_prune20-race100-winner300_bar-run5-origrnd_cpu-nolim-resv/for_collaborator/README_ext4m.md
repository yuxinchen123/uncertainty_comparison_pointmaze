# 4M extension sweep (ext4m) — collaborator instructions (added 2026-08-13)

A SECOND sweep now runs in this same run folder, alongside (and after) the 1M sweep you already
know: three configurations × 300 fresh seeds × **4,000,000 steps** = 900 runs (run-5 original
RND at β=1000; algorithm 2.3 at lr 0.01, β=30; the ground-truth bonus min(1, 1/√n) at β=1).
Everything below is what changes for you; everything not mentioned (id-file discipline, the
problems/ channel, never blanket-cancelling) is exactly as in README.md.

## Submit workers

```bash
bash ext4m_launch_workers_collaborator.sh
```

One command: it reads the sweep id from `slurm/EXT4M_SWEEP_ID.txt`, sizes jobs from the live
cluster state under YOUR caps (cpu 400 / nolim 80, minus 16 headroom), submits with the `e4mclb`
job-name prefix, and appends every id to
`for_collaborator/submitted_jobids_<sweep_id>_$USER.txt`. Re-run it any time to top up; it
plans nothing once `SWEEP4M_COMPLETE` exists and never submits more worker slots than there are
runs left to claim.

## What is DIFFERENT from the 1M sweep — please read

1. **Workers are ONE-SHOT.** Each worker claims exactly one 4M run, finishes it, and exits; the
   job ends when all its workers are done (~60–76 h, inside the 96 h cpu walltime). Nothing to
   configure — but expect your jobs to END on their own after roughly three days; that is
   normal, not a crash. Some of the jobs the launcher submits are UNPINNED and sit PENDING in
   the Slurm queue on purpose: they are the replacement ladder and start as nodes free.
2. **Your share of the workload is 300 of the 900 runs** (the owner takes 600). The launcher
   enforces this through a slots ledger next to your id file
   (`ext4m_slots_<sweep_id>_$USER.txt`) — one `jobid ntasks` line per submission. Re-running
   the launcher never exceeds your share; if it prints `budget remaining 0`, your share is
   fully queued and you are done submitting.
3. **Runs are resumable.** Each run writes a model checkpoint every 0.5M steps; a rare log line
   `rc=3 ... -> pending` is a run suspended at a checkpoint waiting for a fresh claimer (a
   safety net), not a failure. Only `-> failed` lines are problems.
4. **After your first job of any new shape starts**, check `sacct -j <id> -X -o JobID,AllocCPUS`
   shows AllocCPUS == the ntasks the plan printed (as before).

## What to watch

- Your jobs reach RUNNING (not pending forever) and their claims appear in the job log
  (`for_collaborator/logs/e4mclb*_<jobid>.log`, lines `claimed NNN_of_900_...`).
- Anything odd: write a report into `problems/open/` exactly as before (include `MARKER: <name>`
  lines when a specific run is affected — the monitor re-pends them automatically; the
  checkpoint means nothing is lost).

The owner's standing monitor handles orphan recovery, top-ups, and the completion sentinel; your
only job is submitting workers and reporting problems.
