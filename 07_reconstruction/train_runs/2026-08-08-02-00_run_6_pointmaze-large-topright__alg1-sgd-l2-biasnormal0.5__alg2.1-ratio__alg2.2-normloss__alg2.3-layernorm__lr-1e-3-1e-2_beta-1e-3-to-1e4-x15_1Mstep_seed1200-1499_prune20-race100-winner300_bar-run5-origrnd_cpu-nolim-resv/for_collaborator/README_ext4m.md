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

1. **Runs are resumable.** Each run writes a model checkpoint every 0.5M steps. When a worker's
   job has too little walltime left for the next 0.5M-step chunk, the trainer exits with code 3
   and the worker moves the marker **back to pending/** — a log line like
   `rc=3 ... -> pending` is NORMAL operation (a suspended run waiting for its next claimer),
   not a failure. Only `-> failed` lines are problems.
2. **A claim needs only 12 h of remaining walltime** (one chunk), not a whole run — the guard
   is set in the worker; you do not need to configure anything.
3. **After your first job of any new shape starts**, check `sacct -j <id> -X -o JobID,AllocCPUS`
   shows AllocCPUS == the ntasks the plan printed (as before).

## What to watch

- Your jobs reach RUNNING (not pending forever) and their claims appear in the job log
  (`for_collaborator/logs/e4mclb*_<jobid>.log`, lines `claimed NNN_of_900_...`).
- Anything odd: write a report into `problems/open/` exactly as before (include `MARKER: <name>`
  lines when a specific run is affected — the monitor re-pends them automatically; the
  checkpoint means nothing is lost).

The owner's standing monitor handles orphan recovery, top-ups, and the completion sentinel; your
only job is submitting workers and reporting problems.
