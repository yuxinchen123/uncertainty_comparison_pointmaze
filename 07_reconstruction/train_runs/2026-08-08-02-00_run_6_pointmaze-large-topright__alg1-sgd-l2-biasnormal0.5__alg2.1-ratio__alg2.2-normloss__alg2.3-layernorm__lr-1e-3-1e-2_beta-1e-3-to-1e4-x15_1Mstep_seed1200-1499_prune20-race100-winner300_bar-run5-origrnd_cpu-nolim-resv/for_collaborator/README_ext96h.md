# 96-hour extension sweep (ext96h) — collaborator instructions (2026-08-13, REPLACES README_ext4m.md)

The 4M checkpoint/resume sweep was retired before any run completed; this is its replacement.
Same three configurations and seeds, but every run is now ONE FRESH 96-HOUR ATTEMPT: it trains
until its job's 96-hour walltime (10M-step cap that is never reached) and its record is complete
there. No checkpoints, no resume, no suspend codes. Everything not mentioned here (id-file
discipline, problems/ channel, never blanket-cancelling) is as in README.md.

## Submit workers

```bash
bash ext96h_launch_workers_collaborator.sh
```

Reads the sweep id from `slurm/EXT96H_SWEEP_ID.txt`, sizes jobs under YOUR caps, submits with
the `e96clb` prefix at exactly `--time=4-00:00:00`, appends ids to
`for_collaborator/submitted_jobids_<sweep_id>_$USER.txt`, and stops at YOUR 300-run share
(ledger `ext96h_slots_<sweep_id>_$USER.txt`; `budget remaining 0` = fully queued). Some jobs
pend on purpose — they are the second-wave ladder and start as nodes free after 96 hours.

## What to expect

1. Each worker slot runs exactly one run for the job's whole 96 hours; your jobs END at their
   walltime with every run complete — that is success, not a failure.
2. `-> done` is the only good end; `-> failed` lines are real problems (report them with
   `MARKER: <name>` lines in problems/open/ — a re-pended run restarts fresh by design).
3. After your first job of a new shape starts, check AllocCPUS == ntasks via sacct (as before).
