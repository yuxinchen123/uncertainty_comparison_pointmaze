# Train run 5 (set baseline) — collaborator packet

You (a collaborator, e.g. `yuxinchen`) can add worker jobs to this already-launched sweep, using
YOUR OWN Slurm caps, to speed it up. You never build the queue, never prune, never requeue, and
never touch another user's files — you only add workers and report problems.

## One-time check

The shared env must be reachable and the queue readable:
```
bash smoke_test.sh
```
This runs a few short canary configs under your uid and confirms `train.py` starts. If it fails,
tell the owner (do not debug shared state).

## Add workers

```
bash launch_workers_collaborator.sh
```
- Submits 32×1 / 16×1 worker jobs to the **cpu** and **nolim** partitions ONLY (user directive
  2026-07-20: NO gpu partition, NO gnolim, NO reservation).
- Trims the count live against YOUR caps (cpu 400 / nolim 80), keeping ~16 CPUs of headroom per
  pool, and against the remaining queue depth.
- Records every job id in YOUR OWN file `submitted_jobids_<SWEEP_ID>_<you>.txt`. That file is the
  ONLY thing you may `scancel` from — never `scancel -u`, `-t`, or `-n`.

Each worker claims one run at a time from the shared queue, runs it to completion, and exits when
the queue is empty. `srun --wait=0` is set so workers exiting at different times don't kill each
other.

## Monitor your workers

```
bash monitor_collaborator.sh      # your jobs' states + queue progress
bash refresh_my_ids.sh            # refresh your id file with any requeued/new ids
```

## If a worker's environment is broken

`worker_collab.py` fails fast: if the first few claimed configs all die in under 10 minutes (an
import/permissions problem, not real 15-hour training), the worker writes ONE report to
`problems/open/` naming the markers it wrongly failed, and stops. The OWNER's monitor loop reads
those reports and requeues the named markers — you never rename anything back yourself.

## What this sweep is

Three RND baselines at 300 fresh seeds (600–899) on PointMaze; see `resource_facts.md` and the run
folder's `experiment_background.md`. SWEEP_ID and env are in `packet_env.sh` (the one place the
volatile values live).
