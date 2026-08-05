> **STOPPED 2026-08-05.** This sweep was ended by the owner before the stage-1 race reached a
> verdict. The queue's pending pools are empty (every remaining marker is parked in
> `queue/2026-08-01-02-03_run812/stopped_2026-08-05/`), so any worker you still have running will
> find nothing to claim and exit by itself. **Do not submit new workers for this run.** See
> `../SWEEP_STOPPED.md`. The follow-up run has its own collaborator packet.

# Run 8.1.2 (AntMaze UMaze + Medium) — collaborator packet

You (any rlprojects member — the known collaborator is `yuxinchen`) can add worker jobs to this
already-launched sweep, using YOUR OWN Slurm caps, to finish it faster. You never build the queue,
never prune, never requeue, and never touch another user's files — you only add workers and, if your
environment breaks, write a problem report.

**What this run tests (one line):** on the two AntMaze environments where RND is the only method that
ever reaches the goal, can any of four new RND variants (algorithms 1, 2.1, 2.2, 2.3) beat the best
known RND configuration — each variant raced to 100 seeds at 1M steps against a frozen bar taken from
run 8.1, while the RND baseline itself is run out to 10M steps for the comparison curve. (Full
context: `resource_facts.md` and the run folder's `../experiment_background.md`.)

## How the queue works — TWO pending pools under one sweep

- A **work queue** lives in the run folder:
  `queue/2026-08-01-02-03_run812/{pending_1m,pending_10m,running,done,failed,pruned}/`, one small
  JSON marker per run (24,200 to start). Each of your workers atomically claims a marker (renames it
  from a pending pool into `running/`), runs `train.py` to completion, then moves it to `done/`
  (success) or `failed/` (non-zero exit), and claims the next one.

| pool | task | runs | steps per run | which of YOUR shapes claim it |
|---|---|---|---|---|
| `pending_1m` | S — four new RND arms | 24,000 | 1,000,000 (about 15.4 h) | `worker_{30,28,14,12}x1_collab.slurm` on cpu / nolim / gpu-partition-CPU-only; the gnolim shapes as fallback |
| `pending_10m` | R — RND baseline | 200 | 10,000,000 (about 154 h on cpu) | `worker_gnolim_{14,12}x1_collab.slurm` first, and the owner's GPU jobs |

- **Claim order** is set by each `.slurm`'s `WORKER_POOLS`: the task-S shapes carry `"pending_1m"`;
  the gnolim shapes carry `"pending_10m pending_1m"` (10M first, 1M as fallback).
- **The 10M walltime guard is automatic.** This run has no checkpoints, so the run's own `worker.py`
  allows a `pending_10m` claim only when the job has at least 170 h of walltime LEFT. A gnolim job
  submitted before the 2026-08-05 maintenance window gets a short walltime and therefore serves
  task S; a 20-day gnolim job submitted after the maintenance takes 10M runs first. Nothing for you
  to configure.
- **Why `pending_1m` shrinks faster than runs finish:** the OWNER's stage-1 controller prunes a
  task-S configuration whose 99% upper confidence bound falls below the frozen bar and moves its
  pending markers to `pruned/`. That is the owner's job. A worker of yours that claims a marker
  pruned in-flight is HARMLESS (it just runs one extra seed); you never react to it.
- **Single-writer discipline:** every shared file has exactly one writer. You write only your own id
  file, your own logs under `logs/`, and new problem files under `problems/open/`. The one shared
  action is a worker's atomic claim-rename. You never rename markers by hand.

## The five-step pipeline

1. **Setup check (once).** Prove your access and the environment WITHOUT touching the real queue:
   ```
   bash smoke_test.sh
   ```
   It runs non-mutating permission probes on both pools, then >= 5 short canary runs of the real
   `train.py` (isolated under `smoke_data/<you>/<timestamp>/`, each with a hard 20-minute timeout,
   configs read from the COLD high-id end of each pool and never claimed), and prints a PASS/FAIL
   checklist. If it FAILS, tell the owner — do not debug shared state.

2. **First wave.** Submit worker jobs under your own caps:
   ```
   bash launch_workers_collaborator.sh          # or:  DRY=1 bash launch_workers_collaborator.sh  (preview only)
   ```
   - Fills, in order: CPU-ONLY jobs on LOW-capability **gpu**-partition nodes (`--gpus-per-node=0`),
     CPU-ONLY jobs on **gnolim** nodes (`--gpus-per-node=0`), then the open **cpu** and **nolim**
     partitions. The open partitions of your own pools only — NO reservation, NO GPU-USING job.
   - Trims the count live against YOUR caps (cpu 400 / gpu 400 / nolim 80 / gnolim 80), keeping
     ~16 CPUs of headroom per pool, against the remaining unclaimed queue depth, and against live
     node availability (`--ntasks` never exceeds a node's physical core count).
   - Sets `--time` for you: the smallest of the partition limit (4 days cpu/gpu, 20 days
     nolim/gnolim) and the time to the next maintenance window minus 30 minutes. **Every job
     submitted now ends before the 2026-08-05 07:30 maintenance.**
   - Records every job id in YOUR OWN file `submitted_jobids_2026-08-01-02-03_run812_<you>.txt`.

3. **First-job checklist.** After the first wave, verify one job of each shape actually got the CPUs
   it asked for:
   ```
   sacct -j <id> --format=JobID,JobName,AllocCPUS,ReqMem,State,NodeList
   ```
   `AllocCPUS` must EQUAL `ntasks` (30 / 28 / 14 / 12). If it is doubled, `--ntasks-per-core=2` was
   lost — write a problem file and stop that shape. Then confirm workers are claiming:
   ```
   grep -m3 'claimed' logs/<jobname>_<id>.log
   ls ../queue/2026-08-01-02-03_run812/running | wc -l      # should be growing
   ```
   A gnolim worker's first log line names its pools and its job end time, so you can see directly
   whether it will take 10M runs or task S.

4. **Monitor + auto-refill.** Start the passive 10-minute loop once; it refreshes your id file (so
   requeued jobs' new ids are recorded), snapshots your jobs and both pools, and re-runs the
   self-gating launcher to top up when useful:
   ```
   nohup bash monitor_collaborator.sh >> logs/monitor_collaborator.log 2>&1 &
   ```
   To refresh your id file by hand (always do this BEFORE any manual scancel):
   ```
   bash refresh_my_ids.sh
   ```

5. **Am I done?** When the file `../SWEEP_COMPLETE` exists (the owner's monitor touches it once both
   pools and `running/` are empty and every task-S configuration has a stage-1 verdict), the sweep is
   finished — the launcher and the monitor both stop on their own. Nothing more to submit.

## GPU jobs are owner-only

The task-R 10M runs also execute on GPUs (`--gpus-per-node=1`, `WORKER_DEVICE=cuda`, two packed
workers per GPU). **Those jobs are OWNER-ONLY and this packet does not ship them** — do not build or
submit one. Every script here is CPU-only (`WORKER_DEVICE=cpu`, and `--gpus-per-node=0` wherever the
partition would otherwise attach a GPU). The launcher also refuses to place CPU-only jobs on the
nodes the owner's GPU jobs use (cheetah02/04/08/09, jaguar01/02, adriatic01-06), so your jobs can
never squat the CPUs a 10M GPU run needs.

## If the launcher's big jobs pend while the partition holds idle fragments

If several of your `30x1` cpu jobs sit `PENDING (Resources)`/`(Priority)` and `sinfo -p cpu -N` shows
nodes with 12-30 idle threads (and no competing pending jobs), apply the fragment-conversion
procedure of `submit-cpu-sweep` SKILL.md section 10 IMMEDIATELY (do not wait to "confirm" the pend):
`bash refresh_my_ids.sh`, `scancel` YOUR OWN still-pending big-shape ids (from your id file only),
then resubmit the same CPU budget as unpinned fragment fillers — the largest shape each live fragment
takes (`sbatch --partition=cpu --job-name=<your cpu prefix> --ntasks=<N> worker_14x1_collab.slurm`,
e.g. 28 / 26 / 24 / 16 / 14 / 12) — keeping your pool total at or under cap minus 16 headroom. See
`resource_facts.md`.

## Cancelling

Cancel ONLY the job ids in your own file, and refresh it first:
```
bash refresh_my_ids.sh
scancel $(cat submitted_jobids_2026-08-01-02-03_run812_<you>.txt)
```
NEVER `scancel -u`, `scancel -t`, or `scancel -n` — those hit other people's and other sessions'
jobs. Your id file is the ONLY legal source of ids to cancel.

## If a worker's environment is broken

`worker_collab.py` fails fast: if the first few claimed configs all die in under 10 minutes (an
import / permissions problem, not 15 h of real training), the worker writes ONE report to
`problems/open/` naming the markers it wrongly failed, and stops. The OWNER's monitor loop reads
those reports and re-pends the named markers into their origin pool — you never rename anything back
yourself.

## Problem channel

One NEW timestamped, user-named file per issue in `problems/open/` (never edit or append to an
existing one). The owner's monitor picks it up and moves it to `problems/resolved/`.

## The coordination contract

- **Two monitors, disjoint write sets.** The OWNER's monitor runs every ~20 minutes and ACTIVELY
  fixes things: orphan reclaim, re-pending failed markers, the stage-1 prune controller and its
  invariant checker, the disk guard, and the `SWEEP_COMPLETE` sentinel. YOUR monitor is PASSIVE: your
  own ids, gated refill, reporting. They can never fight.
- **When in doubt, write a problem file and stop.** You never repair shared state.

## The DON'Ts

1. Don't run `build_queue.py`, `stage1_controller.py`, `requeue_orphans.py`, the owner monitor, or
   any queue-marker rename by hand. (All owner-only; several refuse to run as a non-owner.)
2. Don't use a reservation or `--qos` you don't hold; don't submit a GPU-USING job
   (`device=cuda` / `--gpus-per-node>0`) — task-R GPU jobs are owner-only.
3. Don't write the owner's id file or log into `../logs/` (owner's). Your ids -> your `_<you>.txt`
   file; your logs -> `logs/`.
4. Don't blanket-cancel (`scancel -u/-t/-n`). Only ids from your own file, refreshed first.
5. Don't point a job at any `/u/<user>` private path. Everything you need is under this run folder and
   the shared env `/p/rlprojects/RND/.venvs/exploration`.

Volatile values (run dir, sweep id, env, pool paths, your id file, job-name prefixes, targets) all
live in one place: `packet_env.sh`.
