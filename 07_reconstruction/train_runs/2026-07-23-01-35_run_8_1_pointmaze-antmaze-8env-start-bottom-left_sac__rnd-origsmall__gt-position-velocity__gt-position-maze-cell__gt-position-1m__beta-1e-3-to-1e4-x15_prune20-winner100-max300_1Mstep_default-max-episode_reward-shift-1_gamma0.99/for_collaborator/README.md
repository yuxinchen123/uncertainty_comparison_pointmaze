# Run 8.1 (point maze + ant maze) — collaborator packet

You (any rlprojects member — the known collaborator is `yuxinchen`) can add worker jobs to this
already-launched sweep, using YOUR OWN Slurm caps, to finish it faster. You never build the queue,
never prune, never requeue, and never touch another user's files — you only add workers and, if your
environment breaks, write a problem report.

**What this run tests (one line):** on 8 fixed start-bottom-left environments (4 PointMaze + 4
AntMaze), does a ground-truth visit-count bonus beat RND (and SAC alone) at driving exploration,
swept over 15 bonus weights with seed racing. (Full context: `resource_facts.md` and the run folder's
`../experiment_background.md`.)

## How the queue works

- A **work queue** lives in the run folder: `queue/2026-07-23-02-05_pm-am-run1/{pending,running,done,failed,pruned}/`,
  one small JSON marker per run (92,400 to start). Each of your workers atomically claims a marker
  (renames it `pending/` -> `running/`), runs `train.py` to completion, then moves it to `done/`
  (success) or `failed/` (non-zero exit), and claims the next one — until `pending/` is empty.
- **Why the pending count shrinks faster than runs finish:** the OWNER's controller races configs
  per (environment, algorithm) cell — floor 20 seeds, winner-only at 100, max 300 — and moves losing
  configs' pending markers to `pruned/`. That is the owner's job. A worker of yours that happens to
  claim a marker pruned in-flight is HARMLESS (it just runs one extra seed); you never react to it.
- **Single-writer discipline:** every shared file has exactly one writer. You write only your own id
  file, your own logs under `logs/`, and new problem files under `problems/open/`. The one shared
  action is a worker's atomic claim-rename. You never rename markers by hand.

## The five-step pipeline

1. **Setup check (once).** Prove your access and the environment WITHOUT touching the real queue:
   ```
   bash smoke_test.sh
   ```
   It runs non-mutating permission probes, then >= 5 short canary runs of the real `train.py`
   (isolated under `smoke_data/<you>/<timestamp>/`, each with a hard 20-minute timeout, the queue
   never claimed), and prints a PASS/FAIL checklist. If it FAILS, tell the owner — do not debug
   shared state.

2. **First wave.** Submit worker jobs under your own caps:
   ```
   bash launch_workers_collaborator.sh          # or:  DRY=1 bash launch_workers_collaborator.sh  (preview only)
   ```
   - Submits `30x1` / `14x1` / `12x1` worker jobs to the **cpu** and **nolim** partitions, and
     CPU-ONLY jobs to LOW-capability **gpu**-partition nodes (`--gpus-per-node=0`) — the open
     partitions of your own pools only. NO reservation, NO gnolim, NO GPU-using job.
   - Trims the count live against YOUR caps (cpu 400 / gpu 400 / nolim 80), keeping ~16 CPUs of
     headroom per pool, against the remaining queue depth (never more worker CPUs than unclaimed
     runs, counting every submitter's live workers), and against node availability.
   - Sets `--time` for you (min of 4 days and the time to the next maintenance window).
   - Records every job id in YOUR OWN file `submitted_jobids_2026-07-23-02-05_pm-am-run1_<you>.txt`.

3. **First-job checklist.** After the first wave, verify one job of each shape actually got the CPUs
   it asked for:
   ```
   sacct -j <id> --format=JobID,JobName,AllocCPUS,ReqMem,State,NodeList
   ```
   `AllocCPUS` must EQUAL `ntasks` (30 / 14 / 12). If it is doubled, `--ntasks-per-core=2` was lost —
   write a problem file and stop that shape. Then confirm workers are claiming:
   ```
   grep -m3 'claimed' logs/<jobname>_<id>.log
   ls queue/2026-07-23-02-05_pm-am-run1/running | wc -l      # should be growing
   ```

4. **Monitor + auto-refill.** Start the passive 10-minute loop once; it refreshes your id file (so
   requeued jobs' new ids are recorded), snapshots your jobs, and re-runs the self-gating launcher to
   top up when useful:
   ```
   nohup bash monitor_collaborator.sh >> logs/monitor_collaborator.log 2>&1 &
   ```
   To refresh your id file by hand (always do this BEFORE any manual scancel):
   ```
   bash refresh_my_ids.sh
   ```

5. **Am I done?** When the file `../SWEEP_COMPLETE` exists (the owner's monitor touches it once
   `pending/` and `running/` are both empty), the sweep is finished — the launcher and the monitor
   both stop on their own. Nothing more to submit.

## If the launcher's big jobs pend while the partition holds idle fragments

If several of your `30x1` cpu jobs sit `PENDING (Resources)`/`(Priority)` for a while and
`sinfo -p cpu -N` shows nodes with 12-30 idle threads (and no competing pending jobs), apply the
fragment-conversion procedure of `submit-cpu-sweep` SKILL.md section 10 IMMEDIATELY (do not wait to
"confirm" the pend): `bash refresh_my_ids.sh`, `scancel` YOUR OWN still-pending big-shape ids (from
your id file only), then resubmit the same CPU budget as unpinned fragment fillers — the largest
shape each live fragment takes (`sbatch --partition=cpu --job-name=<your cpu prefix> --ntasks=<N>
worker_14x1_collab.slurm`, e.g. 28 / 26 / 24 / 16 / 14 / 12) — keeping your pool total at or under
cap minus 16 headroom. See `resource_facts.md`.

## Cancelling

Cancel ONLY the job ids in your own file, and refresh it first:
```
bash refresh_my_ids.sh
scancel $(cat submitted_jobids_2026-07-23-02-05_pm-am-run1_<you>.txt)
```
NEVER `scancel -u`, `scancel -t`, or `scancel -n` — those hit other people's and other sessions'
jobs. Your id file is the ONLY legal source of ids to cancel.

## If a worker's environment is broken

`worker_collab.py` fails fast: if the first few claimed configs all die in under 10 minutes (an
import/permissions problem, not real 9-25 h training), the worker writes ONE report to
`problems/open/` naming the markers it wrongly failed, and stops. The OWNER's monitor loop reads
those reports and requeues the named markers — you never rename anything back yourself.

## Problem channel

One NEW timestamped, user-named file per issue in `problems/open/` (never edit or append to an
existing one). The owner's monitor picks it up and moves it to `problems/resolved/`.

## The coordination contract

- **Two monitors, disjoint write sets.** The OWNER's monitor runs every ~10-20 minutes and ACTIVELY
  fixes things: liveness of the controller, orphan reclaim, pruning, problem handling, and the
  `SWEEP_COMPLETE` sentinel. YOUR monitor is PASSIVE: your own ids, gated refill, reporting. They can
  never fight.
- **When in doubt, write a problem file and stop.** You never repair shared state.

## The DON'Ts

1. Don't run `build_queue.py`, the prune controller, `requeue_orphans.py`, the owner monitor, or any
   queue-marker rename by hand. (All owner-only; several refuse to run as a non-owner.)
2. Don't use a reservation or `--qos` you don't hold; don't submit `gnolim`; don't submit a
   GPU-using job (`device=cuda` / `--gpus-per-node>0`).
3. Don't write the owner's id file or log into `../logs/` (owner's). Your ids -> your `_<you>.txt`
   file; your logs -> `logs/`.
4. Don't blanket-cancel (`scancel -u/-t/-n`). Only ids from your own file, refreshed first.
5. Don't point a job at any `/u/<user>` private path. Everything you need is under this run folder and
   the shared env `/p/rlprojects/RND/.venvs/exploration`.

Volatile values (run dir, sweep id, env, your id file, job-name prefixes, targets) all live in one
place: `packet_env.sh`.
