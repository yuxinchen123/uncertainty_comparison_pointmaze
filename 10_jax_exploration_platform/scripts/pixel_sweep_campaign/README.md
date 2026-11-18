# pixel_sweep_campaign: the sweep queue of the pixel Montezuma study

A sweep is a list of **units** in a run folder's queue. A unit is one `train_final.py` process on
one card: one run (a single), or four runs packed into one process (a pack). A run is one policy:
one method, one value of each swept setting, one seed. Worker jobs are generic: a job holds no unit
until it has started and its worker has claimed one. `P` is
`/p/rlprojects/RND/10_jax_exploration_platform`, `RUN` a run folder under `P/runs/`, `M` this
folder. Plan: `~/.claude/plans/2026-11-17-19-00_montezuma-fast-last-layer-against-rnd.md`.

Every file here uses the standard library only and may run on a login node, except
`smoke_trainer_checks.py`, which imports JAX and runs only inside a smoke job. The trainer is
started by the worker as a process of its own. The login node's `python3` (3.10) runs every command
below; jobs use `/p/rlprojects/RND/.venvs/atari_pixels_jax/bin/python` (3.12).

## Files

| file | what it does |
|---|---|
| `build_units.py` | study description (JSON) → single units in `RUN/queue/pending/`;<br>packs four never-started singles into one unit; raises the target of finished units |
| `card_profiles.json` | the two execution profiles: "one copy" and "four copies" |
| `pixel_queue.py` | the claim by profile, progress from `progress.json`, where a unit's records are |
| `worker.py` | one process per card: claims a unit of its card's profile, runs `train_final.py`,<br>records how it ended, takes the next |
| `submit.py` | submits worker jobs on UVA under both accounts, and the persistent monitor job |
| `monitor.py` | one pass or a loop: returns units of ended jobs to pending, counts, progress,<br>status files, and rebuilds the early-signals report |
| `report.py` | rebuilds `RUN/early_signals.md` from the runs' record files |
| `remote_fleet.py` | Purdue Gautschi and Gilbreth, TAMU ACES, NCSA DeltaAI: server table, smoke job,<br>hand-over, return and check of units, `hand-over-tick` |
| `server_table.json` | the four servers' entries a run folder's table starts from (all not ready) |
| `smoke_check.py`,<br>`smoke_trainer_checks.py` | the smoke job of a server: its check and its speed check |
| `study_descriptions/` | the description of stage 1 |
| `campaign_imports.py` | loads the maze and grid campaigns' files under names that cannot clash |

Used unchanged from `P/scripts/maze_campaign/`: `freeze_code.py` (the frozen code copy) and
`cancel_ledger.py` (the only cancel path).

## Run folder

```
RUN/
├── source.json, code/source_<commit>/   the frozen code copy new jobs run (maze freeze_code.py)
├── study_descriptions/                  a copy of every description that was built
├── specifications/<unit>.json           the unit file
├── queue/pending|running|done|failed/<unit>.json    the unit's record; one rename per change of state
├── queue/packed/<unit>.json             singles a pack replaced (no worker claims them)
├── queue/remote_<server>/               units a remote server holds
├── data/<unit>/run/                     the trainer's run folder: run.json, checkpoint.pkl, metrics.jsonl,
│                                        game_first_times.json, game_windows.jsonl, events.jsonl, progress.json
├── logs/<unit>_job-<job id>.log         everything the unit's trainer printed in that job
├── logs/<job name>_<job id>.log         a Slurm job's output
├── slurm/submitted_jobids_<phase>_<account>.txt     the job-id ledgers (UVA); _<server>.txt for a remote server
├── slurm/scripts/, slurm/submissions/<job id>.json  job scripts; what each job asked for
├── slurm/jobs/<job id>_<node>_card-<i>/             hardware.json, assignment.json of one worker
├── slurm/worker_env.sh, slurm/job_starts.txt        the job environment; one line per start of a worker
├── slurm/remote_live/<server>/<unit>/run/           record files of units that train on a server
├── infrastructure/remote_fleet_servers.json         this run folder's server table
├── status.json, status.txt, monitor_state.json      written by every monitor pass
├── 20_mins_monitoring/outputs/<time>_monitoring.md  one report per pass
├── early_signals.md, early_signals_manifest.json    the early-signals report and its contents record
├── archived_baseline/<unit>/            record files folded from the archived baseline (report.py --fold-archive)
├── infrastructure_events.jsonl          units that did not complete, units recovered from ended jobs
└── STOP, SWEEP_COMPLETE                 sentinels
```

`experiment_background.md` and the other hand-written files of a run folder are not written here.

## Study description

One JSON object.

| key | meaning |
|---|---|
| `study` | the study's name in words |
| `phase` | one word group without underscores (`grid`); the first part of every unit name |
| `seeds` | `{"first": 13, "count": 8}` or `{"list": [1, 2, 5]}` |
| `target_updates` | updates per run (12,207 is 200 million steps; one update is 16,384 steps) |
| `total_timesteps` | the horizon of the learning-rate schedule (default 2,000,000,000) |
| `record_game_events`,<br>`record_bonus_profile` | the trainer's two record flags; the first also sets the environment's `report_game_events` |
| `methods` | a list; each entry has `method` (words), `pack_over`, and `grid` or `configurations` |
| `learner_changes`,<br>`environment_changes` | further trainer settings for every unit (optional) |

A method's `grid` is the cross product of `step_size`, `sharing_group` and `intrinsic_weight`
(RND names the weight only). `configurations` lists them one by one, for a later stage:
`[{"step_size": 0.1, "sharing_group": 16, "intrinsic_weight": 1}]`. `pack_over` names the setting
whose four values share one four-copy process: `step_size`, `intrinsic_weight` (RND) or `seed`.

Stage 1 is `study_descriptions/stage-1_grid_...json`: RND at 6 weights and the fast last layer at
4 step sizes × 3 sharing groups × 6 weights, seeds 13 to 20, 624 units.

Order of claims: seed by seed. Every configuration at seed 13 stands before any at seed 14. Inside
a seed the methods stand in the description's order (RND first in stage 1), and the four step
sizes of one sharing group and one weight are neighbours.

Stage 2 uses the same format with `configurations`, seeds 1 to 12 and `target_updates` 61035
(1 billion steps). Stage 3 is the stage-2 description with `target_updates` 122070, built into the
same run folder with `--continue-units`: every finished unit goes back to pending with the higher
target and continues its run folder.

## Unit names

```
grid_rnd_intrinsic-weight-1_seed-13
grid_fast-last-layer_step-size-0.03_sharing-group-16_intrinsic-weight-1_seed-13
grid_rnd_four-copies_intrinsic-weights-0.25-0.5-1-2_seed-13
grid_fast-last-layer_four-copies_step-sizes-0.01-0.03-0.1-0.3_sharing-group-16_intrinsic-weight-1_seed-13
```

The last two are packs; `four-copies` stands in every pack's name and in no other.

## Execution profiles

`card_profiles.json` holds the learner and environment settings of the archived baseline's two
card classes (a test compares them with the archived `run.json` files).

| profile | cards | copies per process | settings of |
|---|---|---|---|
| one copy | card memory from 10,000 MB to under 60,000 MB | 1 | the archive's RTX A4500 runs |
| four copies | card memory of 60,000 MB or more | 4 | the archive's H100 NVL run |

1. A worker takes the profile of its card and claims only units with that profile's number of
   copies: singles on small cards, packs on large cards.
2. A unit's profile is fixed at its first start. The claim writes the profile's name and settings
   into the unit's record, and every later start uses the recorded settings.
3. A unit may continue on another card type of the same profile. The worker adds a `worker_start`
   line with the card to the run folder's `events.jsonl` at every start, and the report names
   the units that changed card type.
4. A unit that ran out of card memory returns to pending for cards with more memory.

## Steps

```bash
PY="env PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 python3"
M=/p/rlprojects/RND/10_jax_exploration_platform/scripts/pixel_sweep_campaign
MAZE=/p/rlprojects/RND/10_jax_exploration_platform/scripts/maze_campaign
RUN=/p/rlprojects/RND/10_jax_exploration_platform/runs/<run folder>
```

1. **Freeze the code.** Commit and push first; the command refuses code that is not committed.
   Check that `git status` lists this folder's JSON files (`card_profiles.json`, `server_table.json`,
   `study_descriptions/*.json`) among the files of the commit: the repository ignores `*.json`, and
   this folder's own `.gitignore` takes its JSON files out of that rule. A worker of a frozen copy
   without `card_profiles.json` stops at its start.

   ```bash
   $PY $MAZE/freeze_code.py --run-dir $RUN
   ```

2. **Build units.** Look first, then write. Building the same description again adds nothing.

   ```bash
   $PY $M/build_units.py --run-dir $RUN --study $M/study_descriptions/stage-1_grid_*.json --dry-run
   $PY $M/build_units.py --run-dir $RUN --study $M/study_descriptions/stage-1_grid_*.json
   ```

3. **Pack units for the large cards that exist.** Four never-started singles become one four-copy
   unit. Make only as many packs as there are large cards for them (Gautschi: 4 jobs; 8 packs keep
   them supplied). `--unpack-pending` takes never-started packs apart again.

   ```bash
   $PY $M/build_units.py --run-dir $RUN --pack-pending --packs 8 --dry-run
   $PY $M/build_units.py --run-dir $RUN --pack-pending --packs 8
   ```

4. **Submit workers on UVA** (both accounts). Look first, then send. New worker cards of a profile
   never exceed that profile's pending units without a worker. `--only-cards-free-now` sends only
   jobs pinned to nodes with a free card now. `--leave-cards N` keeps N free cards out of the plan
   for another study's waiting units. For a first small submission add `--max-new-cards 1`.

   ```bash
   $PY $M/submit.py --run-dir $RUN --phase grid --only-cards-free-now --leave-cards 4 --dry-run
   $PY $M/submit.py --run-dir $RUN --phase grid --only-cards-free-now --leave-cards 4
   sacct -j <the new ids> -X -n -P -o JobID,State,Elapsed,NodeList      # within a minute of the submission
   ```

   Each card asks for 4 processors and 16 GiB. Card types: `--gpu-types` replaces the default list
   (RTX 2080 Ti, RTX A4000, RTX A4500, A40, RTX 5080, A100, H100 NVL). The RTX 4000 Ada is not in it:
   its gate-6 speed check (3.039 s per update) is below the RTX A4000's (2.038 s). On an RTX 2080 Ti
   the worker gives the trainer `XLA_PYTHON_CLIENT_MEM_FRACTION=0.95` (the default update ran out
   of card memory at 0.85, gate-6 job 6955822); every other card keeps the job's 0.85.

   `--cancel-waiting-after 120` cancels, by id from the run's own ledger, every job of the call
   that still waits in the scheduler's queue 120 seconds after the last submission; its units stay
   pending. With it, no worker job is left waiting in UVA's queue.

5. **Start the persistent monitor** (one processor on `nolim`, a pass every 20 minutes), in the
   same turn as the first submission, and create the session's 20-minute check then too.

   ```bash
   $PY $M/submit.py --run-dir $RUN --monitor --top-up-phase grid --only-cards-free-now --leave-cards 4 \
       --cancel-waiting-after 120
   $PY $M/monitor.py --run-dir $RUN                                     # one pass by hand
   ```

   `--top-up-phase` lets every pass submit workers for pending units that have none, with the same
   plan options (`--cancel-waiting-after` included: the pass then cancels its own jobs that did
   not start). `--complete-when-drained` writes `RUN/SWEEP_COMPLETE` once every unit is done or
   failed.

6. **Remote servers.** Every server starts not ready and becomes ready only after its smoke job.

   ```bash
   $PY $M/remote_fleet.py --run-dir $RUN init-table                     # the table, every server not ready
   $PY $M/remote_fleet.py --run-dir $RUN smoke --server gautschi --dry-run
   $PY $M/remote_fleet.py --run-dir $RUN smoke --server gautschi        # prepare, then one job per pool
   $PY $M/remote_fleet.py --run-dir $RUN smoke-result --server gautschi # reads the verdict; ready when passed
   $PY $M/remote_fleet.py --run-dir $RUN hand-over-tick --dry-run
   $PY $M/remote_fleet.py --run-dir $RUN hand-over-tick                 # every monitoring tick of the session
   $PY $M/remote_fleet.py --run-dir $RUN status
   $PY $M/remote_fleet.py --run-dir $RUN take-back --server gautschi --units 9999 --hold
   ```

   Before the first smoke job a server needs its environment (JAX 0.11.0, JAXtari from the pinned
   source) at the table's `python` path; `prepare` sends the code copy, the asset pack and the
   smoke files. A server's limits are in the table: `pools.<pool>.max_jobs`, `time_limit`,
   `sbatch_options`, `execution_profile`, and `hand_over.pools.<pool>.units_at_once`.

   | server | pool | request | profile | units at once |
   |---|---|---|---|---|
   | Gautschi | `ai-h100` | `-p ai -q normal`, one H100, 14 cores,<br>24 hours, at most 4 jobs | four copies | 8 |
   | Gilbreth | `standby-a100-80gb` | `-q standby`, account `dsmichil`, 4 hours | four copies | 4 |
   | Gilbreth | `standby-a30-a10` | `-q standby`, account `dsmichil`, 4 hours | one copy | 0 until its<br>speed check |
   | ACES | `gpu-h100` | `-p gpu --gres=gpu:h100:1`<br>`--cpus-per-task=8 --mem=39G`,<br>through portal05 | four copies | 0 (stages 2 and 3) |
   | DeltaAI | `ghx4-h100` | `ghx4`, one H100, through portal04; ARM | four copies | 0 (not built) |

   What `hand-over-tick` does for each ready server: one status call; finished units copied to UVA
   and checked; units whose job ended put back in the server's queue; pending units of each pool's
   profile handed over up to the pool's limit, the first of the claim order first; worker jobs
   submitted for units no job will take; the record files of the units that train there copied to
   `RUN/slurm/remote_live/`.

   A finished unit counts as done only when its returned files pass the check: `progress.json`
   completed at the unit's target, one metric line per update, the record files folded to the
   target, `run.json` with the unit's seeds and settings, the trainer's completed event, and a
   checkpoint.

7. **Stop a sweep** (rule `submit-the-whole-queue-up-front.md`, "Cancelling a whole queue"):

   1. `touch $RUN/STOP`: no worker claims a further unit and no monitor pass submits a job. A
      cancelled job's trainer gets the stop signal, finishes its update, saves and ends.
   2. Take waiting units back from every remote server (`take-back ... --hold`).
   3. Cancel every ledger of the run, one call per file, the monitor's ledger last:

      ```bash
      for ledger in $RUN/slurm/submitted_jobids_*_sl5nw.txt $RUN/slurm/submitted_jobids_*_yuxinchen.txt; do
          $PY $MAZE/cancel_ledger.py --ledger "$ledger" < /dev/null
      done
      $PY $MAZE/cancel_ledger.py --ledger $RUN/slurm/submitted_jobids_<server>.txt --server <server>
      ```

   4. Delete the session's recurring check. Never `scancel -u`, `-t` or `-n`.

## What a worker does with a unit

| how the unit's trainer ended | where the unit goes | counts toward the 3 failures |
|---|---|---|
| `progress.json` says completed at the unit's target | `done` | — |
| exit code 0 without that, with a checkpoint<br>(time limit or stop signal) | `pending`; a later job continues it | no |
| ran out of card memory | `pending`, for cards with more memory | no |
| any other end (a loss that is not finite included) | `pending`; `failed` at the third failure | yes |
| its job ended without a result | `pending` at the next monitor pass | no |

The trainer's time limit is what is left of the job, read from the scheduler, less 15 minutes. The
worker only ever sends the trainer SIGTERM, on which it finishes its update, saves and ends. After
a unit that did not complete, the worker takes no further unit.

## The early-signals report

`report.py` reads `game_first_times.json`, `game_windows.jsonl` and `progress.json` of every unit
and writes `RUN/early_signals.md`; it reads no metrics file and stops no run. Sections: fleet
status; speed by card type (copies, seconds per iteration, total steps per second, steps per second
per copy, hours per million steps per copy); every configuration at one common mark (seeds, mean
score over the run, capped first exit, rooms reached); the mean score at every mark; medians of the
first-time steps; shares of games with the key and out of the start room, and lives lost by cause;
score levels; standing checks. Its definitions section states each measure.

```bash
$PY $M/report.py --run-dir $RUN                       # rebuild by hand (every monitor pass does it)
$PY $M/report.py --run-dir $RUN --mark-read all       # the reader has read the page as it stands
```

The archived baseline beside the new runs: its record files are made once from the archived
metrics, inside a Slurm job (it reads about 340 MB of metric lines):

```bash
$PY $M/report.py --run-dir $RUN --fold-archive \
    /p/rlprojects/RND_worktrees/merge-atari-cleanrl/10_jax_exploration_platform/runs/2026-11-06-01-14_*/data
```

## Tests

```bash
cd /p/rlprojects/RND/10_jax_exploration_platform
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s tests/pixel_sweep_campaign -p "test_pixel_*.py"
```

88 tests; they import no numerical library and start no trainer (a stand-in script plays
`train_final.py`), and they also pass under pytest with the project's Python 3.12.

## Taken from the maze and grid campaigns

Imported, not copied: the queue's return, finish, recover and counts (`unit_queue.py`); the
submission planner, the scheduler's state, the ledger names and the time limits (maze `submit.py`);
the job's end time, the claim records and the memory reading (maze `worker.py`); the table and
node tables of a monitoring pass (maze `monitor.py`); the remote fleet's ssh and rsync calls,
status call, queue moves, job submission and `take-back` (grid and maze `remote_fleet.py`).

Copied with the source named in a comment, because the pixel sweep differs: the claim (by
profile), adding a unit (no configuration file), the worker and monitor job scripts, the UVA
submission (job name), the status script's progress section, the return of an unfinished unit
(a whole run folder), and the order numbers of a later build.

**Exercised and not exercised.** The tests cover the builder, the claim, the worker, the plan, one
monitoring pass, the report, and the remote fleet's local parts. `submit.py --dry-run` has read the
live UVA scheduler. No job of this folder has been submitted yet, no remote server has been
contacted by `remote_fleet.py`, and `smoke_check.py` and `smoke_trainer_checks.py` have not run on
a card.
