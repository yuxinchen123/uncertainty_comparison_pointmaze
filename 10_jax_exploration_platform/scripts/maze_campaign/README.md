# maze_campaign: the sweep queue of the PointMaze and AntMaze study

A sweep is a list of **units** in a run folder's queue. A unit is one process on one card: one
environment, one or both reward settings, a list of cells (one value per per-copy setting: learning
rate, entropy coefficient, intrinsic weight, ...) and a range of seeds. Its policies are
cells × reward settings × seeds. Worker jobs are generic: a job holds no unit until it has started
and its worker has claimed one. `P` is `/p/rlprojects/RND/10_jax_exploration_platform`, `RUN` a run
folder under `P/runs/`, `M` this folder. Every command runs with
`PYTHONNOUSERSITE=1 /p/rlprojects/RND/.venvs/platform_jax/bin/python` (Python 3.11 or later is
needed; the login node's `python3` is 3.10). Only the worker's child process, `run_training.py`,
imports JAX; every file here uses the standard library and may run on a login node.

## Files

| file | what it does |
|---|---|
| `build_units.py` | study description (JSON) → units in `RUN/queue/pending/` |
| `freeze_code.py` | copies the committed code into `RUN/code/source_<commit>/`;<br>jobs run that copy |
| `submit.py` | submits worker jobs on UVA under both accounts,<br>and the persistent monitor job |
| `worker.py` | one process per card: claims a unit, runs `run_training.py`,<br>records how it ended, takes the next |
| `monitor.py` | one pass or a loop: returns units of ended jobs to pending,<br>counts, progress, status files |
| `remote_fleet.py` | MIT, Purdue Gautschi, TAMU ACES, NCSA DeltaAI:<br>thin wrappers over the grid campaign's file of the same name |
| `cancel_ledger.py` | the only cancel path: the jobs of one ledger file of a stopped run,<br>or single ids of one ledger |
| `unit_queue.py` | the queue: claim by rename, return, finish, recover,<br>progress read from a shard |
| `grid_campaign.py` | loads the grid campaign's functions that are reused here |
| `analysis/` | records, measures and the report: a separate piece of work with its own files;<br>nothing above imports it |

## Run folder

```
RUN/
├── source.json                      the frozen code copy new jobs run (freeze_code.py)
├── code/source_<commit>/            that copy: 10_jax_exploration_platform/{src,scripts,configs}
├── study_descriptions/              a copy of every description that was built
├── specifications/<unit>.json       the unit file; never changes
├── queue/pending|running|done|failed/<unit>.json    the unit's record; one rename per change of state
├── queue/remote_<server>/, queue/deferred/          units a remote server holds; units held back
├── data/<unit>.jsonl                the unit's shard (run_training.py)
├── checkpoints/<unit>.npz           the unit's saved state; deleted when the unit completes
├── logs/<unit>.log                  run_training.py's own log of the unit
├── logs/<unit>_job-<job id>.log     everything the unit's process printed in that job
├── logs/<job name>_<job id>.log     a Slurm job's output
├── slurm/submitted_jobids_<phase>_<account>.txt     the job-id ledgers (UVA); _<server>.txt for a remote server
├── slurm/unit_configurations/<unit>.toml            the configuration file run_training.py reads for the unit
├── slurm/scripts/, slurm/submissions/<job id>.json  job scripts; what each job asked for
├── slurm/jobs/<job id>_<node>_card-<i>/             hardware.json, assignment.json of one worker
├── slurm/worker_env.sh, slurm/job_starts.txt        the job environment; one line per start of a worker
├── status.json, status.txt, monitor_state.json      written by every monitor pass
├── 20_mins_monitoring/outputs/<time>_monitoring.md  one report per pass
├── infrastructure_events.jsonl      units that did not complete, units recovered from ended jobs
└── STOP, SWEEP_COMPLETE             sentinels
```

`experiment_background.md` and the other hand-written files of a run folder are not written here.

## Study description

One JSON object. With a `sweeps` list, each entry is one sweep and takes every key it does not set
from the top level. A value marked "per environment" may be one value, or an object keyed by
family (`pointmaze`, `antmaze`) or by environment.

| key | meaning |
|---|---|
| `study`, `phase`, `method` | names in words (lower case, hyphens); `phase` and `method` are part of every unit id |
| `bonus` | the bonus family, as `run_training.py --bonus` takes it |
| `environments` | of `pointmaze-medium`, `pointmaze-large`, `antmaze-umaze`,<br>`antmaze-medium`, `antmaze-large` |
| `reward_settings` | one or both of `arrival_plus_one`, `step_cost` |
| `reward_settings_in_one_process` | with both settings: `true` (default) puts each cell once per setting<br>into the same process on the same seeds; `false` gives each setting its own units |
| `configuration_files` | per family: a recorded configuration file (path relative to `P`).<br>The other maps of a family are made from it (start, goal, step limit of the plan) |
| `environments_per_policy` | per environment; default: the file's `n_envs` |
| `steps_per_policy` or `iterations` | per environment; exactly one of the two.<br>Steps are rounded up to whole iterations |
| `windows_per_unit` or `window_iterations` | per environment; exactly one of the two |
| `cells` | see below |
| `seeds` | per environment: `{"first": 0, "count": 16}` |
| `policies_per_unit` | per environment: a number, or `{"card": "<card of card_facts.json>", "default": 512}`.<br>With a card, the number is that card's `copies` in `P/device/estimates/<workload>.json`;<br>the default is used when the file or the row is missing |
| `minimum_card_memory_mb` | per environment: the least card memory a unit needs.<br>Without it: 1.25 × the estimate's largest memory per policy × the unit's policies |
| `half_size_units` | the first so many units get half the policies (default 0),<br>so a card class sees two sizes (copies-per-card skill, step 4) |
| `trainer_changes` | other `[trainer]` values of the configuration file,<br>e.g. `{"gather_minibatches_per_epoch": true}` |
| `extra_options` | further `run_training.py` options, as a list of words |
| `workload` | per environment: the workload's name; default<br>`<environment>_environments-per-policy-<n>_update-<style>_bonus-<bonus>` |
| `run_seed` | one run seed for every unit; default: keyed by environment and seed range (see below) |

Cells, three kinds:

1. `{"kind": "first_pass_grid"}`: the plan's grid, 3 learning rates (1e-4, 3e-4, 1e-3) × 2 entropy
   coefficients (0.001, 0.01) × 10 intrinsic weights (4^-6 to 4^3) = 60 cells.
2. `{"kind": "grid", "values": {"learning_rate": [...], "intrinsic_weight": [...]}}`: the cross product
   of the listed values (the edge rule's extra weights, the second pass).
3. `{"kind": "list", "cells": [{"learning_rate": 3e-4, "entropy_coefficient": 0.01, "intrinsic_weight": 0.25}]}`:
   exactly these cells (a confirmation is one cell with `"seeds": {"first": 30000, "count": 1024}`).

A cell may name `learning_rate`, `entropy_coefficient`, `intrinsic_weight`, `clip_coefficient`,
`value_coefficient`, or any other per-copy setting by the trainer's own name (the keys of
`SETTING_RULES` in `P/src/exploration_platform/training/sweep.py`, read at build time).

Example, the first-pass development sweep on PointMaze Large in both reward settings:

```json
{
  "study": "cleanrl-ppo-rnd-baseline",
  "phase": "development-first-pass",
  "method": "cleanrl-ppo-rnd",
  "bonus": "rnd_next_state",
  "environments": ["pointmaze-large"],
  "reward_settings": ["arrival_plus_one", "step_cost"],
  "configuration_files": {
    "pointmaze": "configs/experiments/pointmaze_large_cleanrl-ppo-rnd_goal-termination_reward-arrival-plus-one_example_steps-1024000.toml"
  },
  "environments_per_policy": {"pointmaze": 4},
  "steps_per_policy": {"pointmaze": 50331648},
  "windows_per_unit": 200,
  "cells": {"kind": "first_pass_grid"},
  "seeds": {"first": 0, "count": 16},
  "policies_per_unit": {"card": "RTX A4500", "default": 512},
  "minimum_card_memory_mb": 7000
}
```

It gives 4 units at 512 policies per unit (16, 16, 16 and 12 cells × 2 reward settings × 16 seeds).

How units are cut:

1. A cell's seeds stay in one unit when cell × seeds × reward settings fits a unit; a unit then
   holds as many whole cells as fit, and the last unit takes the rest.
2. A cell larger than a unit is cut along its seeds; each piece is its own unit with
   `copy_seed_offset` set to its first seed.
3. The unit id spells the content: phase, environment, reward setting, method, the cell (one cell:
   its values; several: their places in the sweep's cell list), the seeds.
4. **Run seed.** `run_training.py` draws the action noise and the minibatch order of a whole process
   from its run seed, place by place. Two units with the same run seed and the same size would give
   the policies at equal places the same draws. Each unit therefore gets a run seed keyed by
   (environment, first seed, seed count): units with different seeds draw differently, and two
   methods trained on the same seeds draw alike.

## The unit file

`RUN/specifications/<unit>.json` holds the unit's identity (`unit_id`, `study`, `phase`, `method`,
`bonus`, `environment`, `reward_setting`, `workload`), its content (`cell_setting_names`, `cells`,
`cell_positions`, `first_seed`, `seed_count`, `policies`), its length (`iterations`,
`window_iterations`, `windows_expected`, `steps_per_policy`), `run_seed`, its size class
(`minimum_card_memory_mb`, `policies_per_unit` with its source), and what `run_training.py` gets:
`configuration_toml` (the whole text of the unit's configuration file: the base file with the
values of `configuration_changes` replaced, among them the cells as `setting_names` and
`parameter_settings`) and `options` (the reward setting and the description). The worker adds
`--run-dir`, `--unit-id`, `--configuration-file`, `--checkpoint-seconds 1200`,
`--time-limit-seconds <what is left of the job, less 10 minutes>` and `--no-aggregate`.

The cells go through the configuration file because `run_training.py --parameter-settings` takes
(learning rate, intrinsic weight) pairs only.

## Steps

```bash
PY="env PYTHONNOUSERSITE=1 /p/rlprojects/RND/.venvs/platform_jax/bin/python"
M=/p/rlprojects/RND/10_jax_exploration_platform/scripts/maze_campaign
RUN=/p/rlprojects/RND/10_jax_exploration_platform/runs/<run folder>
```

1. **Freeze the code.** Commit and push first (commit-before-submit rule); the command refuses code
   that is not committed. Jobs run the frozen copy, and units are built against it.

   ```bash
   $PY $M/freeze_code.py --run-dir $RUN
   ```

2. **Build units.** Look first, then write. Building the same description again adds nothing.
   A unit's configuration file lists every field of the trainer, so it must be built against the
   code copy that runs it: the builder reads the run folder's frozen copy. After a later freeze
   that changed the trainer's fields, `--replace-pending` rebuilds the pending units that never
   started; a unit that started keeps its file.

   ```bash
   $PY $M/build_units.py --run-dir $RUN --study <description>.json --dry-run
   $PY $M/build_units.py --run-dir $RUN --study <description>.json
   ```

3. **Submit workers on UVA** (both accounts). Look first, then send. New worker cards never exceed
   the pending units that have no worker yet. Order: nodes with free cards in the open partitions,
   most capable card first (these start now); the reservation's nodes; then jobs by card type that
   wait in the scheduler's queue. For a first small submission add `--max-new-cards 1`.

   ```bash
   $PY $M/submit.py --run-dir $RUN --phase <phase> --dry-run
   $PY $M/submit.py --run-dir $RUN --phase <phase>
   sacct -j <the new ids> -X -n -P -o JobID,State,Elapsed,NodeList      # within a minute of the submission
   ```

4. **Start the persistent monitor** (one processor on `nolim`, a pass every 20 minutes), in the same
   turn as the first submission, and create the session's 20-minute cron then too.

   ```bash
   $PY $M/submit.py --run-dir $RUN --monitor --top-up-phase <phase>
   $PY $M/monitor.py --run-dir $RUN                                     # one pass by hand
   ```

   `--top-up-phase` lets every pass submit workers for pending units that have none. Without it
   the monitor only reports them. `--complete-when-drained` writes `RUN/SWEEP_COMPLETE` once every
   unit is done or failed; without it the sentinel is written by hand, and the loop ends at the
   first pass that finds it.

5. **Add workers** at any time: step 3 again. A worker that finds nothing it can run ends within
   seconds. A job that was pinned to a node and still waits after a monitoring pass is named in the
   pass's warnings; cancel that one id and submit again:

   ```bash
   $PY $M/cancel_ledger.py --ledger $RUN/slurm/submitted_jobids_<phase>_<account>.txt --job-id <id> < /dev/null
   $PY $M/submit.py --run-dir $RUN --phase <phase>
   ```

6. **Remote servers** (H100 and H200 work; order MIT, Purdue Gautschi, TAMU ACES, NCSA DeltaAI).
   Needs `RUN/infrastructure/remote_fleet_servers.json` in the grid campaign's format, with
   `remote_run`, `code` and `python` set for this study; `unit_seconds` is the least training time
   a job must have left to start a unit (units continue from checkpoints, so 1800 is enough). No
   probe, benchmark or canary job runs on these servers.

   ```bash
   $PY $M/remote_fleet.py --run-dir $RUN stage --server mit --units 4 --match pointmaze --dry-run
   $PY $M/remote_fleet.py --run-dir $RUN submit --server mit --jobs 4 --pool preemptable --dry-run
   $PY $M/remote_fleet.py --run-dir $RUN tick          # every monitoring tick of the session
   $PY $M/remote_fleet.py --run-dir $RUN status
   ```

7. **Stop a sweep** (rule `submit-the-whole-queue-up-front.md`, "Cancelling a whole queue"):

   1. `touch $RUN/STOP`: no worker claims a further unit and no monitor pass submits a job. A unit
      that runs goes on until its job ends. A cancelled job's unit is asked to save its state (the
      scheduler waits 128 s before it ends the processes); if the save does not finish, the unit
      continues later from its last checkpoint, at most 20 minutes back.
   2. Take waiting units back from every remote server:
      `$PY $M/remote_fleet.py --run-dir $RUN take-back --server <s> --units 9999 --hold`.
   3. Cancel every ledger of the run, one call per file, the monitor's ledger last:

      ```bash
      for ledger in $RUN/slurm/submitted_jobids_*_sl5nw.txt $RUN/slurm/submitted_jobids_*_yuxinchen.txt; do
          $PY $M/cancel_ledger.py --ledger "$ledger" < /dev/null
      done
      $PY $M/cancel_ledger.py --ledger $RUN/slurm/submitted_jobids_<server>.txt --server <server>
      ```

   4. Delete the session's cron monitor. Never `scancel -u`, `-t` or `-n`.
   5. To go on later: remove `STOP`, run a monitor pass (it returns the units of the cancelled jobs
      to pending), and submit workers; every unit continues from its checkpoint.

## What a worker does with a unit

| how the unit's process ended | where the unit goes | counts toward the 3 failures |
|---|---|---|
| its shard ends with `unit_complete` | `done`; one line is added to<br>`P/device/observed/<workload>.jsonl` | — |
| stopped with its state saved<br>(time limit, stop signal) | `pending`; a later job continues it | no |
| ran out of card memory | `pending`, for cards with more memory<br>than the one it failed on | no |
| any other failure | `pending`; `failed` at the third failure,<br>with the log's path | yes |
| its job ended without a result | `pending` at the next monitor pass;<br>`failed` after 6 such ends | no |

After a unit that did not complete, the worker takes no further unit. Every attempt is kept in the
unit's record (`history`: job, node, card, code copy, start, end, outcome, log).

## Tests

```bash
cd /p/rlprojects/RND/10_jax_exploration_platform
PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 /p/rlprojects/RND/.venvs/platform_jax/bin/python \
    -m unittest discover -s tests/maze_campaign -p "test_maze_*.py"
```

These import no numerical library and start no trainer (a stand-in script plays `run_training.py`).
`tests/maze_campaign/compute_node_check_unit_configuration_with_trainer.py` imports the trainer and
runs on a compute node only, through the test job
(`bash <benchmark run folder>/infrastructure/submit_tests.sh <that file>`).

## Taken from the grid campaign

Imported, not copied (`grid_campaign.py` loads them from `scripts/grid_initialization_campaign/`):

1. `common.py`: `read_json`, `write_json`, `append_json`, `queue_lock`, `now`, `pacific`, `seed_for`,
   `own_live_jobs`, and the paths `PROJECT`, `PLATFORM`, `ACCOUNTS`.
2. `submit.py`: `allowed_seconds`, `node_resources`, `capacity_budgets`, `reservation_budget`,
   `owner_reservation`, `slurm_fields`, `RESOURCE_FILE`.
3. `worker.py`: `hardware`.
4. `remote_fleet.py`: every command and helper except the four replaced parts named at the head of
   this folder's `remote_fleet.py`.

Written new here: the unit builder, the queue (the grid campaign's `claim`, `finish` and
`recover_claims` do not return a stopped unit to pending or count failures), the worker's loop,
the submission plan, and the monitor's pass.

**Adapted and not exercised:** `remote_fleet.py` and the worker's `--remote-records` records have
not run against any server; `submit.py`'s calls to the scheduler and `submit_as.sh` have not sent
a job; `worker.py` has not started the real `run_training.py`. The first submission of each is its
first real run.
