# scripts

Command-line entry points for the platform.

| script | what it does |
|---|---|
| `create_platform_jax_env.sh` | builds the shared environment `/p/rlprojects/RND/.venvs/platform_jax` — python 3.12 with `jax[cuda12]` 0.11.0 and numpy 2.5.2, `--copy` and group-readable. Run once; it is kept here so the environment can be rebuilt exactly |
| `run_training.py` | runs one training configuration and writes a run folder under `runs/` (the folder layout is described in `../runs/README.md`) |

The environment registry is `/p/rlprojects/RND/.venvs/ENVS.md`, and `platform_jax`'s own
`PROVENANCE.md` sits inside the environment directory. Neither is in git — the whole `.venvs/`
directory is ignored by this repository — so the build script here is the tracked record of how the
environment was made.

Every python invocation sets `PYTHONNOUSERSITE=1`, so a package in `~/.local` cannot shadow the
environment's own. Graphics-card work on serval05 goes through the lock, one command at a time:

```
bash /p/rlprojects/RND/09_parallelization/locks/gpu_run.sh "<command>"
```

`run_training.py --parameter-settings settings.json --bonus rnd_next_state
--copies-per-cell 128` trains exactly the pairs listed for that algorithm. For example:

```json
{
  "rnd_next_state": [[0.001, 10], [0.0001, 100]],
  "gt_position_velocity_sqrt": [[0.001, 1]],
  "none": [[0.001, 0]]
}
```

Each pair contains a learning rate and an intrinsic-advantage weight. Algorithms can have
different pairs; no extra Cartesian-product settings are created. Copies share seed indices
across settings. The no-bonus policy control uses weight zero.

Options added for the PointMaze and AntMaze study (2026-11-12); every default leaves a run as it
was:

| option | what it does |
|---|---|
| `--clip-coef`, `--prime-iterations` | the clip coefficient of the epoch-minibatch update and the number of random-action rollouts before training (CleanRL: 0.1 and 50) |
| `--entropy-coefficients`, `--clip-coefficients`, `--value-coefficients` | comma-separated lists that are crossed with `--learning-rates` and `--intrinsic-weights`; each cell of the cross product gets `--copies-per-cell` copies on the same seeds |
| `--reward-setting` | `arrival_plus_one` (the episode ends at the goal, which pays 1), `step_cost` (every step costs 1 / step limit, arrival pays 0), or `arrival_plus_one_and_step_cost` (both in one process, each cell once per reward on the same seeds); the name goes into the environment specification |
| `--episode-log` | per copy and window: episodes ended, arrivals, sums of returns, lengths and steps to the goal, first training step with an arrival; required when the episode ends at the goal |
| `--gather-minibatches-per-epoch` | holds one shuffled copy of the batch at a time in the epoch-minibatch update |
| `--checkpoint-seconds`, `--time-limit-seconds` | save the unit's whole state to `checkpoints/<unit_id>.npz` at window ends, on a stop signal and at the time limit; the same command then continues the unit |
| `--bonus-options` | options of the bonus family the unit changes, as `name=value,...`<br>(the fields of `RNDConfig` in `src/exploration_platform/bonuses/rnd/config.py`);<br>a pair replaces the same name of a configuration file's `[bonus_options]` table.<br>One value for the whole process. `unit_start` holds every option under `bonus_options`.<br>`record_bonus_on_maze_grid=true` adds the maze-grid fields to every window record<br>and one `maze_grid_at_start` record |

| `--physics` | AntMaze only: `jax` (default; MJX's JAX implementation) or `warp` (MuJoCo Warp, about 8 times as fast on an H100 NVL). `warp` needs the interpreter `/p/rlprojects/RND/.venvs/platform_jax_warp/bin/python`; the environment specification then holds `_physics-warp` |

`configs/experiments/` holds one example file per family for that study, and one for AntMaze with
the MuJoCo Warp physics; a run starts with
`run_training.py --run-dir <folder> --unit-id <name> --configuration-file <file>`.

A unit with the Warp physics:

1. writes MuJoCo Warp's kernel folder into its log. The folder is node-local and created if
   missing: `WARP_CACHE_PATH` when the job sets it, else `/localtmp/<user>/warp_kernel_cache`,
   else `<temporary folder>/<user>/warp_kernel_cache`;
2. sizes the contact pool that all environments share from the batch: 48 slots per environment
   (`warp_contact_slots_per_environment` in the file's `[environment]`), and at most 512 constraint
   rows per environment (`warp_constraint_rows_per_environment`);
3. records the largest pool use in every window record and in `unit_complete` (`physics_pool`);
4. stops with an error, written to its log as `STOPPED: ...`, when MuJoCo Warp dropped a contact
   or a constraint row. The check runs after priming, at every window end, before every
   checkpoint, and between window ends about once a minute, so no record and no checkpoint is
   written after a drop. The unit must then start again from its beginning with a larger pool;
5. saves and continues from checkpoints like every other unit. On processors the continued unit
   equals the uninterrupted one bit for bit. On a card two runs of the same unit already differ,
   because the contacts enter the shared pool in an order that changes from run to run.

Separate UVA programs use `--no-aggregate`; `aggregate_selected_run.py` publishes their
combined scores and curves, and `monitor_selected_run.py` monitors progress on a compute node.
`write_corrected_report.py` reads that aggregate to generate the corrected results table and
figures. The selected report run uses one 128-copy setting per queue unit; the shared GPU
manager claims those units as cards become available.
