# experiments

Whole-experiment files that name one environment, one agent, one bonus and one runtime.

Six files record earlier runs. The six files whose names hold `_example_` start a small run of
the PointMaze and AntMaze study that began 2026-11-11 (CleanRL PPO with random network
distillation, the episode ending at the goal, reward 1 on arrival); adding
`--reward-setting step_cost` to the command runs the other reward setting. The AntMaze examples
whose names hold `_warp-physics_` step the ant with MuJoCo Warp (`physics = "warp"` under
`[environment]`) and run only with `/p/rlprojects/RND/.venvs/platform_jax_warp/bin/python`; the
other AntMaze files use MJX's JAX implementation (`physics = "jax"`).

The physics the AntMaze baseline runs with (a time step of 0.005 s with 10 physics steps per
environment step, the fault guard at 30 m/s and 100 rad/s) has one file per map, the three files
whose names hold `_time-step-0.005_`:

| map | start cell (row, column) | goal cell | step limit |
|---|---|---|---|
| U-maze | (3, 1) | (1, 1) | 700 |
| Medium | (6, 1) | (1, 6) | 1,000 |
| Large | (7, 1) | (1, 10) | 1,000 |

The Medium and Large files are the U-maze file with these four values and `runtime.environment`
changed, and nothing else (`tests/integration/test_experiment_configuration_files.py` checks it
against the presets of `src/exploration_platform/envs/antmaze/am_common.py`).

A file may hold one optional table, `[bonus_options]`, with the options of the bonus family that
it changes: the fields of `RNDConfig` in `src/exploration_platform/bonuses/rnd/config.py`, for
example `even_start = true`. On the command line, `--bonus-options name=value,...` replaces the
same names. No file here holds the table.
