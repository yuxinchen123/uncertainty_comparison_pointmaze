# experiments

Whole-experiment files that name one environment, one agent, one bonus and one runtime.

Six files record earlier runs. The three files whose names hold `_example_` start a small run of
the PointMaze and AntMaze study that began 2026-11-11 (CleanRL PPO with random network
distillation, the episode ending at the goal, reward 1 on arrival); adding
`--reward-setting step_cost` to the command runs the other reward setting. The AntMaze example
whose name holds `_warp-physics_` steps the ant with MuJoCo Warp (`physics = "warp"` under
`[environment]`) and runs only with `/p/rlprojects/RND/.venvs/platform_jax_warp/bin/python`; the
other AntMaze files use MJX's JAX implementation (`physics = "jax"`).

A file may hold one optional table, `[bonus_options]`, with the options of the bonus family that
it changes: the fields of `RNDConfig` in `src/exploration_platform/bonuses/rnd/config.py`, for
example `even_start = true`. On the command line, `--bonus-options name=value,...` replaces the
same names. No file here holds the table.
