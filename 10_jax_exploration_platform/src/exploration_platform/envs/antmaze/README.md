# antmaze

The Gymnasium Ant in the three maze maps (U-maze, medium, large), specified in `spec.md`.

| file | what it holds |
|---|---|
| `am_common.py` | the configuration (`AntMazeConfig`), the three presets, the model's XML |
| `mjx_antmaze.py` | `JaxAntMaze`: the physics stepped by MJX's JAX implementation |
| `warp_antmaze.py` | `WarpAntMaze`: the same model stepped by MuJoCo Warp, called from JAX |

`AntMazeConfig.physics` chooses between the two: `"jax"` (the default, and the stepper of every
run before 2026-11-12) or `"warp"`. The composer (`training/compose.py`) reads it; the training
command sets it with `--physics` or from a configuration file's `[environment]`.

The Warp stepper needs an interpreter with warp-lang
(`/p/rlprojects/RND/.venvs/platform_jax_warp/bin/python`). What differs from the JAX stepper:

1. **One contact pool for all environments**, of `warp_contact_slots_per_environment` (48) slots
   per environment, and at most `warp_constraint_rows_per_environment` (160) constraint rows per
   environment. MuJoCo Warp drops what does not fit; `raise_on_overflow` then raises, and the
   training command calls it after priming, at every window end, before every checkpoint, and
   between window ends about once a minute.
2. **A kernel folder** on node-local storage (`prepare_kernel_folder`), created if missing.
3. **Non-finite states**: an environment whose positions or velocities are non-finite when the
   step starts, or after any of its physics steps, is respawned and counted. The test at the
   start is needed because MuJoCo Warp turns a torso quaternion that holds a NaN into a valid
   one.
4. **Repeatability**: bit for bit on processors. On a card the order of the contacts in the pool
   changes from run to run, so two runs of the same program differ.

Measurements and checks:
`benchmark_runs/2026-11-12-00-17_pointmaze-antmaze_cleanrl-ppo-rnd_speed-30-ideas_*/profiles/antmaze_warp/antmaze_warp.md`
(the environment) and `.../profiles/antmaze_warp_training/antmaze_warp_training.md` (the whole
training iteration).

The two counting states of the plan, `gt_position_maze_cell` (the maze cell the ant is in) and
`gt_position_1m` (position rounded to a 1 metre grid), are not implemented.
