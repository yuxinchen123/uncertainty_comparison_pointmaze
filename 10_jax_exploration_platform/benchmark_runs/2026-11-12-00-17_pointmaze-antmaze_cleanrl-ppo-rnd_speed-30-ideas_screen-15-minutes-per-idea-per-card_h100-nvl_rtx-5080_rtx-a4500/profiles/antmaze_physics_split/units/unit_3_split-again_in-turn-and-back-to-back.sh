#!/bin/bash
# EXPECTED_MINUTES=8
# AntMaze physics split, unit 3: the split at 4,096 and 1,024 environments again, each function timed in
# turn and then back to back. In unit 1 the whole-step functions of the split cells read 5% to 7% above the
# same function in the cells that time only whole steps; this unit checks whether the order of timing is why.
exec bash /p/rlprojects/RND/10_jax_exploration_platform/benchmark_runs/2026-11-12-00-17_pointmaze-antmaze_cleanrl-ppo-rnd_speed-30-ideas_screen-15-minutes-per-idea-per-card_h100-nvl_rtx-5080_rtx-a4500/profiles/antmaze_physics_split/units/run_unit.sh \
    "unit 3 split again, in turn and back to back" split
