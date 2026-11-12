#!/bin/bash
# EXPECTED_MINUTES=15
# AntMaze physics split, unit 1: the split at 4,096 and 1,024 environments, the Large map, the solver and
# contact settings, the wide batches, then the unrolled frame loop and the compiler option if time is left.
exec bash /p/rlprojects/RND/10_jax_exploration_platform/benchmark_runs/2026-11-12-00-17_pointmaze-antmaze_cleanrl-ppo-rnd_speed-30-ideas_screen-15-minutes-per-idea-per-card_h100-nvl_rtx-5080_rtx-a4500/profiles/antmaze_physics_split/units/run_unit.sh \
    "unit 1 split, large map, settings, wide batch, unroll" split large variations wide unroll
