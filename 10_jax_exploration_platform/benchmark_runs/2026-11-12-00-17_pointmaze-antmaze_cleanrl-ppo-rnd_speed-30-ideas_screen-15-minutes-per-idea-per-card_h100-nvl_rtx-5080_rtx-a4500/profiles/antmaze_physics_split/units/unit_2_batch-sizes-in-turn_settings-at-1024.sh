#!/bin/bash
# EXPECTED_MINUTES=8
# AntMaze physics split, unit 2: the environment step at 1,024 to 16,384 environments timed in turn in one
# cell (unit 1 read each size in its own cell, and the same function differed by up to 5% between cells),
# then the solver and contact settings at 1,024 environments.
exec bash /p/rlprojects/RND/10_jax_exploration_platform/benchmark_runs/2026-11-12-00-17_pointmaze-antmaze_cleanrl-ppo-rnd_speed-30-ideas_screen-15-minutes-per-idea-per-card_h100-nvl_rtx-5080_rtx-a4500/profiles/antmaze_physics_split/units/run_unit.sh \
    "unit 2 batch sizes in turn, settings at 1,024" batches variations --variation-batches 1024
