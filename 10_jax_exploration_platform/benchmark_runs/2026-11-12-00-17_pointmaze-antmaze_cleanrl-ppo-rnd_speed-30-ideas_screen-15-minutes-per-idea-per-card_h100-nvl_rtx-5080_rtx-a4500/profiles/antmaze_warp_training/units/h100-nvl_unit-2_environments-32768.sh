#!/bin/bash
# EXPECTED_MINUTES=15
# Training iteration with the Warp AntMaze physics on the H100 NVL, unit 2: 32,768 environments in total,
# with 16, 4 and 128 environments per policy.
exec bash /p/rlprojects/RND/10_jax_exploration_platform/benchmark_runs/2026-11-12-00-17_pointmaze-antmaze_cleanrl-ppo-rnd_speed-30-ideas_screen-15-minutes-per-idea-per-card_h100-nvl_rtx-5080_rtx-a4500/profiles/antmaze_warp_training/units/run_cells.sh "h100-nvl unit 2, 32,768 environments in total" h100-nvl "NVIDIA H100 NVL" 2048x16 8192x4 256x128
