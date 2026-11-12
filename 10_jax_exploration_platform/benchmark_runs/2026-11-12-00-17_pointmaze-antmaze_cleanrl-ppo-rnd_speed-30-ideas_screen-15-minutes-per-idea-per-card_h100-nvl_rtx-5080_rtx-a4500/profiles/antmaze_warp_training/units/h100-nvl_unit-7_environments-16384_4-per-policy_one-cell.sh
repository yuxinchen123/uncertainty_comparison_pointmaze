#!/bin/bash
# EXPECTED_MINUTES=15
# Training iteration with the Warp AntMaze physics on the H100 NVL, unit 7: 4,096 policies x 4 environments
# (16,384 environments in total), alone in its unit: building and priming 4,096 policies took 469 s in unit 1
# (about 0.115 s per policy), so two such cells do not fit into 15 minutes.
exec bash /p/rlprojects/RND/10_jax_exploration_platform/benchmark_runs/2026-11-12-00-17_pointmaze-antmaze_cleanrl-ppo-rnd_speed-30-ideas_screen-15-minutes-per-idea-per-card_h100-nvl_rtx-5080_rtx-a4500/profiles/antmaze_warp_training/units/run_cells.sh "h100-nvl unit 7, 4,096 policies x 4 environments, one cell" h100-nvl "NVIDIA H100 NVL" 4096x4
