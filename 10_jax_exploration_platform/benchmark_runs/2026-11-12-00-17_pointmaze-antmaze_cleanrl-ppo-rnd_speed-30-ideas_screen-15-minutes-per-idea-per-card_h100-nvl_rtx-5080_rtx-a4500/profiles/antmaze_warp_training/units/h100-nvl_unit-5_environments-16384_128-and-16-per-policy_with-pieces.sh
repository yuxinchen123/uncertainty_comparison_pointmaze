#!/bin/bash
# EXPECTED_MINUTES=15
# Training iteration with the Warp AntMaze physics on the H100 NVL, unit 5: 16,384 environments in total with
# 128 and with 16 environments per policy. Two cells, so that each has time for the rollout and the update
# timed alone (one cell takes about 5 minutes on serval02: 80 s to load the libraries, 140 s to build and prime).
exec bash /p/rlprojects/RND/10_jax_exploration_platform/benchmark_runs/2026-11-12-00-17_pointmaze-antmaze_cleanrl-ppo-rnd_speed-30-ideas_screen-15-minutes-per-idea-per-card_h100-nvl_rtx-5080_rtx-a4500/profiles/antmaze_warp_training/units/run_cells.sh "h100-nvl unit 5, 16,384 environments in total, two cells with the pieces timed alone" h100-nvl "NVIDIA H100 NVL" 128x128 1024x16
