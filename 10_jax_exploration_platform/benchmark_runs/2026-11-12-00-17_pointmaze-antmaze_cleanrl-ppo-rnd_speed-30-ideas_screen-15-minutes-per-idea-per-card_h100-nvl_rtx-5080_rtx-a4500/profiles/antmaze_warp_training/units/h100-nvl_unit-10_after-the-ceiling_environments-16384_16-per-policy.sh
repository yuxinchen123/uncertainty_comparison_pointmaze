#!/bin/bash
# EXPECTED_MINUTES=8
# Training iteration with the Warp AntMaze physics on the H100 NVL, unit 10, with the ceiling in the model:
# 1,024 policies x 16 environments, the cell of unit 5 measured again, with the rollout and the update timed
# alone. Its record goes to records_after_the_ceiling/.
export RECORDS_FOLDER=records_after_the_ceiling
exec bash /p/rlprojects/RND/10_jax_exploration_platform/benchmark_runs/2026-11-12-00-17_pointmaze-antmaze_cleanrl-ppo-rnd_speed-30-ideas_screen-15-minutes-per-idea-per-card_h100-nvl_rtx-5080_rtx-a4500/profiles/antmaze_warp_training/units/run_cells.sh "h100-nvl unit 10, after the ceiling, 1,024 policies x 16 environments" h100-nvl "NVIDIA H100 NVL" 1024x16
