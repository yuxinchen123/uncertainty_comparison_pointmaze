#!/bin/bash
# EXPECTED_MINUTES=12
# Training iteration with the Warp AntMaze physics on the H100 NVL, unit 11, with the ceiling in the model:
# 2,048 policies x 16 environments (32,768 environments, not measured before), one cell alone because building
# and priming 2,048 policies takes about 4 minutes on serval02. Its record goes to records_after_the_ceiling/.
export RECORDS_FOLDER=records_after_the_ceiling
exec bash /p/rlprojects/RND/10_jax_exploration_platform/benchmark_runs/2026-11-12-00-17_pointmaze-antmaze_cleanrl-ppo-rnd_speed-30-ideas_screen-15-minutes-per-idea-per-card_h100-nvl_rtx-5080_rtx-a4500/profiles/antmaze_warp_training/units/run_cells.sh "h100-nvl unit 11, after the ceiling, 2,048 policies x 16 environments" h100-nvl "NVIDIA H100 NVL" 2048x16
