#!/bin/bash
# EXPECTED_MINUTES=14
# Training iteration with the Warp AntMaze physics on an H100 NVL, unit 12, with the physics the AntMaze
# baseline runs with since 2026-11-12: time step 0.005 s, 10 physics steps per environment step, the fault
# guard at 30 m/s and 100 rad/s (the configuration file below). 16,384 environments in both cells:
# 128 policies x 128 environments, then 1,024 policies x 16 environments. Records go to
# records_stable_physics_time-step-0-005_frame-skip-10_fault-guard/.
export RECORDS_FOLDER=records_stable_physics_time-step-0-005_frame-skip-10_fault-guard
export CONFIGURATION_FILE=/p/rlprojects/RND/10_jax_exploration_platform/configs/experiments/antmaze_umaze_warp-physics_time-step-0.005_frame-skip-10_fault-guard-30-metres-and-100-radians-per-second_cleanrl-ppo-rnd_goal-termination_reward-arrival-plus-one_example_steps-204800.toml
exec bash /p/rlprojects/RND/10_jax_exploration_platform/benchmark_runs/2026-11-12-00-17_pointmaze-antmaze_cleanrl-ppo-rnd_speed-30-ideas_screen-15-minutes-per-idea-per-card_h100-nvl_rtx-5080_rtx-a4500/profiles/antmaze_warp_training/units/run_cells.sh "serval02 card 1 unit 12, stable physics, 16,384 environments, 128 and 16 per policy" h100-nvl "NVIDIA H100 NVL" 128x128 1024x16
