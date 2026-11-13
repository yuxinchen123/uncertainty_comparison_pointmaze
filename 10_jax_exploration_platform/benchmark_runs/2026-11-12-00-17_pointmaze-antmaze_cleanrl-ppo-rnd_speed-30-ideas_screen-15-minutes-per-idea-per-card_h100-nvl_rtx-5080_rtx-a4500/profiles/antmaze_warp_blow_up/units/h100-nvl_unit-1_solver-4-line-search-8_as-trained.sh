#!/bin/bash
# EXPECTED_MINUTES=12
# Blow-up diagnosis, unit 1: the checkpoint's policies rolled out with the solver settings of training
# (4 solver iterations, 8 line-search iterations) for 300 s, every onset of a torso speed above 100 m/s kept
# with its lead-up; then 4 whole training iterations timed.
exec bash /p/rlprojects/RND/10_jax_exploration_platform/benchmark_runs/2026-11-12-00-17_pointmaze-antmaze_cleanrl-ppo-rnd_speed-30-ideas_screen-15-minutes-per-idea-per-card_h100-nvl_rtx-5080_rtx-a4500/profiles/antmaze_warp_blow_up/units/run_rollout_unit.sh h100-nvl_unit-1_solver-4-line-search-8_as-trained 4 8 300 4
