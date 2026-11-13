#!/bin/bash
# EXPECTED_MINUTES=10
# Blow-up diagnosis, unit 3: the checkpoint's policies (trained with 4 solver and 8 line-search iterations)
# rolled out with 16 solver iterations and 16 line-search iterations for 300 s, every onset of a torso speed
# above 100 m/s kept with its lead-up; then 4 whole training iterations timed.
exec bash /p/rlprojects/RND/10_jax_exploration_platform/benchmark_runs/2026-11-12-00-17_pointmaze-antmaze_cleanrl-ppo-rnd_speed-30-ideas_screen-15-minutes-per-idea-per-card_h100-nvl_rtx-5080_rtx-a4500/profiles/antmaze_warp_blow_up/units/run_rollout_unit.sh h100-nvl_unit-3_solver-16-line-search-16 16 16 300 4
