#!/bin/bash
# EXPECTED_MINUTES=10
# Blow-up diagnosis, unit 5: the checkpoint's policies rolled out for 300 s with 4 solver and 8 line-search
# iterations and these changes to the physics: --integrator RK4. Every onset of a torso speed above 100 m/s is kept
# with its lead-up; then 4 whole training iterations are timed.
exec bash /p/rlprojects/RND/10_jax_exploration_platform/benchmark_runs/2026-11-12-00-17_pointmaze-antmaze_cleanrl-ppo-rnd_speed-30-ideas_screen-15-minutes-per-idea-per-card_h100-nvl_rtx-5080_rtx-a4500/profiles/antmaze_warp_blow_up/units/run_rollout_unit.sh h100-nvl_unit-5_solver-4-line-search-8_rk4 4 8 300 4 --integrator RK4
