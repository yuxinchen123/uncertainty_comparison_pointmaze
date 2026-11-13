#!/bin/bash
# EXPECTED_MINUTES=10
# Blow-up diagnosis, unit 7: the checkpoint's policies rolled out for 300 s with 4 solver and 8 line-search
# iterations and these changes to the physics: --timestep 0.005 --frame-skip 10. Every onset of a torso speed above 100 m/s is kept
# with its lead-up; then 4 whole training iterations are timed.
exec bash /p/rlprojects/RND/10_jax_exploration_platform/benchmark_runs/2026-11-12-00-17_pointmaze-antmaze_cleanrl-ppo-rnd_speed-30-ideas_screen-15-minutes-per-idea-per-card_h100-nvl_rtx-5080_rtx-a4500/profiles/antmaze_warp_blow_up/units/run_rollout_unit.sh h100-nvl_unit-7_solver-4-line-search-8_implicitfast-time-step-0-005 4 8 300 4 --timestep 0.005 --frame-skip 10
