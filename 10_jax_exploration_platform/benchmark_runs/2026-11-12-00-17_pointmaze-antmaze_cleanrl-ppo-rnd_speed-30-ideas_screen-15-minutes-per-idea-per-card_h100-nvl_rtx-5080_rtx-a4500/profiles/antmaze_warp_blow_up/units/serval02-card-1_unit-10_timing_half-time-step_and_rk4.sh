#!/bin/bash
# EXPECTED_MINUTES=13
# Blow-up diagnosis, unit 10, timing on serval02 card 1: implicitfast with a time step of 0.005 s (10 physics
# steps per environment step), and RK4 at 0.01 s; 6 whole training iterations timed in each cell.
exec bash /p/rlprojects/RND/10_jax_exploration_platform/benchmark_runs/2026-11-12-00-17_pointmaze-antmaze_cleanrl-ppo-rnd_speed-30-ideas_screen-15-minutes-per-idea-per-card_h100-nvl_rtx-5080_rtx-a4500/profiles/antmaze_warp_blow_up/units/run_timing_unit.sh serval02-card-1_unit-10_timing_half-time-step_and_rk4 "implicitfast_time-step-0-005:--timestep 0.005 --frame-skip 10" "rk4_time-step-0-01:--integrator RK4"
