#!/bin/bash
# EXPECTED_MINUTES=6
# Blow-up diagnosis, unit 11, timing on serval02 card 1: the training physics (implicitfast, time step 0.01 s)
# with both bounds of the fault guard on, 30 m/s and 100 rad/s; 6 whole training iterations timed.
exec bash /p/rlprojects/RND/10_jax_exploration_platform/benchmark_runs/2026-11-12-00-17_pointmaze-antmaze_cleanrl-ppo-rnd_speed-30-ideas_screen-15-minutes-per-idea-per-card_h100-nvl_rtx-5080_rtx-a4500/profiles/antmaze_warp_blow_up/units/run_timing_unit.sh serval02-card-1_unit-11_timing_guard-at-30-metres-and-100-radians-per-second "implicitfast_guard-at-30-and-100:--fault-torso-speed 30 --fault-torso-angular-speed 100"
