#!/bin/bash
# EXPECTED_MINUTES=11
# Blow-up diagnosis, unit 9, timing on serval02 card 1: the training physics (implicitfast, time step 0.01 s)
# with the fault guard off and with the guard on at 30 m/s; 6 whole training iterations timed in each cell.
exec bash /p/rlprojects/RND/10_jax_exploration_platform/benchmark_runs/2026-11-12-00-17_pointmaze-antmaze_cleanrl-ppo-rnd_speed-30-ideas_screen-15-minutes-per-idea-per-card_h100-nvl_rtx-5080_rtx-a4500/profiles/antmaze_warp_blow_up/units/run_timing_unit.sh serval02-card-1_unit-9_timing_training-physics_guard-off-and-on "implicitfast_guard-off:--fault-torso-speed 0" "implicitfast_guard-at-30:--fault-torso-speed 30"
