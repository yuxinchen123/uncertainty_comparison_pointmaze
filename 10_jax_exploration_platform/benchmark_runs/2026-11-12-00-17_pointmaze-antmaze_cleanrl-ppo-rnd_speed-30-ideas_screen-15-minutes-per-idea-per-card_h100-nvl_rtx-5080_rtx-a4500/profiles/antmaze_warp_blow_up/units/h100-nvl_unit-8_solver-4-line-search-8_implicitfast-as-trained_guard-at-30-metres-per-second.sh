#!/bin/bash
# EXPECTED_MINUTES=10
# Blow-up diagnosis, unit 8: the checkpoint's policies rolled out for 300 s with the physics of training
# (implicitfast, 4 solver and 8 line-search iterations) and the fault guard on at 30 m/s: how many
# environments the guard respawns, the largest speed any physics step still shows, the histogram of the
# torso's angular speed with the training integrator, and the cost of the guard (4 training iterations timed).
exec bash /p/rlprojects/RND/10_jax_exploration_platform/benchmark_runs/2026-11-12-00-17_pointmaze-antmaze_cleanrl-ppo-rnd_speed-30-ideas_screen-15-minutes-per-idea-per-card_h100-nvl_rtx-5080_rtx-a4500/profiles/antmaze_warp_blow_up/units/run_rollout_unit.sh h100-nvl_unit-8_solver-4-line-search-8_implicitfast-as-trained_guard-at-30-metres-per-second 4 8 300 4 --fault-torso-speed 30
