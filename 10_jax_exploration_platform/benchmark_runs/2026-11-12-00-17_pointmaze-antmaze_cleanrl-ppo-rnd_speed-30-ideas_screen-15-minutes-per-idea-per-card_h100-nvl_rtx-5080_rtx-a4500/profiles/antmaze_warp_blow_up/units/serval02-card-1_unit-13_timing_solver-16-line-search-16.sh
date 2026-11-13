#!/bin/bash
# EXPECTED_MINUTES=6
# Blow-up diagnosis, unit 13, timing on serval02 card 1: implicitfast at 0.01 s with 16 solver and 16
# line-search iterations (the replays on processors say more iterations do not remove the fault; this unit
# gives the rate on the card and the cost); 6 whole training iterations timed.
exec bash /p/rlprojects/RND/10_jax_exploration_platform/benchmark_runs/2026-11-12-00-17_pointmaze-antmaze_cleanrl-ppo-rnd_speed-30-ideas_screen-15-minutes-per-idea-per-card_h100-nvl_rtx-5080_rtx-a4500/profiles/antmaze_warp_blow_up/units/run_timing_unit.sh serval02-card-1_unit-13_timing_solver-16-line-search-16 "implicitfast_solver-16-line-search-16:--solver-iterations 16 --ls-iterations 16"
