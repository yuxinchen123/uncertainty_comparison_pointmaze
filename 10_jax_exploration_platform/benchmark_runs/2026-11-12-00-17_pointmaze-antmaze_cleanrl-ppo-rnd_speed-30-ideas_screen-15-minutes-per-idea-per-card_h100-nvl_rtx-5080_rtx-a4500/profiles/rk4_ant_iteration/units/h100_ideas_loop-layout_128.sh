#!/bin/bash
# EXPECTED_MINUTES=15
# serval05 profile unit: loop-layout ideas of the RK4 AntMaze iteration at 128 policies x 16 environments,
# the unit's program and each variant built in one process and timed in alternating rounds.
[ "$(hostname -s)" = serval05 ] || { echo "this unit runs on serval05 only"; exit 1; }
exec bash /p/rlprojects/RND/10_jax_exploration_platform/benchmark_runs/2026-11-12-00-17_pointmaze-antmaze_cleanrl-ppo-rnd_speed-30-ideas_screen-15-minutes-per-idea-per-card_h100-nvl_rtx-5080_rtx-a4500/profiles/rk4_ant_iteration/units/run_on_card.sh ideas_loop-layout_h100-nvl_serval05_copies-128 ideas.py --copies 128 \
    --variants frame-unroll-5,rollout-unroll-1,rollout-unroll-4_frame-unroll-5,rollout-unroll-32_frame-unroll-5 \
    --rounds 8 --budget-seconds 800 --out /p/rlprojects/RND/10_jax_exploration_platform/benchmark_runs/2026-11-12-00-17_pointmaze-antmaze_cleanrl-ppo-rnd_speed-30-ideas_screen-15-minutes-per-idea-per-card_h100-nvl_rtx-5080_rtx-a4500/profiles/rk4_ant_iteration/records/ideas_loop-layout_h100-nvl_serval05_copies-128.json
