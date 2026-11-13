#!/bin/bash
# EXPECTED_MINUTES=10
# serval05 profile unit: the whole RK4 AntMaze iteration and its pieces at 32 policies x 16 environments
# (with the 128-policy unit, splits each piece into a fixed part and a part per policy).
[ "$(hostname -s)" = serval05 ] || { echo "this unit runs on serval05 only"; exit 1; }
exec bash /p/rlprojects/RND/10_jax_exploration_platform/benchmark_runs/2026-11-12-00-17_pointmaze-antmaze_cleanrl-ppo-rnd_speed-30-ideas_screen-15-minutes-per-idea-per-card_h100-nvl_rtx-5080_rtx-a4500/profiles/rk4_ant_iteration/units/run_on_card.sh breakdown_h100-nvl_serval05_copies-32 breakdown.py --copies 32 \
    --parts iteration --budget-seconds 800 --out /p/rlprojects/RND/10_jax_exploration_platform/benchmark_runs/2026-11-12-00-17_pointmaze-antmaze_cleanrl-ppo-rnd_speed-30-ideas_screen-15-minutes-per-idea-per-card_h100-nvl_rtx-5080_rtx-a4500/profiles/rk4_ant_iteration/records/breakdown_h100-nvl_serval05_copies-32.json
