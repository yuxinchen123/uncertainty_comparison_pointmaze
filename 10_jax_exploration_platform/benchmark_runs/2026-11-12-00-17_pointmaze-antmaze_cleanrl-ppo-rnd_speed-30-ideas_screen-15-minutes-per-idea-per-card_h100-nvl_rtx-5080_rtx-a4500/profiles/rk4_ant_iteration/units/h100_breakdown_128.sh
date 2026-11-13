#!/bin/bash
# EXPECTED_MINUTES=15
# serval05 profile unit: the RK4 AntMaze physics step split natively at 2,048 and 512 worlds, its
# same-number variants, then the whole iteration and its pieces at 128 policies x 16 environments.
[ "$(hostname -s)" = serval05 ] || { echo "this unit runs on serval05 only"; exit 1; }
exec bash /p/rlprojects/RND/10_jax_exploration_platform/benchmark_runs/2026-11-12-00-17_pointmaze-antmaze_cleanrl-ppo-rnd_speed-30-ideas_screen-15-minutes-per-idea-per-card_h100-nvl_rtx-5080_rtx-a4500/profiles/rk4_ant_iteration/units/run_on_card.sh breakdown_h100-nvl_serval05_copies-128 breakdown.py --copies 128 \
    --parts native,native_variants,iteration --native-worlds 2048,512 --budget-seconds 820 \
    --out /p/rlprojects/RND/10_jax_exploration_platform/benchmark_runs/2026-11-12-00-17_pointmaze-antmaze_cleanrl-ppo-rnd_speed-30-ideas_screen-15-minutes-per-idea-per-card_h100-nvl_rtx-5080_rtx-a4500/profiles/rk4_ant_iteration/records/breakdown_h100-nvl_serval05_copies-128.json
