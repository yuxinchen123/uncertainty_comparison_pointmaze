#!/bin/bash
# EXPECTED_MINUTES=12
# serval05 profile unit: MuJoCo Warp's solver iterations as a plain loop of 4 instead of the conditional graph
# loop, in the whole RK4 AntMaze iteration, at 128 and then 32 policies x 16 environments; the unit's
# program and the idea built in one process and timed in alternating rounds.
[ "$(hostname -s)" = serval05 ] || { echo "this unit runs on serval05 only"; exit 1; }
H=/p/rlprojects/RND/10_jax_exploration_platform/benchmark_runs/2026-11-12-00-17_pointmaze-antmaze_cleanrl-ppo-rnd_speed-30-ideas_screen-15-minutes-per-idea-per-card_h100-nvl_rtx-5080_rtx-a4500/profiles/rk4_ant_iteration
for copies in 128 32; do
    bash $H/units/run_on_card.sh ideas_solver-plain-loop_h100-nvl_serval05_copies-$copies ideas.py --copies $copies \
        --variants solver-plain-loop --rounds 10 --budget-seconds 330 \
        --out $H/records/ideas_solver-plain-loop_h100-nvl_serval05_copies-$copies.json || exit $?
done
