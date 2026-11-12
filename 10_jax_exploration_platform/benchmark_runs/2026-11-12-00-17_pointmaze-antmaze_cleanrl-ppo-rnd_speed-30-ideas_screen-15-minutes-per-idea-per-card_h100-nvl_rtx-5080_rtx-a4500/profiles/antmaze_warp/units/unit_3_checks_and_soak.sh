#!/bin/bash
# EXPECTED_MINUTES=15
# Warp AntMaze, unit 3: the correctness measurements on the card (one step against C MuJoCo and against the
# JAX implementation, the resting height, copy isolation with a nearly full pool at 4,096 and 32,768
# environments, a pool that is too small, the semantics against the JAX environment, non-finite inputs, the
# resting height in the staged graph mode and with fewer solver and line-search iterations), then the
# random-torque soak at 4,096 environments: 2 million steps and on for 200 seconds.
exec bash /p/rlprojects/RND/10_jax_exploration_platform/benchmark_runs/2026-11-12-00-17_pointmaze-antmaze_cleanrl-ppo-rnd_speed-30-ideas_screen-15-minutes-per-idea-per-card_h100-nvl_rtx-5080_rtx-a4500/profiles/antmaze_warp/units/run_unit.sh "unit 3 correctness on the card and the soak" \
    --parts checks soak --soak-seconds 200
