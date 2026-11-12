#!/bin/bash
# EXPECTED_MINUTES=15
# Warp AntMaze, unit 4: the platform's own training iteration (Runner.iterate with the study's program) with
# the Warp physics and with the JAX physics, each in its own process, at 512 policies x 4 environments and at
# 256 policies x 16 environments, 128 steps per rollout. Then, in the time that is left, Warp with fewer
# solver iterations: one step against C MuJoCo of the specified model, and a 40-second soak, per setting.
exec bash /p/rlprojects/RND/10_jax_exploration_platform/benchmark_runs/2026-11-12-00-17_pointmaze-antmaze_cleanrl-ppo-rnd_speed-30-ideas_screen-15-minutes-per-idea-per-card_h100-nvl_rtx-5080_rtx-a4500/profiles/antmaze_warp/units/run_unit.sh "unit 4 the platform training iteration, Warp and JAX, and fewer solver iterations" \
    --parts platform-trainer-warp --trainer-shapes 512x4 256x16 \
    -- --parts platform-trainer-jax --trainer-shapes 512x4 256x16 \
    -- --parts fewer-iterations --soak-seconds 40
