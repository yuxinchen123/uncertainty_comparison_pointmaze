#!/bin/bash
# EXPECTED_MINUTES=15
# Warp AntMaze, unit 1: does it run on the card. Warp at 1,024 and 4,096 environments with an empty kernel
# folder (so the first call shows the kernel compilation), the trainer's kind of program (a donated state
# carried through lax.scan with resets inside) in two graph modes, each with the scan unrolled 1 and 16 steps,
# a second process on the same card, then
# the JAX implementation at the same two batch sizes in its own process.
exec bash /p/rlprojects/RND/10_jax_exploration_platform/benchmark_runs/2026-11-12-00-17_pointmaze-antmaze_cleanrl-ppo-rnd_speed-30-ideas_screen-15-minutes-per-idea-per-card_h100-nvl_rtx-5080_rtx-a4500/profiles/antmaze_warp/units/run_unit.sh "unit 1 first run, speed at 1,024 and 4,096, trainer scan, second process" \
    --parts speed-warp trainer second-process --batches 1024 4096 --trainer-shapes 1024x4 --graph-modes WARP WARP_STAGED --trainer-unrolls 1 16 \
    -- --parts speed-jax --batches 1024 4096
