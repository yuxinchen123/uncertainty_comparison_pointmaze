#!/bin/bash
# EXPECTED_MINUTES=15
# Warp AntMaze, unit 2: the speed table and the Warp settings.
# First process (Warp): the environment step at 1,024, 4,096, 8,192, 16,384, 32,768 and 65,536 environments
# (also 131,072 if the doubling to 65,536 still gains 10%); fewer solver iterations, fewer line-search
# iterations, a smaller pool, fewer rows, the staged graph mode and one compiler option, timed in turn at
# 4,096 and at 32,768 environments; the three maps timed in turn at the same two batch sizes; the trainer's
# kind of program with the scan unrolled 1 and 16 steps (the trainer's default is 16) at 1,024 x 4 and at
# 256 x 128 environments.
# Second process (JAX): the environment step at the same batch sizes up to 65,536.
# Each list of batch sizes stops when a doubling gains under 10% of total steps per second or when the
# card's memory runs out.
exec bash /p/rlprojects/RND/10_jax_exploration_platform/benchmark_runs/2026-11-12-00-17_pointmaze-antmaze_cleanrl-ppo-rnd_speed-30-ideas_screen-15-minutes-per-idea-per-card_h100-nvl_rtx-5080_rtx-a4500/profiles/antmaze_warp/units/run_unit.sh "unit 2 speed table of Warp and JAX, Warp settings, maps, trainer scan unrolled" \
    --parts speed-warp settings maps trainer --batches 1024 4096 8192 16384 32768 65536 131072 \
    --settings-batches 4096 32768 --trainer-shapes 1024x4 256x128 --graph-modes WARP --trainer-unrolls 1 16 \
    -- --parts speed-jax --batches 1024 4096 8192 16384 32768 65536
