#!/bin/bash
# EXPECTED_MINUTES=5
# serval05 profile unit: the RK4 physics step split into MuJoCo Warp's stages natively at 2,048 and 512 worlds,
# started from the unit's spawn (the first native split started the ant inside a wall), and its same-number variants.
[ "$(hostname -s)" = serval05 ] || { echo "this unit runs on serval05 only"; exit 1; }
H=/p/rlprojects/RND/10_jax_exploration_platform/benchmark_runs/2026-11-12-00-17_pointmaze-antmaze_cleanrl-ppo-rnd_speed-30-ideas_screen-15-minutes-per-idea-per-card_h100-nvl_rtx-5080_rtx-a4500/profiles/rk4_ant_iteration
exec bash $H/units/run_on_card.sh native-split-from-spawn_h100-nvl_serval05 breakdown.py --copies 128 \
    --parts native,native_variants --native-worlds 2048,512 --budget-seconds 600 \
    --out $H/records/native-split-from-spawn_h100-nvl_serval05.json
