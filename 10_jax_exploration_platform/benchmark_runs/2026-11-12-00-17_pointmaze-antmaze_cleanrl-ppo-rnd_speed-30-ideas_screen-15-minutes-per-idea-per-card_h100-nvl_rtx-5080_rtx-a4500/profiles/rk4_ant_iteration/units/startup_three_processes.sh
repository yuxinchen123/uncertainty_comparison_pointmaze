#!/bin/bash
# EXPECTED_MINUTES=10
# Start-up of a resumed RK4 AntMaze unit at 128 policies x 16 environments, four processes in a row on one
# card: without JAX's compilation cache (state as built, then committed to the card), then with a new cache folder twice (the first fills it, the second
# reads it). Runs as a serval05 profile unit or inside a Slurm job on a card node.
#   startup_three_processes.sh <card label>
set -uo pipefail
H=/p/rlprojects/RND/10_jax_exploration_platform/benchmark_runs/2026-11-12-00-17_pointmaze-antmaze_cleanrl-ppo-rnd_speed-30-ideas_screen-15-minutes-per-idea-per-card_h100-nvl_rtx-5080_rtx-a4500/profiles/rk4_ant_iteration
LABEL=$1
BASE=/localtmp/$USER; [ -d /localtmp ] && mkdir -p $BASE 2>/dev/null || BASE=/tmp/$USER
CACHE=$BASE/jax_compilation_cache_rk4-startup-test_$(date +%s)
OUT=$H/records/startup_${LABEL}.jsonl
rm -rf "$CACHE"
for run in "no-cache||" "no-cache_state-committed||--commit-state" "cache-first-process-fills-it_state-committed|$CACHE|--commit-state" "cache-second-process-reads-it_state-committed|$CACHE|--commit-state"; do
    name=${run%%|*}; rest=${run#*|}; dir=${rest%%|*}; flag=${rest#*|}
    bash $H/units/run_on_card.sh startup_${LABEL} startup.py --copies 128 --label "$name" ${dir:+--cache-dir $dir} $flag --out $OUT \
        || { echo "run $name failed"; exit 1; }
done
du -sh "$CACHE" >> $H/logs/startup_${LABEL}.log 2>&1
rm -rf "$CACHE"
