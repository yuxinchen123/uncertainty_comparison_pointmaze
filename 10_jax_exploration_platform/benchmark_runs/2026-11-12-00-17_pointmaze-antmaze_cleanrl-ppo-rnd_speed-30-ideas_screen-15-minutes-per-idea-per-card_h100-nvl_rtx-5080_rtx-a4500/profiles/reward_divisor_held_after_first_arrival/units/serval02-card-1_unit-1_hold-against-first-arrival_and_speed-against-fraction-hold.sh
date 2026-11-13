#!/bin/bash
# EXPECTED_MINUTES=10
# Card check of reward_divisor=held_after_first_arrival on serval02 card 1 (serval05's card held processes
# of another account on 2026-11-13 00:40 PT): the study's PointMaze Medium unit (256 policies, learning rate
# 6e-3, intrinsic weight 2, reward 1 on arrival, 1,048,576 steps per policy) trained once with the new
# option, reading every policy's hold step against its first-arrival step; then the new option and
# held_after_fraction (1/16) timed in alternating blocks of 128 iterations. At most 14.5 minutes of wall
# time, startup and compilation included. Started through serval02_card1_run.sh --kind profile.
set -uo pipefail
START=$(date +%s)
UNIT=serval02-card-1_unit-1_hold-against-first-arrival_and_speed-against-fraction-hold
P=/p/rlprojects/RND/10_jax_exploration_platform
OUT=$P/benchmark_runs/2026-11-12-00-17_pointmaze-antmaze_cleanrl-ppo-rnd_speed-30-ideas_screen-15-minutes-per-idea-per-card_h100-nvl_rtx-5080_rtx-a4500/profiles/reward_divisor_held_after_first_arrival
PY=/p/rlprojects/RND/.venvs/platform_jax/bin/python
exec >> "$OUT/logs/$UNIT.log" 2>&1
pt() { TZ=America/Los_Angeles date '+%Y-%m-%d %H:%M:%S PT'; }
CARD1=GPU-fed69954-5a71-dbb2-ee91-544e0eb6b8a3
[ "$(hostname -s)" = serval02 ] && [ "${CUDA_VISIBLE_DEVICES:-}" = $CARD1 ] || {
    echo "$(pt) refused: this unit runs on serval02 card 1 only, started through serval02_card1_run.sh"; exit 1; }
export PYTHONNOUSERSITE=1 XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
echo "$(pt) unit \"$UNIT\" on serval02 card 1"
nvidia-smi -i $CARD1 --query-gpu=name,memory.total,memory.used --format=csv,noheader
nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader
# GOAL: every 15 seconds, note any process on card 1 that this unit did not start; a unit that saw one is
# discarded and repeated.
OTHERS="$OUT/logs/$UNIT.other_processes"; rm -f "$OTHERS"
( while sleep 15; do
      for pid in $(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader | grep "$CARD1" | cut -d, -f2); do
          [ "$(ps -o sid= -p $pid | tr -d ' ')" = "$(ps -o sid= -p $$ | tr -d ' ')" ] || echo "$(pt) pid $pid user $(ps -o user= -p $pid)" >> "$OTHERS"
      done
  done ) &
WATCHER=$!
timeout -s TERM 860 $PY -u $OUT/units/check_hold_and_speed.py \
    --configuration-file $OUT/units/unit_configuration_pointmaze-medium_256-policies_learning-rate-0.006_intrinsic-weight-2.toml \
    --out $OUT/records/$UNIT.json --block-iterations 128 --wall-seconds 720
rc=$?
kill $WATCHER 2>/dev/null
nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader
[ -s "$OTHERS" ] && { echo "$(pt) another process was on card 1 during the unit: the measurement is discarded"; cat "$OTHERS"; rc=90; }
echo "$(pt) unit \"$UNIT\" ended with exit code $rc after $(( $(date +%s) - START )) s"
exit $rc
