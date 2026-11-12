#!/bin/bash
# EXPECTED_MINUTES=8
# On the H100 NVL (serval02 card 1), with the ceiling in the model: 16,384 ants with random torques for 1,400
# steps (two episodes each), the episode ending at the goal, first with the Warp physics and then with the JAX
# physics, each in its own process: the measurement of unit 8 again
# (units/diagnose_arrivals_with_random_torques.py). Records go to records_arrivals_after_the_ceiling/.
# At most 14.5 minutes of wall time.
set -uo pipefail
START=$(date +%s)
OUT=/p/rlprojects/RND/10_jax_exploration_platform/benchmark_runs/2026-11-12-00-17_pointmaze-antmaze_cleanrl-ppo-rnd_speed-30-ideas_screen-15-minutes-per-idea-per-card_h100-nvl_rtx-5080_rtx-a4500/profiles/antmaze_warp_training
PY=/p/rlprojects/RND/.venvs/platform_jax_warp/bin/python
exec >> "$OUT/logs/h100-nvl-unit-9-arrivals-with-random-torques-after-the-ceiling.log" 2>&1
pt() { TZ=America/Los_Angeles date '+%Y-%m-%d %H:%M:%S PT'; }
CARD1=GPU-fed69954-5a71-dbb2-ee91-544e0eb6b8a3
[ "$(hostname -s)" = serval02 ] && [ "${CUDA_VISIBLE_DEVICES:-}" = $CARD1 ] || {
    echo "$(pt) refused: this unit runs on serval02 card 1 only, started through serval02_card1_run.sh"; exit 1; }
export PYTHONNOUSERSITE=1 XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
mkdir -p "$OUT/records_arrivals_after_the_ceiling"
echo "$(pt) unit 9 on serval02 card 1"
nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader
# GOAL: every 15 seconds, note any process on card 1 that this unit did not start; a unit that saw one is
# discarded and repeated.
OTHERS="$OUT/logs/h100-nvl-unit-9.other_processes"; rm -f "$OTHERS"
( while sleep 15; do
      for pid in $(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader | grep "$CARD1" | cut -d, -f2); do
          [ "$(ps -o sid= -p $pid | tr -d ' ')" = "$(ps -o sid= -p $$ | tr -d ' ')" ] || echo "$(pt) pid $pid user $(ps -o user= -p $pid)" >> "$OTHERS"
      done
  done ) &
WATCHER=$!
worst=0
for physics in warp jax; do
    left=$((START + 870 - $(date +%s)))
    [ $left -gt 120 ] || { echo "$(pt) not started, $left s are left: $physics"; continue; }
    echo "$(pt) physics $physics"
    timeout -s TERM $left $PY -u $OUT/units/diagnose_arrivals_with_random_torques.py --physics $physics \
        --environments 16384 --steps 1400 --out "$OUT/records_arrivals_after_the_ceiling/h100-nvl_arrivals-with-random-torques_physics-$physics.json"
    rc=$?; echo "$(pt) physics $physics ended with exit code $rc"; [ $rc -gt $worst ] && worst=$rc
done
kill $WATCHER 2>/dev/null
nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader
[ -s "$OTHERS" ] && { echo "$(pt) another process was on card 1 during the unit: the measurement is discarded"; cat "$OTHERS"; worst=90; }
echo "$(pt) unit 9 ended with exit code $worst after $(( $(date +%s) - START )) s"
exit $worst
