#!/bin/bash
# One timing unit of the blow-up diagnosis on serval02 card 1 (serval05's card was shared by another account
# on 2026-11-12 from 18:23 PT, which spoils timings there): for each cell, the checkpoint's policies rolled
# out for 45 s and then 6 whole training iterations timed, each cell its own process, at most 14.5 minutes
# of wall time for the unit, startup and compilation included.
#   run_timing_unit.sh <unit name> <cell slug>:<options of roll_out_checkpoint_policies.py> [...]
# The checkpoint and the unit's configuration file are read from /localtmp/sl5nw/antmaze_blow_up (copied
# there before the unit, because serval02 reads network storage slowly).
set -uo pipefail
START=$(date +%s)
UNIT=${1:?unit name}; shift
P=/p/rlprojects/RND/10_jax_exploration_platform
OUT=$P/benchmark_runs/2026-11-12-00-17_pointmaze-antmaze_cleanrl-ppo-rnd_speed-30-ideas_screen-15-minutes-per-idea-per-card_h100-nvl_rtx-5080_rtx-a4500/profiles/antmaze_warp_blow_up
PY=/p/rlprojects/RND/.venvs/platform_jax_warp/bin/python
LOCAL=/localtmp/sl5nw/antmaze_blow_up
exec >> "$OUT/logs/$UNIT.log" 2>&1
pt() { TZ=America/Los_Angeles date '+%Y-%m-%d %H:%M:%S PT'; }
CARD1=GPU-fed69954-5a71-dbb2-ee91-544e0eb6b8a3
[ "$(hostname -s)" = serval02 ] && [ "${CUDA_VISIBLE_DEVICES:-}" = $CARD1 ] || {
    echo "$(pt) refused: this unit runs on serval02 card 1 only, started through serval02_card1_run.sh"; exit 1; }
export PYTHONNOUSERSITE=1 XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
echo "$(pt) unit \"$UNIT\" on serval02 card 1, cells: $*"
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
worst=0
for cell in "$@"; do
    slug=${cell%%:*}; options=${cell#*:}
    left=$((START + 870 - $(date +%s)))
    [ $left -gt 200 ] || { echo "$(pt) not started, $left s are left: $slug"; worst=3; continue; }
    echo "$(pt) cell $slug: $options"
    timeout -s TERM $left $PY -u $OUT/units/roll_out_checkpoint_policies.py \
        --checkpoint $LOCAL/rollout-shape-check_antmaze-umaze_reward-arrival-plus-one_cleanrl-ppo-rnd-with-128-environments-per-policy_cells-33-to-40-of-48_seeds-0-to-15.npz \
        --configuration-file $LOCAL/unit_configuration_cells-33-to-40.toml \
        --rollout-seconds 45 --time-iterations 6 --max-events 0 --max-moderate 0 \
        --out $OUT/records/${UNIT}_$slug $options
    rc=$?; echo "$(pt) cell $slug ended with exit code $rc"; [ $rc -gt $worst ] && worst=$rc
done
kill $WATCHER 2>/dev/null
nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader
[ -s "$OTHERS" ] && { echo "$(pt) another process was on card 1 during the unit: the measurement is discarded"; cat "$OTHERS"; worst=90; }
echo "$(pt) unit \"$UNIT\" ended with exit code $worst after $(( $(date +%s) - START )) s"
exit $worst
