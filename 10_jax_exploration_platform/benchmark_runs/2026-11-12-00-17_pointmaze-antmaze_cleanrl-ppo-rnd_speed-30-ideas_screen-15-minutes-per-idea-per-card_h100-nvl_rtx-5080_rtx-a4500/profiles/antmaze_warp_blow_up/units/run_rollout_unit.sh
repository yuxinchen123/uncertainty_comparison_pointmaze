#!/bin/bash
# One unit of the blow-up diagnosis on the H100 NVL: the policies of the copied training checkpoint rolled
# out in the Warp AntMaze with the given solver settings (units/roll_out_checkpoint_policies.py), at most
# 14.5 minutes of wall time, startup and compilation included.
#   run_rollout_unit.sh <unit name> <solver iterations> <line-search iterations> <rollout seconds> <training iterations to time> [more options of the script]
# Output is appended to logs/<unit name>.log; records go to records/<unit name>.json and _events.npz.
# Where it may run: serval05 through its queue, or serval02 card 1 through serval02_card1_run.sh.
set -uo pipefail
START=$(date +%s)
UNIT=${1:?unit name}; SOLVER=${2:?solver iterations}; LINE=${3:?line-search iterations}; ROLL=${4:?rollout seconds}; TIMED=${5:?training iterations}; shift 5
P=/p/rlprojects/RND/10_jax_exploration_platform
OUT=$P/benchmark_runs/2026-11-12-00-17_pointmaze-antmaze_cleanrl-ppo-rnd_speed-30-ideas_screen-15-minutes-per-idea-per-card_h100-nvl_rtx-5080_rtx-a4500/profiles/antmaze_warp_blow_up
PY=/p/rlprojects/RND/.venvs/platform_jax_warp/bin/python
exec >> "$OUT/logs/$UNIT.log" 2>&1
pt() { TZ=America/Los_Angeles date '+%Y-%m-%d %H:%M:%S PT'; }
CARD1=GPU-fed69954-5a71-dbb2-ee91-544e0eb6b8a3
case "$(hostname -s)" in
    serval05) MACHINE="serval05"; CARD=$(nvidia-smi --query-gpu=uuid --format=csv,noheader | head -1);;
    serval02) MACHINE="serval02 card 1"; CARD=$CARD1
              [ "${CUDA_VISIBLE_DEVICES:-}" = "$CARD1" ] || { echo "$(pt) refused: on serval02 the unit must be started through serval02_card1_run.sh (card 1 only)"; exit 1; };;
    *) echo "$(pt) refused: this unit runs on serval05 or serval02 card 1 only"; exit 1;;
esac
export PYTHONNOUSERSITE=1 XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
echo "$(pt) unit \"$UNIT\" on $MACHINE: solver $SOLVER, line search $LINE, rollout $ROLL s, $TIMED training iterations timed, more: $*"
nvidia-smi --query-gpu=index,uuid,name,memory.used,utilization.gpu --format=csv,noheader
# GOAL: every 15 seconds, note any process on the card that this unit did not start; a unit that saw one
# is discarded and repeated.
OTHERS="$OUT/logs/$UNIT.other_processes"; rm -f "$OTHERS"
( while sleep 15; do
      for pid in $(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader | grep "$CARD" | cut -d, -f2); do
          [ "$(ps -o sid= -p $pid | tr -d ' ')" = "$(ps -o sid= -p $$ | tr -d ' ')" ] || echo "$(pt) pid $pid user $(ps -o user= -p $pid)" >> "$OTHERS"
      done
  done ) &
WATCHER=$!
left=$((START + 870 - $(date +%s)))
timeout -s TERM $left $PY -u $OUT/units/roll_out_checkpoint_policies.py \
    --checkpoint $OUT/checkpoint_copy/rollout-shape-check_antmaze-umaze_reward-arrival-plus-one_cleanrl-ppo-rnd-with-128-environments-per-policy_cells-33-to-40-of-48_seeds-0-to-15.npz \
    --configuration-file $OUT/checkpoint_copy/unit_configuration_cells-33-to-40.toml \
    --solver-iterations $SOLVER --ls-iterations $LINE --rollout-seconds $ROLL --time-iterations $TIMED \
    --out $OUT/records/$UNIT "$@"
rc=$?
kill $WATCHER 2>/dev/null
[ $rc -eq 124 ] && echo "$(pt) stopped at the unit's time limit"
nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader
[ -s "$OTHERS" ] && { echo "$(pt) another process was on the card during the unit: the measurement is discarded"; cat "$OTHERS"; rc=90; }
echo "$(pt) unit \"$UNIT\" ended with exit code $rc after $(( $(date +%s) - START )) s"
exit $rc
