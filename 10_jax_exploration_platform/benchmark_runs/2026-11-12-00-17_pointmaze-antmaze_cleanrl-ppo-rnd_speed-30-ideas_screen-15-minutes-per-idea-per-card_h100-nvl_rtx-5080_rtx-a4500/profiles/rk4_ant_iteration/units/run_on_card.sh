#!/bin/bash
# EXPECTED_MINUTES=15
# One profile unit of the RK4 AntMaze iteration study on one card: runs the given python script with
# its arguments under the platform's Warp interpreter, at most 15 minutes of wall time, and notes every
# process on the card that this unit did not start (a timing with one is discarded: exit code 90).
#   run_on_card.sh <log name> <script under ../scripts> [arguments ...]
# On serval05 it runs as a queue unit (enqueue.sh --kind profile); on a Slurm node inside the job.
set -uo pipefail
START=$(date +%s)
HERE=/p/rlprojects/RND/10_jax_exploration_platform/benchmark_runs/2026-11-12-00-17_pointmaze-antmaze_cleanrl-ppo-rnd_speed-30-ideas_screen-15-minutes-per-idea-per-card_h100-nvl_rtx-5080_rtx-a4500/profiles/rk4_ant_iteration
PY=/p/rlprojects/RND/.venvs/platform_jax_warp/bin/python
NAME=$1; SCRIPT=$2; shift 2
LOG=$HERE/logs/$NAME.log
exec >> "$LOG" 2>&1
pt() { TZ=America/Los_Angeles date '+%Y-%m-%d %H:%M:%S PT'; }
export PYTHONNOUSERSITE=1 XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
# GOAL: Warp's kernels on node-local storage, as the training units keep them
if [ -d /localtmp ] && mkdir -p /localtmp/$USER/warp_kernel_cache 2>/dev/null; then
    export WARP_CACHE_PATH=/localtmp/$USER/warp_kernel_cache
else
    mkdir -p /tmp/$USER/warp_kernel_cache; export WARP_CACHE_PATH=/tmp/$USER/warp_kernel_cache
fi
echo "$(pt) unit $NAME started on $(hostname -s) job ${SLURM_JOB_ID:-none}: $SCRIPT $*"
nvidia-smi --query-gpu=name,memory.used --format=csv,noheader
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader
# GOAL: every 10 seconds, note any process on this unit's card that this unit did not start
OTHERS=$HERE/logs/$NAME.other_processes; rm -f "$OTHERS"
MYSID=$(ps -o sid= -p $$ | tr -d ' ')
( while sleep 10; do
      for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader); do
          [ "$(ps -o sid= -p $pid 2>/dev/null | tr -d ' ')" = "$MYSID" ] || echo "$(pt) pid $pid user $(ps -o user= -p $pid 2>/dev/null)" >> "$OTHERS"
      done
  done ) &
WATCHER=$!
timeout -s TERM $((900 - ($(date +%s) - START))) $PY -u $HERE/scripts/$SCRIPT "$@"
rc=$?
kill $WATCHER 2>/dev/null
[ -s "$OTHERS" ] && { echo "$(pt) another process was on the card during the unit: timings discarded"; sort -u -k4 "$OTHERS" | head; rc=90; }
echo "$(pt) unit $NAME ended with exit code $rc after $(( $(date +%s) - START )) s"
exit $rc
