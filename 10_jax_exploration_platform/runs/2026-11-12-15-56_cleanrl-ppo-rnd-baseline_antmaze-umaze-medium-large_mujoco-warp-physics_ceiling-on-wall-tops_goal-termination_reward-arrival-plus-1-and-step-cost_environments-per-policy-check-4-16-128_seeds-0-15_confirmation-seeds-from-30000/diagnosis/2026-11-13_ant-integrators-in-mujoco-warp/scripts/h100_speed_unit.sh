#!/bin/bash
# EXPECTED_MINUTES=25
# Integrator check, serval05 profile unit: seconds per training iteration of the U-maze trainer on the H100 NVL for
# each physics setting, 128 policies x 16 environments first, then 512 x 16 for three settings while time is left.
# At most 25 minutes of wall time for the unit (about 5 per setting), compilation included. A process on the card
# that this unit did not start makes the unit end with exit code 90 (timings discarded).
#   enqueue.sh --kind profile <slug> h100_speed_unit.sh
set -uo pipefail
START=$(date +%s)
P=/p/rlprojects/RND/10_jax_exploration_platform
HERE=$P/runs/2026-11-12-15-56_cleanrl-ppo-rnd-baseline_antmaze-umaze-medium-large_mujoco-warp-physics_ceiling-on-wall-tops_goal-termination_reward-arrival-plus-1-and-step-cost_environments-per-policy-check-4-16-128_seeds-0-15_confirmation-seeds-from-30000/diagnosis/2026-11-13_ant-integrators-in-mujoco-warp
PY=/p/rlprojects/RND/.venvs/platform_jax_warp/bin/python
OUT=$HERE/records/speed_NVIDIA-H100-NVL_serval05.jsonl
exec >> "$HERE/logs/speed_h100-nvl_serval05.log" 2>&1
pt() { TZ=America/Los_Angeles date '+%Y-%m-%d %H:%M:%S PT'; }
[ "$(hostname -s)" = serval05 ] || { echo "$(pt) refused: this unit runs on serval05 only"; exit 1; }
export PYTHONNOUSERSITE=1 XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
mkdir -p /localtmp/$USER/warp_kernel_cache && export WARP_CACHE_PATH=/localtmp/$USER/warp_kernel_cache
echo "$(pt) unit started on serval05"
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader
# GOAL: every 15 seconds, note any process on the card that this unit did not start
OTHERS="$HERE/logs/speed_h100-nvl_serval05.other_processes"; rm -f "$OTHERS"
( while sleep 15; do
      for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader); do
          [ "$(ps -o sid= -p $pid | tr -d ' ')" = "$(ps -o sid= -p $$ | tr -d ' ')" ] || echo "$(pt) pid $pid user $(ps -o user= -p $pid)" >> "$OTHERS"
      done
  done ) &
WATCHER=$!
worst=0
for job in implicitfast_time-step-0.005_frame-skip-10:128 rk4_time-step-0.005_frame-skip-10:128 implicit-sign-corrected_time-step-0.005_frame-skip-10:128 \
           rk4_time-step-0.01_frame-skip-5:128 implicit-sign-corrected_time-step-0.01_frame-skip-5:128 \
           implicitfast_time-step-0.005_frame-skip-10:512 rk4_time-step-0.005_frame-skip-10:512 implicit-sign-corrected_time-step-0.005_frame-skip-10:512; do
    slug=${job%%:*}; copies=${job#*:}
    left=$((START + 1500 - $(date +%s)))
    [ $left -gt 240 ] || { echo "$(pt) not started, $left s left: $slug x $copies"; continue; }
    echo "$(pt) $slug, $copies policies x 16 environments"
    timeout -s TERM $left $PY -u $HERE/scripts/profile_integrators.py \
        --configuration-file $HERE/configurations/antmaze_umaze_${slug}_guard-30-and-100.toml --map umaze \
        --environments-per-policy 16 --copies $copies --timing-seconds 20 --budget-minutes 5 --out $OUT
    rc=$?; [ $rc -gt $worst ] && worst=$rc
done
kill $WATCHER 2>/dev/null
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader
[ -s "$OTHERS" ] && { echo "$(pt) another process was on the card during the unit: timings discarded"; cat "$OTHERS"; worst=90; }
echo "$(pt) unit ended with exit code $worst after $(( $(date +%s) - START )) s"
exit $worst
