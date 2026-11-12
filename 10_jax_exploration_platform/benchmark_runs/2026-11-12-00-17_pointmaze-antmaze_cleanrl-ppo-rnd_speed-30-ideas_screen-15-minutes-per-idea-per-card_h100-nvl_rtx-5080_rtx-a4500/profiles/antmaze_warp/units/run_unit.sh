#!/bin/bash
# One unit of the Warp AntMaze measurement: at most 15 minutes of wall time on one H100 NVL.
#   run_unit.sh "<unit name>" <arguments of profile_antmaze_warp.py> [ -- <arguments of a second process> ... ]
# Each group of arguments between "--" is one Python process, run one after the other inside the unit's
# 15 minutes (the Warp list and the JAX list of batch sizes each get their own process, so that the memory a
# process holds on the card belongs to one implementation).
# Sets its own environment. Appends its output to logs/<unit name>.log in the profile folder and its
# measurements to antmaze_warp.json there; rebuilds antmaze_warp.md at the end.
# The interpreter is platform_jax_warp (platform_jax plus warp-lang 1.14.0), for the JAX rows too.
# Warp writes its compiled kernels to the node-local folder /localtmp/$USER/warp_kernel_cache, not to
# the home folder's ~/.cache/warp.
# On serval02 it must be started through serval02_card1_run.sh, which pins card 1 by its UUID.
set -uo pipefail
START=$(date +%s)
UNIT=${1:?unit name}; shift
[ $# -gt 0 ] || { echo "no arguments given" >&2; exit 1; }
P=/p/rlprojects/RND/10_jax_exploration_platform
OUT=$P/benchmark_runs/2026-11-12-00-17_pointmaze-antmaze_cleanrl-ppo-rnd_speed-30-ideas_screen-15-minutes-per-idea-per-card_h100-nvl_rtx-5080_rtx-a4500/profiles/antmaze_warp
PY=/p/rlprojects/RND/.venvs/platform_jax_warp/bin/python
SCRIPT=$P/benchmarks/env/profile_antmaze_warp.py
SLUG=$(echo "$UNIT" | tr -c 'A-Za-z0-9\n' '-')
exec >> "$OUT/logs/$SLUG.log" 2>&1
echo $$ > "$OUT/logs/$SLUG.pid"
pt() { TZ=America/Los_Angeles date '+%Y-%m-%d %H:%M:%S PT'; }

# GOAL: name the machine and card; on serval02 only card 1, pinned by its UUID, is ever used.
CARD1=GPU-fed69954-5a71-dbb2-ee91-544e0eb6b8a3
case "$(hostname -s)" in
    serval05) MACHINE="serval05";;
    serval02) MACHINE="serval02 card 1"
              [ "${CUDA_VISIBLE_DEVICES:-}" = "$CARD1" ] || { echo "$(pt) refused: on serval02 the unit must be started through serval02_card1_run.sh (card 1 only)"; exit 1; };;
    *) echo "$(pt) refused: $(hostname -s) is not one of the two H100 NVL machines"; exit 1;;
esac
mkdir -p /localtmp/$USER/warp_kernel_cache
export PYTHONNOUSERSITE=1 PYTHONPATH=$P/src XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1 WARP_CACHE_PATH=/localtmp/$USER/warp_kernel_cache
echo "$(pt) unit \"$UNIT\" on $MACHINE, arguments: $*"
nvidia-smi --query-gpu=index,uuid,name,memory.used,utilization.gpu --format=csv,noheader
echo "processes on the cards before the unit:"; nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader
echo "warp kernel folder $WARP_CACHE_PATH: $(du -sh $WARP_CACHE_PATH 2>/dev/null | cut -f1)"

# GOAL: one measurement process, stopped 14.5 minutes after the unit started whatever it is doing.
measure() {
    local left=$((START + 870 - $(date +%s)))
    [ $left -gt 20 ] || { echo "$(pt) not started, the unit's time is used: $*"; return 124; }
    timeout -s TERM $left $PY -u $SCRIPT --out $OUT/antmaze_warp.json \
        --machine "$MACHINE" --unit "$UNIT" --unit-start-epoch $START "$@"
}
# GOAL: split the arguments at "--" and run one process per group; the unit's exit code is the worst of them.
worst=0; group=()
run_group() {
    [ ${#group[@]} -gt 0 ] || return 0
    measure "${group[@]}"; rc=$?
    if [ $rc -eq 76 ]; then
        echo "$(pt) JAX did not load the card; trying once more with /usr/lib/nvidia on LD_LIBRARY_PATH"
        export LD_LIBRARY_PATH=/usr/lib/nvidia${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
        measure "${group[@]}"; rc=$?
    fi
    [ $rc -eq 124 ] && echo "$(pt) the measurement was stopped at the unit's time limit"
    echo "$(pt) process ended with exit code $rc: ${group[*]}"
    [ $rc -gt $worst ] && worst=$rc
    group=()
}
for arg in "$@"; do
    if [ "$arg" = "--" ]; then run_group; else group+=("$arg"); fi
done
run_group
echo "processes on the cards after the unit:"; nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader
echo "warp kernel folder $WARP_CACHE_PATH: $(du -sh $WARP_CACHE_PATH 2>/dev/null | cut -f1)"

# GOAL: the tables are rebuilt from the record even when a measurement was stopped.
[ -f $OUT/antmaze_warp.json ] && $PY $SCRIPT --out $OUT/antmaze_warp.json --render
echo "$(pt) unit \"$UNIT\" ended with exit code $worst after $(( $(date +%s) - START )) s"
rm -f "$OUT/logs/$SLUG.pid"
exit $worst
