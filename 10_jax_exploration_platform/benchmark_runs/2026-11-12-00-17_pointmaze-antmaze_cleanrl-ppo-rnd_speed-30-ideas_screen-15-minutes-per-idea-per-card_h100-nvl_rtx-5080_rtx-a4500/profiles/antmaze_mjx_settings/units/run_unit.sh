#!/bin/bash
# One unit of the AntMaze MJX settings study: at most 15 minutes of wall time on one H100 NVL.
#   run_unit.sh "<unit name>" <arguments of part 1> -- <arguments of part 2> -- ...
# Each part is one call of benchmarks/env/profile_antmaze_mjx_settings.py (for example
# `--part speed --group "the two speed options" --batch 4096 --models default solver_loop=fixed_count`).
# Sets its own environment. Appends its output to logs/<unit name>.log in the profile folder and its
# measurements to antmaze_mjx_settings.json there; rebuilds antmaze_mjx_settings.md at the end.
# On serval02 it must be started through serval02_card1_run.sh, which pins card 1 by its UUID.
set -uo pipefail
START=$(date +%s)
UNIT=${1:?unit name}; shift
[ $# -gt 0 ] || { echo "no parts given" >&2; exit 1; }
P=/p/rlprojects/RND/10_jax_exploration_platform
OUT=$P/benchmark_runs/2026-11-12-00-17_pointmaze-antmaze_cleanrl-ppo-rnd_speed-30-ideas_screen-15-minutes-per-idea-per-card_h100-nvl_rtx-5080_rtx-a4500/profiles/antmaze_mjx_settings
PY=/p/rlprojects/RND/.venvs/platform_jax/bin/python
SCRIPT=$P/benchmarks/env/profile_antmaze_mjx_settings.py
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
export PYTHONNOUSERSITE=1 PYTHONPATH=$P/src XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONUNBUFFERED=1
echo "$(pt) unit \"$UNIT\" on $MACHINE"
nvidia-smi --query-gpu=index,uuid,name,memory.used,utilization.gpu --format=csv,noheader
echo "processes on the cards before the unit:"; nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader

# GOAL: one part, stopped 14.5 minutes after the unit started whatever it is doing.
measure() {
    local left=$((START + 870 - $(date +%s)))
    [ $left -gt 20 ] || { echo "$(pt) no time left for: $*"; return 124; }
    echo "$(pt) part: $*"
    timeout -s TERM $left $PY -u $SCRIPT --out $OUT/antmaze_mjx_settings.json \
        --machine "$MACHINE" --unit "$UNIT" --unit-start-epoch $START "$@"
}

# GOAL: split the arguments at each "--" and run the parts in order; the unit's exit code is the worst one.
worst=0; part=()
run_part() {
    [ ${#part[@]} -gt 0 ] || return 0
    measure "${part[@]}"; rc=$?
    if [ $rc -eq 76 ]; then
        echo "$(pt) JAX did not load the card; trying once more with /usr/lib/nvidia on LD_LIBRARY_PATH"
        export LD_LIBRARY_PATH=/usr/lib/nvidia${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
        measure "${part[@]}"; rc=$?
    fi
    [ $rc -eq 124 ] && echo "$(pt) the part was stopped at the unit's time limit"
    [ $rc -gt $worst ] && worst=$rc
    part=()
}
for argument in "$@"; do
    if [ "$argument" = "--" ]; then run_part; else part+=("$argument"); fi
done
run_part
echo "processes on the cards after the unit:"; nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader

# GOAL: the tables are rebuilt from the record even when a part was stopped.
[ -f $OUT/antmaze_mjx_settings.json ] && $PY $SCRIPT --out $OUT/antmaze_mjx_settings.json --render
echo "$(pt) unit \"$UNIT\" ended with exit code $worst after $(( $(date +%s) - START )) s"
rm -f "$OUT/logs/$SLUG.pid"
exit $worst
