#!/bin/bash
# One PointMaze speed profile on the H100 NVL of serval02 (card 1): at most 15 minutes of wall time.
#   h100_unit.sh "<unit name>" <record file> <python script, relative to the platform folder> <its arguments ...>
# Must be started through serval02_card1_run.sh, which pins card 1 by its UUID and holds the card's lock for the
# unit. The script is given --output <record file>. Output goes to logs/h100-nvl_<unit name>.log of the run folder.
# If a process of anyone else is seen on card 1 while the unit runs, the record is renamed to
# <record file>.discarded-another-process-on-the-card, because the timing cannot be trusted.
set -uo pipefail
START=$(date +%s)
UNIT=${1:?unit name}; RECORD=${2:?record file}; SCRIPT=${3:?python script}; shift 3
P=/p/rlprojects/RND/10_jax_exploration_platform
RUN=$(cd "$(dirname "$0")/../../.." && pwd)
CARD1=GPU-fed69954-5a71-dbb2-ee91-544e0eb6b8a3
SLUG=$(echo "$UNIT" | tr -c 'A-Za-z0-9\n' '-')
exec >> "$RUN/logs/h100-nvl_$SLUG.log" 2>&1
pt() { TZ=America/Los_Angeles date '+%Y-%m-%d %H:%M:%S PT'; }
[ "$(hostname -s)" = serval02 ] && [ "${CUDA_VISIBLE_DEVICES:-}" = "$CARD1" ] || {
    echo "$(pt) refused: this unit runs on serval02 card 1 only, started through serval02_card1_run.sh"; exit 1; }
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 XLA_PYTHON_CLIENT_PREALLOCATE=false
echo "$(pt) unit \"$UNIT\" on serval02 card 1"
nvidia-smi --query-gpu=index,uuid,name,memory.used,utilization.gpu --format=csv,noheader
cd "$P"

# GOAL: the measurement, stopped 14.5 minutes after the unit started whatever it is doing
timeout -s TERM $((START + 870 - $(date +%s))) /p/rlprojects/RND/.venvs/platform_jax/bin/python "$SCRIPT" "$@" --output "$RECORD" &
PY=$!
echo "$PY" > "$RUN/logs/h100-nvl_$SLUG.pid"

# GOAL: every 15 seconds, note any process on card 1 that is not this unit's own
FOREIGN="$RUN/logs/h100-nvl_$SLUG.other_processes"
rm -f "$FOREIGN"
while kill -0 "$PY" 2>/dev/null; do
    for pid in $(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader | grep "$CARD1" | cut -d, -f2 | tr -d ' '); do
        # the unit's own processes are the timeout process $PY and everything started below it
        # (the measuring python, and for the two-process unit the processes that python starts)
        up=$pid
        while [ -n "$up" ] && [ "$up" != "$PY" ] && [ "$up" != 1 ] && [ "$up" != 0 ]; do
            up=$(ps -o ppid= -p "$up" 2>/dev/null | tr -d ' ')
        done
        [ "$up" = "$PY" ] || echo "$(pt) pid $pid $(ps -o user=,args= -p "$pid" 2>/dev/null | cut -c1-120)" >> "$FOREIGN"
    done
    sleep 15
done
wait "$PY"; rc=$?
[ $rc -eq 124 ] && echo "$(pt) the measurement was stopped at the unit's time limit"
if [ -s "$FOREIGN" ]; then
    echo "$(pt) another process was on card 1 during the unit; the record is discarded:"; sort -u -k4 "$FOREIGN" | head -5
    [ -f "$RECORD" ] && mv "$RECORD" "$RECORD.discarded-another-process-on-the-card"
fi
echo "$(pt) unit \"$UNIT\" ended with exit code $rc after $(( $(date +%s) - START )) s"
rm -f "$RUN/logs/h100-nvl_$SLUG.pid"
exit $rc
