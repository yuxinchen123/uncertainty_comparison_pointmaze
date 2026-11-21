#!/usr/bin/env bash
# The Gilbreth loop of pixel Montezuma stage 2: every 5 minutes, one pass of
#   1. the shared connection to Gilbreth checked; when it is closed the pass makes no tick and the loop never logs
#      in (the session's 20-minute check opens it, with its logins counted for all sessions);
#   2. remote_fleet.py hand-over-tick: finished runs copied back and verified, a run whose 4-hour idle-card job
#      ended gets the next job, live record files mirrored. The table allows 3 runs on Gilbreth at once.
# Why: on 2026-11-21 at 02:13 PT three stage-2 runs (the seed-12 runs) were handed to Gilbreth's A30 cards because
# no listed UVA card was free; the session's 20-minute check also runs the tick, and this loop keeps the gap between
# two 4-hour jobs of a run near 5 minutes.
# It ends when Gilbreth holds no unit of this run folder, when RUN/STOP or RUN/SWEEP_COMPLETE exists, or after 3 days.
# Every pass is logged with a Pacific time to montezuma_stage2_gilbreth_loop.log beside this script. Run it on
# portal04 (the machine holding the shared connections):
#   setsid nohup bash <this script> > /dev/null 2>&1 < /dev/null &
# Standard-library Python and ssh only; nothing here imports a numerical library.
set -u

HERE=$(cd "$(dirname "$0")" && pwd)
LOG="$HERE/montezuma_stage2_gilbreth_loop.log"
PY="env PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 python3"
M=/p/rlprojects/RND/10_jax_exploration_platform/scripts/pixel_sweep_campaign
RUN=$(cd "$HERE/../.." && pwd)
SERVER=gilbreth
PASS_SECONDS=300
LIMIT_SECONDS=$((3 * 24 * 3600))

pacific() { TZ=America/Los_Angeles date '+%Y-%m-%d %H:%M:%S PT'; }
note() { echo "$(pacific) | $*" >> "$LOG"; }
pause() { read -t "$1" <> <(:) || true; }

# one copy of the loop at a time, on portal04 only
exec 9> "$HERE/montezuma_stage2_gilbreth_loop.lock"
if ! flock -n 9; then note "another copy of the loop holds the lock; this one ends"; exit 1; fi
[ "$(hostname -s)" = portal04 ] || { note "this loop runs on portal04 only (it is $(hostname -s)); it ends"; exit 1; }

started=$(date +%s)
pass=0
note "loop starts (process $$), a pass every $PASS_SECONDS s, at most $((LIMIT_SECONDS / 86400)) days; run folder $RUN"
while :; do
    pass=$((pass + 1))
    # the end conditions
    if [ -e "$RUN/STOP" ] || [ -e "$RUN/SWEEP_COMPLETE" ]; then note "RUN/STOP or RUN/SWEEP_COMPLETE exists; the loop ends"; break; fi
    if [ $(( $(date +%s) - started )) -ge "$LIMIT_SECONDS" ]; then note "3 days have passed; the loop ends"; break; fi
    if ! ls "$RUN/queue/remote_$SERVER"/*.json > /dev/null 2>&1; then note "Gilbreth holds no unit of this run; the loop ends"; break; fi

    # 1. only through the open shared connection: this loop never logs in (rule purdue-cluster-slurm.md, "never
    #    retries the login in a loop"); a closed connection is left to the session's check, which counts its logins
    if ssh -O check "$SERVER" > /dev/null 2>&1; then
        # 2. the tick
        (cd /p/rlprojects/RND && timeout 1800 $PY "$M/remote_fleet.py" --run-dir "$RUN" hand-over-tick < /dev/null >> "$LOG" 2>&1) \
            || note "hand-over-tick ended with exit $?"
    else
        note "the shared connection to $SERVER is closed; this pass makes no tick (the session's check opens it)"
    fi
    pause "$PASS_SECONDS"
done
note "loop ends after $pass pass(es)"
