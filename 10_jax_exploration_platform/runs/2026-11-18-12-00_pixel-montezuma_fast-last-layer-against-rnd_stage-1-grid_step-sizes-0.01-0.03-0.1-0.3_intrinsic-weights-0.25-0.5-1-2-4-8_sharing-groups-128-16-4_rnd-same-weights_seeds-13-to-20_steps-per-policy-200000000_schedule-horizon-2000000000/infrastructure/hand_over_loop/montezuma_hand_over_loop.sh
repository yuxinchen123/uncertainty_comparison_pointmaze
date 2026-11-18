#!/usr/bin/env bash
# The hand-over loop of the pixel Montezuma stage-1 grid: every 5 minutes, one pass of
#   1. the shared connections to Gautschi and Gilbreth checked (a closed one is opened once with the key,
#      counted with outside_connections.sh; three failed logins in a row for all sessions together end the loop);
#   2. Gautschi's share: this run's Gautschi pool loses the H100 cards the account's other member (long635)
#      runs or waits for (account dsimchil holds 4 at once);
#   3. Gilbreth's A30 smoke job 12434389 (B nodes): its verdict read once it has ended, by remote_fleet.py
#      smoke-result, which makes Gilbreth ready only if it passed;
#   4. packs: never-started singles packed in fours until the ready four-copy pools can be filled;
#   5. remote_fleet.py hand-over-tick: finished units copied back and verified, units of ended jobs put back,
#      pending units handed over within each pool's limits, worker jobs submitted, live record files mirrored;
#   6. every unit checked to stand in one queue folder only (no unit with a job on UVA and one on a server).
# It ends when no unit is pending on UVA or held by a remote server, when RUN/STOP or RUN/SWEEP_COMPLETE
# exists, or after 4 days. Every pass is logged with a Pacific time to montezuma_hand_over_loop.log beside
# this script. Run it on portal04 (the machine holding the shared connections), for example:
#   setsid nohup bash <this script> > /dev/null 2>&1 < /dev/null &
# Standard-library Python and ssh only; nothing here imports a numerical library.
set -u

HERE=$(cd "$(dirname "$0")" && pwd)
LOG="$HERE/montezuma_hand_over_loop.log"
PY="env PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 python3"
M=/p/rlprojects/RND/10_jax_exploration_platform/scripts/pixel_sweep_campaign
RUN=/p/rlprojects/RND/10_jax_exploration_platform/runs/2026-11-18-12-00_pixel-montezuma_fast-last-layer-against-rnd_stage-1-grid_step-sizes-0.01-0.03-0.1-0.3_intrinsic-weights-0.25-0.5-1-2-4-8_sharing-groups-128-16-4_rnd-same-weights_seeds-13-to-20_steps-per-policy-200000000_schedule-horizon-2000000000
STEPS="$HERE/montezuma_hand_over_steps.py"
CONNECTIONS=/p/rlprojects/.claude/skills/uva-logins-and-outside-connections/scripts/outside_connections.sh
SERVERS=(gautschi gilbreth)
PASS_SECONDS=300
LIMIT_SECONDS=$((4 * 24 * 3600))

pacific() { TZ=America/Los_Angeles date '+%Y-%m-%d %H:%M:%S PT'; }
note() { echo "$(pacific) | $*" >> "$LOG"; }
pause() { read -t "$1" <> <(:) || true; }

# one copy of the loop at a time
exec 9> "$HERE/montezuma_hand_over_loop.lock"
if ! flock -n 9; then note "another copy of the loop holds $HERE/montezuma_hand_over_loop.lock; this one ends"; exit 1; fi
[ "$(hostname -s)" = portal04 ] || { note "this loop runs on portal04 only (it is $(hostname -s)); it ends"; exit 1; }

started=$(date +%s)
pass=0
note "loop starts (process $$), a pass every $PASS_SECONDS s, at most $((LIMIT_SECONDS / 86400)) days; run folder $RUN"
while :; do
    pass=$((pass + 1))
    if [ -e "$RUN/STOP" ] || [ -e "$RUN/SWEEP_COMPLETE" ]; then note "RUN/STOP or RUN/SWEEP_COMPLETE exists; the loop ends"; break; fi
    if [ $(( $(date +%s) - started )) -ge "$LIMIT_SECONDS" ]; then note "4 days have passed; the loop ends"; break; fi
    note "pass $pass starts"

    # 1. the shared connections; a login that can fail is counted for all sessions on this machine
    reachable=yes
    for server in "${SERVERS[@]}"; do
        ssh -O check "$server" > /dev/null 2>&1 && continue
        if ! bash "$CONNECTIONS" may-try "$server" >> "$LOG" 2>&1; then
            note "three failed logins to $server in a row (all sessions together): the loop ends; tell the user"
            exit 1
        fi
        if timeout 120 ssh -o BatchMode=yes -o ConnectTimeout=30 "$server" true < /dev/null >> "$LOG" 2>&1; then
            bash "$CONNECTIONS" ok "$server" >> "$LOG" 2>&1
            note "the connection to $server was closed and is open again"
        else
            bash "$CONNECTIONS" failed "$server" >> "$LOG" 2>&1
            note "login to $server FAILED (counted); this pass makes no tick"
            reachable=no
        fi
    done

    if [ "$reachable" = yes ]; then
        # 2. Gautschi's share of the account's 4 H100 cards
        $PY "$STEPS" --run-dir "$RUN" gautschi-share < /dev/null >> "$LOG" 2>&1 || note "gautschi-share did not complete (exit $?); the tick uses the last limits"
        # 3. Gilbreth's A30 smoke job on a B node: its verdict is read once it has ended (ready only if it passed)
        $PY "$STEPS" --run-dir "$RUN" smoke-wait --server gilbreth --job 12434389 < /dev/null >> "$LOG" 2>&1 \
            || note "smoke-wait did not complete (exit $?)"
        # 4. packs for the four-copy pools
        $PY "$STEPS" --run-dir "$RUN" pack-top-up < /dev/null >> "$LOG" 2>&1 || note "pack-top-up did not complete (exit $?)"
        # 5. the tick
        timeout 3600 $PY "$M/remote_fleet.py" --run-dir "$RUN" hand-over-tick < /dev/null >> "$LOG" 2>&1 \
            || note "hand-over-tick ended with exit $?"
    fi
    # 6. one queue folder per unit
    $PY "$STEPS" --run-dir "$RUN" queue-check < /dev/null >> "$LOG" 2>&1 || note "WARNING: a unit stands in two queue folders (see the line above)"

    if $PY "$STEPS" --run-dir "$RUN" finished < /dev/null >> "$LOG" 2>&1; then
        note "no unit is pending on UVA or held by a remote server; the loop ends"
        break
    fi
    note "pass $pass ends"
    pause "$PASS_SECONDS"
done
note "loop ends after $pass pass(es)"
