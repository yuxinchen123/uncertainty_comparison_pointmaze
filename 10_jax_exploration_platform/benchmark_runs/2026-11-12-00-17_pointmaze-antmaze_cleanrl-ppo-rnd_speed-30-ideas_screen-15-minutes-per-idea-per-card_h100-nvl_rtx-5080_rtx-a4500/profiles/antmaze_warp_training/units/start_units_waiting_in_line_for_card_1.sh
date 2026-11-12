#!/bin/bash
# Run ON serval02: start the given units on card 1, one after the other. Before each try this script waits
# in line for the card's lock (a blocking flock that takes the lock and gives it back at once), as the units
# of the other sessions of this account do, so the units keep their order of arrival. A script that only
# looks every 30 seconds is passed by every unit that waits in line (seen 2026-11-12 13:12 PT).
# Nothing is stopped: serval02_card1_run.sh exits with code 75 while the card is busy, and this script
# then waits in line again. At most 2 hours per unit. Between two units the card is left free for 45
# seconds, so that another session's waiting unit can take its turn.
#   start_units_waiting_in_line_for_card_1.sh <unit script> [<unit script> ...]
S=/p/rlprojects/.claude/skills/serval05-queue/scripts
LOCK=/tmp/sl5nw_serval02_card1.lock
LOGS="$(dirname "$1")/../logs"
pt() { TZ=America/Los_Angeles date '+%Y-%m-%d %H:%M:%S PT'; }
echo $$ > "$LOGS/waiting_for_card_1.pid"
for UNIT in "$@"; do
    ran=no; began=$(date +%s); attempt=0
    while [ $(( $(date +%s) - began )) -lt 7200 ]; do
        attempt=$((attempt + 1))
        flock -w 600 $LOCK true
        bash $S/serval02_card1_run.sh --kind profile --session 9ca0cb45-e2aa-43c6-9353-3a3d17cddb5f -- bash "$UNIT"
        rc=$?
        if [ $rc -ne 75 ]; then
            echo "$(pt) $(basename "$UNIT") attempt $attempt: the unit ran and ended with exit code $rc"
            ran=yes; break
        fi
        [ $((attempt % 10)) -eq 1 ] && echo "$(pt) $(basename "$UNIT") attempt $attempt: card 1 busy, waiting in line for its lock"
        sleep 5
    done
    [ $ran = yes ] || { echo "$(pt) card 1 stayed busy for 2 hours; $(basename "$UNIT") was not run"; break; }
    sleep 45
done
echo "$(pt) no more units"
rm -f "$LOGS/waiting_for_card_1.pid"
