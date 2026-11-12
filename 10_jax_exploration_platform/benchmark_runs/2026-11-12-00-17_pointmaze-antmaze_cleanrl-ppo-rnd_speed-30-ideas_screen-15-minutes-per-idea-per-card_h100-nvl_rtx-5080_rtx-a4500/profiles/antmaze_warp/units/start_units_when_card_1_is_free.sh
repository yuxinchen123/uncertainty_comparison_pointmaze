#!/bin/bash
# Run ON serval02: start the given units on card 1, one after the other, each as soon as the card and our
# lock are free, trying every 30 seconds for at most 2 hours per unit. Nothing is stopped:
# serval02_card1_run.sh exits with code 75 while the card is busy. Between two units the card is left
# free for 45 seconds, so that another session's waiting unit can take its turn.
#   start_units_when_card_1_is_free.sh <unit script> [<unit script> ...]
S=/p/rlprojects/.claude/skills/serval05-queue/scripts
LOGS="$(dirname "$1")/../logs"
pt() { TZ=America/Los_Angeles date '+%Y-%m-%d %H:%M:%S PT'; }
echo $$ > "$LOGS/waiting_for_card_1.pid"
for UNIT in "$@"; do
    ran=no
    for attempt in $(seq 1 240); do
        bash $S/serval02_card1_run.sh --kind profile --session 9ca0cb45-e2aa-43c6-9353-3a3d17cddb5f -- bash "$UNIT"
        rc=$?
        if [ $rc -ne 75 ]; then
            echo "$(pt) $(basename "$UNIT") attempt $attempt: the unit ran and ended with exit code $rc"
            ran=yes; break
        fi
        [ $((attempt % 10)) -eq 1 ] && echo "$(pt) $(basename "$UNIT") attempt $attempt: card 1 busy, trying every 30 s"
        sleep 30
    done
    [ $ran = yes ] || { echo "$(pt) card 1 stayed busy for 2 hours; $(basename "$UNIT") was not run"; break; }
    sleep 45
done
echo "$(pt) no more units"
rm -f "$LOGS/waiting_for_card_1.pid"
