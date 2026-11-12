#!/bin/bash
# Run ON serval02: start one unit on card 1 as soon as the card and our lock are free, trying every 8 seconds
# for at most 90 minutes. Nothing is stopped: serval02_card1_run.sh exits with code 75 while the card is busy.
# The shorter interval is for hours in which other sessions' waiters take the card within seconds of its
# becoming free; the try itself is one nvidia-smi query and one lock test.
#   start_unit_when_card_1_is_free_try-every-8-seconds.sh <unit script>
UNIT=${1:?unit script}
S=/p/rlprojects/.claude/skills/serval05-queue/scripts
LOGS="$(dirname "$UNIT")/../logs"
pt() { TZ=America/Los_Angeles date '+%Y-%m-%d %H:%M:%S PT'; }
echo $$ > "$LOGS/waiting_for_card_1.pid"
for attempt in $(seq 1 675); do
    bash $S/serval02_card1_run.sh --kind profile --session 9ca0cb45-e2aa-43c6-9353-3a3d17cddb5f -- bash "$UNIT" > "$LOGS/last_try.txt" 2>&1
    rc=$?
    if [ $rc -ne 75 ]; then
        cat "$LOGS/last_try.txt"
        echo "$(pt) $(basename "$UNIT") attempt $attempt: the unit ran and ended with exit code $rc"
        rm -f "$LOGS/waiting_for_card_1.pid" "$LOGS/last_try.txt"
        exit $rc
    fi
    [ $((attempt % 40)) -eq 1 ] && echo "$(pt) $(basename "$UNIT") attempt $attempt: $(cat "$LOGS/last_try.txt"); trying every 8 s"
    sleep 8
done
echo "$(pt) card 1 stayed busy for 90 minutes; $(basename "$UNIT") was not run"
rm -f "$LOGS/waiting_for_card_1.pid" "$LOGS/last_try.txt"
exit 75
