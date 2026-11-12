#!/bin/bash
# Run ON serval02: start one unit on card 1 as soon as the card and our lock are free, trying every 30 seconds
# for at most 30 minutes. Nothing is stopped: serval02_card1_run.sh exits with code 75 while the card is busy.
#   start_when_card_1_is_free.sh <unit script>
UNIT=${1:?unit script}
S=/p/rlprojects/.claude/skills/serval05-queue/scripts
pt() { TZ=America/Los_Angeles date '+%Y-%m-%d %H:%M:%S PT'; }
echo $$ > "$(dirname "$UNIT")/../logs/waiting_for_card_1.pid"
for attempt in $(seq 1 60); do
    bash $S/serval02_card1_run.sh --kind profile --session 9ca0cb45-e2aa-43c6-9353-3a3d17cddb5f -- bash "$UNIT"
    rc=$?
    if [ $rc -ne 75 ]; then
        echo "$(pt) attempt $attempt: the unit ran and ended with exit code $rc"
        rm -f "$(dirname "$UNIT")/../logs/waiting_for_card_1.pid"
        exit $rc
    fi
    echo "$(pt) attempt $attempt: card 1 busy, waiting 30 s"
    sleep 30
done
echo "$(pt) card 1 stayed busy for 30 minutes; the unit was not run"
rm -f "$(dirname "$UNIT")/../logs/waiting_for_card_1.pid"
exit 75
