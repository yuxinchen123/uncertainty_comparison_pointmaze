#!/bin/bash
# Run profile or evaluation units one after another on serval02 card 1 (run ON serval02, under nohup):
#   run_units_on_serval02_card1.sh <kind> <unit name> [<unit name> ...]
# Each unit goes through serval02_card1_run.sh, which checks that the card is free, holds the card's lock for that
# unit only and writes the holder file. A unit that finds the card busy (exit 75) stops the chain.
set -u
G=/p/rlprojects/RND/10_jax_exploration_platform/runs/2026-11-17-21-26_pixel-montezuma_fast-last-layer-against-rnd_gates_unit-tests_archive-identity_resume_speed-ideas-10_card-speed-checks
RUN=/p/rlprojects/.claude/skills/serval05-queue/scripts/serval02_card1_run.sh
kind=$1; shift
stamp() { TZ=America/Los_Angeles date '+%Y-%m-%d %H:%M:%S PT'; }
for name in "$@"; do
  echo "$(stamp) start $name kind $kind" >> "$G/logs/serval02_chain.log"
  bash "$RUN" --kind "$kind" --session session_01XYBVA6MLDrN9NQnJsR5uTV -- bash "$G/serval_units/$name/unit.sh" > "$G/serval_units/$name/unit.log" 2>&1
  rc=$?
  echo "$rc" > "$G/serval_units/$name/unit.rc"
  echo "$(stamp) end $name exit $rc" >> "$G/logs/serval02_chain.log"
  [ $rc -eq 75 ] && { echo "$(stamp) card busy; chain stopped" >> "$G/logs/serval02_chain.log"; exit 75; }
done
exit 0
