#!/bin/bash
# Runs the listed serval02 card-1 units one after another; the card-1 lock is held for one unit
# at a time and released between units (rule uva-serval02-card1-fallback.md; study exception in
# plan section 6). Usage: run_serval02_serial.sh <unit list file>
set -u
LIST=$1
echo $$ > /p/rlprojects/RND_worktrees/merge-atari-cleanrl/10_jax_exploration_platform/runs/2026-11-06-01-14_pixel-montezuma_corrected-room-geometry_episode-limit-27000-steps_cleanrl-ppo-rnd-final-configuration_checks-then-steps-per-copy-2000000000_h100-nvl-serval02-card-1-copies-4-seeds-1-to-4_rtx-a4500-jaguar03-8-cards-copies-1-seeds-5-to-12/logs/serval02_runner_$(basename "$LIST" .txt).pid
mapfile -t UNITS < "$LIST"
for unit in "${UNITS[@]}"; do
  [ -e /p/rlprojects/RND_worktrees/merge-atari-cleanrl/10_jax_exploration_platform/runs/2026-11-06-01-14_pixel-montezuma_corrected-room-geometry_episode-limit-27000-steps_cleanrl-ppo-rnd-final-configuration_checks-then-steps-per-copy-2000000000_h100-nvl-serval02-card-1-copies-4-seeds-1-to-4_rtx-a4500-jaguar03-8-cards-copies-1-seeds-5-to-12/STOP ] && { echo "STOP found"; break; }
  echo "$(TZ=America/Los_Angeles date '+%Y-%m-%d %H:%M:%S PT') start $unit" >> /p/rlprojects/RND_worktrees/merge-atari-cleanrl/10_jax_exploration_platform/runs/2026-11-06-01-14_pixel-montezuma_corrected-room-geometry_episode-limit-27000-steps_cleanrl-ppo-rnd-final-configuration_checks-then-steps-per-copy-2000000000_h100-nvl-serval02-card-1-copies-4-seeds-1-to-4_rtx-a4500-jaguar03-8-cards-copies-1-seeds-5-to-12/logs/serval02_runner_$(basename "$LIST" .txt).log
  flock -w 3600 /tmp/sl5nw_serval02_card1.lock bash "$unit" < /dev/null
  rc=$?
  echo "$(TZ=America/Los_Angeles date '+%Y-%m-%d %H:%M:%S PT') end rc=$rc $unit" >> /p/rlprojects/RND_worktrees/merge-atari-cleanrl/10_jax_exploration_platform/runs/2026-11-06-01-14_pixel-montezuma_corrected-room-geometry_episode-limit-27000-steps_cleanrl-ppo-rnd-final-configuration_checks-then-steps-per-copy-2000000000_h100-nvl-serval02-card-1-copies-4-seeds-1-to-4_rtx-a4500-jaguar03-8-cards-copies-1-seeds-5-to-12/logs/serval02_runner_$(basename "$LIST" .txt).log
done
echo "$(TZ=America/Los_Angeles date '+%Y-%m-%d %H:%M:%S PT') runner finished" >> /p/rlprojects/RND_worktrees/merge-atari-cleanrl/10_jax_exploration_platform/runs/2026-11-06-01-14_pixel-montezuma_corrected-room-geometry_episode-limit-27000-steps_cleanrl-ppo-rnd-final-configuration_checks-then-steps-per-copy-2000000000_h100-nvl-serval02-card-1-copies-4-seeds-1-to-4_rtx-a4500-jaguar03-8-cards-copies-1-seeds-5-to-12/logs/serval02_runner_$(basename "$LIST" .txt).log
rm -f /p/rlprojects/RND_worktrees/merge-atari-cleanrl/10_jax_exploration_platform/runs/2026-11-06-01-14_pixel-montezuma_corrected-room-geometry_episode-limit-27000-steps_cleanrl-ppo-rnd-final-configuration_checks-then-steps-per-copy-2000000000_h100-nvl-serval02-card-1-copies-4-seeds-1-to-4_rtx-a4500-jaguar03-8-cards-copies-1-seeds-5-to-12/logs/serval02_runner_$(basename "$LIST" .txt).pid
