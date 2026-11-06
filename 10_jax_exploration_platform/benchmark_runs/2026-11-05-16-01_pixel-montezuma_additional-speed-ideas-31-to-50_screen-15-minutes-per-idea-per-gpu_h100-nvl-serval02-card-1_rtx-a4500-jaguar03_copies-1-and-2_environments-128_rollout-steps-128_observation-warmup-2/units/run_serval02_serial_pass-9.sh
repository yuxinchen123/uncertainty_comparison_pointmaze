#!/bin/bash
# Runs the listed serval02 card-1 units one after another; the card-1 lock is held for one unit
# at a time and released between units (rule uva-serval02-card1-fallback.md; study exception in
# plan section 6). Usage: run_serval02_serial.sh <unit list file>
set -u
LIST=$1
echo $$ > /p/rlprojects/RND_worktrees/merge-atari-cleanrl/10_jax_exploration_platform/benchmark_runs/2026-11-05-16-01_pixel-montezuma_additional-speed-ideas-31-to-50_screen-15-minutes-per-idea-per-gpu_h100-nvl-serval02-card-1_rtx-a4500-jaguar03_copies-1-and-2_environments-128_rollout-steps-128_observation-warmup-2/logs/serval02_runner_pass-9.pid
mapfile -t UNITS < "$LIST"
for unit in "${UNITS[@]}"; do
  [ -e /p/rlprojects/RND_worktrees/merge-atari-cleanrl/10_jax_exploration_platform/benchmark_runs/2026-11-05-16-01_pixel-montezuma_additional-speed-ideas-31-to-50_screen-15-minutes-per-idea-per-gpu_h100-nvl-serval02-card-1_rtx-a4500-jaguar03_copies-1-and-2_environments-128_rollout-steps-128_observation-warmup-2/STOP ] && { echo "STOP found"; break; }
  echo "$(TZ=America/Los_Angeles date '+%Y-%m-%d %H:%M:%S PT') start $unit" >> /p/rlprojects/RND_worktrees/merge-atari-cleanrl/10_jax_exploration_platform/benchmark_runs/2026-11-05-16-01_pixel-montezuma_additional-speed-ideas-31-to-50_screen-15-minutes-per-idea-per-gpu_h100-nvl-serval02-card-1_rtx-a4500-jaguar03_copies-1-and-2_environments-128_rollout-steps-128_observation-warmup-2/logs/serval02_runner_pass-9.log
  flock -w 3600 /tmp/sl5nw_serval02_card1.lock bash "$unit" < /dev/null
  rc=$?
  echo "$(TZ=America/Los_Angeles date '+%Y-%m-%d %H:%M:%S PT') end rc=$rc $unit" >> /p/rlprojects/RND_worktrees/merge-atari-cleanrl/10_jax_exploration_platform/benchmark_runs/2026-11-05-16-01_pixel-montezuma_additional-speed-ideas-31-to-50_screen-15-minutes-per-idea-per-gpu_h100-nvl-serval02-card-1_rtx-a4500-jaguar03_copies-1-and-2_environments-128_rollout-steps-128_observation-warmup-2/logs/serval02_runner_pass-9.log
done
echo "$(TZ=America/Los_Angeles date '+%Y-%m-%d %H:%M:%S PT') runner finished" >> /p/rlprojects/RND_worktrees/merge-atari-cleanrl/10_jax_exploration_platform/benchmark_runs/2026-11-05-16-01_pixel-montezuma_additional-speed-ideas-31-to-50_screen-15-minutes-per-idea-per-gpu_h100-nvl-serval02-card-1_rtx-a4500-jaguar03_copies-1-and-2_environments-128_rollout-steps-128_observation-warmup-2/logs/serval02_runner_pass-9.log
rm -f /p/rlprojects/RND_worktrees/merge-atari-cleanrl/10_jax_exploration_platform/benchmark_runs/2026-11-05-16-01_pixel-montezuma_additional-speed-ideas-31-to-50_screen-15-minutes-per-idea-per-gpu_h100-nvl-serval02-card-1_rtx-a4500-jaguar03_copies-1-and-2_environments-128_rollout-steps-128_observation-warmup-2/logs/serval02_runner_pass-9.pid
