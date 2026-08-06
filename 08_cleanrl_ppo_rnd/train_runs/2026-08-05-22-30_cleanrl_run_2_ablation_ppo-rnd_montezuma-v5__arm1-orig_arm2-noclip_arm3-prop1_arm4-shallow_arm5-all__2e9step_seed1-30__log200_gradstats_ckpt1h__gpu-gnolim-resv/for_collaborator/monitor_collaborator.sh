#!/bin/bash
# Passive monitoring loop for a collaborator. Every ten minutes it refreshes YOUR id file, prints
# your jobs, and refills only if the queue still has unclaimed runs. It never repairs, requeues,
# prunes or cancels anything — those are owner actions.
#
#   NODES="cheetah08:4 ai07:3" bash monitor_collaborator.sh      # refill those nodes each tick
#   bash monitor_collaborator.sh                                 # watch only, never submit
#
# Run it under sbatch or nohup so it survives your shell.
set -uo pipefail
source /p/rlprojects/RND/08_cleanrl_ppo_rnd/train_runs/2026-08-05-22-30_cleanrl_run_2_ablation_ppo-rnd_montezuma-v5__arm1-orig_arm2-noclip_arm3-prop1_arm4-shallow_arm5-all__2e9step_seed1-30__log200_gradstats_ckpt1h__gpu-gnolim-resv/for_collaborator/packet_env.sh
[ "$USER" = "sl5nw" ] && { echo "This is the collaborator monitor. Owner: use the owner's tick."; exit 1; }
PACKET="$RUN_DIR/for_collaborator"
while true; do
  echo "=== $(date -Is) ==="
  if [ -f "$RUN_DIR/SWEEP_COMPLETE" ]; then echo "SWEEP_COMPLETE - the run is finished. Stopping."; exit 0; fi
  bash "$PACKET/refresh_my_ids.sh"
  echo "  queue pending: $(ls "$RUN_DIR/queue/pending" 2>/dev/null | wc -l)"
  # Refill only when asked, and only through the launcher, which owns every guard.
  [ -n "${NODES:-}" ] && NODES="$NODES" bash "$PACKET/launch_workers_collaborator.sh"
  sleep 600
done
