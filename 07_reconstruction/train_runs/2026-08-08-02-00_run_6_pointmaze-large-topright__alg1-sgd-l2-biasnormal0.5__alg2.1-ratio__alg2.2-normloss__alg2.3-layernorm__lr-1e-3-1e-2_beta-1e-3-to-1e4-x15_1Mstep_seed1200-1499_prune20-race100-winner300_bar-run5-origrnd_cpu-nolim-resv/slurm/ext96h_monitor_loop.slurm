#!/bin/bash
# Standing 20-minute monitor for the ext96h sweep: one light Slurm job that ticks until the
# SWEEP96H_COMPLETE sentinel exists. Submit:
#   sbatch --export=ALL,SWEEP_ID=<id> slurm/ext96h_monitor_loop.slurm
#SBATCH --job-name=r6e96mon
#SBATCH --partition=nolim
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=20-00:00:00
#SBATCH --output=/p/rlprojects/RND/07_reconstruction/train_runs/2026-08-08-02-00_run_6_pointmaze-large-topright__alg1-sgd-l2-biasnormal0.5__alg2.1-ratio__alg2.2-normloss__alg2.3-layernorm__lr-1e-3-1e-2_beta-1e-3-to-1e4-x15_1Mstep_seed1200-1499_prune20-race100-winner300_bar-run5-origrnd_cpu-nolim-resv/logs/ext96h_monitor_loop_%j.log
RUN_DIR="/p/rlprojects/RND/07_reconstruction/train_runs/2026-08-08-02-00_run_6_pointmaze-large-topright__alg1-sgd-l2-biasnormal0.5__alg2.1-ratio__alg2.2-normloss__alg2.3-layernorm__lr-1e-3-1e-2_beta-1e-3-to-1e4-x15_1Mstep_seed1200-1499_prune20-race100-winner300_bar-run5-origrnd_cpu-nolim-resv"
SWEEP_ID="${SWEEP_ID:?SWEEP_ID must be passed via sbatch --export=ALL,SWEEP_ID=<id>}"
while [ ! -f "$RUN_DIR/SWEEP96H_COMPLETE" ]; do
  bash "$RUN_DIR/slurm/ext96h_monitor.sh" "$SWEEP_ID"
  sleep 1200
done
echo "sentinel present; monitor loop ends $(date '+%Y-%m-%dT%H:%M:%S')"
