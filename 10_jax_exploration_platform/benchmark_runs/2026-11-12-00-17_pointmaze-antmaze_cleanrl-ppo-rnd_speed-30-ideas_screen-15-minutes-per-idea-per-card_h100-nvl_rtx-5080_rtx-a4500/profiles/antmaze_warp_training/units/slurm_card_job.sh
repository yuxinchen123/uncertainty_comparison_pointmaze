#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -J antmaze-warp-training-profile
# One 15-minute Slurm job on one card: the training iteration with the Warp AntMaze physics at 16 environments
# per policy, with 4,096 and 16,384 environments in total (256 and 1,024 policies).
#   sbatch ... slurm_card_job.sh <card slug> "<card name>"
# The partition, the node, the card, the processors, the memory and the time limit are given to sbatch.
exec bash /p/rlprojects/RND/10_jax_exploration_platform/benchmark_runs/2026-11-12-00-17_pointmaze-antmaze_cleanrl-ppo-rnd_speed-30-ideas_screen-15-minutes-per-idea-per-card_h100-nvl_rtx-5080_rtx-a4500/profiles/antmaze_warp_training/units/run_cells.sh "$1 Slurm job $SLURM_JOB_ID, 16 environments per policy" "$1" "$2" 256x16 1024x16
