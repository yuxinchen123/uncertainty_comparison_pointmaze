#!/bin/bash
#SBATCH -c 4
#SBATCH --mem=48G
#SBATCH -t 01:15:00
#SBATCH -J rk4-ant-iteration-profile
# The RK4 AntMaze iteration profile on one Slurm card: the breakdown at 128 policies (with the native physics
# split at 2,048 and 512 worlds), the breakdown at 32 policies, and the start-up test; each part at most
# 15 minutes (run_on_card.sh). Records go to ../records with the card's name in the file name.
#   sbatch -p gpu -w <node> --gres=gpu:<type>:1 -o <log> slurm_card_job.sh
set -uo pipefail
H=/p/rlprojects/RND/10_jax_exploration_platform/benchmark_runs/2026-11-12-00-17_pointmaze-antmaze_cleanrl-ppo-rnd_speed-30-ideas_screen-15-minutes-per-idea-per-card_h100-nvl_rtx-5080_rtx-a4500/profiles/rk4_ant_iteration
CARD=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 | tr ' ' '-' | tr -d '()')
TAG=${CARD}_$(hostname -s)_job-${SLURM_JOB_ID}
echo "job $SLURM_JOB_ID on $(hostname -s), card $CARD, started $(TZ=America/Los_Angeles date '+%Y-%m-%d %H:%M:%S PT')"
bash $H/units/run_on_card.sh breakdown_${TAG}_copies-128 breakdown.py --copies 128 --parts iteration,native \
    --native-worlds 2048,512 --budget-seconds 760 --out $H/records/breakdown_${TAG}_copies-128.json
bash $H/units/run_on_card.sh breakdown_${TAG}_copies-32 breakdown.py --copies 32 --parts iteration \
    --budget-seconds 800 --out $H/records/breakdown_${TAG}_copies-32.json
bash $H/units/startup_three_processes.sh ${TAG}
echo "job ended $(TZ=America/Los_Angeles date '+%Y-%m-%d %H:%M:%S PT')"
