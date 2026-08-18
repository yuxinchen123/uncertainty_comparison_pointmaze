#!/bin/bash
# Smoke test: one coinflip_count run at 20k steps — verifies the new algorithm end to end in
# train.py (env stack, buffer observe/compute path, JSON record) and measures steps/min.
#SBATCH --job-name=11decay_pilot_smoke
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --gpus-per-node=0
#SBATCH --time=00:40:00
set -euo pipefail
export OMP_NUM_THREADS=1
export PYTHONNOUSERSITE=1
cd /p/rlprojects/RND/07_reconstruction
exec /p/rlprojects/RND/.venvs/exploration/bin/python train.py \
  --algorithm=coinflip_count --beta=1 --a_seed=0 \
  --env_setup=PointMaze_Large-v3_start_bottom_left \
  --z_logging_mode=local --use_wandb=False \
  --local_log_dir="$1" \
  --total_timesteps=20000 --eval_freq=5000 --n_eval_episodes=20 \
  --eval_standalone=False --log_distance=False --device=cpu \
  --rnd_obs_norm=True --rnd_distance=mse --rnd_output_dim=128 --n_predictors=1
