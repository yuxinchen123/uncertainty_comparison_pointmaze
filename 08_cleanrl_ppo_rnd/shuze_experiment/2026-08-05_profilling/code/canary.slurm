#!/bin/bash
#SBATCH --job-name=rndcanary
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gpus-per-node=1
#SBATCH --mem=8G
#SBATCH --time=00:20:00
set -euo pipefail
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=1
BASE=/p/rlprojects/RND/08_cleanrl_ppo_rnd/shuze_experiment/2026-08-05_profilling
/p/rlprojects/RND/.venvs/cleanrl_rnd/bin/python "$BASE/code/canary_gpu.py" "$BASE/data/canary"
