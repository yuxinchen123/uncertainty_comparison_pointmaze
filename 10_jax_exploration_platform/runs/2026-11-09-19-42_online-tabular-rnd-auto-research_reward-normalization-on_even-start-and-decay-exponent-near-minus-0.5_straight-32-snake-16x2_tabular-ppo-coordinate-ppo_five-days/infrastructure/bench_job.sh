#!/bin/bash
#SBATCH -p gpu
#SBATCH -c 2
#SBATCH --mem=12G
#SBATCH --gres=gpu:1
#SBATCH -t 06:00:00
#SBATCH -J rnd-decay-bench
# Run the distillation bench for a subset of ideas on one card: $1 idea file, $2 output folder, then idea names.
set -euo pipefail
P=/p/rlprojects/RND/10_jax_exploration_platform
export PYTHONPATH=$P/src PYTHONNOUSERSITE=1 XLA_PYTHON_CLIENT_PREALLOCATE=false
hostname; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
ideas=$1; output=$2; shift 2
/p/rlprojects/RND/.venvs/platform_jax/bin/python -u $P/scripts/grid_initialization_campaign/rnd_decay_bench.py --ideas $ideas --output $output --only "$@"
