#!/bin/bash
#SBATCH -p cpu
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 01:00:00
#SBATCH -J rnd-unchanged-paths
# Bit-identity of the count path (against 0d14eb3) and the online coordinate path (against 5abf6b2) on processors.
set -euo pipefail
P=/p/rlprojects/RND/10_jax_exploration_platform
E=$1
cd $P/scripts/grid_initialization_campaign
/p/rlprojects/RND/.venvs/platform_jax/bin/python check_count_path_unchanged.py --run-dir $E
/p/rlprojects/RND/.venvs/platform_jax/bin/python check_online_path_unchanged.py --run-dir $E
