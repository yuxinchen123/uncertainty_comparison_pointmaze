#!/bin/bash
#SBATCH --job-name=rndhigh
#SBATCH --ntasks=1
set -euo pipefail
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=1
BASE=/p/rlprojects/RND/08_cleanrl_ppo_rnd/shuze_experiment/2026-08-05_profilling
/p/rlprojects/RND/.venvs/cleanrl_rnd/bin/python "$BASE/code/profile_variants.py" \
    --out "$BASE/data/highend" --iterations 12 --warmup 4 \
    --only_variant 10_tier_a_plus_bugfix --skip_startup
