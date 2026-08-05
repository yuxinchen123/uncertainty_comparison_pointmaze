#!/bin/bash
#SBATCH --job-name=rndpack
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
set -euo pipefail
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=1
BASE=/p/rlprojects/RND/08_cleanrl_ppo_rnd/shuze_experiment/2026-08-05_profilling
/p/rlprojects/RND/.venvs/cleanrl_rnd/bin/python "$BASE/code/pack_test.py" \
    --n "$PACK_N" --cpus_per_run "${PACK_CPUS:-8}" --iterations 12 --warmup 4 \
    --out "$BASE/data/packing"
