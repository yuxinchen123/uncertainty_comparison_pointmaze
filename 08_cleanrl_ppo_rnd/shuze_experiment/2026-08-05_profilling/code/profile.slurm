#!/bin/bash
# Run the throughput-option ladder on one GPU node.
# Submitted by code/submit_profiling.sh, which pins the node and sets the cpu count.
#SBATCH --job-name=rndprof
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
set -euo pipefail
# The shared env is shadowed by /u/sl5nw/.local unless this is set; without it the job would import
# torch 2.10+cu128, which has no kernels for Pascal or Maxwell.
export PYTHONNOUSERSITE=1
# envpool owns the cores; torch's own OpenMP pool would only contend with it.
export OMP_NUM_THREADS=1
BASE=/p/rlprojects/RND/08_cleanrl_ppo_rnd/shuze_experiment/2026-08-05_profilling
/p/rlprojects/RND/.venvs/cleanrl_rnd/bin/python "$BASE/code/profile_variants.py" \
    --out "$BASE/data" --iterations "${PROF_ITERATIONS:-12}" --warmup "${PROF_WARMUP:-4}"
