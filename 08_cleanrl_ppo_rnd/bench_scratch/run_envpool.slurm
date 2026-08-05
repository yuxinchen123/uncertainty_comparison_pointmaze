#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --mem=32G
B=/p/rlprojects/RND/08_cleanrl_ppo_rnd/bench_scratch
export PYTHONPATH=$B/envpool_pkgs
export OMP_NUM_THREADS=1
$B/../../.venvs/exploration/bin/python -u $B/bench_envpool.py $B/results/envpool_cpu$1.json
