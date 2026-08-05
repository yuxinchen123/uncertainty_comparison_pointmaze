#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gpus-per-node=1
export OMP_NUM_THREADS=8
B=/p/rlprojects/RND/08_cleanrl_ppo_rnd/bench_scratch
$B/../../.venvs/exploration/bin/python -u $B/bench_det_uint8.py $B/results/det_$1.json
