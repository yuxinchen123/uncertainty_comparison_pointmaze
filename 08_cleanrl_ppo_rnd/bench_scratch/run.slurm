#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --gpus-per-node=1
export OMP_NUM_THREADS=4
B=/p/rlprojects/RND/08_cleanrl_ppo_rnd/bench_scratch
nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv
$B/../../.venvs/exploration/bin/python -u $B/bench_ppo_rnd.py --out $B/results/$1.json
