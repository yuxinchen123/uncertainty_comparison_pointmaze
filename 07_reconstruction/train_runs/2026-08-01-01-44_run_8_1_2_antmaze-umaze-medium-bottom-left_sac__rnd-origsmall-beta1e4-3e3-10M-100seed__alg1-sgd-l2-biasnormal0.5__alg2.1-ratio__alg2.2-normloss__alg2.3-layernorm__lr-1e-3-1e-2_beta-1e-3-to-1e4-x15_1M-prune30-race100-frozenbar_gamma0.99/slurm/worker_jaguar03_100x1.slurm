#!/bin/bash
# Run 8.1.2 jaguar03 CPU worker job = 100 single-cpu tasks (pool pending_1m; the user decision:
# jaguar03 serves as a CPU server for task S in this run, NOT as a GPU node).
# jaguar03: 224 hardware threads / 112 physical cores / ~1 TB RAM — sbatch rejects task counts
# above the CORE count, so 100 tasks (12 cores of headroom) with --ntasks-per-core=2 and 2G per
# worker (100 x 2G = 200G of ~1 TB). Submit LAST, via the user's reservation when one is active:
#   --nodelist=jaguar03 --reservation=<discovered live> [--qos=csresnolim]
# Partition (gpu), --time, --job-name and --export=ALL,SWEEP_ID=<id> come on the sbatch line.
# srun --wait=0 is MANDATORY.
#SBATCH --nodes=1
#SBATCH --nodelist=jaguar03
#SBATCH --ntasks=100
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-core=2
#SBATCH --gpus-per-node=0
#SBATCH --mem=200G
#SBATCH --output=/p/rlprojects/RND/07_reconstruction/train_runs/2026-08-01-01-44_run_8_1_2_antmaze-umaze-medium-bottom-left_sac__rnd-origsmall-beta1e4-3e3-10M-100seed__alg1-sgd-l2-biasnormal0.5__alg2.1-ratio__alg2.2-normloss__alg2.3-layernorm__lr-1e-3-1e-2_beta-1e-3-to-1e4-x15_1M-prune30-race100-frozenbar_gamma0.99/logs/%x_%j.log
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export TMPDIR=/tmp
export WORKER_DEVICE=cpu
export WORKER_POOLS="pending_1m"
export PROJ_DIR="/p/rlprojects/RND/07_reconstruction"
export RUN_DIR="/p/rlprojects/RND/07_reconstruction/train_runs/2026-08-01-01-44_run_8_1_2_antmaze-umaze-medium-bottom-left_sac__rnd-origsmall-beta1e4-3e3-10M-100seed__alg1-sgd-l2-biasnormal0.5__alg2.1-ratio__alg2.2-normloss__alg2.3-layernorm__lr-1e-3-1e-2_beta-1e-3-to-1e4-x15_1M-prune30-race100-frozenbar_gamma0.99"
export SWEEP_ID="${SWEEP_ID:?SWEEP_ID must be passed via sbatch --export=ALL,SWEEP_ID=<id>}"
PY="/p/rlprojects/RND/.venvs/exploration/bin/python"
srun --wait=0 "$PY" "$RUN_DIR/slurm/worker.py"
