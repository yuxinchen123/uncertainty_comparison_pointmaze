#!/bin/bash
# The matrix work of one training iteration, timed on its own, so the sweep's rungs can be given
# as a percent of something measured rather than of a paper figure.
#
# This is the processor analogue of the graphics-processor roofline in analysis/ceiling: every
# matrix multiply of one iteration timed alone, summed, and compared against the iteration. It is
# measured with the same worker count as the training rung it will be compared against, because a
# processor's matrix multiply is only as fast as the shared memory system lets it be.
#
# 512 copies per worker is left out: the isolated multiplies allocate their own operands, and at
# that size across 224 workers the node would not hold them. Those rungs' rows say so.
#SBATCH --job-name=rnd09_cpu_plateau_floor
#SBATCH --partition=gpu
#SBATCH --nodelist=jaguar03
#SBATCH --exclusive
#SBATCH --gpus-per-node=0
#SBATCH --cpus-per-task=112
#SBATCH --ntasks=1
#SBATCH --mem=960G
#SBATCH --time=4:00:00
#SBATCH --output=/p/rlprojects/RND/09_parallelization/analysis/2026-08-15-17-05_jaguar03-copies-per-worker-plateau/slurm/floor_%j.out
set -euo pipefail

RUN=/p/rlprojects/RND/09_parallelization/analysis/2026-08-15-17-05_jaguar03-copies-per-worker-plateau
PY=/p/rlprojects/RND/.venvs/exploration/bin/python
export PYTHONNOUSERSITE=1

echo "host $(hostname), $(nproc) processors visible, started $(date)"

$PY -u $RUN/code/matmul_floor_cpu.py --copies 1 4 16 32 64 128 256 \
  --procs-list 112 224 --styles full_batch epoch_minibatch

echo "CPU_PLATEAU_FLOOR_DONE $(date)"
