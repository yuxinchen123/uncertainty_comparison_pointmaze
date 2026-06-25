#!/bin/bash
#SBATCH --job-name=cpuprof
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --gpus-per-node=0
#SBATCH --mem=32G
#SBATCH --time=04:00:00
# nodelist / partition / job-name / output are set per submission on the sbatch CLI.
# args: $1 = absolute OUTDIR (analysis dir), $2 = device (cpu|cuda)
OUTDIR="$1"
DEVICE="${2:-cpu}"
TOTAL="${3:-4000}"
WARMUP="${4:-1200}"
echo "host=$(hostname) job=$SLURM_JOB_ID partition=$SLURM_JOB_PARTITION cpus=$SLURM_CPUS_PER_TASK device=$DEVICE total=$TOTAL"
bash "$OUTDIR/code/run_node_sweep.sh" "$OUTDIR" "$DEVICE" "$TOTAL" "$WARMUP"
