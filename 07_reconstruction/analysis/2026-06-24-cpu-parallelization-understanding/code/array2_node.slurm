#!/bin/bash
#SBATCH --job-name=arrayexp2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=0
#SBATCH --mem=20G
#SBATCH --time=00:30:00
# nodelist / partition / output set per submission. $1 = absolute OUTDIR.
bash "$1/code/run_array_taskset.sh" "$1"
