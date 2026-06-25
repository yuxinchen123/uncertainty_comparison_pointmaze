#!/bin/bash
#SBATCH --job-name=arrayexp
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=0
#SBATCH --mem=20G
#SBATCH --time=00:40:00
# nodelist / partition / output set per submission. $1 = absolute OUTDIR.
bash "$1/code/run_array_experiments.sh" "$1"
