#!/bin/bash

# SLURM script for running uncertainty comparison experiments on CPU

#SBATCH --job-name=uncertainty_comp_cpu
#SBATCH --partition=main
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --output slurm_output.log            
#SBATCH --ntasks=8

# Run wandb agent with the provided sweep ID
srun wandb agent catresearch/$1
