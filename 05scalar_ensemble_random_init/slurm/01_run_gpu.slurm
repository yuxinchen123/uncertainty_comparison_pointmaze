#!/bin/bash

# SLURM script for running uncertainty comparison experiments on GPU

#SBATCH --job-name=uncertainty_comp
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --mem=50G
#SBATCH --exclude=serval03,serval[06-09],cheetah01,cheetah04,lotus,jaguar06
#SBATCH --time=3-00:00:00
#SBATCH --output slurm_output.log            
#SBATCH --ntasks=4

# Run wandb agent with the provided sweep ID
srun wandb agent catresearch/$1
