#!/bin/bash

# Example of running python script with a job array

#SBATCH --job-name=exploration_gt_gnolim
#SBATCH --partition=gnolim
#SBATCH --cpus-per-task=2
#SBATCH --gpus-per-node=1
#SBATCH --mem=50G
#SBATCH --time=3-00:00:00
#SBATCH --output slurm_output.log            
#SBATCH --ntasks=4

# Run python script with a command line argument
srun wandb agent catresearch/$1