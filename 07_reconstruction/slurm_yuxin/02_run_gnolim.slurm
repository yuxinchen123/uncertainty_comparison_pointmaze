#!/bin/bash

# Example of running python script with a job array

#SBATCH --partition=gnolim
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --mem=50G
#SBATCH --time=10-00:00:00
#SBATCH --output slurm_output.log            
#SBATCH --ntasks=4

# Run python script with a command line argument
srun wandb agent $1