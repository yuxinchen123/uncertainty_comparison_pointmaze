#!/bin/bash

# Example of running python script with a job array

#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=0
#SBATCH --mem=20G
#SBATCH --time=4-00:00:00
#SBATCH --output slurm_output.log            
#SBATCH --ntasks=4

# Run python script with a command line argument
srun wandb agent $1