#!/bin/bash

# Example of running python script with a job array

#SBATCH --job-name=03
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=3
#SBATCH --gpus-per-node=0
#SBATCH --reservation=sl5nw_99
#SBATCH --qos=csresnolim
#SBATCH --mem=20G
#SBATCH --time=4-00:00:00
#SBATCH --output slurm_output.log            
#SBATCH --ntasks=8

# Run python script with a command line argument
srun wandb agent $1