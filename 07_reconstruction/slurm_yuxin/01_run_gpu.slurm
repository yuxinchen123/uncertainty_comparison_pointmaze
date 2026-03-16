#!/bin/bash

# Example of running python script with a job array

#SBATCH --partition=gpu
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --mem=100G
#SBATCH --exclude=serval03,serval[07-09]
#SBATCH --time=3-00:00:00
#SBATCH --output slurm_output.log            
#SBATCH --ntasks=4

# Run python script with a command line argument
srun wandb agent $1