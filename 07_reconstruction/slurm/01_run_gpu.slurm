#!/bin/bash

# Example of running python script with a job array

#SBATCH --job-name=exploration
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --mem=50G
#SBATCH --exclude=serval03,serval[06-09],cheetah01,cheetah03,cheetah04,lotus,jaguar06,nekomata01
#SBATCH --time=4-00:00:00
#SBATCH --output slurm_output.log            
#SBATCH --ntasks=4

# Run python script with a command line argument
srun wandb agent $1