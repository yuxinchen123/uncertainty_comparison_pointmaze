#!/bin/bash
# Download the best-beta-group eval-reward curves (line-plot data) on a compute
# node. Pinned to jaguar03, which has outbound internet (W&B API) and the shared
# home where ~/.netrc (W&B auth) and the conda env live. API-only job: 0 GPUs.
#SBATCH --job-name=hist_dl_y24vyh06
#SBATCH --partition=gpu
#SBATCH --nodelist=jaguar03
#SBATCH --reservation=sl5nw_151
#SBATCH --gpus-per-node=0
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --chdir=/p/rlprojects/RND/07_reconstruction/development_document/train_run_aggregate/2026-06-24-17-31_run_1_before_reorganization
#SBATCH --output=data/history_download.slurm.log

# Make the claude/local binaries and conda reachable, then run via the project env.
export PATH=/u/sl5nw/.local/bin:$PATH
CONDA=/sw/ubuntu2204/ebu082024/software/common/core/miniforge/24.7.1-py3.11/bin/conda

srun "$CONDA" run -n exploration python code/02_download_history.py --workers 16
