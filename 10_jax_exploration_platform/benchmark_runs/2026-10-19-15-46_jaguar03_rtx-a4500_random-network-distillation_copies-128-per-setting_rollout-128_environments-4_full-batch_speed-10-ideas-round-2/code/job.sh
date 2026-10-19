#!/bin/bash
# One Slurm job on jaguar03 holding 4 RTX A4500 cards: runs every round of driver.py.
set -euo pipefail
B=/p/rlprojects/RND/10_jax_exploration_platform/benchmark_runs/2026-10-19-15-46_jaguar03_rtx-a4500_random-network-distillation_copies-128-per-setting_rollout-128_environments-4_full-batch_speed-10-ideas-round-2
mkdir -p $B/slurm/jobs/${SLURM_JOB_ID}_${SLURMD_NODENAME}
scontrol write batch_script "$SLURM_JOB_ID" "$B/slurm/jobs/${SLURM_JOB_ID}_${SLURMD_NODENAME}/job.sbatch"
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv > "$B/slurm/jobs/${SLURM_JOB_ID}_${SLURMD_NODENAME}/hardware.csv"
export LD_LIBRARY_PATH=/usr/lib/nvidia:${LD_LIBRARY_PATH:-}
exec /usr/bin/python3 -u $B/code/driver.py $B
