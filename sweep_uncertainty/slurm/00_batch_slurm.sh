#!/bin/bash

# Batch script for launching uncertainty comparison experiments

# Define sweep ID (replace with your actual wandb sweep ID)
job_id="uncertainty_comparison/YOUR_SWEEP_ID"

echo "Starting uncertainty comparison experiments with sweep ID: $job_id"
echo "Launching multiple SLURM jobs..."

# Launch GPU jobs
for i in $(seq 1 5); do
    echo "Launching GPU job $i..."
    sbatch slurm/01_run_gpu.slurm $job_id&
done

# Launch CPU jobs for backup/overflow
for i in $(seq 1 3); do
    echo "Launching CPU job $i..."
    sbatch slurm/02_run_cpu.slurm $job_id&
done

echo "All jobs submitted!"

# Wait and check status
sleep 60
squeue -u $USER > squeue.log

echo "Current job status:"
cat squeue.log

# Count running jobs
count=0
while IFS= read -r line
do
    IFS=' ' read -r -a array <<< "$line"
    if [[ ! ${array[-1]} =~ ^\( ]]; then
        count=$((count+1))
    fi
done < squeue.log

printf "Running jobs: %d\n" $count

# Optional: Cancel jobs that run too long
sleep 7200  # Wait 2 hours

echo "Checking for jobs that have run too long..."
squeue -u $USER > squeue_check.log

while IFS= read -r line
do
    IFS=' ' read -r -a array <<< "$line"
    if [[ ${array[-2]} =~ ^[0-9]+$ ]]; then
        if [ ${array[-2]} -gt 5 ]; then
            echo "Cancelling long-running job: $line"
            job_id=${array[0]}
            scancel $job_id
        fi
    fi
done < squeue_check.log

echo "Batch script completed!"
