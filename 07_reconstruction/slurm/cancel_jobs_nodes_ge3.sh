#!/bin/bash
# Cancel all of my jobs that have NODES >= 3.
# Usage: ./cancel_jobs_nodes_ge3.sh   (or: bash cancel_jobs_nodes_ge3.sh)

USER="${USER:-$(whoami)}"
MIN_NODES=3

echo "Listing and cancelling jobs for $USER with NODES >= $MIN_NODES ..."
count=0

while read -r job_id nodes rest; do
    # Skip header or empty
    [[ -z "$job_id" || "$job_id" == "JOBID" ]] && continue
    if [[ "$nodes" =~ ^[0-9]+$ ]] && [[ "$nodes" -ge "$MIN_NODES" ]]; then
        echo "Cancelling job $job_id (NODES=$nodes)"
        scancel "$job_id"
        ((count++)) || true
    fi
done < <(squeue -u "$USER" -o "%.18i %.6D %.50j" --noheader)

if [[ $count -eq 0 ]]; then
    echo "No jobs with NODES >= $MIN_NODES found."
else
    echo "Cancelled $count job(s)."
fi
