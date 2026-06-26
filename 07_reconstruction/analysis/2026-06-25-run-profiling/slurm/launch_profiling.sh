#!/bin/bash
# Submit run_profile.slurm to 12 distinct idle allowlist nodes (one single-process job each, pinned by
# --nodelist so a later baseline-vs-improved run can land on the SAME node). gpu partition, no GPU used
# (CPU-bound workload). Records every job id for cancellation safety (only ever scancel ids in this file).
set -u
PROF=/p/rlprojects/RND/07_reconstruction/analysis/2026-06-25-run-profiling
SLURM=$PROF/slurm
LOGS=$PROF/logs
IDFILE=$SLURM/submitted_jobids.txt

# 12 homogeneous 32-CPU allowlist nodes (per .claude/rules/slurm-submission.md include-list), all idle.
NODES=(adriatic01 adriatic02 adriatic03 adriatic04 adriatic05 adriatic06 lynx01 lynx02 lynx03 lynx04 ai01 ai02)

: > "$IDFILE"
i=0
for node in "${NODES[@]}"; do
  id=$(sbatch --parsable --job-name=prof-base --nodelist="$node" --nodes=1 \
       -p gpu --gpus-per-node=0 \
       --output="$LOGS/prof_${node}_%j.out" "$SLURM/run_profile.slurm")
  echo "$id" >> "$IDFILE"
  echo "submitted $node -> job $id"
  i=$((i+1))
  if (( i % 10 == 0 )); then sleep 20; fi   # stagger in batches of 10
done
echo "submitted ${#NODES[@]} profiling jobs; ids recorded in $IDFILE"
