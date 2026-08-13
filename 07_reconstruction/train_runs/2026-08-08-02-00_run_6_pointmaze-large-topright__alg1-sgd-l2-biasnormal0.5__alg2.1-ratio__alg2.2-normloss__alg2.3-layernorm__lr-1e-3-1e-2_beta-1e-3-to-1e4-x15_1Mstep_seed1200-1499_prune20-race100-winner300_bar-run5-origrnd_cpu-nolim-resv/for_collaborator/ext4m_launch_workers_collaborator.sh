#!/bin/bash
# COLLABORATOR one-command submission for the run-6 4M extension sweep (ext4m).
# Sizes jobs from the live cluster state under YOUR OWN cpu/nolim pool caps, submits with YOUR
# job-name prefix, and appends every id to YOUR OWN id file. The reservation bucket is
# owner-only and skipped automatically for your uid.
set -eu
HERE="$(cd "$(dirname "$0")" && pwd)"
RUN_DIR="$(dirname "$HERE")"
SWEEP_ID="$(cat "$RUN_DIR/slurm/EXT4M_SWEEP_ID.txt")"
PY="/p/rlprojects/RND/.venvs/exploration/bin/python"
export PYTHONNOUSERSITE=1
exec "$PY" "$RUN_DIR/slurm/ext4m_plan_jobs.py" \
  --sweep_id "$SWEEP_ID" --submit \
  --jobname "e4mclb" \
  --script "$HERE/ext4m_worker_cpu_collab.slurm" \
  --idfile "$HERE/submitted_jobids_${SWEEP_ID}_$USER.txt"
