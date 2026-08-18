#!/bin/bash
# Submit one 11_decay_rate experiment to jaguar03 under the active reservation (discovered
# live, never hardcoded) and append the job id to the campaign's id file (cancel-safety rule).
# Usage: submit_experiment.sh <campaign_dir> <exp_name> [extra run_experiment.py args...]
# Prints "<jobid> <out_dir>". Falls back to any gpu-partition node when no reservation exists.
set -euo pipefail
CAMPAIGN="$1"; EXP="$2"; shift 2
OUT="$CAMPAIGN/$EXP"
mkdir -p "$OUT" "$CAMPAIGN/slurm"
IDFILE="$CAMPAIGN/slurm/submitted_jobids.txt"

# freeze the method at SUBMIT time: the runner imports $OUT/method.py, so later edits to
# code/method.py can never change what a queued experiment runs
cp /p/rlprojects/RND/11_decay_rate/code/method.py "$OUT/method.py"

# reservation discovery per the shared cluster rule (empty when none is active).
# PARTITION=cpu (or any partition) overrides the default jaguar03+reservation placement —
# used when the reserved node is full and jobs would pend on ReqNodeNotAvail.
RESARGS=(-p gpu)
if [ -n "${PARTITION:-}" ]; then
    RESARGS=(-p "$PARTITION")
else
    RES=$(scontrol show reservation -o 2>/dev/null | grep -i "Users=.*$USER" \
          | grep -oP 'ReservationName=\K\S+' | head -1)
    if [ -n "$RES" ]; then
        RESARGS+=(--nodelist=jaguar03 --reservation="$RES")
    fi
fi

ID=$(sbatch --parsable "${RESARGS[@]}" \
     --output="$OUT/slurm.out" --error="$OUT/slurm.out" \
     /p/rlprojects/RND/11_decay_rate/code/slurm/experiment.slurm "$OUT" "$@")
echo "$ID $(date '+%Y-%m-%dT%H:%M:%S%z') $EXP" >> "$IDFILE"
echo "$ID $OUT"
