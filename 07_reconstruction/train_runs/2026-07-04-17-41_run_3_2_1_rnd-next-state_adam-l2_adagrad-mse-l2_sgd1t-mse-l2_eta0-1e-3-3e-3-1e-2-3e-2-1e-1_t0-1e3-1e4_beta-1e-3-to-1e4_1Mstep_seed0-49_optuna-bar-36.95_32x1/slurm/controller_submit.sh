#!/bin/bash
# Submit the run-3.2.1 optuna controller as a 1-CPU Slurm job (job name idx-build).
# Usage: bash controller_submit.sh <sweep_id> [nolim|reservation]
#
# Liveness strategy (plan): try nolim first (20-day MaxTime, per-user cap unused); check squeue
# ~1 min after submitting — if PENDING, scancel THAT id (ours, just submitted) and rerun this
# script with "reservation" to place it on puma01 via the reservation (the worker plan leaves
# ~64 reserved threads idle there, so a 1-CPU job starts immediately; 4-day time, the monitor
# resubmits when it ends). The controller is restart-safe by design.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
RUN_DIR="$(dirname "$HERE")"
LOGDIR="$RUN_DIR/logs"; mkdir -p "$LOGDIR"
PY="/u/sl5nw/.conda/envs/exploration/bin/python"
SWEEP_ID="${1:?usage: controller_submit.sh <sweep_id> [nolim|reservation]}"
MODE="${2:-nolim}"

if [[ "$MODE" == "nolim" ]]; then
  PLACE=(--partition=nolim --time=14-00:00:00)
else
  # reservation fallback: discover the name dynamically (never hardcode)
  RES=$(scontrol show reservation -o 2>/dev/null | grep -i 'Users=.*sl5nw' | grep -oP 'ReservationName=\K\S+' | head -1)
  [[ -z "$RES" ]] && { echo "ERROR: no active reservation for the fallback"; exit 1; }
  PLACE=(--partition=cpu --reservation="$RES" --nodelist=puma01 --qos=csresnolim --time=4-00:00:00)
fi

out=$(sbatch --job-name=idx-build "${PLACE[@]}" --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=4G \
      --output="$LOGDIR/controller_%j.log" \
      --wrap="$PY $HERE/optuna_controller.py --sweep_id $SWEEP_ID")
id=$(grep -oP '[0-9]+$' <<< "$out")
[[ -n "$id" ]] || { echo "ERROR: sbatch gave no id: $out"; exit 1; }
echo "$id" >> "$HERE/submitted_jobids_${SWEEP_ID}.txt"
echo "$id" >> "$RUN_DIR/optuna/controller_jobid.txt"
echo "[controller] job $id submitted ($MODE). Verify RUNNING within ~1 min:  squeue -j $id"
echo "[controller] if it stays PENDING: scancel $id && bash $HERE/controller_submit.sh $SWEEP_ID reservation"
