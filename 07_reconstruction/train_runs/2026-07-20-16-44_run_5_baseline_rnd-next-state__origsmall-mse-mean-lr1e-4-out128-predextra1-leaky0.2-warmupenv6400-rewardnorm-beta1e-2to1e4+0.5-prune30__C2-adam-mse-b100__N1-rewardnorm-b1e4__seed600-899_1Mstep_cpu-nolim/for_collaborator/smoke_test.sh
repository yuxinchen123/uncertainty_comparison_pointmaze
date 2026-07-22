#!/bin/bash
# Smoke test: submit ONE small (16-worker) job to the open cpu partition, then print the checks.
# Run this before your first real wave; scale up only after every check passes.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/packet_env.sh"
[[ "$USER" == "sl5nw" ]] && { echo "you are the sweep owner — the owner fleet already ran its smoke checks"; exit 3; }
[[ -f "$RUN_DIR/optuna/SWEEP_COMPLETE" ]] && { echo "sweep complete — nothing to test against"; exit 0; }
mkdir -p "$LOGDIR"

# one 16x1 job, no nodelist (any >=16-core cpu-partition node takes 16 tasks)
out=$(sbatch --job-name="${PREFIX}0" --chdir="$PROJ_DIR" --nodes=1 --ntasks=16 --cpus-per-task=1 \
      --ntasks-per-core=2 --mem-per-cpu=2G --output="$LOGDIR/${PREFIX}0_%j.log" --time=4-00:00:00 \
      --comment="${SWEEP_ID}_${USER}" --export=ALL,SWEEP_ID="$SWEEP_ID" --partition=cpu \
      "$HERE/worker_16x1_collab.slurm")
id=$(grep -oP '[0-9]+$' <<< "$out")
[[ -n "$id" ]] || { echo "ERROR: sbatch gave no id: $out"; exit 1; }
flock "$IDFILE.lock" bash -c "echo $id >> '$IDFILE'"
echo "smoke job $id submitted. Now run these checks (allow ~2 min for start + claim):"
echo "  1. state + shape :  sacct -j $id --format=JobID,JobName,AllocCPUS,ReqMem,State,NodeList"
echo "     -> AllocCPUS must be 16 (if it is 32, --ntasks-per-core=2 was lost — report a problem)"
echo "  2. workers claim :  grep -m3 'claimed' $LOGDIR/${PREFIX}0_${id}.log"
echo "  3. running marks :  ls $RUN_DIR/queue/$SWEEP_ID/running | wc -l   (should grow by up to 16)"
echo "  4. first records :  ls -t $RUN_DIR/data/$SWEEP_ID/local/ | head -3   (JSONs appear after ~30-40 min)"
echo "  5. no fast fails :  ls $FC/problems/open/   (must stay empty; a fast-failure report means"
echo "     your environment access is broken — stop and wait for the owner's monitor to requeue)"
echo "All five pass -> start the monitor loop, then run launch_workers_collaborator.sh."
