#!/bin/bash
# Add worker jobs to the owner's already-built queue, under YOUR OWN Slurm caps.
#
# This script ONLY submits worker jobs. It never builds the queue, never renames a queue marker,
# never truncates a configuration, never touches another user's files. Sizing is done by the run's
# own slurm/plan_jobs.py, which reads the live node table and reads the per-user pool caps for
# whoever runs it — so it computes YOUR room, not the owner's.
#
#   preview:  DRY=1 bash launch_workers_collaborator.sh
#   submit:        bash launch_workers_collaborator.sh
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/packet_env.sh"

# 1. owner guard — the owner has their own launcher and their own id file
if [[ "$USER" == "sl5nw" ]]; then
  echo "You are the sweep owner. Use \$RUN_DIR/slurm/launch_queue.sh, not this packet." >&2
  exit 3
fi
# 2. finished-sweep guard
if [[ -f "$COMPLETE_SENTINEL" ]]; then
  echo "SWEEP_COMPLETE exists — this sweep is finished. Nothing to submit."
  exit 0
fi
# 3. the queue must exist and still have work
if [[ ! -d "$POOL" ]]; then
  echo "ERROR: no pending pool at $POOL — the owner has not built the queue yet." >&2
  exit 1
fi
left=$(ls "$POOL" | wc -l)
echo "[queue] $left runs still waiting to be claimed"
if (( left == 0 )); then
  echo "Nothing left to claim. Not submitting."
  exit 0
fi

mkdir -p "$LOGDIR" "$FC/problems/open" "$FC/problems/resolved"
export PYTHONNOUSERSITE=1

# 4. one call per partition, so each bucket carries that partition's own random job-name prefix.
#    plan_jobs.py applies, for whoever runs it: ntasks <= the node's physical cores, ntasks <= its
#    free allocatable threads, memory per worker <= 92% of RealMemory/CPUEfctv capped at 2 GB, the
#    per-user pool cap minus 16 threads of headroom, --ntasks-per-core=2 only where the node has two
#    threads per core, reserved nodes excluded, the partition's own MaxTime as --time, the
#    SWEEP_COMPLETE guard, the queue-depth guard, batch-of-10-then-sleep-30, and the flock-guarded
#    append of every returned id to YOUR id file.
MODE=(--submit)
[[ "${DRY:-0}" == "1" ]] && MODE=()
"$PY" "$RUN_DIR/slurm/plan_jobs.py" \
  --sweep_id "$SWEEP_ID" \
  --script "$FC/worker_cpu_collab.slurm" \
  --jobname "$(prefix_for_partition cpu)" \
  --idfile "$IDFILE" \
  "${MODE[@]}"

if [[ "${DRY:-0}" == "1" ]]; then
  echo
  echo "[dry run] nothing was submitted. Drop DRY=1 to submit the plan above."
  exit 0
fi
echo
echo "[done] your ids are in $IDFILE — the ONLY list you may ever scancel from."
echo "[verify] sacct -j \$(head -1 \"$IDFILE\") --format=JobID,JobName,AllocCPUS,ReqMem,NodeList,Timelimit"
echo "         AllocCPUS must equal the job's --ntasks. If it is double, tell the owner."
echo "[next] start the passive monitor:  nohup bash $HERE/monitor_collaborator.sh > $LOGDIR/monitor.log 2>&1 &"
