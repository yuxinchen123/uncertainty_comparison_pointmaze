#!/bin/bash
# The body of one job of this sweep: canary, resume check, then the science run — on one card,
# without ever releasing it.
#
#   RUN_DIR=<run folder> bash run_chunk.sh <unit_id> <job_folder_name>
#
# Three phases in one job, in this order:
#
#   1. CANARY. The chunk's own arguments with `--iterations 400` appended (a repeated argument wins
#      in argparse), written into <run>/canary/. It proves the card holds the program at this copy
#      count, writes the node-local compiled-program cache the science run then reads, and measures
#      this chunk's real rate on this card so the 20-minute tick can compare the plan against it.
#      If it does not end in a completion record the job stops here and the science run never
#      starts.
#   2. RESUME CHECK. The identical canary command again. The runner must log that the unit is
#      already complete and skip it, leaving the shard's record count unchanged. This is the whole
#      resume contract of this platform: no model state is saved, so the resumable thing is the
#      unit, and a unit whose shard ends in a completion record makes a re-run a logged no-op.
#   3. SCIENCE RUN. The chunk's own arguments, into the run folder itself, moving the unit's queue
#      marker pending -> running -> done|failed.
#
# Why the three are one job rather than a canary phase followed by a submission phase: the cards
# this run wants are being taken by other users within minutes, and a canary that finishes releases
# its card. Holding the card across all three keeps the plan the plan. What the canary phase is for
# is not lost — a bad card still stops the job before any science runs, and the canary's own record
# is on disk for the tick to read.
#
# The job folder is <run>/slurm/jobs/<job_folder_name>/ and holds this submission's whole record:
# the script as submitted, the unit it was given, the hardware it landed on, and its log.
# Re-running the same command is safe — a unit whose shard already ends in a completion record is
# skipped, so a requeued job resumes at the first phase that has not finished.
set -uo pipefail

# RUN_DIR is passed in by the caller; sbatch copies its script to a spool directory, so a job must
# never work out the run folder from its own path
: "${RUN_DIR:?RUN_DIR must be set to the run folder}"
source "$RUN_DIR/code/worker_env.sh"

UNIT_ID="$1"
JOB_FOLDER="$2"
CANARY_ITERATIONS="${CANARY_ITERATIONS:-400}"

JOB_DIR="$RUN_DIR/slurm/jobs/$JOB_FOLDER"
mkdir -p "$JOB_DIR" "$RUN_DIR/canary"
# everything this job prints goes to its own folder as well as to the scheduler's file
exec > >(tee -a "$JOB_DIR/job.log") 2>&1

echo "[$(TZ=America/Los_Angeles date '+%Y-%m-%d %H:%M:%S PT')] chunk job, folder $JOB_DIR"
echo "host $(hostname), slurm job ${SLURM_JOB_ID:-none}, unit $UNIT_ID"

# find the unit's queue entry wherever it currently sits; its folder is its state
UNIT_FILE=""
for state in pending running done failed; do
  CANDIDATE="$RUN_DIR/queue/$state/$UNIT_ID.json"
  if [ -f "$CANDIDATE" ]; then UNIT_FILE="$CANDIDATE"; fi
done
if [ -z "$UNIT_FILE" ]; then echo "no queue entry for unit $UNIT_ID"; exit 2; fi

# what this job was told to do, and the card it landed on — both kept beside its log, because one
# sweep's jobs run on different hardware and the run manifest cannot hold that
"$PLATFORM_PYTHON" "$RUN_DIR/code/write_job_records.py" \
    --unit-file "$UNIT_FILE" --job-dir "$JOB_DIR" --mode "canary_then_real"
cat "$JOB_DIR/hardware.json"

# the unit's own arguments come from its queue entry, so the job script never restates them. This
# happens BEFORE the claim below, because claiming RENAMES the entry and would leave this path
# pointing at a file that no longer exists — the job would then start with the runner's defaults
# instead of the unit's arguments.
mapfile -t UNIT_ARGS < <("$PLATFORM_PYTHON" -c "
import json, sys
print('\n'.join(json.load(open(sys.argv[1]))['arguments']))" "$UNIT_FILE")
if [ "${#UNIT_ARGS[@]}" -lt 4 ]; then
  echo "read no arguments from $UNIT_FILE; refusing to start with the runner's defaults"
  exit 3
fi

CANARY_SHARD="$RUN_DIR/canary/data/$UNIT_ID.jsonl"

# phase 1: the canary, at a few hundred iterations, into the canary folder
echo "[$(TZ=America/Los_Angeles date '+%Y-%m-%d %H:%M:%S PT')] canary: $CANARY_ITERATIONS iterations"
"$PLATFORM_PYTHON" "$PLATFORM_ROOT/scripts/run_training.py" --run-dir "$RUN_DIR/canary" \
    "${UNIT_ARGS[@]}" --iterations "$CANARY_ITERATIONS"
CANARY_STATUS=$?
if [ $CANARY_STATUS -ne 0 ] || ! grep -q '"record": "unit_complete"' "$CANARY_SHARD"; then
  echo "canary did not complete (exit $CANARY_STATUS); the science run will not start"
  exit 4
fi
CANARY_RECORDS=$(wc -l < "$CANARY_SHARD")

# phase 2: the resume check — the same command again must be a logged no-op that adds no record
echo "[$(TZ=America/Los_Angeles date '+%Y-%m-%d %H:%M:%S PT')] resume check: the same command again"
"$PLATFORM_PYTHON" "$PLATFORM_ROOT/scripts/run_training.py" --run-dir "$RUN_DIR/canary" \
    "${UNIT_ARGS[@]}" --iterations "$CANARY_ITERATIONS"
RESUME_STATUS=$?
RESUME_RECORDS=$(wc -l < "$CANARY_SHARD")
if [ $RESUME_STATUS -ne 0 ] || [ "$RESUME_RECORDS" != "$CANARY_RECORDS" ]; then
  echo "resume check failed: exit $RESUME_STATUS, $CANARY_RECORDS records before and "\
       "$RESUME_RECORDS after; the science run will not start"
  exit 5
fi
echo "resume check passed: $CANARY_RECORDS records before and after, nothing repeated"

# phase 3: claim the unit and run it. A claim is a rename, so two jobs cannot both take it.
if [ -f "$RUN_DIR/queue/pending/$UNIT_ID.json" ]; then
  mv "$RUN_DIR/queue/pending/$UNIT_ID.json" "$RUN_DIR/queue/running/$UNIT_ID.json"
fi
echo "[$(TZ=America/Los_Angeles date '+%Y-%m-%d %H:%M:%S PT')] science run: ${UNIT_ARGS[*]}"
"$PLATFORM_PYTHON" "$PLATFORM_ROOT/scripts/run_training.py" --run-dir "$RUN_DIR" "${UNIT_ARGS[@]}"
STATUS=$?

# the unit's final state, and the job's own exit code recorded with it
if [ $STATUS -eq 0 ]; then
  [ -f "$RUN_DIR/queue/running/$UNIT_ID.json" ] && \
    mv "$RUN_DIR/queue/running/$UNIT_ID.json" "$RUN_DIR/queue/done/$UNIT_ID.json"
else
  [ -f "$RUN_DIR/queue/running/$UNIT_ID.json" ] && \
    mv "$RUN_DIR/queue/running/$UNIT_ID.json" "$RUN_DIR/queue/failed/$UNIT_ID.json"
fi
echo "[$(TZ=America/Los_Angeles date '+%Y-%m-%d %H:%M:%S PT')] unit $UNIT_ID exit $STATUS"
exit $STATUS
