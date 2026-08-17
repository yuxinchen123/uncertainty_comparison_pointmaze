#!/bin/bash
# The body of one job of this sweep: set up the per-submission folder, then run one work unit.
#
#   RUN_DIR=<run folder> bash run_unit.sh <mode> <unit_id> <job_folder_name> [extra run_training args]
#
# mode is `real` or `canary`. A real job writes into the run folder itself and moves the unit's
# queue marker pending -> running -> done|failed. A canary writes into <run>/canary/, touches no
# queue marker, and is expected to carry a short `--iterations` on the end of the command line (a
# repeated argument wins in argparse), so it proves the card, the memory and the compiled-program
# cache without producing science records.
#
# The job folder is <run>/slurm/jobs/<job_folder_name>/ and holds this submission's whole record:
# the script as submitted, the unit it was given, the hardware it landed on, and its log. Re-running
# the same command is safe — a unit whose shard already ends in a completion record is skipped.
set -uo pipefail

# RUN_DIR is passed in by the caller; sbatch copies its script to a spool directory, so a job must
# never work out the run folder from its own path
: "${RUN_DIR:?RUN_DIR must be set to the run folder}"
source "$RUN_DIR/code/worker_env.sh"

MODE="$1"
UNIT_ID="$2"
JOB_FOLDER="$3"
shift 3

JOB_DIR="$RUN_DIR/slurm/jobs/$JOB_FOLDER"
OUTPUT_DIR="$RUN_DIR"
if [ "$MODE" = "canary" ]; then OUTPUT_DIR="$RUN_DIR/canary"; fi
mkdir -p "$JOB_DIR" "$OUTPUT_DIR"
# everything this job prints goes to its own folder as well as to the scheduler's file
exec > >(tee -a "$JOB_DIR/job.log") 2>&1

echo "[$(TZ=America/Los_Angeles date '+%Y-%m-%d %H:%M:%S PT')] $MODE job, folder $JOB_DIR"
echo "host $(hostname), slurm job ${SLURM_JOB_ID:-none}, unit $UNIT_ID, output into $OUTPUT_DIR"

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
    --unit-file "$UNIT_FILE" --job-dir "$JOB_DIR" --mode "$MODE"
cat "$JOB_DIR/hardware.json"

# claim the unit: a claim is a rename, so two jobs cannot both take it. A canary claims nothing.
if [ "$MODE" = "real" ] && [ -f "$RUN_DIR/queue/pending/$UNIT_ID.json" ]; then
  mv "$RUN_DIR/queue/pending/$UNIT_ID.json" "$RUN_DIR/queue/running/$UNIT_ID.json"
fi

# the unit's own arguments come from its queue entry, so the job script never restates them
mapfile -t UNIT_ARGS < <("$PLATFORM_PYTHON" -c "
import json, sys
print('\n'.join(json.load(open(sys.argv[1]))['arguments']))" "$UNIT_FILE")

echo "running: $PLATFORM_PYTHON $PLATFORM_ROOT/scripts/run_training.py --run-dir $OUTPUT_DIR ${UNIT_ARGS[*]} $*"
"$PLATFORM_PYTHON" "$PLATFORM_ROOT/scripts/run_training.py" --run-dir "$OUTPUT_DIR" \
    "${UNIT_ARGS[@]}" "$@"
STATUS=$?

# the unit's final state, and the job's own exit code recorded with it
if [ "$MODE" = "real" ]; then
  if [ $STATUS -eq 0 ]; then
    [ -f "$RUN_DIR/queue/running/$UNIT_ID.json" ] && \
      mv "$RUN_DIR/queue/running/$UNIT_ID.json" "$RUN_DIR/queue/done/$UNIT_ID.json"
  else
    [ -f "$RUN_DIR/queue/running/$UNIT_ID.json" ] && \
      mv "$RUN_DIR/queue/running/$UNIT_ID.json" "$RUN_DIR/queue/failed/$UNIT_ID.json"
  fi
fi
echo "[$(TZ=America/Los_Angeles date '+%Y-%m-%d %H:%M:%S PT')] unit $UNIT_ID exit $STATUS"
exit $STATUS
