#!/bin/bash
# The body of a rate-probe job: measure THIS run's two arms at every chunk size on one card.
#
#   RUN_DIR=<run folder> bash probe_rates.sh <job_folder_name> <copies per cell, comma separated> \
#       <arms, comma separated>
#
# Why it exists: the submission planner needs the rate of each arm at each chunk size it might
# choose, and the shared throughput survey measures a different trainer at 512 / 1,024 / 2,048 /
# 4,096 copies — none of which is a chunk size of this run, whose unit is 33 cells x 128 copies =
# 4,224 copies and whose chunks hold 16, 32, 64 or 128 copies per cell. Each invocation below is a
# real 400-iteration run of one arm at one chunk size, in the SAME 33-cell fused shape the science
# jobs use, written into <run>/probe/. The planner then prices every option from a measurement of
# this arm at this copy count, and transfers only the ratio between two cards.
#
# It also warms this node's compiled-program cache for every program the science jobs might run.
set -uo pipefail

: "${RUN_DIR:?RUN_DIR must be set to the run folder}"
source "$RUN_DIR/code/worker_env.sh"

JOB_FOLDER="$1"
COPIES_PER_CELL="$2"
ARMS="$3"

JOB_DIR="$RUN_DIR/slurm/jobs/$JOB_FOLDER"
PROBE_DIR="$RUN_DIR/probe"
mkdir -p "$JOB_DIR" "$PROBE_DIR"
exec > >(tee -a "$JOB_DIR/job.log") 2>&1

echo "[$(TZ=America/Los_Angeles date '+%Y-%m-%d %H:%M:%S PT')] rate probe on $(hostname), slurm job ${SLURM_JOB_ID:-none}"
echo "copies per cell $COPIES_PER_CELL, arms $ARMS, output into $PROBE_DIR"

# one 400-iteration run per (arm, chunk size), smallest chunk first so the cheapest cells land
# early and the planner has something to price if the job is cut short. The unit id spells the arm
# and the chunk's total copy count into the shard name, which is what the reader keys on.
for PER_CELL in ${COPIES_PER_CELL//,/ }; do
  for ARM in ${ARMS//,/ }; do
    TOTAL=$((PER_CELL * 33))
    UNIT="probe_${ARM//_/-}_copies-per-cell-${PER_CELL}_copies-${TOTAL}_$(hostname)"
    echo "--- $UNIT"
    "$PLATFORM_PYTHON" "$PLATFORM_ROOT/scripts/run_training.py" --run-dir "$PROBE_DIR" \
        --unit-id "$UNIT" --bonus "$ARM" \
        --learning-rates 1e-3,1e-4,1e-5 \
        --intrinsic-weights 1e-5,1e-4,1e-3,1e-2,1e-1,1,10,100,1000,10000,100000 \
        --copies-per-cell "$PER_CELL" --rollout-steps 128 --envs-per-copy 4 \
        --update-style full_batch --iterations 400 --window-iterations 200 --episode-steps 400 \
        --base-seed 0 --run-seed 0 --track-coverage
    echo "--- $UNIT exit $?"
  done
done
echo "[$(TZ=America/Los_Angeles date '+%Y-%m-%d %H:%M:%S PT')] rate probe finished"
