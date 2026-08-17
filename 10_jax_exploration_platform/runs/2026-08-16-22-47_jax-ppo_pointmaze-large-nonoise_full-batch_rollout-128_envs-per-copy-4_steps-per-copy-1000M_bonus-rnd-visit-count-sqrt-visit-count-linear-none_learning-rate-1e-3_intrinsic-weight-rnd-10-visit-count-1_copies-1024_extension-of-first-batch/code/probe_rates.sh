#!/bin/bash
# The body of a rate-probe job: measure THIS run's arms at several copy counts on one card.
#
#   RUN_DIR=<run folder> bash probe_rates.sh <job_folder_name> <copy counts, comma separated> \
#       <arms, comma separated>
#
# Why it exists: the submission planner needs the rate at each copy count it might choose, and the
# shared throughput survey stops at 512 copies. A four-way split of a 1,024-copy unit needs the
# rate at 256, which nothing has measured. Each invocation below is a real 200-iteration run of one
# arm at one copy count, written into <run>/probe/, so the planner's finer-split option is priced
# from a measurement instead of from an assumption.
#
# It also warms this node's compiled-program cache for the copy counts the science jobs will use.
set -uo pipefail

: "${RUN_DIR:?RUN_DIR must be set to the run folder}"
source "$RUN_DIR/code/worker_env.sh"

JOB_FOLDER="$1"
COPY_COUNTS="$2"
ARMS="$3"

JOB_DIR="$RUN_DIR/slurm/jobs/$JOB_FOLDER"
PROBE_DIR="$RUN_DIR/probe"
mkdir -p "$JOB_DIR" "$PROBE_DIR"
exec > >(tee -a "$JOB_DIR/job.log") 2>&1

echo "[$(TZ=America/Los_Angeles date '+%Y-%m-%d %H:%M:%S PT')] rate probe on $(hostname), slurm job ${SLURM_JOB_ID:-none}"
echo "copy counts $COPY_COUNTS, arms $ARMS, output into $PROBE_DIR"

# one 200-iteration run per (arm, copy count). The unit id spells both into the shard name, so the
# planner's reader can key on it, and the intrinsic weight is each arm's own winning value.
for COPIES in ${COPY_COUNTS//,/ }; do
  for ARM in ${ARMS//,/ }; do
    case "$ARM" in
      rnd_next_state) WEIGHT_ARGS=(--intrinsic-weights 10) ;;
      gt_position_velocity_sqrt|gt_position_velocity_linear) WEIGHT_ARGS=(--intrinsic-weights 1) ;;
      none) WEIGHT_ARGS=() ;;
      *) echo "unknown arm $ARM"; exit 2 ;;
    esac
    UNIT="probe_${ARM//_/-}_copies-${COPIES}_$(hostname)"
    echo "--- $UNIT"
    "$PLATFORM_PYTHON" "$PLATFORM_ROOT/scripts/run_training.py" --run-dir "$PROBE_DIR" \
        --unit-id "$UNIT" --bonus "$ARM" --learning-rates 1e-3 "${WEIGHT_ARGS[@]}" \
        --copies-per-cell "$COPIES" --rollout-steps 128 --envs-per-copy 4 \
        --update-style full_batch --iterations 200 --window-iterations 200 --episode-steps 400 \
        --base-seed 0 --run-seed 0 --track-coverage
    echo "--- $UNIT exit $?"
  done
done
echo "[$(TZ=America/Los_Angeles date '+%Y-%m-%d %H:%M:%S PT')] rate probe finished"
