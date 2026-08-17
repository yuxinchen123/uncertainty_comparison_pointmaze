#!/bin/bash
# Submit ONE job of this sweep — one work unit, one card, one job id. Never an array: one id per
# unit is what makes a single unit cancellable and resumable without touching the others.
#
#   RUN_DIR=<run folder> bash submit_one.sh <mode> <unit_id> <node> <walltime> [extra args...]
#
# mode is `real` or `canary`. The script writes the sbatch file it is about to submit into
# slurm/staged/, submits that exact file, and appends the returned id to slurm/submitted_jobids.txt
# in the same command — the id file is the only thing a cancel may ever read from.
set -euo pipefail

: "${RUN_DIR:?RUN_DIR must be set to the run folder}"
MODE="$1"; UNIT_ID="$2"; NODE="$3"; WALLTIME="$4"; shift 4
EXTRA_ARGS="$*"

STAGED_DIR="$RUN_DIR/slurm/staged"
mkdir -p "$STAGED_DIR" "$RUN_DIR/slurm/logs"
SCRIPT="$STAGED_DIR/${MODE}_${UNIT_ID}_${NODE}.sbatch"
JOB_NAME="pmjax-${MODE}-${UNIT_ID%%_*}"

# the script as submitted: every scheduler directive is written into the file rather than passed on
# the command line, so the copy kept beside the job's log is the whole submission
cat > "$SCRIPT" <<SBATCH
#!/bin/bash
#SBATCH --job-name=$JOB_NAME
#SBATCH --partition=gpu
#SBATCH --nodelist=$NODE
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=$WALLTIME
#SBATCH --output=$RUN_DIR/slurm/logs/%j.out
#SBATCH --error=$RUN_DIR/slurm/logs/%j.out
set -uo pipefail
export RUN_DIR="$RUN_DIR"
JOB_FOLDER="\${SLURM_JOB_ID}_\${SLURMD_NODENAME}"
mkdir -p "\$RUN_DIR/slurm/jobs/\$JOB_FOLDER"
# the submitted file itself, copied from its staged path: sbatch runs a spool copy, so a job can
# never find its own source by \$0
cp "$SCRIPT" "\$RUN_DIR/slurm/jobs/\$JOB_FOLDER/job.sbatch"
exec bash "\$RUN_DIR/code/run_unit.sh" "$MODE" "$UNIT_ID" "\$JOB_FOLDER" $EXTRA_ARGS
SBATCH

JOB_ID=$(sbatch --parsable "$SCRIPT" | tee -a "$RUN_DIR/slurm/submitted_jobids.txt")
echo "submitted $MODE $UNIT_ID on $NODE as job $JOB_ID (walltime $WALLTIME, extra args: ${EXTRA_ARGS:-none})"
