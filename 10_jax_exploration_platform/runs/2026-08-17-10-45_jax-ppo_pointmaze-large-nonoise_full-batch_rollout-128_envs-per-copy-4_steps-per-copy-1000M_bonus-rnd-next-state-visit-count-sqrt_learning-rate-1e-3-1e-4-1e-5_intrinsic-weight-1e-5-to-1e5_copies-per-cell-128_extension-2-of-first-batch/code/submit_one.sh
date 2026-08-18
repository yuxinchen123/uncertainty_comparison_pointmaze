#!/bin/bash
# Submit ONE job of this sweep — one chunk, one card, one job id. Never an array: one id per chunk
# is what makes a single chunk cancellable and resumable without touching the others.
#
#   RUN_DIR=<run folder> [RESERVATION=<name>] bash submit_one.sh <unit_id> <node> <walltime> \
#       [cpus] [memory]
#
# The processor and memory asks are small on purpose. The trainer keeps its arrays on the card and
# the host only dispatches, and the rate probes of this run measured 1.5 to 2.9 GB of host memory
# per process; meanwhile this plan puts eight jobs on one lotus card slot group and four on
# affogato11, whose free memory and processors would not hold eight or four of the previous run's
# 48 GB, 8-processor jobs.
#
# The job runs the chunk's canary, its resume check and its science run on the same card, in that
# order — see code/run_chunk.sh for why the three are one job.
#
# The script writes the sbatch file it is about to submit into slurm/staged/, submits that exact
# file, and appends the returned id to slurm/submitted_jobids.txt in the same command — the id file
# is the only thing a cancel may ever read from.
set -euo pipefail

: "${RUN_DIR:?RUN_DIR must be set to the run folder}"
UNIT_ID="$1"; NODE="$2"; WALLTIME="$3"; CPUS="${4:-6}"; MEMORY="${5:-24G}"
RESERVATION="${RESERVATION:-}"

STAGED_DIR="$RUN_DIR/slurm/staged"
mkdir -p "$STAGED_DIR" "$RUN_DIR/slurm/logs"
SCRIPT="$STAGED_DIR/chunk_${UNIT_ID}_${NODE}.sbatch"
# the job name a human reads in squeue: the unit number and the chunk, which is what a monitoring
# table is keyed on
JOB_NAME="pmjax2-$(echo "$UNIT_ID" | sed -E 's/^(unit-[0-9]+).*(chunk-[0-9]+-of-[0-9]+).*/\1-\2/')"
RESERVATION_LINE=""
if [ -n "$RESERVATION" ]; then RESERVATION_LINE="#SBATCH --reservation=$RESERVATION"; fi

# the script as submitted: every scheduler directive is written into the file rather than passed on
# the command line, so the copy kept beside the job's log is the whole submission
cat > "$SCRIPT" <<SBATCH
#!/bin/bash
#SBATCH --job-name=$JOB_NAME
#SBATCH --partition=gpu
#SBATCH --nodelist=$NODE
$RESERVATION_LINE
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=$CPUS
#SBATCH --mem=$MEMORY
#SBATCH --time=$WALLTIME
#SBATCH --open-mode=append
#SBATCH --output=$RUN_DIR/slurm/logs/%j.out
#SBATCH --error=$RUN_DIR/slurm/logs/%j.out
set -uo pipefail
export RUN_DIR="$RUN_DIR"
JOB_FOLDER="\${SLURM_JOB_ID}_\${SLURMD_NODENAME}"
mkdir -p "\$RUN_DIR/slurm/jobs/\$JOB_FOLDER"
# the submitted file itself, copied from its staged path: sbatch runs a spool copy, so a job can
# never find its own source by \$0
cp "$SCRIPT" "\$RUN_DIR/slurm/jobs/\$JOB_FOLDER/job.sbatch"
exec bash "\$RUN_DIR/code/run_chunk.sh" "$UNIT_ID" "\$JOB_FOLDER"
SBATCH

JOB_ID=$(sbatch --parsable "$SCRIPT" | tee -a "$RUN_DIR/slurm/submitted_jobids.txt")
# which unit this id was given, recorded at submit time. A job writes `assignment.json` only once
# it starts, so without this a PENDING resubmission is invisible and the status table falls back to
# the cancelled job it replaced.
printf '%s\t%s\n' "$JOB_ID" "$UNIT_ID" >> "$RUN_DIR/slurm/unit_jobs.tsv"
echo "submitted $UNIT_ID on $NODE as job $JOB_ID (walltime $WALLTIME, $CPUS cpus, $MEMORY, reservation ${RESERVATION:-none})"
