#!/bin/bash
# Submit ONE rate-probe job: one card, several chunk sizes, both arms, 400 iterations each.
#
#   RUN_DIR=<run folder> bash submit_probe.sh <node> <walltime> <copies per cell> <arms> [cpus]
#
# The script writes the sbatch file it is about to submit into slurm/staged/, submits that exact
# file, and appends the returned id to slurm/submitted_jobids.txt in the same command — the id file
# is the only thing a cancel may ever read from.
set -euo pipefail

: "${RUN_DIR:?RUN_DIR must be set to the run folder}"
NODE="$1"; WALLTIME="$2"; COPIES_PER_CELL="$3"; ARMS="$4"; CPUS="${5:-8}"

STAGED_DIR="$RUN_DIR/slurm/staged"
mkdir -p "$STAGED_DIR" "$RUN_DIR/slurm/logs"
SCRIPT="$STAGED_DIR/probe_${NODE}.sbatch"

cat > "$SCRIPT" <<SBATCH
#!/bin/bash
#SBATCH --job-name=pmjax2-probe-$NODE
#SBATCH --partition=gpu
#SBATCH --nodelist=$NODE
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=$CPUS
#SBATCH --mem=48G
#SBATCH --time=$WALLTIME
#SBATCH --output=$RUN_DIR/slurm/logs/%j.out
#SBATCH --error=$RUN_DIR/slurm/logs/%j.out
set -uo pipefail
export RUN_DIR="$RUN_DIR"
JOB_FOLDER="\${SLURM_JOB_ID}_\${SLURMD_NODENAME}"
mkdir -p "\$RUN_DIR/slurm/jobs/\$JOB_FOLDER"
cp "$SCRIPT" "\$RUN_DIR/slurm/jobs/\$JOB_FOLDER/job.sbatch"
exec bash "\$RUN_DIR/code/probe_rates.sh" "\$JOB_FOLDER" "$COPIES_PER_CELL" "$ARMS"
SBATCH

JOB_ID=$(sbatch --parsable "$SCRIPT" | tee -a "$RUN_DIR/slurm/submitted_jobids.txt")
echo "submitted rate probe on $NODE as job $JOB_ID (walltime $WALLTIME, copies per cell $COPIES_PER_CELL, arms $ARMS, cpus $CPUS)"
