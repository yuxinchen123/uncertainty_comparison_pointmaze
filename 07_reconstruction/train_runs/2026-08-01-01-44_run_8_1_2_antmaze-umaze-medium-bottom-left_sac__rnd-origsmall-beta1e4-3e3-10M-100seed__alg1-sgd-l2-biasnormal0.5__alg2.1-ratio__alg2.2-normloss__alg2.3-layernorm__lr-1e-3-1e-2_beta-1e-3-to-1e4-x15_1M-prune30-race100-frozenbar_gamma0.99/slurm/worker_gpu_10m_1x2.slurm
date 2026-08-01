#!/bin/bash
# ============================================================================================
# Run 8.1.2 TASK-R GPU worker job = 1 GPU with W=2 packed cuda worker tasks (pool pending_10m).
#
# Conservative-by-design: run-8.1 measured 7.0-7.7 GiB host RSS per cuda worker at 1M steps, and
# memory growth BEYOND 1M steps has never been measured, so each 10M worker gets 16G (2 x 16 = 32G)
# and only 2 are packed per GPU. GPU memory is ~230 MiB/run (never the constraint). Submit ONLY on
# host classes measured <= 7 h per 1M steps (cheetah02, jaguar01, cheetah04, adriatic01-06,
# cheetah08/09, jaguar02) — host CPU generation, not GPU class, sets the speed.
# worker.py's walltime guard: a pending_10m claim needs >= 90 h of remaining job walltime, so
# submit with the largest --time the partition/maintenance allows; a job with less remaining time
# simply exits (it never starts a run it cannot finish — this run has NO checkpoints).
# Partition, --time, --job-name and --export=ALL,SWEEP_ID=<id> come on the sbatch command line.
# srun --wait=0 is MANDATORY: workers exit at different times as the pool drains.
# Raising W above 2 happens only on canary/measured evidence (skill rule), by editing the THREE
# marked places: W=2 / --ntasks=2 / --mem=32G.
# ============================================================================================
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-core=2
#SBATCH --mem=32G
#SBATCH --output=/p/rlprojects/RND/07_reconstruction/train_runs/2026-08-01-01-44_run_8_1_2_antmaze-umaze-medium-bottom-left_sac__rnd-origsmall-beta1e4-3e3-10M-100seed__alg1-sgd-l2-biasnormal0.5__alg2.1-ratio__alg2.2-normloss__alg2.3-layernorm__lr-1e-3-1e-2_beta-1e-3-to-1e4-x15_1M-prune30-race100-frozenbar_gamma0.99/logs/%x_%j.log

# Number of cuda worker tasks packed on the single GPU — raise only on measured evidence.
W=2

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export TMPDIR=/tmp
export WORKER_DEVICE=cuda
export WORKER_POOLS="pending_10m"
export PROJ_DIR="/p/rlprojects/RND/07_reconstruction"
export RUN_DIR="/p/rlprojects/RND/07_reconstruction/train_runs/2026-08-01-01-44_run_8_1_2_antmaze-umaze-medium-bottom-left_sac__rnd-origsmall-beta1e4-3e3-10M-100seed__alg1-sgd-l2-biasnormal0.5__alg2.1-ratio__alg2.2-normloss__alg2.3-layernorm__lr-1e-3-1e-2_beta-1e-3-to-1e4-x15_1M-prune30-race100-frozenbar_gamma0.99"
export SWEEP_ID="${SWEEP_ID:?SWEEP_ID must be passed via sbatch --export=ALL,SWEEP_ID=<id>}"
PY="/p/rlprojects/RND/.venvs/exploration/bin/python"

# per-minute memory sampling (the endurance measurement this code has never had past 1M steps):
# one background loop per job writes host-RSS of every worker python + GPU memory to logs/.
# The sampler dies with the job (it is a child of the batch step).
SAMPLE_CSV="$RUN_DIR/logs/rss_${SLURM_JOB_ID}.csv"
(
  echo "epoch,rss_kb_per_python_pid,gpu_mem_mib" > "$SAMPLE_CSV"
  while true; do
    rss=$(ps -u "$USER" -o rss=,comm= | awk '$2 ~ /python/ {printf "%s;", $1}')
    gpu=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | paste -sd';')
    echo "$(date +%s),${rss},${gpu}" >> "$SAMPLE_CSV"
    sleep 60
  done
) &
srun --wait=0 --ntasks="$W" "$PY" "$RUN_DIR/slurm/worker.py"
