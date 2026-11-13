#!/bin/bash
#SBATCH --job-name=rnd-maze-warp-import-check-card
#SBATCH --partition=mit_normal_gpu
#SBATCH --account=mit_amf_advanced_gpu
#SBATCH --qos=mit_amf_advanced_gpu
#SBATCH --gpus=h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH -x node3401-c1
#SBATCH --open-mode=append
# Import check of the Warp environment on one MIT H100 card: the card guard, then imports and the device lists of
# jax and warp (import_check_rnd_platform_jax_mujoco_warp.py card). Nothing is compiled or trained: no probe,
# benchmark or canary. The card guard below is the shared one
# (/p/rlprojects/.claude/skills/mit-submit-gpu-sweep/scripts/card_guard.sh) with its count taken from jax, as in the
# fleet's worker jobs; a wrong card count ends this check (GUARD_ACTION=exit), it is not requeued.
#   sbatch --parsable -o <setup folder>/import_check_card_%j.out import_check_card_job_mit.sh <setup folder>
HERE=${1:?give the folder that holds import_check_rnd_platform_jax_mujoco_warp.py}
PY=/orcd/scratch/orcd/008/shuzel/envs/rnd_platform_jax_mujoco_warp/bin/python
export PYTHONNOUSERSITE=1 XLA_PYTHON_CLIENT_PREALLOCATE=false
unset LD_LIBRARY_PATH
# warp keeps compiled kernels in ~/.cache/warp unless told otherwise; this check compiles none, but its folder
# goes to the node-local temporary folder, as in the trainer, and not to the home folder (90% full)
export WARP_CACHE_PATH="${TMPDIR:-/tmp}/$(id -un)_warp_kernel_cache"
echo "cluster mit, job ${SLURM_JOB_ID:-none}, host $(hostname), start $(date -Is), SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-unset}"
NEED_CARDS=1
RUN_DIR="$HERE"
GUARD_ACTION=exit
# CARD GUARD (rule ~/.claude/rules/mit-single-gpu-request.md; P10 of ../submission_principle.md).
# Paste right after the #SBATCH lines, before any worker, or copy this file into the run folder and `source` it.
# Set NEED_CARDS and a PY with torch first. Ends the job within seconds when the allocation is short or long:
# nvidia-smi is compared first, and torch (its import took 273 to 433 s on a cold node) is imported only when
# nvidia-smi already shows the right count.
#   GUARD_ACTION=exit     (default) print the counts and exit 3
#   GUARD_ACTION=requeue  requeue this job (keeps the id, resets the age), at most
#                         GUARD_MAX_RESTARTS times (default 2), then exit 3
# GUARD_ACTION comes from the environment or, when that is unset, from the file $RUN_DIR/guard_action, so a job
# that is already pending can be switched to requeue by writing that file before it starts.
NEED_CARDS=${NEED_CARDS:?set NEED_CARDS to the number of cards the run needs}
GUARD_ACTION=${GUARD_ACTION:-$(tr -d '[:space:]' 2>/dev/null < "${RUN_DIR:-/nonexistent}/guard_action")}
GUARD_ACTION=${GUARD_ACTION:-exit}
have_smi=$(nvidia-smi -L 2>/dev/null | grep -c '^GPU ')
if [ "$have_smi" -eq "$NEED_CARDS" ]; then
  have_torch=$("${PY:-python}" -c 'import jax; print(len(jax.devices("gpu")))' 2>/dev/null || echo "?")
else
  have_torch="not-checked"
fi
echo "[card guard] $(date -Is) job=${SLURM_JOB_ID:-?} node=$(hostname) need=$NEED_CARDS nvidia-smi=$have_smi torch=$have_torch restart=${SLURM_RESTART_COUNT:-0} action=$GUARD_ACTION"
if [ "$have_smi" -ne "$NEED_CARDS" ] || [ "$have_torch" != "$NEED_CARDS" ]; then
  echo "[card guard] MISMATCH: the job holds $have_smi card(s) by nvidia-smi and $have_torch by torch, needs $NEED_CARDS."
  if [ "$GUARD_ACTION" = requeue ] && [ "${SLURM_RESTART_COUNT:-0}" -lt "${GUARD_MAX_RESTARTS:-2}" ]; then
    echo "[card guard] requeueing job $SLURM_JOB_ID (scontrol requeue) so it is allocated afresh; never cancel it"
    scontrol requeue "$SLURM_JOB_ID"
    sleep 30
  fi
  echo "[card guard] exiting 3 so the card(s) go back to the pool at once"
  exit 3
fi
echo "[card guard] ok: $have_smi card(s), as needed"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
"$PY" -I "$HERE/import_check_rnd_platform_jax_mujoco_warp.py" card
echo "end $(date -Is)"
