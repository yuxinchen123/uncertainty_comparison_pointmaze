#!/bin/bash
# Pack-size profile of the pixel Montezuma stage-1 grid on one MIT H200 (pack_size_profile.py beside this file).
# Filled by write_job_script.py: change the template, not the filled copy. One card per job; four settings of at
# most 15 minutes each, one after another on the same card.
RUN=/home/shuzel/rnd_pixel/runs/2026-11-18-12-00_pixel-montezuma_fast-last-layer-against-rnd_stage-1-grid_step-sizes-0.01-0.03-0.1-0.3_intrinsic-weights-0.25-0.5-1-2-4-8_sharing-groups-128-16-4_rnd-same-weights_seeds-13-to-20_steps-per-policy-200000000_schedule-horizon-2000000000
PY=/home/shuzel/envs/atari_pixels_jax/bin/python
CODE=/home/shuzel/rnd_pixel/code/source_08b7c4c89c8d885704bb559f5b623ca58e36ec2a/10_jax_exploration_platform
OUT=$RUN/mit_pack_size_profile/job-$SLURM_JOB_ID
unset PYTHONPATH PYTHONHOME
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.85
export XLA_FLAGS=--xla_gpu_exclude_nondeterministic_ops=true
export XDG_DATA_HOME=/home/shuzel/rnd_pixel/assets/jaxtari_pixel_study_assets_montezuma_room_geometry_from_png
export JAXTARI_SKIP_SPRITE_UPDATE=1
unset LD_LIBRARY_PATH
JOB=${SLURM_JOB_ID:?this script runs only as a Slurm job}
cd "$RUN" || exit 2
NEED_CARDS=1
RUN_DIR="$RUN"
GUARD_ACTION=${GUARD_ACTION:-requeue}
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

echo "job $JOB on $(hostname), start $(date -Is)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
mkdir -p "$OUT"
"$PY" "$RUN/mit_pack_size_profile/pack_size_profile.py" --code "$CODE" --out "$OUT" --updates 60
echo "job $JOB ends $(date -Is)"
