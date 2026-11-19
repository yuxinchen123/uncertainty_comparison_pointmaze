#!/bin/bash
# Pack-size profile of the pixel Montezuma stage-1 grid on one MIT H200 (pack_size_profile.py beside this file).
# Filled by write_job_script.py: change the template, not the filled copy. One card per job; four settings of at
# most 15 minutes each, one after another on the same card.
RUN=@REMOTE_RUN@
PY=/home/shuzel/envs/atari_pixels_jax/bin/python
CODE=@CODE@
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
@CARD_GUARD@
echo "job $JOB on $(hostname), start $(date -Is)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
mkdir -p "$OUT"
"$PY" "$RUN/mit_pack_size_profile/pack_size_profile.py" --code "$CODE" --out "$OUT" --updates 60
echo "job $JOB ends $(date -Is)"
