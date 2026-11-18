#!/bin/bash
set -u
unset PYTHONPATH PYTHONHOME
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.85
export XLA_FLAGS=--xla_gpu_exclude_nondeterministic_ops=true
export XDG_DATA_HOME=/p/rlprojects/RND/reference_repos/jaxtari_pixel_study_assets_montezuma_room_geometry_from_png
export JAXTARI_SKIP_SPRITE_UPDATE=1
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
PY=/p/rlprojects/RND/.venvs/atari_pixels_jax/bin/python
G=/p/rlprojects/RND/10_jax_exploration_platform/runs/2026-11-17-21-26_pixel-montezuma_fast-last-layer-against-rnd_gates_unit-tests_archive-identity_resume_speed-ideas-10_card-speed-checks
cd /p/rlprojects/RND/10_jax_exploration_platform
export PYTHONPATH=/p/rlprojects/RND/10_jax_exploration_platform/src
stamp() { TZ=America/Los_Angeles date '+%Y-%m-%d %H:%M:%S PT'; }
NAME=task-1-part-4_h100_copies-4_default-and-game-events_same-process
OUT="$G/data/speed-profiles/$NAME"
mkdir -p "$OUT"
echo "start $(stamp) host $(hostname) card $CUDA_VISIBLE_DEVICES unit $NAME"
nvidia-smi --query-gpu=index,uuid,name,memory.used --format=csv,noheader | tee "$OUT/hardware.txt"
timeout --signal=TERM 600 $PY -u scripts/atari_pixel_campaign/profile_fast_last_layer.py --output-dir "$OUT" --compilation-cache /localtmp/sl5nw/jax_campaign_caches/pixel_montezuma_fast_last_layer --card h100 --copies 4 --seeds 1\,2\,3\,4 --variant default --variant game-events --timed-updates 8 --budget-seconds 560
rc=$?
echo "end $(stamp) exit $rc"
exit $rc
