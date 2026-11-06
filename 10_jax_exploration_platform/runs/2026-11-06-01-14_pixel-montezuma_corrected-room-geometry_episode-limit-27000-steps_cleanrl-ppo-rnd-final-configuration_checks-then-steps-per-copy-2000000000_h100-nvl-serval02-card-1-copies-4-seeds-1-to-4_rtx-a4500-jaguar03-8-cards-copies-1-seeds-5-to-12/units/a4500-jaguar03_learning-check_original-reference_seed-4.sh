#!/bin/bash
# Unit: learning check on the corrected game, original reference, seed 4, up to 300 updates. At most 15 minutes (user rule 2026-11-05).
set -u
unset PYTHONPATH PYTHONHOME
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1
export XLA_FLAGS=--xla_gpu_exclude_nondeterministic_ops=true XLA_PYTHON_CLIENT_PREALLOCATE=false JAXTARI_SKIP_SPRITE_UPDATE=1 XDG_DATA_HOME=/p/rlprojects/RND/reference_repos/jaxtari_pixel_study_assets_montezuma_room_geometry_from_png XLA_PYTHON_CLIENT_MEM_FRACTION=0.85
CACHE=/tmp/sl5nw_pixel_screen_cache_a4500
export JAX_COMPILATION_CACHE_DIR="$CACHE"
PY=/p/rlprojects/RND/.venvs/atari_pixels_jax/bin/python
TRAIN="$PY -u /p/rlprojects/RND_worktrees/merge-atari-cleanrl/10_jax_exploration_platform/scripts/atari_pixel_campaign/train_final.py"
COMPARE="$PY /p/rlprojects/RND_worktrees/merge-atari-cleanrl/10_jax_exploration_platform/scripts/atari_pixel_campaign/compare_resumed.py"
DATA=/p/rlprojects/RND_worktrees/merge-atari-cleanrl/10_jax_exploration_platform/runs/2026-11-06-01-14_pixel-montezuma_corrected-room-geometry_episode-limit-27000-steps_cleanrl-ppo-rnd-final-configuration_checks-then-steps-per-copy-2000000000_h100-nvl-serval02-card-1-copies-4-seeds-1-to-4_rtx-a4500-jaguar03-8-cards-copies-1-seeds-5-to-12/data/a4500-jaguar03_learning-check_original-reference_seed-4
mkdir -p "$CACHE" $DATA
cd /p/rlprojects/RND_worktrees/merge-atari-cleanrl/10_jax_exploration_platform
export PYTHONPATH=/p/rlprojects/RND_worktrees/merge-atari-cleanrl/10_jax_exploration_platform/src
echo "start $(TZ=America/Los_Angeles date '+%Y-%m-%d %H:%M:%S PT') host $(hostname) pid $$" >> $DATA/unit.log
(
timeout 840 $TRAIN --run-dir $DATA/run --copies 1 --seed-start 4 --target-updates 300 --time-limit-seconds 760

) >> $DATA/unit.log 2>&1
rc=$?
echo "end $(TZ=America/Los_Angeles date '+%Y-%m-%d %H:%M:%S PT') rc $rc" >> $DATA/unit.log
exit $rc
