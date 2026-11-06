#!/bin/bash
# Unit: learning check: original reference, one copy, seed 1, up to 300 updates. At most 15 minutes (user rule 2026-11-05).
set -u
unset PYTHONPATH PYTHONHOME
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1
export XLA_FLAGS=--xla_gpu_exclude_nondeterministic_ops=true XLA_PYTHON_CLIENT_PREALLOCATE=false JAXTARI_SKIP_SPRITE_UPDATE=1 XLA_PYTHON_CLIENT_MEM_FRACTION=0.85
CACHE=/tmp/sl5nw_pixel_screen_cache_a4500
export JAX_COMPILATION_CACHE_DIR="$CACHE"
PY=/p/rlprojects/RND/.venvs/atari_pixels_jax/bin/python
TRAIN="$PY -u /p/rlprojects/RND_worktrees/merge-atari-cleanrl/10_jax_exploration_platform/scripts/atari_pixel_campaign/train_final.py"
COMPARE="$PY /p/rlprojects/RND_worktrees/merge-atari-cleanrl/10_jax_exploration_platform/scripts/atari_pixel_campaign/compare_resumed.py"
DATA=/p/rlprojects/RND_worktrees/merge-atari-cleanrl/10_jax_exploration_platform/benchmark_runs/2026-11-05-16-01_pixel-montezuma_additional-speed-ideas-31-to-50_screen-15-minutes-per-idea-per-gpu_h100-nvl-serval02-card-1_rtx-a4500-jaguar03_copies-1-and-2_environments-128_rollout-steps-128_observation-warmup-2/data/a4500-jaguar03_learning-check_original-reference_seed-1
mkdir -p "$CACHE" $DATA
cd /p/rlprojects/RND_worktrees/merge-atari-cleanrl/10_jax_exploration_platform
export PYTHONPATH=/p/rlprojects/RND_worktrees/merge-atari-cleanrl/10_jax_exploration_platform/src
echo "start $(TZ=America/Los_Angeles date '+%Y-%m-%d %H:%M:%S PT') host $(hostname) pid $$" >> $DATA/unit.log
(
timeout 840 $TRAIN --run-dir $DATA/run --copies 1 --seed-start 1 --target-updates 300 --time-limit-seconds 760

) >> $DATA/unit.log 2>&1
rc=$?
echo "end $(TZ=America/Los_Angeles date '+%Y-%m-%d %H:%M:%S PT') rc $rc" >> $DATA/unit.log
exit $rc
