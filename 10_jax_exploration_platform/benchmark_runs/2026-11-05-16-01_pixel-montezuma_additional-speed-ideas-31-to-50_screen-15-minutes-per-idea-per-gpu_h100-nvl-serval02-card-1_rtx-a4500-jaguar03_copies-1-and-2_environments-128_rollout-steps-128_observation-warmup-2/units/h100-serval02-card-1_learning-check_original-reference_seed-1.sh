#!/bin/bash
# Unit: learning check: original reference on the H100, one copy, seed 1, as many updates as fit. At most 15 minutes (user rule 2026-11-05).
set -u
unset PYTHONPATH PYTHONHOME
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1
export XLA_FLAGS=--xla_gpu_exclude_nondeterministic_ops=true XLA_PYTHON_CLIENT_PREALLOCATE=false JAXTARI_SKIP_SPRITE_UPDATE=1 XLA_PYTHON_CLIENT_MEM_FRACTION=0.85
U=GPU-fed69954-5a71-dbb2-ee91-544e0eb6b8a3
read MIG USED <<<"$(nvidia-smi -i $U --query-gpu=mig.mode.current,memory.used --format=csv,noheader,nounits | tr -d " " | tr "," " ")"
PROCS=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep -c "$U")
CC=$(nvidia-smi conf-compute -f | grep -c "CC status: OFF")
if ! { [ "$MIG" = Disabled ] && [ "$USED" -lt 1024 ] && [ "$PROCS" -eq 0 ] && [ "$CC" -eq 1 ]; }; then echo "card 1 busy: mig=$MIG used=$USED procs=$PROCS cc=$CC"; exit 3; fi
export CUDA_VISIBLE_DEVICES=$U
CACHE=/tmp/sl5nw_pixel_screen_cache_h100
export JAX_COMPILATION_CACHE_DIR="$CACHE"
PY=/p/rlprojects/RND/.venvs/atari_pixels_jax/bin/python
TRAIN="$PY -u /p/rlprojects/RND_worktrees/merge-atari-cleanrl/10_jax_exploration_platform/scripts/atari_pixel_campaign/train_final.py"
COMPARE="$PY /p/rlprojects/RND_worktrees/merge-atari-cleanrl/10_jax_exploration_platform/scripts/atari_pixel_campaign/compare_resumed.py"
DATA=/p/rlprojects/RND_worktrees/merge-atari-cleanrl/10_jax_exploration_platform/benchmark_runs/2026-11-05-16-01_pixel-montezuma_additional-speed-ideas-31-to-50_screen-15-minutes-per-idea-per-gpu_h100-nvl-serval02-card-1_rtx-a4500-jaguar03_copies-1-and-2_environments-128_rollout-steps-128_observation-warmup-2/data/h100-serval02-card-1_learning-check_original-reference_seed-1
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
