#!/bin/bash
# Unit: learning check on the corrected game, final H100 configuration, seeds 1 to 4 packed, up to 300 updates. At most 15 minutes (user rule 2026-11-05).
set -u
unset PYTHONPATH PYTHONHOME
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1
export XLA_FLAGS=--xla_gpu_exclude_nondeterministic_ops=true XLA_PYTHON_CLIENT_PREALLOCATE=false JAXTARI_SKIP_SPRITE_UPDATE=1 XDG_DATA_HOME=/p/rlprojects/RND/reference_repos/jaxtari_pixel_study_assets_montezuma_room_geometry_from_png XLA_PYTHON_CLIENT_MEM_FRACTION=0.85
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
DATA=/p/rlprojects/RND_worktrees/merge-atari-cleanrl/10_jax_exploration_platform/runs/2026-11-06-01-14_pixel-montezuma_corrected-room-geometry_episode-limit-27000-steps_cleanrl-ppo-rnd-final-configuration_checks-then-steps-per-copy-2000000000_h100-nvl-serval02-card-1-copies-4-seeds-1-to-4_rtx-a4500-jaguar03-8-cards-copies-1-seeds-5-to-12/data/h100-serval02-card-1_learning-check_final-configuration_seeds-1-to-4
mkdir -p "$CACHE" $DATA
cd /p/rlprojects/RND_worktrees/merge-atari-cleanrl/10_jax_exploration_platform
export PYTHONPATH=/p/rlprojects/RND_worktrees/merge-atari-cleanrl/10_jax_exploration_platform/src
echo "start $(TZ=America/Los_Angeles date '+%Y-%m-%d %H:%M:%S PT') host $(hostname) pid $$" >> $DATA/unit.log
(
timeout 840 $TRAIN --run-dir $DATA/run --copies 4 --seed-start 1 --target-updates 300 --time-limit-seconds 760 --learner '{"convolution_algorithm": "space_to_depth", "cache_target_features": true, "predictor_selected_rows": 1280, "iteration_compiler_options": [["xla_gpu_enable_while_loop_double_buffering", "true"]]}' --environment '{"separable_rectangle_drawing": true, "unroll_fixed_render_loops": true, "masked_conditional_draws": true, "two_way_masked_draws": true, "render_final_two_together": true}'

) >> $DATA/unit.log 2>&1
rc=$?
echo "end $(TZ=America/Los_Angeles date '+%Y-%m-%d %H:%M:%S PT') rc $rc" >> $DATA/unit.log
exit $rc
