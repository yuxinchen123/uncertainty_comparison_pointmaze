#!/bin/bash
# Final training, H100 NVL serval02 card 1, four packed copies, seeds 1 to 4, 122,070 updates.
# Holds the card-1 lock for the whole run (study exception, plan section 6); yields the card if any
# other process appears on it and resumes from the checkpoint once it is free again.
set -u
unset PYTHONPATH PYTHONHOME
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1
export XLA_FLAGS=--xla_gpu_exclude_nondeterministic_ops=true XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAXTARI_SKIP_SPRITE_UPDATE=1 XDG_DATA_HOME=/p/rlprojects/RND/reference_repos/jaxtari_pixel_study_assets_montezuma_room_geometry_from_png XLA_PYTHON_CLIENT_MEM_FRACTION=0.85
export PYTHONPATH=/p/rlprojects/RND_worktrees/merge-atari-cleanrl/10_jax_exploration_platform/src
cd /p/rlprojects/RND_worktrees/merge-atari-cleanrl/10_jax_exploration_platform
stamp() { TZ=America/Los_Angeles date '+%Y-%m-%d %H:%M:%S PT'; }
export JAX_COMPILATION_CACHE_DIR=/tmp/sl5nw_pixel_final_cache_h100
export CUDA_VISIBLE_DEVICES=GPU-fed69954-5a71-dbb2-ee91-544e0eb6b8a3
U=GPU-fed69954-5a71-dbb2-ee91-544e0eb6b8a3
DATA=/p/rlprojects/RND_worktrees/merge-atari-cleanrl/10_jax_exploration_platform/runs/2026-11-06-01-14_pixel-montezuma_corrected-room-geometry_episode-limit-27000-steps_cleanrl-ppo-rnd-final-configuration_checks-then-steps-per-copy-2000000000_h100-nvl-serval02-card-1-copies-4-seeds-1-to-4_rtx-a4500-jaguar03-8-cards-copies-1-seeds-5-to-12/data/h100-serval02-card-1_final-training_seeds-1-to-4
mkdir -p $DATA "$JAX_COMPILATION_CACHE_DIR"
echo $$ > $DATA/supervisor.pid
card_free() {
  read MIG USED <<<"$(nvidia-smi -i $U --query-gpu=mig.mode.current,memory.used --format=csv,noheader,nounits | tr -d ' ' | tr ',' ' ')"
  PROCS=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep -c "$U")
  CC=$(nvidia-smi conf-compute -f | grep -c 'CC status: OFF')
  [ "$MIG" = Disabled ] && [ "$USED" -lt 1024 ] && [ "$PROCS" -eq 0 ] && [ "$CC" -eq 1 ]
}
others_on_card() {
  # Processes on card 1 other than our trainer ($1).
  nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader | grep "$U" | tr -d ' ' | cut -d, -f2 | grep -v "^$1\$" | wc -l
}
exec 9>/tmp/sl5nw_serval02_card1.lock
flock -w 7200 9 || { echo "$(stamp) card-1 lock not obtained" >> $DATA/unit.log; exit 4; }
nvidia-smi -i $U --query-gpu=name,uuid,driver_version --format=csv,noheader >> $DATA/hardware.txt
failures=0
while ! grep -q '"completed": true' $DATA/run/progress.json 2>/dev/null; do
  until card_free; do echo "$(stamp) card 1 not free; waiting" >> $DATA/unit.log; sleep 300; done
  echo "$(stamp) starting trainer (failures so far $failures)" >> $DATA/unit.log
  /p/rlprojects/RND/.venvs/atari_pixels_jax/bin/python -u scripts/atari_pixel_campaign/train_final.py --run-dir /p/rlprojects/RND_worktrees/merge-atari-cleanrl/10_jax_exploration_platform/runs/2026-11-06-01-14_pixel-montezuma_corrected-room-geometry_episode-limit-27000-steps_cleanrl-ppo-rnd-final-configuration_checks-then-steps-per-copy-2000000000_h100-nvl-serval02-card-1-copies-4-seeds-1-to-4_rtx-a4500-jaguar03-8-cards-copies-1-seeds-5-to-12/data/h100-serval02-card-1_final-training_seeds-1-to-4/run --copies 4 --seed-start 1 --target-updates 122070 --observation-warmup-iterations 50 --checkpoint-seconds 1200 --learner '{"convolution_algorithm": "space_to_depth", "cache_target_features": true, "predictor_selected_rows": 1280, "iteration_compiler_options": [["xla_gpu_enable_while_loop_double_buffering", "true"]]}' --environment '{"separable_rectangle_drawing": true, "unroll_fixed_render_loops": true, "masked_conditional_draws": true, "two_way_masked_draws": true, "render_final_two_together": true, "max_episode_steps": 27000, "reset_lane_budget": 16}' >> $DATA/unit.log 2>&1 &
  trainer=$!
  echo $trainer > $DATA/trainer.pid
  yielded=0
  while kill -0 $trainer 2>/dev/null; do
    sleep 60
    if [ "$(others_on_card $trainer)" -gt 0 ]; then
      echo "$(stamp) another process appeared on card 1; stopping our trainer (it checkpoints)" >> $DATA/unit.log
      nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader | grep "$U" >> $DATA/unit.log
      kill -TERM $trainer; yielded=1; break
    fi
  done
  wait $trainer; rc=$?
  echo "$(stamp) trainer exit $rc" >> $DATA/unit.log
  if [ $yielded -eq 0 ] && [ $rc -ne 0 ]; then
    failures=$((failures + 1))
    [ $failures -ge 5 ] && { echo "$(stamp) five failures; supervisor stops" >> $DATA/unit.log; exit 1; }
    sleep 120
  fi
done
echo "$(stamp) training complete" >> $DATA/unit.log
rm -f $DATA/supervisor.pid $DATA/trainer.pid
