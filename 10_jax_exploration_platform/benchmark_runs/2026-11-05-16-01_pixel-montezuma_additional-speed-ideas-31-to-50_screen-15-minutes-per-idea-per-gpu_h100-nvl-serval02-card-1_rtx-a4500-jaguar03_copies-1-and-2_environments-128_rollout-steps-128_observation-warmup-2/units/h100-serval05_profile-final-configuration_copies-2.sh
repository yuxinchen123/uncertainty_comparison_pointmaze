#!/bin/bash
# Unit: profile of the final H100 configuration, 2 copies, h100-serval05. At most 15 minutes (user rule 2026-11-05).
set -u
unset PYTHONPATH PYTHONHOME
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1
export XLA_FLAGS=--xla_gpu_exclude_nondeterministic_ops=true XLA_PYTHON_CLIENT_PREALLOCATE=false JAXTARI_SKIP_SPRITE_UPDATE=1
if [ "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)" -gt 0 ]; then echo "serval05 card busy"; exit 3; fi
CACHE=/tmp/sl5nw_pixel_screen_cache_h100
mkdir -p "$CACHE" /p/rlprojects/RND_worktrees/merge-atari-cleanrl/10_jax_exploration_platform/benchmark_runs/2026-11-05-16-01_pixel-montezuma_additional-speed-ideas-31-to-50_screen-15-minutes-per-idea-per-gpu_h100-nvl-serval02-card-1_rtx-a4500-jaguar03_copies-1-and-2_environments-128_rollout-steps-128_observation-warmup-2/data/h100-serval05_profile-final-configuration_copies-2
cd /p/rlprojects/RND_worktrees/merge-atari-cleanrl/10_jax_exploration_platform
echo "start $(TZ=America/Los_Angeles date '+%Y-%m-%d %H:%M:%S PT') host $(hostname) pid $$" >> /p/rlprojects/RND_worktrees/merge-atari-cleanrl/10_jax_exploration_platform/benchmark_runs/2026-11-05-16-01_pixel-montezuma_additional-speed-ideas-31-to-50_screen-15-minutes-per-idea-per-gpu_h100-nvl-serval02-card-1_rtx-a4500-jaguar03_copies-1-and-2_environments-128_rollout-steps-128_observation-warmup-2/data/h100-serval05_profile-final-configuration_copies-2/unit.log
timeout --signal=TERM --kill-after=30 900 /p/rlprojects/RND/.venvs/atari_pixels_jax/bin/python -u scripts/atari_pixel_campaign/screen_idea.py \
  --name 'profile of the final H100 configuration' --learner '{}' --environment '{}' --reference-learner '{"convolution_algorithm": "space_to_depth", "cache_target_features": true, "predictor_selected_rows": 1280, "iteration_compiler_options": [["xla_gpu_enable_while_loop_double_buffering", "true"]]}' --reference-environment '{"separable_rectangle_drawing": true, "unroll_fixed_render_loops": true, "masked_conditional_draws": true, "two_way_masked_draws": true, "render_final_two_together": true}' --copies 2 --profile \
  --budget-seconds 840 --observation-warmup-iterations 2 \
  --output /p/rlprojects/RND_worktrees/merge-atari-cleanrl/10_jax_exploration_platform/benchmark_runs/2026-11-05-16-01_pixel-montezuma_additional-speed-ideas-31-to-50_screen-15-minutes-per-idea-per-gpu_h100-nvl-serval02-card-1_rtx-a4500-jaguar03_copies-1-and-2_environments-128_rollout-steps-128_observation-warmup-2/data/h100-serval05_profile-final-configuration_copies-2 --cache-directory "$CACHE" >> /p/rlprojects/RND_worktrees/merge-atari-cleanrl/10_jax_exploration_platform/benchmark_runs/2026-11-05-16-01_pixel-montezuma_additional-speed-ideas-31-to-50_screen-15-minutes-per-idea-per-gpu_h100-nvl-serval02-card-1_rtx-a4500-jaguar03_copies-1-and-2_environments-128_rollout-steps-128_observation-warmup-2/data/h100-serval05_profile-final-configuration_copies-2/unit.log 2>&1
rc=$?
echo "end $(TZ=America/Los_Angeles date '+%Y-%m-%d %H:%M:%S PT') rc $rc" >> /p/rlprojects/RND_worktrees/merge-atari-cleanrl/10_jax_exploration_platform/benchmark_runs/2026-11-05-16-01_pixel-montezuma_additional-speed-ideas-31-to-50_screen-15-minutes-per-idea-per-gpu_h100-nvl-serval02-card-1_rtx-a4500-jaguar03_copies-1-and-2_environments-128_rollout-steps-128_observation-warmup-2/data/h100-serval05_profile-final-configuration_copies-2/unit.log
exit $rc
