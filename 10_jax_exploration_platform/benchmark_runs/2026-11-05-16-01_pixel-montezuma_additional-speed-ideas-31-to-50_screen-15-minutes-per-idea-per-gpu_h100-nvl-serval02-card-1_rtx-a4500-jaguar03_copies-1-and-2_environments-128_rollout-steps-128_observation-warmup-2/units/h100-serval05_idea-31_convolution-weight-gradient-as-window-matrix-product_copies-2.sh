#!/bin/bash
# Unit: convolution weight gradient as one matrix product of copied image windows, 2 copies, h100-serval05. At most 15 minutes (user rule 2026-11-05).
set -u
unset PYTHONPATH PYTHONHOME
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1
export XLA_FLAGS=--xla_gpu_exclude_nondeterministic_ops=true XLA_PYTHON_CLIENT_PREALLOCATE=false
OTHER=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | xargs -r -n1 ps -o user= -p | grep -vc "^sl5nw$" || true)
if [ "${OTHER:-0}" -gt 0 ] || [ "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)" -gt 0 ]; then echo "serval05 card busy"; exit 3; fi
CACHE=/tmp/sl5nw_pixel_screen_cache_h100
mkdir -p "$CACHE" /p/rlprojects/RND_worktrees/merge-atari-cleanrl/10_jax_exploration_platform/benchmark_runs/2026-11-05-16-01_pixel-montezuma_additional-speed-ideas-31-to-50_screen-15-minutes-per-idea-per-gpu_h100-nvl-serval02-card-1_rtx-a4500-jaguar03_copies-1-and-2_environments-128_rollout-steps-128_observation-warmup-2/data/h100-serval05_idea-31_convolution-weight-gradient-as-window-matrix-product_copies-2
cd /p/rlprojects/RND_worktrees/merge-atari-cleanrl/10_jax_exploration_platform
echo "start $(TZ=America/Los_Angeles date '+%Y-%m-%d %H:%M:%S PT') host $(hostname) pid $$" >> /p/rlprojects/RND_worktrees/merge-atari-cleanrl/10_jax_exploration_platform/benchmark_runs/2026-11-05-16-01_pixel-montezuma_additional-speed-ideas-31-to-50_screen-15-minutes-per-idea-per-gpu_h100-nvl-serval02-card-1_rtx-a4500-jaguar03_copies-1-and-2_environments-128_rollout-steps-128_observation-warmup-2/data/h100-serval05_idea-31_convolution-weight-gradient-as-window-matrix-product_copies-2/unit.log
nvidia-smi --query-gpu=name,uuid,driver_version,memory.used --format=csv,noheader >> /p/rlprojects/RND_worktrees/merge-atari-cleanrl/10_jax_exploration_platform/benchmark_runs/2026-11-05-16-01_pixel-montezuma_additional-speed-ideas-31-to-50_screen-15-minutes-per-idea-per-gpu_h100-nvl-serval02-card-1_rtx-a4500-jaguar03_copies-1-and-2_environments-128_rollout-steps-128_observation-warmup-2/data/h100-serval05_idea-31_convolution-weight-gradient-as-window-matrix-product_copies-2/unit.log
timeout --signal=TERM --kill-after=30 900 /p/rlprojects/RND/.venvs/atari_pixels_jax/bin/python -u scripts/atari_pixel_campaign/screen_idea.py \
  --name 'convolution weight gradient as one matrix product of copied image windows' --learner '{"convolution_algorithm": "patch_weight_gradient"}' --copies 2  \
  --budget-seconds 840 --observation-warmup-iterations 2 \
  --output /p/rlprojects/RND_worktrees/merge-atari-cleanrl/10_jax_exploration_platform/benchmark_runs/2026-11-05-16-01_pixel-montezuma_additional-speed-ideas-31-to-50_screen-15-minutes-per-idea-per-gpu_h100-nvl-serval02-card-1_rtx-a4500-jaguar03_copies-1-and-2_environments-128_rollout-steps-128_observation-warmup-2/data/h100-serval05_idea-31_convolution-weight-gradient-as-window-matrix-product_copies-2 --cache-directory "$CACHE" >> /p/rlprojects/RND_worktrees/merge-atari-cleanrl/10_jax_exploration_platform/benchmark_runs/2026-11-05-16-01_pixel-montezuma_additional-speed-ideas-31-to-50_screen-15-minutes-per-idea-per-gpu_h100-nvl-serval02-card-1_rtx-a4500-jaguar03_copies-1-and-2_environments-128_rollout-steps-128_observation-warmup-2/data/h100-serval05_idea-31_convolution-weight-gradient-as-window-matrix-product_copies-2/unit.log 2>&1
rc=$?
echo "end $(TZ=America/Los_Angeles date '+%Y-%m-%d %H:%M:%S PT') rc $rc" >> /p/rlprojects/RND_worktrees/merge-atari-cleanrl/10_jax_exploration_platform/benchmark_runs/2026-11-05-16-01_pixel-montezuma_additional-speed-ideas-31-to-50_screen-15-minutes-per-idea-per-gpu_h100-nvl-serval02-card-1_rtx-a4500-jaguar03_copies-1-and-2_environments-128_rollout-steps-128_observation-warmup-2/data/h100-serval05_idea-31_convolution-weight-gradient-as-window-matrix-product_copies-2/unit.log
exit $rc
