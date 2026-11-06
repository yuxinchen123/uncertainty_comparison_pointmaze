#!/bin/bash
# Per-layer convolution cost probe on the H100 (vmap batching). At most 15 minutes.
set -u
unset PYTHONPATH PYTHONHOME
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1
export XLA_FLAGS=--xla_gpu_exclude_nondeterministic_ops=true XLA_PYTHON_CLIENT_PREALLOCATE=false JAXTARI_SKIP_SPRITE_UPDATE=1
U=GPU-fed69954-5a71-dbb2-ee91-544e0eb6b8a3
read MIG USED <<<"$(nvidia-smi -i $U --query-gpu=mig.mode.current,memory.used --format=csv,noheader,nounits | tr -d " " | tr "," " ")"
PROCS=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep -c "$U")
CC=$(nvidia-smi conf-compute -f | grep -c "CC status: OFF")
if ! { [ "$MIG" = Disabled ] && [ "$USED" -lt 1024 ] && [ "$PROCS" -eq 0 ] && [ "$CC" -eq 1 ]; }; then echo "card 1 busy"; exit 3; fi
export CUDA_VISIBLE_DEVICES=$U
cd /p/rlprojects/RND_worktrees/merge-atari-cleanrl/10_jax_exploration_platform
PYTHONPATH=src timeout 840 /p/rlprojects/RND/.venvs/atari_pixels_jax/bin/python /p/rlprojects/RND_worktrees/merge-atari-cleanrl/10_jax_exploration_platform/benchmark_runs/2026-11-05-16-01_pixel-montezuma_additional-speed-ideas-31-to-50_screen-15-minutes-per-idea-per-gpu_h100-nvl-serval02-card-1_rtx-a4500-jaguar03_copies-1-and-2_environments-128_rollout-steps-128_observation-warmup-2/probes/per_layer_convolution_cost.py /p/rlprojects/RND_worktrees/merge-atari-cleanrl/10_jax_exploration_platform/benchmark_runs/2026-11-05-16-01_pixel-montezuma_additional-speed-ideas-31-to-50_screen-15-minutes-per-idea-per-gpu_h100-nvl-serval02-card-1_rtx-a4500-jaguar03_copies-1-and-2_environments-128_rollout-steps-128_observation-warmup-2/data/probe_per_layer_convolution_cost/h100-serval02-card-1.jsonl vmap > /p/rlprojects/RND_worktrees/merge-atari-cleanrl/10_jax_exploration_platform/benchmark_runs/2026-11-05-16-01_pixel-montezuma_additional-speed-ideas-31-to-50_screen-15-minutes-per-idea-per-gpu_h100-nvl-serval02-card-1_rtx-a4500-jaguar03_copies-1-and-2_environments-128_rollout-steps-128_observation-warmup-2/data/probe_per_layer_convolution_cost/h100-serval02-card-1.log 2>&1
