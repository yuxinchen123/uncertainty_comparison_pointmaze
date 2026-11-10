#!/bin/bash
# serval05 unit (under the queue's lock): total throughput against copies per process, for the baseline's kind of unit
# and the finalist's, to choose how many copies one H100 or H200 process should hold.
set -euo pipefail
FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
if [ "$FREE" -lt 60000 ]; then echo "card busy: $FREE MiB free"; nvidia-smi --query-compute-apps=pid,used_memory --format=csv; exit 75; fi
P=/p/rlprojects/RND/10_jax_exploration_platform
export PYTHONPATH=$P/src PYTHONNOUSERSITE=1 XLA_PYTHON_CLIENT_PREALLOCATE=false
date; echo 'machine: serval05'
timeout 1700 /p/rlprojects/RND/.venvs/platform_jax/bin/python -u $P/scripts/grid_initialization_campaign/profile_online_tabular.py \
  --ideas /p/rlprojects/RND/10_jax_exploration_platform/runs/2026-11-09-19-42_online-tabular-rnd-auto-research_reward-normalization-on_even-start-and-decay-exponent-near-minus-0.5_straight-32-snake-16x2_tabular-ppo-coordinate-ppo_five-days/batches/06_finalists-development-on-the-full-protocol/profile_ideas.json --output /p/rlprojects/RND/10_jax_exploration_platform/runs/2026-11-09-19-42_online-tabular-rnd-auto-research_reward-normalization-on_even-start-and-decay-exponent-near-minus-0.5_straight-32-snake-16x2_tabular-ppo-coordinate-ppo_five-days/infrastructure/profiles/serval05_h100_nvl_snake_coordinate-ppo_copies_per_card.json \
  --copies 64 128 256 512 1024 2048 4096 --timed 20 > /p/rlprojects/RND/10_jax_exploration_platform/runs/2026-11-09-19-42_online-tabular-rnd-auto-research_reward-normalization-on_even-start-and-decay-exponent-near-minus-0.5_straight-32-snake-16x2_tabular-ppo-coordinate-ppo_five-days/logs/profile_serval05_copies_per_card.log 2>&1
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader >> /p/rlprojects/RND/10_jax_exploration_platform/runs/2026-11-09-19-42_online-tabular-rnd-auto-research_reward-normalization-on_even-start-and-decay-exponent-near-minus-0.5_straight-32-snake-16x2_tabular-ppo-coordinate-ppo_five-days/logs/profile_serval05_copies_per_card.log
