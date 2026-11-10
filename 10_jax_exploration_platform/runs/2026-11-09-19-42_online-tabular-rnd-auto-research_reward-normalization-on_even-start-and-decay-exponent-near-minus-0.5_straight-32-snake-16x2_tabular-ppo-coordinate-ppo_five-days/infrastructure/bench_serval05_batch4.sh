#!/bin/bash
# serval05 unit (under the queue's lock): four-law bench of batch 4's rollout-fit ideas.
set -euo pipefail
FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
if [ "$FREE" -lt 60000 ]; then echo "card busy: $FREE MiB free"; nvidia-smi --query-compute-apps=pid,used_memory --format=csv; exit 75; fi
P=/p/rlprojects/RND/10_jax_exploration_platform
export PYTHONPATH=$P/src PYTHONNOUSERSITE=1 XLA_PYTHON_CLIENT_PREALLOCATE=false
date; echo 'machine: serval05'
timeout 1750 /p/rlprojects/RND/.venvs/platform_jax/bin/python -u $P/scripts/grid_initialization_campaign/rnd_decay_bench.py \
  --ideas /p/rlprojects/RND/10_jax_exploration_platform/runs/2026-11-09-19-42_online-tabular-rnd-auto-research_reward-normalization-on_even-start-and-decay-exponent-near-minus-0.5_straight-32-snake-16x2_tabular-ppo-coordinate-ppo_five-days/batches/04_rnd-error-follows-a-set-bonus-schedule-by-one-fit-per-rollout/bench_ideas.json --output /p/rlprojects/RND/10_jax_exploration_platform/runs/2026-11-09-19-42_online-tabular-rnd-auto-research_reward-normalization-on_even-start-and-decay-exponent-near-minus-0.5_straight-32-snake-16x2_tabular-ppo-coordinate-ppo_five-days/bench/batch_04_four_laws > /p/rlprojects/RND/10_jax_exploration_platform/runs/2026-11-09-19-42_online-tabular-rnd-auto-research_reward-normalization-on_even-start-and-decay-exponent-near-minus-0.5_straight-32-snake-16x2_tabular-ppo-coordinate-ppo_five-days/logs/bench_serval05_batch4.log 2>&1
