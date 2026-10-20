#!/bin/bash
# A second attempt of idea A3 on a card with a one-day limit: the first attempt's
# 3.4-hour need sits within 2% of its 3.5-hour reservation-bound limit.
set -euo pipefail
export PYTHONNOUSERSITE=1 XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.75 PYTHONUNBUFFERED=1
export LD_LIBRARY_PATH=/usr/lib/nvidia:${LD_LIBRARY_PATH:-}
export JAX_COMPILATION_CACHE_DIR=/p/rlprojects/RND/10_jax_exploration_platform/.jax_compilation_cache
/p/rlprojects/RND/.venvs/platform_jax/bin/python -u /p/rlprojects/RND/10_jax_exploration_platform/scripts/run_training.py --run-dir /p/rlprojects/RND/10_jax_exploration_platform/runs/2026-10-19-16-46_reward-decline-batch-1_large-cap400_rnd-weights-1-10-100-1000_learning-rate-1e-3-annealed-over-1000M_steps-50M_seeds-0-127_control-e1-e2-e3-a1-a2-a3-a4-a5-a6-a7 --unit-id a3_clipped-ppo_epochs-4_minibatches-4_second-attempt --bonus rnd_next_state --parameter-settings /p/rlprojects/RND/10_jax_exploration_platform/runs/2026-10-19-16-46_reward-decline-batch-1_large-cap400_rnd-weights-1-10-100-1000_learning-rate-1e-3-annealed-over-1000M_steps-50M_seeds-0-127_control-e1-e2-e3-a1-a2-a3-a4-a5-a6-a7/code/parameters/rnd_weights-1-10-100-1000.json --copies-per-cell 128 --iterations 97657 --window-iterations 250 --track-coverage --diagnostics --no-aggregate --map large --start-cell 7,1 --goal-cell 1,10 --episode-steps 400 --lr-anneal-iterations 1953125 --description 'reward-decline idea A3: a3_clipped-ppo_epochs-4_minibatches-4' --update-style epoch_minibatch
