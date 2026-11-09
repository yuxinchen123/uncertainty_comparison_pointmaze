#!/bin/bash
#SBATCH --job-name=grid-fixed-release-held-units-to-fast-cards
#SBATCH --partition=nolim
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=4-00:00:00
#SBATCH --output=/p/rlprojects/RND/10_jax_exploration_platform/runs/2026-11-07-20-03_fixed-cleanrl-ppo-rnd_straight-32-snake-16x2_tabular-neural_uniform-alternating-frozen-network-online-rnd_state-and-reward-normalization_learning-rate-1e-4-1e-3_entropy-1e-3-1e-2_steps-16777216_seeds-0-15_confirmation-20000-20127/logs/grid-fixed-release-held-units_%j.log
exec /usr/bin/python3 /p/rlprojects/RND/10_jax_exploration_platform/runs/2026-11-07-20-03_fixed-cleanrl-ppo-rnd_straight-32-snake-16x2_tabular-neural_uniform-alternating-frozen-network-online-rnd_state-and-reward-normalization_learning-rate-1e-4-1e-3_entropy-1e-3-1e-2_steps-16777216_seeds-0-15_confirmation-20000-20127/infrastructure/release_to_fast_cards.py /p/rlprojects/RND/10_jax_exploration_platform/runs/2026-11-07-20-03_fixed-cleanrl-ppo-rnd_straight-32-snake-16x2_tabular-neural_uniform-alternating-frozen-network-online-rnd_state-and-reward-normalization_learning-rate-1e-4-1e-3_entropy-1e-3-1e-2_steps-16777216_seeds-0-15_confirmation-20000-20127
