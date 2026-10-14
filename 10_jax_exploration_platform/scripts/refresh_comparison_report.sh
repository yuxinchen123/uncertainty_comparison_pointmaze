#!/usr/bin/env bash
set -euo pipefail
export PYTHONNOUSERSITE=1
python_path=/p/rlprojects/RND/.venvs/platform_jax/bin/python
"$python_path" /p/rlprojects/RND/10_jax_exploration_platform/scripts/extract_historical_comparison_curves.py --run-dir /p/rlprojects/RND/10_jax_exploration_platform/runs/2026-10-13-22-13_pointmaze_full-batch_steps-100-million_seeds-0-127_rollout-128_environments-4_random-network-distillation_weights-1-10-100-1000_visit-count-square-root_weights-1-10_visit-count-inverse_weight-1_no-bonus_learning-rates-1e-3-1e-4
"$python_path" /p/rlprojects/RND/10_jax_exploration_platform/development_document/code/2026-10-13-23-52_corrected-pointmaze-100-million-step-tables/make_all.py
"$python_path" /p/rlprojects/RND/10_jax_exploration_platform/tests/integration/test_corrected_report_generator.py
