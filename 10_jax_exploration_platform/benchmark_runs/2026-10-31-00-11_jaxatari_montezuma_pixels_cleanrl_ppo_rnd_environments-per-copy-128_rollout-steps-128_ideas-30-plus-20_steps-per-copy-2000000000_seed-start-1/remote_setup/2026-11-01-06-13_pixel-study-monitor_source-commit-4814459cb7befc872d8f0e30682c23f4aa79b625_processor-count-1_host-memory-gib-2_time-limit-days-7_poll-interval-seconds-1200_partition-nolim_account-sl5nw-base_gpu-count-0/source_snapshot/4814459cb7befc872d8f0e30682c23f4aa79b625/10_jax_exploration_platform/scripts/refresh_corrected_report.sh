#!/usr/bin/env bash
set -euo pipefail
export PYTHONNOUSERSITE=1
python_path=/p/rlprojects/RND/.venvs/platform_jax/bin/python
"$python_path" /p/rlprojects/RND/10_jax_exploration_platform/development_document/code/2026-10-13-23-52_corrected-pointmaze-100-million-step-tables/make_all.py
"$python_path" /p/rlprojects/RND/10_jax_exploration_platform/tests/integration/test_corrected_report_generator.py
