#!/bin/bash
#SBATCH -p cpu
#SBATCH -c 2
#SBATCH --mem=16G
#SBATCH -t 01:00:00
#SBATCH -J rnd-fixed-report
# Regenerate the fixed-settings report (original study plus extension rows) and run the table tests.
set -euo pipefail
P=/p/rlprojects/RND/10_jax_exploration_platform
export PYTHONPATH=$P/src PYTHONNOUSERSITE=1 MPLBACKEND=Agg
cd $P/scripts/grid_initialization_campaign
/p/rlprojects/RND/.venvs/platform_jax/bin/python fixed_settings_report.py --run-dir "$1"
/p/rlprojects/RND/.venvs/atari_pixels_jax/bin/python -m pytest -q $P/tests/test_fixed_settings_report_table.py
