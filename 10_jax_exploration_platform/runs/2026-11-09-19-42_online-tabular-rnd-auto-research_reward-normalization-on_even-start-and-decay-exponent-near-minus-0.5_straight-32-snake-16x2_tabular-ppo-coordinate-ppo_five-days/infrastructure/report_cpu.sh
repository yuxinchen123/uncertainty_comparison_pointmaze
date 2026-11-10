#!/bin/bash
#SBATCH -p cpu
#SBATCH -c 2
#SBATCH --mem=16G
#SBATCH -t 01:00:00
#SBATCH -J rnd-auto-research-report
# Generate the auto research's analysis.md from the finished units of the run folder given as $1.
set -euo pipefail
P=/p/rlprojects/RND/10_jax_exploration_platform
export PYTHONPATH=$P/src PYTHONNOUSERSITE=1 MPLBACKEND=Agg
cd $P/scripts/grid_initialization_campaign
/p/rlprojects/RND/.venvs/platform_jax/bin/python auto_research_report.py --run-dir "$1"
