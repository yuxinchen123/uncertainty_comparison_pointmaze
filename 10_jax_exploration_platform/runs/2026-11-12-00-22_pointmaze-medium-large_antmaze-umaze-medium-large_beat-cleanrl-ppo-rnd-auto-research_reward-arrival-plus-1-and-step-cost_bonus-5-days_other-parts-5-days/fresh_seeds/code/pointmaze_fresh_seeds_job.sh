#!/bin/bash
#SBATCH --job-name=pointmaze-fresh-seeds-analysis
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:30:00
# The PointMaze fresh-seed comparison: its tests, then the page (numpy, so on a compute node).
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1
PY=/p/rlprojects/RND/.venvs/exploration/bin/python
$PY -m pytest -q -p no:cacheprovider test_pointmaze_fresh_seeds.py
$PY pointmaze_fresh_seeds.py
