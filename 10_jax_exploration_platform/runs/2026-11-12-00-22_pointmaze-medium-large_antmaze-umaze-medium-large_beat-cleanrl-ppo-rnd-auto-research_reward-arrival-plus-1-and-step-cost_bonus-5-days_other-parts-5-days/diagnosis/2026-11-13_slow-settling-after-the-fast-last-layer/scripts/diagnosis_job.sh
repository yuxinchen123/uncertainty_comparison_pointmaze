#!/bin/bash
# One diagnosis training run on one card. Arguments: <run name> <configuration file> <bonus options or "none">
# [further run_training.py options]. Records go to runs/data/<run name>.jsonl beside this folder's notes.
set -uo pipefail
umask 0002
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 XLA_PYTHON_CLIENT_PREALLOCATE=false
export LD_LIBRARY_PATH=/usr/lib/nvidia:${LD_LIBRARY_PATH:-}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PLATFORM=/p/rlprojects/RND/10_jax_exploration_platform
HERE="$DIAGNOSIS_FOLDER"
name="$1"; configuration="$2"; options="$3"; shift 3
extra=()
[ "$options" = none ] || extra+=(--bonus-options "$options")
cd "$PLATFORM"
echo "[$(TZ=America/Los_Angeles date '+%Y-%m-%d %H:%M PT')] $name on $(hostname), $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1), commit $(git rev-parse --short=8 HEAD)"
/p/rlprojects/RND/.venvs/platform_jax/bin/python scripts/run_training.py --run-dir "$HERE/runs" --unit-id "$name" \
  --configuration-file "$configuration" --reward-setting arrival_plus_one --description "diagnosis: $name" \
  "${extra[@]}" --no-aggregate "$@"
