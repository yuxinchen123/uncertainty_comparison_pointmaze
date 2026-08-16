#!/bin/bash
# The only way the queue is advanced. Holds an exclusive lock for the whole submission pass, so
# the 20-minute monitor and a hand-run pass cannot both decide a cell is free and submit it twice
# — two jobs of this survey on one machine spoil each other's timing, which cost one jaguar03
# measurement before the lock existed.
set -euo pipefail
RUN="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONNOUSERSITE=1
exec flock "$RUN/slurm/.advance.lock" \
    /p/rlprojects/RND/.venvs/jax_gpu/bin/python "$RUN/code/advance_queue.py" "$@"
