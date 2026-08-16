#!/bin/bash
# The follow-up probe: would letting the compiler generate the matrix multiplications, with the
# bias and the activation folded into them, close the distance to the JAX trainer? Run from
# serval05 itself for the same reason as the main batch.
set -uo pipefail
W=/p/rlprojects/RND/.claude/worktrees/agent-ab3b4d042ce36522c/09_parallelization
L=$W/locks/gpu_run.sh
LOGS=$W/benchmarks/round5_logs
mkdir -p "$LOGS"
PYT="PYTHONNOUSERSITE=1 /localtmp/sl5nw/venvs/rnd09_torch/bin/python"

echo "=== $(date -Is) START probe_autotune" >> "$LOGS/driver.log"
bash "$L" "cd $W/benchmarks && $PYT probe_autotune.py --n-copies 1024 --style epoch_minibatch --rounds 3" \
  > "$LOGS/probe_autotune.log" 2>&1
echo "=== $(date -Is) END   probe_autotune rc=$?" >> "$LOGS/driver.log"
