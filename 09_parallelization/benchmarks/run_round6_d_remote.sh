#!/bin/bash
# Round six, batch D: the two gates that had to be re-run after the shipped configuration was
# settled, and nothing else. Queues behind whatever else holds the lock.
#
#   ssh serval05 "setsid nohup bash <this file> > <logs>/remote_driver_d.log 2>&1 < /dev/null &"
set -uo pipefail
W=/p/rlprojects/RND/.claude/worktrees/agent-a9d3932a1c6532feb/09_parallelization
L=$W/locks/gpu_run.sh
LOGS=$W/benchmarks/round6_logs
PYT="PYTHONNOUSERSITE=1 /localtmp/sl5nw/venvs/rnd09_torch/bin/python"
mkdir -p "$LOGS"

run () {  # run <logname> <working dir> <one command string>
  local name=$1 dir=$2 cmd=$3
  echo "=== $(date -Is) START $name" >> "$LOGS/driver.log"
  bash "$L" "cd $dir && $cmd" > "$LOGS/$name.log" 2>&1
  echo "=== $(date -Is) END   $name rc=$?" >> "$LOGS/driver.log"
}

# the gradient-form gate compared the flat buffer, which the three forms no longer lay out the
# same way; it now compares the parameters, which is what has to agree
run p1_gradient_form_gate "$W/ppo/torch_ppo" "$PYT tests/test_gradient_form_gpu.py"

echo "=== $(date -Is) BATCH D COMPLETE" >> "$LOGS/driver.log"
