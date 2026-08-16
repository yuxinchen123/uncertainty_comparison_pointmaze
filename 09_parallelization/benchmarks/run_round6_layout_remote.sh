#!/bin/bash
# Round six, the layout decision: does holding one contiguous block per parameter beat holding
# one row per copy, once the optimizer is twenty-one programs either way? Queues behind whatever
# else holds the lock.
#
#   ssh serval05 "setsid nohup bash <this file> > <logs>/remote_driver_layout.log 2>&1 < /dev/null &"
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

# the gate first: the two layouts must train to the same parameters, and every window must still
# start on a sixteen-byte boundary for every copy
run m1_layout_gates "$W/ppo/torch_ppo" "\
  $PYT tests/test_torch_ppo.py; \
  $PYT tests/test_gradient_form_gpu.py; \
  $PYT tests/test_capture_gpu.py"

# both sides read the gradients where they were written, so the only difference is the layout
run m2_layout_paired "$W/benchmarks" "\
  $PYT bench_torch_change.py --knob set \
       --off 'gradient_buffer=False' \
       --on 'parameter_layout=parameter_major,gradient_buffer=False' \
       --copies 128 1024 4096 --rounds 11 --iters 15 --style epoch_minibatch --tag _layout"

# and the same programs measured on their own, so the whole-iteration verdict has a cause
run m3_layout_programs "$W/benchmarks" "\
  $PYT probe_gradient_form.py --n-copies 4096 --parameter-layout parameter_major \
       --tag _parameter_major"

echo "=== $(date -Is) LAYOUT DECISION COMPLETE" >> "$LOGS/driver.log"
