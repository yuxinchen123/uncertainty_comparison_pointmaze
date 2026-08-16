#!/bin/bash
# Round six, the last thing: every gate against the code exactly as the round leaves it.
#
#   ssh serval05 "setsid nohup bash <this file> > <logs>/remote_driver_gates.log 2>&1 < /dev/null &"
set -uo pipefail
W=/p/rlprojects/RND/.claude/worktrees/agent-a9d3932a1c6532feb/09_parallelization
L=$W/locks/gpu_run.sh
LOGS=$W/benchmarks/round6_logs
PYT="PYTHONNOUSERSITE=1 /localtmp/sl5nw/venvs/rnd09_torch/bin/python"
mkdir -p "$LOGS"

echo "=== $(date -Is) START u1_final_gates" >> "$LOGS/driver.log"
bash "$L" "cd $W/ppo/torch_ppo && \
  $PYT tests/test_torch_ppo.py; \
  $PYT tests/test_gradient_form_gpu.py; \
  $PYT tests/test_parameter_layout_gpu.py; \
  $PYT tests/test_flat_optimizer_gpu.py; \
  $PYT tests/test_bias_form_gpu.py; \
  $PYT tests/test_capture_gpu.py; \
  $PYT tests/test_sweep.py; \
  $PYT tests/test_hoist_equivalence_gpu.py; \
  $PYT tests/test_compile_post_gpu.py" > "$LOGS/u1_final_gates.log" 2>&1
echo "=== $(date -Is) END   u1_final_gates rc=$?" >> "$LOGS/driver.log"
echo "=== $(date -Is) FINAL GATES COMPLETE" >> "$LOGS/driver.log"
