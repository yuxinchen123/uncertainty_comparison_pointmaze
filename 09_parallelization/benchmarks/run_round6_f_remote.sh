#!/bin/bash
# Round six, batch F: the corrected compiler-generated-multiplication probe, on its own, because
# batch E was already running when the correction was made.
#
# The accuracy columns of the previous run compared weights each arm's own kernels had already
# moved: forcing the compilation runs a real update step. The probe now snapshots the weights
# before that and puts them back, so the columns compare kernels rather than two different sets
# of weights. The timings were never affected.
#
#   ssh serval05 "setsid nohup bash <this file> > <logs>/remote_driver_f.log 2>&1 < /dev/null &"
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

run r1_epilogue_restored "$W/benchmarks" "\
  $PYT probe_epilogue_fusion.py --n-copies 4096 --rounds 7 \
       --arms library generated generated_triton generated_tf32 --tag _pertrainer"

echo "=== $(date -Is) BATCH F COMPLETE" >> "$LOGS/driver.log"
