#!/bin/bash
# Round six, batch G: the shuffle change, measured by the instrument that can resolve it.
#
# It was kept on a kernel profile — 2.9 ms of device-to-device copying an iteration at 4,096
# copies that the change removes — and the cross-process comparison of the whole round then found
# 8 copies 3.8% SLOWER with only that change applied. A profile is not an end-to-end measurement
# and a cross-process comparison cannot resolve a percent, so it is now a knob and gets the paired
# round-by-round treatment at every size.
#
#   ssh serval05 "setsid nohup bash <this file> > <logs>/remote_driver_g.log 2>&1 < /dev/null &"
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

run s1_gather_knob "$W/benchmarks" "\
  $PYT bench_torch_change.py --knob gather_into_place --off False --on True \
       --copies 8 128 1024 4096 --rounds 11 --iters 15 --style epoch_minibatch"

echo "=== $(date -Is) BATCH G COMPLETE" >> "$LOGS/driver.log"
