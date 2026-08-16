#!/bin/bash
# Round six, batch E: the small copy counts, where two instruments disagree.
#
# The paired in-process harness says the one knob that changes at 8 copies — summing the gradient
# limit inside the copy — costs +0.64% there (0 of 11 rounds, 0.05 ms). The cross-process ABBA
# comparison of the whole round against the revision it started from says +4.2% (0.33 ms) at 8
# copies and +0.9% at 128. Those cannot both be descriptions of the same change, so this batch
# splits the round in two through the SAME instrument: the revision against the working tree with
# the fused limit turned off (which leaves only the shuffle change and the refactor), and then
# against the working tree as it ships. It also repeats the full comparison, because a single
# cross-process reading at these sizes is worth less than a repeated one.
#
#   ssh serval05 "setsid nohup bash <this file> > <logs>/remote_driver_e.log 2>&1 < /dev/null &"
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

R5=7287209

run q1_small_split "$W/benchmarks" "\
  $PYT ab_compare.py --name r6-shuffle-only-C8-styleB --iters 60 --warmup 10 \
       --a '{\"__rev__\": \"$R5\", \"n_copies\": 8}' \
       --b '{\"n_copies\": 8, \"fuse_copy_and_limit\": false}'; \
  $PYT ab_compare.py --name r6-shuffle-only-C128-styleB --iters 60 --warmup 10 \
       --a '{\"__rev__\": \"$R5\", \"n_copies\": 128}' \
       --b '{\"n_copies\": 128, \"fuse_copy_and_limit\": false}'; \
  $PYT ab_compare.py --name r6-all-C8-styleB-repeat --iters 60 --warmup 10 \
       --a '{\"__rev__\": \"$R5\", \"n_copies\": 8}' --b '{\"n_copies\": 8}'; \
  $PYT ab_compare.py --name r6-all-C128-styleB-repeat --iters 60 --warmup 10 \
       --a '{\"__rev__\": \"$R5\", \"n_copies\": 128}' --b '{\"n_copies\": 128}'"

echo "=== $(date -Is) BATCH E COMPLETE" >> "$LOGS/driver.log"
