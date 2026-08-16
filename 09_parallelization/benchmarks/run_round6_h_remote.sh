#!/bin/bash
# Round six, batch H: the small copy counts against the revision the round started from, this
# time measuring the configuration that actually ships there.
#
# The earlier run of this comparison built its sides from the dataclass DEFAULTS rather than
# through production_config. Those defaults are the large-copy-count form, so at 8 and 128 copies
# it timed a configuration nobody runs — the one the trainer chooses only at 1,024 and above —
# and reported it as the round's result at those sizes. ab_compare.py now builds each side
# through that revision's own production_config, which is what the rest of the project means by
# "the shipped configuration". The 4,096-copy readings were unaffected, because there the
# defaults and production_config agree.
#
#   ssh serval05 "setsid nohup bash <this file> > <logs>/remote_driver_h.log 2>&1 < /dev/null &"
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

run t1_small_shipped "$W/benchmarks" "\
  $PYT ab_compare.py --name r6-all-C8-styleB --iters 60 --warmup 10 \
       --a '{\"__rev__\": \"$R5\", \"n_copies\": 8}' --b '{\"n_copies\": 8}'; \
  $PYT ab_compare.py --name r6-all-C128-styleB --iters 60 --warmup 10 \
       --a '{\"__rev__\": \"$R5\", \"n_copies\": 128}' --b '{\"n_copies\": 128}'; \
  $PYT ab_compare.py --name r6-all-C512-styleB --iters 60 --warmup 10 \
       --a '{\"__rev__\": \"$R5\", \"n_copies\": 512}' --b '{\"n_copies\": 512}'"

echo "=== $(date -Is) BATCH H COMPLETE" >> "$LOGS/driver.log"
