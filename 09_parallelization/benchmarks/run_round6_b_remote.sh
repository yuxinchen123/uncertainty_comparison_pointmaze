#!/bin/bash
# Round six, batch B: the copy counts batch A did not cover for the gradient-form change, so
# the default can be chosen from a measurement rather than a guess. Same lock discipline as
# batch A — one acquisition per group.
#
#   ssh serval05 "setsid nohup bash <this file> > <logs>/remote_driver_b.log 2>&1 < /dev/null &"
set -uo pipefail
W=/p/rlprojects/RND/.claude/worktrees/agent-a9d3932a1c6532feb/09_parallelization
L=$W/locks/gpu_run.sh
LOGS=$W/benchmarks/round6_logs
PYT="PYTHONNOUSERSITE=1 /localtmp/sl5nw/venvs/rnd09_torch/bin/python"
PYJ="PYTHONNOUSERSITE=1 /localtmp/sl5nw/venvs/rnd09_jax/bin/python"
mkdir -p "$LOGS"

run () {  # run <logname> <working dir> <one command string>
  local name=$1 dir=$2 cmd=$3
  echo "=== $(date -Is) START $name" >> "$LOGS/driver.log"
  bash "$L" "cd $dir && $cmd" > "$LOGS/$name.log" 2>&1
  echo "=== $(date -Is) END   $name rc=$?" >> "$LOGS/driver.log"
}

# the sizes batch A did not measure, so the crossover between the two gradient forms is known
# rather than assumed. 8,192 is included because the change also removes a whole [C, P] buffer,
# which is what decides whether a size fits on the card at all.
run h1_gradient_form_sizes "$W/benchmarks" "\
  $PYT bench_torch_change.py --knob gradient_buffer --off True --on False \
       --copies 8 32 128 512 2048 8192 --rounds 11 --iters 15 --style epoch_minibatch"

run h2_gradient_form_styleA "$W/benchmarks" "\
  $PYT bench_torch_change.py --knob gradient_buffer --off True --on False \
       --copies 128 1024 4096 --rounds 11 --iters 15 --style full_batch"

# the JAX curve again, one update style per process. JAX reports peak device memory as a
# process high-water mark that is never reset, so a single process measuring both styles gives
# the second style the first one's peak — which is what the combined run in batch A did.
run h3_jax_per_style "$W/benchmarks" "\
  $PYJ bench_train_jax.py --n-copies 1024 2048 4096 --styles epoch_minibatch \
       --timing sync --tag _r6_base_styleB_large_sync; \
  $PYJ bench_train_jax.py --n-copies 1024 2048 4096 --styles full_batch \
       --timing sync --tag _r6_base_styleA_large_sync"

echo "=== $(date -Is) BATCH B COMPLETE" >> "$LOGS/driver.log"
