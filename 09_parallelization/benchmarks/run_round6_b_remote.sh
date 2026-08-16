#!/bin/bash
# Round six, batch B: the three gradient forms against each other, the change the kernel profile
# found in the epoch shuffle, the compiler-generated multiplication measured properly, and the
# sizes batch A did not cover. Same lock discipline as batch A — one acquisition per group.
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

# 1. the gate for the three gradient forms, and the epoch-shuffle change is inside every gate
run h1_gates "$W/ppo/torch_ppo" "\
  $PYT tests/test_gradient_form_gpu.py; \
  $PYT tests/test_capture_gpu.py; \
  $PYT tests/test_sweep.py"

# 2. the change the kernel profile found: writing the shuffled batch straight into its buffer
#    instead of allocating a second copy of it and copying across
run h2_gather_out "$W/benchmarks" "\
  $PYT ab_compare.py --name r6-gather-out-C4096-styleB --iters 25 --warmup 6 \
       --a '{\"__rev__\": \"197b76c\", \"n_copies\": 4096}' --b '{\"n_copies\": 4096}'"

# 3. the fused copy-and-limit against the no-buffer form, which is the current best. Both are
#    measured against round five's form so the three are on one scale.
run h3_fused_limit "$W/benchmarks" "\
  $PYT bench_torch_change.py --knob fuse_copy_and_limit --off False --on True \
       --copies 1024 4096 --rounds 11 --iters 15 --style epoch_minibatch; \
  $PYT bench_torch_change.py --knob set \
       --off 'gradient_buffer=True,fuse_copy_and_limit=True' \
       --on 'gradient_buffer=False' \
       --copies 1024 4096 --rounds 11 --iters 15 --style epoch_minibatch --tag _fused_vs_nobuffer"

# 4. why the gradient forms differ, program by program
run h4_gradient_programs "$W/benchmarks" "\
  $PYT probe_gradient_form.py --n-copies 4096; \
  $PYT probe_gradient_form.py --n-copies 128"

# 5. the compiler-generated multiplication again, with one whole trainer per arm. The first
#    attempt swapped four compiled functions on ONE trainer and every arm came out bitwise
#    identical and the same speed — the compiler had handed each of them the first one's kernels.
run h5_epilogue_rebuilt "$W/benchmarks" "\
  $PYT probe_epilogue_fusion.py --n-copies 4096 --rounds 7 \
       --arms library generated_triton generated_tf32 --tag _pertrainer"

# 6. the copy counts batch A did not measure, so the crossover between the gradient forms is
#    known rather than assumed. 8,192 is included because the no-buffer form also removes a whole
#    [C, P] buffer, which is what decides whether a size fits on the card at all.
run h6_gradient_form_sizes "$W/benchmarks" "\
  $PYT bench_torch_change.py --knob gradient_buffer --off True --on False \
       --copies 8 32 128 512 2048 8192 --rounds 11 --iters 15 --style epoch_minibatch"

run h7_gradient_form_styleA "$W/benchmarks" "\
  $PYT bench_torch_change.py --knob gradient_buffer --off True --on False \
       --copies 128 1024 4096 --rounds 11 --iters 15 --style full_batch"

# 7. the JAX curve again, one update style per process. JAX reports peak device memory as a
#    process high-water mark that is never reset, so a single process measuring both styles gives
#    the second style the first one's peak — which is what the combined run in batch A did.
run h8_jax_per_style "$W/benchmarks" "\
  $PYJ bench_train_jax.py --n-copies 1024 2048 4096 --styles epoch_minibatch \
       --timing sync --tag _r6_base_styleB_large_sync; \
  $PYJ bench_train_jax.py --n-copies 1024 2048 4096 --styles full_batch \
       --timing sync --tag _r6_base_styleA_large_sync"

echo "=== $(date -Is) BATCH B COMPLETE" >> "$LOGS/driver.log"
