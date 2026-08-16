#!/bin/bash
# Round six, batch A: the gates, the refreshed head-to-head, the profiles that choose the
# round's candidates, and the first candidate's paired measurement. Each group takes the
# exclusive H100 lock on its own, so another agent's work interleaves between groups instead
# of waiting for the whole batch.
#
# Start it detached so it outlives the session that launched it:
#   ssh serval05 "setsid nohup bash <this file> > <logs>/remote_driver.log 2>&1 < /dev/null &"
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

CAP="--rollout-mode capture --capture-update --one-graph --tf32 --fused-adam --timing sync"
R5=7287209   # the merged round-five state this round starts from

# 1. every gate, including the new one for this round's first change
run g1_gates "$W/ppo/torch_ppo" "\
  $PYT tests/test_gradient_form_gpu.py; \
  $PYT tests/test_flat_optimizer_gpu.py; \
  $PYT tests/test_bias_form_gpu.py; \
  $PYT tests/test_capture_gpu.py; \
  $PYT tests/test_sweep.py; \
  $PYT tests/test_hoist_equivalence_gpu.py; \
  $PYT tests/test_compile_post_gpu.py"

# 2. the head-to-head, both frameworks measured in this session; the torch side pinned to the
#    revision this round starts from so the baseline cannot drift with the working tree
run g2_torch_curve "$W/benchmarks" "\
  $PYT bench_train.py --style epoch_minibatch --n-copies 1024 2048 4096 --iters 20 --warmup 5 \
       $CAP --rev $R5 --tag _r6_base_styleB_large; \
  $PYT bench_train.py --style full_batch --n-copies 1024 2048 4096 --iters 20 --warmup 5 \
       $CAP --rev $R5 --tag _r6_base_styleA_large"

run g3_jax_curve "$W/benchmarks" "\
  $PYJ bench_train_jax.py --n-copies 1024 2048 4096 --styles epoch_minibatch full_batch \
       --timing sync --tag _r6_base_large_sync"

# 3. where the time and the bytes go now, at the size the round is about
run g4_profiles "$W/benchmarks" "\
  $PYT profile_kernels.py --n-copies 4096 --style epoch_minibatch --rev $R5 --tag _r6_base; \
  $PYT profile_update.py --n-copies 4096 --tag _r6_base; \
  $PYT profile_phases.py --n-copies 4096 --style epoch_minibatch --rev $R5 --tag _r6_base"

# 4. the question round five left open: what the compiler-generated multiplication is worth
#    once the reduced-precision matrix units are not switched off along with it
run g5_epilogue "$W/benchmarks" "\
  $PYT probe_epilogue_fusion.py --n-copies 4096 --rounds 7"

# 5. this round's first change is a refactor in its OFF position, so before measuring it,
#    check that the refactor itself moved nothing
run g6_refactor_neutral "$W/benchmarks" "\
  $PYT ab_compare.py --name r6-refactor-neutral-C1024-styleB --iters 40 --warmup 8 \
       --a '{\"__rev__\": \"$R5\", \"n_copies\": 1024}' --b '{\"n_copies\": 1024}'"

# 6. and then the change itself, paired round by round in one process
run g7_gradient_form "$W/benchmarks" "\
  $PYT bench_torch_change.py --knob gradient_buffer --off True --on False \
       --copies 1024 4096 --rounds 11 --iters 15 --style epoch_minibatch"

echo "=== $(date -Is) BATCH A COMPLETE" >> "$LOGS/driver.log"
