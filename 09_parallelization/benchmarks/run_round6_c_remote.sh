#!/bin/bash
# Round six, batch C: the state the round leaves. Every gate, the curves at the sizes in use and
# at the small sizes the earlier rounds were tuned on, the profiles repeated on the changed
# trainer, and the round measured end to end against the revision it started from.
#
#   ssh serval05 "setsid nohup bash <this file> > <logs>/remote_driver_c.log 2>&1 < /dev/null &"
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

CAP="--rollout-mode capture --capture-update --one-graph --tf32 --fused-adam --timing sync"
R5=7287209   # the merged round-five state this round starts from

run k1_gates "$W/ppo/torch_ppo" "\
  $PYT tests/test_gradient_form_gpu.py; \
  $PYT tests/test_flat_optimizer_gpu.py; \
  $PYT tests/test_bias_form_gpu.py; \
  $PYT tests/test_capture_gpu.py; \
  $PYT tests/test_sweep.py; \
  $PYT tests/test_hoist_equivalence_gpu.py; \
  $PYT tests/test_compile_post_gpu.py"

run k2_after_curves "$W/benchmarks" "\
  $PYT bench_train.py --style epoch_minibatch --n-copies 1024 2048 4096 8192 --iters 20 \
       --warmup 5 $CAP --tag _r6_after_styleB_large; \
  $PYT bench_train.py --style full_batch --n-copies 1024 2048 4096 8192 --iters 20 \
       --warmup 5 $CAP --tag _r6_after_styleA_large"

# the small sizes the earlier rounds were tuned on, before and after, so a regression there is
# found by measurement rather than assumed away
run k3_small_sizes "$W/benchmarks" "\
  $PYT bench_train.py --style epoch_minibatch --n-copies 8 32 128 512 --iters 40 --warmup 8 \
       $CAP --tag _r6_after_styleB_small; \
  $PYT bench_train.py --style epoch_minibatch --n-copies 8 32 128 512 --iters 40 --warmup 8 \
       $CAP --rev $R5 --tag _r6_before_styleB_small; \
  $PYT bench_train.py --style full_batch --n-copies 8 32 128 512 --iters 40 --warmup 8 \
       $CAP --tag _r6_after_styleA_small; \
  $PYT bench_train.py --style full_batch --n-copies 8 32 128 512 --iters 40 --warmup 8 \
       $CAP --rev $R5 --tag _r6_before_styleA_small"

run k4_after_profiles "$W/benchmarks" "\
  $PYT profile_kernels.py --n-copies 4096 --style epoch_minibatch --tag _r6_after; \
  $PYT profile_update.py --n-copies 4096 --tag _r6_after; \
  $PYT profile_phases.py --n-copies 4096 --style epoch_minibatch --tag _r6_after"

run k5_ab_all "$W/benchmarks" "\
  $PYT ab_compare.py --name r6-all-C4096-styleB --iters 25 --warmup 6 \
       --a '{\"__rev__\": \"$R5\", \"n_copies\": 4096}' --b '{\"n_copies\": 4096}'; \
  $PYT ab_compare.py --name r6-all-C4096-styleA --iters 25 --warmup 6 \
       --a '{\"__rev__\": \"$R5\", \"n_copies\": 4096, \"update_style\": \"full_batch\"}' \
       --b '{\"n_copies\": 4096, \"update_style\": \"full_batch\"}'; \
  $PYT ab_compare.py --name r6-all-C128-styleB --iters 60 --warmup 10 \
       --a '{\"__rev__\": \"$R5\", \"n_copies\": 128}' --b '{\"n_copies\": 128}'; \
  $PYT ab_compare.py --name r6-all-C8-styleB --iters 60 --warmup 10 \
       --a '{\"__rev__\": \"$R5\", \"n_copies\": 8}' --b '{\"n_copies\": 8}'"

run k6_equivalence "$W/benchmarks" "\
  $PYT compare_revisions.py --a $R5 --b '' --n-copies 32 --iters 1 --tag _r6"

echo "=== $(date -Is) BATCH C COMPLETE" >> "$LOGS/driver.log"
