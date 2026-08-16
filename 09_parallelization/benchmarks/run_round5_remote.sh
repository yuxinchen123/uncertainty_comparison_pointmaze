#!/bin/bash
# The round-five measurement batch, run ON serval05 so it outlives the session that starts it.
#
# A driver held by the calling session is killed after an hour, which on a shared graphics
# processor is less than one batch takes to get through the queue. This script is started with
# setsid/nohup on serval05 instead, takes the exclusive lock once per group, and writes its logs
# to the run's own folder on shared storage so they can be read from anywhere.
#
# Start it with:
#   ssh serval05 "setsid nohup bash <this file> > <log dir>/remote_driver.log 2>&1 < /dev/null &"
set -uo pipefail
W=/p/rlprojects/RND/.claude/worktrees/agent-ab3b4d042ce36522c/09_parallelization
L=$W/locks/gpu_run.sh
LOGS=$W/benchmarks/round5_logs
mkdir -p "$LOGS"
PYT="PYTHONNOUSERSITE=1 /localtmp/sl5nw/venvs/rnd09_torch/bin/python"
R0=25876a8      # the trainer as round five found it (round four)
R1=973b07d      # + sixteen-byte alignment of every parameter window
R2=abf16b5      # + bias added after the multiplication
R3=5859ffc      # + gradients written not accumulated, gradient limit split from the Adam step
CAP="--rollout-mode capture --capture-update --one-graph --tf32 --fused-adam --timing sync"
T=$W/ppo/torch_ppo/tests
B=$W/benchmarks

run () {  # run <logname> <working dir> <one command string>
  local name=$1 dir=$2 cmd=$3
  echo "=== $(date -Is) START $name" >> "$LOGS/driver.log"
  bash "$L" "cd $dir && $cmd" > "$LOGS/$name.log" 2>&1
  echo "=== $(date -Is) END   $name rc=$?" >> "$LOGS/driver.log"
}

run g1_tests "$T" "$PYT test_flat_optimizer_gpu.py; echo '--- capture ---'; $PYT test_capture_gpu.py; echo '--- sweep ---'; $PYT test_sweep.py; echo '--- hoist ---'; $PYT test_hoist_equivalence_gpu.py; echo '--- compile post ---'; $PYT test_compile_post_gpu.py"

run g4_curves "$B" "$PYT bench_train.py --style full_batch --n-copies 1024 2048 4096 8192 --iters 20 --warmup 5 $CAP --tag _after_r5_styleA_large; $PYT bench_train.py --style epoch_minibatch --n-copies 1024 2048 4096 8192 --iters 20 --warmup 5 $CAP --tag _after_r5_styleB_large"

run g2_ab_each "$B" "$PYT ab_compare.py --name round5-align-C1024-styleB --iters 40 --warmup 8 --a '{\"__rev__\":\"$R0\",\"n_copies\":1024}' --b '{\"__rev__\":\"$R1\",\"n_copies\":1024}'; $PYT ab_compare.py --name round5-bias-C1024-styleB --iters 40 --warmup 8 --a '{\"__rev__\":\"$R1\",\"n_copies\":1024}' --b '{\"__rev__\":\"$R2\",\"n_copies\":1024}'; $PYT ab_compare.py --name round5-gradient-C1024-styleB --iters 40 --warmup 8 --a '{\"__rev__\":\"$R2\",\"n_copies\":1024}' --b '{\"__rev__\":\"$R3\",\"n_copies\":1024}'"

run g3_ab_all "$B" "$PYT ab_compare.py --name round5-all-C4096-styleB --iters 25 --warmup 6 --a '{\"__rev__\":\"$R0\",\"n_copies\":4096}' --b '{\"__rev__\":\"$R3\",\"n_copies\":4096}'; $PYT ab_compare.py --name round5-all-C4096-styleA --iters 25 --warmup 6 --a '{\"__rev__\":\"$R0\",\"n_copies\":4096,\"update_style\":\"full_batch\"}' --b '{\"__rev__\":\"$R3\",\"n_copies\":4096,\"update_style\":\"full_batch\"}'; $PYT ab_compare.py --name round5-all-C128-styleB --iters 60 --warmup 10 --a '{\"__rev__\":\"$R0\",\"n_copies\":128}' --b '{\"__rev__\":\"$R3\",\"n_copies\":128}'; $PYT ab_compare.py --name round5-all-C8-styleB --iters 60 --warmup 10 --a '{\"__rev__\":\"$R0\",\"n_copies\":8}' --b '{\"__rev__\":\"$R3\",\"n_copies\":8}'"

run g5_small "$B" "$PYT bench_train.py --style epoch_minibatch --n-copies 8 32 128 512 --iters 40 --warmup 8 $CAP --tag _after_r5_styleB_small; $PYT bench_train.py --style epoch_minibatch --n-copies 8 32 128 512 --iters 40 --warmup 8 $CAP --rev $R0 --tag _before_r5_styleB_small; $PYT bench_train.py --style full_batch --n-copies 8 32 128 512 --iters 40 --warmup 8 $CAP --tag _after_r5_styleA_small; $PYT bench_train.py --style full_batch --n-copies 8 32 128 512 --iters 40 --warmup 8 $CAP --rev $R0 --tag _before_r5_styleA_small"

run g6_profiles "$B" "$PYT profile_kernels.py --n-copies 1024 --style epoch_minibatch --tag _after; $PYT profile_phases.py --n-copies 4096 --style epoch_minibatch --tag _after_r5; $PYT profile_update.py --n-copies 4096 --tag _after_r5"

run g7_equivalence "$B" "$PYT compare_revisions.py --a $R0 --b $R1 --n-copies 32 --iters 1 --tag _align; $PYT compare_revisions.py --a $R1 --b $R2 --n-copies 32 --iters 1 --tag _bias; $PYT compare_revisions.py --a $R2 --b $R3 --n-copies 32 --iters 1 --tag _gradient"

echo "=== $(date -Is) ALL DONE" >> "$LOGS/driver.log"
