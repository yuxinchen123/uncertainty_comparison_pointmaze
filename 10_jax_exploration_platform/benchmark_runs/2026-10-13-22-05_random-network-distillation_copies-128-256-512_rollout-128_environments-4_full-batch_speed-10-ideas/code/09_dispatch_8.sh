#!/bin/bash
set -euo pipefail
export PYTHONNOUSERSITE=1 XLA_PYTHON_CLIENT_PREALLOCATE=false
exec timeout --signal=TERM --kill-after=5 580 /localtmp/sl5nw/venvs/rnd09_jax/bin/python -u /p/rlprojects/RND/10_jax_exploration_platform/benchmarks/harness/bench_speed_ideas.py --idea 09_dispatch_8 --output /p/rlprojects/RND/10_jax_exploration_platform/benchmark_runs/2026-10-13-22-05_random-network-distillation_copies-128-256-512_rollout-128_environments-4_full-batch_speed-10-ideas/data/09_dispatch_8.json
