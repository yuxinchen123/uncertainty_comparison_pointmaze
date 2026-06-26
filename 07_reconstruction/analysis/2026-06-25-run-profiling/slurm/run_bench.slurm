#!/bin/bash
# Benchmark all optimization conditions on ONE node (same-node comparison removes node-type noise).
# Each condition is a single clean process; conditions run sequentially. fps written per (algo,condition,node).
# Conditions are listed below -- extend the CONDITIONS array as optimizations are accepted from research.
#SBATCH --job-name=bench-opt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G
#SBATCH --time=03:00:00

set -e
PROF=/p/rlprojects/RND/07_reconstruction/analysis/2026-06-25-run-profiling
CODE=$PROF/analysis/code
DATA=$PROF/data
PY=/u/sl5nw/.conda/envs/exploration/bin/python
NODE=${SLURMD_NODENAME:-$(hostname -s)}
DEV=${BENCH_DEVICE:-cpu}

# bench <condition> <algorithm> <beta> <threads> <extra-flags...>: one clean timed run, OMP matched to threads
bench() {
  local cond=$1 algo=$2 beta=$3 thr=$4; shift 4
  OMP_NUM_THREADS=$thr MKL_NUM_THREADS=$thr OPENBLAS_NUM_THREADS=$thr \
  $PY $CODE/profile_train.py --condition "$cond" --algorithm "$algo" --beta "$beta" --device "$DEV" \
    --threads "$thr" --seed 0 --bench_steps 6000 --warmup 1000 "$@" \
    --out "$DATA/bench_${algo}_${cond}_${NODE}.json"
}

echo "[run_bench] node=$NODE device=$DEV start $(date)"

# --- rnd_state (the RND hot path): full condition sweep ---
bench baseline       rnd_state 100 2
bench torch_reward   rnd_state 100 2 --opt_torch_reward
bench polyak         rnd_state 100 2 --opt_polyak_foreach
bench threads1       rnd_state 100 1
bench threads4       rnd_state 100 4
bench tf8            rnd_state 100 2 --sac_train_freq 8 --sac_gradient_steps 8
bench combined_exact rnd_state 100 2 --opt_torch_reward --opt_polyak_foreach
bench combined_all   rnd_state 100 2 --opt_torch_reward --opt_polyak_foreach --sac_train_freq 8 --sac_gradient_steps 8

# --- gt_position_velocity + rnd_elliptical: baseline + bit-exact combined + all-combined ---
bench baseline       gt_position_velocity 1 2
bench combined_exact gt_position_velocity 1 2 --opt_torch_reward --opt_polyak_foreach
bench combined_all   gt_position_velocity 1 2 --opt_torch_reward --opt_polyak_foreach --sac_train_freq 8 --sac_gradient_steps 8
bench baseline       rnd_elliptical 0.01 2
bench combined_exact rnd_elliptical 0.01 2 --opt_torch_reward --opt_polyak_foreach
bench combined_all   rnd_elliptical 0.01 2 --opt_torch_reward --opt_polyak_foreach --sac_train_freq 8 --sac_gradient_steps 8

echo "[run_bench] node=$NODE done $(date)"
