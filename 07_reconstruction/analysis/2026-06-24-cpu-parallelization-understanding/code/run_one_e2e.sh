#!/bin/bash
# One profiled training run, tagged by srun task id. Run via `srun ... bash run_one_e2e.sh <tag> <cpt> <logdir>`.
# Uses --n_threads 0 (auto): torch picks its default from the srun-assigned CPU affinity,
# so this measures what a real run actually does under ntasks/cpus-per-task binding.
TAG="$1"; N="$2"; LOG="$3"
PY=/u/sl5nw/.conda/envs/exploration/bin/python
CODE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export RND_PROFILE=1 PYTHONUNBUFFERED=1
"$PY" "$CODE/profile_e2e.py" --algorithm rnd_linear_next_state --device cpu --n_threads 0 \
  --total_timesteps 3000 --warmup_steps 1000 \
  --out "$LOG/${TAG}_cpt${N}_proc${SLURM_PROCID:-0}.json"
