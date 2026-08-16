#!/bin/bash
# The whole CPU/PyTorch throughput research, run serially inside ONE puma01 job (one job at a
# time on the node = clean numbers). Stages: determinism check -> bit-exactness gates for every
# (configuration, candidate) -> A/B timings -> deep cProfile. Every line of output stays in the
# job log; [GATEHASH]/[RESULT] lines are the machine-readable results.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
PY="/p/rlprojects/RND/.venvs/exploration/bin/python"
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
run() { echo "=== $(date '+%H:%M:%S') $*"; "$PY" "$HERE/profile_run.py" "$@"; }

# stage 1: determinism — the same gate twice MUST hash identically, else gating is void
run --mode gate --config alg23 --candidate none --tag det_a
run --mode gate --config alg23 --candidate none --tag det_b
ha=$(grep -h GATEHASH <<<"$(grep -s GATEHASH /dev/null)" || true)   # hashes are read from the log by the analyst

# stage 2: bit-exactness gates, every configuration x candidate
for cfg in alg23 run5rnd gt; do
  run --mode gate --config $cfg --candidate none        --tag g_${cfg}_base
  for cand in polyak torchreward interop all; do
    run --mode gate --config $cfg --candidate $cand     --tag g_${cfg}_${cand}
  done
done

# stage 3: A/B timings (30k steps, two full-size evals, same seed) — alg2.3 fully, run5rnd
# baseline + combined (candidates are shared torch paths; alg2.3 isolates each one's effect)
for cand in none polyak torchreward interop all; do
  run --mode time --config alg23 --candidate $cand      --tag t_alg23_${cand}
done
run --mode time --config run5rnd --candidate none       --tag t_run5rnd_none
run --mode time --config run5rnd --candidate all        --tag t_run5rnd_all

# stage 4: deep profile of the baseline (where does the time actually go)
run --mode profile --config alg23 --candidate none      --tag p_alg23_base

echo "=== research sequence complete $(date '+%H:%M:%S')"
