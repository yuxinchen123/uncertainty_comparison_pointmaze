#!/bin/bash
# The machine half of the parity audit (the reading half is ../parity_check.md), plus one probe
# the pilot raised: two JAX processes with the same seed are not bitwise identical, and the
# obvious suspect is the compiler benchmarking matrix-multiply algorithms when it builds. Turning
# that off and repeating the comparison says whether it is the cause.
set -uo pipefail
RUN="$1"
CODE="$RUN/code"
OUT="$RUN/data/parity"
BASE=/p/rlprojects/RND/09_parallelization
TORCH=/localtmp/sl5nw/venvs/rnd09_torch/bin/python
JAX=/localtmp/sl5nw/venvs/rnd09_jax/bin/python
export PYTHONNOUSERSITE=1
mkdir -p "$OUT"

echo "=== PyTorch trainer, sweep and precision-knob suites ==="
( cd "$BASE/ppo/torch_ppo/tests" \
  && $TORCH test_torch_ppo.py && $TORCH test_sweep.py && $TORCH test_tf32_knob.py \
) > "$OUT/tests_torch.txt" 2>&1
echo "torch tests exit $? — $(tail -1 "$OUT/tests_torch.txt")"

echo "=== JAX trainer and sweep suites ==="
( cd "$BASE/ppo/jax_ppo/tests" && $JAX test_jax_ppo.py && $JAX test_sweep_jax.py \
) > "$OUT/tests_jax.txt" 2>&1
echo "jax tests exit $? — $(tail -1 "$OUT/tests_jax.txt")"

echo "=== cross-framework forward agreement ==="
( cd "$BASE/ppo/torch_ppo/tests" && $TORCH dump_forward_fixture.py \
  && cd "$BASE/ppo/jax_ppo/tests" && $JAX test_forward_fixture.py \
) > "$OUT/forward_fixture.txt" 2>&1
echo "forward fixture exit $? — $(tail -2 "$OUT/forward_fixture.txt")"

echo "=== is the JAX run-to-run difference the compiler's algorithm choice? ==="
for TAG in a b; do
  XLA_FLAGS="--xla_gpu_autotune_level=0" $JAX "$CODE/precision_effect_check.py" dump \
      --framework jax --precision reduced --out "$OUT/jax_noautotune_$TAG.npy"
done
$JAX - <<PYEOF > "$OUT/jax_determinism.txt" 2>&1
"""Two JAX processes with the same seed, with algorithm benchmarking turned off."""
import json
import numpy as np
a = np.load("$OUT/jax_noautotune_a.npy")
b = np.load("$OUT/jax_noautotune_b.npy")
prev = json.load(open("$RUN/data/pilot/precision_effect_jax.json"))
print(json.dumps({
    "with_algorithm_benchmarking_largest_difference": prev["repeat_largest_difference"],
    "without_algorithm_benchmarking_bitwise_identical": bool(np.array_equal(a, b)),
    "without_algorithm_benchmarking_largest_difference": float(np.abs(a - b).max()),
}, indent=1))
PYEOF
cat "$OUT/jax_determinism.txt"
echo PARITY_CHECKS_DONE
