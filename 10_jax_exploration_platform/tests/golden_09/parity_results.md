# Golden-parity results

What `test_golden_parity.py` reported, run by run. Times are Pacific with a `PT` marker (the
machines run on Eastern and the times were converted for display).

Configuration in every run: 4 copies, 4 environments per copy, 32 rollout steps, 3 iterations,
seed 17, both update styles (`full_batch` and `epoch_minibatch`), the platform's trainer against
`09_parallelization/ppo/jax_ppo/jax_ppo_rnd.py` at tag `jax-rnd-baseline-v0.1.0`.

| when | platform side | device | interpreter | result | worst absolute difference |
|---|---|---|---|---|---|
| 2026-08-16 16:22 PT | the copied single module | processor (`JAX_PLATFORMS=cpu`), login node | `/p/rlprojects/RND/.venvs/platform_jax/bin/python` (jax 0.11.0) | every array bit-identical | 0.0 |
| 2026-08-16 16:31 PT | the copied single module | H100 NVL on serval05, under the lock | `/localtmp/sl5nw/venvs/rnd09_jax/bin/python` (jax 0.11.0, node-local) | every array bit-identical | 0.0 |
| 2026-08-16 16:34 PT | the copied single module | H100 NVL on serval05, under the lock | `/p/rlprojects/RND/.venvs/platform_jax/bin/python` (jax 0.11.0, the shared platform environment) | every array bit-identical | 0.0 |
| 2026-08-16 17:05 PT | after the split into environment / agent / bonus / composition | processor (`JAX_PLATFORMS=cpu`), login node | `/p/rlprojects/RND/.venvs/platform_jax/bin/python` (jax 0.11.0) | every array bit-identical | 0.0 |
| 2026-08-16 17:41 PT | after the bonus registry and the `none` family | processor (`JAX_PLATFORMS=cpu`), login node | `/p/rlprojects/RND/.venvs/platform_jax/bin/python` (jax 0.11.0) | every array bit-identical | 0.0 |
| 2026-08-16 17:41 PT | after the bonus registry and the `none` family | H100 NVL on serval05, under the lock | `/p/rlprojects/RND/.venvs/platform_jax/bin/python` (jax 0.11.0) | every array bit-identical | 0.0 |
| 2026-08-16 18:20 PT | after the visit-count family | processor (`JAX_PLATFORMS=cpu`), login node | `/p/rlprojects/RND/.venvs/platform_jax/bin/python` (jax 0.11.0) | every array bit-identical | 0.0 |
| 2026-08-16 18:20 PT | after the visit-count family | H100 NVL on serval05, under the lock | `/p/rlprojects/RND/.venvs/platform_jax/bin/python` (jax 0.11.0) | every array bit-identical | 0.0 |
| 2026-08-16 18:52 PT | after the sweep over learning rates and intrinsic weights | processor (`JAX_PLATFORMS=cpu`), login node | `/p/rlprojects/RND/.venvs/platform_jax/bin/python` (jax 0.11.0) | every array bit-identical | 0.0 |

"Bit-identical" is the literal comparison the test makes: the raw bytes of every array in the
starting state, the primed state, and each iteration's metrics and state are compared, so a
not-a-number in the same place counts as equal and a negative zero against a positive zero counts as
different. No tolerance is involved anywhere.

The automatic-algorithm-selection workaround (`GOLDEN_PARITY_AUTOTUNE_OFF=1`, which sets
`XLA_FLAGS=--xla_gpu_autotune_level=0`) was **not needed**: every run above passed with the
compiler's default settings.

## The losses printed by the first three runs

The gate is that the two modules agree with each other, and they do, exactly, in every run. The
numbers themselves move slightly between devices and between environment builds, which is expected
and is not what the gate measures:

| iteration, `full_batch` | processor, platform env | H100, node-local env | H100, platform env |
|---|---|---|---|
| 1 | 1.88423753 | 1.88431644 | 1.88431644 |
| 2 | 0.82756799 | 0.82756615 | 0.82756615 |
| 3 | 0.57105303 | 0.57097453 | 0.57097459 |

Processor against graphics card differs in the fifth decimal: the two devices accumulate a sum in a
different order. The two graphics-card runs differ only in iteration 3 of `full_batch`, in the eighth
decimal — the same jax version and the same pinned nvidia packages, but two separately built
environments, and the compiler is free to pick a different reduction order for the same program. In
both runs the reference and the copy agreed to the last bit, which is what this gate is for.
