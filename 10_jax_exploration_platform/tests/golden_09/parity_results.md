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
| 2026-08-16 17:27 PT | the split into environment / agent / bonus / composition (commit `5718f94`) | processor (`JAX_PLATFORMS=cpu`), login node | `/p/rlprojects/RND/.venvs/platform_jax/bin/python` (jax 0.11.0) | every array bit-identical | 0.0 |
| 2026-08-16 17:31 PT | the bonus registry and the `none` family (commit `1abf758`) | processor (`JAX_PLATFORMS=cpu`), login node | `/p/rlprojects/RND/.venvs/platform_jax/bin/python` (jax 0.11.0) | every array bit-identical | 0.0 |
| 2026-08-16 17:41 PT | the same, on the card, in the batch that also measured no-bonus against distillation | H100 NVL on serval05, under the lock | `/p/rlprojects/RND/.venvs/platform_jax/bin/python` (jax 0.11.0) | every array bit-identical | 0.0 |
| 2026-08-16 18:08 PT | the visit-count family (commit `7476aa0`) | processor (`JAX_PLATFORMS=cpu`), login node | `/p/rlprojects/RND/.venvs/platform_jax/bin/python` (jax 0.11.0) | every array bit-identical | 0.0 |
| 2026-08-16 18:08 PT | the same, on the card | H100 NVL on serval05, under the lock | `/p/rlprojects/RND/.venvs/platform_jax/bin/python` (jax 0.11.0) | every array bit-identical | 0.0 |
| 2026-08-16 18:18 PT | the sweep over learning rates and intrinsic weights (commit `91886fb`) | processor (`JAX_PLATFORMS=cpu`), login node | `/p/rlprojects/RND/.venvs/platform_jax/bin/python` (jax 0.11.0) | every array bit-identical | 0.0 |
| 2026-08-16 18:44 PT | the finished stage, at the head of the branch, as the first test of the whole suite | processor (`JAX_PLATFORMS=cpu`), login node | `/p/rlprojects/RND/.venvs/platform_jax/bin/python` (jax 0.11.0) | every array bit-identical | 0.0 |
| 2026-08-16 18:47 PT | the finished stage, at the head of the branch, as the first test of the whole suite | H100 NVL on serval05, under the lock | `/p/rlprojects/RND/.venvs/platform_jax/bin/python` (jax 0.11.0) | every array bit-identical | 0.0 |

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

## The whole test suite at the end of the refactor

Every check the platform has, run start to finish on 2026-08-16 after the last commit of the
stage — on the processor (18:44 to 19:05 PT) and on the H100 NVL under the serval05 lock (18:47 to
19:07 PT). Both ended with exit code 0.

| file | processor | graphics card |
|---|---|---|
| `golden_09/test_golden_parity.py` | pass, worst difference 0.0 | pass, worst difference 0.0 |
| `agents/test_jax_ppo.py` | pass | not run (device-independent) |
| `agents/test_sweep_jax.py` | pass | not run (device-independent) |
| `agents/test_hoist_equivalence.py` | pass, worst field 2.9e-07 against a limit of 2e-04 | not run |
| `agents/test_flat_params_equivalence.py` | pass, gradients agree exactly in double precision | not run |
| `bonuses/test_registry.py` | pass | pass |
| `bonuses/test_none_has_no_bonus_arithmetic.py` | pass, 132 matrix multiplications with the bonus against 117 without | pass, 115 against 105 |
| `bonuses/test_visit_count.py` | pass | pass |
| `bonuses/test_visit_count_learning_sanity.py` | pass (run separately) | pass, busiest entry 1,172 visits, bonus 0.0292 there against 1.0000 at an unreached state |
| `parity_07/test_visit_count_against_07.py` | pass, 0 of 10,800 table entries differ | pass, 0 of 10,800 |
| `copy_isolation/test_bonus_copy_isolation.py` | pass, all four families, both update styles | pass, all four families, both update styles |
| `integration/test_jit_eager_equivalence.py` | pass, worst 1.3e-06 by the mixed rule | not run (uncompiled execution on the card is very slow) |

## The whole test suite at the visit-count throughput gate

Run again at the end of the `algo/visit-count` fork's round 1, on 2026-08-16 — on the processor
(20:16 to 20:29 PT) and on the H100 NVL under the serval05 lock (20:22 to 20:25 PT). Both ended
with exit code 0, 0 failed. **No source file changed in that round**, so this is the same code the
row above tested; the point of re-running it is that a fork does not merge on a remembered green.

One command runs the whole suite now: `bash tests/run_all.sh` on the processor,
`bash tests/run_all.sh gpu` on the card (through the lock). Each test runs in its own interpreter,
so one test's device buffers cannot reach the next. The card skips the four agent-only checks,
whose results do not depend on the device, and the uncompiled-execution comparison, which takes
very long there.

| file | processor | graphics card |
|---|---|---|
| `golden_09/test_golden_parity.py` | pass, worst difference 0.0 | pass, worst difference 0.0 |
| `agents/test_jax_ppo.py` | pass | skipped (device-independent) |
| `agents/test_sweep_jax.py` | pass | skipped (device-independent) |
| `agents/test_hoist_equivalence.py` | pass | skipped (device-independent) |
| `agents/test_flat_params_equivalence.py` | pass | skipped (device-independent) |
| `bonuses/test_registry.py` | pass | pass |
| `bonuses/test_none_has_no_bonus_arithmetic.py` | pass | pass, 115 matrix multiplications with the bonus against 105 without |
| `bonuses/test_visit_count.py` | pass | pass |
| `bonuses/test_visit_count_learning_sanity.py` | pass | pass, intrinsic reward 0.1932 to 0.0600 for `1/sqrt(n)` and 0.0503 to 0.0070 for `1/n`, coverage 0.049 to 0.141 in both |
| `parity_07/test_visit_count_against_07.py` | pass | pass, 0 of 10,800 table entries differ, worst bonus difference 9.93e-09 |
| `copy_isolation/test_bonus_copy_isolation.py` | pass | pass, all four families, both update styles |
| `integration/test_jit_eager_equivalence.py` | pass | skipped (uncompiled execution on the card is very slow) |

The card's run took 2.6 minutes against the 20 of the earlier one, because jax's persistent
compilation cache was switched on for this work
(`JAX_COMPILATION_CACHE_DIR=/localtmp/sl5nw/platform_jax_cache`, node-local on serval05). The cache
key is the compiled program, so a changed source recompiles; it saves repeats of an unchanged one,
which at 8,448 copies was 500 seconds a program.
