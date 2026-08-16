# Parity between the two trainers, checked before any card time was spent

The comparison in this run folder is only meaningful if the two implementations are running the
same algorithm. Everything below was read out of
`ppo/torch_ppo/torch_ppo_rnd.py` and `ppo/jax_ppo/jax_ppo_rnd.py` (and the two environments
`pointmaze/torch_env/torch_pointmaze.py`, `pointmaze/jax_env/jax_pointmaze.py`) line by line,
and the machine checks at the end were run.

## The environment is not merely equivalent, it is the same numbers

The reset noise is a counter-based hash of `(base_seed, copy seed index, env index, reset
generation, quantity)` — two rounds of the murmur3 finaliser, then `(h >> 8) * 2^-24`, which is
exactly representable in float32. Both implementations compute that same hash, so **copy k of the
PyTorch run and copy k of the JAX run start every episode at the same position with the same goal**.
The physics is a transliteration of one closed form (eight unrolled wall-box candidates, the exact
two-contact quadratic program, one implicit-damping Euler step) with no matrix multiplication in it,
so the card's matrix precision does not touch the environment at all. The reward is the same sparse
rule: 1 for every step whose post-step position is within 0.45 of the goal, no shift, no
termination, truncation at 400 steps with automatic reset.

## Item by item

| What the specification asks to match | PyTorch | JAX | Same? |
|---|---|---|---|
| Weight distribution | orthogonal by QR of an i.i.d. Gaussian matrix with the sign of R's diagonal folded in, then scaled by the gain | same construction, `jax.random.normal` in place of `torch.randn` | yes — same distribution, different draws |
| Gains per layer | actor sqrt(2), sqrt(2), 0.01; critic sqrt(2), sqrt(2); critic heads 1.0; RND target sqrt(2), sqrt(2); RND predictor sqrt(2), sqrt(2), sqrt(2) | identical list | yes |
| Fan-in convention | drawn on [out, in] then transposed to the [in, out] the forward pass uses | identical | yes |
| Layer sizes | actor 4-64-64-2, critic 4-64-64 with two 64-1 heads, RND target 4-256-128, predictor 4-256-128-128 | identical | yes |
| Biases | zero | zero | yes |
| Log standard deviation | zero, one per action dimension, trainable | identical | yes |
| Which tensors are frozen | the RND target is outside the trainable set and wrapped in `no_grad` | outside the parameter tree and wrapped in `stop_gradient` | yes |
| Optimiser | hand-written Adam, betas 0.9 / 0.999, epsilon 1e-5, bias correction on a step counter | identical formula | yes |
| Learning-rate annealing | linear, `1 - (iteration - 1) / total`, one multiplier over all groups | identical | yes |
| Per-copy gradient-norm limit | `min(1, 0.5 / (norm + 1e-6))`, the norm taken over that copy's own slice of every parameter | identical | yes |
| Advantage normalisation | per copy, per minibatch, sample standard deviation (denominator n-1), epsilon 1e-8 | identical | yes |
| Advantage mixture | `1.0 * intrinsic + 2.0 * extrinsic` | identical | yes |
| Policy loss | clipped surrogate, clip 0.2, on the ratio of new to old log-probability | identical | yes |
| Value loss | clipped extrinsic head (clip 0.2 around the old value) plus unclipped intrinsic head, both halved, coefficient 0.5 | identical | yes |
| Entropy term | coefficient 0, the Gaussian entropy expression is present in both | identical | yes |
| RND predictor loss | mean over feature dimension then over rows of the squared difference to the frozen target | identical | yes |
| Observation statistics | per copy, float64, gymnasium parallel-variance update, initial count 1e-4 | identical | yes |
| When the observation statistics update | the intrinsic bonus uses the statistics as they stood at the START of the iteration; they are then updated; the update phase's whitened input uses the NEW ones | identical order | yes |
| Whitening | subtract mean, divide by sqrt(var + 1e-8) with the float64 statistics cast to float32 first, clip to +-5 | identical | yes |
| Intrinsic-reward normalisation | forward filter with discount 0.99 carried across iterations and never reset at an episode boundary; divided by the square root of the running variance of the filter values, epsilon 1e-8 | identical | yes |
| Advantage estimation | two streams; extrinsic discount 0.999, intrinsic 0.99, lambda 0.95; the extrinsic recursion is cut at an episode end, the intrinsic one is not; the extrinsic bootstrap passes through truncation | identical | yes |
| Episode boundary handling | automatic reset in the same step, the pre-reset observation returned separately for bootstrapping | identical | yes |
| Priming | 10 blocks of 128 random-action steps drawn uniformly on [-1, 1), one statistics update per block, then a respawn at the carried reset generation with step counts zeroed | identical | yes |
| Update style | 4 epochs x 4 shuffled minibatches of 128 rows, permutations independent per copy and per epoch | identical | yes |
| What is recorded, and when | per copy: the summed extrinsic reward of the iteration's 512 steps, the mean intrinsic reward, the cumulative fraction of open maze cells visited; every 200 iterations | identical | yes |

## Known differences, and why none of them is expected to move a learning outcome

1. **The random draws differ.** The initial weights come from different generators, and so do the
   action noise and the minibatch shuffles. This is deliberate and is what the brief asks for: the
   two must draw from the same distribution, not the same numbers. The environment reset noise,
   which is keyed rather than sequential, IS identical.
2. **PyTorch packs two matrix multiplications that JAX leaves separate** — the actor's and critic's
   first layers share one, and the RND target's and predictor's first layers share another. The
   product is the same; only the order in which the accumulator is filled differs.
3. **PyTorch computes the frozen RND target's features once per iteration and gathers them into the
   minibatches; JAX recomputes them inside each minibatch.** The target does not change during an
   iteration and a row's features do not depend on the other rows in the batch, so the values are
   the same; the batch shape the library sees differs (512 rows against 128), which can change the
   last bits.
4. **The gradient-norm sum runs over the parameters in different orders** — PyTorch in the
   trainer's fixed list order, JAX in the parameter tree's own order. Floating-point addition is not
   associative, so the last bits differ.
5. **JAX unrolls its sixteen-step update loop by two.** Its own ledger measured the effect as
   3.6e-07 absolute on one update stage, and 3.7e-16 when the same update is repeated in double
   precision — an accumulation-order difference, not a different function.
6. **Two JAX processes with the same seed are not bitwise identical** (measured below: 2.3e-05 on a
   parameter after three iterations). The cause was probed and confirmed: the compiler benchmarks
   matrix-multiply algorithms when it builds and can choose differently in different processes —
   with that benchmarking turned off, the two processes agree to the last bit. Two PyTorch
   processes with the same seed are identical either way, and so are two JAX trainers built in ONE
   process. This matters for one thing only: a resumed JAX run is not guaranteed to be bit-for-bit
   the run it would have been, and the resume check below is read with that floor in mind.

Differences 2 to 5 are last-bit effects of the size the two implementations already agree to
(8.6e-07 across their forward passes in exact single precision). Difference 1 is the point of the
experiment. None of them is a difference in the algorithm.

## Machine checks

All run on serval05 on 2026-08-16 (01:25 to 01:40 PT) under the exclusive lock, before the
campaign started. Raw output in `data/parity/` and `data/pilot/`.

| Check | How | Result |
|---|---|---|
| Cross-framework forward agreement | `ppo/torch_ppo/tests/dump_forward_fixture.py` writes shared weights and a fixed observation batch; `ppo/jax_ppo/tests/test_forward_fixture.py` reproduces every output | **PASS**, worst 1.174e-06 in exact single precision. The same comparison in the card's reduced mode: 2.175e-03 (`forward_fixture.txt`) |
| PyTorch trainer suite | `test_torch_ppo.py` — same-seed determinism, copy isolation, both update styles, the flat buffer, window alignment, the two buffer layouts | 6 of 6 pass (`tests_torch.txt`) |
| PyTorch sweep suite | `test_sweep.py` — uniform sweep reproduces the plain run, a zero-rate group stays frozen, groups do not influence each other, paired and distinct seeding, and the same under graph capture | 6 of 6 pass; uniform sweep against the plain run 1.192e-07 |
| PyTorch precision-knob suite | `test_tf32_knob.py` (new — see the defect below) | 3 of 3 pass |
| JAX trainer and sweep suites | `test_jax_ppo.py`, `test_sweep_jax.py` | 10 of 10 pass; uniform sweep against the plain run 1.751e-07 (`tests_jax.txt`) |
| The precision knob is not silently ignored | `code/precision_effect_check.py` — the real trainer, three iterations at 64 copies, one process per precision | PyTorch: the two precisions end **3.8e-03** apart on a largest parameter of 7.5e-01; the same precision twice is bitwise identical. JAX: **2.9e-03** apart, against a same-precision floor of 2.3e-05 — 127 times larger (`data/pilot/precision_effect_*.json`) |
| The requested precision is what the card used | `code/train_learning_outcome.py` multiplies matrices of the trainer's own shapes against a double-precision reference before training, and refuses to continue on a mismatch | reduced 3.5e-04 (PyTorch) / 3.6e-04 (JAX); exact 3.6e-07 / 3.7e-07. Recorded in every run's `record.json` |
| A resumed run reproduces an uninterrupted one | `code/pilot2.sh`, 60 iterations against 20 + 40 with the annealing denominator held at 60 in both | PyTorch: **bitwise identical at every recorded iteration**. JAX: differs, but already before the resume point, so the cause is the framework's own cross-process floor and not the resume (`resume_check.json`) |
| Why two JAX processes differ | the same comparison with the compiler's matrix-multiply algorithm benchmarking turned off (`XLA_FLAGS=--xla_gpu_autotune_level=0`) | **bitwise identical**, difference exactly 0.0, against 2.3e-05 with it on. The cause is the compiler choosing a different algorithm in a different process (`jax_determinism.txt`) |

### The defect this audit found

`PPOConfig.tf32` was applied only when it was ON: `if cfg.tf32:
torch.set_float32_matmul_precision("high")`. That setting is process-global, so a trainer built
with `tf32=False` did not get exact single precision — it got whatever the process had last been
put in. A fresh process starts at `"highest"`, which is why no earlier result in this project is
affected (every production configuration sets `tf32=True`, and every benchmark runs one process
per arm). But the first thing that builds both precisions in one process is this campaign's own
precision check, and it would have compared a reduced-precision trainer against another
reduced-precision trainer while calling one of them exact — a null result that reads exactly like
the finding "precision does not matter". Fixed by setting the precision in both directions and
asserting it took; test `ppo/torch_ppo/tests/test_tf32_knob.py`; recorded in
`ppo/torch_ppo/progress_and_changes.md`.

### A mistake in the first version of the resume check, and what it cost

The first resume test ran the uninterrupted arm for 60 iterations and the interrupted arm as 20
then 60 — by passing `--iterations 20` to the first process. The learning rate anneals as
`1 - (iteration - 1) / total`, so the interrupted arm annealed three times as fast over its first
20 iterations and the two runs diverged from iteration 10, before any resume had happened. The
test was measuring its own set-up. A `--stop-after` argument now ends a run early while leaving
`--iterations` at the value the full run uses, and with that the PyTorch resume is bitwise exact.
