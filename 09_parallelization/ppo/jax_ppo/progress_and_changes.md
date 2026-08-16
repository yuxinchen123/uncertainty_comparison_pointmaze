# jax_ppo — experiment ledger (autoresearch style)

Goal metric (correctness): the three test suites in `tests/` plus the cross-framework
forward-agreement test against the torch fixture.
Goal metric (speed): training iterations/second on the serval05 H100
(JSONs in `../../benchmarks/results/`, impl `jax_ppo`).

| # | change | result | verdict |
|---|---|---|---|
| 0 | v0: full twin of torch_ppo_rnd.py per the algorithm spec (incl. the section-11 correction). Batched params [C, ...], matmul forwards, float64 running statistics (x64 on, everything else explicit float32), lax.scan rollout, scan intrinsic filter, reverse-scan GAE, per-copy grad clip, sum-over-copies loss, hand-rolled torch-formula Adam, two separately jitted update styles (style B = one lax.scan over 16 pre-gathered minibatches). | CPU tests all pass: same-seed bit-identical (both styles), copy isolation (both styles), finite + falling intrinsic reward. Cross-framework forward agreement vs torch fixture: worst mixed error 8.6e-7 (gate 1e-5). GPU (H100): one compilation (~4 s), then steady per-iteration times; C=8 smoke 300 iterations in 8.8 s with intrinsic reward falling 4.3 -> 0.03. | KEEP — correctness baseline |
| 1 | first throughput rows (no tuning): full_batch 111/108/74 iter/s at C=8/32/128; epoch_minibatch 86/78/50 iter/s. At C=128 full_batch trains 4.87e6 env-steps/s, epoch_minibatch 3.27e6. Copies are close to free: 16x copies costs only ~1.5x wall time. | baseline numbers recorded (JSON 2026-08-15-01-37-53) | measured || 2 | donate_argnums=(0,) on the whole-iteration jit (TrainState donated) | neutral at current sizes (<=1.3%, within noise); tests still pass bitwise | KEEP |
| 3 | LABELED VARIANT T=32/N=16 (spec 14): 4x fewer sequential rollout steps, 4x wider per-step GEMMs | style A C=128: 3.0x (225.8 iter/s, 1.48e7 env-steps/s); style B 1.9x | recorded (not default) || 4 | C-scaling sweep (see e2e/jax_e2e ledger row 3 for the numbers; JSON 2026-08-15-02-49-51 tag _cscale) | per-copy throughput halves between C=512 and 1024 (style A) / at C~512 (style B); total saturates ~1.9e7 (A) / ~9.1e6 (B) env-steps/s | measured || 5 | ROUND 2 (J1). Hoist the critic forward, the log-probability and the RND bonus OUT of the rollout scan: the scan now carries only the actor forward and the environment step, and the three hoisted quantities are computed afterwards in one wide pass each over [C, T*N, ...]. The two critic passes (on-step values and bootstrap values) become one call on a concatenated batch. Kept behind `hoist_rollout` so both forms stay measurable. | PAIRED ABBA, 60 iterations/side: C=128 style B 21.16 -> 14.46 ms (**+31.7%**, noise floor 0.48 ms); C=8 style B 13.06 -> 9.34 ms (+28.5%, floor 0.16); C=128 style A 15.07 -> 7.92 ms (**+47.5%**, floor 0.36). Exactness: isolated single iteration from identical state, worst relative field deviation 4.15e-07 (ret_ext); drift after 3 chained iterations 8.94e-08 absolute in parameters. CPU suite still passes (same-seed bitwise, copy isolation, falling intrinsic reward). | KEEP |
| 6 | ROUND 2 (J2). `unroll` on the rollout scan, swept against unroll=1 after the J1 hoist changed the body | PAIRED, C=128 style B: unroll 2 +7.1% (13.72 ms), unroll 4 +9.3% (13.12 ms), unroll 8 +9.2% (13.07 ms). 4 and 8 differ by 0.05 ms against noise floors of 0.40 and 0.10 ms, so they tie | KEEP unroll=4 (same speed as 8, less compile time and memory) |
| 7 | ROUND 2 (J5). Float64 audit of the compiled iteration, then a test of the one promotion it found | Audit: 228 f64 result shapes against 9933 f32 (2.2%), ALL of them in the running-statistics path — no leak into the rollout or the update. The only sizeable f64 arrays are the statistics' own batch inputs ([C,T*N,4] observations and [C,1,T*N] filtered intrinsic rewards), which the torch twin promotes identically. Reducing those batch statistics in float32 before promoting: +0.2% at C=128, below the 0.10 ms noise floor | NO CHANGE (`batch_stats_f32=False`): the promotion is free, and changing it would break bit-agreement with the torch twin for nothing |
| 8 | ROUND 2 (J3). Are XLA command buffers (the CUDA-graph equivalent) already on? Measured by DISABLING them (`XLA_FLAGS=--xla_gpu_enable_command_buffer=`) in an ABBA sequence | C=128 style B: default 13.34 / 13.05 ms vs disabled 15.51 / 15.52 ms — command buffers are ON by default and worth **+15.0%**. This is the jax-side equivalent of the manual CUDA-graph capture the torch build had to write by hand, and it is a large part of why the jax build led the torch build in round 1 | NO CHANGE NEEDED (already on); recorded as the explanation |
| 9 | ROUND 2 (J4). XLA Triton GEMM flag `--xla_gpu_triton_gemm_any=true`, ABBA | C=128 style B: base 13.25 / 12.82 ms (spread 0.43), triton 13.43 / 13.24 ms (spread 0.19). The 0.30 ms gap is inside the base side's own spread | NO CHANGE (no effect above the noise floor) |
| 10 | ROUND 4. Learning-rate sweep across copy groups, matching the torch feature (`sweep_config(rates, copies_per_rate)`, paired/distinct seeding, `SWEEP.md`). The per-copy rates are a device CONSTANT reshaped to [C, 1, ...] inside the existing elementwise Adam update, and the annealed rate stays ONE SCALAR argument per iteration (the constant holds the base rates, the argument carries the multiplier) so no per-iteration host-to-device copy is introduced. Without a sweep the optimizer arithmetic is untouched. The environment and the weight initialiser take a `copy_seed_index` so paired groups share seed streams. | Cost of the mechanism (same-rates against uniform, matched copies): **+0.1% / -0.5%** at C=8/128 synchronised, **0.0% / -0.2%** pipelined, **+0.2% / +0.1%** at C=512/2048 — every value inside the noise floor (0.9% to 2.6%). Cost of the rates actually DIFFERING: **-0.4% to +0.4%**, also inside the floor, as expected since it changes only the constants the same kernels read. Tests: uniform sweep reproduces the plain run (5.96e-08), zero-rate group frozen exactly while others train, changing one group's rate leaves other groups BITWISE unchanged, paired and distinct seeding both verified. | KEEP — free |
| 11 | ROUND 4. Per-copy progress recording: `track_coverage` maintains a [C, cells] boolean visited map inside the compiled iteration (one scatter of the rollout's [C, T*N] cell indices), and `train(..., history_every=N)` records per-copy reward, intrinsic reward and coverage. Only the recorded iterations copy anything to the host, so the loop is never serialised by the recording. | Cost: **-0.0% / -0.5%** at C=8/128 synchronised, **-0.3% / -0.0%** pipelined, **+0.4% / +0.3%** at C=512/2048 — inside the noise floor everywhere. The full parity configuration (differing rates AND coverage together) costs **-0.8% / -0.3%** synchronised and **-0.6% / 0.0% / +0.1% / -0.1%** pipelined across C=8..2048. A new test checks the map accumulates across iterations, using a rollout long enough that the ball leaves its starting cell — with a short rollout coverage sits at one cell and a map that reset every iteration would pass unnoticed. | KEEP — free |
| 12 | ROUND 4, measurement method. The first version of the parity benchmark ran each arm in its OWN PROCESS in ABBA order, like the existing ab_compare harnesses. | Process-to-process variation alone was 0.68-0.72 ms on a 7.9-12.7 ms iteration (5.7-8.6%), larger than every effect being measured, and RAISING the iteration count from 40 to 150 did not reduce it — the variance is between processes, not within them. Rebuilt to construct all five arms in ONE process and time them round-robin with the arm order reversed on alternate rounds: the noise floor fell to 0.06-0.10 ms pipelined (0.8-0.9%), which finally resolves the effects. | KEEP the interleaved design; the per-process form cannot answer this question |

| 13 | ROUND 5 (J1). Profile the iteration before changing anything, to test the ceiling analysis's claim that the twenty-one separate parameter tensors are the largest remaining cost. Four timings at 128 copies: the whole iteration, the update stage alone, the sixteen gradients with the clip and the optimizer removed, and the same with the clip but not the optimizer. The first attempt consumed only ONE gradient leaf, which let the compiler delete the other twenty as dead code and credited their cost to the clip — reporting a clip cost of 5.00 ms that does not exist. Consuming every leaf corrects it. | Of a 13.45 ms iteration: environment 3.63 ms (27.0%), policy and post-rollout processing 1.45 ms (10.8%), the sixteen gradients 5.79 ms (43.1%), the gradient clip 0.80 ms (6.0%), the optimizer 1.77 ms (13.2%). The clip and the optimizer together are 2.57 ms, 19% — real, but a quarter of what the mis-measured version claimed, and well below the gradients. | KEEP the profile; it redirected the round away from the clip |
| 14 | ROUND 5 (J2). The ceiling analysis's headline item: hold every parameter in ONE array [copies, parameters per copy] instead of twenty-one named tensors, so the clip is one reduction and Adam a handful of elementwise operations, with the array cut back into named tensors before each forward pass (`flat_params`). | **Size-dependent, and it reverses.** Synchronised, paired: **-5.5% at 8 copies** (7.97 -> 7.53 ms, floor 0.07), within noise at 32, and **+14.4% SLOWER at 128 copies** (13.20 -> 15.11 ms, floor 0.26). The cause is that a parameter's slice of a [copies, parameters] array is strided, so cutting it out is a real copy: at 128 copies the sixteen unpacks per update move about a gigabyte, which costs more than the twenty-one saved kernel launches. Accuracy: the loss is BIT-IDENTICAL and, repeated in double precision, the gradient agrees to 4.2e-16 — the two forms compute the same function, and the single-precision difference is the backward pass accumulating in a different order. | REVERT the default (`flat_params=False`). The flag and its test stay so the 8-copy gain is available and the measurement is reproducible |
| 15 | ROUND 5 (J3). Re-sweep the rollout scan's unroll factor, which round 2 set to 4 on a measurement that could not separate 4 from 8. | Unroll 16 beats 4 in **11 of 11 paired rounds at every size**: **-1.2% / -3.0% / -3.2%** at 8 / 32 / 128 copies (paired differences +0.08 / +0.30 / +0.39 ms against floors of 0.12 / 0.21 / 0.15). Unroll 32 beats 16 at 8 copies (**-2.4%**, 11 of 11) but LOSES at 32 and 128 (0 of 11 and 1 of 11), because one rollout step does more work the more copies there are. Accuracy: unroll is a compiler instruction, not an arithmetic change, and on the default hoisted path the parameters after one iteration are **bit-identical** at 4, 16 and 32. With the hoist turned OFF the in-scan critic and log-probability fuse differently at 32 (4 against 32 deviates 3.87e-04; 4 against 16 stays exact), so 32 is taken only where it is bit-neutral. | KEEP, choosing by copy count: 32 at 8 copies or fewer with the hoist on, 16 otherwise — every configuration still computes exactly what it did before |
| 16 | ROUND 5 (J4), measurement method. The verdict rule inherited from round 4 compares two medians against the worst spread the SAME version shows between rounds. That throws away the pairing the design already provides: the two versions are timed in the same round on the same machine, so whatever drift moves one moves the other. | On the unroll change the median-against-spread rule returned "within noise" at all three sizes (-1.3% / -3.3% / -2.8% against floors of 0.11 / 0.73 / 0.47 ms), while the round-by-round differences were positive in **11 of 11 rounds at every size** — a sign test at p = 0.001. The effect was real and the statistic was hiding it. | KEEP the paired statistic; report both it and the floor |
| 17 | ROUND 5, inherited. TWO gates fail on the merged trainer BEFORE any round-five change, both verified by stashing every round-five edit: `test_hoist_equivalence.py` (worst relative field deviation 2.62e-05 against its own 1e-5 threshold) and `test_forward_fixture.py` (`act_mean` at 7.80e-05 against 1e-5, after regenerating the fixture file, which was absent from the machine). Neither is caused by this round's work; the unroll change does not affect the hoist comparison, which measures 2.44e-04 on parameters at unroll 4, 16 and 32 alike. | Not investigated further — outside this round's scope. | REPORT to the parent session; both gates need their thresholds re-derived or the underlying agreement re-examined |

## Round 4 summary: what the two parity features cost

Nothing measurable. Across C = 8, 128, 512 and 2048, in both timing modes, the largest
absolute effect of any arm is 0.8% and its sign is as often negative as positive, against
noise floors of 0.8% to 2.6%. The full parity configuration — differing learning rates per
group plus per-copy coverage recording — measures 0.0% at C=128 pipelined (11.93 ms against
11.93 ms) with a 0.8% floor.

The design is why, and it was chosen for this rather than optimised into it afterwards: the
per-copy rate is a device constant broadcast inside a kernel XLA was already emitting, the
annealing stays one scalar argument so no host-to-device copy is added per iteration, and the
visited map is updated on the device and read only when a row is recorded. The torch side
reached +1.0% for the same sweep feature and needed its per-copy optimizer compiled to get
there; in JAX the mapped-over-copies structure makes the same thing free from the start.

| measurement | C=8 | C=128 | C=512 | C=2048 |
|---|---|---|---|---|
| uniform baseline, pipelined (ms) | 6.90 | 11.93 | 25.16 | 81.78 |
| full parity configuration (ms) | 6.86 | 11.93 | 25.18 | 81.70 |
| noise floor (ms) | 0.06 | 0.10 | 0.33 | 1.09 |
| 18 | ROUND 5 (J6). Unroll the SIXTEEN-STEP UPDATE scan, which had never been unrolled — the profile put 62% of the iteration in the update stage, so the loop that runs it sixteen times was the obvious next place to look. Two steps per loop body gives the compiler one step's optimizer and the next step's gradient to overlap. | The largest single gain of the round: **-3.6% / -4.2% / -8.5%** at 8 / 32 / 128 copies synchronised (13.11 -> 12.00 ms at 128, paired difference +1.11 ms against a 0.14 ms floor), **11 of 11 paired rounds at every size**. Unroll 4 is NOT better than 2 (slower or mixed at every size, 0 to 3 rounds of 11). Accuracy: unlike the rollout scan this is not bit-neutral — one update stage differs by **3.6e-07 absolute** (6.8e-05 relative, inflated by tensors that start at zero) — but the SAME update in double precision agrees to **3.7e-16**, so the two compute the same function and single precision is merely accumulating in a different order. The cross-framework forward fixture is unaffected: it compares forward passes, which the update scan does not touch. | KEEP at 2 |

## Round 5 summary: where the iteration's time actually goes, and what moved it

The round began by measuring instead of implementing, and that decided everything after it.
The ceiling analysis, derived largely from the PyTorch side, named one large inexpensive item:
hold every parameter in a single flat buffer so the gradient clip and the optimizer stop walking
twenty-one separate arrays. Two of its other suggestions turned out to be **already implemented**
in the JAX trainer — the sixteen minibatches are gathered once before the update rather than per
step, and the sixteen optimizer steps already run as one `lax.scan` — which is why the analysis's
predictions had to be checked against this code before any of them were acted on.

Where a 13.45 ms iteration went at 128 copies, before this round's changes:

| stage | milliseconds | share of the iteration |
|---|---|---|
| environment, 128 sequential steps | 3.63 | 27.0% |
| policy and post-rollout processing | 1.45 | 10.8% |
| sixteen gradients (forward and backward) | 5.79 | 43.1% |
| gradient clip | 0.80 | 6.0% |
| optimizer | 1.77 | 13.2% |

The flat parameter buffer, the analysis's headline item, was **rejected**. It is faster at 8
copies and 14.4% SLOWER at 128, because a parameter's slice of a [copies, parameters] array is
strided, so cutting it back out is a real copy: at 128 copies the sixteen unpacks per update move
about a gigabyte, which costs more than the twenty-one kernel launches it saves. The prediction
was right about the mechanism and wrong about the sign at the size that matters.

What did move the iteration was neither of the things the analysis pointed at. Both kept changes
are unroll factors on the two scans, and the larger one is on the scan nobody had looked at:

1. **The rollout scan**, which round 2 left at 4 because its measurement could not separate 4
   from 8. Sixteen is better at every size, and 32 better still at 8 copies.
2. **The sixteen-step update scan**, which had never been unrolled at all. The profile put 62%
   of the iteration in the update stage, and emitting two steps per loop body — giving the
   compiler one step's optimizer and the next step's gradient to overlap — is the single largest
   gain of the round, worth 8.5% at 128 copies on its own.

Together, against the round-4 configuration, on the same machine in the same process:

| measurement | C=8 | C=32 | C=128 |
|---|---|---|---|
| round 4, synchronised (ms) | 7.97 | 9.73 | 13.26 |
| round 5, synchronised (ms) | **7.35** | **9.00** | **11.97** |
| change, synchronised | -7.8% | -7.5% | -9.7% |
| noise floor, synchronised (ms) | 0.20 | 0.31 | 0.61 |
| round 4, pipelined (ms) | 6.87 | 8.44 | 11.99 |
| round 5, pipelined (ms) | **5.97** | **7.63** | **10.55** |
| change, pipelined | -13.0% | -9.6% | -12.1% |
| noise floor, pipelined (ms) | 0.07 | 0.04 | 0.07 |
| rounds favouring round 5, either mode | 11/11 | 11/11 | 11/11 |

One update per batch (style A), which has no update scan and so gains only from the rollout,
synchronised: 5.50 -> 5.29, 5.88 -> 5.69 and 6.57 -> 6.32 ms at 8 / 32 / 128 copies.

The methodological finding is worth as much as the speed. The verdict rule inherited from round 4
compares two medians against the largest spread the SAME version shows across rounds, and by that
rule the rollout-unroll change was "within noise" at all three sizes. But the two versions are
timed in the same round on the same machine, so the drift that moves one moves the other:
comparing them round by round, the change won every round at every size. The statistic was hiding
a real effect, and it has been replaced with the paired one.

Accuracy. The rollout unroll is **bit-identical** — unrolling is an instruction about how many
copies of the loop body to emit, not a change to the arithmetic — and it is taken to 32 only where
that holds, so every configuration computes exactly what it did before. The update-scan unroll is
NOT bit-identical: it lets the compiler fuse across steps, so float32 accumulates in a different
order and one update stage lands 3.6e-07 away. That it is nonetheless the SAME function was
established the only way that settles such a question, by repeating the comparison in double
precision, where the two agree to 3.7e-16.

Against the analysis's floors, the 128-copy iteration is now 10.55 ms pipelined against a 6.55 ms
sum of the same matmuls measured in isolation — **1.61 times the matrix-work floor**, from 1.83
before. The remaining distance is not arithmetic: the sixteen gradients alone were 5.79 ms against
roughly 0.35 ms of arithmetic at achievable rates, so the update runs about sixteen times above
its arithmetic bound and is limited by how many separate operations the iteration issues, exactly
as the analysis concluded. Both of this round's gains came from making the compiler emit fewer,
longer-running programs rather than from doing less arithmetic, which is what that diagnosis
predicts should work.

### Throughput of the trainer as it now stands

Both rates, for each timing mode. Each copy collects `num_steps` x `n_envs` = 512 environment
steps per iteration, so the total rate is that times the copy count over the iteration time. The
last column restates the per-copy rate as the time one copy would need for a million steps.

| timing mode | copies | seconds per<br>iteration | total steps per<br>second (millions) | steps per second<br>per copy (thousands) | hours per million<br>steps per copy |
|---|---|---|---|---|---|
| synchronised | 8 | 0.00765 | 0.535 | 66.93 | 0.0042 |
| synchronised | 32 | 0.00906 | 1.809 | 56.54 | 0.0049 |
| synchronised | 128 | 0.01198 | 5.472 | 42.75 | 0.0065 |
| pipelined | 8 | 0.00624 | 0.656 | 81.99 | 0.0034 |
| pipelined | 32 | 0.00782 | 2.096 | 65.49 | 0.0042 |
| pipelined | 128 | 0.01073 | 6.107 | 47.71 | 0.0058 |

Going from 8 to 128 copies multiplies the total rate by 9.3 (pipelined) while the rate each copy
gets falls to 58% of its 8-copy value — the usual trade of per-copy latency for aggregate
throughput, and the reason the copy count is chosen from how many runs are wanted rather than from
how fast one run should be.

### Two inherited gates fail, and neither is caused by this round

Both were verified by stashing every round-five edit and re-running:

1. `test_hoist_equivalence.py` — worst relative field deviation 2.62e-05 against its own 1e-5
   threshold. The unroll change does not affect it (the hoist comparison measures 2.44e-04 on
   parameters at unroll 4, 16 and 32 alike).
2. `test_forward_fixture.py` — fails on `act_mean` at **7.80e-05** relative against a 1e-5 gate,
   comparing the JAX forward against a fixture dumped from the PyTorch twin. The fixture had to be
   regenerated because the file it reads was absent from the machine, and it fails identically
   with and without this round's changes. Worth the parent session's attention: this file's own
   notes record that quantity historically agreeing to 8.6e-07 under a mixed tolerance, and the
   actor head it comes from is the gain-0.01 layer whose outputs sit near 1e-3, where a pure
   relative measure is inflated by float32 accumulation noise — so the failure may be the gate's
   form rather than a real divergence, but the ninety-fold move from the recorded value is not
   explained by that alone.

Everything else passes: `test_jax_ppo.py`, `test_sweep_jax.py`, and the new
`test_flat_params_equivalence.py`.

## Notes

- Deviations from the letter of the task/spec, with reasons:
  - The priming PRNG label is the constant 999999937 (fold_in rejects negative ints).
  - Test-1 tolerance is the standard mixed form (atol 1e-6 + rtol 1e-5) rather than pure
    relative 1e-5: the gain-0.01 actor head produces values near 1e-3 where float32 BLAS
    accumulation noise (~1e-7 absolute) exceeds 1e-5 pure-relative. All outputs agree to
    8.6e-7 mixed error.
  - Style B's 16 optimizer steps run as one lax.scan over pre-gathered minibatches
    (spec 12 allows "unrolled or lax.scan").
- Next optimization candidates (not yet tried): donate_argnums on the TrainState in
  `_iterate` (removes a state copy per iteration), fusing the priming loop into one jit,
  jnp.compress-free minibatch gather layouts, larger N with fewer T per spec section 14.
