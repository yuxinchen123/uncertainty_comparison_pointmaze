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
