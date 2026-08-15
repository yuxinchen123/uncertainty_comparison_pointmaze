# Round 2 — ranked experiment lists for Module 2 (trainers) and Module 3 (end to end)

Written 2026-08-15 after reading the round-one record: `PROGRESS.md`, the seven
`progress_and_changes.md` ledgers, `extra_step_review.md`,
`report/2026-08-15-pointmaze-gpu-parallelization/report.md`,
`ppo/torch_ppo/torch_ppo_rnd.py`, `ppo/jax_ppo/jax_ppo_rnd.py`,
`ppo/research/ppo_rnd_algorithm_spec.md`, the two test suites, and
`benchmarks/{bench_train.py,profile_breakdown.py}`. Working method:
`reference_repo/autoresearch/program.md` — one change, measure, keep or revert, log a row.

## Where the time actually goes, and what that rules out

Round-one end state (serval05 H100, C=128, T=128, N=4):

| build | style B | style A |
|---|---|---|
| torch, one graph + TF32 + fused capturable Adam + target hoist + packing | 36.6 ms/iter | 24.5 ms/iter |
| jax, whole-iteration jit + donation | 20.1 ms/iter (50 iter/s) | 13.5 ms/iter (74 iter/s) |

Torch phase split from `profile_breakdown.py` (separate-graph configuration): rollout replay
20.11 ms, post-processing 14.83 ms, update replay 15.54 ms, noise and permutation refill 0.06 ms.

Two arithmetic facts decide the whole round-2 ranking.

1. **The rollout is 100 to 1000 times above its arithmetic floor.** Per copy per step the four
   networks do 371,712 multiply-accumulates (actor 17,920; critic 17,920; RND target 135,168; RND
   predictor 200,704), so one step at C=128 is 95.2 MFLOP and the whole 128-step rollout is
   12.2 GFLOP. Define throughput as $r = W / t$ with $W$ the work and $t$ the measured time:
   12.2 GFLOP in 20.11 ms is $r = 0.61$ TFLOPS, against roughly 67 TFLOPS of fp32 and 400 TFLOPS of
   TF32 tensor-core peak on this card. The rollout is not doing arithmetic; it is paying per-kernel
   fixed cost.
2. **The update is 30 to 60 times above its arithmetic floor.** One style-B minibatch step is about
   1.93 GFLOP forward, so forward plus backward over 16 steps is about 93 GFLOP; in 15.54 ms that is
   6.0 TFLOPS. The batched shapes involved ([128, 128, 256] by [128, 256, 128] for the predictor's
   widest layer) can sustain far more than that.

A static op count of `_one_step_pure` gives about 47 kernels per rollout step:

| block | kernels (estimate) |
|---|---|
| packed actor + critic trunks and heads | 13 (4 weight concatenations, 5 matmuls, 4 bias/activation) |
| action sample + log-probability | 3 |
| env `step_core` (fused, per the env ledger) | 8 |
| whiten + packed RND target/predictor | 11 |
| intrinsic bonus reduction | 2 |
| the 10 buffer writes | 10 |
| total | 47 |

Measured 20.11 ms over 128 steps is 157 us per step, so about 3.3 us per kernel — the small-kernel
floor on this card, replayed from a graph. Define the rollout time model as $t = T k \ell$ with $T$
the horizon, $k$ the kernels per step and $\ell$ the per-kernel floor. Round one drove $\ell$ down
(compile, then capture). Round two can only move $k$ and $T$.

**Consequence for the ranking.** Every idea whose mechanism is "make the arithmetic cheaper" —
bf16 above all — has a small ceiling here, and every idea whose mechanism is "run fewer kernels in
the serial chain" has a large one. That reverses the naive ordering, and it is the single most
useful thing round-one measurement bought.

**A caveat on all of it.** No number below can be decided without the noise floor and paired
comparison that the benchmark protocol still lacks (`extra_step_review.md` section 4, items 1 and
2). E1 comes before everything.

---

## Module 2, PyTorch — ranked

| # | change | estimated effect | confidence | main gate risk |
|---|---|---|---|---|
| T1 | Move critic, log-probability and RND<br>bonus out of the step loop | -9 to -10 ms<br>(-25%) | high | copy-axis reshape;<br>statistics ordering |
| T2 | Build the packed weights once per<br>iteration, not once per step | -2 to -2.5 ms<br>(-6%) | high | none (bitwise) |
| T3 | Fold the buffer writes into the<br>compiled step | -2 to -3.5 ms<br>(-7%) | medium-high | capture address<br>stability |
| T4 | One flat parameter buffer for clip<br>and Adam | -1.5 to -3 ms<br>(-6%) | medium | a view that<br>detaches a tensor |
| T5 | Compile the two scans in `_post_body` | -2 to -4 ms<br>(-8%) | medium-high | none expected |
| T6 | `max-autotune-no-cudagraphs` | -1 to -2 ms<br>(-4%) | medium | numerics change |
| T7 | Gather all 16 minibatches once | -0.5 to -1 ms<br>(-2%) | medium-high | none (bitwise) |
| T8 | bf16 trunk matmuls behind a switch | -0.5 to -1.5 ms<br>(-3%) | low | intrinsic reward<br>scale |
| T9 | Delete the redundant bootstrap<br>critic forward | -0.15 ms (-0.4%) | high | none (simplification) |
| T10 | Persistent rollout kernel | unknown; probe first | low | whole-path rewrite |
| T11 | T=32, N=16 rollout shape | -15 to -18 ms<br>(1.8x) | high | it is a different<br>algorithm |
| T12 | Draw noise and permutations inside<br>the graph | -0.06 ms (-0.2%) | high | none |

### T1 — take the critic, the log-probability and the RND bonus out of the sequential loop

**What to change.** `_one_step_pure` currently computes, per step: the actor mean, the two critic
values, the sampled action, the log-probability, the environment step, and the RND intrinsic bonus.
Only the actor mean and the environment step are on the serial path — the next observation depends
on the action, and on nothing else. The other three are pure functions of data the loop already
stores, evaluated with parameters that do not move during a rollout:

- `logp` reads only `z` and `logstd` (look at the code: it is `(-0.5*z*z - logstd - 0.5*LOG2PI).sum(-1)`,
  the mean never enters). The whole `[T, C, N]` array is one kernel over the pre-drawn `_Z`.
- `vext`, `vint` at step `t` are the critic applied to `obs`, which is written to `b["obs"][t]`
  anyway. One batched call over `[C, T*N, 4]` after the loop reproduces them.
- `r_int` at step `t` is the RND bonus on `final_obs`, written to `b["nobs"][t]` anyway. One batched
  pass after the loop reproduces it.

So the loop body shrinks to: actor trunk, action sample, `env.step_core`, and the remaining writes.
The three deferred blocks move into `_post_body` as three wide passes.

**Expected effect and why.** Using the kernel table above, the change removes roughly 25 of 47
kernels per step: the RND block (11), the whiten (part of that 11), the bonus reduction (2), the
critic half of the packed trunk plus the packed heads (about 7), the log-probability (1), and four
of the ten buffer writes. At the measured 3.3 us per kernel that is about 82 us per step, or
10.5 ms off the 20.11 ms rollout, against roughly 1 ms added back as three wide passes over 512
rows per copy. Net about -9 to -10 ms, a quarter of the whole iteration. It is the largest single
lever in this list, and it costs about 60 lines.

**How to measure.** `bench_train.py` at C = 8, 128, 1024, both styles, ABBA-paired against the
frozen current file per E1; `profile_breakdown.py` for the new phase split; an `nsys` or
`torch.profiler` kernel count of three replays before and after, which is the direct test of the
mechanism (the kernel count should fall by about half, and if it does not, the model above is wrong
and the ranking needs revisiting).

**Correctness gates it risks.**

- **Statistics ordering, the dangerous one.** The rollout's RND bonus must use the observation
  statistics as they stood *during* the rollout, and the update batch's `rnd_input` uses the
  statistics *after* `obs_rms.update`. So the deferred bonus pass must run before `obs_rms.update`
  in `_post_body`, and the update's whiten after it. That means two whiten passes and two target
  passes per iteration, not one. Reusing a single whiten silently changes the intrinsic reward the
  agent sees — a change to the algorithm that no speed test would catch. `test_capture_gpu.py`'s
  bitwise rollout comparison catches it only if the reference side is the pre-change build, so run
  that comparison once across builds before adopting.
- **Copy-axis reshapes.** Three new `permute`/`reshape` pairs between `[T, C, N]` and `[C, T*N]`
  layouts. A wrong axis order mixes copies and is exactly what `test_copy_isolation` exists for; it
  currently runs on CPU only, so add a GPU run of it to the gate.
- **Bitwise equality is not guaranteed** for the critic and RND values: the batched call has M = 512
  rows per copy where the loop had M = 4, and a different GEMM tile can associate the K-reduction
  differently. Expect agreement to about 1e-6, not bitwise. Loosen the rollout test's assertion for
  those fields to a stated tolerance, keep it bitwise for the fields that did not change, and add
  the behavioural check of E7.

### T2 — build the packed weights once per iteration

**What to change.** `actor_critic` runs `torch.cat` on the actor and critic first-layer weights and
biases and on the two critic heads, and `rnd_features` does the same for the target and predictor
first layer — every call, so 4 to 6 concatenation kernels inside every one of the 128 steps. The
packed tensors are constant for the whole rollout. Build them once per iteration (or hold them as
the parameters themselves, per `extra_step_review.md` item 1, which also shrinks the optimizer's
tensor list) and pass them into the step function.

**Expected effect.** 4 to 6 kernels of 47 per step, about 2 to 2.5 ms of the rollout.

**How to measure.** As T1. This is a five-line change; measure it on its own row so its effect is
attributable.

**Gate risk.** None: the concatenation result is identical, so the rollout capture test stays
bitwise. Note the interaction with T1 — once the critic and the RND nets leave the loop, the
actor-critic pack cannot exist there, and the +1.9% that packing bought in round one has to be
re-earned inside the update instead. Measure T1 and T2 in that order and re-check packing after.

### T3 — fold the ten per-step buffer writes into the compiled step

**What to change.** `_rollout_body` calls `b[k][t].copy_(value)` ten times per step, outside the
compiled region, so each is its own kernel. Pass the destination slices in as arguments and do the
`copy_` inside `_one_step_pure`; inductor turns a mutation of an input into a store epilogue on the
kernel that produced the value, and the separate kernel disappears.

**Expected effect.** The component view measures the ten writes at 8.55 ms per rollout in eager
form; under capture the launch part is already gone, leaving ten kernels of about 2 to 3 us, so 2
to 3.5 ms. After T1 only about six writes remain and the estimate falls to 1.5 to 2 ms.

**How to measure.** As T1, plus the kernel count.

**Gate risk.** The slices are at fixed addresses because the buffers are static, and graph capture
bakes those addresses — correct here, but it makes the capture contract stricter, so any future
code that reallocates `_bufs` breaks silently. Assert the buffer data pointers are unchanged
between capture and each replay. Arithmetic is unchanged, so the bitwise rollout test is the gate.

### T4 — one flat parameter buffer for the clip and Adam

**What to change.** `_clip_per_copy_and_step` walks 19 trainable tensors twice (a squared sum and a
scale multiply) and then steps Adam, in every one of the 16 optimizer steps of style B. Allocate one
contiguous `[C, P]` parameter buffer and make each of the 19 tensors a view into it, with a matching
flat gradient buffer assigned to each `p.grad`. The clip becomes one reduction and one multiply, and
Adam becomes one elementwise kernel.

**Expected effect.** About 60 kernels per optimizer step become about 4, so roughly 900 kernels per
iteration disappear. The component view puts clip plus Adam at 12.03 ms eager per iteration; under
capture the residual is roughly 1.5 to 3 ms. This is the largest update-side lever, and it is the
one that follows from the update's measured 6.0 TFLOPS: the update is not GEMM-bound either.

**How to measure.** As T1, restricted to the update phase (`bench_train.py` split mode reports the
update replay separately; take the phase split from a profiled run rather than from extra
synchronizations, per `extra_step_review.md` section 4 item 3).

**Gate risk.** The real hazard is a view whose storage is not the flat buffer, which quietly removes
a network from the optimizer while every test still passes. Assert at construction that every
trainable tensor's `data_ptr` lies inside the flat buffer and that the 19 spans tile it exactly.
The norm reduction changes float association (one reduction over `[C, P]` instead of a sum of 19),
so the captured-update test's 1e-5 tolerance is the right gate, not bitwise; add E7.

### T5 — compile the two scans in `_post_body`

**What to change.** `_post_body` is not compiled at all. It contains a 128-iteration Python loop for
the intrinsic filter and a 128-iteration reverse loop for the two-stream GAE with two buffer writes
each — 256 iterations of tiny eager kernels on `[C, N]` tensors. Extract each scan as a pure
function of its buffers and compile with `fullgraph=True, dynamic=False`. Because the loops are
unrolled at trace time, inductor sees a chain of dependent elementwise operations of identical
shape and fuses long runs of them into single kernels.

**Expected effect.** The component view measures the filter scan at 1.15 ms and the GAE scan at
2.82 ms; the flatten, whiten and statistics blocks add about 0.9 ms. Both scan rows are measured
with a simplified body — `profile_breakdown.py` times one GAE stream with a two-term recursion and
no per-step buffer write, then doubles it, and times the filter as an in-place multiply-add — so
the real scans cost more than those numbers and the estimate below is a floor. Fusing the two scans
should recover 2 to 4 ms. The one-graph mode already removed the launch gaps (14.83 ms of eager
post-processing collapses to roughly 5 ms of device time inside the graph); this row attacks the
device time that remains, which capture cannot touch. Fixing the two profiler rows to time the real
bodies is a prerequisite, and belongs in E1.

**How to measure.** Phase-level timing from a profiled replay; the whole-iteration ABBA is the
decision.

**Gate risk.** Keep the float64 statistics updates and the in-place writes into `_U` outside the
compiled region at first — mutation of attribute tensors and a float64 reduction are the two things
most likely to force a graph break under `fullgraph=True`. Fusing an elementwise chain does not
change arithmetic, so the captured-update test should stay bitwise; if it does not, that is the
signal that inductor reassociated something and the row needs the E7 gate.

### T6 — `max-autotune-no-cudagraphs`

**What to change.** One keyword on the two `torch.compile` calls in `PPORND.__init__`. Plain
`max-autotune` must not be used: its cudagraph trees collide with the manual capture.

**Expected effect.** The reference campaign measured +5.0% under capture. The specific hope here is
narrower and better founded: with autotuning on, inductor may pick Triton GEMM templates with fused
bias-and-activation epilogues for the tiny batched shapes, which removes kernels rather than making
them faster — the mechanism this workload actually responds to. Estimate -1 to -2 ms.

**How to measure.** As T1, and report cold-compile time separately from steady state. Pin
`TORCHINDUCTOR_CACHE_DIR` to serval05 local disk so the autotune cost is paid once.

**Gate risk.** Kernel selection changes float association, so this is not a bitwise change against
the previous build. It also puts the cross-framework fixture test
(`ppo/jax_ppo/tests/test_forward_fixture.py`, currently agreeing to 8.6e-7) at risk if the fixture
is regenerated from an autotuned build. Gate with E7 and state the tolerance. Watch the compile
time: the round-one attempt at forcing fusion by raising inductor's realize thresholds ran over 40
minutes without producing a kernel, so set a hard timeout and treat an overrun as a discard.

### T7 — gather all 16 minibatches once

**What to change.** `_update_body_captured` gathers 9 fields inside each of the 16 minibatch steps,
so 144 gather kernels per iteration. The jax build already gathers all minibatches up front in one
batched `take_along_axis`; do the same here, producing `[16, C, mb, ...]` tensors before the loop.

**Expected effect.** The component view puts minibatch gathers at 1.11 ms; batching them should
recover about half to two thirds. Memory grows by one copy of the batch (about 4 MB per copy-block
at C=128), which is nothing here.

**Gate risk.** None arithmetically — the same rows in the same order. The permutation buffer
`_perm` must keep its current shape and refill semantics so the style-B permutation draw stays in
step with the eager path (`test_capture_gpu.py` relies on that parity).

### T8 — bf16 in the trunk matmuls, behind a switch

**What to change.** Autocast only the matmuls of the actor, critic, RND target and RND predictor.
Environment recursion, whitening, `BatchedRMS` (already float64), the intrinsic filter, GAE, the
losses, the clip and Adam stay fp32.

**Expected effect: small here, and this is the item most likely to be over-ranked.** The reference
campaign measured the same change at -14.7% when launch-bound and +52.6% when device-bound, and
concluded that precision pays only where the stack is device-bound. Our numbers say we are in the
first regime, not the second: the update's matmuls account for roughly 1 ms of its 15.54 ms and the
rollout's for well under 1 ms of its 20.11 ms, so the whole reachable prize is about 2 ms even if
bf16 halved every GEMM. Estimate -0.5 to -1.5 ms. It becomes somewhat more attractive *after* T1,
because T1 turns the per-step RND work into one wide pass over 512 rows per copy, which is the
shape bf16 can actually help.

**How to measure.** Only after T1 and after the enqueue column of E1 shows where the device is
busy. Measure the two phases separately; a whole-iteration number will hide a rollout regression
behind an update gain.

**Gate risk.** The highest of any row. The RND bonus is a squared difference of predictor and target
features, and it feeds a running standard deviation that sets the exploration signal's scale, so a
bf16 predictor changes what the agent explores, not only how fast. Keep the target and predictor
fp32 in the first variant even though they hold most of the arithmetic, measure that, and only then
try them in bf16 with the full E7 behavioural gate. Ship it as a config field, not as a default —
the reference campaign shipped fp32 with TF32 matmuls after measuring bf16 favourably, for this
reason.

### T9 — delete the redundant bootstrap critic forward

**What to change.** `_post_body` runs `critic_values` over all `T*N` stored next observations on top
of the per-step critic forwards. Under `continuing_task=True` nothing terminates, every environment
truncates at step 400, so `nobs_buf[t]` equals the entry observation of step `t+1` for about 399
rows in 400: `vext_next` can be built by shifting the rollout's own value buffer, with a fresh
evaluation only at `t = T-1` and at reset steps. Two deletions come with it: `boot_mask` is
identically zero, so the `(1 - boot_mask[t])` multiply and the whole `b["term"]` buffer are dead.

**Expected effect: 0.4%, and that is the point of listing it.** The component view measures the
bootstrap forward at 0.15 ms — one wide GEMM, already cheap. `extra_step_review.md` ranked this
alongside the RND target hoist on the strength of the reference campaign's +17.61%, but our own
profile caps it at 0.15 ms, and the replacement (a shift plus a masked correction) adds kernels of
its own, so it could come out negative. Rank it as a simplification: adopt on non-inferiority under
the deletion clause, not as a speed row. If T1 is adopted it becomes nearly free — a single critic
pass over `obs_buf` plus a shifted read gives both `vext_buf` and `vext_next`.

**Gate risk.** The reset-step correction is the whole of the correctness. Test it by forcing a short
episode limit so resets are frequent and comparing GAE outputs against the current build.

### T10 — persistent rollout kernel

**What to change.** One kernel that owns a copy's environment rows for the whole horizon, state in
registers, weights in shared memory, no launch per step.

**Why it is ranked here and not higher.** It is the biggest theoretical prize (the rollout runs at
0.6 TFLOPS against a 400 TFLOPS card) and the biggest build. It also has a hard resource
precondition that T1 changes: with the RND networks inside the loop, one copy's weights are about
150k floats, or 600 KB, which does not fit in an H100 SM's 228 KB of shared memory, so a per-copy
persistent block would stream weights from L2 every step and 128 copies would not fit in L2 either.
After T1 the loop body carries only the actor — about 4.5k floats, 18 KB per copy — which fits
comfortably. **T1 is what makes T10 possible; do not attempt T10 first.**

**How to decide before building.** Run the reference campaign's written criterion on our captured
one-graph iteration at C = 128 before writing any kernel: define $h$ as the device-idle share of one
replayed iteration, $g$ as the matmul share of kernel time, $e$ as the elementwise share, and $\rho$
as the median ratio of best Triton to best cuBLAS time over the iteration's matmul shapes. Go only
if $h > 0.15$, or $e > 0.40$ with those kernels under half of measured copy bandwidth, or
$\rho < 0.90$ on at least half the shapes; the achievable ceiling is $1 / (\rho g + e/2)$. Our
verdict is likely to differ from the reference's no-go: they measured $h = 2.79$% with 26,500 large
kernels per update; we have about 6,000 kernels per rollout each doing well under a microsecond of
arithmetic, so $h$ here could be very large. Write the criterion down, then measure. This probe is
E4 in the end-to-end list.

### T11 — the T = 32, N = 16 rollout shape

**What to change.** Nothing structural: the same 512 rows per copy per iteration as 32 sequential
steps of 16 environments instead of 128 steps of 4. It is a labeled algorithm variant (the GAE
horizon becomes 32), blessed by spec section 14 on the condition that it is recorded and never
silently swapped in.

**Expected effect.** The per-step kernel count is unchanged and the per-kernel floor dominates, so
the rollout should fall by nearly the full factor of 4, from 20.11 ms to about 5 to 6 ms; the two
`_post_body` scans shorten by the same factor. The update is unchanged. That predicts about 20 ms
per iteration for style B, a 1.8x — which is exactly what jax measured for the same variant (1.9x
style B, 3.0x style A). The agreement between the prediction and the jax measurement is the reason
this row carries high confidence without having been run in torch.

**How to measure.** `bench_train.py` with `num_steps=32, n_envs=16`, both styles, C = 8 to 1024, and
a learning run at C = 8 to confirm the shorter GAE horizon does not cost reward — the campaign
already has the T = 128 curve to compare against.

**Gate risk.** It is a different algorithm. Both implementations must move together or not at all,
and every reported number must carry the shape. The correctness suites are shape-agnostic and should
pass unchanged.

### T12 — draw the noise and the permutations inside the graph

Measured at 0.06 ms per iteration, 0.16%. Torch registers the default generator at capture, so both
draws can move into `_iteration_body` and be redrawn per replay. Do it for the simpler capture
contract — it removes the last host work from the iteration — not for the time.

---

## Module 2, JAX — ranked

The jax build is already 1.8x faster than the torch build at the same settings, and it is a single
jitted program with donation, so the levers differ: there is no capture to add, and the wins are in
what XLA is asked to compile.

| # | change | estimated effect | confidence | main gate risk |
|---|---|---|---|---|
| J1 | Same hoist as T1, out of the<br>rollout scan | -20 to -30% | high | statistics ordering |
| J2 | `unroll` sweep on the four scans | -5 to -25% | medium-high | compile time,<br>memory |
| J3 | Confirm or enable XLA command<br>buffers | 0 to -30% | medium | none |
| J4 | Triton GEMM autotuning flags | -3 to -10% | medium | numerics change |
| J5 | Audit for float64 leakage | 0 to -50% if<br>a leak exists | medium | none |
| J6 | Flat parameter vector for clip<br>and Adam | -3 to -8% | medium | pytree rewrite |
| J7 | bf16 dot inputs with fp32<br>accumulation | -2 to -6% | low | intrinsic reward<br>scale |
| J8 | Weight packing (the torch keep) | 0 to +11% worse | low | none |
| J9 | T = 32, N = 16 | already measured<br>3.0x / 1.9x | done | labeled variant |

### J1 — hoist the critic, log-probability and RND bonus out of the rollout scan

Identical reasoning to T1 and identical hazard list. In `_iterate_impl` the scan body computes
`_critic_values`, `_actor_mean`, the action, `logp`, the environment step, the whiten and
`_rnd_features` — of which only `_actor_mean` and `self.env.step` are on the serial path. Move the
other three out as three wide operations on the stacked `[C, T*N, ...]` buffers the scan already
returns. XLA rewards this more than torch does, because a 512-row-per-copy dot is a shape its
autotuner can do something with while a 4-row one is not.

The statistics-ordering hazard is the same and is if anything easier to get wrong here, because
`obs_rms_old` is already captured explicitly at the top of `_iterate_impl` — the deferred bonus pass
must keep using `obs_rms_old` while the update batch's `rnd_input` uses the updated statistics.
Gates: `tests/test_jax_ppo.py` (same-seed bitwise, copy isolation, falling intrinsic reward) plus
the cross-framework fixture, which will move if torch adopts T1 and jax does not, or the reverse —
so land T1 and J1 in the same session and regenerate the fixture once.

### J2 — sweep `unroll` on the four scans

None of the six `lax.scan` calls passes `unroll`. Sweep {1, 2, 4, 8} one scan at a time on the
rollout scan, the intrinsic filter scan, the GAE scan and the style-B minibatch scan. The reference
campaign measured 1.89x then a further 1.11x for this knob on an environment scan and +4.67% on a
trainer scan whose body was a transformer; our body is a tiny multilayer perceptron, so the
environment-track prior is the better one. Unrolling also lets XLA overlap the parts of adjacent
steps that are genuinely independent — which, after J1, there are fewer of, so sweep it again after
J1 rather than inheriting the verdict. Watch compile time and peak memory, both of which the
reference saw rise with the knob.

### J3 — confirm or enable XLA command buffers

XLA:GPU can wrap fusions and library calls into CUDA graphs (`--xla_gpu_enable_command_buffer`).
This is the jax-side equivalent of the manual capture that bought torch a factor of several, and a
program of thousands of sub-microsecond kernels is exactly the case it exists for. The first action
is to find out what the installed jax already does by default, from
`XLA_FLAGS=--xla_dump_to=...` plus the executable's command-buffer annotations; only then try
`FUSION,CUBLAS,CUBLASLT,CUSTOM_CALL` and the minimum graph size. If it turns out to be on by
default, that fact alone explains most of the 1.8x gap to torch and belongs in the report.

### J4 — Triton GEMM autotuning

`--xla_gpu_enable_triton_gemm=true`, `--xla_gpu_triton_gemm_any=true`,
`--xla_gpu_autotune_level=4`. Same mechanism as T6: fused epilogues on tiny batched dots remove
kernels. Report cold-compile time separately; autotuning at these shapes is not free.

### J5 — audit for float64 leakage

`jax.config.update("jax_enable_x64", True)` is set for the running statistics, which makes float64
the promotion target for any literal or operation not explicitly typed. The code is careful (`F32`
almost everywhere), but one leaked float64 in the scan body would cost about 2x on this card and
would be invisible in every correctness test, since float64 is *more* accurate. Dump the optimized
HLO and count f64 operations outside `rms_update`. Zero effect if clean; that is the point of
running it early and cheaply.

### J6 — flat parameter vector

`_clip_per_copy` sums over 19 pytree leaves and `_adam_step` runs four `tree_map` passes over 19
leaves, 16 times per iteration in style B. A single `[C, P]` vector with the networks as slices
makes the clip one reduction and Adam four elementwise operations. XLA fuses elementwise work across
leaves better than torch does, so expect a smaller win here than T4 gives in torch — which is itself
worth recording, since it is the same "do not port a decision across frameworks" rule the reference
campaign states.

### J7 — bf16 dot inputs with fp32 accumulation

The reference measured jax's default matmul-precision knob as a null (+0.07%) but explicit bf16
inputs with `preferred_element_type=jnp.float32` at +6.7 to +19.6%. Our arithmetic share is small
for the same reason as in torch, so cap the expectation at a few percent, and keep the same hard
limit: the intrinsic reward path and the running statistics stay fp32.

### J8 — weight packing

The torch build keeps same-input GEMM packing (+1.9% style B). The reference measured the analogous
change at -11.5% under XLA. Measure it once so the report can state a measured verdict for both
frameworks instead of an inherited one, and expect it to be discarded.

---

## Module 3, end to end — ranked

The torch end-to-end path is the trainer (the environment step is already inside the captured
iteration graph), so this list is about deciding, measuring and re-checking rather than about new
code paths.

| # | action | why it is here | cost |
|---|---|---|---|
| E1 | Noise floor, ABBA pairing,<br>enqueue column | Nothing below 3% can<br>be decided without it | half a day |
| E2 | Environment share by<br>substitution | Caps all further<br>Module 1 work | 2 hours |
| E3 | Re-measure the ladders after<br>T1 to T5 | The knee and the phase<br>split both move | 2 hours |
| E4 | Megakernel go/no-go probe | Decides T10 before a<br>multi-week build | half a day |
| E5 | Whole-rollout environment<br>kernel | Only if E2 justifies it | days |
| E6 | Fresh-process memory ladder | A headline number is<br>currently in-process | 2 hours |
| E7 | Behavioural equivalence gate | TF32 already shipped<br>with no such gate | half a day |
| E8 | Re-measure the CUDA-environment<br>pairing after the trainer shrinks | The environment's share<br>rises as the trainer falls | 1 hour |

### E1 — the measurement protocol, first

Round one's verdicts compare absolute numbers from different sessions ("226 to 45.6 ms"), which is
fine for factors of five and useless for the 2 to 8% rows above. Adopt three things before running
any experiment in this note:

1. A null run: the candidate is a byte-identical copy of the reference, five ABBA pairs, reporting
   the per-cell standard error. The reference campaign's training bench read 1.2 to 1.8%, from which
   they concluded keeps need about +3% to clear the decision rule. We do not have our own number.
2. The decision statistic: interleaved ABBA pairs in one session, score `mean(log ratio)` with its
   standard error, keep only if `mean - 3*SE > 0` and the gain is at least 1% and no cell regresses
   more than 10%. A change that only deletes code keeps on non-inferiority.
3. The enqueue column: host time from region start to just before the closing synchronization,
   recorded beside wall time in every row. Enqueue near wall means launch-bound; enqueue far below
   wall means device-bound. That single number decides whether T6 and T8 are worth running at all,
   and it is the one instrument this project does not have.

Two harness bugs to fix in the same pass: `bench_train.py` synchronizes once per iteration inside
the timed loop in one-graph mode and three times in split mode (the protocol is exactly two, at the
region boundaries — take the phase split from a profiled run), and benchmarks should import a frozen
copy extracted from git rather than the working tree, because a queued session imports its files
when it starts, not when it was queued.

### E2 — the environment's share of a training iteration

Add a `_model_only_step` that runs the identical policy, value and RND forwards but returns a
handed-in next state instead of calling `env.step_core`, and time `_rollout_body` with each. The
difference is the environment's share, attributable rather than argued.

The existing evidence already suggests the answer: swapping the entire environment implementation
for the fused CUDA kernel — 24x faster standalone at these batch sizes — moved the iteration from
36.6 ms to 35.6 ms, about 3%. That bounds the environment at roughly 3 to 5% of a training
iteration, which in turn bounds every further environment optimization at a few percent end to end.
Measure it properly, then let the number decide E5. This is the measurement that retired the whole
cross-stack pairing question in the reference campaign, for the same reason.

### E3 — re-measure the ladders after the trainer changes

The copy-count ladder (C = 8 to 32,768), the per-copy and total throughput curves, the phase split
and the before/after table in the report are all functions of the trainer that produced them. T1
alone moves the rollout by a quarter of the iteration, which moves the knee. Regenerate every table
and figure from the result JSONs through `report/.../code/make_report.py` rather than editing the
report, per the markdown-editing rule.

### E4 — the megakernel probe

The criterion, written before the measurement, is in T10. Run it on the captured one-graph iteration
at C = 128: 30 timed replays plus 3 profiled replays, kernel device time summed against replay wall
time, kernels taxonomised into matmul, elementwise and copy, and $\rho$ read out of a max-autotune
log. Record the verdict whichever way it falls; a written no-go that saves a multi-week build is as
valuable as a go.

### E5 — whole-rollout environment kernel

The reference campaign's largest single number (20.57x over a per-step fused kernel, then 1.87x more
from capturing the rollout), and our CUDA environment is still at their version 0. But E2 caps its
end-to-end value at a few percent, so it is justified by the Module 1 deliverable curves — the
throughput and knee lines the task asks for per implementation — rather than by the training
throughput. Rank it accordingly: a deliverable, not a trainer optimization. The geometry rule comes
with it: size the grid to a few programs per multiprocessor before sizing the tile. The deferred
E3 row in the CUDA ledger (distance-only tournament, coded and CPU-exact but never measured on the
GPU) should be run first since it is already written.

### E6 — fresh-process memory ladder

"32,768 fits, 65,536 out of memory" comes from an in-process sweep over eleven copy counts, and an
allocator pool that never shrinks makes late out-of-memory verdicts unreliable — the reference
campaign turned 7 such verdicts into artifacts by re-measuring one cell per process. Re-measure the
top of the ladder one cell per fresh process with `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
and refine between 32,768 and 65,536. The maximum copy count is a number the report states.

### E7 — the behavioural equivalence gate

There is a bitwise or 1e-5 gate for exact refactors and nothing at all for changes that alter
numerics — yet TF32 is already in the production configuration on the strength of a +9% speed
number. Build the second tier now: 3 seeds times a fixed short run at C = 8, comparing median final
extrinsic reward and median coverage against a cached fp32 reference distribution with a stated
band. Run it retroactively on TF32, then as the gate on T6, T8, J4 and J7. Add
`assert_global_state()` to `PPORND` before two trainers ever share a process:
`torch.set_float32_matmul_precision("high")` is process-global and set in `__init__`, so a paired
comparison built in one process would measure TF32 against TF32.

### E8 — re-measure the pairing after the trainer shrinks

The CUDA environment is worth 3% of a 36.6 ms iteration. If T1 to T5 land, the iteration is nearer
20 ms and the same absolute environment saving is worth 6 to 8%. Every pairing verdict in the report
is relative to the trainer that measured it, and that has to be said in the report rather than
implied.

---

## The single-style build question

**The question.** Does a build containing only the on-policy path (style A, one full-batch update)
run faster than a build that also contains the epoch-and-minibatch path, given that the two are
already separate functions with no runtime branch?

**What the code does today.** The style is resolved at construction:
`update = self.update_full_batch if cfg.update_style == "full_batch" else self.update_epoch_minibatch`,
and `_update_body_captured` branches on `cfg.update_style` once, at capture time. Inside `_losses`,
`style_a` is a Python bool, so `torch.compile` specializes on it and the untaken branch never enters
any graph. In jax the two update functions are separate and only the selected one is ever traced.
So the structural requirement the task statement asked for is already met, and the question is
whether anything *else* costs.

**Mechanisms that could produce a real difference, assessed.**

1. **Python attribute lookups and dispatch.** Nil in the shipped path. In one-graph mode a whole
   iteration is one `graph.replay()`; there is no per-step Python at all. In the split path the
   style is resolved once per training run.
2. **Compile cache pressure and guard chains.** Dynamo keeps one cache entry per specialization, so
   a process that called `_losses` with both values of `style_a` would evaluate a two-entry guard
   chain. But a single-style build already only ever passes one value, and guards are evaluated at
   the Python call — which, under capture, happens three times during warmup and never again.
3. **Graph memory pool size.** A CUDA graph allocates from a private pool. A build that captured
   both an A graph and a B graph would hold two pools and more reserved memory. Today only one
   update graph is ever built per trainer, so there is nothing to reclaim. This is the mechanism
   that *would* matter if a future build captured both.
4. **Inductor autotuning choices.** Kernel selection depends on the shapes actually compiled, and
   the taken path's shapes are identical in both builds. No mechanism.
5. **Dead data movement — the one real difference.** Style A's loss never reads `vext_old`, and
   style A never draws a permutation. A style-A-only build can delete the `_U["vext_old"]` buffer
   and the `_post_body` flatten-and-copy that fills it (one kernel over `[C, 512]`), and the `_perm`
   buffer with it. The permutation draw is already skipped at run time by
   `if cfg.update_style != "full_batch"`, and was measured at 0.06 ms per iteration for both draws
   together. So the honest prediction is that any measurable difference comes from deleting one
   buffer write, and is well under 0.5% of a 24.5 ms style-A iteration.

**Predicted answer: no.** Removing the other style's code does not make the remaining path faster;
removing the other style's *data movement* saves one kernel. The experiment below is designed to
separate those two claims and to state how small a difference it can rule out.

**The experiment.**

1. Build three variants, each frozen from git into its own directory and imported by path (never
   from the working tree):
   - **P** — production, both styles present, run with `update_style="full_batch"`.
   - **A** — style-A-only: delete `update_epoch_minibatch`, the minibatch branch of
     `_update_body_captured`, the `_perm` buffer and its refill, the `style_a` parameter and the
     `else` branch of `_losses`, the `_U["vext_old"]` buffer and the `_post_body` line that fills
     it, and the style-B configuration fields.
   - **A-keep** — identical to A except that `vext_old` and `_perm` are still allocated and still
     filled. This is the variant that isolates "does the mere presence of the other code path cost
     anything" from "does deleting its dead data movement help".
2. Run under the H100 lock, one process per measurement, alternating P A A P over at least five
   pairs, then A-keep A A A-keep over five more. Each process: 20 untimed warmup iterations, 200
   timed iterations, exactly two synchronizations around the timed block, at C = 8, 128 and 1024.
3. Record per row: wall time, enqueue time, peak and reserved memory, cold-compile time, the dynamo
   compile-counter delta across the timed region (must be zero), and the allocation-retry count.
4. Decision statistic per E1: `mean(log(t_P / t_A))` with its standard error over the pairs; the
   claim "A is faster" survives only if `mean - 3*SE > 0`. Report the null run's standard error
   beside it so the reader can see what size of effect the experiment could have detected.
5. Run the two mechanistic checks, which settle the question more directly than the timing does:
   - **Kernel sequence.** Profile three replays of each build and diff the ordered kernel-name list.
     If P and A-keep produce identical sequences, no timing difference between them can be caused by
     the presence of the other code path, and any residual is noise.
   - **Generated code.** Dump inductor's output with `TORCH_LOGS=output_code` for both builds and
     diff the Triton source of the taken path. Byte-identical source is a direct answer to "does the
     other path change codegen", and it does not depend on the noise floor at all.
6. Repeat the same three-variant design in jax, where the mechanism differs: XLA compiles
   `_iterate` including the selected update into one module, so the two builds produce genuinely
   different HLO — but again only the taken path exists in either. Diff the optimized HLO instead of
   the Triton source.

**What to report either way.** If the difference is under the noise floor, say so with the floor
attached, and record the two asymmetries that are real regardless of timing: the both-styles build
pays extra warmup compile time and holds a second set of compiled artifacts, and a build that ever
captured both update graphs would hold a second graph memory pool. If A does come out faster, the
A-keep row says immediately whether the cause is the deleted buffer write or something else, and
"something else" would be a finding worth chasing.

---

## Ordering, interactions, and things not to repeat

**Suggested order.** E1, then T1, T2, T3 (rollout kernel count), then E2 and E3 (re-measure), then
T5, T4, T7 (post-processing and update kernel count), then T6, then E4 and E7, then T8 and T11 as
labeled or gated variants. In jax: J5 and J3 first (both cheap and both potentially explanatory),
then J1, then J2, then J6, J4, J7, J8.

**Interactions worth stating before anyone measures.**

- T1 removes the actor-critic pack from the rollout, so the +1.9% packing keep from round one has to
  be re-earned inside the update. Re-measure packing after T1 rather than assuming it survives.
- T1 makes T10 physically possible (18 KB of weights per copy instead of 600 KB) and makes T8 more
  attractive (wide passes instead of 4-row ones). It also makes T9 nearly free.
- T3's estimate shrinks after T1, because four of the ten writes disappear with the deferred blocks.
- J2's verdict must be re-taken after J1: unrolling pays differently once the scan body is small.
- Every row's percentage is against the current 36.6 ms; adopting several shrinks the denominator,
  so report absolute milliseconds beside percentages in the ledger.

**Do not repeat these.** Round one measured them; each has a recorded reason.

- Raising inductor's realize thresholds to force one giant kernel: over 40 minutes of Triton compile
  at 0% GPU with no kernel produced.
- Capturing the eager kernel sequence instead of the compiled one: slower, because replaying
  thousands of tiny unfused kernels is kernel-time bound.
- `mode="reduce-overhead"` automatic capture: silently skipped capture; manual capture with static
  buffers was both faster and verifiable.
- Cross-framework pairings (jax environment with torch trainer, or the reverse): the dlpack boundary
  costs 6.2 to 7.0 ms per environment step on top of a 70 to 260 us native step, and breaks both
  capture and scan fusion.
