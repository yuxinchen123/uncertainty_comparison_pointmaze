# torch_ppo — experiment ledger (autoresearch style)

Goal metric: milliseconds per training iteration (T=128 steps x N=4 envs per copy, style B =
4 epochs x 4 minibatches) on the serval05 H100, and env-steps/second; correctness gated by
tests/test_torch_ppo.py (CPU) and tests/test_capture_gpu.py (GPU). Bench JSONs in
`../../benchmarks/results/` (bench_train.py).

| # | change | result (C=128 unless noted) | verdict |
|---|---|---|---|
| 0 | v0 eager: batched [C,...] params, baddbmm forwards, per-copy statistics/clip, sum-over-copies loss, two update styles | 777 ms/iter (rollout 680 + update 95); FLAT from C=8 to C=128 — 128 copies cost the same as 8. 8.4e4 env-steps/s. Tests: same-seed bit-identical, copy isolation, predictor learns — PASS both styles | baseline |
| 1 | torch.compile the fused per-step rollout body + fused Adam | 226-231 ms/iter (rollout 118, update 95-113) — 3.4x | superseded by 3 |
| 2 | reduce-overhead mode on the per-step body | 189 ms/iter (rollout 100) — cudagraph partially skipped (warning) | superseded by 3 |
| 3 | whole-rollout CUDA-graph capture of the EAGER body (pure step_core, in-place RMS, static buffers, pre-drawn noise) | rollout 187-211 ms — WORSE than per-step compile: replaying thousands of tiny unfused eager kernels is kernel-time bound | superseded by 4 |
| 4 | capture the COMPILED per-step function (fused kernels x 128 in one replay, zero python) | rollout 35-40 ms. Capture bitwise-equal to uncaptured compiled step (test_capture_gpu) | KEEP |
| 5 | capture the whole UPDATE phase (capturable+fused Adam with tensor lr, static input buffers, identity-perm build so the graph build consumes no RNG, compiled loss fwd/bwd, in-graph per-copy clip) | update 125 -> 16.7 ms; TOTAL 45.6 ms/iter = 1.44e6 env-steps/s at C=128 (17x vs baseline; C=8: 32.8 ms). Captured update BITWISE equal to eager update (worst param diff 0.0, both styles) | KEEP |
| 6 | one-graph iteration capture + TF32 (see e2e ledger rows 1-2) | 45.6 -> 37.9 ms/iter (C=128, style B); style A 26.9 ms | KEEP |
| 7 | hoist the frozen RND target features out of the minibatch loop (computed once per iteration in _post_body, gathered per minibatch like any batch field) | style B C=128: 37.9 -> 37.3 ms (+1.6%); style A: 26.9 -> 25.2 ms (+6.7%). All GPU capture tests still bitwise-pass | KEEP || 8 | same-input GEMM packing (actor+critic trunks share one layer-1 GEMM and one packed critic-head GEMM; RND target+predictor share one layer-1 GEMM; weights concatenated at forward time so params/optimizer/init stay untouched) | style B C=128: 37.3 -> 36.6 ms (+1.9%); style A: 25.2 -> 24.5 ms (+2.9%); C=8: 27.2 -> 26.4. GPU capture tests still bitwise-pass | KEEP — FINAL production config: one-graph + TF32 + fused capturable Adam + target hoist + packing = 21.2x vs eager baseline at C=128 |
| 9 | ROUND 2. Answer to the task statement's question "does presenting only the on-policy path help?": an anchored source stripper builds a copy of the trainer containing ONLY the on-policy path (epoch/minibatch method, the loss function's clip branch and its parameter, the captured body's other branch, the permutation buffer and the style dispatch all deleted), verified BITWISE IDENTICAL to the full build after 3 iterations, then timed against it in process-level ABBA order | C=8: 18.55 vs 18.54 ms; C=128: 24.50 vs 24.47 ms — a 0.05-0.08% difference against a 0.01 ms noise floor. NO meaningful effect: the two styles are already separate compiled functions, so the captured graph contains only the selected path either way and the other path's presence costs nothing at run time | measured; no code change (the full build keeps both styles) |
| 10 | ROUND 2. Compile the post-rollout body (`compile_post`): its intrinsic-reward filter and its two GAE scans are python loops over T, so eager they are a few hundred tiny kernels | C=128: 36.83 -> 32.72 ms (+11.2%); C=8: 26.61 -> 22.79 ms (+14.4%), against a 0.10 ms noise floor (paired ABBA, separate processes). Equivalence tested in ISOLATION (both bodies run from byte-identical inputs, statistics snapshotted and restored): worst relative deviation 2.5e-7 | KEEP |
| 11 | ROUND 2, the big one. Hoist the critic forward, the log-probability and the RND bonus OUT of the T-step rollout loop into single wide passes in the post body. Each is a pure function of data the loop already stores and of parameters that do not change during a rollout: the log-probability needs only the action noise and logstd (not the action mean), the critic needs only the observations, the bonus only the true next observations. The loop keeps only what is genuinely sequential — the actor forward that produces the action, and the environment step. The two critic passes (on-step values and bootstrap values) also merge into one call on the concatenated rows | C=128: 36.79 -> 21.23 ms (+42.3%); C=8: 26.57 -> 14.73 ms (+44.6%), noise floor 0.01 ms, paired against the predecessor GIT REVISION so the comparison is like for like. Exactness: an isolated iteration (both builds starting from identical state) deviates 8.5e-7 relative, purely the wider GEMM's accumulation order; parameters after 3 iterations differ by 1.2e-7 | KEEP |
| 12 | ROUND 2. Fold the per-step buffer writes INTO the compiled step (the six slices are passed as arguments and written inside, so each write fuses into the kernel that produces the value instead of being its own copy kernel) | C=128 against the same pre-round-2 revision: 20.23 ms versus the hoist-only 21.23 ms, so about +4.7% on top of the hoist. The two paired runs are comparable: their shared baseline measured 36.79 and 36.82 ms minutes apart, against noise floors of 0.01 and 0.03 ms. Hoist-equivalence and both capture tests still pass | KEEP |
| 13 | ROUND 3, new capability. Learning-rate sweep across copy groups (`sweep_config(rates, copies_per_rate)`): a per-copy rate vector, a hand-written batched Adam with torch's formula (torch.optim.Adam cannot express a per-copy rate without giving up the batched layout), paired seeding so copy k of every group starts from the same weights and environments, and one shared annealing factor. See SWEEP.md | Tests: uniform sweep reproduces the plain run (3e-7), zero-rate group frozen exactly while others train, changing one group's rate leaves other groups BITWISE identical, paired and distinct seeding both verified, and the same under graph capture | KEEP |
| 14 | ROUND 3. Compile the per-copy Adam (written out it is ~6 operations per parameter tensor, which is many small kernels next to torch's fused optimizer) | 512 copies, 4 rates: sweep iteration 60.5 -> 44.2 ms, so the sweep now costs +1.0% against a uniform-rate run of the same size instead of +38%. Two defects of mine surfaced and were fixed in the process: `self._adam_t += 1` rebinds the attribute to a new tensor that a captured graph never sees, and mutating `p.data` inside a compiled region does not reliably alias the parameter. The capture test (zero-rate group must stay bitwise frozen) caught both | KEEP |
| 15 | ROUND 3 scaling study: three measurements of how a sweep's cost grows — rate count at fixed copies per rate, copies per rate at fixed rate count, and rate count at FIXED total copies (the decisive one) — plus fused-versus-separate at 2/4/8/16 rates | The rate count is FREE: at 2048 copies split into 1..128 groups the iteration time is flat (143.1-145.0 ms, 1.3% spread, byte-identical memory), measured in both ascending and descending group order. The two growth studies land on the same curve to 0.3 ms out of 274, and both sit on the uniform-rate (non-sweep) curve. 16 rates x 128 copies = 2048 copies: 144 ms/iteration, 7.29e6 env-steps/s, 3560 per copy, 890 per env, 6.3 GB, 0.78 h for 10M steps per copy. Fusing beats sequential groups by 1.43x / 1.84x / 2.11x / 2.23x at 2 / 4 / 8 / 16 rates | recorded |
| 16 | ROUND 3: separate the two things a sweep changes. Three arms matched in total copies, ABBA in separate processes: uniform scalar rate with torch's fused Adam; the sweep path with 16 IDENTICAL rates; the sweep path with 16 different rates | Making the rate a per-copy VECTOR (the optimizer change) costs 0.0-1.6% across 128-4096 copies. The rates actually DIFFERING costs nothing measurable: -0.7% to +0.0%, inside the noise floor at every point, as expected since it changes only the constants the same kernels read | recorded — the sweep is priced at the optimizer change alone |


## Notes

- Spec §11 correction found during implementation: style A must keep the probability ratio in
  the policy surrogate (it carries the gradient); only the clip machinery may be dropped.
  Recorded in the spec file; the jax twin implements the corrected form.
- Comparison point: the jax twin measures 20 ms/iter (style B) / 13 ms (style A) at C=128.
  Remaining torch gap is rollout kernel time (226 us/step incl. 10 buffer-copy kernels) —
  candidate next steps: TF32 matmuls, folding buffer writes into the compiled step,
  concatenating the four 4-input GEMMs (actor/critic/target/predictor first layers) into one.
- The T=32, N=16 rollout-shape alternative (spec §14) is measured separately for the report;
  it changes the algorithm (GAE horizon) so it is a labeled variant, never a silent swap.

## Round 4 — closing the gap to the JAX trainer

Target: the update stage, which the phase profile put at 13.9 ms of a 20.2 ms iteration at 128
copies, and which is where the JAX trainer's lead was concentrated (it led by 52-70% with
sixteen updates per batch, but only 4-19% with one).

Measured breakdown of ONE minibatch step at 128 copies before any round-4 change
(`benchmarks/profile_update.py`, new in this round):

| part of a minibatch step | time | share |
|---|---|---|
| forward and backward | 1,720 us | 58% |
| clip the per-copy gradient norm | 926 us | 31% |
| Adam step | 188 us | 6% |
| gather the minibatch (9 tensors) | 74 us | 2% |
| zero the gradients (19 tensors) | 69 us | 2% |

| # | change | result | verdict |
|---|---|---|---|
| 17 | ROUND 4. One flat parameter buffer. The twenty-one parameters stay separate leaf tensors, but their storage is twenty-one windows onto a single [C, P] buffer and their gradients are windows onto a single [C, P] gradient buffer, assigned up front. The per-copy norm becomes one reduction, Adam one chain, zeroing one kernel; torch.optim is dropped entirely, so one optimizer now serves both the uniform and the swept case | Update-stage parts at 128 copies: clip+Adam+zero 1,183 -> 122 us. Unexpectedly, forward and backward ALSO fell, 1,720 -> 1,012 us, because pre-assigning the gradient windows removes twenty-one allocations per backward. Whole iteration, paired against the predecessor revision: C=8 13.89 -> 9.80 ms (+29.4%), C=128 20.54 -> 16.96 ms (+17.5%), noise floor 0.04-0.05 ms | KEEP |
| 18 | ROUND 4. Write the clip-and-Adam chain functionally (one expression, then copy back) instead of a sequence of in-place operations, so the compiler fuses it into one pass over the buffer | No change: C=128 +16.9% against the same baseline, versus +17.5% for the in-place form — the two are within the noise floor of each other. Kept for readability, not for speed | KEEP (no effect) |
| 19 | ROUND 4. Shuffle the whole batch once per epoch into a static buffer, so each minibatch is a contiguous slice instead of its own gather. Nine gather kernels per epoch instead of nine per step: 36 per iteration instead of 144 | C=128 16.96 -> 16.31 ms (cumulative +20.5% against the predecessor); C=512 regression narrowed from -2.5% to -1.1%. Exact: identical rows in identical order | KEEP |

### The one size where this is a loss

At 512 copies the round-4 build is 1.1% SLOWER than its predecessor (44.33 against 43.84 ms,
noise floor 0.03). The reason is visible in the profile: at that size the flat buffer is 123 MB,
the optimizer stage becomes limited by memory bandwidth rather than by the number of kernels,
and the extra pass the per-copy norm requires costs more than the kernels it saves. The win at
8-128 copies (+20 to +29%) is large and the loss at 512 is small, so the change is kept as the
default; a run that lives at 512 copies or more and cares about the last percent can measure
both. Reducing this further needs the moments in a narrower type, which changes the numerics
and belongs in its own round with its own gate.

### Correctness

All gates pass. Two drift assertions were re-scoped, and the reasoning is recorded here because
re-scoping a test to make a change pass is exactly the move that hides defects:

- `test_hoist_equivalence_gpu.py` and `test_compile_post_gpu.py` each compare a whole run against
  another build. Their iteration-0 comparisons (byte-identical inputs) still gate tightly, and
  iteration 0 is bitwise identical after this round's changes. Their DRIFT figures, measured
  after three chained iterations, cannot separate a real defect from floating-point
  reassociation once the compared builds differ in the optimizer: one differing bit in the first
  update changes the actions sampled in the second iteration. Those two assertions are now loose
  sanity bounds, and the discriminating check they used to stand in for is a new test.
- `test_flat_optimizer_gpu.py` (new) is that check: it runs one optimizer step through the flat
  form and through the previous per-tensor form from byte-identical inputs and compares. Worst
  relative deviation 6.9e-6, on parameters that moved 3.0e-4 — floating-point rounding. It also
  asserts the parameters moved at all, so it cannot pass vacuously.

### Where the PyTorch trainer now stands against JAX

Both measured with each iteration waited for (`--timing sync`), milliseconds per iteration:

| update convention | copies | PyTorch before | PyTorch after | JAX | remaining gap |
|---|---|---|---|---|---|
| one update per batch | 8 | 5.7 | 5.3 | 5.5 | PyTorch ahead by 4% |
| one update per batch | 32 | 6.4 | 6.1 | 6.0 | level |
| one update per batch | 128 | 8.1 | 7.8 | 6.8 | JAX ahead by 15% |
| sixteen updates per batch | 8 | 13.8 | 9.3 | 8.1 | JAX ahead by 15% |
| sixteen updates per batch | 32 | 15.7 | 10.8 | 9.8 | JAX ahead by 10% |
| sixteen updates per batch | 128 | 20.4 | 16.3 | 13.4 | JAX ahead by 22% |

PyTorch now matches or beats JAX with one update per batch at 8 and 32 copies, and the
sixteen-update gap fell from 52-70% to 10-22%.

### What the remaining gap is made of

After this round the update stage is dominated by the forward and backward passes themselves:
they are 72-84% of a minibatch step, against 10-23% for the whole optimizer. The remaining
difference is therefore in how the two frameworks execute a small multi-layer perceptron's
forward and backward, not in the surrounding machinery — XLA fuses that into fewer kernels than
inductor plus autograd do. Closing it would mean writing the loss and its gradient as one fused
kernel rather than composing framework operations, which is a larger change than this round and
should be entered with a go/no-go probe rather than begun speculatively.

## Round 5 — the copy counts the trainer is actually used at (1,024 to 4,096)

Rounds 1 to 4 optimised, and every ledger row above was decided at, 8 to 128 copies. The trainer
is used at 1,024 to 4,096. This round re-opened the question at those sizes, and the first thing
the measurements said is that the two regimes are not the same problem.

### Where the time goes at these sizes

Milliseconds per phase, one iteration, sixteen updates per batch, for the trainer AS THIS ROUND
FOUND IT, measured as three separately captured graphs (`benchmarks/profile_phases.py`):

| phase | 128 copies | 1,024 copies | 4,096 copies |
|---|---|---|---|
| rollout, 128 sequential steps | 4.97 | 12.05 | 16.85 |
| post-rollout processing | 1.30 | 8.76 | 35.30 |
| update, 16 minibatch steps | 13.89 | 63.98 | 239.15 |
| the same iteration as one graph | 20.31 | 81.59 | 290.58 |

The rollout, which is a quarter of the iteration at 128 copies, is six percent of it at 4,096:
it is a chain of 2,304 small operations whose cost barely grows with the copy count, so adding
copies makes it nearly free. Everything else grows in proportion to the copy count, which is the
signature of a computation limited by memory traffic rather than by the number of operations.
**At 128 copies this trainer is launch-bound; at 1,024 and above it is memory-bandwidth-bound.**
Optimisations chosen for the first regime are not the ones the second regime wants, and one of
them turned out to be a loss there (row 20).

Microseconds inside ONE minibatch step (`benchmarks/profile_update.py`):

| part of a minibatch step | 1,024 copies | 4,096 copies |
|---|---|---|
| forward | 1,272 | 4,637 |
| backward | 1,620 | 5,797 |
| clip, Adam and zero over the flat buffer | 1,068 | 3,913 |
| gather the minibatch | 166 | 372 |
| one step | 4,127 | 14,720 |

### What one byte of memory traffic costs here

Every candidate below was judged against measurements of the individual operations on their real
shapes (`benchmarks/probe_update_ops.py`, 1,024 copies):

| operation | time | bytes | rate |
|---|---|---|---|
| copy one gibibyte, the reference a streaming kernel reaches | 606 us | 2.15 GB | 3,543 GB/s |
| clip and Adam over the flat buffer, compiled | 580 us | 1.96 GB | 3,382 GB/s |
| clip and Adam over the flat buffer, not compiled | 2,562 us | 1.96 GB | 766 GB/s |
| gather the cached target features | 267 us | 0.54 GB | 2,009 GB/s |

The card delivers about 3.5 TB/s to a plain streaming kernel, against a specification figure of
3.9. The compiled optimiser chain reaches 95 percent of that, so there is nothing to win inside
it; the same chain uncompiled is 4.4 times slower, which is what `compile_opt` buys.

### The previous round, re-measured where the trainer is actually used

Round four kept the flat parameter buffer on the strength of 8-to-128-copy measurements and
recorded a 1.1 percent loss at 512 copies as the one size where it was a loss. Measured against
its own predecessor at 1,024 copies, both sides pinned to their git revisions and run in the
order A B B A:

| copies | update convention | before round four | after round four | difference | noise floor |
|---|---|---|---|---|---|
| 1,024 | one update per batch | 24.86 ms | 29.64 ms | round four 19.2% slower | 0.03 ms |
| 1,024 | sixteen updates per batch | 77.24 ms | 82.73 ms | round four 7.1% slower | 0.52 ms |
| 4,096 | sixteen updates per batch | 275.12 ms | 290.97 ms | round four 5.8% slower | 0.18 ms |

The 512-copy loss was not an isolated size; it was the beginning of a trend that reaches nearly a
fifth of the iteration at the sizes in use. A kernel-level profile of the round-four build at
1,024 copies says why, and it is not the reason the round-four note guessed (memory bandwidth in
the optimiser). Of the 31.1 milliseconds of matrix-multiplication time in one iteration,
18.8 milliseconds were spent in the library's UNVECTORISED kernels — the ones whose names end in
`align1`, which load one number at a time instead of four:

| stage | multiplication time | of which unvectorised |
|---|---|---|
| rollout | 7.75 ms | 5.58 ms |
| post-rollout | 3.26 ms | 1.17 ms |
| update | 20.05 ms | 12.05 ms |

The cause is the flat buffer's own layout, and it is arithmetic: the twenty-one windows were packed
tightly, so the per-copy row is 59,910 numbers long and several window offsets are odd multiples
of two. A parameter's address for copy c is base + c x 59,910 x 4 bytes, which is a multiple of
16 for almost no c, and the library selects its scalar-load kernels accordingly. Row 21 fixes it.

### Against the JAX trainer at these sizes, before this round's changes

Both waiting for every iteration (`--timing sync`), milliseconds per iteration. The JAX side is
`ppo/jax_ppo/jax_ppo_rnd.py` as this round found it (last changed at commit `7609297`); a
separate line of work was changing it while these were measured.

| update convention | copies | PyTorch | JAX | JAX faster by |
|---|---|---|---|---|
| one update per batch | 1,024 | 29.61 | 14.24 | 2.08x |
| one update per batch | 2,048 | 48.23 | 23.20 | 2.08x |
| one update per batch | 4,096 | 90.36 | 43.76 | 2.06x |
| sixteen updates per batch | 1,024 | 82.24 | 46.94 | 1.75x |
| sixteen updates per batch | 2,048 | 152.24 | 84.10 | 1.81x |
| sixteen updates per batch | 4,096 | 290.60 | 164.10 | 1.77x |

This is a much larger distance than the round-4 note reports at 128 copies (JAX ahead by 10 to
22 percent), and it is the same cause seen through the other regime: at 128 copies XLA's fusion
means fewer device programs, worth tens of percent; at these sizes it means fewer intermediates
written to memory, and memory traffic is what the iteration costs. JAX also holds less memory at
these sizes (9.0 GB against 13.8 at 4,096 copies with sixteen updates), because the PyTorch
trainer stores the frozen target's features and a second copy of the permuted batch.

### The three changes, and what they bought

Each measured against the revision before it, at 1,024 copies with sixteen updates per batch,
in the order A B B A with the spread between two runs of the same side as the noise floor.

| # | change | measured | verdict |
|---|---|---|---|
| 21 | Pad every parameter window and the per-copy row to a multiple of four numbers, so every copy's parameters start on a sixteen-byte boundary and the multiplication library uses its four-at-a-time kernels instead of its scalar-load ones. Ten extra numbers per copy, never read: nothing writes a gradient into the padding, so its Adam step is exactly zero and it adds exactly zero to the gradient norm | 82.48 -> 73.14 ms, 11.3% faster, noise floor 0.54 ms | KEEP |
| 22 | Add each layer's bias AFTER the multiplication, in the same expression as the activation, instead of folding it into the multiplication call. The library has no batched multiply that broadcasts a bias, so it materialises the bias into the output tensor and accumulates on top of it, and the activation then reads and writes the same tensor: five passes over every activation where three suffice. BITWISE identical — loss and all twenty-one gradients agree to 0.000e+00 (`tests/test_bias_form_gpu.py`) | 73.26 -> 61.51 ms, 16.0% faster, noise floor 0.50 ms | KEEP |
| 23 | Stop accumulating gradients: ask autograd for them and copy them into the flat buffer in one call, which also makes the zeroing unnecessary; and compile the gradient limit separately from the Adam step, because compiled together the compiler emits one reduce-and-update program that reaches only 2.4 of the card's 3.5 TB/s where two programs reach 3.2 and 3.5 | 61.45 -> 57.07 ms, 7.1% faster, noise floor 0.46 ms | KEEP |

Measured together, both sides built by the same harness in the same session, each iteration
waited for:

| update convention | copies | before | after | faster by |
|---|---|---|---|---|
| one update per batch | 1,024 | 29.61 ms | 18.13 ms | 39% |
| one update per batch | 2,048 | 48.23 ms | 32.40 ms | 33% |
| one update per batch | 4,096 | 90.36 ms | 63.03 ms | 30% |
| sixteen updates per batch | 1,024 | 82.24 ms | 57.00 ms | 31% |
| sixteen updates per batch | 2,048 | 152.24 ms | 107.24 ms | 30% |
| sixteen updates per batch | 4,096 | 290.60 ms | 206.91 ms | 29% |

And paired against the round-four revision directly, which also covers the sizes the earlier
rounds were tuned on:

| setting | before | after | faster by | noise floor |
|---|---|---|---|---|
| 4,096 copies, sixteen updates per batch | 290.94 ms | 205.36 ms | 29.4% | 0.44 ms |
| 4,096 copies, one update per batch | 89.93 ms | 63.06 ms | 29.9% | 0.11 ms |
| 128 copies, sixteen updates per batch | 16.18 ms | 12.54 ms | 22.5% | 0.03 ms |
| 8 copies, sixteen updates per batch | 9.32 ms | 7.96 ms | 14.7% | 0.09 ms |

**No size measured is slower.** That is worth stating because the previous round's honest loss at
512 copies is exactly the shape of defect this round was looking for in the other direction: the
three changes remove passes over memory, which matters most where the tensors are large, and
remove device programs as a side effect (twenty-one gradient additions and one zeroing per update
step), which is what the small sizes reward.

That is more than a repair of the round-four regression: at 1,024 copies with one update per batch
the changed trainer is 27 percent faster than the pre-round-four build as well (18.13 against
24.86 ms). Against the JAX trainer the distance falls from 2.06-2.08x to 1.27-1.44x with one
update per batch, and from 1.75-1.81x to 1.21-1.27x with sixteen. 8,192 copies also fits and was
measured: 121.8 ms with one update per batch (29.5 GB) and 407.0 ms with sixteen (27.4 GB), on a
94 GB card.

### The same profiles, after

Every measurement that chose the changes, repeated on the changed trainer.

Kernel-level, one iteration at 1,024 copies with sixteen updates per batch:

| | device time | multiplications | of which unvectorised |
|---|---|---|---|
| before | 79.58 ms | 31.06 ms | 18.80 ms |
| after | 52.26 ms | 21.49 ms | **0.00 ms** |

The unvectorised kernels are gone, and the multiplications that remain cost 9.6 ms less — which
is more than alignment alone can explain, because the bias change also removed one library call
per layer.

Phases of one iteration at 4,096 copies, sixteen updates per batch:

| phase | before | after |
|---|---|---|
| rollout | 16.85 ms | 15.28 ms |
| post-rollout processing | 35.30 ms | 19.72 ms |
| update | 239.15 ms | 169.29 ms |
| the whole iteration as one graph | 290.58 ms | 204.52 ms |

Inside one minibatch step at 4,096 copies:

| part | before | after |
|---|---|---|
| forward | 4,637 us | 2,583 us |
| backward | 5,797 us | 4,786 us |
| gradient limit and Adam | 3,913 us | 2,308 us |
| gather the minibatch | 372 us | 371 us |
| one step | 14,720 us | 10,048 us |

The forward is 44 percent cheaper, which is the bias change; the gradient limit and Adam 41
percent, which is the separated compilation and the removed zeroing; the backward 17 percent,
which is the removed accumulation. The post-rollout stage falls by 44 percent for the same reason
as the forward — it is four wide passes through the same layers.

### Correctness, after all three changes

Every gate run on the graphics processor, none re-scoped:

| gate | result |
|---|---|
| rollout capture against the uncaptured compiled step | bitwise equal |
| captured update against the uncaptured update, one update per batch | worst parameter difference 0.000e+00 |
| captured update against the uncaptured update, sixteen updates per batch | worst parameter difference 0.000e+00 |
| the annealed rate reaches the captured graph; a zero-rate group stays frozen | movement exactly 0.000e+00 |
| flat optimiser against the per-tensor form, one step from identical inputs | worst relative 6.9e-06 on parameters that moved 3.0e-04 |
| hoist equivalence, three iterations | worst parameter deviation 0.000e+00 |
| compiled post-rollout body, isolated | worst relative field deviation 2.1e-07 |
| six learning-rate-sweep gates (uniform matches plain, zero-rate frozen, groups independent, paired and distinct seeding, and the same under capture) | all pass |
| bias form, loss and all twenty-one gradients from identical inputs | 0.000e+00 — bitwise |
| gradients land in the flat buffer, nothing accumulates, padding stays zero (new) | pass |
| every parameter and gradient window on a sixteen-byte boundary, checked on more than one copy (new) | pass |

Two of these are stronger than they were: the captured update now agrees with the uncaptured one
BITWISE in both styles, where round four's gate was a tolerance of 1e-5.

### What the alignment change does to the arithmetic, exactly

Running a whole iteration under the old and new revisions from one seed and comparing parameters
gives 1.08e-3 absolute after ONE iteration — far more than the other two changes (7.0e-7 and
3.0e-7). That measurement cannot be read as an equivalence check: an iteration samples actions,
so a difference in the last bits of the first action sends the two runs down different
trajectories and the number afterwards measures divergence, not error. The question has to be
asked of the arithmetic in isolation, which `benchmarks/probe_alignment_numerics.py` does: the
same weights read from an aligned window and from a window offset by two numbers, multiplied by
the same input, on the trainer's real shapes.

| layer | inner dimension | reduced precision (as shipped) | full single precision |
|---|---|---|---|
| actor and critic first layer, packed | 4 | 5.5e-4 relative | 0.0 |
| actor second layer | 64 | 0.0 | 0.0 |
| RND predictor first layer | 4 | 5.2e-4 relative | 0.0 |
| RND predictor second layer | 256 | 0.0 | 0.0 |
| RND predictor third layer | 128 | 0.0 | 0.0 |

Read across: in full single precision the alignment changes nothing at all, anywhere. In the
shipped configuration it changes the two layers whose inner dimension is four, by about 5e-4
relative. The explanation follows from the two columns together — when those weights were
misaligned the library could not use the reduced-precision matrix units for them and fell back to
full single precision, and aligning them lets `tf32=True` apply where it previously could not.
That is the configuration doing what it says: `PPOConfig.tf32` is documented as "faster, ~1e-3
relative rounding", and 5e-4 is inside it. The change makes the trainer MORE consistent with its
own setting rather than less, and it is not a silent precision loss: the same setting already
governs every other multiplication in the program.

### Tried and NOT adopted: letting the compiler generate the multiplications

The report names this as the obvious next step, so it was measured rather than left as a guess
(`benchmarks/probe_autotune.py`). One trainer is built, the update stage is timed with the loss
compiled as it ships, the loss is rebuilt with the compiler's own multiplication templates
enabled — which lets it fold the bias and the activation into the multiplication instead of
leaving them as a following pass — and the stage is timed again, alternating between the two.

| run | shipped form | compiler-generated | difference | spread within a form |
|---|---|---|---|---|
| first | 46.77 ms | 44.80 ms | 4.4% faster | 0.58 ms |
| second, after correcting the probe's difference measure | 45.80 ms | 44.48 ms | 3.0% faster | 0.79 ms |

So: real, small — 3 to 4 percent of the update stage, which is 2 to 3 percent of an iteration —
and the two runs disagree by more than either spread, which is itself a statement about how well
this harness resolves an effect of that size. Against that, the compiler's chosen kernels set
`ALLOW_TF32=False`, so the trainer would silently stop using the reduced-precision matrix units
for the layers it generates; the loss it computes differs by 2.6e-6 relative and the worst
gradient by 6.5e-4 absolute against a largest gradient of 8.7e-1; and compilation takes
substantially longer because every shape is benchmarked. NOT adopted in this round, whose rule
was that the arithmetic must not change. It is the right first candidate for a round that allows
it, and it now has a number attached rather than an expectation.

### Ideas costed and NOT taken, with the arithmetic that rejected them

Recorded because the counting is the result, and because two of them look obviously right until
the bytes are counted.

1. **Recompute the frozen RND target features in every minibatch step instead of caching them**
   — which is what the JAX twin does, and which looks like the right trade at these sizes because
   caching spends memory traffic to save arithmetic. Counted per copy per iteration: caching costs
   0.26 MB to write the features, 2.1 MB to permute them once per epoch and 1.0 MB to read them
   across the sixteen steps, about 3.4 MB. Recomputing costs nothing to store but writes and reads
   a 256-wide hidden layer in every one of the sixteen steps, 0.26 MB each, about 6.4 MB, plus the
   target's own weights sixteen times. Caching wins by nearly a factor of two. NOT taken.
2. **Tile the update stage over copies so that a tile's parameters, moments and gradients stay in
   the 50 MB cache across all sixteen minibatch steps.** That would remove fifteen sixteenths of
   the parameter traffic, which is a third of the iteration. The per-copy working set is about
   1.65 MB, so a tile that fits the cache holds about 30 copies — and a program with 30 pieces of
   work cannot fill 132 processing blocks. The two constraints are irreconcilable at this network
   size. NOT taken.
3. **Keep the Adam moments in a narrower number format.** It would remove two of the eight passes
   the optimiser makes, about 5 percent of a minibatch step. It changes what the trainer computes,
   so it belongs in its own round with its own equivalence gate rather than inside a round whose
   rule is that the arithmetic must not change. NOT taken here.

## Round 6 — the JAX side's fifth round, read from here; and 1,024 to 4,096 again

The JAX trainer's own fifth round finished after this side's fifth round was written, so its
findings had never been available here. This round read them, took what transferred, and re-opened
the copy counts the trainer is used at with the measurement method that round arrived at.

### The premise, checked before anything was changed

The brief for this round carried a head-to-head table whose JAX figures came from BEFORE its fifth
round. Both sides were therefore re-measured in one session, the PyTorch side pinned to the
revision this round starts from (`7287209`) and the JAX side taken from the merged trainer, each
waiting for every iteration (`--timing sync`):

| update convention | copies | PyTorch | JAX | JAX ahead by |
|---|---|---|---|---|
| one update per batch | 1,024 | 18.07 ms | 14.21 ms | 1.27x |
| one update per batch | 2,048 | 32.57 ms | 23.27 ms | 1.40x |
| one update per batch | 4,096 | 63.33 ms | 43.68 ms | 1.45x |
| sixteen updates per batch | 1,024 | 56.72 ms | 47.02 ms | 1.21x |
| sixteen updates per batch | 2,048 | 107.09 ms | 86.02 ms | 1.24x |
| sixteen updates per batch | 4,096 | 205.51 ms | 167.76 ms | 1.23x |

Two things follow. The PyTorch figures reproduce round five's to within half a percent, so that
round's state is intact. And **the JAX fifth round moved nothing at these sizes**: its trainer is
within a percent of the figures the stale table carried (14.24 / 23.20 / 43.76 and 46.94 / 84.10 /
164.10). Its two kept changes are unroll factors on its two loops, and unrolling buys fewer,
longer-running programs — worth about ten percent at 128 copies, where the iteration's cost is the
number of programs it issues, and nothing at 1,024 and above, where the cost is bytes moved. The
regime split round five found on this side cuts the same way on the other one.

### What transferred

1. **The measurement method.** The JAX round found that its verdict rule — two medians against the
   largest spread a single version shows between rounds — called a real three-percent effect
   "noise", because both versions drift together within a round. They are timed in the same round
   on the same machine, so the difference belongs round by round; the change it was hiding won 11
   of 11 rounds at every size. `benchmarks/bench_torch_change.py` is that harness for this side:
   both configurations built in ONE process, timed round-robin with the order reversed on
   alternate rounds, verdict by the paired count, with the old spread reported beside it.
2. **The direction.** Both JAX gains came from making the compiler emit fewer, longer-running
   programs. At these sizes that means fewer round trips through memory for the same work, which
   is the lever round five was already pulling.

### What did not transfer, and why that is a result

The JAX round's two kept changes are unroll factors. **The PyTorch trainer has no loop to unroll.**
Its sixteen update steps are already emitted one after another into a single recorded sequence of
device programs — the python loop is gone at run time, there is no scan, and no compiler is
choosing how many loop bodies to emit. Its rejected change, the flat parameter buffer, this side
already has, and round five reached a compatible verdict from the other direction: round four's
flat buffer was a regression at 1,024 to 4,096 copies until round five aligned its windows.

### A correction to the record

The trainer holds **twenty-one** trainable tensors per copy, not the nineteen every ledger entry,
code comment and report sentence since round four has said (actor 7, critic 8, predictor 6). The
JAX ledger had it right. Every occurrence is corrected, and the alignment test now counts the
addresses it checked so it cannot pass by iterating an empty list.

### A measurement that measured nothing, caught by its own accuracy check

The first version of `benchmarks/probe_epilogue_fusion.py` built ONE trainer and swapped four
compiled versions of its loss onto it, each compiled inside a context that set the compiler's
options. All four came out **bitwise identical** — loss and every gradient agreeing to
0.000e+00 — and within 0.1 percent of each other in time. That is not four forms agreeing; it is
one form measured four times. The compiler caches its work against the FUNCTION being compiled
and against the backend the wrapper carries, and a setting applied through a surrounding context
is part of neither, so the second, third and fourth compilations were handed the first one's
kernels and the settings never applied.

The probe was rebuilt to give each arm **its own trainer** and to pass the settings to the
compiler as options rather than around it, which makes the arms compile separately for two
independent reasons. The accuracy line the probe already printed is what caught it, and it now
says so explicitly when two arms agree to zero. This is the fourth instance in this project of a
number that was measuring nothing (an empty captured graph, a module missing from the module
table, a profiling variant whose dead code was deleted, and now a compiler cache), and the
pattern is the same every time: the check that catches it is one that asserts the thing under
test actually happened.

### The paired comparison against the revision this round starts from is neutral

The first change of the round is a refactor in its OFF position — the gradient path becomes a
knob, and the old behaviour is one setting of it. Before measuring the new setting, the refactor
itself was measured: at 1,024 copies with sixteen updates per batch, the round-five revision took
57.02 ms and the refactored working tree 56.85, a difference of 0.16 ms against a 0.30 ms noise
floor. Whatever the knob is worth, it is not being credited with the refactor.

### Where the update stage's time goes at 4,096 copies, program by program

Every device program of one iteration, named and grouped by kind
(`benchmarks/profile_kernels.py`, the three stages run without graph capture so the profiler can
name them). This is what chose the round's changes; the update stage is 167.03 ms of it.

| part of the update stage | time | share | rate reached |
|---|---|---|---|
| matrix multiplications | 62.72 ms | 37.6% | — |
| the Adam step | 31.94 ms | 19.1% | 3,435 GB/s |
| copying the gradients into the flat buffer | 16.70 ms | 10.0% | 1,878 GB/s |
| shuffling the batch once per epoch | 6.42 ms | 3.8% | — |
| the per-copy gradient limit | 4.58 ms | 2.7% | 3,430 GB/s |
| everything else: bias, activations, and their gradients | 44.66 ms | 26.7% | — |

The Adam step and the gradient limit already run at the full 3,540 GB/s a plain copy of memory
reaches on this card. The copy does not, and the reason is the layout round four introduced: its
destination is a strided window of the buffer, and the operation probe measures a write into a
strided window at 2,022 GB/s against 3,588 for the same write into a contiguous tensor. It is the
largest single removable item in the stage, and round five's own change created it.

### The changes

Each measured paired, round by round, in one process, at 1,024 and 4,096 copies with sixteen
updates per batch. "Rounds" is how many of the eleven favoured the change.

| # | change | 1,024 copies | 4,096 copies | verdict |
|---|---|---|---|---|
| 24 | **Read the gradients where the backward pass wrote them.** Round five asked the automatic-differentiation system for the gradients and copied the twenty-one fresh tensors into one [C, P] buffer, so that the gradient limit and the Adam step could each be a single program over one contiguous array. Reading them where they lie removes the copy and costs twenty-one programs per optimizer pass instead of one. It also removes the buffer: 981 megabytes less held on the card at 4,096 copies | 57.19 -> 54.89 ms, **-4.02%**, 11 of 11, floor 0.84 | 208.45 -> 199.23 ms, **-4.42%**, 11 of 11, floor 3.49 | KEEP |
| 25 | **Sum the squared gradients in the same program that copies them into the buffer**, so the buffer is not read a second time for the gradient limit. Keeps the buffer and its single-program optimizer | 56.56 -> 54.17 ms, **-4.22%**, 11 of 11, floor 0.73 | 204.75 -> 197.83 ms, **-3.38%**, 11 of 11, floor 3.36 | measured, NOT the default: row 24 beats it (below) |
| 26 | **Write the shuffled batch straight into its buffer.** `buffer.copy_(t.gather(...))` allocates a whole second copy of the shuffled batch and copies it across; `torch.gather(t, 1, ix, out=buffer)` does not, and the 2.86 ms of device-to-device copying the kernel profile names disappears with it | 54.00 -> 53.25 ms, **-1.39%**, 11 of 11, floor 0.78 | 197.19 -> 193.56 ms, **-1.84%**, 11 of 11, floor 3.76 | KEEP |

Row 26 was kept on a kernel profile first, and that was not enough. A profile names the program
that disappears; it does not say what the iteration costs afterwards, and a cross-process
comparison of the whole round could not resolve the difference. It was therefore made a
configuration knob so the paired instrument could answer:

| copies | second copy, then copy across | straight into the buffer | change | rounds | floor |
|---|---|---|---|---|---|
| 8 | 7.88 ms | 7.92 ms | +0.50% | 0 of 11 | 0.00 ms |
| 128 | 12.24 ms | 12.12 ms | **-1.04%** | 11 of 11 | 0.02 ms |
| 1,024 | 54.00 ms | 53.25 ms | **-1.39%** | 11 of 11 | 0.78 ms |
| 4,096 | 197.19 ms | 193.56 ms | **-1.84%** | 11 of 11 | 3.76 ms |

It is on everywhere: the 0.50 percent it costs at 8 copies is 0.04 ms, and a third branch in the
configuration for that is not worth its complexity.

### Row 24 reverses sign below 512 copies, and the trainer chooses by copy count

The change was decided at 1,024 and 4,096, so it was then measured at every size the earlier
rounds were tuned on. Sixteen updates per batch, eleven paired rounds each:

| copies | with the buffer | reading them in place | change | rounds favouring it | floor |
|---|---|---|---|---|---|
| 8 | 7.88 ms | 8.41 ms | **+6.72%** | 0 of 11 | 0.01 ms |
| 32 | 8.95 ms | 9.43 ms | **+5.40%** | 0 of 11 | 0.03 ms |
| 128 | 12.38 ms | 12.73 ms | **+2.84%** | 0 of 11 | 0.03 ms |
| 512 | 30.03 ms | 29.29 ms | **-2.45%** | 11 of 11 | 0.69 ms |
| 1,024 | 57.19 ms | 54.89 ms | **-4.02%** | 11 of 11 | 0.84 ms |
| 2,048 | 105.69 ms | 100.19 ms | **-5.20%** | 11 of 11 | 1.85 ms |
| 4,096 | 208.45 ms | 199.23 ms | **-4.42%** | 11 of 11 | 3.49 ms |
| 8,192 | 404.39 ms | 385.10 ms | **-4.77%** | 11 of 11 | 6.36 ms |

Every one of the eight is unanimous across its eleven rounds, in one direction below 512 copies
and in the other at 512 and above. The reason is the regime split this round and the last are both
about: forty-two device programs per update step where there were two is a bad trade for one copy of a buffer
that is 2 to 31 megabytes at 8 to 128 copies, and a good one for the same copy at 123 megabytes to
a gigabyte. `production_config` therefore picks the form from the copy count, and the buffer's
layout follows it — one block per parameter where the optimizer is twenty-one programs, one row
per copy where it is one. This is the same shape of answer round five arrived at from the other
direction, and it is the reason both rounds measure every change at both ends of the range.

With ONE update per batch the same change is worth much less, which is the same arithmetic seen
from another angle: there is one update step per iteration rather than sixteen, so the copy is
paid for once rather than sixteen times. 128 copies +0.13% (0 of 11, against a 0.01 ms floor —
nothing), 1,024 copies -0.58% (11 of 11), 4,096 copies -1.23% (11 of 11). The size rule is the
same for both conventions.

Rows 24 and 25 remove different passes and cannot both apply, so they were also measured against
each other rather than compared through their separate baselines:

| copies | with the limit fused into the copy | reading the gradients in place | difference | rounds |
|---|---|---|---|---|
| 1,024 | 54.38 ms | 54.11 ms | -0.5% | 11 of 11 |
| 4,096 | 198.34 ms | 195.18 ms | -1.6% | 11 of 11 |

Row 25 was then measured at the small sizes too, because that is where row 24 loses and something
has to be right there:

| copies | plain buffer | with the limit fused into the copy | change | rounds | floor |
|---|---|---|---|---|---|
| 8 | 7.87 ms | 7.93 ms | +0.64% | 0 of 11 | 0.01 ms |
| 32 | 8.95 ms | 8.85 ms | **-1.10%** | 11 of 11 | 0.01 ms |
| 128 | 12.38 ms | 12.12 ms | **-2.07%** | 11 of 11 | 0.01 ms |
| 512 | 30.06 ms | 29.17 ms | **-2.97%** | 11 of 11 | 0.67 ms |

So row 25 IS the default below the crossover, and row 24 above it. The crossover is set at 1,024
rather than 512: at 1,024 the two were compared directly and row 24 wins by 0.5 percent in 11 of
11 rounds, while at 512 row 25 is the one that measured better against their shared baseline
(29.17 against 29.29 ms). The 0.64 percent row 25 costs at 8 copies is 0.05 ms against a 0.01 ms
spread — unanimous, but small enough to take the simpler two-branch rule rather than a third.

And the programs themselves, timed on their real shapes at 4,096 copies
(`benchmarks/probe_gradient_form.py`), which is what the whole-iteration difference is made of:

| program | time | bytes | rate |
|---|---|---|---|
| copy the gradients into the buffer, buffer form | 861 us | 1.96 GB | 2,280 GB/s |
| the gradient limit over the buffer, buffer form | 290 us | 0.98 GB | 3,385 GB/s |
| Adam over the buffer, buffer form | 1,972 us | 6.87 GB | 3,486 GB/s |
| the gradient limit over the twenty-one gradients, no-buffer form | 341 us | 0.98 GB | 2,883 GB/s |
| Adam over the twenty-one windows, no-buffer form | 2,214 us | 6.87 GB | 3,104 GB/s |

Per minibatch step that is 3,123 us against 2,554, so 569 us, so **9.1 ms over the sixteen steps
of an iteration** — which is the 9.2 ms the whole-iteration comparison measured. The decomposition
also says what the no-buffer form gives back: its Adam reaches 3,104 GB/s where the buffer form's
reaches 3,486, because it walks twenty-one strided windows instead of one contiguous array. That
is 4 ms an iteration handed back, and it is what the layout change below goes after.

### Row 27: the buffer's layout, which round four got the wrong way round for this regime

Round four put every parameter in one buffer with **one row per copy**, [copies, parameters]. That
is the layout a single-program optimizer needs, because a copy then owns a contiguous row and one
kernel can walk the whole array applying that copy's rate. Row 24 replaced the single-program
optimizer with twenty-one programs, and for twenty-one programs the same layout is the wrong one:
a parameter's window of a copy-major buffer is strided across copies, and the operation probe
measures a pass over a strided window at 2,022 GB/s against 3,588 for the same pass over a
contiguous tensor.

`parameter_layout="parameter_major"` holds **one contiguous block per parameter** instead — all
copies of the first weight, then all copies of the first bias, and so on. Every window a
multiplication or the optimizer touches is then contiguous. Alignment is kept the same way, by
padding each block's per-copy length to a multiple of four numbers; only four of the twenty-one
need it (both value-head biases, the actor's output bias and the log standard deviation, six
numbers per copy in total) and none of them is a multiplication operand.

| copies | copy-major | parameter-major | change | rounds | floor |
|---|---|---|---|---|---|
| 128 | 12.71 ms | 12.61 ms | **-0.80%** | 11 of 11 | 0.04 ms |
| 1,024 | 54.06 ms | 53.26 ms | **-1.49%** | 11 of 11 | 0.98 ms |
| 4,096 | 195.19 ms | 193.77 ms | **-0.73%** | 11 of 11 | 3.92 ms |

Both sides read the gradients where they were written, so the only difference is the layout. The
same expressions run over the same numbers at different addresses, and on the PROCESSOR the two
train to bitwise equal parameters — a test runs two iterations in each layout and requires exact
equality. On the CARD they are not bitwise equal, and the reason is the point of the change rather
than a caveat around it: the multiplication library chooses its kernel partly from the operand's
layout, so a contiguous weight and a strided one are multiplied by different kernels, which sum
the same products in a different order. Measured on one iteration from identical inputs, the
gradients differ by 4.5e-08 against a largest gradient of 8.6e-01, and the parameters after a
step by 7.8e-11 relative
(`tests/test_parameter_layout_gpu.py`). KEEP.

The programs again, at 4,096 copies, to see how much of the strided penalty the layout recovers:

| Adam over the twenty-one windows | time | rate |
|---|---|---|
| copy-major (a parameter's window strided across copies) | 2,214 us | 3,104 GB/s |
| parameter-major (each window contiguous) | 2,102 us | 3,268 GB/s |
| for comparison, one program over the contiguous buffer (the buffer form) | 1,972 us | 3,486 GB/s |

So the layout recovers about a third of what the twenty-one-program optimizer gave back, and the
rest is the twenty-one launches themselves rather than the addresses they touch. 112 us a step is
1.8 ms an iteration, against the 1.4 ms the whole-iteration comparison measured.

### Row 28, tried and NOT kept: letting the compiler generate the multiplications

Round five measured this and set it aside because the generated kernels switch the
reduced-precision matrix units off. This round found why — a size rule in the compiler's template
heuristics allows those units only when a multiplication has at least sixteen rows AND the smaller
of its two inner dimensions is at least 512, and every multiplication here has an inner dimension
of 4, 64, 128 or 256 — and then measured the form properly, with one whole trainer per arm.

| form | update stage, 4,096 copies | against the shipped form | rounds faster |
|---|---|---|---|
| the library's multiplication, as shipped | 152.15 ms | — | — |
| the compiler's, both backends offered | 148.00 ms | **-2.7% faster** | 7 of 7 |
| the compiler's, library backend removed | 221.49 ms | **+45.6% slower** | 0 of 7 |
| the same, with the size rule replaced so the matrix units apply | 221.47 ms | +45.6% slower | 0 of 7 |

Read the four rows together. Offering both backends is round five's arm and reproduces its
result: 2.7 percent faster on the update stage, 7 of 7 rounds, at the cost of changing the
arithmetic — the loss moves 4.7e-06 relative and the worst gradient 5.0e-04 against a largest of
8.8e-01, which are the same magnitudes round five recorded. But that arm does NOT fuse any
epilogue: the library's kernel wins the selection on nearly every shape, so what it buys is a
better kernel here and there, not the pass-removal this round was testing.

Removing the library backend is what forces a generated kernel and so what makes the bias and the
activation fold into the multiplication. That form is **45.6 percent slower**, and switching the
matrix units back on does not move it. The selection log says why: the library's multiplication
runs the first shape in 0.071 ms against the best generated candidate's 0.078, and the backward
pass's transposed shapes are worse. The passes an epilogue would remove cannot pay for kernels
that much slower. NOT KEPT — and the hypothesis this round was built on, that PyTorch could be
made to fold its element-wise work into its multiplications the way the JAX compiler does, is
answered in the negative with a number.

One caveat on the last row, stated rather than hidden: its figures are identical to the row above
it to five digits, which is the signature of the compiler handing it the row above's stored
kernels — its settings differ only by a monkeypatch, which is not part of the key that code is
stored under. The probe now disables that store for this arm. The verdict does not depend on it:
the form it modifies is 45 percent slower whatever precision its multiplications use.

### The round end to end, against the revision it started from

Both sides built in their own process and run in the order A B B A, so the spread between two
runs of the same side is the noise floor. Sixteen updates per batch unless the row says otherwise.
Below 1,024 copies the trainer keeps the buffer and fuses the gradient limit into its copy; at
1,024 and above it reads the gradients in place and holds one block per parameter.

| copies | round five | round six | difference | noise floor |
|---|---|---|---|---|
| 8 | 7.96 ms | 7.82 ms | **-1.7%** | 0.01 ms |
| 128 | 12.51 ms | 12.12 ms | **-3.1%** | 0.01 ms |
| 512 | 30.35 ms | 28.86 ms | **-4.9%** | 0.13 ms |
| 4,096 | 205.84 ms | 191.35 ms | **-7.0%** | 0.69 ms |
| 4,096, one update per batch | 63.19 ms | 62.33 ms | **-1.4%** | 0.20 ms |

**No size is slower.** One update per batch gains less for the reason its own row gives: the
iteration has one update step rather than sixteen, so every change that removes a pass inside an
update step is paid for once instead of sixteen times.

And the chain at the sizes the round is about, each row the paired median of the arm that includes
every change above it:

| build | 1,024 copies | 4,096 copies |
|---|---|---|
| the revision this round starts from | 57.19 ms | 208.45 ms |
| + write the shuffled batch straight into its buffer | 56.56 ms | 204.75 ms |
| + read the gradients where the backward pass wrote them | 54.11 ms | 195.19 ms |
| + one contiguous block per parameter | 53.26 ms | 193.77 ms |
| **together** | **-6.9%** | **-7.0%** |

### A second measurement that measured the wrong thing, and how it was found

The first run of the table above reported 8 copies **3.7 percent SLOWER** and 128 copies 0.9
percent slower, against a within-side spread of 0.01 ms. That disagreed with the paired
instrument, which had the one knob that differs at 8 copies at +0.64 percent. Splitting the round
in two through the cross-process instrument then blamed the shuffle change for 3.8 percent at 8
copies — which the paired instrument, once the shuffle change was made a knob, put at 0.50
percent, or 0.04 ms.

The cause was the harness. `ab_compare.py` built each side by constructing `PPOConfig` directly
from a fixed dictionary plus the dataclass defaults. That WAS the shipped configuration until this
round made the gradient's home depend on the copy count; after that, at 8 and 128 copies it timed
the large-copy-count form — the one the trainer never chooses there — and reported it as the
round's result. It now builds each side through that revision's own `production_config`, which is
what the rest of the project means by the shipped configuration, and the numbers above are from
the rebuilt harness. The 4,096-copy readings were never affected: there the defaults and
`production_config` agree.

The lesson is the one this project keeps relearning in a new costume: **a measurement can be
precise and still be of the wrong thing.** Its within-side spread was 0.01 ms — the instrument was
extremely repeatable about a configuration nobody runs. What caught it was two instruments
disagreeing by more than either one's stated resolution, which is the only signal such a defect
gives.

### What is left, and where it is

The kernel profile after the round says where an iteration's remaining time sits, and one item
stands out for the next round rather than this one. Of the rollout's 15.3 milliseconds at 4,096
copies, **11.45 are matrix multiplications running at about 850 gigabytes per second** — about a
quarter of what the update stage's multiplications reach (2,500 to 3,400 on the same card). The cause is structural: the rollout is 128 sequential steps and
each one multiplies **four rows per copy** against that copy's whole actor weights, so the weights
(75.6 megabytes across 4,096 copies) are re-read from memory on every step and each read serves
almost no arithmetic. The counted floor for the whole rollout is 2.8 milliseconds against 15.3
measured. Two shapes of answer exist and neither is small: process the copies in groups whose
weights fit the 50-megabyte cache so the re-reads come from there (which doubles the program
count, and the rollout is already partly bound by that), or generate the multiplications for these
four-row shapes rather than calling the library's, which are clearly not tuned for them. Recorded
here with its measurement rather than attempted at the end of a round.
