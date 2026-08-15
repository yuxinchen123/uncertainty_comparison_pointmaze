# Training many independent PPO+RND copies at once on one H100

Design brief for Module 2. Target: `n_copies` independent training copies (8 to 128 and beyond),
each with its own environment batch, its own policy, value, RND predictor and RND target networks,
its own optimizer state and its own random stream, all stepping and updating in lockstep as one
batched computation. Networks are small: observation dimension 4, action dimension 2, hidden width
64 to 256.

All timings below were measured on serval05 (NVIDIA H100 NVL, 132 SMs, 95 GB, driver 580.159.04,
CUDA 13.0) with PyTorch 2.13.0+cu130, under the repository's H100 lock. Scripts:
`ppo/research/code/bench_batched_mlp.py`, `bench_batched_step.py`, `bench_copy_scaling.py`.
Raw results: `benchmarks/results/batched_mlp_layout_fp32.json`,
`batched_step_modes.json`, `copy_scaling_h256_b256.json`, `copy_scaling_h64_b256.json`.

## 1. The measured answer to the main question

**Per-copy GEMMs do not come close to saturating the H100 at these sizes, and that is the reason
the whole approach works.** Up to a knee, adding copies costs nothing at all.

Copy-scaling sweep. Three-layer MLP (4 → hidden → hidden → 5), weights stored `[copies, in, out]`,
`torch.baddbmm` per layer, `torch.compile(mode="reduce-overhead")` so the step runs as a CUDA
graph. `rollout` is forward-only under `no_grad` with 256 rows per copy; `update` is
forward + backward + Adam on 256 rows per copy. Median of 30 to 50 calls after warmup.

| copies | rollout, hidden 256 | update, hidden 256 | rollout, hidden 64 | update, hidden 64 |
|---|---|---|---|---|
| 1 | 57 us | 607 us | 59 us | 625 us |
| 8 | 57 us | 512 us | 57 us | 636 us |
| 32 | 78 us | 522 us | 56 us | 636 us |
| 64 | 121 us | 545 us | 57 us | 637 us |
| 128 | 213 us | 527 us | 63 us | 623 us |
| 256 | 423 us | 992 us | 107 us | 637 us |
| 512 | 802 us | 1,923 us | 196 us | 639 us |

Read this as follows.

- At hidden 256, one update step for **128 independent copies takes the same wall time as one
  update step for a single copy** (527 us against 607 us). The GPU is doing 128 times the
  arithmetic in the same time. Past 256 copies the time grows in proportion to the copies.
- At hidden 64, the update time is flat all the way to 512 copies. It is 100 percent fixed cost.
- The rollout step (small batch, forward only) is flat to 16 copies at hidden 256 and to 128
  copies at hidden 64, then grows.
- The 0.5 to 0.6 ms update floor is **not arithmetic**. At 128 copies, hidden 256, 256 rows per
  copy, the useful arithmetic is 13.3 GFLOP, which at 527 us is 25 TFLOP/s — a few percent of what
  the card can do. The floor is the number and shape of the kernels in the step (autograd
  bookkeeping plus a multi-tensor Adam over six small parameter tensors), not the GEMMs.

The largest case measured, 128 copies at hidden 256 with 2,048 rows per copy (262,144 rows total),
reached 47 TFLOP/s of useful work with TF32 matmuls enabled. Vendor peak for this card is roughly
500 TFLOP/s TF32 and 60 TFLOP/s on the non-tensor FP32 units (scaled from the published H100
figures by the 1,785 MHz clock; not measured here). So even the biggest configuration in this
project uses on the order of a tenth of the tensor-core capability.

**Consequence for the design.** The first optimization target is the fixed per-step cost, not GEMM
efficiency. Cutting the number of kernels in the update (one fused Adam over one flat parameter
buffer, the whole epoch-and-minibatch loop inside one CUDA graph) buys far more than anything done
to the matmuls.

## 2. Layout: batched matmul, not block-diagonal, not a loop

The question was whether to fold `(n_copies, batch)` into one GEMM dimension with block-diagonal
weights, or to keep a batched matmul. Measured, forward only, strict FP32 (TF32 off):

| shape | loop over copies | bmm | vmap, eager | block-diagonal |
|---|---|---|---|---|
| 128 copies, h 256, 256 rows | 12,053 us | 296 us | 453 us | 12,131 us |
| 128 copies, h 256, 2,048 rows | — | 2,092 us | 2,203 us | 98,084 us |
| 32 copies, h 256, 2,048 rows | — | 551 us | 675 us | 6,285 us |
| 8 copies, h 64, 64 rows | 1,445 us | 61 us | 404 us | 47 us |

Block-diagonal weight memory, for the three layers together: 20 MB at 8 copies and hidden 256;
280 MB at 32 copies; **4.45 GB at 128 copies**. The dense matrix is `copies` times larger and
costs `copies` times the arithmetic, because only the diagonal blocks are useful.

Conclusions.

- **Use a batched matmul.** Store each layer's weights as `[n_copies, in_features, out_features]`
  and call `torch.baddbmm(bias, x, w)`. This lowers to `cublasGemmStridedBatched`, which is the
  right primitive: every copy has identical shapes, so the strided-batched form applies directly.
  CUTLASS grouped GEMM solves the different-shapes-per-group problem, which this workload does not
  have; it is only relevant if copies ever get different hidden widths.
- **Never build a block-diagonal weight.** It is 47 times slower and needs 130 times the weight
  memory at 128 copies and hidden 256. It only wins in the corner where everything is tiny
  (8 copies, hidden 64, 64 rows), where the total time is 50 us either way and irrelevant.
- **The copy axis cannot be folded into the row axis**, because each copy multiplies by a different
  weight matrix. What can and must be folded into the row axis is the time and environment axes:
  reshape rollout data from `[copies, T, n_envs, obs]` to `[copies, T * n_envs, obs]` and issue one
  batched matmul per layer for the whole update batch.
- **Never loop over copies in Python.** 41 times slower at 128 copies, and the gap grows.

## 3. PyTorch: recommended v0 design

### 3.1 Two ways to write it, and which to pick

`torch.func` (`stack_module_state` + `functional_call` + `vmap`) and hand-written `[copies, out, in]`
weights with `baddbmm` produce the same computation. The measured difference is in the execution
mode. Full train step (forward + backward + Adam), TF32 on:

| shape | bmm eager | bmm compile | bmm + CUDA graph | vmap eager | vmap compile | vmap + CUDA graph |
|---|---|---|---|---|---|---|
| 128 copies, h 256, 256 rows | 1,298 us | 877 us | 607 us | 2,075 us | 877 us | 696 us |
| 128 copies, h 256, 2,048 rows | 3,257 us | 2,869 us | 2,897 us | 3,377 us | 2,276 us | 2,287 us |
| 128 copies, h 64, 2,048 rows | 1,025 us | 794 us | 783 us | 2,053 us | 811 us | 651 us |
| 32 copies, h 256, 2,048 rows | 1,137 us | 894 us | 812 us | 1,943 us | 899 us | 657 us |
| 8 copies, h 256, 256 rows | 959 us | 785 us | 539 us | 1,672 us | 791 us | 568 us |

("compile" is `torch.compile(dynamic=False)`; "CUDA graph" is `mode="reduce-overhead"`.)

- In eager mode, `vmap` is 1.6 to 2 times slower than hand-written `baddbmm`. It carries a fixed
  dispatch cost of roughly 0.4 ms per forward and 2.3 ms per backward that does not depend on the
  problem size at all. In a forward-only rollout step at 8 copies and hidden 64 this is the
  difference between 61 us and 404 us.
- After `torch.compile`, the two are the same to within noise, and in three of the five shapes the
  `vmap` version is actually faster after compilation, because Dynamo traces the `vmap` away and
  Inductor is free to pick its own schedule.
- Both compose with CUDA graphs. `reduce-overhead` is worth 1.2 to 1.4 times more on top of
  compilation whenever the batch is small — which is exactly the rollout regime.

**Recommendation for v0: hand-written batched layers with `baddbmm`.** Reasons:

1. It is fast in eager mode too, which matters while debugging, before anything is compiled.
2. Parameters are plain tensors with an explicit copy axis, so the fused per-copy Adam, the
   per-copy gradient-norm clipping and the checkpoint format are all trivial to write and read.
3. It has no interaction with the documented `torch.func` restrictions (see §3.6).
4. It costs about forty lines.

Keep the `vmap` version as a second implementation used only as a correctness oracle: it is written
from an ordinary `nn.Module`, so agreement between the two is strong evidence the batched layers
are right. Re-measure both once the final shapes are fixed; if the compiled `vmap` version wins on
the shape actually used, switching is a one-line change behind the same interface.

### 3.2 Parameter storage

One flat buffer per parameter group, with views into it, so the optimizer is one kernel over one
tensor rather than a multi-tensor loop over many small ones.

```python
class BatchedMLP:
    # dims: [(4, h), (h, h), (h, out)]
    # flat: a single 1-D tensor of length n_copies * sum(in*out + out)
    # w[i]: flat[off_i : off_i + C*in*out].view(C, in, out)
    # b[i]: flat[off_b : off_b + C*out]  .view(C, 1, out)
    def forward(self, x):                    # x: [C, N, 4]
        h = torch.baddbmm(self.b0, x, self.w0).tanh()
        h = torch.baddbmm(self.b1, h, self.w1).tanh()
        return torch.baddbmm(self.b2, h, self.w2)
```

Four such objects: policy trunk with actor and both value heads, the RND predictor, the RND target
(no gradient, still `[copies, ...]` so every copy gets a different random target), plus a
per-copy log-standard-deviation parameter `[copies, 1, act_dim]` for the Gaussian policy.

Initialization must be per copy and must match what a single-copy run would draw: derive each copy's
initialization from its own base seed (§5), not from one global stream, so that copy `k` in a
128-copy run starts from the same weights as a solo run with the same seed.

Per-copy running statistics are tensors with a leading copy axis too: observation mean and variance
`[copies, obs]` and count `[copies]` for the RND input normalizer, and the running estimate of the
intrinsic-return standard deviation `[copies]`. Updating them is a reduction over the batch axis
only.

### 3.3 Every reduction keeps the copy axis

This is where correctness is lost, and it is silent when it goes wrong. In batched form:

- The policy loss, value losses, entropy and RND loss are reduced over the batch axis, giving a
  vector of shape `[copies]`.
- The scalar handed to autograd is `per_copy_loss.sum()`. Use **sum**, not mean over copies:
  summing makes each copy's gradient exactly the gradient of its own loss, so the run is
  numerically the single-copy run. Averaging over copies would divide every gradient by
  `n_copies`, silently scaling the learning rate.
- Advantage normalization is per copy: subtract `adv.mean(dim=1, keepdim=True)` and divide by
  `adv.std(dim=1, keepdim=True)`, never a global mean.
- Gradient-norm clipping is per copy. Compute `norm[c] = sqrt(sum over that copy's parameter
  slices)` and scale each copy's gradients by `min(1, max_norm / norm[c])`. `clip_grad_norm_` on
  the stacked tensors would compute one norm across all copies and couple them.
- The RND predictor-loss dropout mask (`update_proportion` in the cleanrl reference) is drawn per
  copy, and its normalization `sum(mask) / max(sum(mask), 1)` is per copy.
- The observation normalizer, the intrinsic-reward scale, and the learning-rate schedule are per
  copy.

Per-copy hyperparameters come free from this layout: store the intrinsic-reward coefficient, the
learning rate, the clip coefficient as `[copies, 1, 1]` tensors and the same kernel runs a
hyperparameter sweep instead of a seed sweep.

### 3.4 Per-copy optimizer

Adam moments live in the same flat layout as the parameters: `m` and `v` are single 1-D tensors the
same length as the flat parameter buffer. The update is then one elementwise kernel over three
arrays, entirely independent per element, so per-copy independence is automatic. The bias-correction
step counter can be a scalar if all copies update in lockstep, which they do.

If per-copy learning rates are wanted, the learning rate becomes a `[copies]` tensor expanded to the
flat layout once at construction time.

Do not use `torch.optim.Adam` over a list of six per-layer tensors in the hot path: that is a
multi-tensor kernel launch chain, and at these sizes it is a visible share of the 0.5 ms floor.
If `torch.optim.Adam` is used inside a CUDA graph anyway, it must be constructed with
`capturable=True`.

### 3.5 Per-copy randomness

Two things need randomness during training: sampling actions from the Gaussian policy, and the
minibatch permutation. Both must be independent per copy, reproducible, and stable when unrelated
parts of the design change (the project's keyed-seeding rule).

- **Do not use a `torch.Generator` per copy.** Drawing `n_copies` separate small tensors is
  `n_copies` kernel launches, and generator objects cannot be batched.
- **Do not rely on the global RNG stream either.** Drawing one `[copies, envs, act]` normal from the
  global generator is fast, but the numbers a given copy sees then depend on `n_copies` and on how
  many draws happened earlier, so a solo re-run of copy `k` will not reproduce it.
- **Use a counter-based generator written in tensor ops.** Key each copy by
  `key[c] = hash(base_seed, copy_index[c])`, and address each draw by a counter built from
  `(global_step, env_index, quantity_name)`. A Philox-style or Threefry-style round function is a
  handful of integer shifts, multiplies and XORs on `int32`/`int64` tensors; it compiles cleanly,
  captures in a CUDA graph (no hidden RNG state to advance), and gives byte-identical draws whether
  the copy runs alone or inside a batch of 128. Register one integer stream identifier per named
  quantity — action noise, minibatch permutation, RND dropout mask, reset noise — so adding a
  quantity later does not shift the existing ones.
- **Minibatch permutation per copy** without a shuffle kernel: draw uniforms `u` of shape
  `[copies, batch]` from that generator and take `perm = u.argsort(dim=1)`. `argsort` is a single
  batched kernel, has a static shape, and compiles.

### 3.6 Pitfalls, and what they cost

- **No BatchNorm anywhere.** Any normalization whose statistics are computed over a batch axis
  shared across copies couples the copies. LayerNorm and RMSNorm are safe because they reduce over
  the feature axis of each row independently. The running observation normalizer that RND needs is
  a per-copy statistic maintained by hand; it is not a BatchNorm.
- **No `.item()`, no `float(tensor)`, no `if tensor > x` in the step.** Each one forces a
  device-to-host synchronization and breaks the compiled graph. Everything logged (approximate KL,
  clip fraction, explained variance) stays a tensor and is copied to the host on the sparse logging
  cadence only, once every few thousand iterations.
- **No KL-based early stopping.** `if approx_kl > target: break` is data-dependent control flow.
  Either drop it, or replace it with a per-copy multiplicative mask on the loss so the shape of the
  computation never changes.
- **Two update styles, two separate compiled functions.** The single full-batch update and the
  four-epochs-of-four-minibatches update are built as two top-level functions selected once at
  construction. A runtime `if` inside the compiled region either specializes and recompiles or
  breaks the graph. This is also what the task statement asked for.
- **Static shapes throughout.** `n_copies`, `n_envs`, rollout length, minibatch count and minibatch
  size are fixed for a run. Compile with `dynamic=False`. Dynamic shapes force `reduce-overhead` to
  re-record a CUDA graph per shape, which is the single most common way this design loses its
  speed.
- **Static input addresses for CUDA graphs.** `reduce-overhead` requires input tensors at fixed
  addresses, and does not support graphs that mutate their inputs. Allocate the rollout buffers
  once at startup and copy into them in place; never allocate a fresh observation tensor per step.
- **`torch.func.vmap` restrictions**, if the `vmap` path is used: no mutation of arbitrary Python
  state inside the function, no data-dependent control flow, and randomness inside a vmapped
  function requires an explicit `randomness=` setting. The counter-based generator sidesteps the
  randomness question entirely because it is pure arithmetic.
- **Do not put the environment step outside the compiled region** if the environment is already a
  GPU tensor program (Module 1). The rollout step is only 57 to 213 us; a Python round trip per
  environment step would dominate it.

### 3.7 Ordered optimization list (PyTorch)

Do these in order and measure after each, keeping or reverting per the project's method.

1. Correct batched implementation in eager mode, `baddbmm` everywhere, per-copy reductions
   verified by the tests in §5. This alone is roughly 40 times faster than a loop over copies.
2. `torch.compile(dynamic=False)` on the rollout step and on each of the two update steps
   separately. Measured 1.3 to 1.5 times on the update at small batch.
3. `mode="reduce-overhead"` (CUDA graphs) once shapes are static and buffers are preallocated.
   Measured a further 1.2 to 1.4 times whenever the batch is small; no gain at large batch, so
   measure rather than assume.
4. Fuse the optimizer: one flat parameter buffer, one flat gradient buffer, one flat moment pair,
   one elementwise Adam kernel. This attacks the 0.5 to 0.6 ms floor directly, which at hidden 64
   is 100 percent of the update cost even at 512 copies.
5. Put the whole update phase inside one graph: all epochs and all minibatches, with the minibatch
   loop unrolled (4 by 4 is small) so there is one graph launch per iteration rather than sixteen.
6. Fold time and environment axes into the row axis for the update batch, so each layer is one
   batched GEMM over `T * n_envs` rows rather than one per timestep.
7. Fuse the elementwise tails: GAE, advantage normalization, ratio and clipped surrogate, RND
   feature error. Inductor does much of this automatically once the region is compiled; check the
   generated kernel count before hand-writing anything.
8. Only then consider precision. Allow TF32 for matmuls (all §1 and §3.1 numbers already have it
   on). Consider bf16 autocast for the trunk GEMMs afterwards, keeping parameters, the log
   probability, the ratio and the losses in FP32, and re-run the correctness tests, because the
   ratio and the KL estimate are the numerically delicate parts of PPO.
9. Last, and only if profiling still shows kernel-launch gaps in the rollout: a hand-written
   persistent kernel that keeps one copy's weights resident and walks the whole small MLP. At
   hidden 64 a copy's three layers are about 20 KB of weights, well inside the 228 KB of shared
   memory per SM, so this is feasible; at hidden 256 the middle layer alone is 256 KB and it is
   not. This is the idea behind the LLM megakernel work (§6) and it is a last resort, not a
   starting point.

## 4. JAX: recommended v0 design

### 4.1 Shape of the program

The pattern from PureJaxRL and the Anakin design is to write the entire training for one seed as a
pure function of a random key and then map it:

```python
train_fn = make_train(config)                      # rng -> final state, metrics
keys     = jax.random.split(jax.random.PRNGKey(base_seed), n_copies)
out      = jax.jit(jax.vmap(train_fn))(keys)
```

Everything downstream follows automatically: the network parameters, the optax state, the
environment state and the running normalizer statistics all gain a leading copy axis because they
are derived from the mapped key, and no code inside `train_fn` knows the copy axis exists.

**One deviation from PureJaxRL is recommended here.** PureJaxRL JIT-compiles the whole training run
because its runs finish in seconds to minutes. This project's runs are hours long, need a log every
20 minutes and must be resumable, and the whole-run form produces no output until it returns.
So: vmap and JIT **one training iteration** (collect `T` steps, then run the update), and drive it
from a short Python loop that carries the state.

```python
@partial(jax.jit, donate_argnums=(0,))
def iteration(state):                                # state: params, opt_state, env_state, obs, key
    ...
    return new_state, metrics

state = init(keys)                                   # every leaf has leading axis n_copies
for it in range(n_iterations):
    state, metrics = iteration(state)
    if it % log_every == 0:
        host_metrics = jax.device_get(metrics)       # the only synchronization
```

The Python loop costs one dispatch per iteration, which against a rollout of hundreds of
environment steps is nothing, and it buys sparse logging, checkpointing and a resume that satisfies
the project's resumable-generation rule.

### 4.2 Inside one iteration

- **Rollout**: `jax.lax.scan` over `T` environment steps, carrying `(env_state, obs, params, key)`
  and stacking the transitions. The environment step is the Module 1 JAX environment, itself vmapped
  over `n_envs`.
- **GAE**: `jax.lax.scan(..., reverse=True, unroll=16)`, exactly as PureJaxRL does.
- **Update, style A (one full-batch update, then discard)**: a single loss-and-gradient call on the
  whole batch, one `optimizer.update`, done.
- **Update, style B (4 epochs of 4 minibatches)**: `lax.scan` over epochs, and inside it `lax.scan`
  over minibatches. Shuffle by `perm = jax.random.permutation(key, batch_size)`, gather with
  `jnp.take(x, perm, axis=0)` on every leaf, then reshape to `[n_minibatches, mb_size, ...]` and
  scan over the leading axis.
- **Two separate functions**, chosen by a static config flag at build time, so the traced graph
  never contains a branch. Same reason as the PyTorch side.
- **Optimizer**: `optax.chain(optax.clip_by_global_norm(max_norm), optax.adam(lr))`. Under `vmap`
  the global-norm clip is computed **per copy**, which is what is wanted, and it is one of the
  places where the JAX version is safer than the PyTorch one by construction.
- **Randomness**: `jax.random.split` on the carried key each iteration and `jax.random.fold_in`
  keyed by `(iteration, quantity)` for named draws. Because the top-level key was split per copy,
  every derived key is already per copy. This satisfies the project's keyed-seeding rule directly.
- **Buffer donation**: `donate_argnums` on the carried state so XLA writes the new parameters and
  optimizer moments into the old buffers instead of allocating fresh ones. Donated buffers must not
  be reused afterwards, and only positional arguments can be donated.

### 4.3 Ordered optimization list (JAX)

1. Correct single-copy `train_iteration`, tested against the PyTorch implementation on one seed.
2. `jax.vmap` over the copy axis, `jax.jit` the result. Verify the independence tests of §5.
3. `lax.scan` for both the rollout and the update loops, so the compiled program size does not grow
   with the rollout length. Tune `unroll` on the GAE scan and the rollout scan (start at 1 and 16).
4. `donate_argnums` on the carried state.
5. Keep the whole environment inside the same JIT region as the network. This is the point of the
   Anakin design: one XLA program covers acting and learning with no host round trip.
6. Set the matmul precision explicitly (`jax.config.update("jax_default_matmul_precision", ...)`)
   and record which setting produced each number, so JAX and PyTorch are compared at the same
   precision. Leaving it at the default makes the two frameworks incomparable.
7. If more than one GPU is ever used, shard the copy axis with `pmap` or `shard_map`. The copy axis
   is embarrassingly parallel — this is the split the podracer paper uses (vmap to fill one core,
   pmap to fill the device).
8. Check the compiled program with `jax.jit(...).lower(...).compile().cost_analysis()` and the XLA
   dump before hand-optimizing anything.

## 5. Verifying that the copies are genuinely independent

Batching is only worth anything if copy `k` learns exactly what a solo run with seed `k` would
learn. These tests belong in the unit-test suite, not in a one-off script.

1. **Same seed twice is bit-identical.** Run the same configuration and base seed twice, compare
   every parameter tensor with an exact equality check. Any difference means an uncontrolled
   randomness source (a global generator, a non-deterministic kernel, uninitialized memory). Run
   this on both the eager and compiled paths — an exact match in eager and a mismatch after
   compilation points at the compiled region's randomness handling.
2. **Different seeds diverge.** With `n_copies` different base seeds, parameters must differ from
   the first update onward, and the spread must keep growing. Identical curves across copies mean
   the seeds never reached the sampling.
3. **The slice test — the strongest one.** Train `n_copies = C` with base seeds `s_0 .. s_{C-1}`
   for a few hundred iterations. Separately train `n_copies = 1` with base seed `s_k`. Copy `k`'s
   slice of the batched run and the solo run must agree. Expect exact agreement on the environment
   trajectories, actions and random draws; expect agreement to a small tolerance (relative 1e-5 or
   so, and it must not grow faster than ordinary floating-point drift) on the parameters, because
   cuBLAS may select a different tile shape for a batch of 1 than for a batch of C, and a different
   summation order gives different last bits. State this distinction explicitly in the test:
   bit-identity is required for test 1, a tolerance is expected for test 3.
4. **The perturbation test.** Change only copy 0's base seed and re-run. Every other copy's
   trajectory, actions and parameters must be byte-identical to the previous run. This catches any
   accidental reduction across the copy axis in one cheap run.
5. **The NaN-isolation test.** Inject a NaN into copy `j`'s observations for one iteration. After
   the update, copy `j`'s parameters must be NaN and **every other copy's parameters must be
   finite and unchanged**. This is the fastest way to find a stray `.mean()` that swallowed the
   copy axis, and it takes one iteration to run.
6. **Reduction audit.** A test that walks the loss function and asserts that every reduction
   carries an explicit `dim=` that excludes axis 0, and that the only reduction over axis 0 is the
   final `sum()` into the scalar handed to autograd.
7. **Agreement with the reference implementation.** One copy against the cleanrl-style single-copy
   PPO+RND, same seed, same hyperparameters, compared on the first few hundred iterations of loss,
   approximate KL, clip fraction and explained variance. This checks the algorithm, not the
   batching.
8. **Throughput sanity as a coupling detector.** If the measured per-copy throughput at 128 copies
   is not close to 128 times the throughput at 1 copy in the flat region of the §1 table, something
   is serializing — usually a Python loop over copies that crept back in, or a per-copy
   synchronization.

Record for each run: the base seed, `n_copies`, the per-copy seed list, the git hash, the
framework and matmul-precision setting, and the measured wall time. Two runs that claim the same
seed but used different precision settings will not match, and without the record that looks like a
correctness bug.

## 6. What the surveyed work contributes

- **PureJaxRL** (Lu et al.). The reference for the pattern: write the whole training as a pure
  function, `jit` it, `vmap` over seeds. Reported figures: about 10 times faster than a
  single-threaded CleanRL PyTorch baseline, and 2,048 PPO agents on CartPole trained in roughly
  half the wall time of one CleanRL agent, on one GPU; 512 agents for 1,024 generations of
  meta-evolution on an A40 in about 9 hours. The claimed "over 1000x" is that combination — a per
  agent figure, not a speedup of one agent. The measurements in §1 reproduce the mechanism on this
  hardware in PyTorch: 128 copies for the price of one, up to a knee.
- **Podracer architectures (Hessel et al., 2021)** — the Anakin and Sebulba designs. The useful
  ideas: make the environment step part of the same compiled program as the learning update so the
  whole thing is one XLA computation; `vmap` the unit of computation until one core is full, then
  `pmap` across cores. The second half only matters if this project moves past one GPU. Sebulba
  (separate acting and learning devices) is the design to reach for if the environment ever stops
  being a GPU tensor program.
- **Stoix** implements both the Anakin and the Sebulba shapes over the same algorithms, and is the
  cleanest place to read how the two are kept interchangeable behind one config.
- **Cleanba** is about a different failure: distributed actor-learner setups whose learning curves
  are not reproducible across hardware even with fixed hyperparameters, because the number of
  actors changes the data ordering. Relevant here as a warning — the batched design must not let
  `n_copies` change what any one copy sees, which is exactly what the tests in §5 check.
- **vLLM and SGLang** contribute two transferable ideas. First, capture the repeated step as a CUDA
  graph and replay it, with one graph per shape, because a captured graph fixes kernel arguments,
  grid dimensions and pointers — hence the static-shape and static-address requirements in §3.6.
  Second, their piecewise-graph approach (split the program at a few points and capture each piece)
  is the fallback if one part of this training step turns out to need a dynamic shape: capture the
  rest.
- **Megakernel work** (persistent kernels that keep threadblocks resident and walk the whole model
  in place) is the answer to what CUDA graphs cannot fix: kernel boundaries still serialize, still
  cost a launch each, and still flush intermediates through memory. It is item 9 on the PyTorch
  list, viable only at hidden 64.
- **Batched small-GEMM literature** agrees with the §2 measurement: batched GEMM at small `K` (64
  to 256) runs well below peak on every library, and the gap is not closed by switching library.
  This is a reason to stop optimizing the GEMMs early and spend the effort on kernel count instead.

## 7. Open items to measure before Module 2 is fixed

- The knee moves with hidden width and with rows per copy. Re-run `bench_copy_scaling.py` at the
  hidden width and rollout shape finally chosen, and use the flat region to pick `n_copies`.
- Whether the compiled `vmap` path beats the compiled `baddbmm` path at the final shape. It did in
  three of five measured shapes, and it is a one-line swap.
- How much of the 0.5 ms update floor is Adam. Profile the compiled update and count kernels before
  writing the fused optimizer.
- Whether bf16 for the trunk GEMMs changes the learning curve. Measure the speed first, then decide
  whether the correctness re-test is worth it.
- The same copy-scaling curve for the JAX implementation, at the same precision setting, so the
  two framework numbers in the final report are comparable.

## Sources

- [PureJaxRL](https://github.com/luchris429/purejaxrl) and [Achieving 4000x Speedups and Meta-Evolving Discoveries with PureJaxRL](https://chrislu.page/blog/meta-disco/)
- [Podracer architectures for scalable Reinforcement Learning](https://arxiv.org/pdf/2104.06272)
- [Stoix](https://github.com/EdanToledo/Stoix)
- [Cleanba: A Reproducible and Efficient Distributed Reinforcement Learning Platform](https://arxiv.org/abs/2310.00036)
- [PyTorch model ensembling tutorial (stack_module_state, functional_call, vmap)](https://docs.pytorch.org/tutorials/intermediate/ensembling.html)
- [torch.func UX limitations](https://docs.pytorch.org/docs/stable/func.ux_limitations.html)
- [CUDAGraph Trees (torch.compile reduce-overhead)](https://docs.pytorch.org/docs/main/user_guide/torch_compiler/torch.compiler_cudagraph_trees.html)
- [torch.compile documentation](https://docs.pytorch.org/docs/2.12/generated/torch.compile.html)
- [JAX PRNG design](https://docs.jax.dev/en/latest/jep/263-prng.html) and [buffer donation](https://docs.jax.dev/en/latest/buffer_donation.html)
- [Optax documentation](https://optax.readthedocs.io/en/latest/getting_started.html)
- [CUDA Graph in SGLang](https://sgl-project-sglang-93.mintlify.app/optimization/cuda-graph) and [Inside vLLM](https://vllm.ai/blog/2025-09-05-anatomy-of-vllm)
- [Compiling LLMs into a MegaKernel](https://zhihaojia.medium.com/compiling-llms-into-a-megakernel-a-path-to-low-latency-inference-cf7840913c17)
- [New cuBLAS 12.0 Features and Matrix Multiplication Performance on NVIDIA Hopper GPUs](https://developer.nvidia.com/blog/new-cublas-12-0-features-and-matrix-multiplication-performance-on-nvidia-hopper-gpus/)
- [Fast Batched Matrix Multiplication for Small Sizes](https://www.netlib.org/utk/people/JackDongarra/PAPERS/ipdps-batched-2019.pdf)
