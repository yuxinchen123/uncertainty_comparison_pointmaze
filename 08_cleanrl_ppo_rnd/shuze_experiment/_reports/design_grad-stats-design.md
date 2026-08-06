## 1. What `clip_grad_norm_` returns — confirmed from source

`torch/nn/utils/clip_grad.py` (torch 2.10.0+cu128, at `/u/sl5nw/.local/lib/python3.11/site-packages/torch/nn/utils/clip_grad.py`):

```python
def clip_grad_norm_(parameters, max_norm, norm_type=2.0, error_if_nonfinite=False, foreach=None) -> torch.Tensor:
    r"""...
    Returns:
        Total norm of the parameter gradients (viewed as a single vector).
    """
    ...
    grads = [p.grad for p in parameters if p.grad is not None]
    total_norm = _get_total_norm(grads, norm_type, error_if_nonfinite, foreach)
    _clip_grads_with_norm_(parameters, max_norm, total_norm, foreach)
    return total_norm
```

Three facts follow, all load-bearing:

1. **It is the norm BEFORE clipping.** `total_norm` is computed from `grads` and only then handed to `_clip_grads_with_norm_`, which mutates the gradients in place. The returned tensor is not recomputed afterwards.
2. **It is computed over exactly the parameters passed in** — `grads` is built from that argument list, nothing wider.
3. **It is a GPU tensor, not a float.** So the headline number *is* free: today's call already computes it and the trainer throws it away. Capturing it costs zero extra kernels; only reading it on the host costs anything.

The clip factor itself, from `_clip_grads_with_norm_`:

```python
clip_coef = max_norm / (total_norm + 1e-6)
# Note: multiplying by the clamped coef is redundant when the coef is clamped to 1, but doing so
# avoids a `if clip_coef < 1:` conditional which can require a CPU <=> device synchronization
# when the gradients do not reside in CPU memory.
clip_coef_clamped = torch.clamp(clip_coef, max=1.0)
```

PyTorch's own comment states the design rule this whole specification follows.

**The consequence that reframes the question.** The run clips over `list(agent.parameters()) + list(rnd_model.predictor.parameters())` (line 437), so the predictor is never clipped on its own — every gradient in the optimizer gets multiplied by the *same* `clip_coef_clamped`. Therefore **"how often is the predictor's gradient clipped" is exactly "how often did the combined norm exceed 0.5"**, and that number is already sitting in a discarded return value. The predictor-only norm answers the different, and more useful, follow-up: *is the predictor the reason the clip fired, or is it collateral damage from the policy gradient?*

## 2. Collection design — no host synchronisation per optimizer step

Written, tested, and in the repo: **`/p/rlprojects/RND/08_cleanrl_ppo_rnd/src/grad_norm_stats.py`**

Structure: a preallocated GPU ring buffer of two columns (combined norm, predictor norm), one row per optimizer step, sized to exactly one logging interval; a GPU moment accumulator behind it; one host transfer at logging time. All the arithmetic the brief asks for — sum, sum of squares, max, count of steps over the threshold — happens in the accumulator, but *batched*: it runs once per interval over the whole buffer instead of once per step. That is why the per-step cost is 2 kernels instead of ~9.

```python
class GradNormStats:
    """Clip the combined gradient and record its norm and the predictor's norm, without any host sync."""

    def __init__(self, agent_parameters, predictor_parameters, clip_threshold, capacity, device):
        # The parameter order is fixed and never changes, so the predictor's gradients are always the
        # tail of the list and `split` is a constant slice index into the norms vector.
        # before: agent 20 tensors (1,407,476 params), predictor 12 tensors (2,203,296 params)
        # after:  self.parameters is 32 tensors, self.split == 20, norms[20:] is the predictor's
        self.agent_parameters = list(agent_parameters)
        self.predictor_parameters = list(predictor_parameters)
        self.parameters = self.agent_parameters + self.predictor_parameters
        self.split = len(self.agent_parameters)
        self.clip_threshold = float(clip_threshold)
        self.checked = False

        # Column 0 is the combined pre-clip norm, column 1 is the predictor-only pre-clip norm.
        # Capacity is one logging interval's optimizer steps, so the buffer folds once, at drain time.
        self.buffer = torch.zeros((capacity, 2), dtype=torch.float32, device=device)
        self.fill = 0

        # Ten running sums and counts, one running maximum pair, one running minimum. All on device,
        # and all float64: the standard deviation comes from a sum minus a sum of squares, which in
        # float32 over a few hundred steps loses about four of the seven digits to cancellation.
        self.sums = torch.zeros(10, dtype=torch.float64, device=device)
        self.maxima = torch.zeros(2, dtype=torch.float64, device=device)
        self.minimum_scale = torch.ones(1, dtype=torch.float64, device=device)

    def clip_and_record(self):
        """Clip the combined gradient exactly as clip_grad_norm_ does, keeping the per-tensor norms."""
        grads = [p.grad for p in self.parameters]
        if not self.checked:
            self._check_grads(grads)

        # before: norms is a list of 32 zero-dim tensors; total_norm = ||stack(norms)||
        # after:  norms shape (32,); total_norm = ||norms||; predictor_norm = ||norms[20:]||
        norms = torch.stack(torch._foreach_norm(grads, 2.0))
        total_norm = torch.linalg.vector_norm(norms, 2.0)
        predictor_norm = torch.linalg.vector_norm(norms[self.split:], 2.0)
        torch.nn.utils.clip_grads_with_norm_(self.parameters, self.clip_threshold, total_norm)

        # Park the two numbers and return. No comparison, no reduction, no host copy in the hot loop.
        torch.stack((total_norm, predictor_norm), out=self.buffer[self.fill])
        self.fill += 1
        if self.fill == self.buffer.shape[0]:
            self._fold()
```

The fold — this is where every comparison lives, and every comparison stays a GPU tensor:

```python
    def _fold(self):
        """Reduce the filled part of the ring buffer into the running accumulators, on the GPU."""
        if self.fill == 0:
            return
        rows = self.buffer[: self.fill].double()
        combined, predictor = rows[:, 0], rows[:, 1]

        # A non-finite norm is only reachable under float16 autocast, where the gradient scaler makes
        # the optimizer skip the step. Counted separately, excluded from every mean and maximum.
        # before: combined = [0.31, inf, 0.72], threshold 0.5
        # after:  usable = [1, 0, 1]; steps counted 2; clipped 1; non-finite 1
        finite = torch.isfinite(combined) & torch.isfinite(predictor)
        zero = torch.zeros((), dtype=combined.dtype, device=combined.device)
        combined = torch.where(finite, combined, zero)
        predictor = torch.where(finite, predictor, zero)
        usable = finite.to(combined.dtype)

        # `combined > threshold` is a GPU bool tensor. It is never read on the host; it is cast to
        # the accumulator dtype and summed on the GPU, which is what makes the count free.
        fired = (combined > self.clip_threshold).to(combined.dtype) * usable
        # The predictor's share of the SQUARED norm, because squared shares partition: the agent's
        # share is one minus this one. The clamp only guards the all-zero-gradient step.
        fraction = (predictor * predictor) / (combined * combined).clamp_min(1e-30)
        # The factor every gradient in the optimizer is actually multiplied by. 1.0 means no clipping.
        scale = torch.clamp(self.clip_threshold / (combined + 1e-6), max=1.0)
        scale = torch.where(finite, scale, torch.ones_like(scale))

        self.sums += torch.stack((
            usable.sum(), (combined * usable).sum(), (combined * combined * usable).sum(),
            fired.sum(), (predictor * usable).sum(), (predictor * predictor * usable).sum(),
            (fraction * usable).sum(), (fraction * fired).sum(), (scale * usable).sum(),
            (1.0 - usable).sum(),
        ))
        self.maxima = torch.maximum(
            self.maxima, torch.stack(((combined * usable).max(), (predictor * usable).max()))
        )
        self.minimum_scale = torch.minimum(self.minimum_scale, scale.min().reshape(1))
        self.fill = 0

    def drain(self):
        """Fold, copy the accumulators to the host in ONE transfer, reset them, and name the numbers."""
        self._fold()
        # One transfer, 13 float64 values (104 bytes). Everything after this line is host arithmetic.
        host = torch.cat((self.sums, self.maxima, self.minimum_scale)).cpu().tolist()
        self.sums.zero_(); self.maxima.zero_(); self.minimum_scale.fill_(1.0)
        ...   # names the fields listed in §5
```

Why the ring buffer rather than accumulating moments per step: accumulating per step needs a square, a `>` comparison, a cast, a division, a `stack`, an `add_`, and a `maximum` — about 9 tiny kernels every step. Parking two floats needs 2. The moment accumulator is still there; it just runs 400 steps' worth at a time.

Things that would have reintroduced a sync, and are absent: `.item()`, `.cpu()`, `float(...)`, `if total_norm > threshold`, `bool(...)`, `print` of a tensor, and `torch.tensor([...], device=device)` built from a Python list. `self.fill` is a host-side Python integer and `self.buffer[self.fill]` is a host-side index into a device tensor — no sync.

Aside, in the same loop: `if args.target_kl is not None and approx_kl > args.target_kl` (line 820) *is* a host sync, four per iteration. It is dead today because `target_kl` defaults to `None` and Python short-circuits, but anyone who sets it pays for it.

**Measured op counts** (`TorchDispatchMode`, real network shapes, one full minibatch — forward, backward, clip, Adam step):

| | ATen ops per minibatch | real kernel launches added |
|---|---|---|
| today | 675 | — |
| with the statistics | 679 | **2** (`linalg_vector_norm` on 12 floats, `stack.out` writing 2 floats) |

The other two of the four added ops are `slice` and `select` — views, no kernel. The fold costs 61 ATen ops **once per 400 steps**. The drain costs 4 ops including the single transfer.

## 3. The predictor-only norm, cheapest correct way

**Reuse the per-tensor norms `clip_grad_norm_` already computes and discards.** `_get_total_norm` calls `torch._foreach_norm(grads)`, giving one scalar norm per gradient tensor, stacks them, and takes the vector norm of the stack. Doing those three steps by hand keeps the stack; the predictor's norm is then one vector norm over a 12-element slice — it never touches gradient memory a second time.

**This is bit-identical to the current clip**, verified: same `total_norm` to the bit, and `torch.equal` on every clipped gradient, because `clip_grads_with_norm_` is literally the function `clip_grad_norm_` calls, given literally the same `total_norm`. (Pinned by `test_clipping_is_bit_identical_to_clip_grad_norm`.) The one assumption is that all 32 gradients form a single (device, dtype) group so the ordering matches — checked once, and it raises rather than silently mis-attributing.

**Cost.** The predictor is **2,203,296 parameters across 12 tensors** (largest: the `512 x 3136` linear, 1,605,632 values). The agent is 1,407,476 across 20; combined 3,610,772 across 32. Extra kernel launches per optimizer step attributable to the predictor number: **1**. Extra gradient memory traffic: **zero**.

The obvious alternative — a second `torch._foreach_norm` over the 12 predictor gradients — costs 3 extra ATen ops (measured) and re-reads 8.81 MB of gradients per step. At ~400 GB/s that is ~22 µs per step, ~0.35 ms per iteration, ~0.012% on jaguar03. Still small, but about ten times the reuse route for a number the reuse route gets for free.

## 4. Alternatives, ranked

1. **Reuse the per-tensor norms (recommended).** Exact predictor norm on every step, 1 extra kernel, bit-identical clipping. Gives up: nothing. Costs: a 12-line refactor of one call site and a one-time invariant check.
2. **Combined clip fraction only, no predictor norm.** Free — literally the discarded return value. This answers the question *as literally asked*, exactly and with no approximation, because the predictor is clipped if and only if the combined norm exceeds 0.5. What it gives up is the actionable half: you cannot tell whether the predictor is being throttled by its own gradient spikes (an RND-specific failure — the predictor loss spikes on genuinely novel frames, which is precisely when you least want its step shrunk) or whether the policy gradient spiked and the predictor is collateral damage. Those two situations have opposite fixes: the first points at `update_proportion` / the RND loss scale, the second at clipping the two groups separately. If you only ever ship one number, ship this one; but it will not close the investigation.
3. **Second `_foreach_norm` over the predictor only.** Same answer as (1), ~10x the cost, and duplicates work already done. Only worth it if you refuse to touch the clip call site.
4. **Subsample: predictor norm on every k-th step.** Given (1) costs one kernel, this saves nothing worth having; it is only relevant on top of (3). It gives up precision on the share (noise scales as `1/sqrt(n/k)`) and, worse, invites a **bias** trap: a fixed stride of 16 always samples the same position in the 4-epoch x 4-minibatch cycle, and gradient magnitude is systematically larger in epoch 0 than epoch 3 because the parameters have already moved. If you must subsample, use a stride coprime with `update_epochs * num_minibatches`, or a random offset per iteration.
5. **Derive the predictor share from the combined norm without a second pass — rejected as impossible.** The combined squared norm is `agent_norm**2 + predictor_norm**2`: one equation, two unknowns. Nothing recovers the split. The only available "derivation" is to assume the share is constant, which is exactly the assumption under test.

Not a logging option but the intervention the statistic exists to motivate: **clip the agent and the predictor to separate thresholds.** That changes the numbers, so it belongs behind an `--opt_*` flag in the second group, and only after the statistics say it is warranted.

## 5. Fields to log

Appended to the existing `eval_entry` dict, so they land in `eval_history` beside `losses/*` and `charts/*`. Every one covers the logging interval (25 policy updates = 400 optimizer steps) and resets after.

| field | meaning | mean or max |
|---|---|---|
| `grad/clip_threshold` | the clip setting, 0.5, echoed so the record reads on its own | constant |
| `grad/optimizer_steps_counted` | how many optimizer steps these numbers cover | count |
| `grad/clip_fired_fraction` | fraction of steps where the combined norm exceeded the threshold, so every gradient — the predictor's included — was scaled down. **This is the answer to the question.** | mean |
| `grad/mean_combined_norm_before_clipping` | average pre-clip norm over the agent and predictor together | mean |
| `grad/max_combined_norm_before_clipping` | the single largest pre-clip norm in the interval | **max** |
| `grad/standard_deviation_combined_norm_before_clipping` | spread of the pre-clip norm; separates "always a bit over" from "usually fine, occasionally enormous" | derived from mean + mean of squares |
| `grad/mean_predictor_norm_before_clipping` | average pre-clip norm of the predictor's gradients alone | mean |
| `grad/max_predictor_norm_before_clipping` | the predictor's largest pre-clip norm | **max** |
| `grad/standard_deviation_predictor_norm_before_clipping` | spread of the predictor's own norm | derived |
| `grad/mean_predictor_fraction_of_squared_norm` | the predictor's share of the squared combined norm; the agent's share is one minus this. Squared shares are used because they partition — plain norm ratios do not add to one | mean |
| `grad/mean_predictor_fraction_of_squared_norm_when_clipped` | the same share, restricted to the steps where the clip actually fired. **This is the number that says whether the predictor causes the clipping.** `None` if the clip never fired | conditional mean |
| `grad/mean_scale_applied_to_gradients` | average of `min(0.5 / (norm + 1e-6), 1)` — the factor the gradients were actually multiplied by. 1.0 means never clipped. Strictly more informative than the fire fraction: it says *how hard*, not just *how often* | mean |
| `grad/min_scale_applied_to_gradients` | the harshest single step in the interval | **min** |
| `grad/nonfinite_norm_steps` | steps whose norm was not finite; only reachable under `--opt_amp_fp16`, where the gradient scaler skips the step. Excluded from every mean and maximum above | count |

How to read them together: high `clip_fired_fraction` with a small `mean_predictor_fraction_of_squared_norm_when_clipped` means the policy gradient is the culprit and the predictor is being throttled for someone else's spike. High fire fraction with a large fraction-when-clipped means the RND predictor's own loss is the driver. `mean_scale_applied_to_gradients` near 1.0 with a low `min_scale` means it fires rarely but hard.

**Optional extra, free if you want it:** because the buffer capacity equals one interval, the buffer holds the *whole* interval at fold time, so `torch.quantile(combined, torch.tensor([0.5, 0.9, 0.99]))` gives exact percentiles for three more kernels per interval. Caveat: this is only exact while no mid-interval fold happens (it does not happen with the sizing below).

## 6. Verification

Written and passing: **`/p/rlprojects/RND/08_cleanrl_ppo_rnd/tests/test_grad_norm_stats.py`** — 7 passed, 1 skipped (`/p/rlprojects/RND/.venvs/exploration/bin/python -m pytest tests/test_grad_norm_stats.py`).

The construction that makes it deterministic: four scalar parameters, two "agent" and two "predictor", with gradients `(3s, 4s)` and `(0, 12s)` — a 5-12-13 triangle. For a power-of-two `s` every value and every norm is exact in float32: agent norm `5s`, predictor norm `12s`, combined norm `13s`.

- `test_clipping_is_bit_identical_to_clip_grad_norm` — random gradients over six shapes, clipped both ways; asserts the recorded norm equals `clip_grad_norm_`'s return exactly and `torch.equal` on every gradient. This is the regression guard for the refactor.
- `test_reported_statistics_match_hand_computed_values` — four steps at `s = 0.25, 1, 2, 4` against threshold 13, so the combined norms are 3.25, 13, 26, 52. The step sitting **exactly on** the threshold pins the comparison convention (strict `>`, so it does not count). Checks: fire fraction 0.5; mean 23.5625; max 52; standard deviation `sqrt(334.69921875)`; predictor mean 21.75, max 48; share `144/169` both overall and when clipped; mean scale 0.6875; min scale 0.25; non-finite 0.
- `test_gradients_are_actually_scaled_by_the_reported_factor` — one step at norm 26 against threshold 13; asserts the gradients that reach the optimizer are exactly halved and the reported scale is 0.5. Ties the *reported* number to the *applied* number.
- `test_folding_at_capacity_matches_a_buffer_that_never_folds` — eight steps through a capacity-2 buffer against a capacity-8 buffer; every field must agree. This is the guard on the batched-accumulator arithmetic.
- `test_draining_resets_the_accumulators` — a second drain reports zero steps, not the previous interval again.
- `test_nonfinite_norms_are_counted_separately_and_left_out_of_the_means` — an `inf` gradient is counted as non-finite and excluded from mean and max.
- `test_missing_gradient_raises_instead_of_mis_attributing_the_split` — a `None` gradient raises rather than silently shifting the predictor slice.
- `test_recording_performs_no_host_device_synchronisation` — wraps four recorded steps in `torch.cuda.set_sync_debug_mode("error")`, which raises on any host-device sync. **This is the direct test of the whole constraint.** It is currently **skipped** — the login node has no CUDA. It must be run on a GPU node before this is merged, and the same guard should be wrapped around one full real iteration.

## 7. Overhead, and how to confirm it

Baseline from the existing profiling data (`shuze_experiment/2026-08-05_profilling/data/profile_*.json`, 16 cpus per run, all options on):

| node | GPU | iteration (s) | update phase (s) | per optimizer step (ms) |
|---|---|---|---|---|
| jaguar03 | RTX A4500 | 3.031 | 0.978 | 61 |
| cheetah02 | RTX 4000 Ada | 3.508 | 1.276 | 80 |
| adriatic01 | Quadro RTX 4000 | 5.280 | 2.219 | 139 |
| jaguar02 | A16 | 6.321 | 4.006 | 250 |

Added work per iteration: **32 tiny kernel launches** (2 per step x 16 steps), each touching 12 or 2 floats; plus one fold of ~57 kernels every 25 iterations; plus one 104-byte device-to-host copy every 25 iterations, landing inside the block that *already* synchronises six times for `losses/value_loss`, `losses/policy_loss` and the rest — so it adds **no new synchronisation point at all**.

At a deliberately pessimistic 8 µs per tiny launch on these older cards: 0.26 ms per iteration, i.e. **0.0085% on jaguar03**, and smaller in relative terms on every slower node. The fold amortises to ~0.02 ms per iteration. **Total under 0.02% everywhere.** In op-count terms the same conclusion: 679 vs 675 ATen ops per minibatch, +0.6% of ops and +0.3% of kernel launches.

For scale, the thing this design refuses to do: an `.item()` on `total_norm` per step forces the CPU to wait for every kernel queued for that minibatch, 16 times per iteration. The ladder row that removed three such syncs measured between -0.1% and +0.5% across the seven profiled GPUs — small, because at 16 dedicated cpus the loop is GPU-bound and the CPU has already run ahead by the time the sync lands. That margin is exactly what disappears in the packed configuration the real sweep uses, where several runs share the cpus. There is no reason to spend it when the number is available for free.

**What I would measure, in this order:**

1. **Correctness of the no-sync claim, not an estimate of it.** Run one full training iteration inside `torch.cuda.set_sync_debug_mode("error")` on a GPU node. It either raises or it does not. This is the only check that matters for the hard constraint, and it is binary. Also un-skips `test_recording_performs_no_host_device_synchronisation`.
2. **Kernel-launch delta, directly.** One iteration under `torch.profiler.profile(activities=[CPU, CUDA])`; count launches and sum `cuda_time_total` for `aten::linalg_vector_norm` and `aten::stack`. Expected: +32 launches per iteration and a summed CUDA time in the tens of microseconds. This measures the overhead instead of inferring it from wall clock, which is the point — see (3).
3. **Wall clock A/B, interleaved, and expect a null result.** Reuse `bench_scratch/bench_ppo_rnd.py`: same node, same seed, warmup then at least 8 measured iterations, arms ABABAB at least 5 times each, reporting `iteration_seconds_mean` and `update_seconds_mean`. **Set the acceptance bar as a null result, not a small number:** adjacent no-op rows of the existing ladder on jaguar03 measured 3.297 / 3.311 / 3.293 s — a 0.55% spread between variants that differ by essentially nothing. The predicted overhead is roughly fifty times below that floor, so the honest expected outcome is "indistinguishable from noise", and a *measurable* slowdown is the signal that something is wrong with the implementation, not a cost to accept.
4. **In the configuration that actually runs.** Repeat (3) in the packed layout (several runs per GPU sharing cpus), because CPU launch time is the only regime where 32 extra launches could matter, and it is the regime the sweep runs in.

## Splice into the trainer

Not applied — `ppo_rnd_envpool_shuze.py` is unchanged. Three edits:

```python
# after line 44, next to the other local imports
from grad_norm_stats import GradNormStats
```

```python
# after line 438 (optimizer = optim.Adam(...))
    # One optimizer step per minibatch per update epoch, and the record logs every
    # log_every_updates policy updates, so this is exactly one logging interval of steps: the ring
    # buffer folds once, at drain time, and never mid-interval.
    # before: 4 update epochs x 4 minibatches x 25 updates
    # after:  capacity 400 rows x 2 columns of float32 = 3.2 KB of device memory
    grad_stats = GradNormStats(
        agent.parameters(), rnd_model.predictor.parameters(),
        clip_threshold=args.max_grad_norm,
        capacity=args.update_epochs * args.num_minibatches * args.log_every_updates,
        device=device,
    )
```

```python
# lines 811 and 817, both branches
-                        nn.utils.clip_grad_norm_(combined_parameters, args.max_grad_norm)
+                        grad_stats.clip_and_record()
```

```python
# inside the eval_entry dict, after "losses/old_approx_kl"
                    **grad_stats.drain(),
```

Notes on the splice. The `if args.max_grad_norm:` guards stay exactly as they are, so a run with clipping off records nothing and constructs no statistics it cannot fill. The float16 branch already calls `scaler.unscale_(optimizer)` before the clip, so the recorded norms are true unscaled norms; steps the scaler then skips show up as `grad/nonfinite_norm_steps` rather than polluting the means. No checkpoint change is needed — the accumulators cover one logging interval, and a resume simply starts a fresh one. I would **not** put this behind an `opt_*` flag: 2 kernels per step does not warrant another axis in the profiling matrix, and a statistic that is off by default is a statistic nobody has.