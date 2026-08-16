# Applied changes — ~9% faster, and provably the same experiment

Everything in this file was measured on a node held exclusively, against a baseline measured in the
same job on the same node, and — the condition that decides whether it may be applied at all —
**produced a record identical to the baseline's, field for field**: same `eval_history`, same
`train_history`, same `train_episode_history`, same `visit_counts`. Across the campaign that check
passed on **176 of 176** candidate-versus-baseline comparisons — 38 single-worker (jobs 6536785,
6536795, 6536801, covering all three configurations) and 138 packed-worker (jobs 6536803, 6536805,
6536808, 46 workers each) — plus 4 baseline-repeat comparisons that confirm the workload is
deterministic run to run, which is what makes the other 176 mean anything. Nothing here re-defines
the experiment, so every result already produced by run 5, run 6 and the 1M sweep stays valid.

## The headline

| | per worker | one 4M run | across the sweep's 900 runs |
|---|---|---|---|
| baseline | 15.366 steps/s | **72.3 h** | — |
| with this stack | 16.821 steps/s | **66.1 h** | — |
| **saved** | **+9.5%** | **6.2 h per run** | **~5,600 worker-hours** |

Measured with 46 workers packed on one 24-core node — the shape the sweep actually runs — at 40,000
steps per worker (job 6536808). Single-worker measurements on an idle node give the same answer:
+7.8% to +9.0% (job 6536801).

## The stack, and what each part is worth

Single-worker gains, best-of-2, job 6536801 (clean build, all three configs):

| # | change | how it is applied | alg2.3 | gtposvel | run5origrnd | record |
|---|---|---|---|---|---|---|
| 1 | **tcmalloc** — `LD_PRELOAD=/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4` | one line in the worker `.slurm`; **no code change** | +6.1% | +5.3% | +6.4%¹ | identical |
| 2+3 | **`--opt_polyak_foreach=True --opt_torch_reward=True`** — `torch._foreach_` soft target update instead of SB3's `zip_strict` loop, and the reward combine kept in torch | flags that already ship in `train.py`; set them in the queue markers' `fixed` dict; **no code change** | +2.2% | +3.3% | +4.3% | identical |
| 4 | **foreach Adam/SGD** — torch's CPU default is the single-tensor Python loop, and SAC steps three optimizers plus the RND predictor every step | small code change (`build_sac`, RND optimizer construction) | +1.0% | +0.3% | +1.7% | identical |
| 5 | **cached observation-norm tensors** — `RND._normalize_obs` rebuilt `mean`/`var`/`sqrt(var+1e-8)` from numpy on every call, twice per gradient step plus once per env step, though the statistics move only inside `RND.update()` | small code change (`methods/rnd.py`) | +0.6% | +0.3% | +0.2% | identical |
| 6 | **`gc.freeze()` after construction** — the env, model and buffer live for the whole run and were rescanned by every generational collection | three lines in `train4m.run4m` | +1.0% | +0.5% | +1.2% | identical |
| | **all together** | | **+8.9%** | **+7.8%** | **+9.0%** | **identical** |

¹ corrected: experiment 1's +6.9% for this cell straddled a mid-experiment edit to the live
trainer; experiment 3 re-measured it clean.

**Two of the six carry ~80% of the win and need no code at all** (items 1-3: tcmalloc plus the two
shipped flags). Items 4-6 are ~+1.9% combined and require touching files that 450 runs are executing
right now.

## Why each one cannot change a result

1. **tcmalloc** replaces the allocator, not the arithmetic. Addresses change; floats do not.
2. **`opt_polyak_foreach`** performs `target *= (1-τ); target += τ·param` as two `torch._foreach_`
   kernels over the same tensors in the same element order as SB3's Python loop — proven bit-exact
   by `analysis/2026-06-25-run-profiling/analysis/code/test_optimizations.py::test_polyak_foreach_bit_exact`.
3. **`opt_torch_reward`** does `ext + β·int` in float32 torch instead of float32 numpy — the same
   IEEE operations on the same values — and skips a re-wrap of tensors SB3 had already built.
4. **foreach Adam/SGD** applies the same per-element update formula; only the loop moves from
   Python into a fused kernel.
5. **cached norm tensors** hand back the same values, converted once per statistics update instead
   of once per call.
6. **`gc.freeze()`** moves objects out of the collector's scan set. It frees nothing and computes
   nothing.

Points 1-3 and 6 are safe by construction; 4 and 5 are the kind of claim that deserves evidence
rather than argument, which is why the record check ran on all three configurations and again on
every one of 46 packed workers.

## How this reaches the 96-hour training

```bash
# 1. tcmalloc — one line in the worker .slurm, next to the other exports  (~6% of the ~9%)
export LD_PRELOAD=/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4

# 2+3. the two shipped switches, into every not-yet-claimed marker of the live sweep  (~3%)
python analysis/2026-08-13-ext4m-efficiency/analysis/code/apply_optflags.py \
    --queue <run>/queue/<sweep_id>             # dry run first
python .../apply_optflags.py --queue <run>/queue/<sweep_id> --apply
```

`apply_optflags.py` is safe against a **live** sweep: it rewrites each marker in place through the
same file descriptor and never creates a directory entry, so it cannot resurrect a claimed marker
into `pending/` and cannot make a run execute twice. And because the flags are record-identical, a
sweep that is half-flagged is still one uniform experiment — runs started before and after the edit
produce the same numbers. That property is exactly what the 223 record comparisons verify; without
it no live edit would be defensible.

**Items 4-6 are deliberately not applied.** They are code changes to `train.py` / `methods/rnd.py` /
`train4m.py`, and 450 runs are executing out of those files. Editing them mid-flight would leave the
sweep straddling two code versions for +1.9%, when the two no-code items already deliver ~7%. The
exact implementations are in `analysis/code/patched_entry.py`, each already proven identical, ready
for the next sweep boundary.

## What was tried and rejected

Measured, record-identical, but not worth it: `mkldnn_off` (never beat the stack; oneDNN earns its
dispatch cost on these shapes), `MALLOC_ARENA_MAX=1` (−0.1%, so tcmalloc's win is its allocator, not
the arena count), `inference_mode` for the bonus (**−0.8%**: inference-mode tensors cannot be
consumed by autograd-tracked code, so the bonus needs a `.clone()` costing more than the bookkeeping
it removes). `torch.set_num_interop_threads(1)` was found independently and applied to the live
trainer by the sweep owner at 05:06:59 (~+0.5%); torch raises if it is set twice, so this side keeps
it as a no-op.

Everything faster-but-different is in `future_changes.md`, with its code in `../deferred_experiments/`.

## Honest limits of these numbers

- The longest measurement is 40,000 steps per worker: **1% of a 4M run**. The changes are per-step
  overheads, and the gain held from 3,000 to 40,000 steps, but the late-run regime (a full
  1M-transition replay buffer, a large resident set at step 2M) was never measured.
- One packed pair is worth about ±3%: the shared `/p/rlprojects` filesystem, which the live sweep
  writes to, produced a −10% outlier in one single-worker run and a −3% dip in the 12,000-step
  packed pair. Differences smaller than that were not treated as real.
- All measurements are on `cortado01` (2×12 cores, Xeon class). Other node classes in the sweep
  (`bigcat`, `affogato`, `slurm*`) were not measured; the allocator win in particular could differ
  with core count and memory bandwidth.
