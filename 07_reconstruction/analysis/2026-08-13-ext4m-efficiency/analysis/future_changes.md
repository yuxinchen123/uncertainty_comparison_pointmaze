# Future changes — meaningful, but they would invalidate the results already produced

Everything here makes the training faster and is worth doing **for a future sweep that starts a
fresh comparison**. None of it is applied now, because each one changes numbers that are already
recorded in run 5, run 6 and the 1M sweep — the user's constraint is that those stay valid, and a
speed-up that silently re-defines the experiment is not a speed-up.

The rule used to sort a candidate into this file rather than `applied_changes.md`: run the same
config, same seed, same horizon, and compare the record JSON field by field (the check in
`analysis/code/run_conditions.py`). One differing number is enough to land here.

Code for the items that have an implementation lives in `../deferred_experiments/<name>/`, each with
its own README describing what it does and what it would cost to adopt.

---

## 1. Port the SAC update to JAX (sbx) — the only large lever

**Gain: ×1.66 measured on this exact task** (PointMaze + RND, 12-node same-node A/B,
`analysis/2026-06-25-run-profiling/analysis.md` §0 and §6.3); up to ~×2.5 if the RND predictor is
ported too, since with SAC fused the torch RND becomes the bottleneck (27% of the step).

**Why it breaks validity:** it is a different SAC implementation, not a faster spelling of the same
one — different initialization, different update math order, different RNG. Worse, the prior
campaign's full sweep found sbx **under-trains** relative to SB3 (run 4 reached 0.39–0.52× run 2's
reward at 500K steps), so it is not a drop-in reproduction: adopting it means re-running the
baselines it is compared against.

**Cost to adopt:** a separate trainer + a convergence-parity study against SB3 before any science
run. Prototype already on disk:
`analysis/2026-06-25-run-profiling/experiments/2026-06-25-21-40-jax-ab/`.

## 2. Fuse the RND bonus and predictor-update forwards

**Gain: ~1 of the ~5 ms/step the RND slice costs (~3-5% end to end).** Today every gradient step
runs the predictor and target **twice** over the same 256-row batch: once in `compute()` (no-grad,
for the reward) and once in `update()` (with grad, for the loss).

**Why it breaks validity:** the two forwards are *not* redundant. `RND.update()` calls
`obs_rms.update(x)` **before** normalizing its input (`methods/rnd.py:631`), so the update's forward
sees a different normalization than the bonus's forward one line earlier. Fusing them forces a
choice — reuse the bonus-time normalization for the loss, or move the statistics update ahead of the
bonus — and either one changes the predictor that gets trained, hence every downstream number.

**Cost to adopt:** small code change, but it must be introduced at a sweep boundary.
Implementation: `../deferred_experiments/fused_rnd/`.

## 3. Batch the SAC updates (`sac_train_freq=N`, `sac_gradient_steps=N`)

**Gain: +1.0% measured** (2026-06-25). Both knobs already exist in `Config`; keeping them equal
holds the total number of gradient steps constant.

**Why it breaks validity:** updates move from "one per env step" to "N every N steps", so the
policy that collects the data is stale by up to N-1 steps. Same expected work, different trajectory.

**Cost to adopt:** none — it is two flags. It is here only because +1% is not worth a fresh baseline.

## 4. `torch.compile` the SAC update

**Gain: ~1% estimated** (not built; the nets are 256×256 MLPs, so there is little for the compiler
to fuse, and CPU inductor pays a warm-up per process — 900 short-lived processes make that worse).

**Why it breaks validity:** compiled kernels are free to reassociate float operations.

## 5. Drop the per-env-step intrinsic logging wrapper

**Gain: one batch-1 predictor+target forward per env step** (~0.1-0.2 ms of a ~30-40 ms step).
`ComputeIntrinsicRewardWrapper` exists to log `train/intrinsic_reward`; the reward that trains the
agent is formed in the replay buffer, not here.

**Why it breaks validity:** `train_episode_history[*]["train/intrinsic_reward"]` is a recorded
metric — dropping it removes a column the analysis plots.

## 6. GPU (`--device=cuda`)

**Gain: ~+20% over the best CPU configuration** for a single run (2026-06-25 GPU bench).

**Why it is not adopted:** not a validity break per se (same code path) but a throughput loss for a
*sweep* — a GPU node runs far fewer concurrent runs than 30-60 CPU workers, and the sweep's
bottleneck is total runs per node-hour, not latency per run. Also float accumulation order differs
from CPU, so mixing devices inside one sweep would break comparability.

## 7. Vectorized environments (`n_envs > 1`) / lean CleanRL-style trainer

**Gain: +6.8% (RND) measured for the lean trainer**, more for vectorized rollouts on heavier envs.

**Why it breaks validity:** both change the order in which transitions enter the buffer, which is
the experiment. Prototype on disk:
`analysis/2026-06-25-run-profiling/experiments/2026-06-25-19-55-fast-trainer/fast_sac_rnd.py`.

## 8. Reduced precision (bf16/fp16) for the MLPs

**Gain:** small on this CPU class (no native bf16 matmul on the older nodes; the 256×256 GEMMs are
already MKL-bound).

**Why it breaks validity:** obviously — different arithmetic.

---

## Rejected outright (measured, not worth building)

| idea | why not |
|---|---|
| C/C++ environment | the env is 0.69 ms of a ~30-40 ms step: **≤1.4% ceiling**, and MuJoCo is already C |
| more threads per worker | 2 threads beat 1 by 22.8% per run, but two 1-thread workers on a core's sibling threads beat one 2-thread worker by ~30% per core-hour — the sweep already uses the better packing |
| `MALLOC_ARENA_MAX=1` | **-0.1%** measured (2026-08-13, alg23, 2 repeats). Collapsing glibc's arenas does not reproduce tcmalloc's +6% — the win comes from the allocator's algorithm, not the arena count |
| `inference_mode` for the RND bonus | **-0.8%** measured. Inference-mode tensors may not be consumed by autograd-tracked code, so the bonus must be `.clone()`d before it becomes a reward — and the clone costs more than the version-counter bookkeeping it removes |
| `torch.set_num_interop_threads(1)` | not rejected — **already applied upstream** by the sweep owner in `train4m.py` at 2026-08-13 05:07, worth ~+0.5%. torch raises if it is set twice, so this side keeps it as a tolerated no-op |
