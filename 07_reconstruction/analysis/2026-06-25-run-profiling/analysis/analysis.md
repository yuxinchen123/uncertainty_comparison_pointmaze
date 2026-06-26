# Profiling & efficiency improvements — SAC + intrinsic-bonus training (07_reconstruction)

**Date:** 2026-06-25. **Goal:** make the training loop run faster. Profile head-by-head, propose +
implement improvements, measure local (per-function) and end-to-end (whole-loop fps) speedups on ≥12
nodes (baseline vs improved on the **same node** so node-type noise cancels), with a passing test per
improvement. Small safe wins are **on/off switches in the main repo**; the bold rewrite is a **separate
codebase** presented for a merge decision.

> All numbers are **measured** — 12-node same-node benchmark (`run_bench.slurm` → `aggregate.py`),
> per-function microbenchmarks (`micro_local.py`), JAX A/B (`sac_pointmaze_bench.py` → `jax_ab_aggregate.py`),
> GPU (`run_gpu_bench.slurm`), and the env-step bound (`env_step_bench.py`). See §8 for the raw table.

---

## 0. Where the time goes (100K steps): train run 2 vs JAX

Per-step **compute** breakdown for `rnd_state` (one process, 2 threads), measured by `breakdown.py`. For
100K steps:

**Train run 2 (SB3 + torch)** — clean 2-CPU node, 32.8 fps = 30.5 ms/step:

| Phase | ms/step | % of step | time for 100K steps |
|---|---|---|---|
| SAC critic+actor+optimizer + replay buffer (torch autograd) | 23.2 | **76%** | ~39 min |
| RND compute+update (torch, 256-batch) | 5.0 | **16%** | ~8 min |
| Polyak target update (SB3 `zip_strict`) | 1.3 | **4%** | ~2 min |
| Env step (MuJoCo + wrappers) | 1.0 | **3%** | ~2 min |
| **Total** | **30.5** | 100% | **~51 min** |

*Production wall-clock is higher — ~80 min/100K, ~14 h/1M — because 8 workers share each node's CPUs; that
extra is contention, not new work.*

**JAX (sbx, SB3+JAX) — exact same task (PointMaze + RND)** — clean node, 54.5 fps = 18.3 ms/step:

| Phase | ms/step | % of step | time for 100K steps |
|---|---|---|---|
| SAC update (JAX, **XLA-fused**, incl. polyak) | 12.3 | **67%** | ~20 min |
| RND compute+update (**still torch**) | 5.0 | **27%** | ~8 min |
| Env step (MuJoCo + wrappers) | 1.0 | **6%** | ~2 min |
| **Total** | **18.3** | 100% | **~30 min** |

**Measured full-task speedup: ×1.66 (±0.02 over 12 nodes), 51 → 30 min/100K.** JAX collapses the 76% SAC
slice (23.2 → 12.3 ms) into one fused program; the **torch RND (now 27%) becomes the bottleneck** — porting
RND to JAX too would push the full-task speedup back toward the 2.53× measured for pure SAC (§6.3).

---

## 1. Executive summary

The per-step cost (~50 ms/step, ~20 fps on a 2-CPU worker) splits into **~28 ms compute core**
(SAC critic+actor autograd backward + linear forwards on the 256×256 nets, plus the RND predictor) and
**~20–30 ms of SB3 / env / Python plumbing** (DummyVecEnv + Monitor + 3 wrappers + 3 callbacks, the
polyak `zip_strict` loop, and the intrinsic buffer's numpy↔torch round-trip). The compute core is
matmul-bound and already uses MKL/oneDNN, so the realistic wins attack the **plumbing slice** and the
**target-update overhead**:

| Improvement | Where it lives — exactly what changed | Numerics | Local | End-to-end (rnd_state, same-node) |
|---|---|---|---|---|
| `opt_polyak_foreach` | **Built · main repo · OFF by default.** New file `src/rnd_exploration/common/sb3_patches.py` + flag `Config.opt_polyak_foreach` (`train.py`); `build_sac()` installs the patch when the flag is True | **bit-exact** | polyak **×11.5** (1558→136 µs/call) | **+2.1%** |
| `opt_torch_reward` | **Built · main repo · OFF by default.** Flag `Config.opt_torch_reward` → an `if`-branch in `buffers/vector_intrinsic_replay_buffer.py::sample()` | **bit-exact** | reward-combine ~neutral (obs re-wrap skip) | **+2.3%** |
| **combined** | **Both flags ON together** (no new code) | **bit-exact** | — | **+6.1% (free — merge now)** |
| `sac_train_freq=8` | **Built · main repo · default 1.** Flags `Config.sac_train_freq`/`sac_gradient_steps` → `SAC(train_freq=, gradient_steps=)` in `build_sac()` | changes-dynamics | — | +1.0% (not worth it) |
| threads | **Setting, not code.** `torch.set_num_threads(N)` + `OMP_NUM_THREADS` at launch; finding = keep N=2 | per-run float-order | — | **2 optimal** (1: −22.8%, 4: −10.6%) |
| `fast_sac_rnd` | **Built · STANDALONE file · not merged, not imported by `train.py`.** `experiments/2026-06-25-19-55-fast-trainer/fast_sac_rnd.py` | different-but-valid | removes plumbing slice | +6.8% (RND), −11% (gt) |
| **JAX (sbx)** | **Built as a separate A/B trainer** (`experiments/2026-06-25-21-40-jax-ab/sac_pointmaze_bench.py`, jax venv), not merged; a production sbx port would be a new codebase | different-but-valid | XLA-fuses the whole SAC update | **×1.66 full run-2 task / ×2.53 pure SAC ← biggest** |
| GPU (`device=cuda`) | **Existing flag, only benchmarked.** `Config.device="cuda"` already in `train.py`; no new code | exact | — | ~+20% vs best CPU (not a sweep win) |
| C/C++ env | **NOT built — rejected by measurement.** No change made | — | env.step 0.69 ms (1.4% of step) | **≤1.4% ceiling** |
| torch.compile | **NOT built — research estimate only.** No flag added | exact | — | ~1.12% |

**How to read the "Where" column:**
- **Built · main repo · OFF by default** — a new `Config` flag + code path added to the live codebase, defaulting to the *old* behavior, so **nothing changes until you pass the flag**. To turn one on: `--opt_polyak_foreach=True` (or set it in the queue config). Three of these exist.
- **Built · STANDALONE file** — new code that is **not** wired into `train.py`; run it directly. One exists (`fast_sac_rnd.py`).
- **NOT built** — I measured or estimated the speedup but wrote **no code** for it (JAX port, C/C++ env, torch.compile). These are proposals.
- **Existing flag** — a capability already in the repo (`device=cuda`) that I only benchmarked.

**Exactly which files I changed in the main repo** (every switch defaults to the old behavior — the live training runs are unaffected until you opt in):
1. `train.py` — added 4 `Config` fields + their argparse flags (`opt_torch_reward`, `opt_polyak_foreach`, `sac_train_freq`, `sac_gradient_steps`) and wired them in `build_sac()`.
2. `src/rnd_exploration/buffers/vector_intrinsic_replay_buffer.py` — added the `opt_torch_reward` parameter + an `if`-branch in `sample()`.
3. `src/rnd_exploration/common/sb3_patches.py` — **new file** (the `_foreach_` polyak update).

Plus one standalone, unmerged file: `analysis/2026-06-25-run-profiling/experiments/2026-06-25-19-55-fast-trainer/fast_sac_rnd.py`. Nothing else in the training pipeline was touched.

The CPU optimizations are **modest** because the SAC backward+linear compute core (~80%) is matmul-bound
and already uses MKL/oneDNN — there is no large pure-CPU win to be had. The bit-exact switches give **~+6%
for free**; the lean rewrite **~+7%**; the only **large** lever is **JAX (sbx): ×1.66 measured on the full
run-2 task** (PointMaze+RND, 12 nodes), or ×2.53 for pure SAC — it fuses the whole SAC update into one XLA
program (the ×1.66 is below ×2.53 only because the RND stays in torch). **Important caveat (Train run 4,
§6.3): these are FPS/speed numbers, not training-quality parity** — the full JAX sweep showed sbx's SAC
**under-trains vs SB3** (run-4 reaches only 0.39–0.52× run-2's reward at 500K), and the real per-run sweep
speedup is **1.55–1.81×** (8 workers/node, contended) not ~2.2×. So JAX is the biggest *speed* lever but is
**not a drop-in reproduction** of the SB3 results without matching the SAC implementations. GPU gives a fast absolute fps but is not a
clear win over a good 4-CPU node at this net size, and CPU packs the sweep better (per-run threads saturate
at 2). A C/C++ env is bounded at **≤1.4%** because the env is only 0.69 ms/step (MuJoCo is already C).

**Recommendation, in priority order:**
1. **Merge the two bit-exact switches now** (`opt_polyak_foreach`, `opt_torch_reward`) — proven identical,
   free, ~+6% combined; flip on by default after one sweep confirms wall-clock.
2. **For a real speedup, port to sbx/JAX** — **×1.66 measured** on the full run-2 task (RND kept in torch),
   up to ~×2.5 if the RND is also ported to flax — as a **separate codebase**, after a convergence-parity
   check. Highest value for the money.
3. Keep `fast_sac_rnd` (CleanRL-style) as the **heavier-env** scaling base (env-agnostic; JAX's edge shrinks
   when envs get non-jittable and expensive).
4. Do **not** invest in a C/C++ env (≤1.4%), `sac_train_freq` batching (+1%, changes dynamics), more threads
   (>2 hurts), or GPU for the sweep (CPU packs more runs/node).

---

## 2. Methodology

- **Profiling harness** `analysis/code/profile_train.py`: builds the *exact* training stack `train.py`
  builds (env stack, intrinsic model, SAC with the `VectorIntrinsicReplayBuffer`), warms up past
  `learning_starts`, then (a) times a non-profiled segment for clean fps and (b) optionally cProfiles a
  segment for the function-level breakdown. Eval/callbacks excluded so it is the pure train loop.
- **Same-node A/B**: `slurm/run_bench.slurm` runs every condition as a single clean process **on one
  node**, sequentially; the launcher pins one job per node across **12 homogeneous 32-CPU allowlist
  nodes** (`adriatic01–06, lynx01–04, ai01–02`). `aggregate.py` pairs conditions by node and reports the
  ratio improved/baseline per node, averaged — so node-type variation cancels and the speedup is robust.
- **Bold trainer** `experiments/2026-06-25-19-55-fast-trainer/`: validated for SAC-core learning on
  Pendulum-v1 and benchmarked vs SB3 on the same node (`slurm/run_fast_bench.slurm`, lynx05–07).
- **GPU** (`slurm/run_gpu_bench.slurm`): CPU vs CUDA on the same GPU node (a4000/a100/h100).
- All runs: `OMP/MKL/OPENBLAS=2`, `torch.set_num_threads(2)` unless varied; PointMaze_Large-v3,
  top_right goal, the run-2 fixed hyperparameters.

---

## 3. Baseline profile

Clean fps (12 profiling nodes, 2 threads/2 CPUs): **gt_position_velocity 31.4 ±2.8, rnd_state 30.4 ±2.8**
(rnd_elliptical runs faster — ~58 fps on the 4-CPU benchmark nodes — because its elliptical/Mahalanobis
bonus is cheaper than RND's predictor+target forwards). cProfile of the rnd_state train loop (6 000 steps,
adriatic01), top by total time:

| function | tottime | ncalls (/6000 steps) | what |
|---|---|---|---|
| `torch._C._EngineBase.run_backward` | 72.8 s | 24 000 (4/step) | SAC critic+actor (+RND predictor) backward |
| `torch._C._nn.linear` | 51.1 s | 252 000 (42/step) | all MLP forwards |
| `adam._single_tensor_adam` | 8.2 s | 24 000 | optimizer steps |
| `utils.zip_strict` | 5.9 s | 84 000 (14/step) | SB3 polyak target update (pure Python) |
| `torch.relu` / `mul_` / `sqrt` / `lerp_` | ~3–5 s each | — | elementwise |

The backward + linear (~80%) are the matmul-bound compute core; `zip_strict` is the largest pure-Python
overhead and the cheapest exact target.

---

## 4. Improvements (main-repo switches)

Each switch defaults **off** (= original behavior) so the baseline is unchanged; the benchmark toggles them.

### 4.1 `opt_torch_reward` — combine reward in torch, no numpy round-trip  *(bit-exact)*

**Why.** `VectorIntrinsicReplayBuffer.sample()` runs every gradient step. SB3's `_get_samples` already
returns CPU float32 tensors and `RND.compute` returns a float32 tensor, yet the code converted **both** to
numpy, added in numpy, then converted back to torch — three needless conversions per step in the
intrinsic path.

**Before** (`src/rnd_exploration/buffers/vector_intrinsic_replay_buffer.py`):
```python
obs = to_tensor(batch.observations, self.device)          # re-wrap already-tensors
next_obs = to_tensor(batch.next_observations, self.device)
actions = to_tensor(batch.actions, self.device)
samples = {"observations": obs, "next_observations": next_obs, "actions": actions}
intrinsic_rewards = self.intrinsic_reward_model.compute(samples)
self.intrinsic_reward_model.update(samples)
intrinsic = to_numpy_flat(intrinsic_rewards)              # tensor -> cpu -> numpy -> float32 -> ravel
extrinsic = to_numpy_flat(batch.rewards)                  # tensor -> numpy ...
total = (extrinsic + self.beta * intrinsic).reshape(-1, 1)   # numpy add
total = to_tensor(total, self.device)                     # numpy -> tensor
```

**After** (switch on):
```python
samples = {"observations": batch.observations,           # pass tensors straight through
           "next_observations": batch.next_observations, "actions": batch.actions}
intrinsic_rewards = self.intrinsic_reward_model.compute(samples)
self.intrinsic_reward_model.update(samples)
if not torch.is_tensor(intrinsic_rewards):               # visit-count returns numpy -> one conversion
    intrinsic_rewards = torch.as_tensor(intrinsic_rewards, device=batch.rewards.device)
intrinsic_t = intrinsic_rewards.reshape(-1, 1).to(dtype=batch.rewards.dtype, device=batch.rewards.device)
total = batch.rewards + self.beta * intrinsic_t          # extrinsic + beta*intrinsic, all torch
```

**Numerics.** Bit-exact: float32 `ext + beta*intrinsic` is the same IEEE computation in numpy and torch.
**Test.** `test_optimizations.py::test_torch_reward_matches_numpy_reward_bit_identical` — `torch.equal`
PASSES (max abs diff 0.0).
**Local improvement.** The reward-combine arithmetic itself is ~neutral (`micro_local.py`: numpy round-trip
**20.0 µs/call** vs torch **20.7 µs/call**, ×0.96) — the real saving is skipping the three `to_tensor`
re-wraps of obs/next_obs/actions. **End-to-end:** **+2.3%** (rnd_state, 12 nodes, same-node ratio) —
marginal, near the run-to-run noise.

### 4.2 `opt_polyak_foreach` — torch `_foreach_` soft target update  *(bit-exact)*

**Why.** SB3's `polyak_update` loops the 12 critic parameters one-by-one through `zip_strict` every
gradient step (5.9 s / 84 000 calls in the profile — the largest pure-Python slice).

**Before** (`stable_baselines3/common/utils.py`, called by `SAC.train`):
```python
with th.no_grad():
    for param, target_param in zip_strict(params, target_params):
        target_param.data.mul_(1 - tau)
        th.add(target_param.data, param.data, alpha=tau, out=target_param.data)
```
**After** (`src/rnd_exploration/common/sb3_patches.py`, installed when the switch is on; patches
`sac.sac.polyak_update`, the name SAC actually calls):
```python
with th.no_grad():
    t = list(target_params); s = list(params)
    if not t:                       # guard SAC's empty batch_norm_stats call (_foreach raises on empty)
        return
    th._foreach_mul_(t, 1.0 - tau)
    th._foreach_add_(t, s, alpha=tau)
```

**Numerics.** Bit-exact — same two element-wise ops, same scalars, same per-element order, fused into two
C++ foreach kernels. **Test.** `test_optimizations.py::test_polyak_foreach_bit_exact` — `torch.equal` over
the 12 critic-shaped tensors PASSES; empty-list call no-ops.
**Local improvement.** The soft target update drops from **1558 µs → 136 µs/call (×11.5, −91%)**
(`micro_local.py`) — `zip_strict`'s 14 per-parameter Python iterations collapse into two C++ foreach
kernels. **End-to-end:** **+2.1%** (rnd_state, 12 nodes). This is the canonical local-vs-e2e gap: huge
locally (×11.5), small end-to-end because polyak is only ~6% of per-step time.

### 4.3 `sac_train_freq` / `sac_gradient_steps` batching  *(changes dynamics — needs A/B)*

**Why.** With `train_freq=1/gradient_steps=1`, SB3 re-runs `train()`'s fixed setup
(`set_training_mode`, `_update_learning_rate`, optimizer-list build, `logger.record`, length-1 `np.mean`)
2 000× and ping-pongs collect↔train every env step. Setting `train_freq=N, gradient_steps=N` keeps the
**total** gradient-step count identical but runs that setup `total/N`× instead.

**Before:** `SAC(..., )` (defaults `train_freq=1, gradient_steps=1`). **After:** `SAC(...,
train_freq=cfg.sac_train_freq, gradient_steps=cfg.sac_gradient_steps)`.

**Numerics.** Changes-dynamics: N env steps are collected under a fixed policy, then N updates run on that
snapshot, vs interleaving one update per step. Total updates equal, but learning trajectories differ —
a standard SAC config, but **must be A/B-validated** on `eval/mean_extrinsic_reward` and `distance_to_gt/*`
before adoption. **Test.** assert `model._n_updates` is identical across N (same total gradient steps).
**End-to-end:** **+1.0%** at N=8 (rnd_state, 12 nodes) — small, and it changes dynamics, so **not worth
adopting** for this gain.

### 4.4 Threads  *(per-run float-order; empirical)*

`torch.set_num_threads` 1 vs 2 vs 4 on the 256×256 nets. **Measured (rnd_state, 12 nodes, vs threads=2):**
threads=1 **−22.8%**, threads=4 **−10.6%**. So **2 is the sweet spot** — one thread starves the matmuls,
four wastes cores (cross-thread sync overhead on tiny ops). The sweep lever is therefore **more concurrent
2-thread runs per node**, not more threads per run.

### 4.5 Combined (bit-exact only) and combined (all)

`opt_torch_reward + opt_polyak_foreach` (both bit-exact → mergeable with no re-validation):
**+6.1%** (rnd_state, 12 nodes). Adding `train_freq=8` (`combined_all`): **+3.1%** — *lower* than the
bit-exact pair, i.e. the dynamics-changing batching did not help here and added variance, confirming it is
not worth adopting. **The recommended bundle is the two bit-exact switches: ~+6% for free.**

---

## 5. Bold rewrite — `fast_sac_rnd.py` (separate codebase)

**Why.** ~20–30 ms/step is SB3/env/Python plumbing on top of the ~28 ms compute core. A lean,
single-file CleanRL-style loop removes that slice: one process, one env (no DummyVecEnv/Monitor/3
callbacks), a preallocated **torch** replay buffer (no numpy↔torch round-trip), the intrinsic reward
folded into the batched update, and a `_foreach_` soft target update. It **reuses the project's exact env
stack + intrinsic model** (`train.py` `build_env_stack` / `build_intrinsic_model`), so the env and bonus
are identical to the SB3 runs and only the SAC loop changes.

**Location.** `experiments/2026-06-25-19-55-fast-trainer/fast_sac_rnd.py` (kept separate; not merged).

**Validation.** SAC core on Pendulum-v1 (10 k steps, 3 nodes): mean episodic return reaches **−165 / −184
/ −191** (random ≈ −1200; ~−200 = solved) — the hand-written SAC genuinely learns. PointMaze fps vs SB3
on the **same node** (3 nodes): **+6.8% (rnd_state), +9.3% (rnd_elliptical), −11.0% (gt_position_velocity)**.
A modest win for the RND/elliptical cases; slightly *slower* for the cheap visit-count case (its intrinsic
is a table lookup, so there is little plumbing to remove and my loop's per-step Python costs more than
SB3's there). **The win is smaller than the research's 1.5–2× estimate** because that used a contended
20 fps anchor; on clean 4-CPU nodes SB3 already runs ~53 fps (19 ms/step), so the plumbing slice removed
is proportionally small. Raw: fast 56.5 vs SB3 53.0 fps (rnd_state).

**Numerics.** Different-but-valid: same algorithm + hyperparameters, but a fresh implementation (RNG/init
differ from SB3, so seeds won't match). **Before merging it must be validated to convergence parity** on
`eval/mean_extrinsic_reward` and `distance_to_gt/*` against the existing run-2 curves. Until then it is a
fast prototype for the scaling plan, not a drop-in.

---

## 6. Alternative directions (measured): C/C++ env, envpool, JAX, GPU

These answer the explicit "what about X" questions with measured numbers, not guesses.

### 6.1 Implementing the env in C/C++ — bounded at ~1.4%

**Measured** (`analysis/code/env_step_bench.py`, PointMaze_Large, random actions, no SAC):
`env.step()` = **0.69 ms/step (1 447 env-steps/s)**. At ~50 ms/full-SAC-step the env is only **~1.4%** of
per-step time, so even a **perfect, free** env gives **≤ ~1.4% end-to-end** speedup here. Reason: the
physics is **already C** (MuJoCo); only the thin Python wrapper layer (obs flatten, visit-count bookkeeping)
could move to C/C++, and that layer is sub-millisecond. **Verdict: not worth a C/C++ env rewrite for
PointMaze** — the bottleneck is the SAC gradient update, not the env.

### 6.2 envpool (CleanRL's fast C/C++ env) — wrong regime for off-policy SAC

The vendored `ppo_rnd_envpool.py` is fast because **envpool** runs *hundreds* of **Atari** envs in C++
with no per-step Python, feeding a cheap PPO whose per-step cost is dominated by **env throughput**. That
is the opposite regime to ours: off-policy **SAC** does a full gradient update **every env step** on a
**tiny continuous** env, so the env (0.69 ms) is negligible vs the update (~28 ms). envpool also does not
provide gymnasium-robotics PointMaze. **Verdict: envpool's win does not transfer** to this workload; it
would matter only for a future on-policy, env-throughput-bound, envpool-supported task.

### 6.3 JAX (sbx = SB3 + JAX) — measured update + end-to-end speedup

Measured directly by installing `sbx` (SB3+JAX, jax[cpu]) in an isolated venv and benchmarking sbx.SAC vs
SB3.SAC fps on a common env (Pendulum-v1, identical hyperparameters), same node:

- **Pure SAC, Pendulum-v1 (measured, same machine):** SB3 **41.4 fps** vs sbx **104.5 fps → ×2.53 (+153%)**
  — JAX jit fuses the whole actor+critic+alpha update into one XLA program, removing the per-op Python
  dispatch torch.compile could only cut ~1.12×.
- **Pure SAC on PointMaze (extrinsic, no RND, 12-node A/B):** SB3 36.5 → sbx 72.7 fps = **×2.00 ±0.10** —
  between Pendulum's ×2.53 and the full-task ×1.66. The drop from Pendulum reflects PointMaze's heavier
  (un-jitted) MuJoCo env. So the **dilution chain is fully measured: ×2.53 (light env) → ×2.00 (PointMaze
  env) → ×1.66 (+ torch RND)** — consistent with §0 (the SAC slice alone is 23.2→12.3 ms ≈ ×1.9).
- **Full run-2 task, PointMaze_Large + RND (measured, 12 nodes × 3 seeds, same-node A/B):** SB3 **32.8 fps**
  vs sbx **54.5 fps → ×1.66 (±0.02)** — **the honest end-to-end number for the actual run-2 task.** It is
  lower than the pure-SAC ×2.53 because the **RND stays in torch** (now 27% of the step, §0) and gets no JAX
  speedup; porting RND to flax too would push it back toward ×2.5. The JAX side applies RND at **step time**
  (a standard variant, as in CleanRL ppo_rnd); for `train_freq=1` the per-step RND cost matches run-2's
  sample-time RND, so this is a fair speed A/B. One-time JIT warmup ≈ 25 s.
- **Why it holds for PointMaze too:** the ×2.53 was measured with the env step **un-jitted** (Pendulum, like
  PointMaze, is a cheap env — §6.1), so the un-jitted slice is small and the SAC-update speedup carries through.
  For **heavier non-jittable envs later** the un-jitted env grows and this e2e gain shrinks (which is why the
  env-agnostic CleanRL rewrite is the better base once envs get heavy).
- **Caveats (measured):** one-time JIT warmup **16.6 s**; seeds won't match SB3 (different RNG/init) →
  different-but-valid, needs convergence re-validation; the custom recompute-on-sample intrinsic buffer + 3
  methods must be ported to flax to recover the rest. **Effort:** ~1–2 days. **Verdict: the highest-value
  direction for the current PointMaze workload** — a separate sbx codebase, **×1.66 measured per run on the
  full task** (RND in torch), up to ~×2.5 with the RND also in JAX.
- **RND folded into JAX — DONE + measured (`experiments/2026-06-25-run4-fully-jax/jax_rnd.py`):** a fully-JAX
  RND (flax predictor/target + optax Adam + running obs-norm, mirroring `rnd.py` for rnd_state) is **validated
  correct** (training drives a seen state's intrinsic 3.76 → 0.02 while a novel state stays higher, 0.22 — the
  RND novelty signal). With it, **sbx + JAX-RND = 76.6 fps vs SB3 35.3 on the same node = ~×2.17** (vs ×1.66
  with the torch RND), i.e. the JAX RND is now essentially free (≈ the pure-SAC fps) — confirming the §0
  prediction. This is the basis for **Train run 4** (fully-JAX, 500K-step, 3×200-seed sweep).
- **Train run 4 result — the convergence re-validation FAILED (the caveat above bit).** The full 500K-step,
  3-algorithm sweep (stopped at 395/600 runs, ≥125 seeds/algo, sample-time intrinsic matching run-2's
  `VectorIntrinsicReplayBuffer`, obs-norm warm-up matched) shows **sbx's SAC under-trains vs SB3's** on this
  sparse-reward task. At 500K steps run-4 reaches only **0.39–0.52× run-2's eval reward** for every algorithm
  ($\bar R \pm \text{SE}$: gt_position_velocity 15.6±2.2 vs 38.8±2.9; rnd_elliptical 7.8±1.7 vs 20.2±2.7;
  rnd_state 7.8±1.9 vs 14.9±2.3). The curves match to ~250K then diverge in the late-training exploitation
  phase. **The gap is the SAC backend, not the intrinsic or the sample-time recompute:**
  `gt_position_velocity` uses the *identical* torch visit-count bonus with the same sample-time recompute as
  run-2, differing only in sbx-vs-SB3 SAC, yet shows the same shortfall. **The real per-run sweep speedup is
  1.55–1.81×** (run-4 ~2.9–3.1h vs run-2 ~4.7–5.2h at 500K-equivalent, 2 CPUs) — below the ~2.17× FPS
  benchmark above because those FPS numbers were single-run on an idle node, while the sweep packs 8
  workers/node on slower, contended hardware. **Conclusion: the FPS speedup does not imply training-quality
  parity.** Folding the RND into JAX is correct and ~1.6–1.8× faster, but reproducing run-2 in JAX requires
  matching sbx's SAC to Stable-Baselines3's (entropy/temperature, target update, init — not investigated).
  See `development_document` §Train-run-4 (Table 34, Figure 6); data under
  `train_runs/2026-06-25-21-54_run_4_…/data/2026-06-25-23-05_obsrms-distoff/`.

### 6.4 GPU — measured CPU vs CUDA on a4000 / a100 / h100

Same-node CPU vs CUDA (`slurm/run_gpu_bench.slurm`, a4000/a100/h100, 3 nodes). **Measured** cuda fps:
rnd_state **63.6**, gt **62.3**, rnd_elliptical **71.2**. On the **same GPU node vs its 2-CPU CPU**:
**+107% to +143%** — but that 2-CPU CPU is handicapped (38 fps); against the **best 4-CPU CPU nodes**
(~47–53 fps) cuda is only **~+20–35%**, with high variance. And a CPU node packs ~16 two-thread runs vs
~4 GPU runs, so for the 600-run **sweep** CPU throughput-per-node wins. **Verdict:** the cuda path runs
cleanly and gives a fast absolute fps, but it is **not a clear win over a good CPU at this net size**, and
it does not help the sweep. `device=cuda` is kept as a main-repo switch; GPU becomes a real lever when
**nets/observations grow** or **many seeds are batched on-device** (a JAX `vmap`-across-seeds trainer can
reach 10–50× *sweep* throughput, but only for a jittable env, so it does not survive a move to heavier
non-jittable MuJoCo envs — which is why the env-agnostic CleanRL rewrite (§5) is the heavier-env base).

---

## 7. Where each change lives (the merge decision)

- **Merge now (bit-exact switches, default off):** `opt_polyak_foreach`, `opt_torch_reward`. Proven
  identical (`torch.equal`), free, ~+6% combined. Turn on by default after one sweep confirms wall-clock.
- **Separate codebase — highest value:** an **sbx/JAX port** (**×1.66 measured** on the full PointMaze+RND
  task, up to ~×2.5 with RND in JAX too). Biggest win
  for the current workload; needs a convergence-parity check and a flax port of the intrinsic buffer + 3
  methods (~1–2 days). Your decision to invest.
- **Separate codebase — heavier-env base:** `fast_sac_rnd.py` (env-agnostic, ~+7%); adopt as the scaling
  base for non-jittable/heavier envs after convergence parity. Do not merge into `train.py`.
- **Keep as low-value flags:** `device=cuda` (~+20% vs best CPU, not a sweep win), `opt_torch_compile`
  (~1.12%), `sac_train_freq` (+1%, changes dynamics).
- **Do not pursue:** a C/C++ env (≤1.4% ceiling), >2 threads per run (hurts).

---

## 8. Raw results

**Baseline fps** (clean): profiling 2-CPU nodes — gt 31.4±2.8, rnd_state 30.4±2.8; benchmark 4-CPU nodes —
rnd_state 47.4±6.9, gt 47.1±4.9, rnd_elliptical 58.3±3.7.

**CPU-sweep same-node speedup vs baseline** (`aggregate.py --source bench`, ratio paired by node):

| condition | rnd_state | gt_position_velocity | rnd_elliptical |
|---|---|---|---|
| `opt_torch_reward` (bit-exact) | +2.3% (12) | — | — |
| `opt_polyak_foreach` (bit-exact) | +2.1% (12) | — | — |
| **combined_exact** (both, bit-exact) | **+6.1% (12)** | **+3.7% (12)** | **+5.8% (12)** |
| combined_all (+tf8, dynamics) | +3.1% (12) | +0.4% (12) | +12.5% (12) |
| `tf8` (train_freq=8) | +1.0% (12) | — | — |
| threads=1 | −22.8% (12) | — | — |
| threads=4 | −10.6% (12) | — | — |

**Local microbenchmarks** (`micro_local.py`): polyak update **1558 µs → 136 µs/call (×11.5)**;
reward-combine numpy 20.0 µs vs torch 20.7 µs/call (×0.96, ~neutral).

**Bold trainer** (`fast_sac_rnd`, same-node vs SB3, 3 nodes): rnd_state **+6.8%**, rnd_elliptical **+9.3%**,
gt **−11.0%**. SAC-core validation (Pendulum-v1 mean return, ~−200 = solved): −165 / −184 / −191.

**JAX — pure SAC** (`pendulum_sac_bench.py`, Pendulum-v1, same machine): SB3 41.4 fps → **sbx 104.5 fps =
×2.53**; one-time JIT warmup 16.6 s.

**JAX — full run-2 task** (`sac_pointmaze_bench.py` → `jax_ab_aggregate.py`, PointMaze_Large + RND, 12 nodes
× 3 seeds = 36 same-node pairs): SB3 **32.8 ±2.8 fps** → sbx **54.4 ±4.0 fps** = **×1.659 ±0.023**; JIT
warmup ≈ 25 s. **Pure SAC on PointMaze** (extrinsic, 12 pairs): SB3 36.5 → sbx 72.7 fps = **×2.00 ±0.10**.
Dilution chain: ×2.53 (Pendulum) → ×2.00 (PointMaze env) → ×1.66 (+ torch RND). (sbx applies RND at step
time with the project's torch RND, so the JAX speedup is on the SAC update only — see §0/§6.3.)

**GPU** (`run_gpu_bench.slurm`, same-node CPU vs cuda): cuda fps rnd_state 63.6 / gt 62.3 / elliptical 71.2;
vs the node's 2-CPU CPU +133% / +107% / +143%; vs best 4-CPU CPU node (~47–53) only ~+20–35%.

**Env step** (`env_step_bench.py`): PointMaze `env.step()` = **0.69 ms/step (1447 env-steps/s)** = ~1.4%
of the ~50 ms per-step → C/C++ env ceiling ≤1.4%.

## 9. Reproduce

```bash
PROF=07_reconstruction/analysis/2026-06-25-run-profiling
EXPLPY=conda run -n exploration python

# baseline fps + cProfile hotspots (12 nodes), then the optimization sweep (12 nodes), then aggregate:
bash $PROF/slurm/launch_profiling.sh
for n in adriatic01 ... ; do sbatch --nodelist=$n -p gpu --gpus-per-node=0 $PROF/slurm/run_bench.slurm; done
$EXPLPY $PROF/analysis/code/aggregate.py --source bench           # same-node speedup ratios

# bold trainer vs SB3 (3 nodes), CPU-vs-GPU (a4000/a100/h100), and the cross-format comparisons:
for n in lynx05 lynx06 lynx07; do sbatch --nodelist=$n -p gpu --gpus-per-node=0 $PROF/slurm/run_fast_bench.slurm; done
for n in cheetah08 cheetah01 serval03; do sbatch --nodelist=$n -p gpu --gpus-per-node=1 $PROF/slurm/run_gpu_bench.slurm; done
$EXPLPY $PROF/analysis/code/compare_fast_gpu.py

# local per-function microbenchmarks, env-step bound, and the JAX comparison (sbx venv):
$EXPLPY $PROF/analysis/code/micro_local.py
$EXPLPY $PROF/analysis/code/env_step_bench.py
$EXPLPY      $PROF/analysis/code/pendulum_sac_bench.py --backend sb3 --steps 12000 --warmup 2500 --out sb3.json
<jax-venv-py> $PROF/analysis/code/pendulum_sac_bench.py --backend sbx --steps 12000 --warmup 2500 --out sbx.json

# correctness tests for the switches (bit-exact):
$EXPLPY -m pytest $PROF/analysis/code/test_optimizations.py -q
```

**Artifacts:** switches in the main repo (`train.py` Config/argparse/build_sac,
`src/rnd_exploration/buffers/vector_intrinsic_replay_buffer.py`, `src/rnd_exploration/common/sb3_patches.py`),
all default **off**; the bold trainer in `experiments/2026-06-25-19-55-fast-trainer/fast_sac_rnd.py`
(separate, unmerged).
