# Maximising throughput for CleanRL PPO+RND on Atari (envpool), PyTorch 2.10 + CUDA 12.8

Everything below is measured on this cluster unless marked as an estimate. Benchmark code, raw JSON, and Slurm logs: `/p/rlprojects/RND/08_cleanrl_ppo_rnd/bench_scratch/` (`bench_ppo_rnd.py`, `bench_isolate.py`, `bench_det_uint8.py`, `bench_envpool.py`, `results/`, `logs/`, `submitted_jobids.txt`).

---

## 0. BLOCKING FINDING — Pascal GPUs cannot run this stack at all

`torch 2.10.0+cu128` ships SASS for **sm_70, sm_75, sm_80, sm_86, sm_90, sm_100, sm_120 and no PTX fallback**. Read directly from the installed build:

```
torch._C._cuda_getArchFlags() -> 'sm_70 sm_75 sm_80 sm_86 sm_90 sm_100 sm_120'
```

Confirmed empirically on a real GTX 1080 (Slurm job 6533800, node `ai05`):

```
UserWarning: Found GPU0 NVIDIA GeForce GTX 1080 which is of cuda capability 6.1.
Minimum and Maximum cuda capability supported by this version of PyTorch is (7.0) - (12.0)
```

Per the CUDA compatibility guides, a cubin runs only on **the same major compute-capability version with equal-or-higher minor**, and without embedded PTX there is no JIT fallback. So:

| GPU | Capability | Runs on torch 2.10+cu128? |
|---|---|---|
| GTX 1080, 1080 Ti, Titan X/Xp, Tesla P100 | sm_60 / sm_61 | **No** — "no kernel image is available for execution on the device" |
| RTX 2080 Ti, Quadro RTX 4000/6000 | sm_75 | Yes (native cubin) |
| A100 | sm_80 | Yes |
| A40, A16, RTX A4000/A4500 | sm_86 | Yes |
| RTX 4000 Ada | sm_89 | Yes (runs the sm_86 cubin — same major version, higher minor) |
| H100 NVL | sm_90 | Yes |
| RTX 5080 | sm_120 | Yes |

**Action:** either drop Pascal from the plan, or run Pascal seeds on a separate `torch<=2.4.1+cu121` environment. Do not attempt to mix — a Pascal node will fail at the first kernel launch, not at import.

---

## 1. The arithmetic you asked for

CleanRL's docs report `ppo_rnd_envpool.py` at **2000M steps** with "extreme long run time (~250 hours)", 1 seed, scoring 7100 on MontezumaRevenge (paper: 8152, 3 seeds).

```
steps/s          = 2e9 / (250 h × 3600 s/h) = 2e9 / 900,000 s   = 2,222.2 agent-steps/s
iterations       = 2e9 / (128 envs × 128 steps) = 2e9 / 16,384  = 122,070 iterations
wall/iteration   = 900,000 s / 122,070                          = 7.373 s
policy steps/s   = 2,222.2 / 128 envs                           = 17.36 rollout steps/s
Atari frames/s   = 2,222.2 × 4 (frameskip)                      = 8,889 frames/s
```

30 seeds at that rate:

```
30 × 250 h = 7,500 GPU-hours = 312.5 GPU-days = 0.86 GPU-years
on 8 GPUs concurrently: 7,500 / 8 = 937.5 h = 39.1 days wall clock
```

| Speedup | h / seed | GPU-hours (30 seeds) | Days on 8 GPUs |
|---|---|---|---|
| 1.0x (published) | 250.0 | 7,500 | 39.1 |
| 1.35x | 185.2 | 5,556 | 28.9 |
| 1.6x | 156.3 | 4,688 | 24.4 |
| 2.0x | 125.0 | 3,750 | 19.5 |
| 3.0x | 83.3 | 2,500 | 13.0 |

For calibration: 8,889 frames/s is very low for envpool — the published number is **not** a GPU-bound rate. My measurement of envpool alone on 4 CPU cores is 2,736 agent-steps/s, i.e. the published 2,222 steps/s is roughly what a 4-core allocation caps you at. **The published run was almost certainly CPU-starved, not GPU-limited.** That reframes the whole ranking below.

---

## 2. Where the time actually goes (measured, RTX A4500 / sm_86, script's real default backend flags)

Per-rollout-step components at batch 128, and per-iteration whole-batch work:

| Phase | as written | fixed | ratio |
|---|---|---|---|
| policy: `get_value` + `get_action_and_value` (two full CNN trunk passes) | 2.042 ms | 0.951 ms | 2.15x |
| RND target+predictor forward, **grad mode on** | 1.359 ms | 0.592 ms | 2.30x |
| `torch.Tensor(np_obs).to(cuda)` (uint8→f32 on CPU, 14.5 MB H2D) | 1.056 ms | 0.190 ms | 5.56x |
| RND input normalisation (float64 stats re-uploaded every step) | 0.164 ms | 0.042 ms | 3.90x |
| `action.cpu().numpy()` | 0.012 ms | — | — |
| **rollout step total (×128 per iteration)** | **4.633 ms → 593 ms** | **1.775 ms → 227 ms** | **2.61x** |
| minibatch update, 4096 (×16 per iteration) | 57.9 ms → 926 ms | 57.9 ms | 1.00x |
| `obs_rms.update(...cpu().numpy())` — 462 MB D2H + numpy moments over 115.6M elems | 388.7 ms | 1.6 ms | **240x** |
| whole-batch RND normalisation in float64 | 21.3 ms | 5.0 ms | 4.27x |
| GAE Python loop (×2, ext + int) | 20.8 ms | 20.8 ms | 1.00x |
| **torch-side total per iteration** | **1,950 ms** | **1,181 ms** | **1.65x** |
| implied GPU-side steps/s ceiling | 8,402 | 13,873 | |

envpool alone (MontezumaRevenge-v5, same wrapper settings, 128 envs, sync):

| CPU cores in cgroup | agent-steps/s | frames/s | async (256 envs / batch 128) |
|---|---|---|---|
| 4 | 2,797 | 11,190 | — |
| 8 | 4,780 | 19,119 | 5,575 |
| 16 (`affogato02`) | 8,069 | 32,276 | — |
| 16 (`jaguar03`) | 12,477 | 49,908 | — |
| 32 (`cortado03`) | 12,568 | 50,272 | **23,144** |

Because envpool stepping and the GPU work are strictly serialised in this script, end-to-end steps/s = `16384 / (16384/env_sps + t_gpu)`:

| Setup | as written | same-numerics fixes | + fp16 update |
|---|---|---|---|
| 8 cores + A4500 | 3,047 | 3,556 (1.17x) | 3,910 (1.28x) |
| 16 cores (jaguar03) + A4500 | 5,010 | 6,551 (1.31x) | 7,984 (1.59x) |

**The single most important knob is CPU cores.** At 4 cores nothing you do to the PyTorch code can exceed 2,797 steps/s.

---

## 3. Tier A — same numerics, faster (safe for a faithful replication)

Ranked by measured value. Line numbers refer to `ppo_rnd_envpool.py`.

### A1. Give the job enough CPU cores, and set `num_threads` explicitly
**(a)** envpool is the throughput floor. **(b)** Slurm `--cpus-per-task=32` (not 4), plus:
```python
envs = envpool.make(..., num_threads=int(os.environ.get("SLURM_CPUS_PER_TASK", 32)))
```
**(c)** 4→32 cores: 2,797 → 12,568 steps/s (**4.5x**). Explicit `num_threads` vs the default is worth 2–13%: at 32 cores, `num_threads=128` gave 11,081 vs 12,568 at `num_threads=32`. Independent of GPU generation. **(d)** None — env stepping is unchanged; `seed` still controls the per-env seeds. **(e)** `bench_envpool.py`, or wall-clock the rollout loop with the policy replaced by random actions.

> Why the default is wrong on this cluster: envpool computes `num_threads = min(batch_size, processor_count)` where `processor_count = std::thread::hardware_concurrency()` (`envpool/core/async_envpool.h`). That is **not cgroup-aware** — on jaguar03 it reads 224 even inside a 16-CPU allocation (verified: job 6533817 reported `cgroup_cpus: 16, nproc_all: 224`), so the default becomes 128 threads. Leave `thread_affinity_offset` at its default `-1`: the binding loop uses absolute core ids `(offset + tid) % processor_count`, which is also not cgroup-aware and would bind outside your cpuset.

### A2. Move `obs_rms.update` to the GPU — **388.7 ms → 1.6 ms per iteration**
**(a)** L444 copies 462 MB device→host every iteration and runs numpy `mean`/`var` over 115.6M elements single-threaded. **(b)**
```python
# BEFORE (L444)
obs_rms.update(b_obs[:, 3, :, :].reshape(-1, 1, 84, 84).cpu().numpy())
# AFTER: keep the same parallel-variance update, but compute the batch moments on device
v = b_obs[:, 3, :, :].reshape(-1, 1, 84, 84)
obs_rms.update_from_moments(v.mean(0).cpu().numpy().astype(np.float64),
                            v.var(0, unbiased=False).cpu().numpy().astype(np.float64),
                            v.shape[0])
```
**(c)** 387 ms/iteration = **5.2% of the published 7.373 s wall**, and ~21% of the torch-side time. All generations. **(d)** Same estimator (`np.var` is population variance, matching `unbiased=False`); only the **summation order** differs, so results are not bit-identical — relative difference ~1e-7. Note the mean is in fact exact either way (values are integers ≤255, batch sum ≤ 4.18e6 < 2^24). If you need bit-identity, keep it on CPU; this is the one place where I would accept the 1e-7 change. **(e)** Time L444 alone with `torch.cuda.synchronize()` either side.

### A3. Merge the two policy trunk passes in the rollout — 2.042 → 0.951 ms/step
**(a)** L351 and L356 run the identical Nature-CNN trunk on the identical input twice. **(b)** Add one method and call it once:
```python
def get_action_and_value_fused(self, x):
    hidden = self.network(x / 255.0)
    features = self.extra_layer(hidden)
    value_ext, value_int = self.critic_ext(features + hidden), self.critic_int(features + hidden)
    probs = Categorical(logits=self.actor(hidden))
    action = probs.sample()
    return action, probs.log_prob(action), value_ext, value_int
```
```python
# BEFORE (L350-356): two trunk passes
with torch.no_grad():
    value_ext, value_int = agent.get_value(obs[step])
    ext_values[step], int_values[step] = value_ext.flatten(), value_int.flatten()
    action, logprob, _, _, _ = agent.get_action_and_value(obs[step])
# AFTER: one trunk pass
with torch.no_grad():
    action, logprob, value_ext, value_int = agent.get_action_and_value_fused(obs[step])
    ext_values[step], int_values[step] = value_ext.flatten(), value_int.flatten()
```
**(c)** 1.09 ms × 128 = **140 ms/iteration**. All generations. **(d)** **Bit-identical — verified on device**: `{'value_ext': True, 'value_int': True, 'action': True, 'logprob': True}`. The RNG stream is unchanged because `get_value` consumes no randomness, and the discarded `entropy` is not used in the rollout. **(e)** The equality check above; it is in `bench_det_uint8.py`.

### A4. Wrap the rollout RND forward in `no_grad` — 1.359 → 0.592 ms/step
**(a)** L371–373 sit **outside** the `with torch.no_grad()` block. `rnd_model.predictor` has trainable parameters, so a full autograd graph is built for a batch-128 forward, 128 times per iteration, then thrown away by `.data`. **(b)**
```python
# BEFORE (L371-373)
target_next_feature = rnd_model.target(rnd_next_obs)
predict_next_feature = rnd_model.predictor(rnd_next_obs)
curiosity_rewards[step] = ((target_next_feature - predict_next_feature).pow(2).sum(1) / 2).data
# AFTER
with torch.no_grad():
    target_next_feature = rnd_model.target(rnd_next_obs)
    predict_next_feature = rnd_model.predictor(rnd_next_obs)
    curiosity_rewards[step] = (target_next_feature - predict_next_feature).pow(2).sum(1) / 2
```
**(c)** 0.767 ms × 128 = **98 ms/iteration** (2.30x on that component), plus it removes 128 allocate/free cycles of activation memory per iteration. All generations. **(d)** **None.** The value was already `.data`-detached; the target tower is already `requires_grad=False`. Bit-identical. **(e)** `timeit` the two forms; also watch `torch.cuda.max_memory_allocated()` drop.

### A5. Store observations as uint8 — 1.85 GB → 0.46 GB, and 5.6x cheaper H2D
**(a)** `obs` at L307 is `float32` of shape `(128, 128, 4, 84, 84)`:
```
462,422,016 elements × 4 B = 1,849,688,064 B = 1.850 GB (1.723 GiB)
as uint8:                      462,422,016 B = 0.462 GB (0.431 GiB)   saving 1.387 GB
```
Worse, `torch.Tensor(next_obs)` (L364) is `torch.FloatTensor(...)` — it converts uint8→float32 **on the CPU** and then copies 14.5 MB per step (1.85 GB per iteration) instead of 3.6 MB (0.46 GB). **(b)**
```python
obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape,
                  dtype=torch.uint8, device=device)
...
# BEFORE (L364)
next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(done).to(device)
# AFTER
next_obs = torch.from_numpy(next_obs).to(device, non_blocking=True)          # uint8
next_done = torch.as_tensor(done, dtype=torch.float32, device=device)
```
The networks already divide by 255.0, so nothing else changes. **(c)** H2D 1.056 → 0.190 ms/step = **111 ms/iteration**; the 16 per-iteration minibatch gathers `b_obs[mb_inds]` also get cheaper (A4500: 10.35 → 8.67 ms, 16%; Quadro RTX 6000: 12.06 → 10.32 ms), and 7.4 GB/iteration of gather traffic becomes 1.85 GB. All generations; the memory saving matters most on 11 GB cards. **(d)** **The network path is bit-identical — verified: `uint8/255.0 == float32/255.0` exactly `True`** (every uint8 value is exactly representable in float32). One caveat: `np.var` promotes uint8 input to float64 but keeps float32 input in float32, so if you *also* keep `obs_rms` on CPU the variance changes by ~1e-7. Combined with A2 this is moot. **(e)** The `torch.equal` check in `bench_det_uint8.py`; `nvidia-smi`/`max_memory_allocated` for the memory.

### A6. Keep the observation-normalisation statistics on the GPU in float32
**(a)** `gym.wrappers.normalize.RunningMeanStd.__init__` uses `np.zeros(shape, dtype=np.float64)` — verified from source. So `torch.from_numpy(obs_rms.mean)` at L367/L451 produces a **float64** tensor, and every arithmetic step promotes to float64 before the trailing `.float()`. On the whole batch that means three transient float64 tensors of `16384×1×84×84 × 8 B = 924.8 MB` each. And the two statistics are re-uploaded host→device on **every one of the 128 rollout steps**. **(b)** Keep a device-resident float32 copy, refreshed once per iteration:
```python
obs_mean_g = torch.as_tensor(obs_rms.mean, dtype=torch.float32, device=device)
obs_std_g  = torch.sqrt(torch.as_tensor(obs_rms.var, dtype=torch.float32, device=device))
# then, in place of L365-370 and L449-454:
rnd_next_obs = ((next_obs[:, 3, :, :].reshape(-1, 1, 84, 84).float() - obs_mean_g) / obs_std_g).clip(-5, 5)
```
**(c)** Rollout: 0.164 → 0.042 ms/step = 16 ms/iteration. Whole batch: 21.3 → 5.0 ms. Total ~32 ms/iteration, plus ~2.8 GB of transient float64 allocation removed. Biggest relative effect on consumer cards where float64 runs at 1/32–1/64 of float32 rate (all the RTX/Quadro/GeForce parts here); smaller on A100 (1/2 rate). **(d)** Same formula, float32 instead of float64 intermediates before a cast to float32 that was happening anyway — differences of 1–2 ulp in float32. Not bit-identical, statistically irrelevant. **(e)** Time L449–454 alone; check `max_memory_allocated`.

### A7. Delete the three forced host-device syncs in the update loop
**(a)** Three separate stalls inside the innermost loop, 16 times per iteration:
- **L483** `clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]` — grep confirms `clipfracs` is assigned at L456, appended at L483, and **never read anywhere**. It is dead code whose only effect is a full pipeline drain.
- **L469** `mask = (mask < args.update_proportion).type(torch.FloatTensor).to(device)` — `torch.FloatTensor` is the **CPU** type (verified: `t.type(torch.FloatTensor).device == cpu`), so this does a device→host copy of 4096 floats and a host→device copy straight back, with a sync in between.
- **L471** `torch.tensor([1], device=device, dtype=torch.float32)` allocates a new device tensor from a Python list every minibatch.

**(b)**
```python
# L468-471
mask = (torch.rand(len(forward_loss), device=device) < args.update_proportion).float()
forward_loss = (forward_loss * mask).sum() / torch.clamp(mask.sum(), min=1.0)
# L456 and L483: delete both lines
```
**(c)** ~30–50 ms/iteration, from restored CPU/GPU overlap. All generations; relatively larger on fast GPUs where the CPU is closer to being the limiter. **(d)** **None.** `.float()` and `.type(torch.FloatTensor).to(device)` produce identical values; `torch.clamp(x, min=1.0)` equals `torch.max(x, tensor([1.]))`; `clipfracs` is unused. Bit-identical. **(e)** `torch.profiler` or `TORCH_LOGS` — or simply time the 16-minibatch loop before and after.

### A8. Rewrite the observation-normalisation init loop
**(a)** L326–335 calls `.tolist()` on a `(128,1,84,84)` uint8 array 6,400 times, building ~115.6M Python objects, then `np.stack` on a list of 16,384 nested lists 50 times. Measured on this machine:

```
.tolist() per env-step batch      :   25.24 ms   -> 2.7 min over 6,400 steps
np.stack of 2,048 nested lists    :  916.22 ms   -> ~7.3 s at the real 16,384 rows, ×50 = 6.1 min
direct uint8 buffer fill per batch:    0.12 ms   (210x faster)
```
**(b)**
```python
buf = np.empty((args.num_steps * args.num_envs, 1, 84, 84), dtype=np.uint8)
fill = 0
for step in range(args.num_steps * args.num_iterations_obs_norm_init):
    acs = np.random.randint(0, envs.single_action_space.n, size=(args.num_envs,))
    s, r, d, _ = envs.step(acs)
    buf[fill:fill + args.num_envs] = s[:, 3, :, :].reshape(-1, 1, 84, 84)
    fill += args.num_envs
    if fill == buf.shape[0]:
        obs_rms.update(buf); fill = 0
```
**(c)** ~9 minutes of pure Python per run → ~1 s. Over 30 seeds that is ~4.5 hours, and it removes a ~2 GB Python-object spike. All generations. **(d)** **Bit-identical — verified**: `np.mean`/`np.var` over the int64 stack and over the uint8 buffer are exactly equal (`mean True`, `var True`), because numpy promotes both integer dtypes to float64 for the moments. **(e)** The timing script above.

### A9. `torch.backends.cudnn.benchmark = True`
**(a)** Lets cuDNN autotune convolution algorithms; shapes here are fixed (128 and 4096), so it tunes once. **(b)** Add after L273. **(c)** Measured, fresh process per config:

| GPU | benchmark off | benchmark on | gain |
|---|---|---|---|
| RTX A4500 (sm_86), TF32 off | 102.26 ms | 69.09 ms | 1.48x |
| RTX A4500, TF32 on | 56.82 ms | 48.33 ms | 1.18x |
| Quadro RTX 6000 (sm_75) | 80.23 ms | 69.95 ms | 1.15x |
| RTX 2080 Ti (sm_75) update | 82.82 ms | 72.72 ms | 1.14x |
| RTX 2080 Ti rollout step | 4.90 ms | 3.37 ms | 1.46x |

**(d)** The tuning tries several algorithms; the docs say "the benchmark may select different algorithms on subsequent runs, even on the same machine." Individual convolutions may therefore use a different (still mathematically correct) algorithm than the default heuristic pick, so results are not bit-reproducible across runs. Combined with `cudnn.deterministic = True` (which the script sets), cuDNN benchmarks only among deterministic algorithms, so within a run it stays deterministic. **I classify this as same-algorithm/different-kernel: safe for a faithful replication, but not bit-identical.** Costs extra workspace memory (A4500 peak went 2.73 → 4.55 GB in my harness). **(e)** Time the first 3 iterations (tuning) separately from the steady state.

### A10. Overlap envpool stepping with GPU work (structural, largest remaining headroom)
**(a)** Today the loop is strictly serial: policy forward → `envs.step` on the CPU → RND forward. Roughly 1.3–3.4 s of every iteration is CPU-only with the GPU idle. envpool's `send`/`recv` API lets you issue the env step and do device work before blocking on the result. **(b)** Use `envs.send(action, env_id)` immediately after sampling, then do the previous step's RND forward and the `obs[step]`/`dones[step]` stores, then `envs.recv()`. **(c)** Upper bound is `min(t_env, t_gpu)/(t_env + t_gpu)` recovered — at 16 cores + A4500 with the other fixes applied, roughly **1.4–1.8x** on top. All generations. **(d)** Requires care: the RND bonus for step *t* depends only on `next_obs` from step *t*, so a one-step software pipeline preserves the values exactly — but getting the buffer indices and the episode-end bookkeeping right is where a silent off-by-one would change the data. Treat as same-numerics **only after** asserting a short run reproduces the serial version's `curiosity_rewards` tensor exactly. **(e)** Diff `curiosity_rewards`, `ext_values`, `int_values` against the serial implementation for 3 iterations at a fixed seed.

### A11. Async envpool mode (`batch_size < num_envs`)
**(a)** In sync mode every step waits for the slowest of 128 emulators. Async returns as soon as `batch_size` environments have finished. **(b)** `envpool.make(..., num_envs=192, batch_size=128, num_threads=<cores>)` plus the `send`/`recv` loop. **(c)** Measured at 32 cores: sync 128 envs 12,568 → async 192/128 **23,138 steps/s (1.84x)**; at 8 cores 4,780 → 5,575 (1.17x). Gain grows with core count. **(d)** **This changes the data distribution** — a rollout batch is no longer 128 synchronised environments but whichever 128 of 192 finished first, so environments contribute unevenly and the GAE bootstrap is over a different env set. **Different numerics. Do not use for a faithful replication of the published run.** **(e)** `bench_envpool.py` async rows.

### A12. Cheap leftovers
- **`torch.set_num_threads(1)` / `OMP_NUM_THREADS=1`** for the torch process. Almost all torch work is on the GPU; the only CPU-heavy torch op is the obs conversion, which A5 removes. Leaving OpenMP at the default spawns one thread per visible core and contends directly with envpool's workers. Set `OMP_NUM_THREADS=1` and give the cores to envpool. Same numerics. Measured indirectly: my envpool runs used `OMP_NUM_THREADS=1` throughout.
- **Pinned host memory + `non_blocking=True`** for the obs copy: 0.217 → 0.190 ms (12%). Small here because the copy is only 3.6 MB once A5 lands. Same numerics. Requires a persistent pinned staging buffer; envpool returns its own array, so you must `copy_` into the pinned buffer first — usually not worth it.
- **`optimizer.zero_grad(set_to_none=True)`** — **already the default.** `torch.optim.Optimizer.zero_grad` signature in 2.10 is `(self, set_to_none: bool = True)`. Measured difference: A4500 99.59 vs 100.10 ms, i.e. none. **No action needed.**
- **Hoist the episode-logging syncs** (L374–388): `np.mean(curiosity_rewards[step].cpu().numpy())` and passing the CUDA tensor `curiosity_rewards[step][idx]` to `add_scalar` force a sync per finished episode. Compute one host-side copy per rollout step outside the `for idx, d` loop. Same numerics.

---

## 4. Tier B — faster but **different numerics**

These change what the network computes. Use them only if you accept a deviation from the published implementation, and if so validate on at least 3 seeds against a Tier-A-only reference.

### B1. AMP fp16 + `channels_last` — the largest single GPU-side win
Measured minibatch update, 4096, fresh process per config:

| Config | Quadro RTX 6000 (sm_75) | RTX 2080 Ti (sm_75) | RTX A4500 (sm_86) | RTX 4000 Ada (sm_89) |
|---|---|---|---|---|
| fp32 + cudnn.benchmark + TF32 flags | 71.93 ms | 73.74 ms | 48.33 ms | 71.94 ms |
| + `channels_last` (no AMP) | 83.66 ms | 86.29 ms | 49.58 ms | 76.40 ms |
| AMP **bf16** | **2951.15 ms** | **2984.82 ms** | 43.78 ms | 61.64 ms |
| AMP bf16 + `channels_last` | **2961.31 ms** | **2952.30 ms** | 36.95 ms | 56.32 ms |
| AMP **fp16** + GradScaler | 47.46 ms | 47.85 ms | 40.45 ms | 54.04 ms |
| AMP fp16 + `channels_last` + GradScaler | **39.42 ms** | **39.05 ms** | **29.83 ms** | **45.39 ms** |

**(b)**
```python
agent = Agent(envs).to(device, memory_format=torch.channels_last)
rnd_model = RNDModel(4, envs.single_action_space.n).to(device, memory_format=torch.channels_last)
scaler = torch.amp.GradScaler("cuda")
...
with torch.autocast("cuda", dtype=torch.float16):
    ...  # everything from the RND forward through `loss = ...`
optimizer.zero_grad()
scaler.scale(loss).backward()
scaler.unscale_(optimizer)                       # required before clip_grad_norm_
nn.utils.clip_grad_norm_(combined_parameters, args.max_grad_norm)
scaler.step(optimizer); scaler.update()
```
**(c)** 1.6–1.8x on the update phase across Turing, Ampere and Ada. **(d)** Large risk, and RND-specific: the intrinsic reward is `(target − predictor)²/2`, so running the RND towers in fp16 changes the **scale and dynamic range of the exploration bonus**, which then feeds `reward_rms` and the intrinsic advantage. Montezuma exploration is known to be sensitive to intrinsic-reward scaling. If you use AMP at all, run the RND target tower in fp32 (`torch.autocast(..., enabled=False)` around it) and only autocast the policy update. GradScaler is mandatory for fp16 and must be `unscale_`d before gradient clipping or `max_grad_norm` clips the *scaled* gradients — a silent correctness bug. **(e)** Compare `curiosity_rewards` distributions and `losses/fwd_loss` curves against fp32 over 200 iterations.

> **bf16 trap.** On Turing (sm_75) bf16 is **41x slower than fp32** — 2951 ms vs 71.9 ms, reproduced independently on a Quadro RTX 6000 and an RTX 2080 Ti. bf16 tensor cores exist only from Ampere (sm_80) onward; PyTorch 2.10 emulates it on older parts. And `torch.cuda.is_bf16_supported()` **will not save you**: its signature is `is_bf16_supported(including_emulation: bool = True)`, and it returned `True` on the GTX 1080 in my run. Gate on `torch.cuda.get_device_properties(0).major >= 8` or `is_bf16_supported(including_emulation=False)`.

### B2. `torch.backends.cuda.matmul.allow_tf32 = True` — Ampere+ only
**(a)** Enables TF32 (10-bit mantissa) for `Linear`/matmul. **(b)** `torch.backends.cuda.matmul.allow_tf32 = True`, or `torch.set_float32_matmul_precision("high")`. **(c)** Small here, because convolutions dominate the FLOPs:

| GPU | matmul TF32 alone | with cudnn.benchmark |
|---|---|---|
| RTX A4500 | 102.26 → 98.31 ms (1.04x) | 56.82 → 52.55 ms (1.08x) |
| Quadro RTX 6000 (sm_75) | 80.23 → 78.58 ms (no-op, no TF32 hardware) | — |

**(d)** Reduces mantissa precision; the docs note "relative error compared to double precision is approximately 2 orders of magnitude larger." Different numerics. **(e)** Toggle in a fresh process.

> **Important correction to the usual advice: `torch.backends.cudnn.allow_tf32` defaults to `True`.** So this script **already runs its convolutions in TF32 on Ampere+**, and that default is worth 1.80x on the A4500 (102.26 ms with it off vs 56.82 ms with it on). The published CleanRL result was produced with TF32 convolutions on. **Do not turn it off** in the name of fidelity — that would deviate from the reference *and* cost you 1.8x.

### B3. `--torch_deterministic False`
**(a)** L273 sets `torch.backends.cudnn.deterministic = args.torch_deterministic`, default `True`. **(c)** Measured (fresh process, cuDNN TF32 on as PyTorch defaults):

| GPU | deterministic=True | deterministic=False | gain |
|---|---|---|---|
| RTX A4500 | 57.89 ms | 56.69 ms | 1.02x |
| Quadro RTX 6000 | 83.99 ms | 78.21 ms | 1.07x |

**(d)** Loses per-run reproducibility of the convolution backward. **(e)** As above. Low value; keep determinism.

### B4. Increase `num_envs`
**(a)** More parallel emulators, larger batch. **(c)** envpool side at 32 cores: 128 envs 12,568 → 256 envs 16,644 steps/s (**1.32x**); at 8 cores 4,780 → 4,977 (1.04x, already saturated). GPU side improves too because batch 128 badly underfills a modern GPU (the whole workload runs at roughly 1.1 TFLOP/s effective — an A100 peaks near 156 TF32 TFLOP/s). **(d)** **Changes the algorithm**: `batch_size` becomes 32768, `num_iterations` halves, the LR-anneal schedule changes, and the number of gradient updates for the same step budget halves. This is a hyperparameter change, not an optimization. **(e)** Sweep `num_envs` and record both steps/s and the learning curve.

---

## 5. Tier C — measured to not help, or to hurt

| Item | Measured result | Verdict |
|---|---|---|
| `optim.Adam(fused=True)` | A4500 48.33 → 48.78 ms; 99.98 vs 99.59 ms in the sequential harness; RTX 6000 71.93 → 70.49 ms | **No effect.** ~3.6M parameters is negligible against a 4096-sample forward/backward. The `foreach` path is already the default when all tensors are CUDA. |
| `zero_grad(set_to_none=False)` | A4500 100.10 vs 99.59 ms | **No effect** (and `True` is already the default). |
| `channels_last` **without** AMP | A4500 48.33 → 49.58; RTX 6000 71.93 → 83.66; 2080 Ti 73.74 → 86.29; Ada 71.94 → 76.40 | **Hurts.** Makes sense: the policy input has C=4 and the RND input C=1, both far from the multiple-of-8 that NHWC tensor-core kernels want; with C=1, NHWC is layout-identical to NCHW so you pay for permutes and get nothing. Only pair it with AMP. |
| `torch.compile` on the rollout, `default` mode | A4500 1.573 → 1.482 ms (1.06x), 9.6 s compile; RTX 6000 1.966 → 1.790 ms (1.10x), 10.8 s; 2080 Ti 2.545 → 2.227 ms (1.14x), 15.2 s | Marginal; see §6. |
| `torch.compile` on the update, `default` mode | A4500 48.33 → 44.47 ms (1.09x), 7.9 s compile; RTX 6000 72.38 → 64.41 ms (1.12x), 12.4 s; 2080 Ti 73.47 → 66.20 ms (1.11x), 15.7 s | Modest, real, cheap to try. |
| `mode="reduce-overhead"` | **Fails at runtime on all three GPUs** | See §6. |
| Manual `torch.cuda.CUDAGraph` capture of the rollout step | **Fails on all three GPUs**: `cudaErrorStreamCaptureInvalidated` | See §6. |
| envpool XLA interface | Docs: works with "Jax/Tensorflow"; PyTorch is not supported | Not applicable. |
| `thread_affinity_offset` | Binds to absolute core ids modulo the machine's core count, not cgroup-aware | Leave at `-1` on a shared cluster. |

---

## 6. torch.compile and CUDA graphs — exactly what breaks

**Compile modes** (`torch.compile` docs): `"default"` is "a good balance between performance and overhead"; `"reduce-overhead"` "reduces the overhead of python with CUDA graphs, useful for small batches"; `"max-autotune"` "leverages Triton or template based matrix multiplications ... It enables CUDA graphs by default on GPU".

**What actually happened here:**

1. **`reduce-overhead` fails on this workload.** Measured on A4500, Quadro RTX 6000 and RTX 2080 Ti:
   ```
   RuntimeError: Error: accessing tensor output of CUDAGraphs that has been overwritten by
   a subsequent run. ... To prevent overwriting, clone the tensor outside of torch.compile
   ```
   This is the documented CUDAGraph Trees limitation: replaying the graph overwrites the previous outputs. The rollout loop *stores* the outputs (`actions[step] = action`, `logprobs[step] = logprob`, `ext_values[step] = ...`) and reads them 128 steps later at GAE time — exactly the pattern that breaks. Workarounds are `torch.compiler.cudagraph_mark_step_begin()` each iteration, or cloning every output, which gives back much of the saving.

2. **Manual CUDA graph capture fails too**, on all three GPUs:
   ```
   AcceleratorError: CUDA error: operation failed due to a previous error during capture
   (cudaErrorStreamCaptureInvalidated)
   ```
   Root cause: `Categorical.sample()` calls `torch.multinomial`, and the CUDAGraph Trees docs list `aten.multinomial.default` among the "incompatible operators" that prevent graphing. You would have to replace the categorical sample with a Gumbel-argmax (`(logits + gumbel_noise).argmax(-1)`) — which **changes the sampling RNG stream and therefore the trajectory**, i.e. Tier B.

3. **Things in this script that force graph breaks** (docs: `.item()`, printing/logging, data-dependent control flow, non-traceable C functions):
   - L362 `envs.step(action.cpu().numpy())` — leaves the graph entirely.
   - L367–368 `torch.from_numpy(obs_rms.mean)` — numpy round trip each step.
   - L374–388 `for idx, d in enumerate(done): if d and info["lives"][idx] == 0:` — data-dependent Python control flow plus `print` plus `.cpu()`.
   - L391 `curiosity_rewards.cpu().data.numpy().T` and the `RewardForwardFilter` Python loop.
   - L444 `.cpu().numpy()`, L469 `.type(torch.FloatTensor)`, L483 `.item()`.
   - L529–536 six `.item()` calls.

   **Therefore do not try to compile the training loop.** Compile the three `nn.Module`s only — `agent.network`, `rnd_model.predictor`, `rnd_model.target` — where the graph is clean.

4. **Dynamic shapes:** the rollout uses batch 128, the update 4096. `torch.compile` will specialise per shape and recompile once for each; the default `torch._dynamo.config.recompile_limit` is 8, so two shapes is fine. Do not let the last minibatch be ragged (`16384 / 4` divides evenly here, so it is not).

5. **Compile-time cost:** first compile 7.9–15.7 s per module per shape on these GPUs (33.1 s for the very first inductor invocation in a cold process, which includes Triton/inductor startup). Amortised over 122,070 iterations this is free; but it is paid **per Slurm job**, so with 30 seeds and requeues it is worth setting `TORCHINDUCTOR_CACHE_DIR` to a shared path.

6. **Pascal/Turing support:** Inductor generates Triton kernels, which need sm_70+. Turing (sm_75) works — measured. Pascal is moot given §0.

**Verdict:** `torch.compile(mode="default")` on the three modules is worth ~1.06–1.14x and is low-risk (Inductor may fuse pointwise ops and reassociate, so not bit-identical). `reduce-overhead`/`max-autotune` are not usable here without changing the sampling op.

---

## 7. Per-GPU-generation cheat sheet

| Optimization | Pascal sm_61 | Turing sm_75 | Ampere sm_80/86 | Ada sm_89 | Hopper sm_90 |
|---|---|---|---|---|---|
| Runs at all on torch 2.10+cu128 | **No** | Yes | Yes | Yes | Yes |
| `cudnn.benchmark=True` | n/a | 1.14–1.15x update, 1.46x rollout | 1.18–1.48x | expect similar | expect similar |
| cuDNN TF32 (default **on**) | n/a | no hardware, no-op | **1.80x — already on** | on | on |
| `matmul.allow_tf32` | n/a | no-op | 1.04–1.08x | small | small |
| `channels_last` alone | n/a | **0.86x (hurts)** | 0.97x | 0.94x | likely neutral |
| AMP fp16 + channels_last | n/a | **1.82x** | **1.62x** | **1.59x** | expect ≥1.6x |
| AMP bf16 | n/a | **0.02x — 41x SLOWER** | 1.10x (1.31x with CL) | 1.17x (1.28x with CL) | expect ≥1.3x |
| `torch.compile` default | n/a | 1.11–1.14x | 1.06–1.09x | expect similar | expect similar |
| `reduce-overhead` / CUDA graphs | n/a | **fails** | **fails** | **fails** | **fails** (op-level, not arch-level) |
| Tier A rollout/host fixes (A2–A8) | n/a | apply | apply | apply | apply |

A100 (sm_80) and H100 NVL (sm_90) measurements are queued as Slurm jobs 6533816 and 6533818 — both nodes are fully allocated to other users right now (`serval03` is under reservation `nkp2mr_155` until 2026-08-07). Results will appear at `bench_scratch/results/det_a100.json` and `det_h100.json` when they run. The Ada sm_89 column is the closest measured proxy for Hopper's relative ordering; absolute Hopper numbers will be much faster, which *increases* the relative importance of the CPU/envpool side (§3, A1) and of the packing strategy in §9.

---

## 8. Practical note: envpool version and API

The pinned dependency is `envpool>=0.6.4,<0.7`, but the current release is **1.2.5** (Python 3.11–3.14 wheels; `requires_python>=3.11`). I installed and verified it works, but its `env_type="gym"` now returns a **Gymnasium-style API**: `reset()` returns `(obs, info)` and `step()` returns **5** values (`obs, reward, terminated, truncated, info`), not the 4 the script expects at L114 and L362. You must either pin an old envpool or adapt `RecordEpisodeStatistics` and the two `envs.step(...)` call sites. This is a correctness matter, not a throughput one, but it will bite on first run.

---

## 9. How to measure, and a plan for the 30 seeds

**Measurement protocol** (each knob in a fresh process — cuDNN autotune caches and allocator state carry over and will fabricate speedups if you toggle knobs in one process; I hit exactly this and had to re-run):

1. Time the three phases separately with `torch.cuda.Event(enable_timing=True)` + `torch.cuda.synchronize()`: the 128-step rollout, the 16-minibatch update, and the per-iteration whole-batch block (L390–L454).
2. Report `charts/SPS` only after ~20 iterations — the script's `global_step / (time.time() - start_time)` includes the multi-minute normalisation-init loop and understates steady-state throughput early on.
3. Separate env from GPU: run the loop once with the policy replaced by `np.random.randint` to get the pure envpool rate, and once with `envs.step` replaced by a cached observation to get the pure torch rate. Their sum should match the real wall time; if it does not, you have found overlap or a hidden sync.
4. `torch.profiler.profile(activities=[CPU, CUDA])` for one iteration, and look for gaps on the CUDA stream — those are the syncs from A7.
5. `TORCH_LOGS="graph_breaks"` before trying any `torch.compile`.

**Recommended job shape on this cluster** (jaguar03, 8× RTX A4500, 224 CPUs):

```
--cpus-per-task=24  --gpus-per-node=1  --mem=48G   OMP_NUM_THREADS=1
envpool.make(..., num_threads=24)
```
24 cores × 8 GPUs = 192 CPUs, inside jaguar03's 224. Estimated ~7,000–8,000 steps/s per run with Tier A applied, i.e. **~70–80 h/seed instead of 250 h**, and 8 seeds in parallel on one node.

**The biggest lever for a 30-seed campaign is not in this list: it is packing.** My harness peaked at 4.5 GB device memory (2.7–3.4 GB with the Tier A/B fixes), against 20 GB on an A4500 and 94 GB on an H100 NVL, and the workload runs at roughly 1.1 TFLOP/s effective — a small fraction of any of these GPUs. Two to four runs per GPU, each with its own CPU slice, will get closer to linear scaling in seeds-per-GPU-hour than any single-run optimization here. Validate by running 2 and 4 concurrent processes on one GPU and checking that aggregate steps/s rises; the limit will be CPU cores, not the GPU.

---

## Sources checked

- [torch.compile (PyTorch 2.10)](https://docs.pytorch.org/docs/2.10/generated/torch.compile.html) — mode semantics, reduce-overhead memory caveat
- [CUDA semantics (PyTorch 2.10)](https://docs.pytorch.org/docs/2.10/notes/cuda.html) — TF32 defaults (`matmul.allow_tf32` False, `cudnn.allow_tf32` True), CUDA graph constraints, pinned memory
- [torch.amp](https://docs.pytorch.org/docs/2.10/amp.html) — autocast dtypes, GradScaler and fp16 underflow
- [torch.optim.Adam](https://docs.pytorch.org/docs/2.10/generated/torch.optim.Adam.html) — foreach is the CUDA default; fused dtype support
- [Channels-last memory format tutorial](https://docs.pytorch.org/tutorials/intermediate/memory_format_tutorial.html) — evidence is all Tensor-Core + fp16
- [Performance tuning guide](https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html) — cudnn.benchmark, the list of sync-forcing ops, set_to_none
- [Reproducibility notes](https://docs.pytorch.org/docs/2.10/notes/randomness.html) — cost of determinism, benchmark vs reproducibility
- [torch.compile troubleshooting](https://docs.pytorch.org/docs/2.10/user_guide/torch_compiler/torch.compiler_troubleshooting.html) — graph-break causes, `recompile_limit` = 8
- [CUDAGraph Trees](https://docs.pytorch.org/docs/2.10/user_guide/torch_compiler/torch.compiler_cudagraph_trees.html) — output overwrite, `aten.multinomial.default` incompatibility, skip conditions
- [envpool python interface](https://envpool.readthedocs.io/en/latest/content/python_interface.html) and [benchmark](https://envpool.readthedocs.io/en/latest/content/benchmark.html) — `num_threads`/`batch_size`/`thread_affinity_offset` semantics, published FPS
- [envpool `async_envpool.h`](https://github.com/sail-sg/envpool/blob/main/envpool/core/async_envpool.h) and [`env_spec.h`](https://raw.githubusercontent.com/sail-sg/envpool/main/envpool/core/env_spec.h) — `hardware_concurrency()`, defaults
- [envpool XLA interface](https://envpool.readthedocs.io/en/latest/content/xla_interface.html) — Jax/TensorFlow only
- [CleanRL PPO-RND docs](https://docs.cleanrl.dev/rl-algorithms/ppo-rnd/) — 2000M steps, ~250 hours, 7100 vs 8152
- [NVIDIA Turing](https://docs.nvidia.com/cuda/turing-compatibility-guide/index.html) / [Ada](https://docs.nvidia.com/cuda//ada-compatibility-guide/index.html) compatibility guides — cubin forward-compatibility within a major version, PTX JIT fallback
- [PyTorch 2.8+cu128 drops Pascal](https://github.com/bnsreenu/digitalsreeni-image-annotator/issues/57)