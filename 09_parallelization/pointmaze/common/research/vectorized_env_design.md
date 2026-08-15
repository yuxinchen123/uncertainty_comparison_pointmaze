# Design brief: a massively parallel batched PointMaze on one H100

Scope: the three Module-1 implementations (PyTorch tensor ops, one fused CUDA kernel, JAX), sized
for 10^3 to 10^8 simultaneous environments on a single H100 NVL. Physics contract:
`../physics_spec.md`. Hardware: serval05, one H100 NVL — 132 SMs, 94 GB HBM3, 3,938 GB/s peak
memory bandwidth, ~67 TFLOP/s fp32 (non-tensor).

## 1. Five facts about this workload that decide every design choice

1. **It is memory bound by a factor of about 23.** One env step is roughly 40 floating-point
   operations over roughly 53 bytes of memory traffic, an arithmetic intensity of 0.75 FLOP/byte.
   The H100's fp32 balance point is 67e12 / 3.9e12 = 17.2 FLOP/byte. Nothing about the arithmetic
   is worth optimizing. Every real optimization either moves fewer bytes or launches fewer kernels.
2. **The state is four floats plus two small counters.** Position, velocity, a step counter, and a
   reset counter. There is no contact solver, no constraint graph, no per-env geometry. This is
   closer to a grid-world than to Brax or MJX, so the design should copy gymnax and Madrona (tiny
   state, huge batch, everything resident) rather than MJX (large per-env state, batch limited by
   memory).
3. **The ball cannot move more than 0.0523 m in one step**, because the velocity clip gives a
   terminal speed of 5.226256 m/s per axis and the timestep is 0.01 s. Cells are 1.0 m. So the ball
   can never tunnel through a wall, can never cross more than one cell boundary per step, and
   always ends the step inside the same cell it started in or an adjacent free one. Continuous
   collision detection is unnecessary and single-cell locality is guaranteed.
4. **The whole maze is 108 bytes.** The Large map is 9 rows x 12 columns of one byte each. It fits
   in CUDA constant memory, in a Triton constant, and in an XLA-inlined literal. Wall lookups cost
   no HBM traffic at any batch size.
5. **At large batch the policy costs more than the environment.** A 4->64->64->2 MLP forward is
   about 8,960 FLOP per env; the env step is about 40. At one million envs the env step takes
   about 16 microseconds of bandwidth time and the policy takes 90-180 microseconds of matmul time.
   Module 1 has a ceiling on end-to-end value, and the optimization loop should stop when the env
   step is well under 10% of the loop.

## 2. Shared roofline and the numbers that define "good"

### Bytes moved per env-step

Three accounting cases. All in fp32 state with int32 counters.

| case | traffic per env-step | what it assumes |
|---|---|---|
| minimal fused | **53 B** | state read+write 32 B, action read 8 B, packed counter read+write 8 B, reward write 4 B, done write 1 B; obs is the state buffer itself |
| training-shaped | **77 B** | the same, plus a separate 16-byte obs write into a rollout buffer, plus unpacked 32-bit step and reset counters |
| PyTorch eager, unfused | **240-400 B** | 15-25 intermediate `[N,2]` tensors, each read and written to HBM |

The 53-byte figure is the one to design against; the 77-byte figure is the one a real PPO rollout
will actually pay, because the trainer needs a stacked `[T, C, E, 4]` observation buffer.

### Sustained throughput to expect

Take achievable bandwidth as 85% of peak, i.e. 3.3 TB/s (a well-written streaming kernel on H100
reaches 80-90% of peak; the prompt's ~3.3 TB/s figure is the H100 SXM peak, and it happens to be
the right *achievable* figure for the NVL card in this box).

| case | roofline | realistic target (60-80% of roofline) |
|---|---|---|
| minimal fused, 53 B | 62 G env-steps/s | **37-50 G env-steps/s** |
| training-shaped, 77 B | 43 G env-steps/s | **26-34 G env-steps/s** |
| eager unfused, 320 B | 10 G env-steps/s | not reachable — launch bound first |

Read as time-per-step at a given batch, which is the more useful form:

| total envs N | minimal-fused step time at 3.3 TB/s | regime |
|---|---|---|
| 10^4 | 0.16 us | launch bound, 20-100x off roofline |
| 10^5 | 1.6 us | launch bound for one kernel, badly so for eager torch |
| 2.7 x 10^5 | 4.3 us | one full occupancy wave (132 SM x 2048 threads = 270,336) |
| 10^6 | 16 us | approaching bandwidth bound |
| 10^7 | 160 us | bandwidth bound |
| 10^8 | 1.6 ms | bandwidth bound |

### Where the curve bends, and why

- **Kernel launch overhead** is 3-7 microseconds for a null kernel launched normally, and roughly
  1-2 microseconds for a CUDA-graph replay of the same work. A single fused kernel therefore stops
  being launch bound at about N = 3-5 x 10^5, and at about N = 10^5 under CUDA graphs.
- **PyTorch eager** issues 20-40 kernels per step, so its floor is 100-300 microseconds of CPU
  dispatch per step no matter how small the batch. It would need N of order 10^7 before the GPU
  became the limit. This is the single strongest argument for `torch.compile` plus CUDA graphs in
  the torch variant — not a 20% tuning gain, a 20x structural one.
- **L2 cache is 50 MB.** At 53 bytes per env the working set fits in L2 up to about 900,000 envs.
  In that range measured throughput can *exceed* the HBM roofline, because the traffic never
  reaches HBM. Do not compute "percent of peak bandwidth" in this regime; it will read above 100%
  and look like a measurement error when it is a cache hit.
- **Occupancy fill** is 270,336 resident threads at one env per thread. Below that the machine is
  not full even if the launch overhead were zero.

Expect the log-scale throughput curve the final report asks for to have four visible parts: a flat
launch-bound plateau, a steep rise, an L2-assisted bump, and a flat HBM-bound asymptote.

### Memory footprint, and what actually caps the batch

Resident env state is about 40 bytes per env including the action buffer. 10^6 envs is 40 MB;
10^8 envs is 4 GB. The environment is not what limits batch size on a 94 GB card. The rollout
buffer is: at T = 128 steps and about 37 bytes per stored transition, 10^6 envs needs 4.7 GB and
10^7 envs needs 47 GB. Plan training at N up to about 10^6, and use N up to 10^8 only for the
env-only throughput curve.

## 3. Decisions that are the same in all three variants

Fixing these first means the three implementations can be diffed element by element, which is the
cheapest correctness tool available.

### 3.1 Memory layout: SoA across envs, packed 4-vector within an env

Store state as one flat `[N, 4]` buffer of `(x, y, vx, vy)`, N = n_copies x n_envs, with the
4-vector contiguous. Expose `[n_copies, n_envs, 4]` by reshape, which is free because a contiguous
`[N,4]` buffer and a contiguous `[C,E,4]` buffer are the same bytes.

- Across envs this is structure-of-arrays: env i and env i+1 are adjacent, so a warp reads 32
  consecutive envs and every load is coalesced.
- Within an env the four floats are packed, which is exactly one `float4` (16-byte) vectorized
  load in CUDA and exactly the `[C, E, 4]` matrix the policy wants as input to a batched matmul.
  No transpose anywhere in the loop.
- The one place to reconsider is JAX, where XLA often prefers four separate `[C,E]` planes over one
  `[C,E,4]` array because the packed last axis of 4 can block its vectorizer. Treat planar-vs-packed
  as a measured experiment in the JAX variant, not a fixed decision.

Keep the two batch axes as *axes*, not as a single flattened dimension, at the interface. The env
kernel treats them as one flat range; only per-copy quantities (network parameters, per-copy base
seed, per-copy episode statistics) need the copy axis materialized. The policy needs `[C, E, in] x
[C, in, out]` batched matmuls, so C must survive to the network.

### 3.2 Wall handling: one byte of table, four clamps, one corner test

This is exact for face contacts, branch free, and costs no HBM traffic.

- Precompute, per cell, an 8-bit neighbour mask: one bit for each of the 4 faces and each of the 4
  diagonal neighbours being a wall. 9 x 12 = 108 bytes total. Put it in CUDA `__constant__` memory,
  a small PyTorch buffer that stays L2-resident, or a JAX constant XLA inlines.
- Index the cell from the **pre-step** position: `j = floor(x + W/2)`, `i = floor(H/2 - y)` with
  W = 12, H = 9 (row 0 at top, y up). The pre-step cell is always free and the ball centre is
  always at least r = 0.1 from any blocked face of it.
- After the integrator step, for each of the 4 faces: if that face's bit is set, clamp the position
  to `face +/- r` and zero the inward velocity component. Faces whose bit is clear impose no bound,
  which is what lets the ball move into an adjacent free cell. Because the per-step displacement is
  at most 0.0523 m and the clamp keeps the centre 0.1 m from blocked faces, the pre-step cell is
  always a valid frame for the whole step. No neighbour search, no loop.
- Corners are real in the Large map: cell (3,1) is free, its diagonal neighbour (2,2) is a wall, and
  both shared edge neighbours are free, so a diagonal move can clip the wall's corner. Test **only
  the nearest corner**, selected by the signs of `x - cell_centre_x` and `y - cell_centre_y`; if
  that diagonal bit is set and the centre is within r of the corner point, push the centre out
  radially to distance r and remove the inward radial velocity component. One extra lookup, one
  distance test.
- Use squared distances everywhere, including the goal test (`dist2 <= 0.45^2`). No `sqrt` in the
  step path except the single corner projection, which can be guarded so it is only evaluated where
  it matters or computed unconditionally with `rsqrt`.

### 3.3 Keyed counter-based RNG: one hash, written three times

The project's seeding rule requires each draw to be a pure function of
`(base_seed, copy_index, env_index, reset_counter, draw_index)`, so adding a parameter never shifts
existing draws. Framework RNGs do not give this: `torch.rand` is positional, `jax.random.split`
chains are order dependent, and `curand_init` with a large subsequence is expensive (it performs a
2^67 skip-ahead).

**Recommendation: write out Philox-4x32-10 by hand in integer ops, identically in all three
implementations.** It is 10 rounds of two 32-bit multiply-high/multiply-low pairs plus xors — about
40 integer operations — and one call produces four 32-bit words, which is exactly the four uniforms
a reset needs (start x noise, start y noise, goal x noise, goal y noise).

- key = `(base_seed, copy_index)`, counter = `(env_index, reset_counter, draw_stream, 0)`.
- CUDA: call the stateless `curand_Philox4x32_10(uint4 counter, uint2 key)` device function from
  `curand_kernel.h` directly. Never call `curand_init` in the step path.
- PyTorch: int32 tensor ops. Torch has no unsigned 32-bit type, but int32 multiply wraps like C, so
  the arithmetic is bit-identical if the final conversion to a uniform masks the sign bit
  explicitly. Inductor fuses these integer ops into the same kernel as the physics.
- JAX: the same code in `jnp.uint32`, which has proper unsigned wrapping. `jax.random.fold_in` on a
  traced integer is a legal and vectorizable alternative (it is a threefry hash), but using the same
  hand-written Philox is what buys bit-identical resets across the three variants.

The payoff is that the three implementations can be compared trajectory by trajectory from the same
seed, not just distributionally. The reference MuJoCo env's noise stream is not reproduced by any of
them, per the physics spec; that comparison stays distributional.

### 3.4 Auto-reset with no host involvement

Always compute the reset state unconditionally and select. Never branch on device data from Python.

```
next_state  = integrate_and_resolve_walls(state, action)
reward      = goal_test(next_state)
done        = (step_counter + 1 >= 400)            # plus terminal test if continuing_task=False
reset_state = draw_reset(base_seed, copy, env, reset_counter + 1)
state_out   = where(done, reset_state, next_state)
step_out    = where(done, 0, step_counter + 1)
reset_out   = reset_counter + done
```

Notes that matter:

- **Expose the terminal observation.** PPO must bootstrap from the observation *before* the reset.
  Return `next_state` as `final_obs` alongside the post-reset `state_out`, or adopt the alternative
  convention of resetting at the start of the following step. Getting this wrong is silent: the
  value target is computed against the reset state and the run trains, badly.
- **Stagger the initial step counters.** With a fixed start cell and a 400-step cap, every env
  resets on the same step, which correlates the batch and produces a periodic throughput spike. Draw
  each env's initial step counter uniformly from [0, 400). It costs nothing because the reset path
  is branch free either way.
- The in-phase case has one genuine advantage worth measuring once: if all envs reset together, the
  done flag is a host-known constant and the reset code can be hoisted out of the per-step kernel
  entirely. Measure it, then decide whether decorrelation is worth the difference.
- Forbidden in the step path in every framework: `nonzero`, boolean mask indexing, `masked_select`,
  `unique`, any operation whose output shape depends on data. These force a device-to-host
  synchronization and make CUDA-graph capture and `jit` impossible.

### 3.5 What to fuse

The target is one kernel per step, then one launch per rollout.

- **Level 1**: integrate + wall resolve + reward + termination + reset draw + select, in one
  kernel or one compiled function. This is where 240-400 bytes per step collapses to 53-77.
- **Level 2**: the whole T-step rollout, including the policy forward, in one CUDA graph
  (PyTorch) or one `lax.scan` under one `jit` (JAX). This removes T x (env launches + policy
  launches) of overhead, and is the structural idea behind PureJaxRL and DeepMind's Anakin
  architecture.
- **Level 3** (stretch, Module 3): one persistent kernel per copy that holds the policy weights in
  shared memory and loops T steps with the state in registers. Interior steps then never touch HBM
  for state at all, dropping the per-step traffic to the obs and action writes. This is what Madrona
  does structurally, and it is only worth attempting after levels 1 and 2 are measured.

## 4. Variant 1 — PyTorch tensor ops

### v0 design

- One module holding preallocated buffers: `state [N,4] float32`, `counters [N] int32` (step count
  in the low 16 bits, reset count in the high 16 bits), `reward [N] float32`,
  `done [N] uint8`, `action [N,2] float32`, plus the 108-byte maze mask as a registered buffer.
  N = n_copies x n_envs; `state.view(C, E, 4)` is the public observation view and is free.
- `step(action) -> (obs, reward, done, final_obs)` writing into the preallocated buffers with
  `out=` where possible, so that no allocator activity happens in the steady state. This is
  mandatory for CUDA-graph capture later, and it is easier to build in from the start than to
  retrofit.
- All constants as Python floats baked into the graph, or as 0-dim GPU tensors created once. Never
  create a tensor from a Python scalar inside the step.
- Correctness first: a `float64` mode that passes the fixture checks at 1e-9, and a `float32` mode
  checked at 1e-4, before any performance work.

### Ordered optimization steps

1. **Baseline in eager.** Record kernel count from `torch.profiler` and steps/s at N = 10^4 ...
   10^7. Expect a flat launch-bound line; that flat line is the thing every later step attacks.
2. **`torch.compile(fullgraph=True, dynamic=False)`.** `fullgraph=True` makes any graph break an
   error rather than a silent slowdown; `dynamic=False` stops Inductor from generating a
   shape-generic kernel. Inspect the generated Triton with `TORCH_LOGS=output_code` and count the
   kernels: the goal is 1-3, and anything above that names the operation that failed to fuse.
3. **Delete every synchronization.** Run development with
   `torch.cuda.set_sync_debug_mode("error")`, which turns any accidental `.item()`, `.cpu()`,
   `bool(tensor)`, or data-dependent-shape operation into a traceback instead of a mystery 10x.
   Episode-return accounting accumulates into GPU tensors and is read once per logging interval,
   not per step.
4. **CUDA graphs.** Two routes: `mode="reduce-overhead"` on `torch.compile`, or manual capture with
   `torch.cuda.CUDAGraph` and static input/output buffers. Prefer manual capture here — it is
   explicit about which buffers are static, it composes with the rollout loop in step 5, and it
   avoids the known throughput regression in `reduce-overhead` in recent PyTorch versions. Capture
   requirements: static shapes, static addresses, no CPU synchronization inside the captured
   region, warmup on a side stream before capture.
   **The classic bug**: a graph replay writes into the same output memory every time, so any tensor
   you keep from a replay is overwritten by the next one. Copy out, or write into a `[T, ...]`
   rollout buffer indexed by a captured-constant offset.
5. **Capture the whole rollout.** Capture T env steps plus T policy forwards as one graph, so a
   T = 128 rollout is one launch instead of hundreds. This is the largest remaining win in the
   torch path after fusion.
6. **Layout sweep.** Measure `[N,4]` packed against a planar `[4,N]`. Packed should win in Inductor
   because it vectorizes the last axis to a 16-byte load, but Triton's handling of a size-4 trailing
   dimension is worth checking rather than assuming.
7. **Shrink the counters.** One packed int32 instead of two int64s takes counter traffic from 32
   bytes per step to 8. On a 53-byte budget that is a 20% change, not a rounding error.
8. **Batch axes.** Confirm that `[C,E,4]` versus flat `[N,4]` makes no difference to the env kernel
   (it should not — both are one flat elementwise loop) and keep whichever the policy prefers.
9. **`torch.set_float32_matmul_precision("high")`** for the policy matmuls once the network enters
   the loop. It does not affect the env step, which has no matmuls.
10. **`mode="max-autotune-no-cudagraphs"`** to let Inductor tune tile and block sizes, then hand the
    tuned function to your own graph capture from step 4.

### Expected bottlenecks

- N below about 10^6: CPU dispatch and launch overhead, entirely. Every gain comes from steps 2, 4,
  and 5, and none from arithmetic.
- N above about 10^6: HBM bandwidth on the 53-77 bytes, plus the rollout buffer store. Check the
  achieved fraction of peak; below about 60% means the generated kernel is not vectorizing or is
  re-reading something.
- Anywhere: an accidental `float64` promotion from a Python constant, which doubles state traffic
  and is invisible in the profile unless you look at the dtypes in the generated code.

## 5. Variant 2 — one fused CUDA kernel

### v0 design

- One kernel, one thread per env, 256 threads per block, grid-stride loop so a single launch handles
  any N. Signature takes raw pointers to the same buffers the PyTorch variant uses.
- `float4` load and store for state (one 16-byte transaction), `float2` load for action,
  `const __restrict__` on every read-only pointer so the compiler can use the read-only path.
- Counters packed into one `uint32`: step count in the low 16 bits, reset count in the high 16.
- The 108-byte face/corner mask table in `__constant__` memory. All threads in a warp read
  different entries, but the table is tiny and stays in the constant cache; there is no HBM traffic
  for it at any batch size.
- RNG: `curand_Philox4x32_10(make_uint4(env, reset_counter, stream, 0), make_uint2(seed, copy))`
  called directly. One call yields the four uniforms a reset needs. Compute it unconditionally in
  v0 (branch-free, about 40 integer operations) and measure a `if (done)`-guarded version later —
  with staggered episode phases the branch is divergent, with in-phase episodes it is free.
- Reset is a `selp` select, not a branch.
- No allocation, no synchronization, no printf in the kernel.

### Integration with PyTorch

- Build with `torch.utils.cpp_extension.load_inline` or a small `setup.py`, `nvcc -arch=sm_90 -O3`.
  Launch on `c10::cuda::getCurrentCUDAStream()` so it interleaves correctly with torch's work and
  can be captured into torch's CUDA graph.
- Register it as a proper custom operator (`torch.library.custom_op` plus a fake/meta
  implementation) so `torch.compile(fullgraph=True)` does not break the graph around it.
- `--use_fast_math` is safe for the integrator, which is only multiplies and adds, but check it
  against the fixtures rather than assuming; avoid it if the corner projection's `rsqrt` moves any
  result past the 1e-4 float32 tolerance.

### Ordered optimization steps

1. **Correctness against the fixtures**, in `double` first if that makes debugging easier, then
   `float`. Then a bit-for-bit comparison of the reset draws against the PyTorch and JAX
   implementations of the same Philox.
2. **`-Xptxas -v` register report.** Full occupancy at 2048 threads per SM needs 32 registers or
   fewer per thread. This kernel should land around 24-32. If it spills, the corner test is the
   likely cause and can be restructured.
3. **`ncu` on the real batch.** The one metric that matters is
   `gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed`. Above 80% means done; below 60% means
   look at `l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum` for uncoalesced access and at the
   memory-workload-analysis section for wasted sectors.
4. **Two or four envs per thread**, unrolled. More outstanding loads per thread means more
   memory-level parallelism, which is usually what closes the gap from 60% to 85% of peak on a
   kernel this light. Measure 1, 2, and 4.
5. **Block size sweep** at 128 / 256 / 512, and grid sized to an exact multiple of 132 SMs for the
   persistent variant.
6. **Guarded versus unconditional reset**, measured under both in-phase and staggered episodes.
7. **Multi-step kernel** that loops S internal steps with state held in registers, for the env-only
   benchmark and for any use where the action is analytic. Label its numbers clearly — see the
   measurement traps; this is not a number a training loop can reach.
8. **Persistent per-copy kernel** for Module 3: one block per copy, policy weights in shared memory,
   T steps in the loop, no HBM round trip for state or observations between steps. This is the
   highest ceiling in the whole project and also the most work; do not start it before levels 1 and
   2 are measured and the policy has been shown to dominate the loop.

### Expected bottlenecks

- Small N: launch overhead, 3-7 microseconds, unavoidable without graphs. Even a perfect kernel is
  flat below N of about 3 x 10^5.
- Large N: HBM bandwidth, and this variant should be the one that gets closest to it. If it does
  not reach 70-85% of peak at N = 10^7, the cause is almost always insufficient memory-level
  parallelism (fix with step 4) or an unvectorized load (fix the layout).
- Warp divergence from the wall clamps is minor because the clamps are selects, not branches. The
  corner test is the only place a real branch is tempting.

## 6. Variant 3 — JAX

### v0 design

- A pure function `step(state, action) -> (state, obs, reward, done)` where `state` is a NamedTuple
  of `pos [C,E,2]`, `vel [C,E,2]`, `step_count [C,E] int32`, `reset_count [C,E] int32`. Write it for
  a single env and apply `jax.vmap` twice (over envs, then over copies), or write it batched
  directly — XLA generates the same thing, and the single-env-plus-vmap form is much easier to
  check against the fixtures.
- The maze mask is a module-level `jnp` constant so XLA inlines it as a literal.
- Auto-reset with `jnp.where`. Do not use `lax.cond` per env: inside a `vmap` it becomes a select
  that evaluates both branches anyway, and it obscures the intent.
- The rollout is `jax.lax.scan` over T steps, and the whole thing goes under one `jax.jit`. This is
  the structural decision that matters most in this variant; everything else is tuning.
- `donate_argnums` on the carried state at the `jit` boundary so XLA updates buffers in place
  instead of copying the batch every call.

### Ordered optimization steps

1. **Correctness under `jax_enable_x64=True`** against the fp64 fixtures, then switch x64 off — it
   is a global flag and it is slow, so it belongs only in the correctness test.
2. **`lax.scan` over T** versus a Python loop of jitted steps. The scan should win by the full
   per-step dispatch cost, and it is the precondition for everything after.
3. **`unroll=` on the scan**, at 1 / 4 / 8 / 16. Unrolling lets XLA keep the carry in registers
   across several steps instead of round-tripping it, which directly attacks the 32 bytes of state
   traffic per step.
4. **Buffer donation**, verified. JAX warns "some donated buffers were not usable" when the donation
   silently did not apply, and that warning is easy to miss in a log. Treat it as an error.
5. **Read the HLO.** `jax.jit(f).lower(args).compile().as_text()` and count the fusions. Look
   specifically for `copy`, `transpose`, and `bitcast-convert` operations that should not exist —
   each one is a full pass over the state.
6. **Layout experiment: packed `[C,E,4]` versus four separate `[C,E]` planes.** This is the one
   place where the right answer probably differs from the CUDA variant. XLA's vectorizer often does
   better with separate planes, while CUDA wants the packed `float4`. Measure, do not assume.
7. **int32 counters**, and `jax.lax.stop_gradient` nowhere needed here since the env is not
   differentiated — but confirm the env is outside any `grad` in Module 3, since XLA will otherwise
   keep intermediates alive.
8. **`jax.experimental.pallas`** if XLA leaves more than about 25% of bandwidth on the table.
   Pallas on GPU lowers to Triton and gives explicit block-level control, which is the bridge
   between this variant and the hand-written CUDA one.
9. **Whole-training-loop jit** for Module 3: scan the rollout, the PPO update, and the iteration
   loop under one `jit`, the Anakin pattern. At that point the host launches one executable per
   training run segment and Python is out of the loop entirely.

### Expected bottlenecks

- JAX's structural advantage is that most synchronization mistakes are compile errors: you cannot
  branch on a traced value, cannot index with a data-dependent mask, cannot call `.item()`. Most of
  the PyTorch checklist is enforced by the language here.
- The costs that remain are the scan carry round-tripping through HBM (fix with `unroll`), donation
  not applying (fix explicitly), and the scan's stacked outputs — a scan that returns per-step
  observations writes a full `[T,C,E,4]` buffer, which is real traffic. That buffer is the rollout
  buffer you needed anyway, so count it as training-shaped traffic and not as overhead.
- XLA preallocates about 75% of GPU memory by default (`XLA_PYTHON_CLIENT_PREALLOCATE`). That is
  fine under the single-user GPU lock and would be a problem without it.

## 7. Benchmarking methodology and the traps

### Protocol

- One measurement produces one JSON in `benchmarks/results/`, recording: git hash, GPU name, driver
  and CUDA versions, framework versions, n_copies, n_envs, T, dtype, layout variant, kernel count,
  wall-clock, env-steps/s, achieved bytes/s, and percent of the 3.9 TB/s peak.
- Every GPU-touching run goes through the `locks/gpu_run.sh` lock. A second process on the card
  invalidates the number silently, and shared-GPU contamination is invisible in the result.
- Warm up at least 10 iterations and discard them: JIT compilation, Inductor autotuning, CUDA-graph
  capture, and the GPU clock ramp all live there.
- Measure a region long enough to reach steady-state clocks — at least one second of GPU work —
  and report the median of five or more repetitions plus the spread. H100 clocks move under
  sustained load; a single fast run is not a result.
- Report both **env-steps per second** and **nanoseconds per env-step**, and plot against total envs
  on a log x-axis, marking the launch-bound plateau, the L2 knee, and the HBM asymptote.
- Re-run the fixture correctness check after every accepted optimization. The autoresearch loop is
  one change, one measurement, keep or revert, one log line — and a speedup that broke the physics
  is not a speedup.

### Timing correctly

- **PyTorch**: `torch.cuda.Event` pairs around the region, with one `torch.cuda.synchronize()` at
  the end, or wall-clock around N repetitions with a single sync after them. Never wall-clock a
  single kernel launch — you will be timing the launch, not the work.
- **JAX**: `block_until_ready()` on every returned array (use `jax.block_until_ready(pytree)` for a
  tree), outside the timed loop's last iteration. Without it you are timing dispatch.
- **CUDA**: `cudaEvent` pairs on the same stream, or `ncu` for per-kernel numbers.
- **Never set `CUDA_LAUNCH_BLOCKING=1` while measuring.** It serializes everything and turns a
  throughput measurement into a latency measurement.

### The specific traps

1. **Asynchronous timing.** The default failure mode in both frameworks is to measure Python
   dispatch and report it as GPU throughput, off by one to three orders of magnitude in the
   flattering direction.
2. **Missing warmup.** The first `torch.compile` call can take tens of seconds; the first `jit`
   likewise; the first CUDA-graph replay after capture is not representative.
3. **`torch.compile` recompiles.** Changing batch size, dtype, or `requires_grad` triggers a silent
   recompilation that can dominate a short benchmark. Set
   `torch._dynamo.config.error_on_recompile = True` during benchmarking and run with
   `TORCH_LOGS=recompiles` when it fires.
4. **CUDA-graph output aliasing.** A replay overwrites the same buffers. A benchmark loop that keeps
   the outputs will silently keep only the last step's data, and a correctness check that passes on
   step 1 will pass on every step for the wrong reason.
5. **JAX donation.** Using a donated buffer after the call raises; a donation that did not apply
   only warns. Both are easy to miss, and the second one costs a full state copy per step.
6. **L2 residency read as bandwidth.** Below about 900,000 envs the state fits in the 50 MB L2 and
   the measured "bandwidth" exceeds HBM peak. Report the regime alongside the number.
7. **The in-kernel multi-step benchmark.** A kernel that loops S steps internally holds state in
   registers and skips almost all traffic. Its steps/s figure can be several times the
   training-shaped roofline and is not reachable by any training loop, because the policy has to run
   between steps. Report it separately and label it.
8. **Comparing variants at different precisions.** TF32 in the policy matmuls, `--use_fast_math` in
   nvcc, and JAX's default matmul precision all differ by default. Fix them explicitly across the
   three variants before comparing.
9. **Reading the CPU-bound plateau as a GPU limit.** A flat line as N grows means the CPU is
   issuing work, not that the GPU is full. Check with `nsys`: gaps between kernels on the timeline
   are launch overhead, not compute.
10. **Profiler distortion.** `torch.profiler` with `with_stack=True` or `record_shapes=True` adds
    per-op overhead comparable to the ops themselves at this scale. Profile to find the shape of the
    problem, then measure without the profiler attached.

## 8. What the reference systems contribute

| system | the idea worth copying here |
|---|---|
| gymnax / PureJaxRL | whole rollout and update under one `jit`; env written as a pure function; thousands of envs at a time |
| DeepMind Anakin | environment and learner both on the accelerator, entire training step compiled — the ceiling this project is aiming at |
| DeepMind Sebulba | actor/learner split for envs that cannot be compiled; not applicable here, and knowing that is the point |
| Brax / MJX | batched physics as composable array primitives; also the warning that their per-env state is large and ours is not, so their batch-size limits do not apply |
| Isaac Gym tensor API | physics buffers exposed directly as GPU tensors, no host copy anywhere in the step |
| Madrona / GPUDrive | thousands of independent worlds on one GPU; aggregate batch throughput preferred over per-world latency; amortize synchronization across the batch |
| PufferLib | native-code environments reaching very high step rates; the reminder that "steps/s" figures are only comparable when the measurement shape is stated |
| EnvPool | asynchronous stepping to hide per-env latency — irrelevant here, since every env is a handful of arithmetic operations and there is no latency to hide |
| CUDA graphs in torch | replay a captured launch sequence with one call; the fix for the launch-bound plateau below 10^6 envs |

## 9. Sources

- [NVIDIA H100 NVL product brief](https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/h100/PB-11773-001_v01.pdf) — 94 GB HBM3, 3,938 GB/s, 132 SMs
- [PyTorch CUDA Graphs](https://pytorch.org/blog/accelerating-pytorch-with-cuda-graphs/) and [NVIDIA's torch CUDA-graph checklist](https://docs.nvidia.com/dl-cuda-graph/torch-cuda-graph/quick-checklist.html) — capture requirements, static shapes and addresses, no in-capture synchronization
- [torch.compile reduce-overhead regression report](https://github.com/pytorch/pytorch/issues/174575) — reason to prefer manual capture
- [Benchmarking JAX code](https://docs.jax.dev/en/latest/benchmarking.html) and [JAX buffer donation](https://docs.jax.dev/en/latest/buffer_donation.html) — `block_until_ready`, `donate_argnums` semantics and the unusable-donation warning
- [Podracer architectures for scalable RL](https://arxiv.org/pdf/2104.06272) — Anakin and Sebulba
- [PureJaxRL](https://chrislu.page/blog/meta-disco/) — end-to-end compiled RL on accelerators
- [MuJoCo XLA (MJX) documentation](https://mujoco.readthedocs.io/en/stable/mjx.html) — batched physics as XLA graphs, `vmap`/`scan` usage
- [Madrona Engine](https://madrona-engine.github.io/) and [GPUDrive](https://arxiv.org/html/2408.01584v1) — batch many-world simulation on GPU
- [Isaac Gym](https://ar5iv.labs.arxiv.org/html/2108.10470) and its [tensor API docs](https://docs.robotsfan.com/isaacgym/programming/tensors.html) — no-host-copy stepping
- [PufferLib performance notes](https://pufferai.github.io/build/html/rst/blog.html) — reported step rates and their measurement shape
- [How PyTorch generates random numbers in parallel on the GPU](https://blog.codingconfessions.com/p/how-pytorch-generates-random-numbers) — Philox subsequence/offset split
- [cuRAND programming guide](https://docs.nvidia.com/cuda/archive/9.0/pdf/CURAND_Library.pdf) — Philox-4x32-10, constant-time skip-ahead, `curand_init` cost
- [CUDA graph replay and launch overhead measurements](https://inferacthq.com/blog/cuda-graph-replay-kernel-launch-overhead) — null-kernel launch 3-7 us, graph replay materially lower
