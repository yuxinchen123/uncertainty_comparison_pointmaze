# Module 1 round-2 experiment list (torch env, CUDA env, jax env)

Written 2026-08-15 after reading the round-one ledgers, the benchmark result JSONs, the unified
report, `extra_step_review.md`, and the three implementations line by line. Every row below is a
candidate that round one did NOT measure. Rows that round one already measured and rejected are
listed at the end with their numbers so they are not proposed again.

## Where each implementation stands (measured, from `benchmarks/results/`)

| implementation | small-batch floor | 1e5 envs | 1e6 envs | 3e6 envs | peak |
|---|---|---|---|---|---|
| CUDA fused kernel | 5.55 us (32-3e4 envs) | 9.06 us | 59.6 us | 171.7 us | 1.75e10 |
| torch compiled | 152-171 us (32-3e5) | 164 us | 257 us | 720 us | 4.17e9 |
| jax jit per step | 73-76 us (3e2-3e5) | 75.1 us | 145.5 us | 506.6 us | 6.87e9 |
| jax scan (labeled upper bound) | 12.3-15.1 us | 19.2 us | 119.0 us | 414.7 us | 9.92e9 |

Two readings from those numbers drive the whole list:

1. Below roughly 1e5 envs every implementation sits on a flat floor, so the floor is per-call host
   and launch cost, not work. The floors differ by 28x (5.55 us CUDA vs 158 us torch) and nothing
   about the physics explains that difference.
2. At 3e6 envs the CUDA kernel moves about 94 bytes per env-step, which is 1.64 TB/s against a
   card that delivers roughly 3.3 TB/s in practice — under half. Ledger row 4 (E2) already
   confirmed this by cutting 16 of those 94 bytes and measuring exactly zero change. The large-batch
   limit is instruction throughput and dependency latency, not bandwidth.

## Gates and measurement rules that every row must satisfy

- **Exactness gates, unchanged**: `common/code/check_against_fixtures.py --impl {torch,cuda,jax}`
  in float64 (one-step error <= 4.4e-16) and float32 (<= 4.6e-7); `tests/test_cross_impl_rng.py`
  (resets bit-identical across the three implementations, including generation-1 respawns);
  `cuda_env/test_ab_vs_torch.py` (500 steps, 320 auto-resets, within 1e-6 of TorchPointMaze).
- **The gate must run through the code path that was timed** (`extra_step_review.md` item 16). A
  compiled, captured, K-step, or Triton path is a different program from the eager one, so the
  checker needs a `--path` switch, and the switch needs the adversarial check: perturb one constant
  by 1% and confirm the checker reports a failure.
- **Paired comparison, not two absolute numbers.** At least 5 ABBA pairs in one process, decision
  statistic `mean(log ratio)` with its standard error, keep only if `mean - 3*SE > 0` and the gain
  is at least 1%. Run the null first (candidate = byte-identical copy of the reference) to get the
  noise floor; without it a +5% row cannot be told from drift.
- **Record enqueue time beside wall time** in every bench row (host time from region start to just
  before the closing synchronization). Enqueue near wall means the row is launch-bound and fusion
  or capture will pay; enqueue far below wall means only fewer or cheaper instructions will.
- **One experiment is one committed edit**, benchmarked from a frozen copy extracted from git, with
  the commit hash written into the result JSON.

## Diagnostics to run before the ranked rows (no speed effect on their own)

### D1. `ncu` and `-Xptxas -v` on the CUDA kernel at 1e5 / 1e6 / 3e6 envs

- **Change** — no code change; collect `gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed`,
  `smsp__inst_executed.sum`, `smsp__warps_active.avg.pct_of_peak_sustained_active`, stall reasons,
  and the per-thread register count.
- **Why** — the CUDA ledger's row 5 justifies E3 by "the compute/latency limit exposed by E4", but
  there is no E4 row in the ledger, so that limit is currently an assertion, not a measurement. The
  three candidate limits (instruction issue, dependent-select latency, occupancy from register
  pressure) call for different fixes, and the ranking below depends on which one it is.
- **Expected** — zero speed effect. It decides whether CUDA row 1 or row 4 is the bigger lever.
- **Measure** — one profiled run per batch size under the H100 lock; store the counters in the
  result JSON beside the timing rows.
- **Risk** — none.

### D2. Kernel inventory of the compiled torch step

- **Change** — no code change; `TORCH_LOGS=output_code` plus a profiler pass at 1e6 envs to list
  the 8 kernels by name, device time, and bytes moved, and the host time per `step` call at N=32.
- **Why** — round one recorded the split as "3 kernels of 58/57/46 us + integration 37 + RNG 15 +
  cat 10+10" but not what forces each split, and the 158 us floor at N=32 is 130+ us of host time
  that no kernel explains. Torch rows 2, 3, and 6 below each attack a different one of those, and
  the inventory says which is worth doing first.
- **Expected** — zero speed effect.
- **Measure** — one profiled run; save the kernel list next to the JSON.
- **Risk** — none.

### D3. HLO read of the jax step and scan

- **Change** — no code change; `jax.jit(f).lower(args).compile().as_text()`, count fusions and look
  for `copy`, `transpose`, `bitcast-convert`, and the two `concatenate` operations the step
  performs per call.
- **Why** — the jax ledger has one baseline row and no structural information at all. The step
  concatenates `[pos, vel]` twice per call (obs and final_obs), which is two full passes over the
  state if XLA does not fuse them away; the HLO says whether jax rows 2 and 3 below have anything
  to remove.
- **Expected** — zero speed effect.
- **Measure** — save the fusion count and the operation histogram in the jax ledger.
- **Risk** — none.

---

## CUDA env — ranked

### 1. Launch on torch's current stream instead of the default stream

- **Change** — `pointmaze_kernel.cu` launches all three kernels as `<<<blocks, threads>>>` with no
  stream argument, so they go to the legacy default stream. Add
  `#include <c10/cuda/CUDAStream.h>` and pass `at::cuda::getCurrentCUDAStream()` as the fourth
  launch argument in `env_step`, `env_reset`, and `env_dynamics`.
- **Why** — `torch_ppo_rnd.py` calls `_rollout_body_cuda()` inside `torch.cuda.graph(g)` for the
  `env_backend="cuda"` pairing, and capture runs on a side stream. A kernel issued to the legacy
  default stream during capture is either an error or is not captured at all, which would mean the
  measured pairing row (35.6 ms at C=128, reported in `e2e/torch_e2e` and in the unified report)
  does not contain the env step. This is a correctness question about a published number, and it
  also blocks row 2 below. The same fix makes the env kernel obey any stream a caller sets.
- **Expected** — no throughput change by itself. It either confirms the pairing row or invalidates
  it.
- **Measure** — after the fix, replay the captured iteration twice and assert `env.state` advances
  by the same amount as two eager iterations; assert the replay draws different action noise (a
  frozen state under capture looks fast and passes a mean test). Re-run
  `bench_train.py --env-backend cuda` and compare against the recorded 25.1 / 35.6 ms rows.
- **Risk** — none to the exactness gates: the arithmetic is untouched. The risk is that the
  previously reported pairing numbers move.

### 2. Winner-only contact law with a rank-by-counting tournament (the ledger's E3, in a safer form)

- **Change** — in `pointmaze_dynamics.h`, keep the 8-candidate unrolled loop but compute only
  `dist_k` in it (rectangle, nearest point, `dn`, wall select). Replace the running two-slot
  tournament with a count: `rank_k = sum_j [ dist_j < dist_k or (dist_j == dist_k and j < k) ]`,
  take `k1 = argrank 0` and `k2 = argrank 1` by 8-way integer selects, then recompute the
  rectangle, the normal `(sx, sy)`, `imp`, `b_qp`, and `Rreg` for those two candidates only. The
  2-contact QP that follows is unchanged.
- **Why** — the ledger's E3 was recorded as "roughly 8x less in-loop arithmetic", but the stronger
  argument is the dependency chain. Today each candidate updates ten carried registers through
  `take1`/`take2` selects that depend on the previous candidate's comparison, so every thread walks
  an ~80-step serial chain, and it carries 10 live registers across the loop. The counting form
  makes all 8 distances independent and reduces the carry to two integers, which cuts both the
  latency chain and the register pressure that limits occupancy. The reference campaign measured
  1.3834x for removing exactly one select from exactly this kind of chain, and stated the win was
  latency rather than arithmetic. Ledger row 4 already ruled out bandwidth as the limit here, so
  this is the right class of change for the 1e5-3e6 region.
- **Expected** — 1.4-2.2x at 1e5 envs and above if D1 shows instruction or latency limits; nothing
  at or below 3e4 envs, where the 5.55 us launch floor dominates and no arithmetic change is
  visible. Peak would move from 1.75e10 toward 2.5-3.5e10 env-steps/s.
- **Measure** — `bench_env_step_cuda.py --n-envs 100 300 1000 3000 10000 30000 100000 300000
  1000000 3000000 --tag _winnerlaw`, paired ABBA against the current build in one process; plus the
  D1 counters again to confirm the mechanism (instructions per env-step down, warp stall reason
  changed).
- **Risk** — the tie rule is the whole exactness question. The current sequential tournament is
  exactly "stable sort by `dist`, take the first two", so the counting form must break ties by
  candidate index in the same direction, which the `(dist_j == dist_k and j < k)` term does. Rank
  on `dist` (that is, keep the 8 square roots and the `- K_R` shift) so the ordering compares the
  same quantity the current code compares. The winners' recomputed expressions must keep the
  current grouping character for character, since `--fmad=false` bit-equality with torch eager is
  the gate. Gates: fixture checker both dtypes, A/B vs torch within 1e-6, cross-impl RNG. Add one
  new fixture with the ball placed at an exactly symmetric position between two wall boxes so the
  tie path is actually exercised.

### 3. CUDA-graph capture of the per-step kernel for the small-batch curve

- **Change** — add a benchmark and an API path that captures K repeated `env_step` launches into
  one `torch.cuda.CUDAGraph` over the preallocated buffers and replays it, with the action buffer
  as a static input written before each replay.
- **Why** — the floor is flat at 5.55-6.06 us from 32 to 30,000 envs, which is pybind dispatch plus
  one kernel launch; the kernel itself at N=32 is a couple of microseconds. Graph replay removes
  the host half. This is the honest fix for three decades of the deliverable curve, and unlike row
  5 it is a path a trainer can actually use, since the trainer already captures.
- **Expected** — floor 5.55 us to roughly 2.5-3.5 us, so 1.6-2.2x for every batch at or below 3e4
  envs, and nothing above 3e5.
- **Measure** — new mode in `bench_env_step_cuda.py`, tag `_graph`, at N = 32 to 1e5; report the
  captured and uncaptured curves as two labeled lines, since the report already labels the jax scan
  line the same way. Record the enqueue column, which should collapse under capture.
- **Risk** — none to the arithmetic, so the fixture gates carry over. Depends on row 1. The
  practical trap is the one the design brief names: a replay writes into the same output buffers
  every time, so anything the benchmark keeps from a replay must be copied out, and the step-count
  and reset-count state must be verified to advance exactly K times per replay.

### 4. Two and four envs per thread, then re-sweep the block size

- **Change** — template `step_kernel` on `ENVS_PER_THREAD` in {1, 2, 4} with the inner work
  unrolled, and re-run the block sweep {128, 256} afterwards.
- **Why** — the design brief's step 4 for this variant: more independent loads in flight per thread
  is the usual way a light streaming kernel closes the gap from about half of peak bandwidth to
  80%. The v0 block sweep (ledger row 2) was measured against the long-chain kernel; after row 2
  the register footprint and the latency profile change, so the sweep's answer can change with it.
- **Expected** — 0-15% at 1e6 envs and above; possibly negative at small batch, where the grid gets
  shorter than the machine. Order it after row 2 because row 2 changes the register count that
  decides this.
- **Measure** — the same grid, tags `_ept2` / `_ept4`, paired against the row-2 build; read
  `-Xptxas -v` registers per thread and the achieved bandwidth from D1.
- **Risk** — none to exactness (per-env math unchanged), provided the tail handling for a batch
  that is not a multiple of `ENVS_PER_THREAD * block` is exercised by a benchmark size that is not
  a round number (for example 3,000,001).

### 5. K-step kernel with the state held in registers (labeled upper bound)

- **Change** — a second entry point taking a horizon K and an action buffer `[K, B, 2]`, running
  the K-step loop inside the kernel with position, velocity, goal, and both counters in registers,
  writing only per-step rewards and end flags. Keep the per-step kernel as the correctness and
  production path.
- **Why** — this is the reference campaign's single largest number (20.57x for moving the time loop
  into the kernel) and our CUDA env is exactly at their version 0. Here it removes K-1 launches and
  K-1 round trips of the 32-byte state, which is the dominant cost below 1e5 envs and a real share
  above it.
- **Expected** — 3-5x below 3e4 envs (the 5.55 us floor is paid once per K steps instead of every
  step) and 1.2-1.6x at 1e6-3e6 envs. Not 20x: our per-step kernel is already fused and our launch
  overhead is a smaller share than theirs.
- **Measure** — a new bench mode reporting a separate curve, labeled in the report as an upper
  bound a trainer cannot reach, exactly as the jax scan line is labeled. Sweep K in {8, 32, 128}
  and the rows-per-program constant so the grid is a few programs per multiprocessor.
- **Risk** — the arithmetic per step is unchanged, so the fixture gate carries over, but the gate
  must be re-run THROUGH the K-step entry point, and the A/B against `TorchPointMaze` must drive it
  with the same action sequence for the full K. The auto-reset path inside the loop is the part
  most likely to differ (the reset generation counter now lives in a register across steps), so the
  A/B must cover at least 400 steps so every env truncates and respawns at least once.

### 6. Measure what `--fmad=false` costs, as a labeled build

- **Change** — build a second extension with fused multiply-add contraction enabled, run the whole
  grid, and record the number. Do not change the default.
- **Why** — `--fmad=false` was chosen to make float32 results bit-identical to torch eager. It
  roughly doubles the instruction count of the multiply-add chains, which under D1's likely finding
  (instruction-throughput limited) is a direct throughput cost. We currently do not know what
  exactness costs, and that number belongs in the report.
- **Expected** — 10-30% at 1e6 envs and above.
- **Measure** — the same grid, tag `_fmad`, paired; report the fixture error at both dtypes for the
  fast build alongside the timing.
- **Risk** — this deliberately breaks the bit-equality gate: the float32 A/B against torch would go
  from bit-equal to about 1e-7, and the float64 one-step error would move off 4.4e-16. It is
  therefore a labeled measurement, never the default build, and the report must state which build
  each number came from.

### 7. Rank on squared distance (follow-up to row 2 only)

- **Change** — after row 2 lands, rank the 8 candidates on `dx*dx + dy*dy` and compute the square
  root only for the two winners.
- **Why** — removes 6 of 8 square roots per env-step. Ordering by squared distance is equivalent to
  ordering by distance for non-negative values, so the pair is the same in every case where the
  distances are distinct.
- **Expected** — a further 5-15% at 1e5 envs and above; nothing at small batch.
- **Measure** — same grid, tag `_dsqrank`, paired against the row-2 build.
- **Risk** — this is the one row that can change results in a way the current fixtures would not
  catch. If two candidates have different squared distances that round to the same square root, the
  squared-distance order and the distance order disagree about which candidate is slot 1, and the
  QP case enumeration is not symmetric in the two slots. Keep it separate from row 2 so the two
  effects can be told apart, and gate it with the symmetric-tie fixture added in row 2.

---

## torch env — ranked

### 1. Contact pipeline restructured to rank-by-counting with winner-only law

- **Change** — the same restructure as CUDA row 2, written in torch: compute the 8 distances,
  build `rank_k` as a sum of 8 comparisons, select the two winners with masked sums
  (`sum_k (rank_k == 0) * value_k`), and evaluate the normal and the contact law twice instead of
  eight times.
- **Why** — the contact pipeline is 161 of the 248 us measured at 1e6 envs (three kernels of 58,
  57, and 46 us), and it is the part with the ten `torch.where` chains per candidate. The chain is
  40 dependent selects per element; the counting form has none. It should also reduce the number of
  live intermediates, which is what forces inductor to split a fused region, so it is the change
  most likely to turn three contact kernels into one or two without touching the realize
  thresholds that blew up compile time in round one (ledger row 5).
- **Expected** — 1.2-1.5x at 1e6-3e6 envs, and a smaller improvement at the floor if the kernel
  count drops. The reference measured 1.3834x for the equivalent chain removal.
- **Measure** — `bench_env_step.py --impl torch --mode compile --n-envs 32 ... 3000000 --tag
  _rankcount`, paired ABBA; plus D2's kernel inventory again, since the kernel count is half the
  hypothesis.
- **Risk** — same tie question as CUDA row 2: the counting rank must break ties by candidate index
  in the same direction as the sequential tournament. Gates: fixture checker both dtypes,
  `tests/test_torch_env.py`, and the A/B against the CUDA kernel (which is the independent
  implementation of the same math and would catch a tie-order change that the fixtures miss).

### 2. Packed `[C, N, 4]` state so the observation is the state

- **Change** — carry one state tensor of shape `[C, N, 4]` with `pos` and `vel` as views, as the
  CUDA env already does after its E2 row. `step_core` then returns the state slice instead of
  building `obs` and `final_obs` with `torch.cat`.
- **Why** — the two `cat` calls are 10 us each at 1e6 envs (20 of 248 us) plus two full writes of a
  `[C, N, 4]` tensor, and at small batch they are two more kernels of dispatch on a 158 us floor.
  The CUDA env measured this change as speed-neutral because that kernel is not bandwidth-bound,
  but the torch path pays it as two extra kernels rather than as bytes, which is a different
  regime; the reference's rule is that a layout decision does not port across implementations
  unmeasured.
- **Expected** — 8% at 1e6 envs, and 10-25 us off the 158 us floor. It also deletes code, which
  keeps it under the simplicity criterion even if it measures flat.
- **Measure** — full grid, tag `_packed`, paired. Report the floor and the 1e6 point separately.
- **Risk** — pure layout, so no arithmetic changes and the fixture numbers must be unchanged to the
  last bit. The contract change touches `torch_ppo_rnd.py` (which stores `_S_pos`, `_S_vel`,
  `_S_obs` as separate static capture buffers) and the fixture checker's `step_batch` adapter, so
  the trainer's bitwise capture test is part of this row's gate, not an afterthought.

### 3. CUDA-graph capture of the compiled step for the small-batch curve

- **Change** — a benchmark mode that captures K replays of the compiled `step` over static buffers,
  mirroring what the trainer already does for the whole rollout.
- **Why** — the floor is 152-171 us flat over four decades and the device time at N=32 is on the
  order of 15-25 us (8 kernels of a couple of microseconds each). So roughly 85% of the floor is
  host work: dynamo guard evaluation, the python wrapper, and 8 launches. Capture removes all of
  it, and the trainer's rollout capture is direct evidence that this works on this code.
- **Expected** — 158 us to 15-25 us at every batch at or below 1e5 envs, so 6-10x on the whole flat
  part of the curve; nothing at 3e6.
- **Measure** — new mode `compile-graph` in `bench_env_step.py`, tag `_graph`, N = 32 to 1e6; two
  labeled lines in the report (uncaptured and captured), with the enqueue column showing the host
  time collapsing.
- **Risk** — none to the arithmetic. Two practical traps: the replay writes into the same output
  buffers, so a benchmark that keeps outputs must copy them out; and the timed region must not
  include a compile, so assert the dynamo compile-counter delta is zero across the timed block.

### 4. K-step compiled chunk (labeled upper bound)

- **Change** — a compiled function taking `[K, C, N, 2]` actions and running K steps in a python
  loop that unrolls at trace time, returning stacked rewards and end flags.
- **Why** — the reference campaign's largest environment-track keep was exactly this (3.84x for
  handing 25 steps at a time to one compiled function). It attacks the same host floor as row 3 but
  additionally lets inductor fuse across the step boundary, which capture cannot do.
- **Expected** — 4-6x on the flat part of the curve at K = 8; possibly better than row 3 at 1e6 if
  cross-step fusion removes the RNG or the tail kernels. Compile time grows with K, so sweep K in
  {4, 8, 16} and stop where compile time exceeds a minute.
- **Measure** — new mode, tag `_chunk8`, same grid, with compile time reported separately from
  steady state.
- **Risk** — exactness is unchanged per step, but the gate must run through the chunked path, and
  the chunk must be driven with the same action sequence as the per-step path in the A/B. Round
  one's row 5 is the warning: a change that enlarges the compiled region can explode compile time
  (it ran over 40 minutes without producing a kernel). Kill any configuration that exceeds a
  10-minute compile.

### 5. Hand-written Triton kernel for the whole step

- **Change** — one Triton kernel covering dynamics, reward, ends, and auto-reset, called through
  `torch.library` so it composes with capture, replacing the compiled 8-kernel region.
- **Why** — this is the ceiling of the torch path and the only way it reaches the CUDA curve. At
  1e6 envs the compiled path takes 257 us against the CUDA kernel's 59.6 us for identical math, and
  the whole difference is kernel count and intermediate traffic. It also collapses the floor to one
  launch.
- **Expected** — 257 us to 70-110 us at 1e6 (2.3-3.6x), floor to about 10-20 us. Rank it below rows
  1-4 because it is days of work against their hours, and because row 1 may recover part of the gap
  for free.
- **Measure** — same grid, tag `_triton`; compare against the CUDA kernel curve directly, since
  that is the target.
- **Risk** — the highest of any row here. Triton's float32 division, square root, and multiply-add
  contraction do not match torch eager's by default, so bit-equality with the torch reference is
  unlikely; the gate would have to move to a documented tolerance (the float64 fixture at 4.4e-16
  is the strict part to keep, the float32 A/B may need 1e-6). Decide the tolerance before writing
  the kernel, not after seeing the numbers.

### 6. Delete the constant outputs and the int64 reset counter

- **Change** — under `continuing_task=True`, `terminated` is identically false, so build it as a
  constant instead of a tensor written every step; `done` is then `truncated`. Separately, make
  `reset_count` int32 and do the reset-noise mixing in 32-bit arithmetic, as the CUDA and jax
  implementations already do.
- **Why** — the RNG block is 15 us of the 248 us at 1e6 envs and it currently runs int64 multiplies
  (emulated on GPU, roughly 4x the cost of int32) and moves 8 bytes per counter instead of 4. The
  `terminated` write is one more kernel and one more allocation per step at every batch size. Both
  are deletions, which the simplicity criterion rewards even at flat speed.
- **Expected** — 2-5% at 1e6 envs; a kernel fewer at the floor.
- **Measure** — full grid, tag `_int32rng`, paired.
- **Risk** — the reset RNG is FROZEN and bit-identity across the three implementations is a gate,
  so the 32-bit form must reproduce the current draws exactly. Torch's int32 multiply wraps in two's
  complement like C, but the python literals are larger than int32, so the dtype promotion must be
  pinned explicitly. `tests/test_cross_impl_rng.py` is the whole gate and must pass including the
  generation-1 respawns. The `terminated` deletion touches the trainer's `boot_mask`, which
  `extra_step_review.md` item 4 already identified as dead work under this configuration.

### 7. Neighbor mask by bit arithmetic instead of a table lookup

- **Change** — replace `self.nb_mask[cell]` with bit extraction from the wall map held as two int64
  constants: neighbor k of cell `c` is bit `c + di*cols + dj` of the map.
- **Why** — the review's item 9 is that the cache-versus-recompute decision has opposite answers in
  eager torch and under XLA, and must be measured in each. Our code currently recomputes the
  rectangles but gathers the mask, which is the untested middle. The gather is also a plausible
  reason inductor splits the contact region into three kernels.
- **Expected** — small, 0-8%, and it could go either way: the table is 432 bytes and stays in cache,
  so the arithmetic replaces a cheap load with about 24 integer operations. Run it, and run the
  opposite direction too (a precomputed `[n_cells, 8, 4]` rectangle table, gathered) so both sides
  of the review's item 9 are on record for this workload.
- **Measure** — two tags, `_bitmask` and `_recttable`, same grid, paired; run both in torch and in
  jax and record the two verdicts separately.
- **Risk** — none to exactness if the extraction reproduces `build_geometry`'s mask exactly; add a
  unit test comparing the bit-extracted mask against `nb_mask` for every cell of all four maps.

---

## jax env — ranked

The ledger has one baseline row, so this list starts from structure rather than from tuning.

### 1. Buffer donation on the jitted step, verified

- **Change** — `donate_argnums` on the carried `EnvState` at the jit boundary, and treat the
  "some donated buffers were not usable" warning as an error rather than a log line.
- **Why** — without donation every jitted step allocates a fresh `pos`, `vel`, `goal`, and two
  counter arrays. At 1e6 envs that is 40 MB of allocation and copy per step against a 145 us step;
  the allocator is fast but the copies are real traffic. It is also the precondition for row 2:
  a layout change is worth much less if the state is being copied anyway.
- **Expected** — 0-10% at 1e5 envs and above, near zero at the floor where dispatch dominates. The
  jax trainer measured donation at about 1% on a much larger state, so treat 10% as the optimistic
  end.
- **Measure** — `bench_env_step_jax.py --mode jit --tag _donate`, paired ABBA at N = 1e3 to 3e6;
  assert no donation warning is emitted.
- **Risk** — none to exactness; donation changes buffer lifetime, not arithmetic. The one real
  hazard is a benchmark that keeps a reference to a donated array and then reads it, which raises
  in jax rather than returning silent garbage.

### 2. Four scalar planes instead of packed `[C, N, 2]` tensors, and no concatenate in the step

- **Change** — carry the state as four `[C, N]` arrays (`x`, `y`, `vx`, `vy`) rather than two
  `[C, N, 2]` arrays, and return them directly instead of building `obs` and `final_obs` with
  `jnp.concatenate`. Let the consumer stack once if it needs a `[C, N, 4]` observation.
- **Why** — the step currently does two `jnp.concatenate` calls and one `jnp.stack` per call, and
  every physics line immediately re-slices `pos[..., 0]` and `pos[..., 1]`. The design brief
  predicted that XLA prefers separate planes where CUDA prefers a packed vector, and named this as
  the one layout decision likely to differ between the implementations. D3's HLO read says directly
  whether the concatenates survive fusion; if they do, this is two full passes over the state per
  step.
- **Expected** — 10-25% at 1e5 envs and above if the concatenates are real; near zero if XLA
  already fuses them away, in which case the row still deletes code.
- **Measure** — full grid in both jit and scan modes, tag `_planes`, paired; re-read the HLO to
  confirm the concatenates are gone.
- **Risk** — pure layout, so the fixture numbers must be identical to the last bit; the checker's
  `make_step_batch` adapter takes `[B, 2]` arrays and must be kept as an adapter over the new
  internal form. The jax trainer's rollout scan carries this state, so its bitwise same-seed test is
  part of the gate.

### 3. Rank-by-counting contact selection, measured independently

- **Change** — the same restructure as CUDA row 2 and torch row 1, in jax.
- **Why** — the 8-candidate loop carries ten values through nested `jnp.where` calls, which XLA
  turns into a long chain of selects inside one fusion, and register pressure inside a fusion is
  what limits its occupancy. The review's item 9 is explicit that this class of change has measured
  opposite signs in eager torch and under XLA, so it must be measured here rather than inherited
  from the torch or CUDA verdict.
- **Expected** — 1.2-1.8x at 1e5 envs and above; unknown sign at the floor.
- **Measure** — full grid, both modes, tag `_rankcount`, paired.
- **Risk** — the tie rule, exactly as in the other two implementations. The jax fixture gate is the
  strictest of the three (one-step error 0 in float64), so a tie-order change would show up there
  only if the fixture set contains a tie; add the symmetric-tie fixture from CUDA row 2 first.

### 4. `unroll` on the rollout scan

- **Change** — sweep `unroll` in {1, 2, 4, 8} on the scan used by the scan-mode benchmark and by
  the trainer's rollout.
- **Why** — unrolling lets XLA keep the scan carry in registers across several steps instead of
  round-tripping the state, which is the 24-32 bytes per env-step the scan mode still pays. The
  reference measured 1.889x for `unroll=4` and a further 1.112x for 8 on their environment scan,
  and 4.67% on a scan whose body was much heavier. Our body is a heavy elementwise block, so expect
  something between those.
- **Expected** — 1.1-1.5x on the scan curve; nothing on the jit curve, which has no scan.
- **Measure** — `bench_env_step_jax.py --mode scan --tag _unroll{1,2,4,8}`, full grid; record
  compile time and peak memory, both of which the reference saw rise with the knob.
- **Risk** — none to exactness; unroll is a scheduling directive. Re-sweep it if the body changes
  (rows 2 and 3 both change the body), because the reference recorded the knob flipping sign after
  a body change.

### 5. Expose a K-step step function, for the benchmark only

- **Change** — add `step_k(state, actions[K, C, N, 2])` implemented as a `lax.scan`, and use it for
  a labeled K-step curve.
- **Why** — the question of whether the jax env should expose a fused K-step entry point for the
  trainer is already answered by the jax e2e ledger: the whole training iteration, including the
  rollout scan and the env step inside it, is ALREADY one jitted XLA program with donated state, so
  a separate K-step env function would be traced into the same program and fuse identically. It
  buys the trainer nothing. It is worth having for two other reasons: it makes the jax curve
  directly comparable to the CUDA K-step kernel of CUDA row 5, and it gives any non-jitted consumer
  a way to amortize the 73-76 us per-call floor.
- **Expected** — no trainer effect (state it as measured, from the e2e ledger, not as an
  assumption). On the benchmark it reproduces the existing scan curve with per-step outputs
  written, which should land between the jit line (145 us at 1e6) and the scan line (119 us).
- **Measure** — a third labeled jax line in the report, with per-step observations written so it is
  an honest trainer-reachable number rather than the reward-sum-only upper bound the current scan
  mode measures.
- **Risk** — none to exactness. The risk is presentational: three jax lines with different output
  contracts invite a wrong comparison, so each must state what it writes per step.

### 6. XLA flags, in this order

- **Change** — try, one at a time: `--xla_gpu_enable_command_buffer=FUSION` together with
  `--xla_gpu_graph_min_graph_size=1` (wraps the compiled program's kernel sequence in a CUDA graph,
  the XLA analogue of the torch capture of torch row 3); `--xla_gpu_autotune_level=4`; and
  `--xla_gpu_enable_priority_fusion` if D3 shows the step is split across more fusions than
  expected.
- **Why** — the jit curve's 73-76 us floor across four decades is per-call dispatch plus launches,
  which is the same problem torch row 3 attacks with capture; command buffers are the only lever
  jax offers against the launch half of it. The others are ordinary compiler knobs and are listed
  second because an elementwise program with no matmul gives an autotuner little to choose from.
  Flags that are worth naming only to be skipped: the matmul precision knobs and the latency-hiding
  scheduler have nothing to act on in this program (no matmul, one device, no collectives).
- **Expected** — command buffers: up to 2-3x on the flat part of the jit curve if the launch share
  is what the floor is made of, nothing above 3e5 envs. The other two: 0-5%.
- **Measure** — one flag per run through `XLA_FLAGS`, full grid, paired; the flag string goes into
  the result JSON, since a row measured under a different flag set is not comparable.
- **Risk** — none to exactness for the scheduling flags. Autotuning can change kernel selection and
  therefore floating-point association, so re-run the float32 fixture gate for any row where
  autotune level changes, and record that the float64 gate is the strict one.

### 7. int32 counters end to end, and the checker path

- **Change** — the jax step already uses int32 counters; make the fixture checker and the benchmark
  drive the same jitted function that is timed, rather than `dynamics_step` alone, and add the
  adversarial 1% perturbation check.
- **Why** — `make_step_batch` currently exposes `dynamics_step`, so the timed path (the full `step`
  with reward, ends, and auto-reset) is never gated by the fixtures. Every row above changes that
  path.
- **Expected** — no speed effect. It is what makes rows 2, 3, and 4 safe to keep.
- **Measure** — checker run with `--path timed`; confirm the adversarial perturbation is reported.
- **Risk** — none.

---

## Questions answered without an experiment

1. **bf16 or fp16 state, in any implementation: rejected on arithmetic, do not measure.** bf16
   carries 8 mantissa bits, so at a position of 4.5 m the spacing between representable values is
   0.031 m. One step moves at most `h * v_max` = 0.05 m and the contact margin is 0.002 m, so bf16
   cannot resolve either. fp16 at the same position has a spacing of 0.0039 m, still coarser than
   the margin. The environment has no matmul, so reduced precision buys only traffic, and the CUDA
   E2 row measured a 17% traffic cut at exactly zero effect. State stays float32.
2. **A K-step fused env step for the jax trainer: not needed.** The jax e2e ledger records that the
   whole iteration, env step included, is already one jitted program with donated state. Build the
   K-step entry point for the benchmark curve only (jax row 5).
3. **Cross-framework pairing (jax env with the torch trainer, or the reverse): closed.** Measured at
   +6.2 to +7.0 ms per env step over the dlpack boundary against a 70-260 us native step. Do not
   re-open.
4. **`mode="reduce-overhead"` and raising inductor's realize thresholds: closed by round one.** The
   first silently skipped capture; the second compiled for over 40 minutes without producing a
   kernel. Torch rows 3 and 4 reach the same goal by explicit capture and by trace-time unrolling.
5. **A guarded versus unconditional auto-reset in the CUDA kernel: already guarded**
   (`if (term || trunc)` in `step_kernel`), so there is nothing left to try there. The torch and jax
   implementations compute the spawn for every env and select it, which is the correct form for
   those two.
6. **The block-size sweep: done, and do not re-use the answer.** 128 and 256 tied at 1e6 envs and
   larger blocks were worse, but that was measured on the long-dependency-chain kernel; CUDA row 4
   re-sweeps it after row 2 changes the register footprint.

## What this list is worth to the end-to-end training path

Worth stating plainly so the ranking is not misread: at the batch sizes the trainer actually uses
(C copies x 4 envs, so 32 to 512 envs), swapping the whole torch env for the CUDA kernel moved a
training iteration by 1 ms out of 36.6 at C=128, because inside a captured graph the env's device
time is a small share. So these rows are worth a large amount to the Module 1 deliverable curves
and to any large-batch use, and a few percent to the training campaign. The one exception is CUDA
row 1, which is a correctness question about a number already in the report.
