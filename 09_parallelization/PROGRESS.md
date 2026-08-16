# 09_parallelization — top-level progress ledger

Single resume point for this multi-day task. Task statement:
`efficiency_improvement_user_prompt.md`. Layout: `STRUCTURE.md`. Update this file whenever a
phase advances; per-subtask experiment logs live in each subtask's `progress_and_changes.md`.

## Phase checklist

- [x] Phase 0 — scaffolding
  - [x] folder structure designed (`STRUCTURE.md`)
  - [x] reference repos cloned (autoresearch, nanoGPT, nanochat, cleanrl)
  - [x] parallel-agents skill unzipped (`parallel_agents_skill/`)
  - [x] MuJoCo dynamics probed; closed-form update rule verified EXACT (`pointmaze/common/physics_spec.md`)
  - [x] reference fixtures generated (`pointmaze/common/fixtures/`, 12 cases)
  - [x] serval05 local envs built (rnd09_torch, rnd09_jax on /localtmp/sl5nw)
  - [x] H100 lock protocol in place (`locks/`)
- [ ] Phase 1 — Module 1: batched GPU envs (each with its own autoresearch loop)
  - [x] pointmaze/common: physics spec (probe-verified EXACT incl. the MuJoCo contact law),
        fixtures, one-step + rollout checker, frozen cross-impl reset RNG
  - [x] torch_env: exact + fused (4.03e9 env-steps/s @1M envs compiled)
  - [ ] cuda_env: fork agent building it (fused single kernel)
  - [x] jax_env: exact (bit-identical resets to torch); 6.87e9 (jit) / 8.39e9 (scan) @1M
  - [ ] throughput curves: n_envs sweep on H100 (log-scale line), step time per env
- [ ] Phase 2 — Module 2: batched multi-copy PPO+RND (torch + jax)
  - [x] algorithm spec written (ppo/research/, incl. a real cleanrl bug found: intrinsic
        filter iterates the wrong axis; and a spec §11 gradient correction)
  - [x] torch trainer: correct (copy-isolation bitwise tests) + optimized 777 -> 45.6 ms/iter
        at C=128 (whole-rollout + whole-update CUDA-graph capture, both bitwise-verified)
  - [x] jax trainer (fork agent): all tests pass, cross-framework agreement 8.6e-7,
        74 iter/s (style A) / 50 iter/s (style B) at C=128
  - [ ] remaining optimization loop + learning sanity run
- [ ] Phase 3 — Module 3: end-to-end (each env variant x each trainer, fused where possible)
- [x] Phase 4 — extra step: review done (`extra_step_review.md`), improvements applied
      (RND-target hoist, GEMM packing, capturable-Adam coupling fix, incremental bench
      writes; CUDA E3 recorded as prepared candidate)
- [x] Phase 5 — final deliverable: campaign 10/10 records (both styles x C=8..128,
      10.24M steps/copy); smallN env benches; cuda-pairing (35.6/23.2 ms at C=128) and
      cross-framework dlpack measurements; unified report generated in
      `report/2026-08-15-pointmaze-gpu-parallelization/` (report.md + figures, all from
      result JSONs via code/make_report.py)

## Rounds

- Round 1 (2026-08-15, early): build all three modules; 777 -> 36.6 ms per iteration at 128
  copies; first campaign; first report.
- Round 2 (2026-08-15, afternoon): improve the finished system under a paired measurement
  protocol. 36.8 -> 20.2 ms (style B) and 24.5 -> 8.1 ms (style A) at 128 copies; jax +32/+47%;
  cuda env +14% and a real stream defect fixed; campaign re-run; on-policy-only question
  answered (no effect).
- Round 3 (2026-08-15, afternoon): learning-rate sweep across copy groups, +1.0% against a
  uniform run and 1.84x faster than running the groups separately; demonstration run recovers
  the expected best rate.
- Round 4 (2026-08-15, afternoon): close the distance to the JAX trainer at 8 to 128 copies —
  one flat parameter buffer and one shuffle per epoch, 20.4 -> 16.3 ms at 128 copies, with a
  1.1% loss at 512 copies recorded as the one size where it was a loss.
- Round 5 (2026-08-15, evening): re-open the question at the copy counts the trainer is
  actually used at, 1,024 to 4,096. Two findings before any change was made. First, the regime
  is different: at 128 copies the iteration's cost is the number of device programs it issues,
  at 1,024 and above it is the number of bytes it moves, and every individual program is
  already at 72 to 100 percent of the bandwidth the card delivers. Second, round four is a
  REGRESSION at these sizes — 19.2% slower at 1,024 copies with one update per batch, 7.1% with
  sixteen, 5.8% at 4,096 with sixteen, each measured against its own predecessor revision — 
  because packing the twenty-one parameter windows tightly left every copy's parameters off a
  sixteen-byte boundary and the multiplication library fell back to its scalar-load kernels.
  Three exact changes followed: pad the windows; add each layer's bias after the multiplication
  rather than folding it in; write gradients into the flat buffer instead of accumulating into
  it, and compile the gradient limit separately from the Adam step. Worth 11.3, 16.0 and 7.1
  percent respectively at 1,024 copies, and together 29 to 39 percent across 1,024 to 4,096 —
  with NO size slower: 14.7 percent at 8 copies, 22.5 at 128, 31 at 512. Head to head at 4,096
  copies with one update per batch: PyTorch 90.4 ms before, 63.0 after, against JAX 43.8; the
  distance to JAX falls from 2.06-2.08x to 1.27-1.44x with one update per batch and from
  1.75-1.81x to 1.21-1.27x with sixteen. 8,192 copies also fits (121.8 ms and 29.5 GB with one
  update per batch; 407.0 ms and 27.4 GB with sixteen) on the 94 GB card.

- Round 6 (2026-08-15 night, Pacific): read the JAX trainer's own fifth round, which had finished
  after this side's was written, and re-open 1,024 to 4,096 with the measurement method that round
  arrived at. Its verdict rule — the paired, round-by-round sign test, both versions built in ONE
  process and timed round-robin — was adopted and is now `benchmarks/bench_torch_change.py`. Its
  two kept changes did not transfer: both are unroll factors, and the PyTorch update has no loop
  to unroll; measured at these sizes the JAX round is worth nothing on its own side either
  (within a percent of its pre-round figures at 1,024 to 4,096), which is the same regime split
  seen from the other direction. A kernel-level profile at 4,096 copies then chose three changes,
  all of them removing passes over memory that round five's own work had created. Measured end to
  end against the revision it started from: -1.7% at 8 copies, -3.1% at 128, -4.9% at 512 and
  -7.0% at 4,096 with sixteen updates per batch, and no size slower. Against JAX at 1,024 to
  4,096, the distance falls from about 1.22x to about 1.13x with sixteen updates per batch. Two
  of the changes reverse sign with the copy count, so the trainer chooses the form from it. The
  round's own hypothesis — that PyTorch could be made to fold its element-wise work into its
  multiplications the way the JAX compiler does — is answered NO: forcing the generated kernels
  that would fuse an epilogue is 45.6% slower.

## State notes (newest first)

- 2026-08-15 ~22:00 PT — round 6 complete, on branch `worktree-agent-a9d3932a1c6532feb`. The
  question was what the JAX trainer's fifth round transferred to the PyTorch one, and whether
  1,024 to 4,096 copies could be improved again. Everything below is measured on serval05 under
  the exclusive lock, both frameworks waiting for every iteration.
  - **The premise was checked first, and it held.** Re-measured in one session, PyTorch reproduces
    round five's figures to within half a percent, and the JAX fifth round is worth **nothing at
    these sizes** — its trainer is within a percent of what it was before that round at 1,024 to
    4,096. Its two kept changes are unroll factors, and unrolling buys fewer, longer-running
    programs, which is worth ten percent at 128 copies and nothing where bytes are the cost.
  - **Its measurement method transferred and is now used here**: both versions built in ONE
    process, timed round-robin with the order reversed on alternate rounds, verdict by the
    round-by-round paired difference and a sign test (`benchmarks/bench_torch_change.py`).
  - **Four changes kept**, chosen by a kernel-level profile at 4,096 copies: read the gradients
    where the backward pass wrote them instead of copying them into one buffer; hold one
    contiguous block per parameter instead of one buffer row per copy; sum the gradient limit
    inside the copy where the buffer is kept; and write the shuffled batch straight into its
    buffer instead of building a second copy of it. End to end against the revision the round
    started from: **-1.7% at 8 copies, -3.1% at 128, -4.9% at 512 and -7.0% at 4,096** with
    sixteen updates per batch, -1.4% at 4,096 with one. **No size is slower.**
  - **Two of them reverse sign with the copy count**, so the trainer picks the form from it:
    reading the gradients in place is +6.7% / +5.4% / +2.8% SLOWER at 8 / 32 / 128 copies and
    -2.5% to -5.2% faster from 512 up, and the crossover is set at 1,024 because that is where
    the two candidate forms were compared directly. That is the same regime split round five
    found, seen from the other direction, and it is why every change is measured at both ends of
    the range.
  - **The hypothesis the round was built on is answered in the negative.** Letting PyTorch's
    compiler generate the multiplications so the bias and the activation fold into them, which is
    what the JAX compiler does for free, is **45.6% SLOWER** once the library backend is removed
    so that an epilogue actually fuses — the generated kernels are far enough behind the
    library's on these shapes that the passes they would remove cannot pay for them. Offering
    both backends reproduces round five's 3% gain, but that arm fuses no epilogue: the library's
    kernel simply wins the selection nearly everywhere.
  - **Three measurements measured the wrong thing, and all three were caught.** The first
    version of the epilogue probe swapped four compiled functions onto one trainer, and all four
    came out bitwise identical and the same speed, because the compiler caches against the
    function and the backend rather than against a surrounding context. The rebuilt probe then
    read its accuracy from weights each arm's own kernels had already moved, because forcing a
    compilation runs a real update. And the cross-process comparison built its sides from the
    dataclass defaults rather than through `production_config`, so once this round made the
    gradient's home depend on the copy count it timed, at 8 and 128 copies, a configuration the
    trainer never chooses there — and reported the round as a 3.7% REGRESSION at 8 copies when
    it is a 1.7% gain. Its within-side spread was 0.01 ms: precise, and about the wrong thing.
    All three were found the same way, by a check that asks whether the thing under test actually
    happened.
  - **Correction to the record**: the trainer has twenty-one parameter tensors per copy, not the
    nineteen every ledger and report sentence since round four has said.
  - **Largest thing left, measured but not attempted**: of the rollout's 15.3 ms at 4,096 copies,
    11.45 are matrix multiplications running at about 850 GB/s, because each of the 128 steps
    multiplies four rows per copy against that copy's whole actor weights and those weights do not
    fit the cache. The counted floor for the whole rollout is 2.8 ms.
  - **Against JAX**: with both trainers measured in one session, the distance at 1,024 / 2,048 /
    4,096 copies falls from 1.20x / 1.25x / 1.22x to **1.12x / 1.15x / 1.13x** with sixteen
    updates per batch, and stands at 1.25x / 1.36x / 1.43x with one. JAX's own fifth round moved
    none of those numbers.
  - Report section: "Training a thousand to four thousand copies at once", subsection "Round six"
    (`report/2026-08-15-pointmaze-gpu-parallelization/report.md`). Ledger: round 6 in
    `ppo/torch_ppo/progress_and_changes.md`. Every result JSON is in `benchmarks/results/` in both
    this branch and the main tree.

- 2026-08-15 ~19:30 PT — round 5 complete, on branch `worktree-agent-ab3b4d042ce36522c`. The
  question was how the PyTorch trainer compares with the JAX one at 1,024 to 4,096 copies, and
  whether it can be improved there. Answers, all measured on serval05 under the exclusive lock,
  both frameworks waiting for every iteration:
  - **The limit is different at these sizes.** At 128 copies the iteration costs what it costs
    because of how many device programs it issues; at 4,096 it is memory traffic, and every
    individual program already reaches 72 to 100 percent of the 3,539 GB/s a plain copy of
    memory gets on this card. Optimisations that remove programs cannot help here; ones that
    remove passes over memory can.
  - **Round four was a regression here**, 19.2 percent at 1,024 copies with one update per batch,
    measured against its own predecessor revision. Its flat parameter buffer packed the twenty-one
    windows tightly, leaving every copy's parameters off a sixteen-byte boundary, and the
    multiplication library answered with its scalar-load kernels: 18.8 of 31.1 milliseconds of
    multiplication time in one iteration.
  - **Three exact changes**, worth 11.3, 16.0 and 7.1 percent at 1,024 copies and 29 to 39
    percent together across 1,024 to 4,096, with no size slower (14.7 percent at 8 copies, 22.5
    at 128, 31 at 512). The unvectorised multiplication time falls from 18.8 milliseconds to
    zero.
  - **Against JAX**: the distance falls from 2.06-2.08x to 1.27-1.44x with one update per batch
    and from 1.75-1.81x to 1.21-1.27x with sixteen. What remains is that PyTorch issues separate
    programs whose intermediates go to memory, where the JAX compiler folds them together;
    measured, letting PyTorch's compiler generate the multiplications recovers 3 to 4 percent of
    the update stage but turns off the reduced-precision matrix units, so it was not adopted.
  - Report section: "Training a thousand to four thousand copies at once" (last section of
    `report/2026-08-15-pointmaze-gpu-parallelization/report.md`). Ledger: round 5 in
    `ppo/torch_ppo/progress_and_changes.md`. Every result JSON is in `benchmarks/results/` in
    both this branch and the main tree.

- Round 4 (2026-08-15 afternoon, Pacific): two branches merged.
  - `feature/torch-speed`: one flat parameter buffer (the twenty-one parameter tensors become
    windows onto one buffer, so the per-copy gradient clip is one reduction and the optimizer
    one chain) plus shuffling once per epoch instead of gathering inside every update step.
    C=128 sixteen-updates 20.4 -> 16.3 ms; C=8 13.8 -> 9.3 ms; one-update-per-batch at C=8 now
    beats jax. Honest loss recorded: 512 copies is 1.1% SLOWER, the buffer reaching 123 MB and
    the optimizer becoming bandwidth-bound.
  - `feature/jax-parity`: the learning-rate sweep and per-copy progress recording added to jax
    at zero measured cost at 8/128/512/2048 copies.
  - Still running: `feature/jax-speed` (ceiling-driven jax work), the clean-node cpu comparison
    on jaguar03, and the Pacific-time display change.

### Lessons worth keeping (they changed how the work is done, not just its result)

1. **Profile before following a plan.** The round-2 research note ranked the minibatch gather as
   a target and the gradient clip nowhere; the profile showed the clip at 31% of a minibatch step
   and the gather at 2%. The plan was rewritten from the measurement.
2. **Per-process A/B comparison is too noisy for small effects here.** Process-to-process
   variation alone is 0.68-0.72 ms on an 8-13 ms iteration, and more timed iterations do not
   reduce it. Constructing every arm in ONE process and timing them round-robin with the order
   reversed on alternate rounds gives a 0.06-0.10 ms floor. Effects below about 2% measured with
   the per-process harnesses should be treated as unresolved.
3. **A "measured" number can be measuring nothing.** Two cases this session: the cuda environment
   launched on the wrong stream, so its captured graph was empty and the pairing numbers never
   stepped the environment; and revision-loaded modules were not registered in `sys.modules`, so
   compiled comparisons against a previous revision failed outright. Both were found by writing a
   test that asserts the thing under test actually happened.
4. **Faster is not the same as near the limit.** The ceiling analysis puts the environment at 67%
   of its instruction-issue limit and the trainers at about a third of the matrix-work floor for
   the algorithm as written; remaining headroom is at most about 3x, not the 99% a naive "percent
   of peak" reading suggests.
5. **Record the losses.** The 512-copy regression, the discarded pairing numbers, and the
   experiments that produced no effect are in the ledgers beside the wins; a ledger of only
   successes would have hidden the stream defect for good.
6. **An optimisation is only established at the sizes it was measured at.** Round 4 was decided
   at 8 to 128 copies, recorded a 1.1% loss at 512 as an isolated exception, and is in fact a
   19.2% regression at 1,024 — the size the trainer is actually used at. Before keeping a change,
   measure it where the code runs, not only where it was developed.
7. **Read the names of the programs, not just their times.** The round-4 regression is invisible
   in a phase breakdown and unmistakable in a kernel-level profile: 18.8 of 31.1 milliseconds of
   multiplication time sat in the library's scalar-load kernels, whose names end in `align1`,
   because a buffer layout had moved every copy's parameters off a sixteen-byte boundary.
8. **Two compiled functions can beat one.** Compiling the gradient limit and the Adam step
   together let the compiler emit a single reduce-and-update program that reached 2.4 of the
   card's 3.5 terabytes per second; compiled separately they reach 3.2 and 3.5. Fusion is not
   free at every size.
9. **A driver held by the session dies with it.** On a graphics processor shared with two other
   agents, a batch spends more time queueing than running, and a background driver in the session
   is killed after an hour. The round-5 batch was moved onto the machine itself (`setsid`, logs on
   shared storage) so the queue could outlast the session that started it.

- 2026-08-15 ~18:40 — processor comparison. End-to-end training measured on jaguar03 (AMD EPYC
  7663, 224 logical processors, 1 TB) held EXCLUSIVELY inside reservation sl5nw_156, chosen as
  the largest completely idle node on the cluster; cheetah04 has more cores but was already
  shared, and serval03 is in maintenance until 2026-08-31. Job 6537825, ids in
  `analysis/2026-08-15-jaguar03-cpu-endtoend/slurm/`. Both parallelisation styles (threads in
  one process, independent single-thread processes) over 8 to 3,584 copies, both update
  conventions. The graphics-processor style-A gap at 2,048 and 4,096 copies was filled at the
  same time (23.3 and 24.2 million environment steps per second). Report sections 5 and 6 added
  to `report/2026-08-15-gpu-parallel-rl-environment-training-endtoend/` via
  `code/cpu_sections.py`, with the graphics processor drawn dashed in every comparison figure.
  DONE at 18:55 Eastern (15:55 PT), every point measured. Independent processes reach 2.11
  million environment steps per second at 3,584 copies (3.80 with one update per batch); the
  threaded form tops out at 0.053 million, a factor of 40 slower on the same machine. Raising
  the thread count from 8 to 112 was slower at 7 of the 8 copy counts measured. Best setup under
  4,096 copies: graphics processor 24.1 million at 0.47 hours to give every copy ten million
  steps, against 3.80 million and 2.61 hours for the processor node — a factor of 6.3.
  Two defects found and fixed while checking the rendered output: the wall-time panel of
  `figures/best_setup.png` was drawn upside down against its shared row labels (it read as the
  graphics processor being the SLOWEST), and the method section was numbered 5, colliding with
  the new processor section.

- 2026-08-15 ~04:45 — TASK COMPLETE. All five phases checked off. Unified report:
  `report/2026-08-15-pointmaze-gpu-parallelization/report.md` (+ a self-contained HTML
  render, regenerable via code/make_artifact_html.py). The 20-minute monitor is retired.
  Open follow-up candidates recorded in ledgers: CUDA env E3 (distance-only tournament),
  compiling _post_body's scans, bf16 gated test.

- 2026-08-15 ~02:15 — torch env validation contract fully satisfied (fixtures float64+float32,
  unit tests, cross-impl RNG identity, random-policy distributional check TV=0.026 PASS).
  Learning sanity run (C=8, 20k iters, 81.92M total steps in 658 s): goal reached from
  ~2.3M steps/copy, oscillating rediscovery, SUSTAINED reward at run end (mean 32.75,
  max 210 per 512-step iteration). The trainer learns end to end. Torch trainer at 45.6 ms/iter (C=128); one-graph mode implemented,
  bench pending. Fork agents: CUDA env in flight; jax agent continuing with e2e fusion tasks.

- 2026-08-15 ~00:30 session start. serval05 idle (H100 NVL 95GB). Dynamics rule verified exact:
  v' = (m*clip(v,±5) + h*g*clip(a,±1))/(m+h*d), q' = q + h*v'; hard-wall clamp within 1.4 mm
  of MuJoCo soft contact. Fixtures written. Next: serval05 env bootstrap + lock + research
  fan-out + torch env v0.

- 2026-08-15 ~17:15 PT — a second processor node measured against jaguar03, and the earlier
  processor numbers corrected. Node: jaguar02 (Intel Xeon Gold 6334, 16 cores / 32 threads,
  the highest clock on this cluster), held exclusively; the Zen 4 serval machines could not be
  used (one in maintenance, four carrying other users' multi-day jobs). Run folder
  `analysis/2026-08-15-16-39_cpu-node-comparison-newer-processor-vs-jaguar03/`; section
  generator `report/2026-08-15-pointmaze-gpu-parallelization/code/cpu_node_comparison.py`
  (`sec_cpu_node_comparison()`), not yet wired into make_report.py.
  **The five-iteration processor measurements are burst numbers, not sustained ones.** Repeating
  them at 150 iterations drops jaguar03 by 27% at 112 workers and 68% at 224; jaguar02 is
  unaffected. Two causes, both growing with worker count: jaguar03's clock settles from 3.52 to
  2.60 GHz with every core busy (jaguar02 holds 3.56), and over seven iterations several hundred
  processes are still starting at different moments, so the benchmark's sum of per-worker rates
  describes a load that never existed.
  Consequences: jaguar03's best setting is **112 workers, not 224** — filling all 224 hardware
  threads is 11% WORSE (0.1276 -> 0.1138 million steps/s), where the burst numbers said it nearly
  doubled. On jaguar02 the second thread is still worth +11%. Every worker count the argument
  rests on was measured three times; the 224-worker point repeated to 0.1138 exactly.
  At equal load (every physical core busy) jaguar02 gives 1.27 thousand steps/s per copy against
  jaguar03's 1.14, but on a clock 37% higher — so jaguar03 does 1.23x as much work per clock
  cycle. Cause traced to the last-level cache, measured on both nodes: at an 8 MiB working set one
  dependent access takes 44.3 ns on jaguar02 and 15.6 on jaguar03, and jaguar03 has 4.57 MiB of
  level-3 cache per core against 2.25. Not the vector units — the array library uses AVX512 on
  jaguar02 and only AVX2 on jaguar03, and the AVX512 machine is the one doing less per cycle.
  Extrapolation: scaled to 112 cores, jaguar02's per-core rate projects to 0.159 million steps/s,
  about 125% of jaguar03's best (140% of jaguar03 at the same 224 threads) — but that assumes the
  clock, the memory bandwidth per core and the cache per core all survive a sevenfold core
  increase, and jaguar03 is itself the evidence that they do not. Read as an upper bound.
