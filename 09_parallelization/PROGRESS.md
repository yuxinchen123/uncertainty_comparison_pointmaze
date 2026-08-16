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

## State notes (newest first)

- Round 4 (2026-08-15 afternoon, Pacific): two branches merged.
  - `feature/torch-speed`: one flat parameter buffer (the nineteen parameter tensors become
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
  jaguar03's 1.16, but on a clock 37% higher — so jaguar03 does 1.24x as much work per clock
  cycle. Cause traced to the last-level cache, measured on both nodes: at an 8 MiB working set one
  dependent access takes 44.3 ns on jaguar02 and 15.6 on jaguar03, and jaguar03 has 4.57 MiB of
  level-3 cache per core against 2.25. Not the vector units — the array library uses AVX512 on
  jaguar02 and only AVX2 on jaguar03, and the AVX512 machine is the one doing less per cycle.
  Extrapolation: scaled to 112 cores, jaguar02's per-core rate projects to 0.159 million steps/s,
  about 125% of jaguar03's best (140% of jaguar03 at the same 224 threads) — but that assumes the
  clock, the memory bandwidth per core and the cache per core all survive a sevenfold core
  increase, and jaguar03 is itself the evidence that they do not. Read as an upper bound.
