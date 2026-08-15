# Round 2 — adversarial review of the current torch implementation

Scope: `ppo/torch_ppo/torch_ppo_rnd.py`, `pointmaze/torch_env/torch_pointmaze.py`,
`pointmaze/cuda_env/{cuda_pointmaze.py,pointmaze_kernel.cu}`, `benchmarks/bench_train.py`,
`benchmarks/bench_env_step.py`, `benchmarks/profile_breakdown.py`, `train_runs/run_final.py`,
`ppo/torch_ppo/tests/`, `pointmaze/tests/`, and the numbers in
`report/2026-08-15-pointmaze-gpu-parallelization/report.md`.

Findings already written down in `extra_step_review.md` section 4 (no paired A/B, no noise floor,
syncs inside the timed region, no behavioural gate, out-of-memory ceiling not confirmed in a fresh
process, thin provenance) are not repeated except where I can add a mechanism, a number, or a
sharper test. Everything below is new or newly quantified.

## Findings, most severe first

### 1. `prime_obs_rms` reads the same buffer 128 times when `env_backend="cuda"` — blocker

- File: `ppo/torch_ppo/torch_ppo_rnd.py:757-764`; the buffers are
  `pointmaze/cuda_env/cuda_pointmaze.py:63-67`, returned at `:88`.
- What is wrong: `CudaPointMaze.step` returns its PREALLOCATED `_final_obs` (its own docstring
  says "callers that keep a step's outputs must copy them"). `prime_obs_rms` does
  `batch.append(final_obs)` for `num_steps` steps and then `torch.stack(batch)`. Every element of
  `batch` is the same tensor object, so the stack contains `num_steps` copies of the LAST step's
  observation. `obs_rms` is then primed with a batch whose variance along time is exactly zero.
  The whitening in `whiten` divides by that variance, so the RND input is far outside +-5 and
  clips, for the whole run. The torch backend is unaffected (`step_core` returns a fresh `cat`).
- Fix: `batch.append(final_obs.clone())`, or write into a preallocated `[T, C, N, 4]` buffer.
- Demonstrating check: with `env_backend="cuda"`, run `prime_obs_rms` and assert
  `trainer.obs_rms.var.min() > 1e-3`; today it will be at the floor. Equivalent one-liner:
  `assert not torch.equal(torch.stack(batch)[0], torch.stack(batch)[-1])`.

### 2. The fused CUDA env launches on the legacy default stream, and the cuda pairing has no equivalence test — blocker for the pairing claim

- File: `pointmaze/cuda_env/pointmaze_kernel.cu:131, 151, 166` — all three launches are
  `kernel<<<blocks, threads>>>(...)` with no stream argument, i.e. stream 0.
- What is wrong: every other operation in the captured region runs on PyTorch's capture stream.
  A launch on the legacy default stream during an active capture is either rejected (CUDA returns
  a capture error, which `C10_CUDA_KERNEL_LAUNCH_CHECK` would raise) or, if it slips through,
  carries no dependency edge to the surrounding PyTorch kernels. In the second case the replayed
  graph has the policy forward reading `env.state` with no ordering against the env kernel that
  writes it — a data race, and the env may not advance at all on replay. The report presents
  "CUDA env + torch trainer, one-graph, 35.6 ms at C=128" as a measured pairing; the JSON
  `benchmarks/results/2026-08-15-04-34-16_..._cudaenv3.json` has `rollout_sec_per_iter = 0.0`,
  which in `bench_train.py` only happens in the one-graph branch, so that number came from a
  captured graph containing these launches. Nothing in the repository tests that the replay
  actually steps the environment.
- Fix: take `at::cuda::getCurrentCUDAStream()` and pass it as the fourth launch argument in all
  three functions. Then add the equivalence test below.
- Demonstrating test: build the iteration graph with `env_backend="cuda"`, snapshot
  `env.state` and `env.reset_count`, call `iteration_captured()` once, and assert the state
  changed and `reset_count` advanced. Stronger: run the same seed through `_rollout_body_cuda`
  eagerly and through the replay and compare `_bufs["obs"]`, `_bufs["rext"]`, `_bufs["rint"]`
  bitwise, the same shape as `test_rollout_capture_bitwise`.

### 3. The learning-rate anneal inside the captured graph is never exercised — should-fix

- Files: `torch_ppo_rnd.py:167-175` (tensor `lr`), `:789-796` (`self._lr_t.fill_(lr)`),
  `benchmarks/bench_train.py:38-61` (never calls `train()`), `tests/test_capture_gpu.py:43-65`
  (calls `update_*` directly, constant lr).
- What is wrong: the whole final campaign runs `train(anneal_lr=True)` through a captured graph.
  Correct annealing requires that `torch.optim.Adam` hold the very tensor `self._lr_t` in
  `param_groups[0]["lr"]` and that the fused capturable kernel read it from device memory at
  replay. If any layer of that chain converts the tensor to a Python float at build time, the
  graph replays a frozen learning rate and every campaign number is a constant-lr run. Nothing
  measures this; the benchmark path never touches `train()`, and both capture tests use a
  constant lr. `extra_step_review.md` section 4 item 6 flagged the missing lockstep test; the
  test below is cheaper and decisive.
- Fix (test, not code): after the graph is built, set `_lr_t` to zero and replay.
- Demonstrating test:
  `trainer._lr_t.fill_(0.0); snap = [p.detach().clone() for p in trainer.trainable];
  trainer.iteration_captured(); assert all(torch.equal(p, s) for p, s in zip(trainer.trainable, snap))`.
  With lr exactly 0 an Adam step changes nothing, so any parameter movement proves the graph
  baked a stale learning rate. Pair it with a step-counter check: read
  `opt.state[p]["step"]` before and after one replay and assert it advanced by 16 (style B) or 1
  (style A).

### 4. `run_final.py` writes nothing until the run finishes, and resumes on a filename that omits the settings — should-fix

- File: `train_runs/run_final.py:27-30, 41-43, 49-66, 91-98`.
- What is wrong, in four parts:
  1. The only record is written after all 20,000 iterations. A kill at iteration 19,999 loses the
     entire run, including the 400 history samples already computed. This is the case the
     resumable-generation rule exists for.
  2. The resume key is `copies_{C}_{style}.json`. Change `--iterations`, `--base-seed`, or
     `--one-graph` and the run is skipped with `[resume] ... skipping`, after which the report
     presents the old numbers under the new settings.
  3. The record stores `n_copies`, `style`, `one_graph`, `iterations`, `base_seed` only. It does
     not store `tf32`, `fused_adam`, `rollout_mode`, `capture_update`, `num_steps`, `n_envs`,
     `learning_rate`, or the env config — all of which `run_one` hard-codes at `:41-43` and all of
     which the reported throughput depends on. Verified against the shipped records: the key set is
     exactly `[base_seed, build_plus_train_seconds, env_steps_per_sec, final_coverage_per_copy,
     git, global_step, gpu, history, iterations, n_copies, one_graph, peak_vram_mb, style, torch,
     train_seconds]`.
  4. The multi-count parent uses `subprocess.run(..., check=True)`, so one failing copy count
     aborts every later one and no record is written for the ones that would have succeeded.
- Fix: stream each history sample to `data/copies_{C}_{style}.jsonl` as it is produced and flush;
  write the summary record with `"completed": false` at start and `true` at the end; put
  `dataclasses.asdict(cfg)` in the record; make the resume compare the stored config against the
  requested one and refuse (not silently skip) when a behaviour-defining field differs; drop
  `check=True` and record the failure.
- Demonstrating check: run with `--iterations 100`, kill at iteration 50, re-run — today nothing
  survives. Then re-run with `--iterations 200` and observe the stale 100-iteration record being
  reported as a 200-iteration run.

### 5. "copies at goal" is a sampling artifact of `history_every=50` — should-fix

- Files: `torch_ppo_rnd.py:803-811` (history sampled every `history_every` iterations),
  `report/.../code/make_report.py:511-519` (union over the last 10% of samples).
- What is wrong: `_log_rext_sum` holds the reward of ONE iteration — the one that happened to be
  sampled. With `history_every=50`, 400 of 20,000 iterations are seen, so the "final 10%" is 40
  sampled iterations out of 2,000. Measured on the shipped C=128 style-B record:
  per sampled iteration 19 to 25 copies have positive reward; the union over the 40 sampled late
  iterations is 27; the union over all 400 samples is 49. Counting all 2,000 late iterations would
  give a number well above 27. The reported "27/128" is therefore a function of `history_every`,
  not of the run.
- Fix: accumulate on the GPU each iteration — `self._ever_positive |= (self._log_rext_sum > 0)`
  and a running per-copy sum — which costs one elementwise kernel and no host sync, and report
  that instead of the sampled union.
- Demonstrating measurement: re-run one configuration with `--history-every 1` and compare the
  "copies at goal" column; if it moves, the current column is measuring the sampler.

### 6. The profiling breakdown does not describe the shipped configuration and cannot be made to — should-fix

- File: `benchmarks/profile_breakdown.py:51-52` (config), `:80-98, 107-116` (component view),
  `:130-138` (totals).
- What is wrong, in four parts:
  1. The trainer is built with `rollout_mode="capture", capture_update=True, fused_adam=True` and
     no `tf32`, no `one_graph`. The shipped configuration is one-graph plus TF32 plus the target
     hoist plus GEMM packing. The script has no flag for either, so the report's phase table
     (total 50.53 ms) describes a configuration that is 14 ms slower than the one whose throughput
     the report quotes (36.6 ms), which is why the phase rows do not sum to the headline.
  2. The component view times each block in isolation at `[C, N] = [128, 4]` = 512 rows. Every one
     of those kernels is launch-bound, so the block ranking is essentially a kernel-count ranking,
     not a work ranking. The report's sentence "in eager form the ENVIRONMENT dominates (about 84%
     of raw kernel time)" is not supported: it is 84% of isolated-call wall time, most of which is
     launch overhead.
  3. Two rows carry arbitrary multipliers: the GAE proxy at `:89-93` times only the extrinsic
     recursion without `vext_next` or the bootstrap mask and multiplies by 2; the flatten/whiten
     proxy at `:96-98` multiplies one whiten by 8. Both feed the published percentages.
  4. `comp` contains both `update: loss forward` and `update: forward + backward`, and the second
     contains the first, so the script's own denominator double-counts it. (`make_report.py`
     removes the `env dynamics_step` row from its denominator but not this one; the script's
     printed percentages and the report's therefore differ.)
- Fix: add `--tf32`/`--one-graph` and profile the shipped config; get the component split from
  `torch.profiler` kernel-time sums over one real captured iteration rather than isolated eager
  calls; delete the two multiplier rows or measure them properly; drop one of the two overlapping
  update rows from the denominator.
- Demonstrating measurement: sum the CUDA kernel durations from a `torch.profiler` trace of one
  `iteration_captured()` and compare the env share against the 84% claim.

### 7. `one_graph=True` does not force a compiled rollout — should-fix

- File: `torch_ppo_rnd.py:118-119` (forces `capture_update`), `:224-231` (compile gated on
  `rollout_mode`), `benchmarks/bench_train.py:21, 80` (`rollout_mode` defaults to `"eager"`).
- What is wrong: the same class of defect that was already fixed for `capture_update` is still
  open for `rollout_mode`. `one_graph=True` with the default `rollout_mode="eager"` builds a
  perfectly valid graph containing 128 steps of unfused eager kernels — the configuration measured
  at 187-211 ms of rollout in ledger row 3 — and reports it as the one-graph number, with no
  warning. `bench_train.py --one-graph` reaches this state unless `--rollout-mode capture` is also
  typed. The shipped JSONs happen to have been run correctly (41.5 ms is a compiled figure), so
  this has not yet produced a wrong number; it is one forgotten flag away.
- Fix: in `__init__`, force `rollout_mode = "capture"` whenever `one_graph` is set, exactly as
  `capture_update` is forced two lines above.
- Demonstrating measurement: run `bench_train.py --one-graph --n-copies 128` without
  `--rollout-mode capture` and compare; a large gap proves the trap is live.

### 8. TF32 is in the production configuration with only a speed number behind it, and the flag is process-global — should-fix

- File: `torch_ppo_rnd.py:122-123`.
- What is wrong: `torch.set_float32_matmul_precision("high")` is set in the constructor and never
  restored, so it applies to every fp32 matmul in the process for the rest of its life, including
  the RND target and predictor. Two consequences:
  1. The intrinsic reward is $r^{\text{int}} = \tfrac{1}{2}\lVert p - t\rVert^2$, a squared
     difference of two nearly equal 128-vectors. TF32 keeps about 10 mantissa bits, so the target
     and predictor features each carry roughly $10^{-3}$ relative rounding; with feature magnitudes
     of order 0.3 that is a few times $10^{-3}$ absolute per feature. The learning run ends with
     `rint` near 0.03, i.e. about 0.022 per feature. The rounding is within one decade of the
     signal, so TF32 plausibly sets a floor on the RND bonus late in training. This needs
     measuring, not assuming.
  2. The flag leaks across trainers in one process. Any paired comparison built in a single
     process (which `bench_train.py:94-103` does for every copy count) measures TF32 against TF32
     once one side has enabled it. `extra_step_review.md` section 1 item 15 predicted exactly this;
     it is still unfixed.
- Fix: record and restore the previous precision in a context manager, or add an
  `assert_global_state()` that refuses to build a `tf32=False` trainer in a process where the
  precision is already "high".
- Demonstrating measurement: with fixed weights and a fixed whitened batch, compute
  $\tfrac{1}{2}\lVert p - t\rVert^2$ under `"highest"` and under `"high"` and report the absolute
  difference next to the observed late-training bonus of 0.03.

### 9. `test_copy_isolation`'s "the perturbation did something" guard is vacuous — should-fix

- File: `ppo/torch_ppo/tests/test_torch_ppo.py:49-57`.
- What is wrong: the test perturbs `b.actor["W0"][2]` by 0.05 and, after training, asserts
  `not torch.equal(a.actor["W0"][2], b.actor["W0"][2])`. That assertion is satisfied by the
  perturbation itself — it holds even if training did nothing at all, if the rollout produced
  identical actions, or if the updates never reached copy 2. So the test can report "copies are
  isolated" while proving nothing about copy 2 having been trained differently.
- Fix: assert on a parameter that was NOT perturbed, so a difference can only have arrived through
  the trajectory and the update. For example
  `assert not torch.equal(a.critic["W0"][2], b.critic["W0"][2])` and
  `assert not torch.equal(a.predictor["W0"][2], b.predictor["W0"][2])`.
- Related gap: the test runs on CPU at `C=4, N=2, T=8` and never touches the GPU paths, so copy
  isolation is unverified for the packed GEMMs, the captured update, and the one-graph iteration.

### 10. The env fidelity gates never run the code path that is timed and shipped — should-fix

- Files: `pointmaze/common/code/check_against_fixtures.py` (drives the eager `step`),
  `pointmaze/tests/test_torch_env.py` (eager, float64, CPU),
  `torch_ppo_rnd.py:224-225` (the shipped path is `torch.compile(self._one_step_pure)` inside a
  CUDA graph).
- What is wrong: the report's correctness section (one-step error 4.4e-16, rollouts to 1e-13) is
  entirely about the eager float64 path. The trained path is `step_core` traced by inductor,
  fused with the policy and RND forwards, in float32, replayed from a graph. There is no test
  comparing the eager env step against the compiled one, so an inductor reassociation or a fusion
  bug in the contact pipeline would not be caught by anything. `test_rollout_capture_bitwise`
  compares `compile-step` against `capture` — the same compiled kernels on both sides — so it can
  only catch a capture bug, never a compile bug.
- Fix: add a `--path {eager,compiled,captured,cuda}` switch to the fixture checker, and add the
  adversarial confirmation `extra_step_review.md` section 1 item 16 asks for (perturb one constant
  by 1%, confirm the checker fails).
- Demonstrating test: `torch.equal(env.step_core(*state, act), torch.compile(env.step_core)(*state, act))`
  over the fixture cases in float32, and the same comparison for a 400-step rollout with a
  divergence-horizon report.

### 11. The target hoist and the GEMM packing are mathematical identities that nothing pins — note

- Files: `torch_ppo_rnd.py:258-270` (`rnd_features`, packed), `:272-285` (`actor_critic`, packed),
  `:293-297` (`target_features`, unpacked), `:516` (the hoist).
- What is wrong: neither change alters the mathematics, and both change the numbers. Packing
  concatenates two `[C, 4, 64]` weights into `[C, 4, 128]`, so cuBLAS picks a different kernel and
  a different accumulation order; the hoist moves the target forward from a `[C, 128, 4]` per
  minibatch GEMM to one `[C, 512, 4]` GEMM, again a different tiling. Under TF32 the difference is
  around $10^{-3}$ relative. Ledger rows 7 and 8 record "GPU capture tests still bitwise-pass",
  but those tests compare capture against no-capture within the SAME build; they say nothing about
  the pre-change build. Nothing in the repository asserts that `target_features(x)` equals the
  `tf` returned by `rnd_features(x)`, or that `actor_critic(x)` equals
  `(actor_mean(x), *critic_values(x))`.
- A second-order consequence worth stating: the critic is now evaluated by two different code paths
  whose outputs are combined in the GAE. `vext_buf[t]` comes from the packed `actor_critic` at
  `:343`; `vext_next[t]` comes from the unpacked `critic_values` at `:470`. For a step that did not
  end an episode these are mathematically the same quantity, but they are not the same number.
- Fix: add the two identity tests with a tolerance tied to the dtype (1e-6 in fp32, 3e-3 in TF32),
  and note the tolerance in the ledger rows so "no value changed" is a checked claim rather than
  an inference.
- Related, still open: the bootstrap forward at `:470-472` is redundant for every step that did not
  end an episode, since `nobs_buf[t] == obs_buf[t+1]` there. Masking it down to the done steps plus
  `t = T-1` removes a full critic pass over 512 rows per copy per iteration and removes the
  two-path inconsistency at the same time. This is `extra_step_review.md` section 1 item 4,
  confirmed against the code and still unapplied.

### 12. `_visited` is not restored after the graph-build warmup — note

- File: `torch_ppo_rnd.py:713-716` (the snapshot list) and `:523` (the scatter).
- What is wrong: `_build_iteration_graph` snapshots and restores env state, the intrinsic filter,
  both running statistics, the parameters, and the optimizer state, but not `self._visited`. The
  three warmup iterations therefore mark cells that the counted run never visited, and the
  campaign's "coverage mean" column includes them. The effect is 3 iterations in 20,000, so it is
  small; it is listed because the same omission on any future accumulator would not be small.
- Fix: add `self._visited` to `stat_tensors`.

### 13. The copy-scaling ladder and the out-of-memory ceiling come from one long-lived process — note

- File: `benchmarks/bench_train.py:94-103`.
- What is wrong: every copy count is built in the same process, and nothing frees the previous
  trainer explicitly or calls `torch.cuda.empty_cache()`. Each trainer holds a CUDA graph with its
  own private memory pool, and dynamo's code cache keeps compiled artifacts alive, so allocator
  state at C=32,768 depends on everything measured before it. The headline "maximum copies that
  fit: 32,768" and the C=65,536 out-of-memory row are both products of that state.
  `extra_step_review.md` section 4 item 8 asked for fresh-process confirmation; the mechanism here
  is the specific reason it matters for this script.
- Fix: run one copy count per process (as `run_final.py:89-98` already does) and report
  `memory_reserved` alongside `max_memory_allocated`.
- Related: `run_final.py:60` reports `torch.cuda.max_memory_allocated()` as "peak VRAM". That
  excludes the CUDA context and cuBLAS workspaces, so the report's 146 MB at C=8 is not what
  `nvidia-smi` would show. Rename the column to peak allocator bytes or add
  `torch.cuda.max_memory_reserved()`.

### 14. The post-rollout phase is about 1,500 tiny kernels of sequential device time — note (performance)

- File: `torch_ppo_rnd.py:477-480` (intrinsic filter, T iterations) and `:491-497` (GAE, T
  iterations, both streams).
- What is wrong: nothing, but it is the largest remaining unexplored win. Inside the graph these
  loops cost no launch time, yet they still execute roughly 1,500 kernels on `[C, N]` tensors of
  512 elements each, all pure overhead and all serially dependent. The measured post-processing
  phase is 14.83 ms of the 50.53 ms iteration, the second-largest block.
- Fix: both recursions are first-order linear scans and can be done in a handful of kernels.
  The intrinsic filter is $F_t = \gamma F_{t-1} + r_t$, computable as a `cumsum` of
  $r_t \gamma^{-t}$ rescaled by $\gamma^{t}$; with $\gamma_{\text{int}} = 0.99$ and $T = 128$ the
  largest rescale factor is $0.99^{-127} \approx 3.6$, so there is no dynamic-range problem. GAE
  has a step-varying coefficient $c_t = \gamma\lambda(1 - d_t)$ and needs a reverse
  `cumprod`/`cumsum` pair; the product over 128 steps is $0.949^{128} \approx 1.3\times10^{-3}$,
  again well inside float32. Note that this is a different change from
  `extra_step_review.md` section 1 item 5 (compile the post phase) — compiling does not remove the
  sequential depth, a log-depth or cumsum formulation does.
- Demonstrating measurement: time `_post_body` before and after with CUDA events at C=8 and
  C=128, and check the GAE result against the loop version to 1e-5.

### 15. The "before" baseline runs env code that was restructured to help the "after" — note

- Files: report table at `report.md:72-84`, ledger `pointmaze/torch_env/progress_and_changes.md`
  row 3.
- What is wrong: the 777 ms eager baseline uses today's contact pipeline — the unrolled
  8-candidate loop with the two-smallest tournament. That restructure was adopted because it
  compiles well, and the same ledger row records that it made the SMALL-batch eager path slower
  ("117 -> 153 us at 1k"). The trainer runs at 512 envs, squarely in that regime. So the reported
  21x compares an optimized configuration against a baseline that the optimization work itself
  moved downward. The env table at `report.md:181-183` states this honestly for the env rows; the
  trainer row's label "BEFORE any optimization (eager)" does not.
- Fix: either relabel the row as "current physics code, run eagerly, no compile or capture", or
  measure the pre-tournament env eagerly once and quote that as the true before.
- Also on that table: the "before" column at C=16 and C=64 (1.05e4, 4.22e4) is the 777 ms figure
  carried across, not a measurement — only C=8, 32, 128 were run. The parenthetical says so; the
  table does not mark the derived cells.

## Smaller notes

- `torch_ppo_rnd.py:118-119` mutates a frozen dataclass with `object.__setattr__`. It works, but a
  `PPOConfig` shared between two trainers silently acquires `capture_update=True`. Resolve into an
  instance attribute instead.
- `torch_ppo_rnd.py:218` assigns `self._open_cells = (self.env.nb_mask >= 0)` and overwrites it four
  lines later. Dead.
- `torch_ppo_rnd.py:775-780` computes `update` and then never uses it in the `one_graph` branch.
- `torch_ppo_rnd.py:636` takes an `example_batch` parameter that is unused.
- `torch_pointmaze.py:178` rebinds `d1, d2` from the tournament distances to the QP diagonal. It is
  correct, because the distances are dead by then, but it makes the two-contact solve hard to audit.
- `pointmaze/tests/test_torch_env.py` covers only the eager float64 CPU path; there is no unit test
  for the env at float32 on the device, which is what runs.
- `ppo/torch_ppo/tests/dump_forward_fixture.py:30` rebinds `t.obs_rms.mean` and `.var` rather than
  copying in place. Harmless in the fixture, but it is exactly the pattern that would break a
  captured graph, so it is worth a comment saying so.
- `benchmarks/bench_env_step.py` sweeps only `n_envs` with `n_copies=1`. The trainer's shape is
  `C x N` with tiny `N`, so the env curve in the report does not cover the aspect ratio the trainer
  actually runs.

## Tests worth adding

Ordered by what they would have caught.

1. **Captured graph reads the live learning rate.** Set `_lr_t` to 0 after the build, replay once,
   assert every trainable parameter is bitwise unchanged. Then set it back and assert they move.
   Catches a frozen Python-side lr, which today would silently invalidate the whole campaign.
2. **Adam step counter advances per replay.** Read `opt.state[p]["step"]` before and after one
   `iteration_captured()`, assert it advanced by the number of optimizer steps in the style.
   Catches a frozen bias correction.
3. **Two replays differ, and two replays with pinned noise agree.** Replay twice and assert
   `_bufs["act"]` differ; then re-fill `_Z` with a saved copy, replay, and assert the rollout is
   bitwise equal to the first. Catches both a frozen noise buffer (which looks very fast and passes
   every mean-based check) and a nondeterministic capture.
4. **CUDA-backend rollout equivalence.** With `env_backend="cuda"`, compare the eager
   `_rollout_body_cuda` against the captured replay bitwise, and assert `env.reset_count` advances.
   This is the only thing standing between the reported pairing number and a graph that may not be
   stepping the environment.
5. **Observation statistics after priming are non-degenerate.** Assert `obs_rms.var` is above a
   floor and that the first and last primed observations differ, for both env backends. Catches
   finding 1 directly.
6. **Copy isolation on an unperturbed parameter, on the GPU, in the shipped configuration.**
   Perturb copy 2's actor weight, train with `one_graph=True, tf32=True`, and assert copies 0, 1, 3
   are bitwise equal AND that copy 2's critic and predictor weights differ between the two runs.
7. **Packed equals unpacked.** `target_features(x)` against `rnd_features(x)[0]`, and
   `actor_critic(x)` against `(actor_mean(x), *critic_values(x))`, at a dtype-appropriate tolerance,
   run once with TF32 off and once on. Turns ledger rows 7 and 8 into checked claims.
8. **Eager env against compiled env against the CUDA kernel, on the fixtures, in float32**, with the
   adversarial confirmation (perturb one physics constant by 1%, confirm the checker fails). This is
   the gate for every fidelity number the report prints.
9. **One-graph against separate-graphs against eager, over 3 iterations, per-parameter deltas.**
   The one-graph mode is the shipped path and currently has no equivalence test of any kind.
10. **A behavioural gate for TF32.** Three seeds at C=8 for a fixed short run, comparing the median
    final extrinsic reward and median coverage against a cached fp32 reference distribution, plus
    the direct measurement of the RND bonus floor described in finding 8.
