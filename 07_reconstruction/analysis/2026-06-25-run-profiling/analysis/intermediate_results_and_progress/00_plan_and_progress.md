# Profiling + optimization — plan and progress log

Goal: make the SAC + intrinsic-bonus training loop run faster. Profile head-by-head, propose +
implement improvements (small ones behind on/off switches in the main repo; bold rewrites in a separate
codebase presented for a merge decision), measure local + end-to-end speedup on >=12 nodes (user has
24-36, so run multiple conditions in parallel; baseline vs improved on the SAME node to remove node-type
noise), write tests for each, report in analysis/analysis.md.

## Environment facts (2026-06-25)
- Python env `exploration` (torch 2.10.0+cu128 -> torch.compile available; numba 0.61 installed;
  jax / sbx / cython NOT installed -> JAX path needs an install, the framework agent will weigh it).
- Reservation `sl5nw_151` active (jaguar03/puma01); many idle 32-CPU allowlist nodes.

## Hot path (read from code) and initial profile (cProfile, 2k steps, 1 process, 2 cpu)
- ~20-30 fps per 2-cpu worker. Dominated by:
  - `torch._C._EngineBase.run_backward` 17.5s / 8000 calls  -> SAC critic+actor (+RND predictor) backward.
  - `torch._C._nn.linear` 12.3s / 84000 calls               -> all MLP forwards (256x256 nets).
  - `adam._single_tensor_adam` 2.2s / 8000                  -> optimizer steps.
  - SB3 `utils.zip_strict` 1.9s / 28000                     -> polyak target update (pure-Python overhead).
  - torch sqrt/relu/mul_/lerp_ ~1s each.
- The SAC core (backward + linear on the critic/actor) DOMINATES; the RND intrinsic path is a smaller
  slice; SB3 polyak (zip_strict) is pure-Python overhead.

## Candidate optimizations identified from the code (to confirm with research + profile)
Small / safe (on/off switch in main repo, exactly or numerically equal):
1. `buffers/vector_intrinsic_replay_buffer.py sample()` -- the reward arithmetic round-trips
   torch->numpy->torch (`to_numpy_flat` x2 + final `to_tensor`). Do it entirely in torch
   (`rewards = batch.rewards + beta*intrinsic`). EXACTLY equal. [local win on buffer overhead]
2. `methods/rnd.py` compute() + update() called back-to-back on the SAME samples -- each re-extracts +
   re-normalizes features + runs the target forward. Share the feature/target work. [local win]
3. obs_rms update does `x.detach().cpu().numpy()` every step -- batch it / keep in torch.
4. Adam `set_to_none=True`; enforce float32.
Tuning (changes dynamics or empirical -- benchmark per node):
5. torch.set_num_threads (1 vs 2 vs 4; memory says >4 slower -- verify).
6. SAC train_freq=N / gradient_steps=N (batch updates; same total updates, less Python overhead).
Bold (separate codebase, present for merge decision):
7. torch.compile on the SAC nets; a lean custom off-policy loop (no SB3 callback/zip_strict overhead),
   batched updates, intrinsic folded into the batched forward; possibly JAX/sbx (needs install).

## Status
- [DONE] commit + push (37f775a) all run-2 work.
- [DONE] folder scaffold; harness `analysis/code/profile_train.py` (clean fps + optional cProfile).
- [RUNNING] 12-node baseline profiling (slurm jobs in slurm/submitted_jobids.txt; nodes adriatic01-06,
  lynx01-04, ai01-02). Each node: rnd_state (cProfile + fps), gt_position_velocity + rnd_elliptical (fps).
- [RUNNING] research workflow w0z1rdgjr (SB3/SAC, RND-impl, torch-CPU micro-opts).
- [RUNNING] research workflow wu0nx2qnz (fast-frameworks, RND-native/C++, rewrite-architect).
- [NEXT] collect baseline + proposals -> implement switches + a separate bold codebase -> benchmark
  baseline vs improved on 24-36 nodes (head-by-head + end-to-end) -> tests -> analysis.md.

## Slurm safety
Only ever `scancel` ids in `slurm/submitted_jobids.txt`; never `scancel -u`. Monitor every ~10 min.

## Update (baseline done, switches started) ~19:15 EDT
- Baseline fps across 12 nodes: gt_position_velocity 34.2 +/-0.1 ; rnd_state 30.4 +/-2.8 (rnd_elliptical pending).
- Confirmed hotspots (cProfile, 6k steps, adriatic01): run_backward 72.8s/24k + linear 51s/252k (SAC critic/actor
  + RND predictor) DOMINATE ~80%; adam 8.2s; zip_strict (polyak) 5.9s/84k; relu/mul_/sqrt/lerp_ ~3-5s.
- IMPLEMENTED switch #1 opt_torch_reward (buffers/vector_intrinsic_replay_buffer.py): torch-only reward combine,
  no numpy round-trip. TEST test_optimizations.py::test_torch_reward_matches_numpy_reward_bit_identical PASSES
  (torch.equal -> bit-identical). Wired through Config + build_sac + harness.
- Added SAC batching switches sac_train_freq / sac_gradient_steps (Config + build_sac), harness --threads /
  --interop_threads / --device. Benchmark infra: slurm/run_bench.slurm (all conditions on one node) + aggregate.py.
- WAITING on 2 research workflows (micro-opts w0z1rdgjr; bold directions wu0nx2qnz) to finalize the condition set
  (esp. torch.compile + the rewrite) before launching the 12-node CPU benchmark + the GPU bold runs.

## Update ~20:35 EDT — CLEAN results (methodology fix applied)
METHODOLOGY BUG CAUGHT + FIXED: aggregate.py was pooling 2-cpu profiling baselines with 4-cpu bench runs
under "baseline" (inflated ratios to ~+26%). Fixed: tag each JSON by _source (filename prefix), restrict
the A/B ratio to the 'bench' experiment (all 4cpu). CLEAN same-node ratios (rnd_state, 12 nodes):
- torch_reward +2.3% (bit-exact), polyak +2.1% (bit-exact), combined_exact +6.1% (bit-exact).
- tf8 (train_freq=8) +1.0% (changes dynamics -> not worth it).
- threads: 2 optimal; threads1 -22.8%, threads4 -10.6%.
- fast_sac_rnd vs SB3: ~+7% (rnd_state). SAC core VALIDATED: Pendulum return -165/-184/-191 (near-optimal).
- GPU (a100/h100/a4000) cuda: 63-80 fps absolute; vs the gpu-node's 2cpu CPU +66-133% but vs best 4cpu CPU
  (~47-53) only ~+20-50%, high variance; 1 GPU/run doesn't pack the sweep like CPU.
- C/C++ env ceiling: 1.4% (env.step 0.69ms measured).
HONEST TAKEAWAY: no huge single CPU win (SAC backward+linear compute core dominates ~80%); bit-exact
micro-opts give a free ~6% combined; JAX (sbx) measurement pending (likely the biggest lever).
PENDING: full benchmark completion (remaining~13) + sbx-vs-sb3 JAX number, then finalize analysis.md.

## DONE ~21:00 EDT — all benchmarks complete, analysis.md finalized
FINAL measured numbers (12-node same-node, all bit-exact unless noted):
- combined_exact (polyak_foreach + torch_reward): rnd_state +6.1%, gt +3.7%, elliptical +5.8%. FREE, merge-ready.
- LOCAL: polyak x11.5 (1558->136us); reward-combine ~neutral. threads: 2 optimal (1:-22.8%, 4:-10.6%).
- fast_sac_rnd rewrite (separate codebase): +6.8% RND / -11% gt; SAC core VALIDATED (Pendulum -165/-184/-191).
- JAX (sbx) = x2.53 (104.5 vs 41.4 fps Pendulum) -- THE big win. C/C++ env <=1.4% (env.step 0.69ms). GPU ~+20% vs best CPU.
- 11 tests pass. analysis.md complete: why/before/after/local/e2e/test per improvement. summary.json written.
SOLUTION (per user's "tell me"): small wins = on/off switches in main repo (default off, mergeable);
bold rewrite (fast_sac_rnd) + JAX(sbx) port = separate codebases for the user's merge decision.
NOT committed (left for user review). Slurm: all 30 ids tracked in slurm/submitted_jobids.txt; no blanket cancels; all jobs finished.

## JAX A/B + breakdown task (~21:45 EDT)
USER asks: (1) JAX end-to-end A/B on the EXACT run-2 task (PointMaze+RND), short runs, many seeds for stats;
(2) does JAX need GPU / is it faster on CPU; (3) a breakdown table at the top: run-2 100K total time + each
part's time/% (same for JAX).
GPU ANSWER: the 2.53x was measured with jax[cpu] -> JAX is faster on CPU, NO GPU needed (GPU helps more for
bigger nets, not required for the win).
RUN-2 PER-STEP BREAKDOWN (rnd_state, measured ms/step): RND compute+update 4.95 | polyak(zip_strict) 1.29 |
env.step(vec) 1.0 | SAC critic+actor+optimizer core = remainder. At ~30 fps (2cpu clean) total=33.3ms/step ->
100K = ~56 min. Shares ~ SAC core 78%, RND 15%, polyak 4%, env 3%.
PLAN: shared venv /p/rlprojects/RND/.venvs/jax_bench (jax[cpu]+sbx+gym-robotics+mujoco+torch) for 12-node A/B.
Trainer experiments/2026-06-25-21-40-jax-ab/sac_pointmaze_bench.py: sbx vs sb3, +/-RND (step-time torch RND),
reuses train.py exact env. A/B 40K steps, many seeds/12 nodes. Then profile sbx for the JAX breakdown table.

## JAX A/B COMPLETE ~20:55 EDT (12 nodes, no errors)
FINAL: full run-2 task (PointMaze+RND) sbx/SB3 = x1.659 +/-0.023 (36 pairs); pure SAC on PointMaze x2.00
+/-0.10 (12 pairs); Pendulum pure SAC x2.53. Dilution chain: 2.53 (light env) -> 2.00 (PointMaze env) ->
1.66 (+ torch RND). GPU NOT needed (all jax[cpu]). Breakdown table §0: run-2 100K=51min (SAC 76%/RND 16%/
polyak 4%/env 3%); JAX 100K=30min (SAC-JAX 67%/RND-torch 27%/env 6%) -> RND becomes the bottleneck.
analysis.md fully filled (0 placeholders). JAX trainer standalone in experiments/2026-06-25-21-40-jax-ab/.
ALL profiling+JAX work uncommitted (for user review). Slurm: all ids tracked, no blanket cancels, jobs done.

## TRAIN RUN 4 (fully-JAX) task started ~21:30 EDT
USER: fold RND into JAX (separate jax repo), run train run 4 = 3 algos x 200 seeds x 500K steps, JAX sweep
only (run-2 data exists), reservation+partitions per slurm rule, monitor every 10 min, plot after 100
seeds/algo. NEW LOGGING: at each eval log eval results + EVERY training episode since last eval (episode-
level, not step), no final-eval-special-case (final eval = same n_eval_episodes). Plot: run-2 eval (dashed)
+ run-4 eval (solid) + run-4 training-eval (darker same-color), same color per algo. Modify jax repo +
python repo + main.tex with SAME logic, subagent-reviewed (esp. main.tex).
PHASE 1 DONE: jax_rnd.py (fully-JAX RND, flax+optax+obs-norm) VALIDATED (seen intrinsic 3.76->0.02, novel
0.22 > seen -> correct novelty). run4_bench.py: sbx+JAX-RND = 76.6 fps vs SB3 35.3 (portal) = ~2.17x (torch
RND was 1.66x) -> folding RND into JAX is ~free, run-4 ~2.1x faster than run-2. Scope: only rnd_state needs
the JAX RND (gt visit-count is a cheap wrapper lookup; rnd_elliptical phi is frozen + cheap Mahalanobis).
NEXT: Phase 2 new logging convention across jax/python/main.tex.

## STATE @ ~22:00 EDT — run-4 foundation done; sweep is the overnight continuation
DONE + validated:
- jax_rnd.py: fully-JAX RND (rnd_state), validated correct (novelty), sbx+JAX-RND ~2.17x vs SB3 (portal). analysis.md §6.3 updated.
- Python logging convention CHANGED (canonical, live repo): TrainEpisodeStatsCallback.episode_history (one row
  per completed training episode); train.py _write_local_log saves train_episode_history; wandb_eval_logging
  final-eval special case REMOVED (n_ep = n_eval_episodes always). NOTE: python sweep is NOT re-run; this is
  for consistency + main.tex docs.
REMAINING (overnight-scale, resumable):
1. JAX run-4 trainer (experiments/2026-06-25-run4-fully-jax/): sbx.SAC + JAX-RND (rnd_state) + gt visit-count
   wrapper + rnd_elliptical (torch ok, cheap) + the NEW logging via SB3-compatible callbacks (sbx supports
   them). Must put extrinsic/intrinsic in info for TrainEpisodeStatsCallback; needs an eval env+callback.
   VALIDATE it learns (reward up) before the sweep.
2. main.tex Logging section: document the new episode-level convention (matching python+jax). Subagent-review.
3. run-4 sweep infra: build_queue (3 algos x 200 seeds, total_timesteps=500000), worker calling the JAX
   trainer, launch_queue with reservation+partitions per slurm-submission.md. Monitor every 10 min.
4. After 100 seeds/algo: plot = run-2 eval (dashed) + run-4 eval (solid) + run-4 training-eval past-n mean
   (darker same-color), same color per algo; main.tex Train-run-4 section. analysis.md final update.
5. .claude/rules/run-id-and-logging.md: update for the new episode-level convention.

## STATE @ ~22:20 EDT — SAMPLE-TIME corrected, sweep LAUNCHED, reviews running
USER CORRECTION (critical): run-4 must recompute the intrinsic at SAMPLE TIME (== run-2
VectorIntrinsicReplayBuffer), not step time; plus audit any other logic difference.
DONE:
- run4_train.py REWRITTEN for sample-time: SACWithIntrinsic subclasses sbx.SAC, recomputes
  reward = ext + beta*intrinsic on the SAMPLED batch at sbx's numpy hook + trains the predictor there
  (== VectorIntrinsicReplayBuffer.sample). Env wrapper (InfoIntrinsicWrapper) only computes the step-time
  intrinsic for the info (logging); buffer stores EXTRINSIC. All 3 algos validated end-to-end.
- PARITY AUDIT complete. Logic differences found + fixed: (1) step-time->sample-time, (2) learning_starts
  1000->100, (3) buffer_size 200000->1e6. All other knobs match run-2 (sbx defaults == SB3 defaults: lr 3e-4,
  batch 256, tau 0.005, train_freq 1, gradient_steps 1, ent_coef auto, gamma 0.999, net 256x256; RND lr 1e-3,
  out 128, n_pred 1, obs-norm, mse, feature=observations; env stack identical). Only intended diff: 500K vs 1M.
- Timing (2-CPU A/B, == sweep worker budget): single vs multi-thread XLA identical (266.6 vs 264.8s for 11k
  train + 1x100-ep eval). Projected ~2.2h/run for 500K+10 evals; run-2 ~9.7h@1M (~4.85h@500K) -> run-4 ~2.2x.
- run-4 folder train_runs/2026-06-25-21-54_run_4_fully_jax_sampletime_500K_3algo_200seed/: slurm/ (build_queue
  600 seed-outermost, worker.py runs run4_train.py under JAX venv, worker.slurm XLA-single-thread, launch_queue
  reservation-aware), code/ snapshot, experiment_background.md, analysis/ (comparison plot script + placeholder).
- SWEEP LAUNCHED: 64 jobs / 512 workers (reservation sl5nw_151 jaguar03+puma01 + cpu/gpu/nolim), all 64 ids in
  slurm/submitted_jobids.txt. Healthy: 424 running, 176 pending, 0 failed at launch. Persistent monitor every
  10 min (flags failures, my-jobs-alive, the >=100-seeds/algo plot milestone).
- main.tex: §Logging updated (5th group train_episode_history + final-eval special case removed), §Train-run-4
  added (parity table tab:trainrun4-parity + comparison figure). Compiles clean, tables no overflow.
- .claude/rules/run-id-and-logging.md updated (5 groups + no final-eval special case).
- 3 subagent reviews RUNNING: code parity, main.tex accuracy, cross-repo logging consistency.
REMAINING:
- Act on review findings (a parity MISMATCH would require fixing run4_train.py + restarting the sweep).
- At >=100 seeds/algo milestone: re-run make_comparison_curve.py (fills run-4 solid + training-eval lines),
  rebuild main.pdf, final analysis.md update with real 2-CPU per-run runtimes + the reproduction verdict.
- scancel ONLY ids in slurm/submitted_jobids.txt if needed (never scancel -u).

## STATE @ ~23:10 EDT — distance OFF by default, sweep-id mechanism, RELAUNCHED (sweep 23:05)
Reviews found: (1) obs_rms warm-up missing in JaxRND (parity bug, rnd_state only) -> FIXED
(jax_rnd.warmup_obs_rms draws 200 obs == run-2 _warmup_obs_rms). (2) run-4 omitted distance_history +
visit_counts/*. User decisions this turn: ADD distance support but make it OFF BY DEFAULT (a switch), in
python+jax+main.tex; ADD a sweep-id mechanism so legacy data never collides + reruns extend; CLEAN the old data.
DONE:
- DISTANCE OFF BY DEFAULT: shared DistanceLoggingCallback gains `enabled` (no-op when off). python train.py
  Config.log_distance=False + --log_distance + build_callbacks passes enabled. jax run4_train.py --log_distance
  (default 0) gates the callback. JaxRND now exposes the torch-style compute(samples)/update(samples) so the
  distance metric grids it; validated rnd_state distance, gt~0, elliptical empty (when on), all-empty (default off).
- visit_counts/* added to run-4 eval_history (eval_cb visit_count_env=train_pos). obs_rms warmup wired for rnd_state.
- SWEEP-ID MECHANISM (sweep-id convention in run-id-and-logging.md): each sweep = `<YYYY-MM-DD-HH-MM>_<tag>`;
  queue/<sweep_id>/{pending,running,done,failed}/, data/<sweep_id>/local/<id>.json, submitted_jobids_<sweep_id>.txt,
  manifest data/SWEEPS.md. build_queue.py --sweep_id (creates ALL 4 dirs -- first launch bug was missing
  running/done/failed -> empty-queue); worker.py reads SWEEP_ID; launch_queue.sh [tag] generates id + sbatch
  --export=ALL,SWEEP_ID. Tests: sweep-scoped ids, two-sweep isolation, 4-dir regression. analysis loads by sweep.
- CLEANED: old flat queue/data + the broken 23:00 sweep (which stalled on the 4-dir bug, 0 data).
- RELAUNCHED sweep 2026-06-25-23-05_obsrms-distoff: 64 jobs/512 workers, claiming healthily (0 failed). Monitor
  (sweep-scoped) every 10 min.
- main.tex: §Logging (distance off-by-default row+caption, sweep-id paragraph, data/<sweep_id>/local path),
  §Train-run-4 logging para (reward groups + visit_counts, distance off, sweep-id). Compiles clean 41pp.
- make_comparison_curve.py auto-detects the latest data/<sweep_id> dir.
REMAINING: at >=100 seeds/algo -> fill the comparison plot, rebuild main.pdf, analysis.md final update.

## STATE @ ~03:15 EDT — OUTCOME: run-4 does NOT reproduce run-2 (SAC backend), DOCUMENTED + stopped
The milestone data (≥125 seeds/algo) showed run-4 (sbx/JAX SAC) reaches only **0.39–0.52× run-2's (SB3/torch
SAC) eval reward @500K** for ALL 3 algorithms. Curves match to ~250K then diverge in late-training exploitation.
CAUSE = the SAC backend, NOT the intrinsic/sample-time: gt_position_velocity shares the IDENTICAL torch
visit-count bonus + same sample-time recompute as run-2, yet shows the same gap. Real per-run sweep speedup
1.55–1.81× (not the ~2.2× single-run FPS benchmark; sweep packs 8 workers/node on contended HW).
User decision: document the gap, then stop the run (chose NOT to debug the sbx-vs-SB3 SAC difference).
DONE:
- Stopped: scancelled the sweep's 64 ids (submitted_jobids_<sid>.txt); monitor stopped. Final 395/600, 0 failed.
- Final comparison plot regenerated (395 runs, MIN_SEEDS=30); shows run-4 solid well below run-2 dashed.
- main.tex §Train-run-4: rewrote the "faithful reproduction tracks" claim -> the actual finding; added result
  Table 34 (run-4 R, run-2 R, ratio 0.39-0.52, speedup 1.55-1.81x). Compiles clean 44pp, table no overflow.
- analysis.md §6.3: added the Train-run-4 result (the convergence re-validation FAILED); §1 caveat (FPS != training-quality parity).
- SWEEPS.md manifest: marked the sweep "stopped 395/600 — DOCUMENTED (run-4 under-trains)".
REMAINING (deferred, user did not request): diagnosing the sbx-vs-SB3 SAC difference (entropy/temperature,
target update, init) to make run-4 a faithful reproduction would be the next step IF a JAX reproduction is wanted.
