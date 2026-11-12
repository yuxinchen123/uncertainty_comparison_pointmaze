# Pixel Montezuma: count-free neural RND, 1B-step screening protocol
Date: 2026-11-11 PT. Status: **prototype implementation on this branch; small mathematical CPU smoke checked, full Atari PPO tests and GPU training not run or submitted**.
Branch: `research/montezuma-neural-rnd-1b-20261111`. Do not modify the completed baseline or the frozen `09_parallelization` directory.

## Objective and decision

Optimize **extrinsic episode score against agent steps** with the existing corrected JAXtari pixel-Montezuma PPO+RND trainer, changing only the RND initialization, bonus computation, and predictor learning rule. First compare with the **existing completed fast 2B-step JAX PPO+RND run** on exactly the same corrected environment, at *matched progress up to 1B*. Later validate a winner on ALE/EnvPool before claiming to beat the official CleanRL result.

**1B is a screen, not final proof.** Existing curves include learning after 1B; the published CleanRL reference rose from about 4,759 at 1B to 7,050 at 2B in the project's trailing-200-episode reanalysis. Seed 4 of the new JAX baseline first reached rooms 24/25 around 1.44B steps. Candidates promoted from 1B should resume from their checkpoint to 2B with no change to the LR schedule.

The target for now is `61_035` completed PPO updates, or `999_997_440` agent steps per independent seed; each update contains 128 environments times 128 rollout steps. **Keep `PixelPPOConfig.total_timesteps=2_000_000_000` even when stopping at `--target-updates 61035`.** That parameter is the PPO linear LR *annealing horizon*, not the stop instruction. The completed baseline used 122,070 updates (~2B agent steps).

## Frozen controls

- Environment: installed JAXtari Montezuma source `04ab9aec` with the corrected room-geometry sprite assets; max episode **27,000 agent steps** (108,000 Atari frames); sticky-action probability 0.25; 30 reset NOOPs; episodic life handling as in the completed baseline.
- Policy and value network: unchanged CleanRL-style pixel CNN and PPO equations; 128 envs per policy, rollout 128, epochs 4, minibatches 4; PPO LR 1e-4 with a **2B-step** linear anneal; GAE .95, gamma_ext .999, gamma_int .99; clip .1; ent .001; value coeff .5; max global norm .5; Adam eps 1e-5.
- Intrinsic weight 1, extrinsic weight 2; predictor update mask .25; observation-RMS warmup 50 rollouts; intrinsic-reward normalization ON with original conventions. Later retune the intrinsic weight only as an explicitly marked secondary study with the same search budget for every arm.
- Use the **same accepted optimized environment and learner backend** as the completed November 6–9 training, with identical deterministic GPU flags, input preprocessing, corrected assets and per-device settings. Test both original and modified methods with the same source commit, backend, game and checkpoint contract. Never use the broken older asset pack.
- Each independent PPO copy has independent policy, predictor, target and optimizer state. Keep the original `initialize_networks` seed naming for the baseline.

## Experimental dimensions

Let `f(x)` be the frozen RND CNN target, `g_theta(x)` the trainable CNN predictor, `r=f-g`, and `b(x)=0.5*||r||_2^2` its raw pixel-based intrinsic reward (rescale only as an explicit, logged ablation). No visit-count table, position ID, room ID, image-hash count, or step counter may enter the reward or predictor loss. A clock may only control an explicitly named optimizer schedule. Passive visit counts may be computed OFFLINE for diagnostics, never for control.

**Uniform start (U):** normalize the target output per image: `z(x)=f(x)/max(||f(x)||_2,epsilon)`, and zero-initialize only the last dense output head of the predictor so `g_theta0(x)=0`. Hence `||z(x)-g_theta0(x)||_2^2≈1` for every nonzero target feature vector. Preserve the 512-dimensional target *direction* and original CNN representation. Baseline remains exactly unchanged when U is off. Test simple U+original MSE+Adam as an ablation: uniformity alone was not enough in the earlier tabular/grid study. Report initial P90/P10, log-std and 1/99 percentile of raw bonus over fixed pixel probes. The first update of the zero head does not train the earlier trunk; diagnose delayed trunk learning.

**Decay candidate D1 (higher-order residual error):** use per-image `E(x)=||z(x)-g_theta(x)||_2^2` and predictor loss `L_p=E(x)^p/p`, p in {2,3}, with a separately tuned **constant-learning-rate SGD predictor optimizer**, leaving the PPO policy on its original Adam. For p=2 and p=3 these are respectively quartic and sextic losses in the residual norm. When U is off, use the original frozen output and an explicitly stated normalization for loss-scale comparability. Adam+same loss is an optimizer-control arm. This is only a *hypothesis* for slower/more consistent decay; Adam can cancel gradient-magnitude shaping, so changing the loss under Adam alone may not change effective step size. Pilot the SGD LR on short runs and assert finite loss and gradients before long runs.

**Decay candidate D2 (residual-adaptive regression target):** with normalized frozen target `z`, let `e=z-g`, `E=||e||_2^2`, and select a positive exponent `a`. Define `q=(1+E^(1/a))^(-a/2)` and a **stop-gradient** regression target `y=stop_gradient(z-q*e)`. Train the predictor toward `y` with separate SGD and preserve the ordinary raw RND bonus for the policy. With exact independent-state fits and initial `E=1`, this would give `E_(n+1)=E_n/(1+E_n^(1/a))^a=(1+n)^(-a)`. **For a shared image CNN, exact per-state fits and that law are not guaranteed**: generalization, interference and optimizer error can change the realized decay. Evaluate a in {0.5,1,2} initially, then refine around the extrinsic-return winner. This scheme uses only neural residuals, not counts. Verify the stop-gradient: otherwise it is not this algorithm.

**Optional D3 (representation/optimizer controls):** target-only layer normalization or normalized target features; predictor Adam with smaller constant LR; SGD+momentum or decoupled AdamW. This is a second wave, not part of the first controlled 2x2 test.

| Arm | Uniform start | Predictor rule | Purpose |
|---|---|---|---|
| B0 | No | Original target, MSE + Adam | archived fast JAX CleanRL-style baseline |
| U | Yes | MSE + Adam | isolate initial-bonus consistency |
| D1 | No | quartic/sextic + SGD | isolate decay/optimizer |
| U+D1 | Yes | quartic/sextic + SGD | test interaction |
| U+D2 | Yes | residual-adaptive target + SGD | transfer count-free schedule idea to pixels |
| U+D3 | Yes | one alternative normalizer/optimizer | optional controlled second wave |

Implementation on this branch: `src/exploration_platform/atari_pixels/rnd_variants.py` introduces U, power, and shrink objectives; opt-in fields in `config.py`, `networks.py`, `training.py`, and predictor-only SGD in `state.py` preserve the stock default. `tests/atari_pixels/test_neural_rnd_variants.py` contains CPU tests that must still be executed in the full repository environment. `scripts/atari_pixel_campaign/launch_neural_rnd_1b.py` constructs exact A4500/H100 run commands without submitting jobs, and `scripts/atari_pixel_campaign/analyze_neural_rnd_1b.py` compares archived baseline and candidate metrics on matched seeds.

If implementation requires a new configuration, add explicit defaults to `src/exploration_platform/atari_pixels/config.py`, behavior only under opt-in flags in `networks.py` and `training.py`, and focused tests in `tests/atari_pixels/`. Reuse the accepted `train_final.py` checkpoint/resume and selected GPU backend. Protect the stock B0 path with original numerical-parity tests; leave policy, intrinsic-reward RMS and frozen target disabled by default. Do not put an optimizer flag into `per_copy_scalar_settings`; optimizer choice changes the compiled program and should be one group per arm.

## Scientific diagnostics (offline only)

Construct a fixed, versioned bank of pixel observations from the completed baseline and new initial rollouts, stratified by observed room/background and by game phase; include start-room and deep-room states. Save the bank and a frozen observation-RMS snapshot. At initialization and every 20M steps:

1. Evaluate raw bonus across all bank images using a **frozen** RMS snapshot to isolate predictor-learning decay. Evaluate again with each run's **live** RMS to measure the extra effect of input-normalizer drift.
2. Record start P90/P10, min/max with epsilon protection, log-std, median and interquartile slopes of log bonus versus log independently logged exposure to similar images/visual contexts. Offline exposure estimates are diagnostic only; do not feed them to RND.
3. Plot per-probe bonus versus time and exposure, reward-normalizer variance, predictor learning rate, predictor loss/gradient norm, room visitation, PPO KL/entropy and extrinsic score. Avoid using only the global mean bonus: changing occupancy changes that mean even if no state's intrinsic reward decays.
4. Test frozen-target output variance and near-zero norms; check that initial U bonus is constant across very different images, no NaNs, and no collapse across rooms. A flat bonus alone is not a success criterion; it must still discriminate novelty after training.

## Evaluation and selection (pre-register)

- **Primary 1B outcome:** equal-weighted **mean across seeds** of the area under the *extrinsic episode-score versus agent-step* curve from 0 to 1B, using fixed 20M-step bins and per-finished-game mean in each bin. Explicitly define empty bins rather than silently dropping them.
- **Co-primary endpoint:** mean extrinsic score over completed games during the last 100M steps (900M–1B). Plot the complete 1B curve with per-seed traces and a seed-level bootstrap band.
- Secondary: capped first extrinsic-reward timing, first-room/unique-room progress, room visitation per update, best last-window *reported but never used to select*, PPO entropy/approx KL, episodic returns and throughput. Use same exact definitions for baseline and methods. No intrinsic reward as an extrinsic score.
- Compare paired seeds and compute a paired bootstrap 95% confidence interval of per-seed AUC and endpoint differences; do not treat 128 parallel games or 20M windows as independent seeds. A best-seed or best-window score is insufficient.
- Early candidate selection may be based on 4 development seeds **13–16** for 200–250M steps. Move the two most promising methods to 1B on **seeds 1–12**, whose archived 2B baseline can be truncated at 1B. Neither the final score nor the 2B official score is a substitute for the baseline's *matched 1B curves*.
- Advance a candidate only if its AUC and endpoint are both directionally better than B0, the seed bootstrap/paired differences support improvement, and no unacceptable increase in nonfinite-loss incidence, output collapse or per-GPU wall time. If inconclusive, report that rather than declare a win.
- After the specified 1B experiment, resume the best validated candidate(s) to 2B using the **same 2B anneal**, then evaluate on ALE/EnvPool with matched controls to support a published CleanRL comparison. JAXtari-recreated game and ALE are not automatically performance-equivalent.

## Compute and implementation safeguards

Existing measured final-run throughput: RTX A4500 11.4–12.1k agent steps/s at **one copy per GPU** (~23–24.4 h per 1B seed); RTX 5080 13.0–13.3k at **one** (~20.9–21.4 h per 1B seed); H100 NVL 31.4k across **four packed** copies (~35.4 h to advance all four seeds by 1B). These are **baseline measurements**, not guaranteed for new predictor objectives. A4500 two-copy packing was slower in earlier experiments; RTX 5080 multi-copy packing is not established; H100 8/16-copy attempts exceeded memory. Preserve device-specific configurations and reprofile each new method's speed and peak memory. Use only authorized UVA/MIT/Purdue allocations and correct account/reservation limits.

Jobs have **not** been submitted from this protocol; no GPU study results exist for the new candidate arms. Only a standalone local JAX CPU test of target normalization and residual-shrinkage gradients passed, not the full package/unit or end-to-end Montezuma tests. This GitHub connection permits repository work but not SSH/Slurm access to the GPU clusters. Before submitting, ensure: (i) new method correctness and baseline parity tests pass; (ii) JAXtari corrected sprite asset hashes match; (iii) per-method short GPU learning/speed smoke tests pass; (iv) the archived raw `metrics.jsonl` baseline is available to calculate true 1B paired AUC (the committed plots alone are insufficient); and (v) commands use the accepted locked GPU execution wrappers and never interfere with existing jobs.

## Reference records

- `10_jax_exploration_platform/runs/2026-11-06-01-14_pixel-montezuma_corrected-room-geometry_.../analysis/2026-11-09-02-46_final-training-results_all-12-copies_.../analysis.md` and plots: final 12-seed curves, metrics, rooms and hardware rates.
- `10_jax_exploration_platform/runs/2026-11-09-19-42_online-tabular-rnd-auto-research_.../STATE.md` and `analysis/analysis.md`: uniform-start/decay evidence, count-like tabular exact fits, novelty limits and four-case confirmations.
- `10_jax_exploration_platform/src/exploration_platform/atari_pixels/`: existing pixel PPO-RND source; `scripts/atari_pixel_campaign/train_final.py` supports raising `--target-updates` later without changing run identity.
- `08_cleanrl_ppo_rnd/train_runs/2026-08-05-22-30_cleanrl_run_2_ablation_.../analysis/analysis.md`: published CleanRL W&B run re-averaged with 200 episodes, 4,759 at 1B and ~7,050 at 2B.
