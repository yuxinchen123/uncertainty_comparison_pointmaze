# Train run 5 (set baseline) — experiment background

Reproducibility record (experiment-background convention). This run establishes a clean baseline:
three RND configurations at 300 fresh seeds each on PointMaze, including the first faithful
translation of the ORIGINAL RND (Burda et al. 2019) onto this task.

## Purpose

The project has two load-bearing Adam RND lines from train run 3 (the benchmark C2 without reward
normalization, and N1 with it) but has never run the original RND's settings. Run 5 adds an
"original-small" arm — the original RND's non-architecture settings on the project's train-run-3
RND-sized network — and re-runs C2 and N1 at 300 fresh seeds so the project has a concrete,
high-precision baseline. The fresh C2/N1 data also feed a per-run substitution test against the
run-3.2.x data (see analysis plan).

## The three arms (10 configs total)

Seeds 600–899 (300 fresh; disjoint from all prior runs, which used 0–599). config_key is the
8-field `optimizer|readout|eta0|t0|beta|bias|norm|variant` (the first 7 match run 3.2.x; the 8th
distinguishes the original-small arm).

1. **original-small** — the original RND's settings on the train-run-3 RND net size, swept over
   **8 betas** {1e-2, 1e-1, 0.5, 1, 10, 1e2, 1e3, 1e4} (0.5 is the paper-faithful ext2:int1 ratio):
   - `rnd_bonus_readout=mse_mean` (original's (1/m)·Σe²), `rnd_optimizer=adam`, `rnd_lr=1e-4`,
     `rnd_update_proportion=1.0` (update all — the original REPO's code default; the paper Table 5
     says 0.25), `rnd_activation=leaky_relu` (slope 0.2, the original's tf default),
     `rnd_predictor_extra_layers=1` (predictor deeper than target by one 128-wide block),
     `rnd_obs_warmup_mode=env_steps`, `rnd_obs_warmup_steps=6400` (128×50 random-agent steps),
     `rnd_reward_norm=True`, `rnd_reward_norm_gamma=0.99`, `rnd_output_dim=128`, orthogonal √2
     weights, zero bias, next-state input, obs-norm RMS+clip±5.
   - Network: target = state(4)→256→LReLU(0.2)→128; predictor = state(4)→256→LReLU(0.2)→128→
     LReLU(0.2)→128 (deeper, output dim 128 so the distillation error matches the target).
   - config_key `adam|mse_mean|-|-|<beta>|zero|rewardnorm|origsmall`.
2. **C2** — the run-3.2.2 benchmark, verbatim: adam, mse readout, β=100, zero bias, NO reward norm.
   config_key `adam|mse|-|-|100|zero|-|-` (= the old 7-field key + `|-`). 300 seeds, never pruned.
3. **N1** — the run-3.2.3 reward-norm winner, verbatim: adam, mse, β=1e4, zero bias, reward norm
   γ=0.99. config_key `adam|mse|-|-|10000|zero|rewardnorm|-`. 300 seeds, never pruned.

C2/N1 set NONE of the new run-5 knobs, so the (getattr) factory defaults reproduce the historical
adam/mse/relu/full-batch/200-space-sample behavior bit-identically (verified by the pre-edit
goldens), making the fresh C2/N1 data behaviorally identical to the run-3.2.x data.

## Beta-race pruning (original-small only)

The 8 original-small betas race under `slurm/prune_controller.py` (owner-only, run each monitoring
cycle). Rule (user, 2026-07-20): each beta gets ≥30 finished seeds, then a beta is pruned when its
one-sided 99% upper limit `mean + 2.576·s/√n` falls below the running best original-small mean
(bar = max mean over betas with n≥30); survivors run to 300. Score = last `train_history` row's
`train/mean_extrinsic_reward` (the 1M-step windowed training-episode reward), completed records
only. C2/N1 are never pruned. Decisions logged to `slurm/prune_decisions_<sweep_id>.jsonl`.

## Fixed stack (identical to run 3.2.4)

PointMaze_Large-v3, goal top_right, env_max_episode 400, no termination wrapper, total_timesteps
1e6, eval_freq 5e4, n_eval_episodes 100, eval_standalone False (scored on training-episode reward),
log_distance False, device cpu, SAC γ 0.999, SB3 MlpPolicy defaults.

## Code / environment

- Env: shared `/p/rlprojects/RND/.venvs/exploration/bin/python` (Python 3.11).
- Code changes for run 5 (train run 5 original-RND switches; all default to the historical behavior
  bit-identically — pre-edit goldens `tests/methods/rnd/goldens/preedit_v0.pt` confirm no RNG shift):
  `src/rnd_exploration/methods/rnd.py` (mse_mean readout, `rnd_activation` leaky_relu 0.2,
  `rnd_predictor_extra_layers`, `rnd_update_proportion` keep-mask), `methods/__init__.py` (expose
  `rnd_lr`, env-steps obs warmup on a FRESH throwaway env so training visit counts are untouched),
  `train.py` (Config fields + argparse + logging). Full suite: 186 tests pass.
- Reference implementations (RND paper + CleanRL + ExPLORe) fact sheets: in the writeup's dev-doc
  `code/` folder; the tex `\subsection{Train run 5}` presents the cross-implementation table.
- **Git commit (Phase-1 commit-before-submit rule):** `<FILLED AT COMMIT>` — the repo is committed
  and pushed before the sweep is submitted, so git reflects exactly the code that ran.

## Launch / monitoring

- Owner launch: `cd slurm && TIME=4-00:00:00 bash launch_queue.sh set-baseline` (builds the 3000-run
  queue, submits worker jobs to **cpu + nolim ONLY** — no reservation/gpu/gnolim per user directive).
- Monitoring: a ~20-minute owner loop refreshes `slurm/submitted_jobids_<sweep_id>.txt`, runs
  `requeue_orphans.py` (reclaim walltime-killed attempts) and `prune_controller.py` (beta race), and
  tops up cpu+nolim; anomalies noted in `infra_history.md`. Collaborators may add workers via
  `for_collaborator/` (cpu+nolim only); they never prune or requeue.

## Known failure mode

The sgd1t arms in run 3.2.4 had ~2/100 seeds NaN-diverge (SAC actor NaN under a large initial
bonus). The original-small arm uses Adam with reward normalization (bounded bonus), so this is less
likely, but any NaN-diverged seeds are counted separately in the analysis as config-caused (NOT an
infrastructure exclusion) and their failed markers stay in `failed/`.

## Analysis plan

`analysis/code/make_results.py`: (1) performance table (rows = original-small winning beta, C2, N1 +
old-C2/old-N1 references; final training-episode reward mean±SE, success fraction, n), (2) the
original-small beta-sweep table, (3) reward-over-training curves. Per-run substitution: for each old
run (3.2.2 & 3.1.1 for C2; 3.2.3 & 3.2.4 for N1), Welch t + Cohen's d + KS vs the run-5 fresh line
for the same config_key, pre-registered "substantially different iff p<0.01 AND |d|>0.3", plus an
executed-path code-diff audit; substitute (pool) exactly the old runs that pass both gates, with
concise provenance notes; old run folders left byte-untouched.
