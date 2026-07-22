# Train run 3.2.3 — nonzero bias initialization and classic reward normalization

## Purpose

Turn two remedies from the run-3.2.2 initial-bonus analysis (development document,
`sec:train-run-3-2-2`) into training sweeps, each measured against the unmodified RND benchmark:

1. **Reward normalization (arm N1).** The classic RND paper divides the intrinsic reward by a
   running standard deviation of the forward-filtered (discounted) intrinsic return. This codebase
   had never enabled any intrinsic-reward normalization (only observation whitening). Arm N1 is the
   benchmark configuration (Adam at PyTorch defaults, mse readout, zero biases) with the new
   `--rnd_reward_norm True` switch, sweeping beta over 9 decades `1e-4 .. 1e4`.
2. **Nonzero bias initialization (arms O1×init, O3×init).** The 2026-07-10 bias ablation
   (`development_document/code/2026-07-10-21-05_rnd-init-bonus-bias-ablation/`) showed the untrained
   RND bonus "bowl" is an artifact of both nets' biases starting at exactly 0, and that nonzero bias
   initializations flatten it (free-point max/min ratio ~2,900 → 5.2 / 1.7 for
   `pytorch_default` / `normal_0.5`). These arms train with the new
   `--rnd_bias_init` switch — same keyed draws as the ablation, so a run's initial bonus field
   equals the ablation's measured field for the same seed — crossed with Adam and SGD-1/t.

The two remedies are deliberately not crossed (N1 keeps zero biases; the init arms keep the
normalization off). The reference is configuration **C2** — the RND benchmark re-measured on 100
fresh seeds in run 3.2.2: final training-episode reward mean **34.50**, SE 3.13, success 0.69,
n=100. C2 is not re-run here.

## Key hyperparameters

| Parameter | Value |
|-----------|-------|
| algorithm | `rnd_next_state` (all arms) |
| env | `PointMaze_Large-v3`, goal `top_right`, `env_max_episode` 400, no termination wrapper |
| agent | SB3 SAC `MlpPolicy` defaults, lr 3e-4, buffer 1e6, batch 256, gamma 0.999, device cpu |
| total_timesteps | 1,000,000 |
| eval_freq / n_eval_episodes | 50,000 / 100 (`eval_standalone` False — scored on the training-episode reward) |
| log_distance | False |
| RND fixed | `rnd_output_dim` 128, `rnd_obs_norm` True, `rnd_distance` mse, `n_predictors` 1, orthogonal weights gain sqrt(2), no gradient clipping |
| Arm N1 (9 configs) | Adam / mse readout / `rnd_bias_init` zero / `rnd_reward_norm` True (`rnd_reward_norm_gamma` 0.99) / beta in {1e-4 .. 1e4} (9 decades) |
| Arm O1×init (20 configs) | Adam / readout l2 only / `rnd_bias_init` {pytorch_default, normal_0.5} / beta in {1e-5 .. 1e4} (10 decades) / reward norm off |
| Arm O3×init (120 configs) | SGD-1/t / readout l2 only / same 2 bias schemes / `rnd_sgd_eta0` {1e-3, 1e-2, 1e-1} / `rnd_sgd_t0` {1e3, 1e4} / same 10-decade beta / reward norm off |
| seeds | `a_seed` 0–99 (100-seed cap; adaptive allocation, 10-seed decision floor) |
| grid | 149 configurations × 100-seed cap = 14,900 queue entries, seed outermost. Revised pre-launch 2026-07-11 (user decision): the init arms run the l2 readout only and two bias schemes (mse cells and normal_1.0 dropped from the original 429-config plan) |
| pruning bar | 34.50 = C2's fresh-seed mean, frozen at first controller start by recomputing from run-3.2.2's C2 records (`../2026-07-07-23-10_run_3_2_2_validate_*/data/2026-07-07-23-15_validate/local/`) |
| stopping rule | stop a configuration when n >= 10 finished seeds and mean + 2.576·s/sqrt(n) < bar (as run 3.2.1) |

## Code and config changes

- `train.py` + `src/rnd_exploration/methods/{__init__,rnd}.py` (uncommitted at launch): new
  switches `--rnd_bias_init` (zero | pytorch_default | normal_<sigma>; overwrites both nets' two
  Linear biases with sha256-keyed draws — key parts `(a_seed, "bias-uniform"/"bias-normal",
  net_name, layer_idx)`, identical to the bias ablation; weights never touched),
  `--rnd_reward_norm` (classic-RND normalization: `observe()` on the buffer's add-time hook
  advances the per-env forward filter and the running std in time order; `compute()` divides the
  readout by the current std; eval rollouts never move the statistics; the predictor's mse training
  loss stays unnormalized), and `--rnd_reward_norm_gamma` (default 0.99, the paper's intrinsic
  discount). All three recorded in the per-run JSON for RND runs. Defaults reproduce the historical
  behavior bit-for-bit; covered by new unit tests in `tests/methods/` (166 tests pass).
- `slurm/` scripts adapted from run 3.2.1 (same worker/queue/controller design): new config grid
  (149 configs, N1 → O1×init → O3×init order, seed outermost), controller bar recomputed from
  run-3.2.2 C2 records instead of run-3.1.1, seed cap 100.
- Development document: run 3.2.2 promoted from a paragraph to `\subsubsection` (table of
  contents); new `\subsubsection` for run 3.2.3 with the per-arm hyperparameter table (Table 48).

## Git state

Commit: `55eb6bb3e21415e309deda48418d48dca3cc524e` (branch `Use-RLexplore-RND`)

Working tree dirty at launch — the run-3.2.3 switches and this run folder are uncommitted; also
in-flight: the `visit_count_decay` oracle knob (run-3.2.2 follow-up), doc edits, and the untracked
run-3.2.1/3.2.2 analysis folders. Snapshot of the training code as launched: `code/` in this
folder.

```
## Use-RLexplore-RND...origin/Use-RLexplore-RND
 M 07_reconstruction/development_document/main.tex
 M 07_reconstruction/src/rnd_exploration/methods/__init__.py
 M 07_reconstruction/src/rnd_exploration/methods/rnd.py
 M 07_reconstruction/tests/methods/rnd/test_rnd.py
 M 07_reconstruction/tests/methods/test_registry.py
 M 07_reconstruction/train.py
 (plus doc/rule edits and untracked analysis folders unrelated to the training path)
```

**Frozen bar (controller, 2026-07-11T00:41:37):** 34.5001 (run-3.2.2 C2 adam/mse/beta=100, n=100, SE=3.125; the validation table reports it as 34.50).
