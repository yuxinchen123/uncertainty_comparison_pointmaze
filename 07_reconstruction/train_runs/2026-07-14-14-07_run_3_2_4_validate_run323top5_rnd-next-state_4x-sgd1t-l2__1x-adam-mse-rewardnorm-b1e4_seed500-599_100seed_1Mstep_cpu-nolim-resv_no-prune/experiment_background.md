# Train run 3.2.4 — fresh-seed validation of run 3.2.3's top-5 racing configs

## Purpose

Out-of-sample validation of the 5 best-performing configurations from run 3.2.3 (the
init+reward-normalization sweep), measured on 100 FRESH seeds (`a_seed` 500–599, disjoint from
every prior run's 0–499). Run 3.2.3 selected these as the argmax over 149 configs at ~40–50 seeds
each, so their in-sweep means are upward-biased by selection (winner's curse). This run re-measures
them on unseen seeds with NO pruning — every config runs all 100 seeds — exactly as run 3.2.2 did
for the earlier SGD-1/t winners. The reference is the run-3.2.2 C2 benchmark (Adam / mse / β=100 /
zero-bias): $\bar R$ = 34.50, SE 3.13, n = 100.

## Key hyperparameters

| Parameter | Value |
|---|---|
| algorithm / env | `rnd_next_state` on `PointMaze_Large-v3`, goal top_right, 400-step episodes |
| agent | SB3 SAC MlpPolicy defaults, lr 3e-4, buffer 1e6, batch 256, γ 0.999, device cpu |
| total_timesteps | 1,000,000; eval_freq 50,000; n_eval_episodes 100; eval_standalone False (scored on training-episode reward) |
| RND fixed | rnd_output_dim 128, rnd_obs_norm True, rnd_distance mse, n_predictors 1 |
| seeds | `a_seed` 500–599 (100 fresh, disjoint from all prior runs) |
| pruning | NONE — validation run; every config runs all 100 seeds |
| grid | 5 configs × 100 seeds = 500 runs, seed-outermost ids |
| placement | open cpu partition + nolim + reservation sl5nw_151 (jaguar03 + puma01); no gpu allowlist, no gnolim (user request 2026-07-14) |

The five configs (run-3.2.3 mean rank; config_key = optimizer|readout|eta0|t0|beta|bias|norm):

1. `sgd1t|l2|0.1|10000|1|normal_0.5|-`  — SGD-1/t, l2, β=1, η0=0.1, t0=1e4, normal_0.5 bias (3.2.3 mean ~48.0)
2. `sgd1t|l2|0.01|10000|0.1|pytorch_default|-`  — SGD-1/t, l2, β=0.1, η0=0.01, t0=1e4, pytorch_default (3.2.3 mean ~44.1)
3. `adam|mse|-|-|10000|zero|rewardnorm`  — Adam, mse, β=1e4, zero-bias, reward-norm on (γ=0.99) (3.2.3 mean ~40.7)
4. `sgd1t|l2|0.01|10000|1|normal_0.5|-`  — SGD-1/t, l2, β=1, η0=0.01, t0=1e4, normal_0.5 (3.2.3 mean ~39.8)
5. `sgd1t|l2|0.1|10000|1|pytorch_default|-`  — SGD-1/t, l2, β=1, η0=0.1, t0=1e4, pytorch_default (3.2.3 mean ~38.4)

## Code and config changes

- No source changes vs run 3.2.3 — same `train.py`, same switches (`rnd_bias_init`,
  `rnd_reward_norm`, `rnd_reward_norm_gamma`, sgd1t schedule). Run through the SHARED canonical
  env `/p/rlprojects/RND/.venvs/exploration/bin/python`.
- `slurm/build_queue.py` (this run): the 5 config keys copied verbatim from run 3.2.3, seeds
  500–599, no controller, no pruning. worker.py / requeue_orphans.py copied from run 3.2.3.
- `slurm/launch_queue.sh`: cpu + nolim + reservation buckets only (no gpu allowlist / gnolim);
  no controller submission (validation = run all seeds).

## Git state

Commit: `55eb6bb3e21415e309deda48418d48dca3cc524e` (branch Use-RLexplore-RND). Working tree dirty (the run-3.2.3 switches and this
run folder are uncommitted, as with prior runs this session). Training code is the shared repo;
snapshot in `code/` deferred (identical trainer to run 3.2.3, whose `code/` snapshot applies).
