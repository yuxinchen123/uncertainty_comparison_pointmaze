## Purpose

The final deliverable runs of the 09_parallelization task: train 8, 16, 32, 64, and 128
INDEPENDENT PPO+RND copies (independent seeds, each with its own networks, environments,
running statistics, and optimizer state) simultaneously on the serval05 H100, in both update
styles, to produce (a) the performance table (per-copy extrinsic-reward and maze-coverage
curves over 10.24M env steps per copy) and (b) the measured training throughput per copy
count for the before/after-optimization table in the unified report.

## Key hyperparameters

| Parameter | Value |
|-----------|-------|
| environment | batched GPU PointMaze (pointmaze/torch_env), map Large, start (7,1), goal (1,10), noise ±0.25, 400-step cap, continuing task, sparse reward |
| algorithm | PPO+RND per ppo/research/ppo_rnd_algorithm_spec.md (incl. §11 correction) |
| n_copies | 8, 16, 32, 64, 128 (one process per count) |
| n_envs per copy (N) | 4 |
| rollout length (T) | 128 (512 rows per copy per iteration) |
| iterations | 20,000 (= 10.24M env steps per copy) |
| update styles | style B = 4 epochs x 4 minibatches of 128; style A = one full-batch step |
| learning rate | 3e-4, linear anneal to 0, Adam eps 1e-5 (fused, capturable, tensor lr) |
| gamma ext / int | 0.999 / 0.99; GAE lambda 0.95; clip 0.2; vf coef 0.5; ent coef 0 |
| int coef / ext coef | 1.0 / 2.0; RND feature width 128, hidden 256, predictor +1 block |
| networks | actor 4-64-64-2 tanh + logstd; critic 4-64-64-(1+1); per-copy batched |
| execution | one-graph CUDA-graph capture (rollout+post+update), TF32, RND-target hoist, same-input GEMM packing |
| base_seed | 0 (per-copy keyed init and reset RNG; copy i identical across C) |
| history sampling | every 50 iterations (per-copy reward sum, intrinsic mean, coverage) |

## Code and config changes

Driver `09_parallelization/train_runs/run_final.py` (this repo), one child process per copy
count for honest peak-VRAM numbers, per-count JSON records (resumable: existing JSONs are
skipped on re-run). No wandb/tensorboard; sparse stdout logs (60 s) + per-count log files in
`logs/`. Trainer code unmodified from the committed state below.

## Git state

Commit `680c436` (branch Use-RLexplore-RND), working tree clean for 09_parallelization at
launch time. Repository: /p/rlprojects/RND.
