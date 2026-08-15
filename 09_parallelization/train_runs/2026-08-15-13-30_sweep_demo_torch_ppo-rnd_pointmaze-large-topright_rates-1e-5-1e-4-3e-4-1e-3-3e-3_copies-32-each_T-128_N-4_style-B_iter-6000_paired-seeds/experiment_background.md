## Purpose

Demonstrate the learning-rate sweep added in round 3: one training run in which the copies are
split into groups, each group trained at a different learning rate, to show (a) that the groups
train independently and correctly and (b) that the sweep recovers a sensible best rate.

## Key hyperparameters

| Parameter | Value |
|-----------|-------|
| learning rates | 1e-5, 1e-4, 3e-4, 1e-3, 3e-3 |
| copies per rate | 32 (160 copies total) |
| seeding | paired — copy k of every group shares initial weights and environments |
| iterations | 6,000 (3.07M environment steps per copy) |
| rollout | T=128, N=4 (512 rows per copy per iteration), style B (4 epochs x 4 minibatches) |
| execution | production configuration: one-graph capture, compiled post body, TF32, compiled per-copy Adam |
| environment | batched GPU PointMaze Large, start (7,1), goal (1,10), 400-step cap |
| base seed | 0 |

## Code and config changes

Driver `train_runs/run_final.py` with `--sweep-rates` and `--copies-per-rate`. The sweep path
replaces torch.optim.Adam with the batched per-copy Adam described in
`ppo/torch_ppo/SWEEP.md`; everything else matches the uniform-rate campaign.

## Git state

Commit 98a45f3 plus the compiled-per-copy-Adam fix committed immediately after (see the
repository log around 2026-08-15 13:30). Repository: /p/rlprojects/RND.
