## Purpose

Make the JAX trainer of `10_jax_exploration_platform` fast on PointMaze (Medium, Large) and AntMaze
(U-maze, Medium, Large) with many seeds and many hyperparameter settings in one process, for the
study "beat CleanRL RND on the training curve" (plan
`~/.claude/plans/2026-11-11-22-40_pointmaze-antmaze-speed-and-beat-rnd.md`, Part 3 and Part 4).

1. Measure where the time of one training iteration goes, in both families.
2. Try 30 speed ideas, each profiled in at most 15 minutes of wall time per card, on an H100 NVL
   first (serval05, or serval02 card 1), kept ideas then on an RTX 5080 and an RTX A4500.
3. Measure the best number of copies per process on those three cards and estimate it for every
   other card.

Times in this folder are Pacific (the machines run on Eastern).

## Key hyperparameters

| Parameter | Value |
|-----------|-------|
| trainer | CleanRL PPO-RND settings: clipped PPO, 4 epochs x 4 minibatches, 128-step rollouts, clip 0.1, learning rate 1e-4 annealed within the run, entropy 0.001, extrinsic and intrinsic coefficients 2 and 1, discounts 0.999 and 0.99, GAE 0.95 |
| predictor | trained on the whole batch (no 25% subsampling) |
| environments | PointMaze Medium and Large (400-step limit); AntMaze U-maze (700), Medium and Large (1,000); fixed start and goal; the episode ends at the goal |
| environments per policy | 4, 16 and 128 are measured; the baseline check chooses one per family |
| precision | 32-bit physics and networks, 64-bit running statistics, as in the platform today |
| time limit per idea | 15 minutes of wall time per card, startup and compilation included |
| accepted when | at least 5% more total steps per second at the production shape, and the idea's check passes |
| cards | H100 NVL (serval05 / serval02 card 1), RTX 5080 (nekomata), RTX A4500 (jaguar03) |

## Code and config changes

Listed per idea in `ledger.md` of this folder (what changed, risk class, check, gain per card, kept
or not). The trainer changes that the study needs before any run (per-copy settings, the two
reward settings, the episode log, checkpoint and resume, the AntMaze preset) are in the plan's
section 3.1 and are tested through `infrastructure/submit_tests.sh` (job ids in
`slurm/submitted_jobids_tests_sl5nw.txt`).

## Git state

Commit at folder creation: `1275b250` (pushed), working tree clean for the code.
