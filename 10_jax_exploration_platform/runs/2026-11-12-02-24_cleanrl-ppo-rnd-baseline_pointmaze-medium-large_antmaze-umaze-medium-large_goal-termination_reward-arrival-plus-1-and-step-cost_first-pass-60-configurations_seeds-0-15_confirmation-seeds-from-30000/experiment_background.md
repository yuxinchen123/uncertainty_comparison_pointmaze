## Purpose

The RND baseline of the PointMaze and AntMaze study (plan
`~/.claude/plans/2026-11-11-22-40_pointmaze-antmaze-speed-and-beat-rnd.md`, Part 5; protocol
`runs/2026-11-12-00-22_pointmaze-medium-large_antmaze-umaze-medium-large_beat-cleanrl-ppo-rnd-auto-research_*/program.md`):
CleanRL PPO-RND swept and then trained on many seeds in each of the five environments and both reward settings. Every
new method is later compared with these runs on the same seeds.

Phases, each added to this folder's queue when it is due:

1. **development, first pass:** 10 intrinsic weights x 3 learning rates x 2 entropy coefficients (60 configurations)
   x 16 seeds (0 to 15), per environment and reward setting, at a generous budget; the budget of each case is then
   fixed from the curves;
2. **development, second pass** around the chosen configuration (and the grid's extension when the chosen intrinsic
   weight is at an end of it);
3. **confirmation:** the chosen configuration on fresh seeds from 30000 (1,024 on PointMaze, 256 on AntMaze), plus two
   reference rows (CleanRL's untouched defaults; PPO without a bonus).

Times in this folder are Pacific (the machines run on Eastern).

## Key hyperparameters

| Parameter | Value |
|-----------|-------|
| learner | CleanRL PPO-RND values: clipped PPO, 4 epochs x 4 minibatches, 128-step rollouts, clip 0.1, learning rate annealed to zero over the run, extrinsic coefficient 2, discounts 0.999 and 0.99, GAE 0.95, value coefficient 0.5, joint gradient clip 0.5, 50 priming rollouts; the predictor trains on the whole batch |
| swept | intrinsic weight 4^-6 to 4^3 (10 values); learning rate 1e-4, 3e-4, 1e-3; entropy coefficient 0.001, 0.01 |
| environments | PointMaze Medium and Large (400-step limit); AntMaze U-maze (700), Medium and Large (1,000); fixed start and goal; the episode ends at the goal (termination); the step limit is a truncation |
| reward settings | 1 on arrival; or a cost of 1/limit per step with 0 on arrival; extrinsic discount 0.999 in both; both settings share a process on the same seeds |
| environments per policy | PointMaze: 4 (rollout-shape check of 2026-11-12); AntMaze: set after the ant's physics settings are decided |
| first-pass budget | PointMaze: 50,331,648 steps per policy, 192 logging windows |
| policies per unit | PointMaze: 256 (an RTX A4500 gives 2.4 million steps per second at 512 and at 960 policies, so smaller units on more cards finish sooner) |
| measure | whole-curve mean of the training extrinsic return; final level over the last 10% of steps |

## Code and config changes

Units are built by `scripts/maze_campaign/build_units.py` from the descriptions in `infrastructure/descriptions/`
and run by generic workers (`scripts/maze_campaign/worker.py`) from the frozen code copy `code/source_<commit>/`.
The AntMaze phases are added after the speed study has fixed the ant's physics.

## Git state

Commit `816c626b` (pushed) at folder creation; the frozen copy's commit is in `source.json`.
