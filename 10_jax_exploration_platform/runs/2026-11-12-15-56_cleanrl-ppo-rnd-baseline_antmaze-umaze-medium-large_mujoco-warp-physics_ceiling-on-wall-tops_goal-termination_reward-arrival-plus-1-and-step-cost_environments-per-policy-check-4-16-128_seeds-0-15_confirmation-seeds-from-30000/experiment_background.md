## Purpose

The RND baseline on AntMaze of the PointMaze and AntMaze study (plan
`~/.claude/plans/2026-11-11-22-40_pointmaze-antmaze-speed-and-beat-rnd.md`, Part 5; protocol
`runs/2026-11-12-00-22_pointmaze-medium-large_antmaze-umaze-medium-large_beat-cleanrl-ppo-rnd-auto-research_*/program.md`).
The PointMaze baseline is in its own folder (`runs/2026-11-12-02-24_cleanrl-ppo-rnd-baseline_*`), because its code
copy is frozen at an earlier commit.

Phases, each added to this folder's queue when it is due:

1. **rollout-shape check (first):** CleanRL PPO-RND with 4, 16 and 128 environments per policy on the U-maze with
   reward 1 on arrival, the same 48 configurations and 16 seeds at each count, 16,777,216 steps per policy. The count
   with the best whole-curve mean at the same number of steps is kept for the three AntMaze maps. The same runs are
   the first reading of the learning rate, the entropy coefficient and the intrinsic weight on the ant, and of how
   many steps the U-maze needs.
2. **development sweeps** at the chosen count for the U-maze, Medium and Large maps and both reward settings; the
   budget of each case from the curves; RND's configuration for the whole-curve mean and, separately, for the mean
   capped first-goal time.
3. **confirmation** on fresh seeds from 30000 (256 per case), with the two reference rows.

Times in this folder are Pacific (the machines run on Eastern).

## Key hyperparameters

| Parameter | Value |
|-----------|-------|
| environment | AntMaze U-maze (700-step limit); fixed start and goal, no reset noise; the episode ends at the goal (termination), the step limit is a truncation |
| physics | MuJoCo Warp inside MJX 3.11.0 (`physics = "warp"`), 5 physics steps per environment step |
| ceiling | one plane on the wall tops (2 m) that only the torso meets, so an ant cannot cross a wall; decided in the session of 2026-11-12 and reported to the user, who may reverse it |
| reward | 1 on arrival, extrinsic discount 0.999 |
| learner | CleanRL PPO-RND values: clipped PPO, 4 epochs x 4 minibatches, 128-step rollouts, clip 0.1, learning rate annealed to zero over the run, extrinsic coefficient 2, intrinsic discount 0.99, GAE 0.95; the predictor trains on the whole batch |
| environments per policy | 4, 16 and 128 (the three arms of the check) |
| swept in every arm | learning rate 1e-4, 3e-4, 1e-3, 3e-3; entropy coefficient 0.001, 0.01; intrinsic weight 0.0625, 0.25, 1, 4, 16, 64 (48 configurations) |
| seeds | 0 to 15 |
| steps per policy | 16,777,216, in 256 logging windows |
| policies per process | 768 at 4 and at 16 environments per policy (3,072 and 12,288 environments); 128 at 128 environments per policy (16,384 environments, 6 processes) |
| measures | whole-curve mean of the training extrinsic return (primary) and mean capped first-goal time (secondary); final level over the last 10% of steps |
| fault record | per logging window and policy: steps with the torso within 0.5 m of the ceiling, and the largest torso speed |

The intrinsic weights below 0.0625 of the PointMaze first pass are left out: on PointMaze with reward 1 on arrival
no policy reached the goal with them.

## Code and config changes

Units are built by `scripts/maze_campaign/build_units.py` from the descriptions in `infrastructure/descriptions/`
and run by generic workers (`scripts/maze_campaign/worker.py`) from the frozen code copy `code/source_<commit>/`.
New since the PointMaze baseline's code copy: the Warp ant in the training command, the ceiling and the fault
record, the bonus options of batches 1 and 2 (all off here).

Measured before these runs (H100 NVL, U-maze, this learner, with the ceiling): 1.67 million steps per second at
16,384 environments (1.25 s per iteration, 17.8 GiB of card memory), 1.93 million at 32,768 environments (34.5 GiB),
0.89 million at 4,096 environments.

## Git state

Commit `30b2c609` (pushed) at folder creation; `source.json` names the frozen copy new jobs run.
