# Reward-decline study, phase 4: confirmation at learning rate 1e-4 for 400M steps

## 1. Purpose

Confirm, at a longer budget and with both bonuses, the fixes the batches found on the large maze at
learning rate 1e-3 for 50M steps: A8 (learning rate annealed to zero within the run), B5 (episode
ends at the goal with a -1 step cost, gradient clip per network) and C2 (entropy coefficient 0.01
with the rate annealed within the run), against the control (rate annealed over 1000M steps, so
near its start value throughout, as in the runs where the fall was first seen). Part A repeats the
batches' learning rate 1e-3 at twice the budget; part B tests learning rate 1e-4, where the
1000M-step runs peaked at 170-260M steps and then fell.

## 2. Key hyperparameters

- Large PointMaze, cap 400 (B5: episodes also end at the goal, -1 per other step).
- Learning rate 1e-4; 400M steps per copy; RND weights 100 and 1000 and oracle 1/sqrt(n) weights 1 and 10 (2 x 128 copies each; the weights that led the 1000M-step runs at 1e-4); seeds 0-127 paired across ideas.
- Control, B5: rate annealed over 1000M steps; A8, C2: annealed over this run's own 400M steps.
- Full-batch updates; records every 2000 iterations with diagnostics and coverage.

## 3. Code and config changes

None beyond batches 1-3 (commit recorded below). Queue built by code/build_queue.py; workers
code/worker.sh from both accounts; commands in code/submission_commands.md.

## 4. Git state

Recorded at submission below.
