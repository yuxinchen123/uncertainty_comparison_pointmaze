# Batch 15: the method on the ant U-maze, RK4 physics, development sweep

Written 2026-11-13 13:09 PT, before any run of the batch. The units run in the AntMaze folder
(`runs/2026-11-12-15-56_cleanrl-ppo-rnd-baseline_antmaze-*`, phase `method-sweep-rk4`), next to RND's sweep at the
same budget and seeds (phase `rnd-sweep-rk4`, 28 configurations per reward setting).

## Setting

MuJoCo Warp with RK4 at 0.01 s x 5 (the user's decision of 2026-11-13), 16 environments per policy, budget
B = 134,217,728 steps per policy (the plan's rule), development seeds 0 to 15, both reward settings.

## Arms (the method of PointMaze at three steps of the fast last layer)

| arm | options |
|---|---|
| divisor held from each policy's first arrival (step 0) | `reward_divisor=held_after_first_arrival` |
| fast last layer at step 0.03, with the hold | `fast_last_layer_step=0.03,reward_divisor=held_after_first_arrival` |
| fast last layer at step 0.1, with the hold | `fast_last_layer_step=0.1,reward_divisor=held_after_first_arrival` |

Configurations: learning rates 1e-4, 3e-4, 1e-2 x intrinsic weights 1/16, 1, 16, entropy 0.001, clip 0.1 (9 per
arm, each also in RND's sweep, so every arm configuration has RND at the same configuration and seeds).

## What the earlier runs showed

On the old physics (implicitfast, 0.005 s x 10) at 16.8 million steps and 16 environments per policy (batches 10
and 14; 16 seeds; against RND at the same configuration): step 0.1 with the hold ahead in 8 of 8 configurations
(+0.002 to +0.008 on a whole-curve mean of 0.02 to 0.05), step 0.03 ahead in 8 of 8 (+0.0005 to +0.005), the hold
alone level (1 of 8 ahead). Under RK4 the arrival rates of the same trained policies were within about 0.01 of the
old physics's for these configurations (the integrator notes). The budget is now 8 times longer, and the first
arrival comes at 0.3 to 1.3 million steps, under 1% of it.

## Predictions

1. **Hold alone:** level with RND at the same configuration (difference within 0.005) in at least 7 of 9
   configurations in each reward setting.
2. **Step 0.1 with the hold, reward 1 on arrival:** whole-curve mean above RND at the same configuration in at
   least 6 of 9 configurations; first-goal time sooner in at least 7 of 9.
3. **Step 0.03 with the hold:** between the hold alone and step 0.1 on both measures in most configurations.
4. **Best against best, reward 1 on arrival:** the best of the 27 method configurations is above the best of RND's
   28 by more than twice its standard error. I am unsure of this one: with a budget 8 times longer than the
   screens, an earlier first arrival weighs less in the whole-curve mean.
5. **Step cost:** the same signs as with reward 1 on arrival, with smaller differences.
6. **Runaways:** at most 5 fault-guard respawns per unit under RK4.

## What decides the next step

Each side's best three configurations by the whole-curve mean go to the first ant validation block (128 seeds,
seeds 10000 to 10127), as the user decided on 2026-11-13.
