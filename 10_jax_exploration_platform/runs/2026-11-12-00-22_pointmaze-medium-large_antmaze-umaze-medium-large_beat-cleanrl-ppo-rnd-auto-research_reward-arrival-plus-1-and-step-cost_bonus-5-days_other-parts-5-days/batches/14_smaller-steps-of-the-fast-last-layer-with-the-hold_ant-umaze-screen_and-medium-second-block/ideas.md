# Batch 14: the smaller steps of the fast last layer with the hold, on the ant U-maze, and the second block on PointMaze Medium

Written 2026-11-13 06:27 PT, before any run of the batch. Bonus-only (program.md, days 1 to 5); existing options.

## What the earlier runs showed

1. **PointMaze Large with reward 1 on arrival: the fast last layer at step 0.03 with the first-arrival hold is
   ahead of RND on both validation blocks** (learning rate 2e-3, weight 2): +0.0195 [+0.0136, +0.0254] on the
   first block and +0.0153 [+0.0096, +0.0208] on the second, with the final level ahead (+0.004) and the first
   arrival after 118 to 128 thousand steps against 266 to 269. Step 0.01 is ahead too (+0.0148 and +0.0103).
2. **The step is the arm's own setting.** On PointMaze Medium a small step gives the hold alone (step 0.01: 0.9400
   against 0.9435 with reward 1 on arrival; −0.3470 against −0.3484 with the step cost) and step 0.1 costs return.
3. **Ant U-maze, step 0.1** (batch 10's screen, 16 development seeds): with 128 environments per policy the fast
   last layer is behind RND in all 8 configurations, with or without the hold (−0.005 to −0.018), and arrives
   later; with 16 environments per policy it was ahead in all 8 in the partial reading (units not complete yet).

## Units

1. **Ant U-maze screen** (phase `bonus-arms-screen` in the AntMaze baseline folder; 16 development seeds,
   16,777,216 steps, the same configurations as batch 10's screen and as the check's RND units): the fast last
   layer with the first-arrival hold at steps 0.03 and 0.01, with 128 and with 16 environments per policy. Four
   units of 8 configurations.
2. **PointMaze Medium, second validation block** (seeds 20000 to 20255) for step 0.01 with the hold at the
   configurations fixed on the first block: reward 1 on arrival (6e-3, 0.01, 4) and step cost (6e-3, 0.01, 2).
   RND and the hold alone have their units on this block.

## Predictions

1. **Ant, 128 environments per policy:** the smaller the step, the closer to RND: at step 0.01 the whole-curve
   mean is within 0.003 of RND's at the same configuration in at least 6 of 8 configurations; at step 0.03 it is
   still behind in most. No step is ahead of RND at this count.
2. **Ant, 16 environments per policy:** step 0.03 is ahead of RND at the same configuration in at least 6 of 8
   configurations, by less than step 0.1; step 0.01 is level with RND.
3. **PointMaze Medium, second block:** step 0.01 with the hold is within 0.005 of the hold alone on the
   whole-curve mean in both cases, and ahead of RND with reward 1 on arrival.
4. **Fault-guard respawns** stay at a few per unit.
