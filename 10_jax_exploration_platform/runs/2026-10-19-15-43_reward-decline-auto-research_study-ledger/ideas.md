# Ideas (20 planned), status

| id | batch | idea | run folder | outcome |
|---|---|---|---|---|
| E1 | 1 | episode ends at the goal, +1 once | `2026-10-19-16-46_reward-decline-batch-1_large-cap400_rnd-weights-1-10-100-1000_learning-rate-1e-3-annealed-over-1000M_steps-50M_seeds-0-127_control-e1-e2-e3-a1-a2-a3-a4-a5-a6-a7` | holds 0.91 at weights 1 and 10 but ends with 53-59% of episodes reaching the goal (control peak 67.5%); weights 100 and 1000 do not learn: not fixed |
| E2 | 1 | episode ends at the goal; 0 at the goal, -1 elsewhere | `2026-10-19-16-46_reward-decline-batch-1_large-cap400_rnd-weights-1-10-100-1000_learning-rate-1e-3-annealed-over-1000M_steps-50M_seeds-0-127_control-e1-e2-e3-a1-a2-a3-a4-a5-a6-a7` | holds 0.92-0.94 at weights 1-100; ends with 36-50% reaching the goal: not fixed |
| E3 | 1 | episode ends at the goal; pays remaining steps of the cap at once | `2026-10-19-16-46_reward-decline-batch-1_large-cap400_rnd-weights-1-10-100-1000_learning-rate-1e-3-annealed-over-1000M_steps-50M_seeds-0-127_control-e1-e2-e3-a1-a2-a3-a4-a5-a6-a7` | holds 0.58-0.89: not fixed |
| A1 | 1 | intrinsic weight decayed linearly to 0 | `2026-10-19-16-46_reward-decline-batch-1_large-cap400_rnd-weights-1-10-100-1000_learning-rate-1e-3-annealed-over-1000M_steps-50M_seeds-0-127_control-e1-e2-e3-a1-a2-a3-a4-a5-a6-a7` | falls more than the control (0.16-0.51): not fixed |
| A2 | 1 | intrinsic weight x (1 - running goal-reaching rate) | `2026-10-19-16-46_reward-decline-batch-1_large-cap400_rnd-weights-1-10-100-1000_learning-rate-1e-3-annealed-over-1000M_steps-50M_seeds-0-127_control-e1-e2-e3-a1-a2-a3-a4-a5-a6-a7` | falls (0.16-0.37): not fixed |
| A3 | 1 | clipped PPO, 4 epochs x 4 minibatches | `2026-10-19-16-46_reward-decline-batch-1_large-cap400_rnd-weights-1-10-100-1000_learning-rate-1e-3-annealed-over-1000M_steps-50M_seeds-0-127_control-e1-e2-e3-a1-a2-a3-a4-a5-a6-a7` | barely learns at lr 1e-3 (best final 0.49; log std to -2.6 in 2M steps): not fixed |
| A4 | 1 | entropy bonus 0.01 | `2026-10-19-16-46_reward-decline-batch-1_large-cap400_rnd-weights-1-10-100-1000_learning-rate-1e-3-annealed-over-1000M_steps-50M_seeds-0-127_control-e1-e2-e3-a1-a2-a3-a4-a5-a6-a7` | highest level (peaks ~90, best final 76.4) but falls (0.55-0.80); log std grows to +5: not fixed |
| A5 | 1 | task reward / max(1, running std of its discounted sum) | `2026-10-19-16-46_reward-decline-batch-1_large-cap400_rnd-weights-1-10-100-1000_learning-rate-1e-3-annealed-over-1000M_steps-50M_seeds-0-127_control-e1-e2-e3-a1-a2-a3-a4-a5-a6-a7` | falls (0.12-0.79), lower level: not fixed |
| A6 | 1 | extrinsic discount 0.99 | `2026-10-19-16-46_reward-decline-batch-1_large-cap400_rnd-weights-1-10-100-1000_learning-rate-1e-3-annealed-over-1000M_steps-50M_seeds-0-127_control-e1-e2-e3-a1-a2-a3-a4-a5-a6-a7` | falls (0.44-0.85): not fixed |
| A7 | 1 | gradient clip per network | `2026-10-19-16-46_reward-decline-batch-1_large-cap400_rnd-weights-1-10-100-1000_learning-rate-1e-3-annealed-over-1000M_steps-50M_seeds-0-127_control-e1-e2-e3-a1-a2-a3-a4-a5-a6-a7` | peak 43 but falls (0.14-0.59): not fixed |
| A8 | screening | learning rate annealed to zero over the run's own 50M steps (the screening's large-maze RND unit: batch 1's control program, same 512 copies, seeds and weights, anneal over 50M instead of 1000M) | `2026-10-19-16-18_map-screening_...` | holds 0.98-0.99 at every weight; peaks at the end; best final 64.9 (weight 100) against the control's peak 31 |
| B1 | 2 | log std floor -0.5 | `2026-10-19-17-27_reward-decline-batch-2_...` | falls (0.32-0.67) though log std stays above -0.3; ball still slows: not fixed |
| B2 | 2 | log std floor -0.2 | batch 2 | falls (0.42-0.75): not fixed |
| B3 | 2 | log std floor 0 | batch 2 | falls (0.23-0.85); highest peak 43.5: not fixed |
| B4 | 2 | critic step 0.1 | batch 2 | barely learns (return below 3): not fixed |
| B5 | 2 | E2 + A7 | batch 2 | FIXED: holds 0.91-0.95 at every weight; weight 1000 ends with 74.1% of episodes reaching the goal (control peak 67.5%), speed 4.4-4.8 m/s |
| C1 | 3 | E2 + rate annealed within the run (A8) | `2026-10-19-18-11_reward-decline-batch-3_...` | FIXED: holds 0.99 at every weight; weight 1000 ends with 75% of episodes at the goal |
| C2 | 3 | A4 entropy 0.01 + A8 | batch 3 | FIXED, best of the study: holds 0.99-1.00 at every weight, final 100-109 (3.5x control peak), 91-97% of episodes at the goal |
| C3 | 3 | actor and critic see whitened observations | batch 3 | learns little (best final 8.4): not fixed |
| C4 | 3 | learning rate 3e-4 (anneal over 1000M) | batch 3 | holds 0.95-0.97 at weights 1-100 within 50M (final 64.7 at weight 100); peaks at the run's end, so a later fall is not ruled out |
| C5 | 3 | anneal over 100M (1e-3 -> 5e-4 within 50M) | batch 3 | holds only at weight 1000 (0.94, final 31.1); others 0.47-0.67 |

## Mechanism notes (screening diagnostics, partial, 2026-10-19 16:45 PT)

Values are means over copies, at the window of the copy-averaged peak and over the final 10%.

1. The fall happens without any bonus: medium map, cap 400, no bonus, episode return 79.7 at its
   peak, 47.9 in the final 10%. So the bonus is not needed for it.
2. The intrinsic part of the advantage is small: root-mean-square of beta x A_int is 0.01-1 for
   beta up to 100, against 2-11 for 2 x A_ext. Only beta 1000 makes it comparable.
3. The extrinsic critic fits badly: explained variance of the extrinsic value is mostly between
   -1 and -10; its predictions are in the hundreds (discount 0.999 with bootstrapping at the cap).
4. The joint gradient clip is active almost always and is set by the critic: critic gradient norm
   10-126 against actor about 1; clip scale 0.06-0.4.
5. Runs that fall lose policy spread and speed: no bonus, medium 400: log std -0.29 -> -0.94, mean
   speed 1.64 -> 1.31; oracle 1/sqrt(n), medium 400: log std -0.07 -> -0.42, speed 3.0 -> 2.5. Runs
   that hold keep log std near -0.15.
6. (complete small-map screening, 17:12 PT) On the medium map with cap 400 the fall
   comes with a WEAK bonus: final / peak 0.60 without a bonus, 0.48 for the oracle at weight 1, 0.65
   and 0.64 for RND at weights 1 and 10; RND at weights 100 and 1000 holds 1.00 and ends highest
   (172, 221). Medium cap 250: only RND weight 10 falls a little (0.86). U-maze: nothing falls.
   This points at exploration pressure (and with it the policy's spread) running out, not at the
   bonus pulling the agent away from the goal. Ideas that keep the spread (A4 entropy; a log-std
   floor) are the first candidates for batch 2.
7. (batch 1 scored, 18:10 PT) The log std falls in almost every run, including those that hold
   (A8 and E2 end near -0.5), so the spread alone does not explain the fall. Mean speed does separate
   them: falling runs slow from ~5 to 2-3 m/s; holding runs (E1, E2, A8) keep 4-5. Reading: the reward
   for staying inside the goal radius teaches braking, which spreads to the whole maze under a high
   learning rate; ending at the goal removes the braking incentive, annealing stops the drift.
8. (batch 2 scored, 18:49 PT) Log-std floors keep the spread but not the peak: the ball still
   slows in the runs that fall. Two fixes so far: A8 (anneal the rate within the run) and B5 (end at
   the goal with a -1 step cost, gradient clip per network).

## Phase 4 (confirmation), submitted 2026-10-19 19:28 PT

Control, A8, B5 and C2 (the best batch-3 result) with RND and oracle 1/sqrt(n), 128 paired seeds:
part A at learning rate 1e-3 for 100M steps (`2026-10-19-19-27_reward-decline-confirmation-part-a_...`),
part B at 1e-4 for 400M steps (`..._part-b_...`).
9. (phase 4 part A, oracle 1/sqrt(n) units, 100M steps, lr 1e-3, 20:26 PT) Control falls almost
   to zero (final/peak 0.04-0.05). A8 holds at weight 10 (0.95) but not at 1 (0.51: peak at 12M,
   while the rate is still ~0.9e-3). B5 falls (0.54-0.67): it does not carry over to the oracle bonus.
   C2 holds at weight 10 (1.00, final 92.5) and mostly at 1 (0.81). RND units still running.
10. (phase 4 part A complete, 21:52 PT; lr 1e-3, 100M) Control falls to 0.04-0.29. C2 holds at
   RND weights 100/1000 (0.94/0.99, finals 91.6/98.5) and oracle weight 10 (1.00, 92.5); 0.81-0.85 at
   small weights (dip at 20-40M, recovers as the rate falls). A8 holds at RND 100/1000 and oracle 10
   only (early peaks fall). B5 only delayed the fall (0.54-0.79 at 100M).
11. (phase 4 part B complete, 2026-10-20 00:48 PT; lr 1e-4, 400M) Control falls later and slower
   (RND weight 100 peak 95.4 at 170M, keeps 0.65; oracle keeps 0.69-0.86). B5 and C2 pass everywhere
   (0.98-1.00; C2 finals 103.6/105.8 with RND); A8 holds (0.97-1.00) but ends below the control's
   peak with RND (80.6-83.1 vs 95.4).

## Outcome

21 ideas tested (batches 1-3 plus A8) and confirmed in phase 4. Recommended for the platform's RND
runs: learning rate annealed to zero over the run's own budget with entropy coefficient 0.01 (C2).
Report: development document subsections 1.3-1.9 (summary in 1.9). Study complete; monitor deleted.
