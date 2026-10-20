# Ideas (20 planned), status

| id | batch | idea | run folder | outcome |
|---|---|---|---|---|
| E1 | 1 | episode ends at the goal, +1 once | `2026-10-19-16-46_reward-decline-batch-1_large-cap400_rnd-weights-1-10-100-1000_learning-rate-1e-3-annealed-over-1000M_steps-50M_seeds-0-127_control-e1-e2-e3-a1-a2-a3-a4-a5-a6-a7` | running since 16:48 PT |
| E2 | 1 | episode ends at the goal; 0 at the goal, -1 elsewhere | `2026-10-19-16-46_reward-decline-batch-1_large-cap400_rnd-weights-1-10-100-1000_learning-rate-1e-3-annealed-over-1000M_steps-50M_seeds-0-127_control-e1-e2-e3-a1-a2-a3-a4-a5-a6-a7` | running since 16:48 PT |
| E3 | 1 | episode ends at the goal; pays remaining steps of the cap at once | `2026-10-19-16-46_reward-decline-batch-1_large-cap400_rnd-weights-1-10-100-1000_learning-rate-1e-3-annealed-over-1000M_steps-50M_seeds-0-127_control-e1-e2-e3-a1-a2-a3-a4-a5-a6-a7` | running since 16:48 PT |
| A1 | 1 | intrinsic weight decayed linearly to 0 | `2026-10-19-16-46_reward-decline-batch-1_large-cap400_rnd-weights-1-10-100-1000_learning-rate-1e-3-annealed-over-1000M_steps-50M_seeds-0-127_control-e1-e2-e3-a1-a2-a3-a4-a5-a6-a7` | running since 16:48 PT |
| A2 | 1 | intrinsic weight x (1 - running goal-reaching rate) | `2026-10-19-16-46_reward-decline-batch-1_large-cap400_rnd-weights-1-10-100-1000_learning-rate-1e-3-annealed-over-1000M_steps-50M_seeds-0-127_control-e1-e2-e3-a1-a2-a3-a4-a5-a6-a7` | running since 16:48 PT |
| A3 | 1 | clipped PPO, 4 epochs x 4 minibatches | `2026-10-19-16-46_reward-decline-batch-1_large-cap400_rnd-weights-1-10-100-1000_learning-rate-1e-3-annealed-over-1000M_steps-50M_seeds-0-127_control-e1-e2-e3-a1-a2-a3-a4-a5-a6-a7` | running since 16:48 PT |
| A4 | 1 | entropy bonus 0.01 | `2026-10-19-16-46_reward-decline-batch-1_large-cap400_rnd-weights-1-10-100-1000_learning-rate-1e-3-annealed-over-1000M_steps-50M_seeds-0-127_control-e1-e2-e3-a1-a2-a3-a4-a5-a6-a7` | running since 16:48 PT |
| A5 | 1 | task reward / max(1, running std of its discounted sum) | `2026-10-19-16-46_reward-decline-batch-1_large-cap400_rnd-weights-1-10-100-1000_learning-rate-1e-3-annealed-over-1000M_steps-50M_seeds-0-127_control-e1-e2-e3-a1-a2-a3-a4-a5-a6-a7` | running since 16:48 PT |
| A6 | 1 | extrinsic discount 0.99 | `2026-10-19-16-46_reward-decline-batch-1_large-cap400_rnd-weights-1-10-100-1000_learning-rate-1e-3-annealed-over-1000M_steps-50M_seeds-0-127_control-e1-e2-e3-a1-a2-a3-a4-a5-a6-a7` | running since 16:48 PT |
| A7 | 1 | gradient clip per network | `2026-10-19-16-46_reward-decline-batch-1_large-cap400_rnd-weights-1-10-100-1000_learning-rate-1e-3-annealed-over-1000M_steps-50M_seeds-0-127_control-e1-e2-e3-a1-a2-a3-a4-a5-a6-a7` | running since 16:48 PT |

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
