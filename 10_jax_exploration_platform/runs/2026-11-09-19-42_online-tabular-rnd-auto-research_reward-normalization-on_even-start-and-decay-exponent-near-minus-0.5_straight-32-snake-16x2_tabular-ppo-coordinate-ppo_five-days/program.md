# Online tabular-RND auto research: the fixed protocol

Written 2026-11-09 20:00 PT, before any RL unit of the auto research ran. Times are Pacific. Plan:
`~/.claude/plans/2026-11-09-19-03_tabular-rnd-extension-and-online-tabular-rnd-auto-research.md`. Deadline: 2026-11-14
20:00 PT.

## Goal

Find a new method that improves online tabular-RND (RND whose networks read a one-hot cell vector) with reward
normalization on. The improvement must come from two properties together:

1. **An even start:** every cell starts with about the same bonus.
2. **A decay exponent near −0.5:** each cell's bonus falls like $N^{-1/2}$ in that cell's own visits $N$.

The result is meant for a paper. So the method must be new on both points: a re-implementation of a published RND
variant, or of a method that claims to beat RND, does not count (user, 2026-11-09).

## The four verification cases

| case | map | policy |
|---|---|---|
| 1 | straight grid, 32 cells | tabular-PPO |
| 2 | straight grid, 32 cells | coordinate-PPO |
| 3 | snake grid, width 16, 2 turns (50 cells) | tabular-PPO |
| 4 | snake grid, width 16, 2 turns (50 cells) | coordinate-PPO |

PPO, the policies, the maps, the budget $B = 2^{24}$ transitions, the evaluation and the seeds follow the fixed-settings
study (`runs/2026-11-07-20-03_fixed-…`).

## The baseline

The baseline is online tabular-RND with state normalization off and reward normalization on, at CleanRL settings, in
the extension run `runs/2026-11-09-19-40_fixed-…-extension_…`.
- It is chosen by the study's own protocol: 52 configurations on development seeds 0-15.
- Its chosen configuration is confirmed on 1,024 seeds, 20000-21023 (field seeds 5000-5255), in each case.

## What may change and what may not

**May change:**
- the RND networks: initialization, target output, which layers train;
- the predictor's optimizer, loss and learning-rate schedule;
- the bonus readout.

**May not change:**
- The bonus never uses visit counts or the step counter. A learning-rate schedule may use the step.
- PPO, the policies, the maps, the seeds, the budget and the evaluation stay as they are.
- CleanRL's reward normalization stays on, so an idea differs from the baseline only in the change it tests. The one
  exception: an idea that replaces the reward normalization with one of our own design turns CleanRL's off. It is
  still compared with the same baseline.
- Published RND variants (NGU's error normalization, NovelD, DRND, coin-flip networks, E3B, random distribution
  distillation, self-supervised network distillation, randomized-prior ensembles) may appear only as labeled
  comparison rows, never as the result.

## Measures

Define $b_t(s)$ as the bonus of cell $s$ after $t$ updates (the readout the policy receives before reward
normalization), and $N_t(s)$ as the visits of cell $s$ by then.

1. **Start evenness:** the ratio $\max_s b_0(s) / \min_s b_0(s)$ over the cells, and the spread
   $\mathrm{sd}_s \log b_0(s)$. Target: a ratio of at most 1.5.
2. **Decay exponent:** the least-squares slope of $\log(b_t(s)/b_0(s))$ on $\log N_t(s)$, per cell and seed. It uses
   checkpoints with $N_t(s) \ge 2$, and cells with at least 3 such checkpoints. This is the method of the original
   study's decay section. Target: a median in [−0.6, −0.4].
3. **Late count consistency:** $\mathrm{sd}_s \log\left(b_T(s)\sqrt{N_T(s)}\right)$ at the last checkpoint $T$. It
   is zero when every cell's bonus is the same function $c/\sqrt{N}$ of its own visits.
4. **Performance:** the mean capped first-goal time $M(B) = \dfrac{1}{K} \sum_{i=1}^{K} \min(T_i, B)$
   (project rule `exploration-metrics.md`). Define $K$ as the number of policies and $T_i$ as policy $i$'s first goal
   transition. Discovery rate, coverage and the final-policy goal rate are reported beside it.

The distillation bench (`scripts/grid_initialization_campaign/rnd_decay_bench.py`) reads measures 1-3 without the
policy, under uniform and start-heavy visits. The RL runs read all four.

## Stages

1. **Bench (minutes).** Measures 1-3 for every idea, and the value of each knob (for example an SGD learning rate),
   before any RL unit is sent.
2. **RL screen (each batch).** Each idea runs in each case:
   - 14 settings: learning rate 1e-3 × entropy {1e-3, 1e-2} × weights 2^-6, 2^-4, …, 2^6;
   - 32 development seeds, 0-31 (field seeds 0-7);
   - stopped at $2^{22}$ transitions on the full $2^{24}$ learning-rate schedule;
   - the same control (the baseline code) in every batch.
   An idea's screen score in a case is the smallest $M(2^{22})$ over its 14 settings.
3. **Finalists (from 2026-11-12 18:00 PT at the latest).** One or two methods run:
   - the baseline's own selection protocol (52 configurations, seeds 0-15, $2^{24}$);
   - the 1,024-seed confirmation paired with the baseline;
   - a 2×2 ablation (even-start part on or off × decay part on or off) on 512 of those seeds.

## Success rule (fixed now)

The final method succeeds in a case when its confirmed $M(2^{24})$ is below the baseline's on the same 1,024 seeds,
and the 95% interval of the paired difference is below zero (bootstrap resampling field-seed groups). In addition:
- its measured start evenness and decay exponent meet the targets;
- the ablation shows whether the gain needs both properties;
- it passes the novelty check in `literature/novelty.md`.

Every case is reported as it comes out, including a case where the method does not win.
