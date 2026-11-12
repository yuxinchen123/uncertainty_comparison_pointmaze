---
paths:
  - "10_jax_exploration_platform/src/exploration_platform/envs/pointmaze/**"
  - "10_jax_exploration_platform/src/exploration_platform/envs/antmaze/**"
  - "10_jax_exploration_platform/src/exploration_platform/training/**"
  - "10_jax_exploration_platform/runs/*pointmaze*/**"
  - "10_jax_exploration_platform/runs/*antmaze*/**"
  - "10_jax_exploration_platform/scripts/maze_campaign/**"
---

# Exploration metrics on PointMaze and AntMaze

The environment-specific part of `exploration-metrics.md` for the study that started 2026-11-11
(plan `~/.claude/plans/2026-11-11-22-40_pointmaze-antmaze-speed-and-beat-rnd.md`). User decisions
of 2026-11-11.

## Environments and the two reward settings

| family | map | start (row, column) | goal | step limit |
|---|---|---|---|---|
| PointMaze | Medium | (6, 1) | (1, 6) | 400 |
| PointMaze | Large | (7, 1) | (1, 10) | 400 |
| AntMaze | U-maze | (3, 1) | (1, 1) | 700 |
| AntMaze | Medium | (6, 1) | (1, 6) | 1,000 |
| AntMaze | Large | (7, 1) | (1, 10) | 1,000 |

- Fixed start and goal, no reset noise.
- Reaching the goal (within 0.45 m) ends the episode as a **termination**: the extrinsic return
  does not continue past it. The step limit is a **truncation**: the extrinsic return continues
  with the value of the last state. The intrinsic return runs across both.
- Define $H$ as the step limit and $t$ as the step of the episode at which the goal is reached.
  - **Setting A:** reward 0 per step, +1 on arrival. Episode return: 1 if reached, else 0.
  - **Setting B:** reward $-1/H$ per step, 0 on arrival. Episode return: $-(t-1)/H$ if reached,
    else $-1$.
  - Extrinsic discount 0.999 in both.

## The primary metric: whole-curve mean of the training return

Define $K$ as the number of seeds, $B$ as the budget in environment steps per policy, and $R_i(s)$
as the mean extrinsic return of the training episodes of policy $i$ that end in the logging window
containing step $s$ (a window with no finished episode takes the last value).

- **Training curve:** $\bar R(s) = \dfrac{1}{K}\sum_{i=1}^{K} R_i(s)$.
- **Whole-curve mean (chooses configurations):** $W(B)$, the mean of $R_i(s)$ over all steps up to
  $B$ and over the $K$ policies (the area under the curve divided by the budget).
- **Final level:** $L(B)$, the same mean over the last 10% of the budget.

Ties in $W(B)$ are broken by the larger $L(B)$.

## What "beat" means

A method beats RND in one environment and reward setting when, on the same fresh seeds:

1. the 95% interval of the difference in $W$ (method minus RND) is entirely above zero;
2. the 95% interval of the difference in $L$ is not entirely below zero.

Each method's configuration is chosen on development seeds 0 to 15; the comparison uses fresh
seeds from 30000 (1,024 on PointMaze, 256 on AntMaze); intervals resample seeds on the paired
differences. A method succeeds when it beats RND in all five environments in setting A, or in all
five in setting B.

## The secondary metric: mean capped first-goal time (user decision 2026-11-12)

The primary metric measures discovery and optimization together. The secondary metric measures discovery alone: how
soon a policy first reaches the goal. It is the metric of the straight and snake grids.

Define $T_i$ as the training step at which policy $i$ first reaches the goal ($T_i = \infty$ if it never does).

$$M(B) = \dfrac{1}{K} \sum_{i=1}^{K} \min(T_i, B)$$

Smaller is better. A policy that never reaches the goal counts as $B$. Ties are broken by the larger goal discovery
rate $F(B) = \dfrac{1}{K} \sum_{i=1}^{K} \mathbf{1}[T_i \le B]$.

1. **Its own configuration.** For this metric each method's configuration, and RND's, is the one with the smallest
   $M(B)$ on the development seeds. It may differ from the configuration chosen for $W$.
2. **Two tables in every report.** One for the primary metric and one for the secondary metric, each with its own
   chosen configurations; the two are never mixed in one row.
3. **Ahead on discovery** (one environment, one reward setting, same fresh seeds): the 95% interval of the
   difference in $M$ (method minus RND, each at its own configuration for $M$) is entirely below zero.
4. **Noted in the report, not part of the success rule:** every method that is ahead of RND on discovery in all
   five environments. Reading of the session, not the user's words: "all environments" is counted per reward
   setting, as for the primary metric (all five in setting A, or all five in setting B), and both counts are shown.
5. Seeds, intervals and budgets are those of the primary metric.

User's words (2026-11-12): "i also want to introudce an secnodnary metrics that is the first step reached the goal
(the one we used in straitghe and snake env) if any algorihtm beats RND on all envs on this, also note it in the
report. (the aglortihm who acehive this may have different hyeperaters configuraiton then the one acheived the final
perofmrance, it is fine, just put those into two tables), this one is to understand the ability to discover, and the
tradiciton extrin metrics is to discouvner + optmizaiton."

## The four parts of `exploration-metrics.md`

| part | definition here |
|---|---|
| goal event | arriving within 0.45 m of the goal during training; $T_i$ is the training step of the first arrival |
| coverage unit | open maze cells; coverage at $B$ is the fraction reached at least once in training |
| evaluation episode | none separate: the measures are read from the training episodes |
| budget $B$ | fixed per environment and reward setting from the RND baseline's curves before any method is compared, and written in the study's `program.md` |

Reported beside the two metrics, never used to choose: the goal discovery rate $F(B)$ (except as
the tie-break of the secondary metric), coverage, mean steps to the goal, and the share of training
episodes that reach the goal in the last 10% of the budget. The final level against the training
length gets its own report section, because it depends on how many steps are trained.
