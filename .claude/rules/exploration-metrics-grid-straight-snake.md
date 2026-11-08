---
paths:
  - "10_jax_exploration_platform/src/exploration_platform/grid_initialization/**"
  - "10_jax_exploration_platform/scripts/grid_initialization_campaign/**"
  - "10_jax_exploration_platform/tests/grid_initialization/**"
  - "10_jax_exploration_platform/runs/*initial-bonus*/**"
  - "10_jax_exploration_platform/runs/*fixed-cleanrl-ppo-rnd*/**"
  - "10_jax_exploration_platform/development_document/grid_initialization.tex"
---

# Exploration metrics on the straight and snake grids

The environment-specific part of `exploration-metrics.md` (user rule, 2026-11-07). The primary
metric is the mean capped first-goal time $M(B)$ defined there.

## The grids

A straight grid is one row of $W$ cells. A snake grid is $T+1$ corridors of width $W$ joined by
$T$ connector cells. Cells are numbered along the path from the start (cell 0) to the goal (the
last cell). An episode ends on reaching the goal or at the episode limit, then restarts at the
start cell; the extrinsic reward is 1 on arriving at the goal and 0 otherwise.

## The four parts

| part | definition on these grids |
|---|---|
| goal event | arriving at the goal cell during training; $T_i$ is the training transition of the first arrival, counted from 1 |
| coverage unit | path cells; coverage at $B$ is the fraction of path cells reached at least once in training by transition $B$, with the start cell counted as reached |
| evaluation episode | from the start cell, up to the training episode limit, with no learning; $J = 32$ episodes with sampled actions (the reported rate) and 1 greedy episode (the grid is deterministic, so one is exact) |
| budget $B$ | the study's transitions per policy (16,777,216 in the fixed-settings study of 2026-11) |

## Conventions that go with these metrics

- Random-action warm-up steps used to start the state-normalization statistics are not training
  transitions: they do not count toward $T_i$, visits, or coverage.
- Count-based bonus fields (uniform, alternating, frozen network field) are scaled to average 1
  over path cells before division by $\sqrt{\text{visits}}$. The online RND bonus is the raw
  prediction error and is not scaled (user decision 2026-11-07).
- For network fields, four policy seeds share each field seed (field seed = policy seed // 4), so
  intervals resample field seeds as groups.
