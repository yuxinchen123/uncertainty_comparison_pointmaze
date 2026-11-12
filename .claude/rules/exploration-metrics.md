# Exploration metrics: one primary metric for every environment

User rule, 2026-11-07. Applies to every exploration study in this project (grids, PointMaze,
Atari, and any later environment): which metric configurations are chosen by, which metrics are
reported beside it, and what each environment must define before a study reports them.

**Amended 2026-11-11 (user decision): on PointMaze and AntMaze, configurations are chosen by the
whole-curve mean of the training extrinsic return, not by the first-goal time.** The mean capped
first-goal time defined below stays the primary metric on the grids and is still computed and
reported beside the primary one on PointMaze and AntMaze. Definitions for those two families:
`exploration-metrics-pointmaze-antmaze.md`.

**Amended 2026-11-12 (user decision): on PointMaze and AntMaze the mean capped first-goal time is the secondary
metric.** It has its own configuration choice and its own table beside the primary one, and a method that is ahead
of RND on it in all five environments is noted in the report. Definitions: `exploration-metrics-pointmaze-antmaze.md`.

## The primary metric: mean capped first-goal time

Every configuration choice (intrinsic weight, learning rate, entropy, or any other swept setting)
minimises the mean capped first-goal time, in every environment except PointMaze and AntMaze
(amendment above):

$$M(B) = \dfrac{1}{K} \sum_{i=1}^{K} \min(T_i, B)$$

- $K$: the number of independently trained policies (seeds) in the configuration.
- $T_i$: the training transition at which policy $i$ first reaches the environment's goal event;
  $T_i = \infty$ if it never does.
- $B$: the training budget in transitions. A policy that never reaches the goal counts as $B$, so
  failures stay in the mean.

Ties in $M(B)$ are broken by the larger goal discovery rate $F(B)$.

## Metrics reported beside it (never used to choose a configuration)

Define $J$ as the number of evaluation episodes per policy, and $G_{ij} = 1$ when evaluation
episode $j$ of policy $i$ reaches the goal ($G_{ij} = 0$ otherwise). Evaluation uses the policy
trained for $B$ transitions, with no learning during evaluation.

1. **Goal discovery rate at $B$**: $F(B) = \dfrac{1}{K} \sum_{i=1}^{K} \mathbf{1}[T_i \le B]$.
2. **State coverage at $B$**: the fraction of the environment's coverage units reached at least
   once in training by transition $B$.
3. **Final-policy goal rate**: $P(B) = \dfrac{1}{KJ} \sum_{i=1}^{K} \sum_{j=1}^{J} G_{ij}$. It is
   reported with actions sampled from the policy; the greedy version (most likely action) is
   recorded beside it when the environment allows it.

## What each environment must define

Each environment has its own rule file that fills in four parts. A study must not report these
metrics in an environment until its row below says "defined".

| part | meaning |
|---|---|
| goal event | what counts as reaching the goal, which sets $T_i$ |
| coverage unit | what is counted for state coverage |
| evaluation episode | start, length limit, $J$, sampled and greedy |
| budget $B$ | the transitions per policy at which all four metrics are read |

| environment | state of the definitions | rule file |
|---|---|---|
| straight and snake grids | defined | `exploration-metrics-grid-straight-snake.md` |
| PointMaze (Medium, Large) | defined 2026-11-11; primary metric is the whole-curve mean of the training return | `exploration-metrics-pointmaze-antmaze.md` |
| AntMaze (U-maze, Medium, Large) | defined 2026-11-11; primary metric is the whole-curve mean of the training return | `exploration-metrics-pointmaze-antmaze.md` |
| Atari (Montezuma's Revenge and others) | to be defined; coverage may be rooms reached, and the goal event is still to be decided | — |

Add a row (and a rule file) when a new environment's study starts. The primary metric stays
$M(B)$ unless the row says otherwise; the goal event and the other three parts change.

User's words (2026-11-07): "the metrics I am going to care most, and optimize to is mean capped
first goal time ... we optimized to mean capped first goal time among all env."

User's words (2026-11-11), on PointMaze and AntMaze: "The goal I am trying to optimize here is the
extrinsic performance in training curve used in the clean RL rnd. I need to beat this."
