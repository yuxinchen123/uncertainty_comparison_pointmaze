"""What a bonus family has to provide, and nothing more.

A bonus is a set of pure functions over immutable trees, gathered into one frozen record. The
composer picks the record BEFORE the program is compiled and calls its functions directly, so the
compiled program contains one bonus and no branch that chooses between bonuses.

The hooks, in the order one iteration calls them:

| hook | when | signature |
|---|---|---|
| `init` | once, when the state is created | `() -> (params, state)` |
| `prime` | once per warm-up rollout, before training | `(params, state, next_obs) -> state` |
| `rollout_step` | inside the rollout scan, one step at a time | `(params, state, next_obs) -> reward` |
| `post_rollout` | once, after the rollout | `(params, state, next_obs, scored) -> (state, reward, batch extras)` |
| `loss` | inside every gradient step | `(params, batch) -> [C] per-copy loss` |
| `metrics` | once per iteration | `(params, state) -> dict of [C] arrays` |

`constants` holds the family's fixed arrays that are neither parameters nor state (the random
network distillation target). A checkpoint saves them, and a resumed run hands them back to the
family's factory as `saved_constants`, so the program holds the saved numbers and not a new draw.

`next_obs` is always float32 [C, M, 4]: [C, N, 4] for one rollout step, [C, T*N, 4] for a whole
rollout flattened. `reward` is [C, M]. `params` and `state` are trees of device arrays and may be
empty — a family with nothing to train passes `{}`, and the compiled program then holds none of
its arithmetic at all.

`prime` and `rollout_step` may be `None`:

- `prime is None` — the family needs no warm-up, and the runner runs no warm-up rollouts.
- `rollout_step is None` — the family cannot score one step at a time (a visit-count table has to
  see the whole rollout before it can say how often a state was reached), so composing it with
  `hoist_rollout=False` is refused when the program is built rather than producing wrong numbers.

`post_rollout` takes `scored`: the per-step rewards the rollout scan already produced, flattened
to [C, T*N], or `None` when the rollout did not score anything and the family must do it now.
That is the whole difference between the two rollout shapes, and it is decided once, in Python,
when the program is built.

Five further fields say how the trainer treats the family. Their defaults are what the trainer
always did, so a family that sets none of them composes into the program it always did:

- `score_alone` — `(params, state, obs [C, M, obs]) -> [C, M]`: the bonus each given observation
  would get on its own, as the parameters and the state stand, changing neither. The trainer
  uses it to score the priming rollouts (for a divisor frozen after priming) and the probe
  observations of the maze grid. `None`: the family offers no such score, and both uses are
  refused when the program is built.
- `reward_divisor` — what the trainer divides the bonus by before it forms the intrinsic
  advantage. "running_std": the running standard deviation of the bonus's discounted sum, moved
  on by every rollout. "frozen_after_priming": the same statistic gathered over the priming
  rollouts only and never moved afterwards. "held_after_fraction": moved on by every rollout
  of the iterations numbered (from 0) below `divisor_hold_fraction` x the configuration's
  `total_iterations`, and not moved from that iteration on. "held_after_first_arrival": for
  each copy on its own, moved on by every rollout up to and including the one in which the
  copy first reaches the goal (the episode log's first arrival), and not moved by any later
  rollout; a copy that never arrives is moved on by every rollout. "none": no division.
- `divisor_hold_fraction` — the fraction read with the "held_after_fraction" divisor; `None`
  with every other divisor.
- `own_learning_rate` — `None`: the family's parameters are stepped by the agent's Adam, at the
  agent's rate, inside the joint gradient clip. A number: they are stepped at this constant
  rate (never annealed) with their own moments, and their gradient is left out of the clip.
- `record_on_maze_grid` — the trainer counts the visits to every maze cell, and the driver
  records the bonus at one probe observation per open cell once per logging window.
"""
from dataclasses import dataclass
from typing import Callable, Optional


@dataclass(frozen=True)
class BonusFunctions:
    """One bonus family, ready to compose: a name and the hooks described in this module."""
    name: str
    init: Callable
    post_rollout: Callable
    loss: Callable
    metrics: Callable
    prime: Optional[Callable] = None
    rollout_step: Optional[Callable] = None
    constants: Optional[dict] = None
    score_alone: Optional[Callable] = None
    reward_divisor: str = "running_std"
    divisor_hold_fraction: Optional[float] = None
    own_learning_rate: Optional[float] = None
    record_on_maze_grid: bool = False
