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
