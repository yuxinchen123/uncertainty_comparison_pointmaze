"""Splitting the copies into groups, so one run answers "which setting is best".

The copies normally differ only by their seed. A sweep divides them into groups and gives each
group its own learning rate, so the answer comes out of one compiled program instead of one run
per setting. Adam is elementwise, so C copies stacked on the leading axis are already C
independent optimizers and the rate is the only thing that was a scalar: it becomes a [C] device
constant, reshaped against each parameter inside the same elementwise update.
"""
from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np

from .. import F32


@dataclass(frozen=True)
class SweepVectors:
    """What a sweep hands the compiled program, plus the bookkeeping the driver reports."""
    is_sweep: bool
    lr_per_copy: jnp.ndarray   # [C] float32, or None when every copy uses the same rate
    copy_seed_index: list      # which seed stream each copy draws from
    copy_group: np.ndarray     # which group each copy belongs to


def build_sweep(cfg) -> SweepVectors:
    """Derive the per-copy vectors from the configuration's group lists.

    before: rates (1e-4, 1e-3), counts (2, 2), paired seeding
    after:  lr [1e-4, 1e-4, 1e-3, 1e-3], seed index [0, 1, 0, 1], group index [0, 0, 1, 1]
            — so copy k of every group starts from the same weights and meets the same
            environments, and a difference between the groups is the rate's doing
    """
    C = cfg.n_copies
    if not cfg.learning_rates:
        return SweepVectors(False, None, list(range(C)), np.zeros(C, dtype=int))

    assert sum(cfg.copies_per_rate) == C, "copies_per_rate must sum to n_copies"
    lr_list, seed_list, group_list = [], [], []
    for g, (rate, count) in enumerate(zip(cfg.learning_rates, cfg.copies_per_rate)):
        lr_list += [rate] * count
        seed_list += (list(range(count)) if cfg.sweep_seed_mode == "paired"
                      else list(range(len(seed_list), len(seed_list) + count)))
        group_list += [g] * count
    # a device CONSTANT, so the annealed rate stays one scalar argument per iteration and no
    # per-iteration host-to-device copy is introduced
    return SweepVectors(True, jnp.asarray(lr_list, F32), seed_list, np.asarray(group_list))


def sweep_config(learning_rates, copies_per_rate, style="epoch_minibatch", **overrides):
    """Build the configuration for a learning-rate sweep across copy groups.

    learning_rates: the rates to try, e.g. [1e-4, 3e-4, 1e-3, 3e-3].
    copies_per_rate: how many independent copies each rate gets — one number for all rates, or
    one per rate. The total copy count is their sum.

    before: sweep_config([1e-4, 1e-3], 128) ; after: 256 copies, the first 128 training at 1e-4
    and the second 128 at 1e-3, with copy k of both groups sharing initial weights and
    environments (paired), so the two groups differ only by the rate.
    """
    from ..agents.ppo.config import PPOConfig
    rates = tuple(float(x) for x in learning_rates)
    counts = ((int(copies_per_rate),) * len(rates) if isinstance(copies_per_rate, int)
              else tuple(int(x) for x in copies_per_rate))
    assert len(counts) == len(rates), "give one copy count per learning rate, or a single number"
    return PPOConfig(n_copies=sum(counts), update_style=style, learning_rates=rates,
                     copies_per_rate=counts, **overrides)
