"""Splitting the copies into groups, so one run answers "which setting is best".

The copies normally differ only by their seed. A sweep divides them into groups and gives each
group its own learning rate, its own weight on the intrinsic reward, or both, so the answer comes
out of one compiled program instead of one run per setting.

Both knobs generalise the same way, because both enter the arithmetic elementwise along the copy
axis. Adam is elementwise, so C copies stacked on the leading axis are already C independent
optimizers and the rate was the only scalar left; the intrinsic weight appears once, in
`advantage = beta * intrinsic + ext_coef * extrinsic`. Each becomes a [C] device constant,
broadcast into the same operation the program was already performing.

A knob that is not being swept keeps its scalar. That is deliberate: it leaves the compiled
program for a run that does not sweep it exactly what it was.
"""
from dataclasses import dataclass
from itertools import product

import jax.numpy as jnp
import numpy as np

from .. import F32


@dataclass(frozen=True)
class SweepVectors:
    """What a sweep hands the compiled program, plus the bookkeeping the driver reports."""
    is_sweep: bool
    lr_per_copy: jnp.ndarray      # [C] float32, or None when every copy uses the same rate
    beta_per_copy: jnp.ndarray    # [C] float32, or None when every copy uses the same weight
    copy_seed_index: list         # which seed stream each copy draws from
    copy_group: np.ndarray        # which group each copy belongs to
    group_settings: tuple         # (learning rate, beta) of each group, in group order


def build_sweep(cfg) -> SweepVectors:
    """Derive the per-copy vectors from the configuration's group lists.

    The groups are the cross product of the learning rates and the intrinsic-reward weights, rate
    outermost, and every cell gets the same number of copies.

    before: rates (1e-4, 1e-3), betas (0, 1), 2 copies per cell, paired seeding
    after:  4 groups of 2 copies = 8 copies;
            lr    [1e-4, 1e-4, 1e-4, 1e-4, 1e-3, 1e-3, 1e-3, 1e-3]
            beta  [   0,    0,    1,    1,    0,    0,    1,    1]
            seed  [   0,    1,    0,    1,    0,    1,    0,    1]
            group [   0,    0,    1,    1,    2,    2,    3,    3]
            — so copy k of every group starts from the same weights and meets the same
            environments, and a difference between groups is the swept settings' doing
    """
    C = cfg.n_copies
    sweeps_rate, sweeps_beta = bool(cfg.learning_rates), bool(cfg.betas)
    if not (sweeps_rate or sweeps_beta):
        return SweepVectors(False, None, None, list(range(C)), np.zeros(C, dtype=int),
                            ((cfg.learning_rate, cfg.int_coef),))

    rates = tuple(cfg.learning_rates) if sweeps_rate else (cfg.learning_rate,)
    betas = tuple(cfg.betas) if sweeps_beta else (cfg.int_coef,)
    settings = tuple(product(rates, betas))
    per_group = cfg.copies_per_group
    assert per_group > 0, "a sweep needs copies_per_group set"
    assert len(settings) * per_group == C, (
        f"{len(rates)} learning rates x {len(betas)} intrinsic weights x {per_group} copies "
        f"per group is {len(settings) * per_group} copies, but n_copies is {C}")

    lr_list, beta_list, seed_list, group_list = [], [], [], []
    for g, (rate, beta) in enumerate(settings):
        lr_list += [rate] * per_group
        beta_list += [beta] * per_group
        seed_list += (list(range(per_group)) if cfg.sweep_seed_mode == "paired"
                      else list(range(len(seed_list), len(seed_list) + per_group)))
        group_list += [g] * per_group

    # device CONSTANTS, so the annealed rate stays one scalar argument per iteration and no
    # per-iteration host-to-device copy is introduced. A knob that is not swept stays None, and
    # its scalar stays in the program.
    return SweepVectors(
        True,
        jnp.asarray(lr_list, F32) if sweeps_rate else None,
        jnp.asarray(beta_list, F32) if sweeps_beta else None,
        seed_list, np.asarray(group_list), settings)


def sweep_config(learning_rates=(), betas=(), copies_per_group=1, style="epoch_minibatch",
                 **overrides):
    """Build the configuration for a sweep across copy groups.

    learning_rates: the rates to try, e.g. [1e-4, 3e-4, 1e-3]. Empty means every copy trains at
    the configuration's own `learning_rate`.
    betas: the weights on the intrinsic advantage to try, e.g. [0, 1e-2, 1]. Empty means every
    copy uses the configuration's own `int_coef`.
    copies_per_group: how many independent copies each (rate, beta) cell gets. The total copy
    count is that times the number of cells.

    before: sweep_config([1e-4, 1e-3], [0.0, 1.0], copies_per_group=128)
    after:  four groups of 128 copies = 512 copies, one group per (rate, beta) cell, with copy k
            of every group sharing initial weights and environments (paired), so the groups differ
            only by the settings being swept.
    """
    from ..agents.ppo.config import PPOConfig
    rates = tuple(float(x) for x in learning_rates)
    weights = tuple(float(x) for x in betas)
    assert rates or weights, "a sweep needs learning rates, intrinsic weights, or both"
    cells = max(len(rates), 1) * max(len(weights), 1)
    return PPOConfig(n_copies=cells * int(copies_per_group), update_style=style,
                     learning_rates=rates, betas=weights,
                     copies_per_group=int(copies_per_group), **overrides)
