"""Splitting the copies into groups, so one run answers "which setting is best".

The copies normally differ only by their seed. A sweep divides them into groups and gives each
group its own learning rate, its own weight on the intrinsic reward, or both, so the answer comes
out of one compiled program instead of one run per setting.

Both knobs generalise the same way, because both enter the arithmetic elementwise along the copy
axis. Adam is elementwise, so C copies stacked on the leading axis are already C independent
optimizers and the rate was the only scalar left; the intrinsic weight appears once, in
`advantage = beta * intrinsic + ext_coef * extrinsic`. Each becomes a [C] device constant,
broadcast into the same operation the program was already performing.

Explicit cells (`parameter_settings`) may also give each copy its own entropy coefficient, clip
coefficient, value coefficient, and scale and offset on the extrinsic reward; `setting_names`
says which value each column of a cell holds.

A knob that is not being swept keeps its scalar. That is deliberate: it leaves the compiled
program for a run that does not sweep it exactly what it was.
"""
from dataclasses import dataclass
from itertools import product

import jax.numpy as jnp
import numpy as np

from .. import F32

# every setting that may differ between copies, and the values it accepts; a new per-copy setting
# adds one row here and reads its value through `SweepVectors.setting` where it is used
SETTING_RULES = {
    "learning_rate": (lambda value: value > 0, "positive"),
    "int_coef": (lambda value: value >= 0, "zero or positive"),
    "ent_coef": (lambda value: value >= 0, "zero or positive"),
    "clip_coef": (lambda value: value > 0, "positive"),
    "vf_coef": (lambda value: value >= 0, "zero or positive"),
    "ext_reward_scale": (lambda value: True, "finite"),
    "ext_reward_offset": (lambda value: True, "finite"),
}
DEFAULT_SETTING_NAMES = ("learning_rate", "int_coef")


@dataclass(frozen=True)
class SweepVectors:
    """What a sweep hands the compiled program, plus the bookkeeping the driver reports."""
    is_sweep: bool
    per_copy: dict                # setting name -> [C] float32 device constant, only for the
                                  # settings that have one value per copy
    per_copy_values: dict         # the same settings as tuples of Python numbers, from which
                                  # derived constants (1 - clip) are formed before rounding
    copy_seed_index: list         # which seed stream each copy draws from
    copy_group: np.ndarray        # which group each copy belongs to
    group_settings: tuple         # the settings of each group, in group order
    setting_names: tuple          # the name of each column of `group_settings`

    @property
    def lr_per_copy(self):
        """[C] float32 learning rates, or None when every copy uses the same rate."""
        return self.per_copy.get("learning_rate")

    @property
    def beta_per_copy(self):
        """[C] float32 intrinsic weights, or None when every copy uses the same weight."""
        return self.per_copy.get("int_coef")

    def setting(self, cfg, name: str, shape=None, transform=None):
        """One setting as the program should hold it: a Python scalar, or one value per copy.

        A setting that is not per copy returns the configuration's own Python number, so the
        compiled program keeps the constant it always had. A per-copy setting returns a float32
        array, reshaped to `shape` so it broadcasts along the copy axis. `transform` is applied to
        the Python numbers BEFORE they become float32, which is where the scalar program rounds.

        before: clip_coef 0.1 for every copy, transform 1 - x      -> 0.9 (a Python float)
        after:  clip_coef (0.1, 0.2) per copy, shape (2, 1), same  -> [[0.9], [0.8]] float32
        """
        values = self.per_copy_values.get(name)
        if values is None:
            value = getattr(cfg, name)
            return transform(value) if transform is not None else value
        if transform is None and shape is None:
            return self.per_copy[name]
        array = jnp.asarray([transform(v) if transform is not None else v for v in values], F32)
        return array.reshape(shape) if shape is not None else array


def check_explicit_settings(names: tuple, rows: tuple) -> None:
    """Refuse unknown or repeated setting names, and rows that do not fit them.

    before: names ("learning_rate", "ent_coef"), rows ((1e-4, 0.001), (1e-4, 0.01)) -> accepted
    before: the same names with a row (1e-4,) or (0.0, 0.001)                       -> ValueError
    """
    unknown = [name for name in names if name not in SETTING_RULES]
    if unknown or len(set(names)) != len(names) or not names:
        raise ValueError(f"setting_names {names!r} must be distinct names from "
                         f"{sorted(SETTING_RULES)}")
    if len(set(rows)) != len(rows):
        raise ValueError("parameter_settings contains duplicate cells")
    for row in rows:
        if len(row) != len(names):
            raise ValueError(f"the cell {row!r} does not hold one value for each of {names!r}")
        for name, value in zip(names, row):
            accepts, wording = SETTING_RULES[name]
            if not np.isfinite(value) or not accepts(value):
                if name in DEFAULT_SETTING_NAMES:
                    raise ValueError("each cell needs a finite positive rate and nonnegative "
                                     f"weight; {name} is {value!r}")
                raise ValueError(f"{name} must be finite and {wording}; a cell holds {value!r}")


def build_sweep(cfg) -> SweepVectors:
    """Derive the per-copy vectors from the configuration's group lists.

    The groups are the cross product of the learning rates and the intrinsic-reward weights, rate
    outermost, or the explicit cells of `parameter_settings`; every cell gets the same number of
    copies.

    before: rates (1e-4, 1e-3), betas (0, 1), 2 copies per cell, paired seeding
    after:  4 groups of 2 copies = 8 copies;
            lr    [1e-4, 1e-4, 1e-4, 1e-4, 1e-3, 1e-3, 1e-3, 1e-3]
            beta  [   0,    0,    1,    1,    0,    0,    1,    1]
            seed  [   0,    1,    0,    1,    0,    1,    0,    1]
            group [   0,    0,    1,    1,    2,    2,    3,    3]
            — so copy k of every group starts from the same weights and meets the same
            environments, and a difference between groups is the swept settings' doing

    before: setting_names ("learning_rate", "ent_coef"), cells ((1e-4, 0.001), (1e-4, 0.01)),
            2 copies per cell
    after:  lr [1e-4] * 4 and entropy [0.001, 0.001, 0.01, 0.01] per copy; the intrinsic weight
            stays the configuration's scalar
    """
    C = cfg.n_copies
    names = tuple(cfg.setting_names)
    sweeps_rate, sweeps_beta = bool(cfg.learning_rates), bool(cfg.betas)
    explicit = tuple(cfg.parameter_settings)
    if explicit and (sweeps_rate or sweeps_beta):
        raise ValueError("parameter_settings cannot be combined with a Cartesian sweep")
    if explicit:
        check_explicit_settings(names, explicit)
    elif names != DEFAULT_SETTING_NAMES:
        raise ValueError("setting_names describes the rows of parameter_settings; without "
                         f"parameter_settings it must stay {DEFAULT_SETTING_NAMES!r}")
    if not (explicit or sweeps_rate or sweeps_beta):
        return SweepVectors(False, {}, {},
                            list(range(cfg.copy_seed_offset, cfg.copy_seed_offset + C)),
                            np.zeros(C, dtype=int), ((cfg.learning_rate, cfg.int_coef),),
                            DEFAULT_SETTING_NAMES)

    # the cells and the names of their columns; a Cartesian sweep's cells are (rate, weight)
    rates = tuple(cfg.learning_rates) if sweeps_rate else (cfg.learning_rate,)
    betas = tuple(cfg.betas) if sweeps_beta else (cfg.int_coef,)
    settings = explicit or tuple(product(rates, betas))
    swept = names if explicit else tuple(
        name for name, on in zip(DEFAULT_SETTING_NAMES, (sweeps_rate, sweeps_beta)) if on)
    per_group = cfg.copies_per_group
    assert per_group > 0, "a sweep needs copies_per_group set"
    assert len(settings) * per_group == C, (
        f"{len(settings)} parameter cells x {per_group} copies "
        f"per group is {len(settings) * per_group} copies, but n_copies is {C}")

    # the seed index of a copy keys BOTH its initial weights and the environment draws it meets,
    # so shifting it by `copy_seed_offset` moves this process onto a different slice of one
    # logical run's copies. That is what lets a 1,024-copy unit be cut into chunks that run on
    # separate cards: chunk j of k passes offset j x per_group and holds copies j x per_group
    # onward, sharing no seed with any other chunk.
    # before: per_group=3, offset=0 -> seeds [0,1,2] per group
    # after:  per_group=3, offset=3 -> seeds [3,4,5] per group
    offset = cfg.copy_seed_offset
    columns = {name: [] for name in names}
    seed_list, group_list = [], []
    for g, row in enumerate(settings):
        for name, value in zip(names, row):
            columns[name] += [value] * per_group
        seed_list += (list(range(offset, offset + per_group)) if cfg.sweep_seed_mode == "paired"
                      else list(range(offset + len(seed_list),
                                      offset + len(seed_list) + per_group)))
        group_list += [g] * per_group

    # device CONSTANTS, so the annealed rate stays one scalar argument per iteration and no
    # per-iteration host-to-device copy is introduced. A setting that is not swept has no entry, and
    # its scalar stays in the program.
    return SweepVectors(
        True,
        {name: jnp.asarray(columns[name], F32) for name in swept},
        {name: tuple(columns[name]) for name in swept},
        seed_list, np.asarray(group_list), settings, names)


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
