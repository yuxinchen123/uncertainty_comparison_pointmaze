"""Named bonus presets: the string a run's configuration carries, and the family it selects.

A preset is a factory — `(cfg, env_cfg, n_copies, base_seed, copy_seed_index) -> BonusFunctions` —
so the family binds to the run's environment, copy count and seeds before anything is traced. The
name is resolved on the host, in Python, and only the resulting concrete functions ever reach the
compiler: there is no place in a compiled program where an algorithm is chosen.
"""
from .none import build as build_none
from .rnd.config import RNDConfig
from .rnd.implementation import build as build_rnd


def rnd_preset(rnd_cfg: RNDConfig):
    """A factory for random network distillation at the given network widths."""
    return lambda cfg, env_cfg, n_copies, base_seed, copy_seed_index: build_rnd(
        cfg, env_cfg, rnd_cfg, n_copies, base_seed, copy_seed_index)


BONUS_REGISTRY = {
    "none": build_none,
    "rnd_next_state": rnd_preset(RNDConfig()),
}

# 07_reconstruction's names for the same families, so a configuration written against that
# project selects the same thing here rather than failing on an unknown name
ALIASES = {
    "no_exploration": "none",
    "rnd_state": "rnd_next_state",
}


def make_bonus(bonus):
    """Turn a preset name into a factory; a factory passed straight in is returned unchanged."""
    if callable(bonus):
        return bonus
    name = ALIASES.get(bonus, bonus)
    if name not in BONUS_REGISTRY:
        raise ValueError(f"unknown bonus {bonus!r}; the presets are {sorted(BONUS_REGISTRY)} "
                         f"and the accepted older names are {sorted(ALIASES)}")
    return BONUS_REGISTRY[name]
