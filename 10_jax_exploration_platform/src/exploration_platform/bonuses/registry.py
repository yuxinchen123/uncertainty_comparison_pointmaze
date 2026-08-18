"""Named bonus presets: the string a run's configuration carries, and the family it selects.

A preset is a factory — `(cfg, env_cfg, n_copies, base_seed, copy_seed_index, obs_dim=4)
-> BonusFunctions` —
so the family binds to the run's environment, copy count and seeds before anything is traced. The
name is resolved on the host, in Python, and only the resulting concrete functions ever reach the
compiler: there is no place in a compiled program where an algorithm is chosen.
"""
from .none import build as build_none
from .rnd.config import RNDConfig
from .rnd.implementation import build as build_rnd
from .visit_count.config import VisitCountConfig
from .visit_count.implementation import build as build_visit_count


def rnd_preset(rnd_cfg: RNDConfig):
    """A factory for random network distillation at the given network widths."""
    return lambda cfg, env_cfg, n_copies, base_seed, copy_seed_index, obs_dim=4: build_rnd(
        cfg, env_cfg, rnd_cfg, n_copies, base_seed, copy_seed_index, obs_dim)


def visit_count_preset(vc_cfg: VisitCountConfig, name: str):
    """A factory for the oracle visit-count bonus at the given decay."""
    return lambda cfg, env_cfg, n_copies, base_seed, copy_seed_index, obs_dim=4: (
        build_visit_count(cfg, env_cfg, vc_cfg, name, n_copies, base_seed, copy_seed_index,
                          obs_dim))


BONUS_REGISTRY = {
    "none": build_none,
    "rnd_next_state": rnd_preset(RNDConfig()),
    # decay -0.5 is 1/sqrt(n), decay -1 is 1/n
    "gt_position_velocity_sqrt": visit_count_preset(VisitCountConfig(decay=-0.5),
                                                    "gt_position_velocity_sqrt"),
    "gt_position_velocity_linear": visit_count_preset(VisitCountConfig(decay=-1.0),
                                                      "gt_position_velocity_linear"),
}

# 07_reconstruction's names for the same families, so a configuration written against that
# project selects the same thing here rather than failing on an unknown name. That project's
# `gt_position_velocity` defaults to decay -0.5, which is the sqrt preset.
ALIASES = {
    "no_exploration": "none",
    "rnd_state": "rnd_next_state",
    "gt_position_velocity": "gt_position_velocity_sqrt",
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
