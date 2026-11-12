"""Named bonus presets: the string a run's configuration carries, and the family it selects.

A preset is a factory — `(cfg, env_cfg, n_copies, base_seed, copy_seed_index, obs_dim=4)
-> BonusFunctions` —
so the family binds to the run's environment, copy count and seeds before anything is traced. The
name is resolved on the host, in Python, and only the resulting concrete functions ever reach the
compiler: there is no place in a compiled program where an algorithm is chosen.

A family may have options of its own (the fields of its configuration class). A run names the
ones it changes, `make_bonus(name, options)` builds the family with them, and
`bonus_options_record(name, options)` is the family's whole configuration as the run's records
hold it, so two runs of one family can be told apart.
"""
from dataclasses import asdict, fields

from .none import build as build_none
from .rnd.config import RNDConfig, resolved_rnd_config
from .rnd.implementation import build as build_rnd
from .visit_count.config import VisitCountConfig
from .visit_count.implementation import build as build_visit_count


def rnd_preset(rnd_cfg: RNDConfig):
    """A factory for random network distillation at the given network widths."""
    return (lambda cfg, env_cfg, n_copies, base_seed, copy_seed_index, obs_dim=4,
            saved_constants=None: build_rnd(cfg, env_cfg, rnd_cfg, n_copies, base_seed,
                                            copy_seed_index, obs_dim, saved_constants))


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


# the families a run may set options of: how a preset is built from the family's configuration,
# the configuration with every default, and the check that fills in what an option implies
OPTION_FAMILIES = {
    "rnd_next_state": (rnd_preset, RNDConfig(), resolved_rnd_config),
}


def option_value(name: str, value, default):
    """One option's value in the type of its default; a value typed on a command line is text.

    before: ("even_start", "true", False)          ; after: True
    before: ("fast_last_layer_step", "0.3", 0.0)   ; after: 0.3
    before: ("fast_last_layer_step", 1, 0.0)       ; after: 1.0
    before: ("reward_divisor", "none", "running_std") ; after: "none"
    """
    if isinstance(default, bool):
        if isinstance(value, bool):
            return value
        if value in ("true", "false"):
            return value == "true"
        raise ValueError(f"bonus option {name} is true or false, not {value!r}")
    if isinstance(default, str):
        if isinstance(value, str):
            return value
        raise ValueError(f"bonus option {name} is a word, not {value!r}")
    if isinstance(value, bool):
        raise ValueError(f"bonus option {name} is a number, not {value!r}")
    try:
        return type(default)(value)
    except (TypeError, ValueError):
        raise ValueError(f"bonus option {name} is a number, not {value!r}") from None


def family_configuration(bonus: str, options=None):
    """The configuration of a named family with a run's options applied, checked and resolved.

    options maps option names to values (text from a command line, or typed values from a
    configuration file). Returns None for a family that has no options.

    before: ("rnd_next_state", {"even_start": "true"})
    after:  RNDConfig(even_start=True) with every other field at its default
    """
    name = ALIASES.get(bonus, bonus)
    if name not in BONUS_REGISTRY:
        raise ValueError(f"unknown bonus {bonus!r}; the presets are {sorted(BONUS_REGISTRY)} "
                         f"and the accepted older names are {sorted(ALIASES)}")
    options = dict(options or {})
    if name not in OPTION_FAMILIES:
        if options:
            raise ValueError(f"the {name} bonus has no options; {sorted(options)} were given")
        return None
    _, defaults, resolve = OPTION_FAMILIES[name]
    known = {field.name: getattr(defaults, field.name) for field in fields(defaults)}
    unknown = sorted(set(options) - set(known))
    if unknown:
        raise ValueError(f"unknown options {unknown} for the {name} bonus; its options are "
                         f"{sorted(known)}")
    return resolve(type(defaults)(**{
        **known, **{option: option_value(option, value, known[option])
                    for option, value in options.items()}}))


def bonus_options_record(bonus, options=None) -> dict:
    """Every option of the family a run uses, by name, as the run's records hold them.

    before: ("rnd_next_state", {"even_start": True})
    after:  {"feature_dim": 128, "hidden": 256, "even_start": True, "reward_divisor":
             "running_std", ...}; {} for a family with no options or a factory passed in
    """
    if callable(bonus):
        return {}
    configuration = family_configuration(bonus, options)
    return {} if configuration is None else asdict(configuration)


def make_bonus(bonus, options=None):
    """Turn a preset name into a factory; a factory passed straight in is returned unchanged.

    options, when given, are the family's own options a run changes (`family_configuration`);
    without them the registered preset itself is returned.
    """
    if callable(bonus):
        if options:
            raise ValueError("options cannot be applied to a bonus factory passed straight in")
        return bonus
    configuration = family_configuration(bonus, options)
    name = ALIASES.get(bonus, bonus)
    if not options:
        return BONUS_REGISTRY[name]
    return OPTION_FAMILIES[name][0](configuration)
