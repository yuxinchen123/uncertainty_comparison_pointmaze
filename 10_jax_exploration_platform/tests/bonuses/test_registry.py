"""The bonus registry: a name selects a family, and it does so before anything is compiled.

Run: PYTHONNOUSERSITE=1 JAX_PLATFORMS=cpu <jax python> test_registry.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))
from exploration_platform.agents.ppo.config import PPOConfig  # noqa: E402
from exploration_platform.bonuses.registry import (ALIASES, BONUS_REGISTRY,  # noqa: E402
                                                   make_bonus)
from exploration_platform.envs.pointmaze.pm_common import EnvConfig  # noqa: E402
from exploration_platform.training.runner import Runner  # noqa: E402

SMALL = dict(n_copies=2, n_envs=2, num_steps=8, prime_iterations=1)


def test_every_preset_builds():
    """Every registered name produces a family that can be built for a real configuration."""
    cfg = PPOConfig(**SMALL)
    for name in sorted(BONUS_REGISTRY):
        bonus = make_bonus(name)(cfg, EnvConfig(), cfg.n_copies, cfg.base_seed, [0, 1])
        params, state = bonus.init()
        print(f"{name:16} -> {bonus.name:16} parameters {sorted(params)} state {sorted(state)}")
        assert callable(bonus.post_rollout) and callable(bonus.loss)
    print("ok test_every_preset_builds")


def test_older_names_still_resolve():
    """07_reconstruction's names select the same families as the platform's own names."""
    for old, new in sorted(ALIASES.items()):
        assert make_bonus(old) is make_bonus(new), f"{old} and {new} resolved differently"
        print(f"{old:16} -> {new}")
    print("ok test_older_names_still_resolve")


def test_unknown_name_is_refused():
    """An unrecognised name fails immediately, with the available names in the message."""
    try:
        make_bonus("elliptical")
    except ValueError as error:
        assert "elliptical" in str(error) and "rnd_next_state" in str(error)
        print(f"refused as expected: {error}")
    else:
        raise AssertionError("an unknown bonus name was accepted")
    print("ok test_unknown_name_is_refused")


def test_runner_selects_by_name():
    """The runner takes the name and ends up holding that family."""
    for name, expected in [("rnd_next_state", "rnd_next_state"), ("rnd_state", "rnd_next_state"),
                           ("none", "none"), ("no_exploration", "none")]:
        runner = Runner(PPOConfig(**SMALL), bonus=name)
        assert runner.bonus.name == expected, f"{name} gave {runner.bonus.name}"
        print(f"Runner(bonus={name!r}) holds {runner.bonus.name}")
    print("ok test_runner_selects_by_name")


if __name__ == "__main__":
    test_every_preset_builds()
    test_older_names_still_resolve()
    test_unknown_name_is_refused()
    test_runner_selects_by_name()
