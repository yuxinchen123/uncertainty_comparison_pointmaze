#!/usr/bin/env python
"""Launch gate: the named EnvSetup this run uses on PointMaze must reproduce train run 5's
environment exactly.

Train run 5 configured its environment with individual flags (`--env_name=PointMaze_Large-v3`,
`--goal_position=top_right`, `--discount_factor=0.999`, `--env_max_episode=400`,
`--apply_termination_wrapper=False`) and left every other env field at the Config default. This run
instead passes one named setup, `--env_setup=initial_single_large_pointmaze_max_400`, whose registry
docstring claims to be that same configuration recorded verbatim.

If the claim is wrong in any field, the PointMaze column of this run would not be "train run 5 with
a different learning rate" at all, and its comparison against train run 5's frozen bar would be
meaningless. So the claim is checked, not trusted: every env-side Config field is compared between
the two parameterizations, and the goal and start cells are resolved through the real
`select_cells` on a real environment.

Run:  /p/rlprojects/RND/.venvs/exploration/bin/python -m pytest slurm/test_pointmaze_env_equivalence.py -q
"""
import os
import sys

import pytest

PROJ = "/p/rlprojects/RND/07_reconstruction"
sys.path.insert(0, PROJ)
sys.path.insert(0, os.path.join(PROJ, "src"))
import train  # noqa: E402

# every env-side field train.py's _apply_env_setup can touch, plus the two that decide the wrapper
# stack; these are the fields that must agree for the two parameterizations to be the same task
ENV_FIELDS = ["env_name", "continuing_task", "reset_target", "position_noise_range",
              "env_max_episode", "reward_shift", "attach_xy_to_state", "include_contact_forces",
              "discount_factor", "apply_termination_wrapper"]

# train run 5's FIXED block, env-side entries only (copied from that run's slurm/build_queue.py)
RUN5_ENV_FLAGS = {"env_name": "PointMaze_Large-v3", "discount_factor": 0.999,
                  "env_max_episode": 400, "goal_position": "top_right",
                  "apply_termination_wrapper": False}


def config_from(**overrides):
    """A Config built the way train.py builds one, with `overrides` standing in for CLI flags."""
    cfg = train.Config(**overrides)
    if cfg.env_setup:
        # _apply_env_setup skips fields the CLI set explicitly; here the overrides ARE the explicit set
        train._apply_env_setup(cfg, set(overrides))
    return cfg


def test_every_environment_field_agrees():
    """The named setup and train run 5's flags produce the same value in every env-side field."""
    run5 = config_from(**RUN5_ENV_FLAGS)
    ours = config_from(env_setup="initial_single_large_pointmaze_max_400")
    mismatches = {f: (getattr(run5, f), getattr(ours, f))
                  for f in ENV_FIELDS if getattr(run5, f) != getattr(ours, f)}
    assert mismatches == {}, f"env fields differ (train run 5, this run): {mismatches}"


def test_the_named_setup_pins_the_cells_train_run_5_resolved_by_name():
    """Train run 5 said `goal_position=top_right`; the named setup writes the cells out explicitly.
    Resolve BOTH through the real select_cells on a real environment and require the same pair."""
    run5 = config_from(**RUN5_ENV_FLAGS)
    ours = config_from(env_setup="initial_single_large_pointmaze_max_400")
    env_run5 = train.make_base_env(run5)
    env_ours = train.make_base_env(ours)
    try:
        assert train.select_cells(run5, env_run5, seed=0) == train.select_cells(ours, env_ours, seed=0)
        # and the pair is the documented one: goal upper-right (1, 10), start lower-left (7, 1)
        assert train.select_cells(ours, env_ours, seed=0) == ((1, 10), (7, 1))
    finally:
        env_run5.close()
        env_ours.close()


def test_the_setup_is_not_a_section_eight_setup():
    """A section-8 setup would bring discount 0.99, the -1 reward shift and exact-center resets —
    a different task. The PointMaze column of this run must NOT accidentally use one."""
    ours = config_from(env_setup="initial_single_large_pointmaze_max_400")
    assert ours.discount_factor == 0.999
    assert ours.reward_shift == 0.0
    assert ours.position_noise_range == 0.25
    assert ours.continuing_task is True
    assert ours.env_max_episode == 400


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
