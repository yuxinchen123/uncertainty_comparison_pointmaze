"""Tests for the settings that may differ between the copies of one program.

Beside the learning rate and the intrinsic weight, a cell of `parameter_settings` may hold an
entropy coefficient, a clip coefficient, a value coefficient, and a scale and offset on the
extrinsic reward; `setting_names` says which. The properties checked:

  1. the cells are turned into per-copy vectors by name, and a name that is not listed stays the
     configuration's own Python number,
  2. unknown names, repeated names and rows of the wrong length are refused,
  3. one value for every copy, given per copy, reproduces the program that held it as a scalar,
     bit for bit, for the three loss coefficients and for the reward scale and offset,
  4. a per-copy scale of 1 and offset of 0 reproduce the program with no reward mapping at all,
  5. changing one group's values leaves every other group bitwise unchanged,
  6. two reward settings on paired seeds start from the same weights and the same state and
     differ in the returns,
  7. the existing check of selected cells and iteration blocks still passes.

Run on a compute node: PYTHONNOUSERSITE=1 JAX_PLATFORMS=cpu <platform_jax python> -m pytest <file>
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from exploration_platform.agents.ppo.config import PPOConfig  # noqa: E402
from exploration_platform.agents.ppo.losses import loss_coefficients  # noqa: E402
from exploration_platform.envs.pointmaze.pm_common import EnvConfig  # noqa: E402
from exploration_platform.training.runner import Runner  # noqa: E402
from exploration_platform.training.sweep import build_sweep  # noqa: E402

import jax  # noqa: E402

SMALL = dict(n_envs=2, num_steps=8, prime_iterations=1)
# an episode that ends at the goal or after 6 steps, so three 8-step rollouts hold truncations
SHORT_EPISODES = EnvConfig(continuing_task=False, max_episode_steps=6)


def state_arrays(state) -> dict:
    """Every array of a training state, by its path in the tree."""
    flat, _ = jax.tree_util.tree_flatten_with_path(state)
    return {jax.tree_util.keystr(path): np.asarray(leaf) for path, leaf in flat}


def run(cfg, iterations=3, env_cfg=None):
    """Train a few iterations from a fresh state; return every array of the final state."""
    runner = Runner(cfg, env_cfg=env_cfg)
    state = runner.prime(runner.init_state(run_seed=17))
    for k in range(1, iterations + 1):
        state, _ = runner.iterate(state, runner.lr_argument(k, iterations))
    return state_arrays(state)


def assert_bitwise_equal(a: dict, b: dict, label: str) -> None:
    """Two states must hold the same arrays with the same bytes."""
    assert a.keys() == b.keys(), label
    for name in a:
        assert a[name].tobytes() == b[name].tobytes(), f"{label}: {name} differs"


def copy_rows(arrays: dict, rows) -> dict:
    """The rows of the given copies from every array whose leading axis is the copy axis."""
    return {name: value[rows] for name, value in arrays.items()
            if value.ndim >= 1 and value.shape[0] == 4}


def test_named_cells_become_per_copy_vectors():
    """Cells are read by name; a name that is not listed keeps the configuration's scalar."""
    cfg = PPOConfig(n_copies=4, copies_per_group=2, ent_coef=0.5,
                    setting_names=("learning_rate", "clip_coef"),
                    parameter_settings=((1e-4, 0.1), (3e-4, 0.2)))
    sweep = build_sweep(cfg)
    assert sweep.setting_names == ("learning_rate", "clip_coef")
    assert sweep.group_settings == cfg.parameter_settings
    np.testing.assert_array_equal(sweep.lr_per_copy, np.float32([1e-4, 1e-4, 3e-4, 3e-4]))
    np.testing.assert_array_equal(sweep.per_copy["clip_coef"], np.float32([.1, .1, .2, .2]))
    assert sweep.beta_per_copy is None and set(sweep.per_copy) == {"learning_rate", "clip_coef"}
    assert sweep.copy_seed_index == [0, 1, 0, 1]

    # a setting that is not per copy is the configuration's Python number, transformed or not
    assert sweep.setting(cfg, "ent_coef") == 0.5 and isinstance(sweep.setting(cfg, "ent_coef"), float)
    assert sweep.setting(cfg, "ent_coef", (4, 1), lambda value: 1 - value) == 0.5
    # a per-copy setting is transformed as Python numbers first, then rounded to float32
    low = sweep.setting(cfg, "clip_coef", (4, 1), lambda value: 1 - value)
    np.testing.assert_array_equal(low, np.float32([[.9], [.9], [.8], [.8]]))
    coef = loss_coefficients(cfg, sweep)
    np.testing.assert_array_equal(coef.clip_high, np.float32([[1.1], [1.1], [1.2], [1.2]]))
    assert coef.ent == 0.5 and coef.vf == cfg.vf_coef

    # the configurations of earlier runs build exactly what they did
    plain = build_sweep(PPOConfig(n_copies=4))
    assert not plain.is_sweep and plain.per_copy == {} and plain.setting_names == (
        "learning_rate", "int_coef")
    scalar = loss_coefficients(PPOConfig(clip_coef=0.2))
    assert (scalar.clip_low, scalar.clip_high, scalar.clip_neg, scalar.clip) == (
        1 - 0.2, 1 + 0.2, -0.2, 0.2)
    print("ok test_named_cells_become_per_copy_vectors")


def test_invalid_names_and_rows_are_refused():
    """Unknown or repeated names, rows of the wrong length and impossible values all fail."""
    good = dict(n_copies=2, copies_per_group=1)
    for bad in (
            dict(setting_names=("learning_rate", "gamma_ext"), parameter_settings=((1e-4, .9),)),
            dict(setting_names=("ent_coef", "ent_coef"), parameter_settings=((.1, .1), (.2, .2))),
            dict(setting_names=("ent_coef",), parameter_settings=((.1, .2), (.3, .4))),
            dict(setting_names=("clip_coef",), parameter_settings=((0.0,), (0.1,))),
            dict(setting_names=("ent_coef",), parameter_settings=((-0.1,), (0.1,))),
            dict(setting_names=("ext_reward_scale",), parameter_settings=((float("inf"),), (1.,))),
            # names without cells: the names describe cells, so they cannot stand alone
            dict(setting_names=("ent_coef",))):
        try:
            build_sweep(PPOConfig(**good, **bad))
        except ValueError as error:
            print(f"  refused: {error}")
            continue
        raise AssertionError(f"accepted {bad}")
    print("ok test_invalid_names_and_rows_are_refused")


def test_uniform_loss_coefficients_equal_the_scalar_program():
    """Entropy, clip and value coefficients per copy, all equal: the scalar program's bits."""
    values = dict(ent_coef=0.01, clip_coef=0.1, vf_coef=0.7)
    for style in ("epoch_minibatch", "full_batch"):
        scalar = run(PPOConfig(n_copies=4, update_style=style, **values, **SMALL))
        vector = run(PPOConfig(n_copies=4, update_style=style, copies_per_group=4,
                               sweep_seed_mode="distinct", setting_names=tuple(values),
                               parameter_settings=(tuple(values.values()),), **SMALL))
        assert_bitwise_equal(scalar, vector, style)
        print(f"  {style}: per-copy entropy, clip and value coefficients are bit-identical to "
              "the scalars")
    print("ok test_uniform_loss_coefficients_equal_the_scalar_program")


def test_uniform_reward_mapping_equals_the_scalar_program():
    """The reward scale and offset per copy, all equal: the scalar program's bits; and a scale of
    1 with an offset of 0 per copy: the bits of the program with no mapping."""
    names = ("ext_reward_scale", "ext_reward_offset")
    per_copy = lambda scale, offset: PPOConfig(
        n_copies=4, copies_per_group=4, sweep_seed_mode="distinct", setting_names=names,
        parameter_settings=((scale, offset),), **SMALL)
    limit = SHORT_EPISODES.max_episode_steps
    scalar = run(PPOConfig(n_copies=4, ext_reward_scale=1 / limit, ext_reward_offset=-1 / limit,
                           **SMALL), env_cfg=SHORT_EPISODES)
    assert_bitwise_equal(scalar, run(per_copy(1 / limit, -1 / limit), env_cfg=SHORT_EPISODES),
                         "step cost")
    unmapped = run(PPOConfig(n_copies=4, **SMALL), env_cfg=SHORT_EPISODES)
    assert_bitwise_equal(unmapped, run(per_copy(1.0, 0.0), env_cfg=SHORT_EPISODES),
                         "scale 1, offset 0")
    # the mapping changes the result: under the step cost the policy learns something else
    assert any(scalar[name].tobytes() != unmapped[name].tobytes() for name in scalar), \
        "the reward mapping had no effect, so the comparison proves nothing"
    print("ok test_uniform_reward_mapping_equals_the_scalar_program")


def test_groups_do_not_influence_each_other():
    """Changing the second group's coefficients and reward leaves the first group's bits alone."""
    names = ("ent_coef", "clip_coef", "vf_coef", "ext_reward_scale", "ext_reward_offset")
    first = (0.001, 0.1, 0.5, 1.0, 0.0)
    build = lambda second: PPOConfig(n_copies=4, copies_per_group=2, setting_names=names,
                                     parameter_settings=(first, second), **SMALL)
    a = run(build((0.01, 0.2, 0.5, 1.0, 0.0)), env_cfg=SHORT_EPISODES)
    b = run(build((0.05, 0.3, 1.0, 0.25, -0.25)), env_cfg=SHORT_EPISODES)
    assert_bitwise_equal(copy_rows(a, slice(0, 2)), copy_rows(b, slice(0, 2)), "unchanged group")
    changed_a, changed_b = copy_rows(a, slice(2, 4)), copy_rows(b, slice(2, 4))
    assert any(changed_a[name].tobytes() != changed_b[name].tobytes() for name in changed_a), \
        "changing the second group's settings had no effect, so the test proves nothing"
    print("ok test_groups_do_not_influence_each_other")


def test_two_reward_settings_on_paired_seeds_start_alike():
    """Reward 1 on arrival and the step cost, paired: copy k of both settings starts from the
    same weights and the same state, and the settings differ in the returns they learn from.

    Paired seeds share the initial weights and the environment's draws. The action noise is
    drawn per copy, so the two settings' rollouts part after the first action; what is identical
    is the first step's observation and the value the critic gives it.
    """
    limit = SHORT_EPISODES.max_episode_steps
    cfg = PPOConfig(n_copies=4, copies_per_group=2,
                    setting_names=("ext_reward_scale", "ext_reward_offset"),
                    parameter_settings=((1.0, 0.0), (1 / limit, -1 / limit)), **SMALL)
    runner = Runner(cfg, env_cfg=SHORT_EPISODES)
    assert runner.sweep.copy_seed_index == [0, 1, 0, 1]
    for weights in jax.tree.leaves(runner.init_trainable()):
        assert (np.asarray(weights)[:2] == np.asarray(weights)[2:]).all(), \
            "the two reward settings did not start from the same weights"
    state = runner.prime(runner.init_state(run_seed=17))
    _, _, batch = jax.jit(runner.composition.iteration_capturing_batch())(
        state, runner.lr_argument(1, 1))
    batch = {name: np.asarray(value) for name, value in batch.items()}
    # before: batch rows [C, T * N], row t * N + n; the first N rows are the first rollout step
    first_step = slice(0, cfg.n_envs)
    for name in ("obs", "vext_old"):
        assert (batch[name][:2, first_step] == batch[name][2:, first_step]).all(), \
            f"{name} of the first step differs between the settings"
    assert (batch["ret_ext"][:2] != batch["ret_ext"][2:]).any(), "the returns do not differ"
    print("ok test_two_reward_settings_on_paired_seeds_start_alike")


def test_selected_cells_and_iteration_blocks():
    """The existing check of selected cells and iteration blocks, at its default block length.

    Its `main` reads the command line, so it is called here with a command line of its own.
    """
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import test_selected_settings_blocks
    command_line = sys.argv
    sys.argv = ["test_selected_settings_blocks.py"]
    try:
        test_selected_settings_blocks.main()
    finally:
        sys.argv = command_line


if __name__ == "__main__":
    test_named_cells_become_per_copy_vectors()
    test_invalid_names_and_rows_are_refused()
    test_uniform_loss_coefficients_equal_the_scalar_program()
    test_uniform_reward_mapping_equals_the_scalar_program()
    test_groups_do_not_influence_each_other()
    test_two_reward_settings_on_paired_seeds_start_alike()
    test_selected_cells_and_iteration_blocks()
