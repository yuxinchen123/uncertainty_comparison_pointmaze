"""Check deterministic maps and frozen initial bonus fields on a compute node."""

import numpy as np
import pytest

from exploration_platform.grid_initialization.environments import (
    ACTION_NAMES,
    make_geometry,
    make_snake_grid_world,
    make_straight_grid_world,
)
from exploration_platform.grid_initialization.fields import (
    ADVERSARIAL_FIELD_KINDS,
    derive_quantity_seed,
    make_initial_bonus_field,
    make_random_network_bonus_field,
)


def test_straight_grid_has_exact_actions_and_blocked_stays():
    """Compare every transition with a hand-written four-cell corridor."""
    # Rows are state identifiers and columns are up, right, down, left.
    geometry = make_straight_grid_world(4)
    expected = np.asarray([[0, 1, 0, 0], [1, 2, 1, 0],
                           [2, 3, 2, 1], [3, 3, 3, 2]], dtype=np.int32)
    np.testing.assert_array_equal(geometry.transition_table, expected)
    np.testing.assert_array_equal(geometry.grid, [[0, 1, 2, 3]])
    np.testing.assert_allclose(geometry.coordinates, [[-1, 0], [-1 / 3, 0], [1 / 3, 0], [1, 0]])
    assert ACTION_NAMES == ("up", "right", "down", "left")
    assert geometry.state_count == geometry.n_states == 4
    assert geometry.start_state == 0 and geometry.goal_state == 3


def test_snake_has_a_single_path_and_bottom_right_goal():
    """Follow an explicit right/down/left/down/right trajectory through a snake."""
    # Width four and two turns produce three corridors and two joining cells.
    geometry = make_snake_grid_world(4, 2)
    np.testing.assert_array_equal(geometry.grid,
                                 [[0, 1, 2, 3], [-1, -1, -1, 4], [8, 7, 6, 5],
                                  [9, -1, -1, -1], [10, 11, 12, 13]])
    forward_actions = [1, 1, 1, 2, 2, 3, 3, 3, 2, 2, 1, 1, 1]
    state = geometry.start_state
    for expected_state, action in enumerate(forward_actions, 1):
        state = int(geometry.transition_table[state, action])
        assert state == expected_state
    assert state == geometry.goal_state == 13
    np.testing.assert_array_equal(geometry.positions[state], [4, 3])

    # Every legal movement changes path distance by one; every other action stays put.
    distances = geometry.path_distance[geometry.transition_table] - geometry.path_distance[:, None]
    assert np.all(np.abs(distances) <= 1)
    assert np.all(np.sum(distances != 0, axis=1)[1:-1] == 2)
    assert np.all(np.sum(distances != 0, axis=1)[[0, -1]] == 1)
    np.testing.assert_array_equal(geometry.path_distance, np.arange(14))
    assert np.all(np.linalg.norm(geometry.coordinates, axis=1) > 0)


def test_straight_and_snake_have_matching_path_transitions():
    """Check the path-graph control after per-state action relabeling."""
    # A 14-cell straight corridor and a 4-by-three-corridor snake share one path graph.
    straight = make_straight_grid_world(14)
    snake = make_snake_grid_world(4, 2)
    np.testing.assert_array_equal(np.sort(straight.transition_table, axis=1),
                                  np.sort(snake.transition_table, axis=1))


@pytest.mark.parametrize("kind,width,turns", [
    ("straight", 1, 0), ("straight", 3, 0), ("straight", 4, 2),
    ("snake", 3, 2), ("snake", 4, 0), ("snake", 4, 3), ("missing", 4, 0),
])
def test_geometry_rejects_settings_that_do_not_match_the_recorded_map(kind, width, turns):
    """Reject widths with a zero-origin cell, inconsistent turns, and unknown kinds."""
    # Invalid settings stop before any learner is initialized.
    with pytest.raises(ValueError):
        make_geometry(kind, width, turns)


@pytest.mark.parametrize("kind", ["uniform", "random_log_uniform", *ADVERSARIAL_FIELD_KINDS])
def test_nonzero_fields_are_strictly_positive_and_mean_one(kind):
    """Keep all ten adversarial fields on the same initial average reward scale."""
    # Before: an arbitrary positive shape; after: 64 positive values averaging one.
    result = make_initial_bonus_field(kind, make_geometry("straight", 64), seed=7)
    assert result.raw.shape == result.normalized.shape == (64,)
    assert result.raw.dtype == result.normalized.dtype == np.float32
    assert np.all(result.raw > 0) and np.all(result.normalized > 0)
    assert np.all(np.isfinite(result.normalized))
    np.testing.assert_allclose(result.normalized.mean(), 1, rtol=5e-7)
    np.testing.assert_allclose(result.normalized.astype(np.float64).mean(), 1, rtol=1e-7)
    assert result.metadata["kind"] == kind
    assert result.metadata["raw_mean"] == float(result.raw.astype(np.float64).mean())


def test_high_start_field_matches_hand_computed_normalization():
    """Check the exact mean-one transformation for a single high start cell."""
    # Before: [4,1,1,1], mean 7/4; after: [16/7,4/7,4/7,4/7].
    field = make_initial_bonus_field("high_start_cell", make_straight_grid_world(4), severity_exponent=2)
    np.testing.assert_array_equal(field.raw, [4, 1, 1, 1])
    np.testing.assert_allclose(field.normalized, [16 / 7, 4 / 7, 4 / 7, 4 / 7])


def test_none_field_has_exactly_zero_reward():
    """Keep the no-bonus comparison separate from positive mean-one fields."""
    # Both the archived and supplied no-bonus arrays contain exact zeros.
    field = make_initial_bonus_field("none", make_straight_grid_world(4))
    np.testing.assert_array_equal(field.raw, np.zeros(4))
    np.testing.assert_array_equal(field.normalized, np.zeros(4))


def test_random_field_seeds_are_independent_and_growth_preserves_raw_prefix():
    """Keep existing random state values fixed when more states are appended."""
    # The raw 16-cell draw must be the first 16 entries of the 32-cell draw.
    short = make_initial_bonus_field("random_log_uniform", make_straight_grid_world(16), seed=23)
    long = make_initial_bonus_field("random_log_uniform", make_straight_grid_world(32), seed=23)
    other = make_initial_bonus_field("random_log_uniform", make_straight_grid_world(16), seed=24)
    np.testing.assert_array_equal(short.raw, long.raw[:16])
    assert not np.array_equal(short.raw, other.raw)
    original_seed = derive_quantity_seed(23, "target", "layer_0")
    assert original_seed != derive_quantity_seed(23, "predictor", "layer_0")
    derive_quantity_seed(23, "new_quantity")
    assert original_seed == derive_quantity_seed(23, "target", "layer_0")


def test_frozen_network_field_is_positive_reproducible_and_archived():
    """Check independent untrained networks and the explicit coordinate-origin control."""
    # This test performs JAX initialization and belongs on an allocated compute node.
    geometry = make_snake_grid_world(4, 2)
    first = make_random_network_bonus_field(geometry, seed=13)
    repeated = make_random_network_bonus_field(geometry, seed=13)
    control = make_random_network_bonus_field(geometry, seed=13, constant_coordinate=True)
    np.testing.assert_array_equal(first.raw, repeated.raw)
    assert first.raw.shape == (14,) and np.all(first.raw > 0)
    np.testing.assert_allclose(first.normalized.astype(np.float64).mean(), 1, rtol=2e-7)
    assert not np.array_equal(first.raw, control.raw)
    assert first.metadata["target_widths"] == [2, 32, 64, 64, 512]
    assert first.metadata["predictor_widths"] == [2, 32, 64, 64, 512, 512, 512]
    assert first.metadata["trained"] is False
    assert control.metadata["constant_coordinate"] is True


@pytest.mark.parametrize("kind,settings", [
    ("missing", {}), ("uniform", {"severity_exponent": -1}),
    ("high_start_region", {"width_exponent": 8}),
    ("low_middle_interval", {"width_exponent": 8}),
    ("repeated_low_intervals", {"width_exponent": 8}),
])
def test_fields_reject_invalid_names_and_nonexistent_regions(kind, settings):
    """Reject a field that cannot have its recorded shape on the chosen map."""
    # Reject instead of silently clipping a region or replacing a failed field.
    with pytest.raises(ValueError):
        make_initial_bonus_field(kind, make_straight_grid_world(16), **settings)
