"""Tests of the property measures of the bonus on the maze grid, on hand-made records whose answers are known."""

import numpy as np
import pytest

from maze_record_examples import goal_windows, grid_records, start_record
from maze_campaign.analysis import maze_grid, records


def test_the_slope_of_an_exact_power_law_is_its_exponent():
    """A bonus of 4 / (1 + visits) has slope -1; the window before the first visit is left out."""
    # One policy, three cells, five windows. Cell 0: visits 0, 1, 3, 7, 15 with the bonus 9 before the first
    # visit and 4 / (1 + visits) after it. Cell 1: 3 * (1 + visits)^-0.5 from the first window on. Cell 2: no visit.
    visits = np.asarray([[[0, 1, 0]], [[1, 2, 0]], [[3, 5, 0]], [[7, 9, 0]], [[15, 20, 0]]])
    bonus = np.stack([np.where(visits[:, 0, 0] >= 1, 4.0 / (1 + visits[:, 0, 0]), 9.0),
                      3.0 * (1.0 + visits[:, 0, 1]) ** -0.5, np.full(5, 2.0)], axis=-1)[:, None, :]
    slopes = maze_grid.decay_slopes(bonus, visits)
    assert slopes.shape == (1, 3)
    assert slopes[0, 0] == pytest.approx(-1.0) and slopes[0, 1] == pytest.approx(-0.5)
    assert np.isnan(slopes[0, 2])


def test_a_pair_without_two_different_visit_counts_has_no_slope():
    """One visited window, or visits that never change after the first, give no slope; a zero bonus is left out."""
    visits = np.asarray([[[0, 4, 1]], [[0, 4, 2]], [[5, 4, 4]]])
    bonus = np.asarray([[[1.0, 2.0, 8.0]], [[1.0, 1.0, 0.0]], [[0.5, 0.5, 2.0]]])
    slopes = maze_grid.decay_slopes(bonus, visits)
    assert np.isnan(slopes[0, 0]) and np.isnan(slopes[0, 1])
    # Cell 2: the window with the bonus 0 does not count, so the slope is that of (1 + 1, 8) and (1 + 4, 2).
    assert slopes[0, 2] == pytest.approx(np.log(2.0 / 8.0) / np.log(5.0 / 2.0))


def test_the_spill_over_ratio_is_read_at_pairs_with_no_visit():
    """The bonus now over the bonus at the start, only where the cumulative visits are zero."""
    ratios = maze_grid.unvisited_ratios(np.asarray([[2.0, 4.0], [1.0, 8.0]]), np.asarray([[8.0, 8.0], [8.0, 8.0]]),
                                        np.asarray([[0, 5], [0, 0]]))
    assert ratios.tolist() == [0.25, 0.125, 1.0]
    # Every pair visited: no ratio at all.
    assert len(maze_grid.unvisited_ratios(np.ones((2, 2)), np.ones((2, 2)), np.ones((2, 2), np.int64))) == 0


def test_the_bonus_at_unvisited_cells_over_the_bonus_at_visited_cells():
    """The median of the pairs with no visit over the median of the pairs with one; none when either kind is too few."""
    bonus = np.asarray([[4.0, 0.2], [6.0, 0.3]])
    visits = np.asarray([[0, 5], [0, 9]])
    assert maze_grid.unvisited_over_visited(bonus, visits, 2) == (20.0, 2, 2)
    assert maze_grid.unvisited_over_visited(bonus, visits, 3) == (None, 2, 2)
    # A value that is zero or not a number is in neither kind.
    bonus[1, 0] = float('nan')
    assert maze_grid.unvisited_over_visited(bonus, visits, 1) == (16.0, 1, 2)


def test_the_evenness_ratio_is_the_largest_over_the_smallest_start_bonus():
    """Per policy; a policy with a start bonus that is zero has no ratio."""
    ratios = maze_grid.start_ratio(np.asarray([[0.5, 0.5, 0.5], [1.0, 4.0, 2.0], [0.0, 1.0, 2.0]]))
    assert ratios[:2].tolist() == [1.0, 4.0] and np.isnan(ratios[2])


def test_a_share_of_the_budget_is_read_at_the_last_window_that_ends_inside_it():
    """Window ends 100, 200, 300: step 250 reads the window that ends at 200; a step before the first is refused."""
    steps = np.asarray([100, 200, 300])
    assert maze_grid.window_at(steps, 250) == 1 and maze_grid.window_at(steps, 300) == 2
    with pytest.raises(ValueError, match='No logging window ends at or before step 50'):
        maze_grid.window_at(steps, 50)


def test_the_weighted_intrinsic_reward_is_the_weight_times_the_mean_reward():
    """Mean over policies and cells of 0.1, 0.3, 0.2, 0.2 is 0.2; times the weight 2."""
    assert maze_grid.weighted_intrinsic_reward(np.asarray([[0.1, 0.3], [0.2, 0.2]]), 2.0) == pytest.approx(0.4)


def test_visit_bins_hold_the_median_and_quartiles_per_factor_of_two():
    """1 + visits of 2 and 3 share a bin, 4 and 6 the next; a point with no visit is left out."""
    bins = maze_grid.visit_bins(np.asarray([9.0, 8.0, 4.0, 2.0, 1.0]), np.asarray([0, 1, 2, 3, 5]))
    assert bins['one_plus_visits'].tolist() == [2.5, 5.0] and bins['median'].tolist() == [6.0, 1.5]
    assert bins['low'].tolist() == [5.0, 1.25] and bins['high'].tolist() == [7.0, 1.75]
    assert bins['points'].tolist() == [2, 2]


def grid_unit(windows=3):
    """A unit of 2 policies and 2 open cells: cell 0 is visited once per window, cell 1 never."""
    start = start_record('grid unit', iterations=windows, probe_cells=[(1, 1), (1, 2)])
    return grid_records(start, goal_windows(start, [None, None]), [8.0, 2.0],
                        lambda index: [8.0 / (index + 2), 1.0], lambda index: [1, 0], divisor=2.0)


def test_a_units_maze_grid_records_become_arrays_with_cumulative_visits():
    """Shapes are (windows, policies, cells); the visits are summed over the windows."""
    grid = maze_grid.read_grid(records.latest_attempt(grid_unit()))
    assert grid.cells == ((1, 1), (1, 2)) and grid.copy_cell.tolist() == [0, 0]
    assert grid.steps.tolist() == [100, 200, 300]
    assert grid.start_bonus.tolist() == [[8.0, 2.0]] * 2 and grid.start_reward.tolist() == [[4.0, 1.0]] * 2
    assert grid.bonus.shape == (3, 2, 2) and grid.bonus[:, 0, 0].tolist() == [4.0, 8.0 / 3, 2.0]
    assert grid.reward[:, 1, 1].tolist() == [0.5, 0.5, 0.5]
    assert grid.start_divisor.tolist() == [2.0, 2.0] and grid.divisor.tolist() == [[2.0, 2.0]] * 3
    assert grid.visits[:, 0, 0].tolist() == [1, 2, 3] and grid.visits[:, 0, 1].tolist() == [0, 0, 0]
    assert maze_grid.problems(grid) == {}


def test_a_unit_without_the_maze_grid_records_is_refused():
    """No record at the start, or a window without the fields: the error names the unit and the option."""
    unit = grid_unit()
    with pytest.raises(ValueError, match='grid unit has 0 maze_grid_at_start records'):
        maze_grid.read_grid([record for record in unit if record['record'] != 'maze_grid_at_start'])
    del unit[-1]['probe_bonus_per_copy']
    with pytest.raises(ValueError, match="'probe_bonus_per_copy'"):
        maze_grid.read_grid(unit)


def test_values_the_measures_cannot_use_are_counted():
    """A zero and a value that is not a number in the windows, a zero at the start."""
    unit = grid_unit()
    unit[1]['probe_bonus_per_copy'][0][1] = 0.0
    unit[2]['probe_bonus_per_copy'][1][0] = float('nan')
    unit[3]['probe_bonus_per_copy'][1][1] = 0.0
    found = maze_grid.problems(maze_grid.read_grid(unit))
    assert found == {'start bonus values that are zero or negative': 1,
                     'bonus values in the logging windows that are not finite': 1,
                     'bonus values in the logging windows that are zero or negative': 1}
