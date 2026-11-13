"""The bonus recorded on the maze grid: one unit's records as arrays, and the property measures read from them.

With the bonus option `record_bonus_on_maze_grid` the trainer (`training/maze_grid.py`) writes, per policy and per
open maze cell: the bonus before the divisor at one probe observation (the cell's centre, zero velocity) as the
bonus stands at the end of each logging window, the same after the divisor, and the number of states the window's
rollouts reached in the cell. One record written after the random-action rollouts and before the first training
iteration (`maze_grid_at_start`) holds the same fields: the bonus's start on the grid.

Measures, with b the bonus before the divisor and n the cumulative visits to a cell:

- start ratio of a policy: the largest over the smallest start bonus over the cells;
- decay slope of a policy and cell: the least-squares slope of log b on log(1 + n), over the logging windows at
  whose end the cell has at least one cumulative visit;
- unvisited ratio at a logging window: for every (policy, cell) pair with no visit so far, the bonus now over the
  bonus at the start;
- weighted intrinsic reward: intrinsic weight x the bonus after the divisor, averaged over cells and policies.

A value the logarithm or a ratio cannot use (not finite, or not above zero) is left out of the measure and counted
by `problems`, so the summary can name it.
"""

from dataclasses import dataclass

import numpy as np

WINDOW_FIELDS = ('probe_bonus_per_copy', 'probe_intrinsic_reward_per_copy', 'intrinsic_reward_divisor_per_copy',
                 'cell_visits_per_copy')


@dataclass(frozen=True, eq=False)
class Grid:
    """One unit's maze-grid records: per policy and open cell, the start and every logging window."""
    unit_id: str
    cells: tuple               # the open cells as (row, column), in the order of the arrays' last axis
    copy_cell: np.ndarray      # [policies] the index of each policy's cell of settings
    steps: np.ndarray          # [windows] environment steps per policy at each window's end
    start_bonus: np.ndarray    # [policies, cells] the bonus before the divisor, before the first training iteration
    start_reward: np.ndarray   # [policies, cells] the same after the divisor
    start_divisor: np.ndarray  # [policies] the divisor of the intrinsic reward in that record
    bonus: np.ndarray          # [windows, policies, cells] the bonus before the divisor at each window's end
    reward: np.ndarray         # [windows, policies, cells] the same after the divisor
    divisor: np.ndarray        # [windows, policies] the divisor of the intrinsic reward at each window's end
    visits: np.ndarray         # [windows, policies, cells] cumulative visits to each cell by each window's end


def read_grid(attempt):
    """The maze-grid fields of a unit's latest attempt as arrays; refused when the unit did not record them."""
    start = attempt[0]
    at_start = [record for record in attempt if record['record'] == 'maze_grid_at_start']
    windows = [record for record in attempt if record['record'] == 'episode_window']
    if 'maze_grid_probe_cells' not in start or len(at_start) != 1:
        raise ValueError(f"Unit {start['unit_id']} has {len(at_start)} maze_grid_at_start records and "
                         f"{'a' if 'maze_grid_probe_cells' in start else 'no'} list of probe cells; the property "
                         'measures need a unit trained with the bonus option record_bonus_on_maze_grid=true.')
    missing = [name for name in WINDOW_FIELDS if any(name not in window for window in windows)]
    if missing or not windows:
        raise ValueError(f"Unit {start['unit_id']}: {len(windows)} logging windows, and the maze-grid fields "
                         f'{missing} are missing from at least one of them.')

    # before: per window a list [policies][cells] of the window's own visits;
    # after:  an array [windows, policies, cells] of the visits summed from the first window to each window.
    visits = np.cumsum(np.asarray([window['cell_visits_per_copy'] for window in windows], np.int64), axis=0)
    grid = Grid(
        unit_id=start['unit_id'], cells=tuple(tuple(cell) for cell in start['maze_grid_probe_cells']),
        copy_cell=np.asarray(start['copy_cell'], np.int64),
        steps=np.asarray([window['env_steps_per_copy'] for window in windows], np.int64),
        start_bonus=np.asarray(at_start[0]['probe_bonus_per_copy'], np.float64),
        start_reward=np.asarray(at_start[0]['probe_intrinsic_reward_per_copy'], np.float64),
        start_divisor=np.asarray(at_start[0]['intrinsic_reward_divisor_per_copy'], np.float64),
        bonus=np.asarray([window['probe_bonus_per_copy'] for window in windows], np.float64),
        reward=np.asarray([window['probe_intrinsic_reward_per_copy'] for window in windows], np.float64),
        divisor=np.asarray([window['intrinsic_reward_divisor_per_copy'] for window in windows], np.float64),
        visits=visits)
    expected = (len(windows), start['copies'], len(grid.cells))
    for name in ('bonus', 'reward', 'visits'):
        if getattr(grid, name).shape != expected:
            raise ValueError(f"Unit {start['unit_id']}: the maze-grid field {name} has the shape "
                             f'{getattr(grid, name).shape}, not {expected} (windows, policies, open cells).')
    if grid.start_bonus.shape != expected[1:] or np.any(np.asarray(at_start[0]['cell_visits_per_copy']) != 0):
        raise ValueError(f"Unit {start['unit_id']}: the maze_grid_at_start record has the shape "
                         f'{grid.start_bonus.shape} (expected {expected[1:]}) or holds visits before training.')
    return grid


def usable(values):
    """Where a bonus can enter a logarithm or the bottom of a ratio: finite and above zero."""
    return np.isfinite(values) & (values > 0)


def start_ratio(start_bonus):
    """Per policy, the largest over the smallest start bonus over the cells; nan when a value is not usable."""
    # before: [[0.5, 0.5, 0.5], [1.0, 4.0, 2.0], [0.0, 1.0, 2.0]]; after: [1.0, 4.0, nan].
    whole = usable(start_bonus).all(axis=-1)
    smallest = np.where(whole, start_bonus.min(axis=-1), 1.0)
    return np.where(whole, start_bonus.max(axis=-1) / smallest, np.nan)


def decay_slopes(bonus, visits):
    """Per policy and cell, the least-squares slope of log(bonus) on log(1 + cumulative visits) over visited windows."""
    # before: one policy and cell over four windows: cumulative visits [0, 1, 3, 7], bonus [9, 2, 1, 0.5], which is
    #         4 / (1 + visits) from the first visit on; after: slope -1.0, the window before the first visit left out.
    # A pair has no slope (nan) when fewer than two windows count, or when its counted windows all hold the same
    # number of visits.
    counted = (visits >= 1) & usable(bonus)
    x = np.log1p(visits.astype(np.float64))
    y = np.log(np.where(counted, bonus, 1.0))
    number = counted.sum(axis=0)
    mean_x = (x * counted).sum(axis=0) / np.maximum(number, 1)
    mean_y = (y * counted).sum(axis=0) / np.maximum(number, 1)
    spread = (counted * (x - mean_x) ** 2).sum(axis=0)
    product = (counted * (x - mean_x) * (y - mean_y)).sum(axis=0)
    # The visits differ among the counted windows exactly when the last counted value is above the first.
    differ = np.where(counted, x, -np.inf).max(axis=0) > np.where(counted, x, np.inf).min(axis=0)
    fitted = (number >= 2) & differ
    return np.where(fitted, product / np.where(fitted, spread, 1.0), np.nan)


def unvisited_ratios(bonus_now, start_bonus, visits_now):
    """The bonus now over the bonus at the start, for every (policy, cell) pair with no visit so far."""
    # before: start [[8, 8], [8, 8]], now [[2, 4], [1, 8]], cumulative visits [[0, 5], [0, 0]]
    # after:  [0.25, 0.125, 1.0]: the pair with 5 visits is left out.
    unvisited = (visits_now == 0) & usable(start_bonus) & np.isfinite(bonus_now)
    return bonus_now[unvisited] / start_bonus[unvisited]


def unvisited_over_visited(bonus_now, visits_now, fewest):
    """The median bonus of the (policy, cell) pairs with no visit so far over the median of the pairs with a visit.

    Returns (the ratio, the number of usable pairs without a visit, the number with a visit); the ratio is None
    when either kind has fewer than `fewest` pairs.
    """
    # before: bonus [[4.0, 0.2], [6.0, 0.3]], cumulative visits [[0, 5], [0, 9]], fewest 2
    # after:  (5.0 / 0.25 = 20.0, 2, 2).
    good = usable(bonus_now)
    never, seen = bonus_now[good & (visits_now == 0)], bonus_now[good & (visits_now > 0)]
    ratio = float(np.median(never) / np.median(seen)) if min(len(never), len(seen)) >= fewest else None
    return ratio, len(never), len(seen)


def window_at(steps, at_steps):
    """The index of the last logging window that ends at or before a step count."""
    # before: window ends [100, 200, 300], step count 250; after: 1, the window that ends at step 200.
    index = int(np.searchsorted(steps, at_steps, side='right')) - 1
    if index < 0:
        raise ValueError(f'No logging window ends at or before step {at_steps}; the first ends at step {int(steps[0])}.')
    return index


def weighted_intrinsic_reward(reward, weight):
    """Intrinsic weight x the bonus after the divisor, the mean over the given policies and cells."""
    # before: reward [[0.1, 0.3], [0.2, 0.2]], weight 2; after: 0.4.
    return float(weight * np.mean(reward))


def visit_bins(bonus, visits):
    """The median and quartiles of the bonus in bins of 1 + cumulative visits, each bin a factor of two wide."""
    # before: points (cumulative visits, bonus) = (0, 9), (1, 8), (2, 4), (3, 2), (5, 1)
    # after:  bin [2, 4) holds 1 + visits = 2 and 3 (bonus 8 and 4): median 6, drawn at 2.5, the median of the
    #         bin's 1 + visits; bin [4, 8) holds 4 and 6 (bonus 2 and 1): median 1.5 at 5; the point with no
    #         visit is left out.
    counted = (visits >= 1) & usable(bonus)
    x, y = 1.0 + visits[counted], bonus[counted]
    order = np.floor(np.log2(x)).astype(np.int64)
    rows = []
    for index in np.unique(order):
        inside = order == index
        low, median, high = np.quantile(y[inside], [0.25, 0.5, 0.75])
        rows.append((float(np.median(x[inside])), float(median), float(low), float(high), int(inside.sum())))
    names = ('one_plus_visits', 'median', 'low', 'high', 'points')
    return {name: np.asarray([row[position] for row in rows]) for position, name in enumerate(names)}


def problems(grid):
    """How many recorded values the measures cannot use, by kind; kinds with none are left out."""
    counts = {
        'start bonus values that are not finite': int((~np.isfinite(grid.start_bonus)).sum()),
        'start bonus values that are zero or negative': int((grid.start_bonus <= 0).sum()),
        'bonus values in the logging windows that are not finite': int((~np.isfinite(grid.bonus)).sum()),
        'bonus values in the logging windows that are zero or negative': int((grid.bonus <= 0).sum()),
        'intrinsic rewards that are not finite': int((~np.isfinite(grid.reward)).sum()
                                                     + (~np.isfinite(grid.start_reward)).sum())}
    return {kind: count for kind, count in counts.items() if count}
