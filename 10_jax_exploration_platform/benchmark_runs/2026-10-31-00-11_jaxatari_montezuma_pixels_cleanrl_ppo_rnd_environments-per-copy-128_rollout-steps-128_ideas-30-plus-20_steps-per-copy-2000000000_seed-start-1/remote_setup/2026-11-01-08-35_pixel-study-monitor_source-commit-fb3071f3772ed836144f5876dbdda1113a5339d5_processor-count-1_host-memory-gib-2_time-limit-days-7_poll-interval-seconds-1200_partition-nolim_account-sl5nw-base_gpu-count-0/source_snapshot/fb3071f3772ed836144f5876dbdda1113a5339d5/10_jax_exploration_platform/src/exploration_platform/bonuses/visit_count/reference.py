"""The visit-count bonus written out as a plain loop, to check the fused form against.

Same semantics, no array tricks: for each copy, walk the rollout's rows in order and add one to
the entry of every open cell reached; then walk them again and read each row's bonus out of the
finished table. Nothing here is meant to be fast — it is meant to be obviously the definition.
"""
import numpy as np


def table_index(observation, wall, velocity_bins: int, velocity_clip: float):
    """One observation (x, y, vx, vy) -> (flat table index, is the cell open).

    before: (-4.5, -3.0, 0.0, 0.0) on the large maze (9 rows, 12 columns)
    after:  column 1, row 7 (row 0 is the top), velocity bins 5 and 5, index
            (7*12 + 1)*100 + 5*10 + 5 = 8555, open
    """
    rows, cols = wall.shape
    x, y, vx, vy = (np.float32(v) for v in observation[:4])

    # position: the world is centred on the origin, one cell is one metre, row 0 is the top
    column = int(np.clip(np.int32(x + np.float32(cols / 2.0)), 0, cols - 1))
    row = int(np.clip(np.int32(np.float32(rows / 2.0) - y), 0, rows - 1))

    # velocity: clipped to the environment's own limit, then cut into equal bins per axis
    span = np.float32(2 * velocity_clip)
    bx = int(np.clip(np.int32((np.clip(vx, -velocity_clip, velocity_clip)
                               + np.float32(velocity_clip)) / span * velocity_bins),
                     0, velocity_bins - 1))
    by = int(np.clip(np.int32((np.clip(vy, -velocity_clip, velocity_clip)
                               + np.float32(velocity_clip)) / span * velocity_bins),
                     0, velocity_bins - 1))

    cell = row * cols + column
    per_cell = velocity_bins * velocity_bins
    return cell * per_cell + bx * velocity_bins + by, bool(wall[row, column] == 0)


def post_rollout(counts, next_obs, wall, decay: float, velocity_bins: int = 10,
                 velocity_clip: float = 5.0):
    """Count a whole rollout, then score its rows. counts [C, table], next_obs [C, M, 4].

    Returns (the counts after this rollout, the bonus of every row [C, M] float32).
    """
    counts = np.array(counts, dtype=np.int32, copy=True)
    n_copies, n_rows = next_obs.shape[0], next_obs.shape[1]
    bonus = np.zeros((n_copies, n_rows), dtype=np.float32)

    # pass one: every open-cell visit adds one to its own entry; a wall visit adds nothing
    index = np.zeros((n_copies, n_rows), dtype=np.int64)
    is_open = np.zeros((n_copies, n_rows), dtype=bool)
    for c in range(n_copies):
        for r in range(n_rows):
            index[c, r], is_open[c, r] = table_index(next_obs[c, r], wall, velocity_bins,
                                                     velocity_clip)
            if is_open[c, r]:
                counts[c, index[c, r]] += 1

    # pass two: the bonus of a row is read from the finished table, so a state reached once in
    # this rollout has count 1 and scores 1.0; a wall cell scores 1.0 because it was never counted
    for c in range(n_copies):
        for r in range(n_rows):
            if not is_open[c, r]:
                bonus[c, r] = 1.0
                continue
            n = max(int(counts[c, index[c, r]]), 1)
            bonus[c, r] = min(np.float32(1.0), np.float32(n) ** np.float32(decay))
    return counts, bonus
