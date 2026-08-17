# Oracle visit counts over position and velocity — `bonus_spec@1`

The identity a run's manifest carries for this family: **`gt_position_velocity@1`** with update
schedule **`post_rollout_update@1`**. Changing anything below is a new version number, not an edit
of this one.

## The counted state

A copy's knowledge is one integer table. The counted state is the discretised position and
velocity of the observation the environment returned, `(x, y, vx, vy)`, exactly as
`07_reconstruction`'s `PositionVelocityVisitCountWrapper` discretises it:

| part | rule |
|---|---|
| column | `clip(int(x + cols/2), 0, cols-1)` — the world is centred on the origin and one cell is one metre |
| row | `clip(int(rows/2 - y), 0, rows-1)` — row 0 is the TOP row, matching the maze map's own indexing |
| velocity bin, per axis | the component is clipped to the environment's own limit, +-5 m/s, and the range is cut into 10 equal bins: `clip(int((v + 5)/10 * 10), 0, 9)` |

The conversion to an integer truncates toward zero. That differs from rounding down only for
negative values, and every negative value here is clipped to 0 anyway, so the two agree on every
input.

## The table

Per copy, an `int32` array of `rows * cols * 100` entries — 10,800 for the large maze, 43 KB per
copy, about 0.35 GB at 8,448 copies. The index is

```
(row * cols + column) * 100 + velocity_bin_x * 10 + velocity_bin_y
```

Wall cells get entries too. They are never incremented and never read, and giving them their own
slots keeps the index arithmetic a plain multiply-add instead of a lookup through a compaction
table.

## The bonus

```
bonus = min(1, n**decay)     for an open cell with count n
bonus = 1.0                  for a wall cell
```

`decay` is negative, so the bonus falls as a state is revisited: **-0.5** gives `1/sqrt(n)` (preset
`gt_position_velocity_sqrt`) and **-1** gives `1/n` (preset `gt_position_velocity_linear`).

A count of zero would give `0**decay = infinity`, so the count is floored at 1 before the power.
Under the update schedule below that floor is never reached — every open row of a rollout has been
counted at least once by the time it is scored — but the arithmetic does not depend on that
argument being right.

A wall cell scoring 1.0 is `07_reconstruction`'s convention: an unreachable or unvisited state
counts as zero visits and therefore earns the maximum exploration bonus. The environment's contact
model keeps the ball out of walls, so this is an edge case about the ball's centre being within a
cell's bounds while resting against it, not a common outcome.

## The update schedule: `post_rollout_update@1`

**Count the whole rollout first, then score it.** One scatter adds the rollout's `C x T*N` visits
into the table; one gather reads the updated counts back for the same rows. Two array operations
per iteration, no sequential dependence along the rollout, and no per-step work at all.

The consequence to be aware of: a state reached once in this rollout is scored with count 1, not
count 0, so the largest bonus any visited state can earn is `min(1, 1**decay) = 1.0` — the same
value an unvisited state would have earned. States reached repeatedly within one rollout are
scored with the total, so a copy that spends a rollout in one corner sees its bonus there fall
inside that rollout rather than only in the next one.

This matches what `07_reconstruction` effectively does. There, the environment wrapper increments
the count at every step as the episode runs, and the bonus is computed later, when a batch is
sampled from the replay buffer — after collection, from a table that already holds those steps.
The per-step ordering inside one collection phase is not part of what that implementation
guarantees, and the batch form is the one that fuses.

## What is checked

| check | where |
|---|---|
| the fused form and a plain python loop agree on the same supplied trajectories | `tests/bonuses/test_visit_count.py` |
| the count table and the bonuses agree with `07_reconstruction`'s own wrapper on a fixed trajectory, its code run in its own interpreter | `tests/parity_07/test_visit_count_against_07.py` |
| one copy's table never moves another copy's | `tests/copy_isolation/test_visit_count_isolation.py` |
| the bonus falls where visits accumulate, and coverage grows | `tests/bonuses/test_visit_count_learning_sanity.py` |
