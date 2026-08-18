"""How many of a configuration's copies share the shape its AGGREGATE curve was given.

The aggregate's shape is not necessarily any single copy's shape: averaging 128 copies can produce
a peak that no copy has, and can hide a peak that most of them do. `curve_shape.copy_level_agreement`
measures that, and this module is what feeds it — it extracts one curve per copy for a named
configuration, which nothing else in the run needs and so nothing else builds.

One pass over the arm's shards, keeping a [copies x windows] array of episode returns: 128 copies
by 9,766 windows is about 10 MB, so the per-copy view is affordable for the handful of
configurations a writeup names, and only for those.

Run:
  PYTHONNOUSERSITE=1 /p/rlprojects/RND/.venvs/platform_jax/bin/python analysis/code/copy_agreement.py
"""
import sys
from pathlib import Path

RUN_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(RUN_DIR / "code"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np                                                       # noqa: E402

from aggregate import cell_copies, live_lines, read_selected             # noqa: E402
from curve_shape import classify, copy_level_agreement                   # noqa: E402

# the four configurations the writeup names: each arm's best, and each arm's best that is NOT
# rise-then-fall
NAMED = [
    ("rnd_next_state", 1e-4, 1e2, "best of its arm"),
    ("gt_position_velocity_sqrt", 1e-4, 1e1, "best of its arm"),
    ("rnd_next_state", 1e-5, 1e1, "best of its arm that is not rise-then-fall"),
    ("gt_position_velocity_sqrt", 1e-4, 1e2, "best of its arm that is not rise-then-fall"),
]


def per_copy_curves(bonus: str, rate: float, weight: float, run_dir: Path = RUN_DIR) -> np.ndarray:
    """One row per copy of this configuration, one column per window, as episode returns.

    before: the arm's shards, each holding this configuration's copies at some slice of the indices;
    after:  an array of shape (128, 9766) — every copy of the configuration, in copy-index order
    """
    setting = (rate, weight)
    blocks = []
    for shard in sorted((run_dir / "data").glob("*.jsonl")):
        live = live_lines(shard)
        if live["start"] is None:
            continue
        _, start = next(read_selected(shard, {live["start"]}))
        if start["bonus"] != bonus:
            continue
        positions = cell_copies(start).get(setting)
        if not positions:
            continue
        rows = {}
        for _, record in read_selected(shard, set(live["windows"].values())):
            episodes = record["episodes_per_copy_in_window"]
            rows[record["last_iteration"]] = [
                record["reward_ext_sum_per_copy"][position] / episodes for position in positions]
        ordered = [rows[key] for key in sorted(rows)]
        blocks.append((start["copy_seed_index_first"], np.asarray(ordered, dtype=float).T))
    blocks.sort(key=lambda item: item[0])
    return np.concatenate([block for _, block in blocks], axis=0)


def main() -> None:
    """Print, per named configuration, how many of its copies share the aggregate's label."""
    print(f"{'arm':28} {'lr':>7} {'beta':>7} {'aggregate':18} {'copies':>7}  copies by their own label")
    for bonus, rate, weight, _ in NAMED:
        curves = per_copy_curves(bonus, rate, weight)
        aggregate_mean = curves.mean(axis=0)
        aggregate_error = curves.std(axis=0, ddof=1) / np.sqrt(curves.shape[0])
        label = classify(aggregate_mean, aggregate_error)["shape"]
        agreement = copy_level_agreement(list(curves))
        counts = ", ".join(f"{name} {n}" for name, n in
                           sorted(agreement["counts"].items(), key=lambda kv: -kv[1]))
        print(f"{bonus:28} {rate:>7.0e} {weight:>7.0e} {label:18} {agreement['copies']:>7}  "
              f"{counts}")


if __name__ == "__main__":
    main()
