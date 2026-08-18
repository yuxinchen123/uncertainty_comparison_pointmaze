"""Every configuration's curve from ONE pass over the shards, instead of one pass per curve.

`aggregate.curve_of` streams the shards afresh for each configuration it is asked for. That is the
right shape for a figure, which wants five curves; it is the wrong shape for the shape
classification, which wants all 66 and would read about 4 GB sixty-six times.

This reads each shard once and carries a running count, sum and sum of squares per (configuration,
window), so it produces exactly the numbers `curve_of` produces — the identity is checked by
`verify_against_curve_of` below, and by the gate that runs it before the classification.
"""
import math
import sys
from pathlib import Path

RUN_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(RUN_DIR / "code"))

from aggregate import cell_copies, live_lines, read_selected            # noqa: E402


def curves_all(run_dir: Path = RUN_DIR) -> dict:
    """Map (bonus, learning rate, intrinsic weight) -> the same dict `curve_of` returns.

    before: 66 configurations x a full pass over every shard of the arm;
    after:  one pass over every shard, all 66 configurations accumulated together
    """
    chunks_seen, per_setting = {}, {}
    for shard in sorted((run_dir / "data").glob("*.jsonl")):
        live = live_lines(shard)
        if live["start"] is None:
            continue
        _, start = next(read_selected(shard, {live["start"]}))
        bonus = start["bonus"]
        positions_of = cell_copies(start)
        for setting in positions_of:
            chunks_seen[(bonus, *setting)] = chunks_seen.get((bonus, *setting), 0) + 1
        for _, record in read_selected(shard, set(live["windows"].values())):
            episodes = record["episodes_per_copy_in_window"]
            rewards = record["reward_ext_sum_per_copy"]
            iteration = record["last_iteration"]
            for setting, positions in positions_of.items():
                entry = per_setting.setdefault((bonus, *setting), {}).setdefault(
                    iteration, {"n": 0, "sum": 0.0, "square_sum": 0.0, "chunks": 0,
                                "env_steps_per_copy": record["env_steps_per_copy"],
                                "phase_blocked": True})
                total = square_total = 0.0
                for position in positions:
                    value = rewards[position] / episodes
                    total += value
                    square_total += value * value
                entry["n"] += len(positions)
                entry["sum"] += total
                entry["square_sum"] += square_total
                entry["chunks"] += 1
                entry["phase_blocked"] = entry["phase_blocked"] and record["phase_blocked"]

    out = {}
    for key, per_window in per_setting.items():
        steps, means, errors, blocked = [], [], [], []
        for iteration in sorted(per_window):
            entry = per_window[iteration]
            if entry["chunks"] != chunks_seen[key]:
                continue
            n = entry["n"]
            mean = entry["sum"] / n
            variance = (max(0.0, (entry["square_sum"] - n * mean * mean) / (n - 1))
                        if n > 1 else 0.0)
            steps.append(entry["env_steps_per_copy"])
            means.append(mean)
            errors.append(math.sqrt(variance / n))
            blocked.append(entry["phase_blocked"])
        out[key] = {"env_steps_per_copy": steps, "mean_episode_return": means,
                    "standard_error": errors, "phase_blocked": blocked}
    return out


def verify_against_curve_of(run_dir: Path = RUN_DIR, samples: int = 3) -> dict:
    """Gate: a few configurations recomputed the slow way must match this one exactly.

    The two readers necessarily run minutes apart, and on a live run more windows are written in
    between, so the comparison is over the windows BOTH readers saw — matched by their step count,
    never by position in the list. Requiring equal lengths would report a growing run as a defect.
    """
    from aggregate import curve_of
    everything = curves_all(run_dir)
    checked, worst, compared = 0, 0.0, 0
    for key in sorted(everything)[:: max(1, len(everything) // samples)][:samples]:
        bonus, rate, weight = key
        reference = curve_of(bonus, rate, weight, run_dir)
        mine = everything[key]
        theirs = dict(zip(reference["env_steps_per_copy"],
                          zip(reference["mean_episode_return"], reference["standard_error"])))
        ours = dict(zip(mine["env_steps_per_copy"],
                        zip(mine["mean_episode_return"], mine["standard_error"])))
        shared = sorted(set(theirs) & set(ours))
        if not shared:
            raise AssertionError(f"{key}: the two readers share no window")
        for step in shared:
            for a, b in zip(theirs[step], ours[step]):
                worst = max(worst, abs(a - b))
        compared += len(shared)
        checked += 1
    return {"configurations_checked": checked, "windows_compared": compared,
            "largest_absolute_difference": worst, "identical": worst == 0.0}


if __name__ == "__main__":
    print(verify_against_curve_of())
