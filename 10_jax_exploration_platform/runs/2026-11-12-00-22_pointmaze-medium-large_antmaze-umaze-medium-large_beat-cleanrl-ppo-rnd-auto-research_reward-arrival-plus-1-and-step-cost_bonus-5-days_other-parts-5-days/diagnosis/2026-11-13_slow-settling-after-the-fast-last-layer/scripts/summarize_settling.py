"""One line per run and configuration: how soon the goal is found and how fast it is kept (standard library).

Usage: summarize_settling.py <runs/data folder> <run name> [<run name> ...]
Columns: mean first arrival (thousand steps); share of episodes that reach the goal 8, 16, 32 and 64 logging
windows after the policy's own first arrival; mean training return over the whole short run; the divisor and
the weighted intrinsic and the extrinsic advantage (root mean square) one window after the first arrival.
"""
import statistics
import sys

from read_diagnostics import read, value


def at_offsets(start, windows, copies, name, offsets):
    """Mean over the given policies of one quantity at window offsets from each policy's first arrival."""
    steps = start["rollout_steps"] * start["envs_per_copy"] * start["window_iterations"]
    first = windows[-1]["first_goal_step_per_copy"]
    out = []
    for offset in offsets:
        cells = []
        for copy in copies:
            if first[copy] is None or first[copy] < 0:
                continue
            index = int(first[copy] // steps) + offset
            if 0 <= index < len(windows):
                x = value(windows[index], name, copy)
                if x is not None:
                    cells.append(x)
        out.append(statistics.fmean(cells) if cells else float("nan"))
    return out


def mean_return(windows, copies):
    """Mean over policies and windows of the training return (0 before a policy's first ended episode)."""
    total = []
    for copy in copies:
        last, series = 0.0, []
        for window in windows:
            if window["episodes_per_copy"][copy] > 0:
                last = window["return_sum_per_copy"][copy] / window["episodes_per_copy"][copy]
            series.append(last)
        total.append(statistics.fmean(series))
    return statistics.fmean(total)


if __name__ == "__main__":
    folder, names = sys.argv[1], sys.argv[2:]
    print(f"{'run':62s} {'weight':>6s} {'first':>6s} | arrival share +8   +16   +32   +64 | {'return':>6s} | "
          f"{'divisor':>7s} {'adv int':>7s} {'adv ext':>7s}")
    for name in names:
        start, windows = read(f"{folder}/{name}.jsonl")
        first = windows[-1]["first_goal_step_per_copy"]
        for cell, settings in enumerate(start["cell_settings"]):
            copies = [c for c, owner in enumerate(start["copy_cell"]) if owner == cell]
            arrived = [first[c] for c in copies if first[c] is not None and first[c] >= 0]
            share = at_offsets(start, windows, copies, "arrival_share", (8, 16, 32, 64))
            one = [at_offsets(start, windows, copies, field, (1,))[0] for field in ("int_std", "adv_int_rms", "adv_ext_rms")]
            weight = dict(zip(start["cell_setting_names"], settings))["int_coef"]
            print(f"{name[:62]:62s} {weight:6g} {statistics.fmean(arrived) / 1000:6.1f} |              "
                  + " ".join(f"{x:5.2f}" for x in share) + f" | {mean_return(windows, copies):6.3f} | "
                  + " ".join(f"{x:7.3g}" for x in one))
