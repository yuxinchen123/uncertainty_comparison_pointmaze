"""Read diagnosis runs: quantities per logging window, counted from each policy's own first arrival.

Usage: read_diagnostics.py <runs/data folder> <run name> [<run name> ...] [--absolute] [--fields a,b,c]
Prints, per run, the mean over policies of each quantity at window offsets from the policy's first arrival
(offset 0 is the window that contains it), or at absolute windows with --absolute. Standard library only.
"""
import json
import statistics
import sys

OFFSETS = (-4, -2, -1, 0, 1, 2, 4, 8, 16, 32, 64)
ABSOLUTE = (0, 1, 2, 4, 8, 16, 32, 64, 127, 255)
FIELDS = ("arrival_share", "coverage", "rint_raw", "int_std", "rint_hat_mean", "adv_int_rms", "adv_ext_rms",
          "logstd_mean", "speed_mean", "grad_norm_actor", "grad_norm_critic", "grad_norm_bonus", "clip_scale",
          "approx_kl", "vext_mean")


def read(path):
    """Return the start record and the window records of a run's last start."""
    start, windows = None, []
    for line in open(path):
        if '"unit_start"' in line:
            start, windows = json.loads(line), []
        elif '"return_sum_per_copy"' in line:
            windows.append(json.loads(line))
    return start, windows


def value(window, name, copy):
    """One quantity of one policy in one window; None where it is not defined."""
    if name == "arrival_share":
        episodes = window["episodes_per_copy"][copy]
        return window["arrivals_per_copy"][copy] / episodes if episodes else None
    if name == "coverage":
        return window["coverage_per_copy"][copy]
    if name == "rint_raw":
        return window["rint_mean_per_copy"][copy]
    # a quantity this run did not record (the predictor's own score is recorded only with the second divisor)
    series = window["diagnostics_per_copy"].get(name)
    return None if series is None else series[copy]


def table(start, windows, absolute, fields, copies):
    """Rows of (field, {position: mean over the given policies}) for one configuration of a run."""
    steps = start["rollout_steps"] * start["envs_per_copy"] * start["window_iterations"]
    first = windows[-1]["first_goal_step_per_copy"]
    rows = []
    for name in fields:
        cells = {}
        for copy in copies:
            step = first[copy]
            if absolute:
                positions = [(p, p) for p in ABSOLUTE]
            elif step is None or step < 0:
                continue
            else:
                # before: first arrival at step 5,000 with 2,048 steps per window; after: window 2 is offset 0
                positions = [(o, int(step // steps) + o) for o in OFFSETS]
            for label, index in positions:
                if 0 <= index < len(windows):
                    x = value(windows[index], name, copy)
                    if x is not None:
                        cells.setdefault(label, []).append(x)
        rows.append((name, {label: statistics.fmean(xs) for label, xs in cells.items()}))
    return rows


if __name__ == "__main__":
    arguments = [a for a in sys.argv[1:] if not a.startswith("--")]
    absolute = "--absolute" in sys.argv
    fields = next((a.split("=", 1)[1].split(",") for a in sys.argv if a.startswith("--fields=")), FIELDS)
    folder, names = arguments[0], arguments[1:]
    positions = ABSOLUTE if absolute else OFFSETS
    print(("absolute windows: " if absolute else "windows from the first arrival: ") + "  ".join(f"{p:>8}" for p in positions))
    for name in names:
        start, windows = read(f"{folder}/{name}.jsonl")
        first = windows[-1]["first_goal_step_per_copy"]
        # one block per configuration of the run: the policies of a configuration share copy_cell
        for cell, settings in enumerate(start["cell_settings"]):
            copies = [c for c, owner in enumerate(start["copy_cell"]) if owner == cell]
            arrived = [first[c] for c in copies if first[c] is not None and first[c] >= 0]
            label = ", ".join(f"{key} {value:g}" for key, value in zip(start["cell_setting_names"], settings))
            print(f"== {name} ({label}): {len(windows)} windows, {len(arrived)} of {len(copies)} policies arrived, "
                  f"mean first arrival {statistics.fmean(arrived) / 1000:.1f} thousand steps")
            for field, cells in table(start, windows, absolute, fields, copies):
                print(f"   {field:17s}" + "  ".join(f"{cells[p]:8.3g}" if p in cells else "       -" for p in positions))
