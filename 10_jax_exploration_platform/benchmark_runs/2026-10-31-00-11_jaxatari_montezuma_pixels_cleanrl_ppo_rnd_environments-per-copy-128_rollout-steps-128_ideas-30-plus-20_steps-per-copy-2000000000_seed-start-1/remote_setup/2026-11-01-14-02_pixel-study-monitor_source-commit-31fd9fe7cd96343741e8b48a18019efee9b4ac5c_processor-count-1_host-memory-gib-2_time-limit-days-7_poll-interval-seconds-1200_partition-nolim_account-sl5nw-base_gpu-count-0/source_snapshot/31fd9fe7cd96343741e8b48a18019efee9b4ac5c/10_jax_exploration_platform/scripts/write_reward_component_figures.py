"""Plot recorded extrinsic and raw combined rewards, with explicit labels and paired seeds."""
import argparse
import importlib.util
import json
import math
from pathlib import Path

from write_corrected_report import NAMES, SETTINGS, power

FIRST = [("rnd_next_state", 1e-3, 10), ("gt_position_velocity_sqrt", 1e-3, 1),
         ("gt_position_velocity_linear", 1e-3, 1), ("none", 1e-3, 1),
         ("rnd_next_state", 1e-4, 100), ("gt_position_velocity_sqrt", 1e-4, 10)]


def episode_components(record, position, episode_steps):
    """Convert one copy's window sums and raw per-step mean to comparable episode totals."""
    episodes = record["episodes_per_copy_in_window"]
    if episodes <= 0 or episode_steps <= 0:
        raise ValueError("episode count and episode length must be positive")
    # Before: 64 task rewards / 32 episodes and raw bonus mean .1; after: 2 + 400*.1 = 42.
    extrinsic = record["reward_ext_sum_per_copy"][position] / episodes
    intrinsic = record["rint_mean_per_copy"][position] * episode_steps
    if not all(math.isfinite(value) for value in (extrinsic, intrinsic)):
        raise ValueError("non-finite recorded reward")
    return extrinsic, intrinsic, extrinsic + intrinsic


def pooled_summary(entry, expected_copies):
    """Finish per-copy running sums, requiring the complete expected seed count."""
    n = entry["n"]
    if n != expected_copies or n < 2:
        raise ValueError(f"expected {expected_copies} copies, found {n}")
    result = {}
    for column, name in enumerate(("extrinsic", "raw_intrinsic", "raw_total")):
        mean = entry["sum"][column] / n
        variance = max(0., (entry["square_sum"][column] - n*mean*mean) / (n-1))
        result[name], result[name+"_standard_error"] = mean, math.sqrt(variance/n)
    return result


def extract(run, settings, reader, expected_copies):
    """Pool selected settings and matching windows, calculating combined errors per copy."""
    pools, expected_seeds = {}, {}
    for shard in sorted((run / "data").glob("*.jsonl")):
        live = reader.live_lines(shard)
        if live["start"] is None or live["complete"] is None:
            raise ValueError(f"incomplete historical input: {shard}")
        _, start = next(reader.read_selected(shard, {live["start"]}))
        available = reader.cell_copies(start)
        wanted = {(b, r, w): available[(r, w)] for b, r, w in settings
                  if b == start["bonus"] and (r, w) in available}
        if not wanted:
            continue
        # Each seed must occur once per setting; average across copies, not across chunks.
        for key, positions in wanted.items():
            seeds = {start["copy_seed_index"][position] for position in positions}
            previous = expected_seeds.setdefault(key, set())
            if previous & seeds:
                raise ValueError(f"duplicate seed for {key} in {shard}")
            previous.update(seeds)
        seen_steps = set()
        for _, record in reader.read_selected(shard, set(live["windows"].values())):
            if not record["phase_blocked"]:
                continue
            steps = record["env_steps_per_copy"]
            if steps in seen_steps:
                raise ValueError(f"duplicate step count {steps} in {shard}")
            seen_steps.add(steps)
            for key, positions in wanted.items():
                window = pools.setdefault(key, {}).setdefault(
                    steps, {"n": 0, "sum": [0.]*3, "square_sum": [0.]*3})
                for position in positions:
                    values = episode_components(record, position, start["episode_steps"])
                    for column, value in enumerate(values):
                        window["sum"][column] += value
                        window["square_sum"][column] += value*value
                    window["n"] += 1
        print(f"read {shard.name}", flush=True)

    # Before: each seed's (extrinsic, raw intrinsic, sum); after: three curves with seed errors.
    output = {}
    for key in settings:
        if expected_seeds.get(key) != set(range(expected_copies)):
            raise ValueError(f"{key} does not contain exactly {expected_copies} seed indices")
        points = []
        for steps, entry in sorted(pools[key].items()):
            point = {"steps": steps, "copies": expected_copies}
            point.update(pooled_summary(entry, expected_copies))
            points.append(point)
        output[f"{key[0]}:{key[1]:g}:{key[2]:g}"] = points
    return output


def draw(curves, settings, output, total=False, first_figure=False, historical=False):
    """Draw one clearly named reward quantity with the established algorithm colors."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    colors = {"rnd_next_state": "black", "gt_position_velocity_sqrt": "#0072B2",
              "gt_position_velocity_linear": "#D55E00", "none": "#009E73"}
    field = "raw_total" if total else "extrinsic"
    figure, axes = plt.subplots(figsize=(8, 5))
    for index, (bonus, rate, weight) in enumerate(settings):
        points = curves[f"{bonus}:{rate:g}:{weight:g}"]
        x = np.asarray([point["steps"] for point in points]) / 1e6
        mean = np.asarray([point[field] for point in points])
        error = np.asarray([point[field + "_standard_error"] for point in points])
        color = ("#D55E00" if not first_figure and bonus == "rnd_next_state" and weight == 1
                 else colors[bonus])
        style = ("--" if index < 4 else "-") if first_figure else (":" if rate == 1e-3 else "-")
        opacity = .55 if historical and (index < 4 if first_figure else rate == 1e-3) else 1.
        label = NAMES[bonus] + rf", $\alpha={power(rate)}$"
        if bonus != "none":
            label += rf", $\beta={power(weight)}$"
        if first_figure:
            label += f", {points[0]['copies']} copies"
        axes.plot(x, mean, color=color, linestyle=style, alpha=opacity, linewidth=1.5, label=label)
        axes.fill_between(x, mean-error, mean+error, color=color, alpha=.10*opacity, linewidth=0)
        # Preserve the four peak markers on the historical extrinsic configuration figure.
        if historical and not first_figure and not total and not (bonus == "rnd_next_state" and weight == 1):
            peak = int(np.argmax(mean))
            axes.plot(x[peak], mean[peak], "o", color=color, markersize=4)
            offsets = {("gt_position_velocity_sqrt", 1e-3): (-58, -6),
                       ("rnd_next_state", 1e-3): (-12, 16),
                       ("gt_position_velocity_sqrt", 1e-4): (-95, 10),
                       ("rnd_next_state", 1e-4): (10, 6)}
            axes.annotate(f"{mean[peak]:.1f} at {x[peak]:.0f}M", (x[peak], mean[peak]),
                          xytext=offsets[(bonus, rate)], textcoords="offset points", fontsize=8,
                          color=color, bbox=dict(facecolor="white", edgecolor="none", alpha=.8))
    ylabel = ("mean total reward per episode\n(extrinsic + raw intrinsic; unweighted)" if total
              else "mean extrinsic reward per episode")
    axes.set(xscale="log", xlabel="environment steps per copy (millions, logarithmic)", ylabel=ylabel)
    if total:
        # Keep both the large initial raw bonuses and the smaller later task rewards readable.
        axes.set_yscale("symlog", linthresh=1)
        axes.set_ylim(bottom=0)
    axes.set_title("Historical training" if historical else "Training with continuing intrinsic returns")
    if historical and not first_figure and not total:
        axes.set_xlim(right=5.5e3)
        axes.set_ylim(top=95)
    axes.grid(alpha=.2, which="both")
    axes.legend(loc="upper left", fontsize=7.5, framealpha=.93)
    figure.tight_layout()
    for suffix in ("pdf", "png"):
        figure.savefig(output.with_suffix("." + suffix), dpi=190)
    plt.close(figure)


def verify_extrinsic(cached, historical, corrected):
    """Check every extrinsic point against the previously published curve aggregates."""
    worst, points = 0., 0
    merged = {**cached["historical_second"], **cached["historical_first"]}
    for name in ("first", "second"):
        actual = merged if name == "first" else cached["historical_second"]
        for key, reference in historical["curves"][name].items():
            key = "none:0.001:1" if key.startswith("none:") else key
            curve = actual[key]
            if [point["steps"] for point in curve] != reference["env_steps_per_copy"]:
                raise ValueError(f"historical window steps changed for {key}")
            for point, mean, error in zip(curve, reference["mean_episode_return"],
                                          reference["standard_error"]):
                worst = max(worst, abs(point["extrinsic"]-mean),
                            abs(point["extrinsic_standard_error"]-error))
                points += 1
    for bonus, rate, weight in SETTINGS:
        key = f"{bonus}:{rate:g}:{weight:g}"
        actual, reference = cached["corrected"][key], corrected["curves"][key]
        if [p["steps"] for p in actual] != [p["steps"] for p in reference]:
            raise ValueError(f"corrected window steps changed for {key}")
        for point, old in zip(actual, reference):
            worst = max(worst, abs(point["extrinsic"]-old["mean"]),
                        abs(point["extrinsic_standard_error"]-old["standard_error"]))
            points += 1
    if worst > 1e-7:
        raise ValueError(f"extrinsic curves changed: largest difference {worst}")
    return {"points_checked": points, "largest_absolute_difference": worst}


def main():
    """Extract completed records once, publish five plots and describe the recorded quantities."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis-dir", type=Path, required=True)
    parser.add_argument("--refresh", action="store_true", help="re-read the original completed shards")
    parser.add_argument("--reference-historical-cache", type=Path,
                        help="published historical curves for an independent preservation check")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    runs = root / "runs"
    first = next(runs.glob("2026-08-16-22-47_*extension-of-first-batch"))
    second = next(runs.glob("2026-08-17-10-45_*extension-2-of-first-batch"))
    corrected = next(runs.glob("2026-10-13-22-13_pointmaze_full-batch*"))
    spec = importlib.util.spec_from_file_location("historical_reward_reader", second / "code/aggregate.py")
    reader = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reader)
    cache = args.analysis_dir / "data/curves.json"
    if args.refresh or not cache.exists():
        old_first = extract(first, FIRST[:4], reader, 1024)
        old_second = extract(second, list(dict.fromkeys(SETTINGS + FIRST[4:])), reader, 128)
        new = extract(corrected, SETTINGS, reader, 128)
        cached = {"historical_first": old_first, "historical_second": old_second, "corrected": new,
                  "sources": [str(first), str(second), str(corrected)],
                  "total_definition": "extrinsic episode sum + 400 * raw intrinsic per-step mean"}
        cache.write_text(json.dumps(cached, allow_nan=False))
    else:
        cached = json.loads(cache.read_text())
    if args.reference_historical_cache:
        result = verify_extrinsic(cached, json.loads(args.reference_historical_cache.read_text()),
                                  json.loads((corrected / "summary.json").read_text()))
        (args.analysis_dir / "data/extrinsic_validation.json").write_text(json.dumps(result))
        print(f"extrinsic preservation check: {result}", flush=True)
    plots = args.analysis_dir / "plots"
    draw({**cached["historical_second"], **cached["historical_first"]}, FIRST,
         plots / "historical_extrinsic_selected", first_figure=True, historical=True)
    for name, historical in (("historical_second", True), ("corrected", False)):
        prefix = "historical" if historical else "corrected"
        for total in (False, True):
            draw(cached[name], SETTINGS, plots / (prefix + ("_total" if total else "_extrinsic")),
                 total=total, historical=historical)
    (args.analysis_dir / "analysis.md").write_text(
        "Generated by scripts/write_reward_component_figures.py.\n\n"
        "Existing reward curves measure extrinsic episode reward only. "
        "Figures 1.3.1 and 1.4.1 show that quantity. Figures 1.3.2 and 1.4.2 add raw intrinsic "
        "reward without weights or normalization. Define $R_{\\mathrm{ext}}$ as extrinsic "
        "reward per episode and $\\bar r_{\\mathrm{int,raw}}$ as mean raw intrinsic reward "
        "per step. Each copy's total is "
        "$R_{\\mathrm{sum}}=R_{\\mathrm{ext}}+400\\bar r_{\\mathrm{int,raw}}$. "
        "Episode length is 400 steps. "
        "Standard errors are calculated from each copy's combined value, retaining covariance. "
        "Combined-reward plots use a vertical axis logarithmic above 1 and linear between -1 and 1. "
        "Historical Figure 1.2 retains its original 1024/128 copy counts; both paired figures "
        "use 128 copies per setting. Settings are held fixed from the original Figure 1.3.\n\n"
        "The logs record raw intrinsic rewards before their per-copy normalization. "
        "PPO combines task and normalized intrinsic advantages with different weights and "
        "discounts. Normalization denominators and normalized intrinsic reward sums were not "
        "saved, so the actual training-weighted normalized total cannot be reconstructed. "
        "The additional curves do not identify the cause of the rise and decline.\n")
    print(f"wrote five plots to {plots}", flush=True)


if __name__ == "__main__":
    main()
