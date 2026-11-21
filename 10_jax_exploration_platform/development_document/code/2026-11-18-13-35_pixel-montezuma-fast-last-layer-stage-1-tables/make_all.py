"""Rebuild the generated blocks of the pixel Montezuma fast-last-layer section from the stage-1 run records.

    python3 make_all.py [--run-dir RUN] [--tex TEX]

It reads the run folder with the live report's own reading functions (scripts/pixel_sweep_campaign/report.py:
collect, configurations, first_step, non_finite): every unit's queue record, progress.json and its two record files
(game_first_times.json, game_windows.jsonl), and the archived baseline's record files. It writes nothing there; the
only side effect is the moment it holds the queue's lock while it reads the queue records, as the live report does
at every monitoring pass. It writes only the .tex, between the
    % >>> AUTO-GENERATED TABLE START: <name>
    % <<< AUTO-GENERATED TABLE END: <name>
markers; prose, captions and labels stay outside them. Standard library only, so it may run on a login node.
"""
import argparse
import importlib.util
import math
from pathlib import Path
import statistics
import sys

HERE = Path(__file__).resolve().parent
DOCUMENT = HERE.parents[1]
PLATFORM = DOCUMENT.parent
SWEEP_SCRIPTS = PLATFORM / "scripts/pixel_sweep_campaign"
RUN_DIR = PLATFORM / ("runs/2026-11-18-12-00_pixel-montezuma_fast-last-layer-against-rnd_stage-1-grid_step-sizes-"
                      "0.01-0.03-0.1-0.3_intrinsic-weights-0.25-0.5-1-2-4-8_sharing-groups-128-16-4_rnd-same-weights_"
                      "seeds-13-to-20_steps-per-policy-200000000_schedule-horizon-2000000000")
TEX = DOCUMENT / "pixel_montezuma_fast_last_layer.tex"
SHARING_GROUPS = (128, 16, 4)
EVENTS = ("key_pickup", "door_opened", "left_start_room")
EXIT = "left_start_room"


def load_report():
    """Load the live report's module (report.py of the pixel sweep) by its path, with its folder importable."""
    # report.py imports campaign_imports and pixel_queue from its own folder; campaign_imports then moves that
    # folder to the end of sys.path itself
    sys.path.insert(0, str(SWEEP_SCRIPTS))
    spec = importlib.util.spec_from_file_location("pixel_sweep_report", SWEEP_SCRIPTS / "report.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


report = load_report()


def steps_trained(run):
    """The agent steps a run has trained so far."""
    return run["update"] * report.pixel_queue.STEPS_PER_UPDATE


def trained(run):
    """Whether a run has trained at least one update (the live report also counts a claimed unit still compiling)."""
    return run["started"] and run["update"] > 0


def median_first(runs, event):
    """Median first-time step of an event over runs (a run without it counts as later than all), and how many had it.

    before: first steps [162176, None, 2400512] ; after: (2400512, 2)
    before: first steps [None] ; after: (inf, 0)
    """
    steps = [report.first_step(run, event) for run in runs]
    return statistics.median([math.inf if step is None else step for step in steps]), sum(s is not None for s in steps)


def recorded(runs, event):
    """Whether the runs' records hold an event: the archived baseline's records hold only the key and the exit."""
    return any(run["has_event_fields"] or event in ("key_pickup", EXIT) for run in runs)


def marks(values):
    """LaTeX formats marking the smallest value of a column bold and the second smallest underlined; ties share.

    before: [160896, inf, 2400512, 160896, None] ; after: ["\\textbf{%s}", "%s", "\\underline{%s}", "\\textbf{%s}", "%s"]
    """
    ranked = sorted({value for value in values if value is not None and math.isfinite(value)})
    style = {ranked[0]: "\\textbf{%s}"} if ranked else {}
    if len(ranked) > 1:
        style[ranked[1]] = "\\underline{%s}"
    return [style.get(value, "%s") for value in values]


def million(steps):
    """Write a number of steps in millions: two decimals below 1,000 million, whole millions above."""
    # before: 162176 ; after: "0.16"      before: 2000015360 ; after: "2,000"
    value = steps / 1e6
    return f"{value:,.2f}" if value < 1000 else f"{value:,.0f}"


def configuration_label(configuration):
    """A configuration's row label inside its own table (the method and the sharing group are in the caption)."""
    # after: "weight 0.25"; "step size 0.03, weight 1"; "RND, archived baseline:\\newline weight 1, seeds 1 to 12"
    settings, weight = configuration["settings"], report.number(configuration["settings"]["intrinsic_weight"])
    if configuration["archived"]:
        seeds = sorted(run["seed"] for run in configuration["runs"])
        return f"RND, archived baseline:\\newline weight {weight}, seeds {seeds[0]} to {seeds[-1]}"
    if "step_size" in settings:
        return f"step size {report.number(settings['step_size'])}, weight {weight}"
    return f"weight {weight}"


def early_row(configuration):
    """One early-table row's values: label, seeds, steps trained, and (median, count) of each event, with sort key."""
    # after: {"label": "weight 1", "seeds": "1 of 8", "steps": "33.69", "events": {"key_pickup": (136704, 1, True),
    #         ...}, "started": 1, "sort": (162176, 136704, <the configuration's key>)}
    runs = [run for run in configuration["runs"] if trained(run)]
    events = {event: (*median_first(runs, event), recorded(runs, event)) for event in EVENTS}
    return {"label": configuration_label(configuration), "started": len(runs),
            "seeds": f"{len(runs)} of {len(configuration['runs'])}",
            "steps": million(statistics.median(steps_trained(run) for run in runs)), "events": events,
            "sort": (events[EXIT][0], events["key_pickup"][0], configuration["key"])}


def early_tabular(trained, baseline):
    """The tabular of one early table: the stage-1 configurations sorted by first exit, a rule, the archived baseline.

    before: trained = [early_row(...) for each started configuration of the block], baseline = early_row(archived)
    after:  "\\begin{tabular}...\\end{tabular}" with one line per row; marks per event column over all rows
    """
    trained = sorted(trained, key=lambda row: row["sort"])
    rows = trained + [baseline]
    # each event column: its marks over every row with the event recorded, then the cell text
    cells = {}
    for event in EVENTS:
        values = [row["events"][event][0] if row["events"][event][2] else None for row in rows]
        cells[event] = []
        for row, style in zip(rows, marks(values)):
            median, count, known = row["events"][event]
            if not known:
                cells[event].append("not recorded")
            else:
                text = million(median) if math.isfinite(median) else "not yet"
                cells[event].append((style % text) + f" ({count}/{row['started']})")
    lines = ["\\begin{tabular}{@{}>{\\raggedright\\arraybackslash}p{3.7cm}rrrrr@{}}", "\\toprule",
             "\\textbf{configuration} & \\textbf{\\shortstack[r]{seeds\\\\started}} & "
             "\\textbf{\\shortstack[r]{steps trained\\\\(million)}} & \\textbf{\\shortstack[r]{first\\\\key}} & "
             "\\textbf{\\shortstack[r]{first\\\\door}} & "
             "\\textbf{\\shortstack[r]{first exit from the\\\\start room $\\downarrow$}} \\\\", "\\midrule"]
    if not trained:
        lines.append("\\multicolumn{6}{@{}l}{no run of this block had started} \\\\")
    for index, row in enumerate(rows):
        if index == len(trained):
            lines.append("\\midrule")
        lines.append(f"{row['label']} & {row['seeds']} & {row['steps']} & "
                     + " & ".join(cells[event][index] for event in EVENTS) + " \\\\")
    return "\n".join(lines + ["\\bottomrule", "\\end{tabular}"])


BUDGET_WINDOWS = 200     # whole windows of 61 updates in 12,207 updates: the budget B of 200 million steps
FINAL_WINDOWS = 20       # the last 10% of the budget's windows, for the final level


def score_row(configuration, budget=BUDGET_WINDOWS, final=FINAL_WINDOWS):
    """One score-table row: label, seeds at the budget, mean score over the run S(B) and final level, with sort key.

    S(B) is report.score_up_to at the budget, a mean over the seeds that have reached it; the final level is the
    mean window value of the last `final` windows before the budget. The archived baseline gives its first
    `budget` windows (its runs share the stage's 2-billion-step learning-rate horizon, so they are a prefix).
    The standard error of S(B) over those seeds (sample standard deviation over the square root of their number)
    stands beside it; with one seed there is none.
    before: a configuration with 2 of 8 seeds at 200 windows, S 1,800 and 2,200 ; after: {"seeds": "2 of 8",
            "score": 2000.0, "standard_error": 200.0, ...}
    """
    runs = [run for run in configuration["runs"] if len(run["windows"]) >= budget]
    scores = [report.score_up_to(run, budget) for run in runs]
    finals = [statistics.fmean(window["mean_score"] for window in run["windows"][budget - final:budget]) for run in runs]
    score = statistics.fmean(scores) if scores else None
    standard_error = statistics.stdev(scores) / math.sqrt(len(scores)) if len(scores) > 1 else None
    return {"label": configuration_label(configuration), "seeds": f"{len(runs)} of {len(configuration['runs'])}",
            "at_budget": len(runs), "score": score, "standard_error": standard_error,
            "final": statistics.fmean(finals) if finals else None,
            "sort": (score is None, -(score or 0.0), configuration["key"])}


def score_tabular(rows, baseline):
    """The tabular of one score table: configurations by S(B), largest first, a rule, the archived baseline.

    before: rows = [score_row(...) for each configuration of the block] ; after: "\\begin{tabular}...\\end{tabular}"
    """
    rows = sorted(rows, key=lambda row: row["sort"]) + [baseline]
    # larger is better in both columns: the marks of the negated values
    cells = {}
    for column in ("score", "final"):
        styles = marks([-row[column] if row[column] is not None else None for row in rows])
        cells[column] = [(style % f"{row[column]:,.0f}") if row[column] is not None else "not yet"
                         for row, style in zip(rows, styles)]
    # the standard error follows the marked mean score, outside the mark: "\\textbf{1,889} $\\pm$ 278"
    cells["score"] = [cell + (f" $\\pm$ {row['standard_error']:,.0f}" if row["standard_error"] is not None else "")
                      for cell, row in zip(cells["score"], rows)]
    lines = ["\\begin{tabular}{@{}>{\\raggedright\\arraybackslash}p{3.7cm}rrr@{}}", "\\toprule",
             "\\textbf{configuration} & \\textbf{\\shortstack[r]{seeds at the\\\\budget}} & "
             "\\textbf{\\shortstack[r]{mean score over the run,\\\\$S(B)$ $\\uparrow$ $\\pm$ standard error}} & "
             "\\textbf{\\shortstack[r]{final level,\\\\$S_{\\text{last}}(B)$}} \\\\", "\\midrule"]
    for index, row in enumerate(rows):
        if index == len(rows) - 1:
            lines.append("\\midrule")
        lines.append(f"{row['label']} & {row['seeds']} & {cells['score'][index]} & {cells['final'][index]} \\\\")
    return "\n".join(lines + ["\\bottomrule", "\\end{tabular}"])


def blocks(groups):
    """Split the configurations into the early tables' blocks: RND, then one per sharing group, and the baseline.

    after: ({"rnd": [stage-1 RND configurations], 128: [...], 16: [...], 4: [...]}, the archived configuration)
    """
    archived = [configuration for configuration in groups if configuration["archived"]]
    if len(archived) != 1:
        raise SystemExit(f"expected one archived configuration (RND at weight 1), found {len(archived)}")
    split = {"rnd": [], **{group: [] for group in SHARING_GROUPS}}
    for configuration in groups:
        if not configuration["archived"]:
            split[configuration["settings"].get("sharing_group", "rnd")].append(configuration)
    return split, archived[0]


def started_rows(configurations):
    """The early-table rows of the configurations that have a started seed."""
    return [early_row(configuration) for configuration in configurations
            if any(trained(run) for run in configuration["runs"])]


def status_line(name, configurations):
    """One status-table row: configurations started, runs planned, started, finished, out of the start room, steps."""
    # after: "fast last layer,\\newline sharing group 128 & 24 of 24 & 192 & 24 & 0 & 17 of 24 & 11.16 to 41.14 \\\\"
    runs = [run for configuration in configurations for run in configuration["runs"]]
    started = [run for run in runs if trained(run)]
    steps = sorted(steps_trained(run) for run in started)
    configurations_started = sum(any(trained(run) for run in configuration["runs"]) for configuration in configurations)
    left = sum(report.first_step(run, EXIT) is not None for run in started)
    return (f"{name} & {configurations_started} of {len(configurations)} & {len(runs)} & {len(started)} & "
            f"{sum(run['state'] == 'done' for run in runs)} & {left} of {len(started)} & "
            + (f"{million(steps[0])} to {million(steps[-1])}" if steps else "N/A") + " \\\\")


def status_tabular(split):
    """The status table: one row per block (RND, each sharing group), then all stage-1 runs together."""
    lines = ["\\begin{tabular}{@{}>{\\raggedright\\arraybackslash}p{3.4cm}rrrrrr@{}}", "\\toprule",
             "\\textbf{arm} & \\textbf{\\shortstack[r]{configurations\\\\started}} & "
             "\\textbf{\\shortstack[r]{runs\\\\planned}} & \\textbf{\\shortstack[r]{runs\\\\started}} & "
             "\\textbf{\\shortstack[r]{runs\\\\finished}} & \\textbf{\\shortstack[r]{started runs that\\\\left the start room}} & "
             "\\textbf{\\shortstack[r]{steps trained\\\\(million)}} \\\\", "\\midrule"]
    names = {"rnd": "RND", **{group: f"fast last layer,\\newline sharing group {group}" for group in SHARING_GROUPS}}
    lines += [status_line(names[block], configurations) for block, configurations in split.items()]
    every = [configuration for configurations in split.values() for configuration in configurations]
    lines += ["\\midrule", status_line("all stage-1 runs", every)]
    return "\n".join(lines + ["\\bottomrule", "\\end{tabular}"])


def check_counts(data):
    """The standing checks of the live report, as counts: (found, checked, what was checked, expected) per check."""
    # the same filters as report.checks_section, which writes them as Markdown lines
    runs = [run for run in data["runs"] if run["first"] is not None and not run["archived"]]
    events = [run for run in runs if run["has_event_fields"]]
    locked = sum(run["first"]["event_totals"][report.LOCKED_EXIT] for run in events)
    # an exit before the first counted key is expected only from a key carried out of the random warm-up
    _, _, unexplained = report.exits_before_counted_key(events)
    broken = sum(report.non_finite(run["first"]) + report.non_finite(run["windows"]) for run in runs)
    silent = sum(not unit["has_records"] and unit["update"] > 0 for unit in data["units"])
    failed = sum(unit["state"] == "failed" for unit in data["units"])
    return [("exits from the start room with both doors locked", locked, len(events), "runs with game events"),
            ("runs whose first exit came before their first counted key with no door opened before that key",
             len(unexplained), len(events), "runs with game events"),
            ("values that are not finite in the record files", broken, len(runs), "runs with records"),
            ("units past their first update without record files", silent, len(data["units"]), "units"),
            ("failed units", failed, len(data["units"]), "units")]


def checks_tabular(checks):
    """The standing-checks table: one row per check, with its count, what it was checked on and the expected count."""
    lines = ["\\begin{tabular}{@{}>{\\raggedright\\arraybackslash}p{6.6cm}rrr@{}}", "\\toprule",
             "\\textbf{check} & \\textbf{found} & \\textbf{checked} & \\textbf{expected} \\\\", "\\midrule"]
    lines += [f"{name} & {found} & {checked} {what} & 0 \\\\" for name, found, checked, what in checks]
    return "\n".join(lines + ["\\bottomrule", "\\end{tabular}"])


def seeds_in_words(runs):
    """The distinct seeds of some runs, written for the prose.

    before: runs of seeds [14, 14, 14] ; after: "seed $14$"      before: seeds [15, 14] ; after: "seeds $14$ and $15$"
    before: no runs ; after: "no seed"
    """
    seeds = sorted({run["seed"] for run in runs})
    if not seeds:
        return "no seed"
    names = [f"${seed}$" for seed in seeds]
    return ("seed " if len(seeds) == 1 else "seeds ") + (names[0] if len(names) == 1 else
                                                         ", ".join(names[:-1]) + " and " + names[-1])


def numbers_block(data, split, archived, checks, read_at):
    """The \\PixelFll... macros the hand-written prose uses, so its counts follow every refresh."""
    # before: the collected runs ; after: "\\newcommand{\\PixelFllRunsStarted}{34}" and the like, one per line
    stage = [run for configurations in split.values() for configuration in configurations for run in configuration["runs"]]
    started = [run for run in stage if trained(run)]
    steps = sorted(steps_trained(run) for run in started)

    def door_no_exit(runs):
        """How many runs opened a door and had not left the start room."""
        return sum(report.first_step(run, "door_opened") is not None and report.first_step(run, EXIT) is None
                   for run in runs)

    # runs that left the start room with a key carried out of the random warm-up, and their seeds
    _, carried, _ = report.exits_before_counted_key([run for run in started if run["has_event_fields"]])
    # seeds that have reached the budget, per stage-1 configuration (for the score tables' text)
    at_budget = [sum(len(run["windows"]) >= BUDGET_WINDOWS for run in configuration["runs"])
                 for configurations in split.values() for configuration in configurations]
    rnd = [run for run in started if run["method"] == "rnd"]
    fast = [run for run in started if run["method"] != "rnd"]
    exits = sorted(report.first_step(run, EXIT) for run in archived["runs"] if report.first_step(run, EXIT) is not None)
    values = {
        "ReadAt": read_at, "RunsPlanned": f"{len(stage):,}", "RunsStarted": len(started),
        "RunsFinished": sum(run["state"] == "done" for run in stage),
        "MostSeedsStarted": max(sum(trained(run) for run in configuration["runs"])
                                for configurations in split.values() for configuration in configurations),
        "FewestSteps": million(steps[0]) if steps else "N/A", "MostSteps": million(steps[-1]) if steps else "N/A",
        "RndStarted": len(rnd), "RndLeft": sum(report.first_step(run, EXIT) is not None for run in rnd),
        "FastStarted": len(fast), "FastLeft": sum(report.first_step(run, EXIT) is not None for run in fast),
        "RndDoorNoExit": door_no_exit(rnd), "FastDoorNoExit": door_no_exit(fast),
        "ArchivedRuns": len(archived["runs"]), "ArchivedLeft": len(exits),
        "ArchivedExit": million(statistics.median(exits)), "ArchivedExitFirst": million(exits[0]),
        "ArchivedExitLast": million(exits[-1]),
        "LockedExits": checks[0][1], "ExitBeforeKey": checks[1][1], "NonFinite": checks[2][1],
        "WarmUpKeyRuns": len(carried), "WarmUpKeySeeds": seeds_in_words(carried),
        "FewestSeedsAtBudget": min(at_budget), "MostSeedsAtBudget": max(at_budget),
    }
    return "\n".join(f"\\newcommand{{\\PixelFll{name}}}{{{value}}}" for name, value in values.items())


def splice(text, name, block):
    """Replace the text between one table's markers; fail when a marker is missing or appears twice.

    before: "...% >>> AUTO-GENERATED TABLE START: x\\n<old>\\n% <<< AUTO-GENERATED TABLE END: x..." ; after: the same
    with <old> replaced by block
    """
    begin, end = f"% >>> AUTO-GENERATED TABLE START: {name}", f"% <<< AUTO-GENERATED TABLE END: {name}"
    if text.count(begin) != 1 or text.count(end) != 1:
        raise SystemExit(f"marker mismatch for {name!r}: START {text.count(begin)}, END {text.count(end)} (expected 1 each)")
    before, _, rest = text.partition(begin)
    _, _, after = rest.partition(end)
    return f"{before}{begin}\n{block}\n{end}{after}"


def build(run_dir, text):
    """Every generated block from the run's records, spliced into the .tex text; returns the new text."""
    data = report.collect(run_dir)
    read_at = report.common.pacific(report.common.now())
    split, archived = blocks(report.configurations(data["runs"]))
    baseline = early_row(archived)
    checks = check_counts(data)
    generated = {"pixel-montezuma-fll-numbers": numbers_block(data, split, archived, checks, read_at),
                 "pixel-montezuma-fll-status": status_tabular(split),
                 "pixel-montezuma-fll-early-rnd": early_tabular(started_rows(split["rnd"]), baseline),
                 **{f"pixel-montezuma-fll-early-group-{group}": early_tabular(started_rows(split[group]), baseline)
                    for group in SHARING_GROUPS},
                 "pixel-montezuma-fll-score-rnd": score_tabular([score_row(c) for c in split["rnd"]], score_row(archived)),
                 **{f"pixel-montezuma-fll-score-group-{group}": score_tabular([score_row(c) for c in split[group]],
                                                                             score_row(archived))
                    for group in SHARING_GROUPS},
                 "pixel-montezuma-fll-checks": checks_tabular(checks)}
    for name, block in generated.items():
        text = splice(text, name, block)
    return text, read_at


def main(argv=None):
    """Rebuild every generated block of the section's .tex from the run records."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", type=Path, default=RUN_DIR)
    parser.add_argument("--tex", type=Path, default=TEX)
    args = parser.parse_args(argv)
    text, read_at = build(args.run_dir.resolve(), args.tex.read_text())
    # written under another name first, so a build that starts meanwhile never reads half a file
    partial = args.tex.with_name(args.tex.name + ".partial")
    partial.write_text(text)
    partial.replace(args.tex)
    print(f"updated {args.tex} from the records read at {read_at}")


if __name__ == "__main__":
    main()
