"""Regenerate the reward-decline study's tables inside the development document.

Each table's body sits between a pair of marker comments inside its table float; this script
rewrites only what lies between the markers, so captions and prose are never touched.

Tables:
- pm-jax-speed-jaguar03: the jaguar03 speed ideas, from the benchmark's analysis_rows.json.
- pm-jax-screening-results / -verdicts: the map screening, from its analysis/data/summary.json.

Usage: write_decline_study_tables.py <document.tex> [--only <table name>]
"""
import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SPEED = ROOT / ("benchmark_runs/2026-10-19-15-46_jaguar03_rtx-a4500_random-network-distillation_"
                "copies-128-per-setting_rollout-128_environments-4_full-batch_speed-10-ideas-"
                "round-2/analysis_rows.json")

SPEED_NAMES = {
    "baseline": "baseline: one setting, one program per card",
    "01_four_settings_one_program": "01 four settings in one program",
    "02_eight_settings_one_program": "02 eight settings in one program",
    "03_two_programs_per_card": "03 two programs share a card",
    "04_four_programs_per_card": "04 four programs share a card",
    "05_command_buffers_off": "05 XLA command buffers off",
    "06_command_buffers_with_loops": "06 XLA command buffers also for loops",
    "07_latency_hiding_scheduler": "07 XLA latency-hiding scheduler on",
    "08_compilation_cache_cold": "08 compilation cache, first run",
    "08_compilation_cache_warm": "08 compilation cache, second run",
    "09_two_processors": "09 two processors instead of eight",
    "10_diagnostics_on": "10 the study's diagnostics on",
}


def mark(values, value, text, higher=True):
    """Bold the best and underline the second best of a column (ties share the mark)."""
    ranked = sorted(set(values), reverse=higher)
    if value == ranked[0]:
        return r"\textbf{" + text + "}"
    if len(ranked) > 1 and value == ranked[1]:
        return r"\underline{" + text + "}"
    return text


def speed_table() -> str:
    """The tabular of the jaguar03 speed ideas, sorted by total rate relative to the baseline."""
    rows = json.loads(SPEED.read_text())
    rows.sort(key=lambda r: -r["relative_total"])
    rel = [round(r["relative_total"], 2) for r in rows]
    tot = [round(r["total_millions"], 2) for r in rows]
    per = [round(r["per_copy_thousands"], 1) for r in rows]
    setup = [round(r["setup_seconds"]) for r in rows]
    lines = [r"\begin{tabular}{@{}>{\raggedright\arraybackslash}p{4.3cm} r r r r r r r"
             r" >{\raggedright\arraybackslash}p{2.6cm}@{}}",
             r"\toprule",
             r"\textbf{idea} & \textbf{\shortstack[c]{copies}} & "
             r"\textbf{\shortstack[c]{seconds per\\iteration}} & "
             r"\textbf{\shortstack[c]{total steps\\per second (M)}} & "
             r"\textbf{\shortstack[c]{steps per second\\per copy (k)}} & "
             r"\textbf{\shortstack[c]{hours per M\\steps per copy}} & "
             r"\textbf{\shortstack[c]{total against\\baseline $\uparrow$}} & "
             r"\textbf{\shortstack[c]{setup\\seconds}} & "
             r"\textbf{\shortstack[c]{copies identical\\to baseline}} \\",
             r"\midrule"]
    for r, a, b, c, d in zip(rows, rel, tot, per, setup):
        # before: "3 of 128 identical; largest difference 2.8e-04"; after: "3 of 128"
        ident = r["agreement"].split(";")[0].replace(" identical", "")
        copies = f"{r['copies']:,}".replace(",", "{,}")
        hours = f"{r['hours_per_million_per_copy']:.4f}"
        lines.append(
            f"{SPEED_NAMES[r['idea']]} & {copies} & {r['seconds_per_iteration']:.4f} & "
            f"{mark(tot, b, f'{b:.2f}')} & {mark(per, c, f'{c:.1f}')} & "
            f"{mark(per, c, hours)} & "
            f"{mark(rel, a, f'{a:.2f}')} & {mark([-x for x in setup], -d, str(d))} & "
            f"{ident} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


SCREEN_RUN = ROOT / ("runs/2026-10-19-16-18_map-screening_umaze-cap150_medium-cap250_medium-"
                     "cap400_large-cap400_rnd-weights-1-10-100-1000_visit-count-sqrt-weight-1_"
                     "no-bonus_learning-rate-1e-3_steps-30M-large-50M_seeds-0-127")
SCREEN_MAPS = [("umaze-cap150", "U-maze, cap 150"), ("medium-cap250", "medium maze, cap 250"),
               ("medium-cap400", "medium maze, cap 400"), ("large-cap400", "large maze, cap 400")]
BONUS_NAMES = {"rnd_next_state": "random network distillation",
               "gt_position_velocity_sqrt": "oracle counts $1/\\sqrt{n}$", "none": "no bonus"}


def screening_rows():
    """The screening summary as {map: [(bonus, weight, numbers), ...]} in a fixed row order."""
    summary = json.loads((SCREEN_RUN / "analysis" / "data" / "summary.json").read_text())
    by_map = {m: [] for m, _ in SCREEN_MAPS}
    order = {"rnd_next_state": 0, "gt_position_velocity_sqrt": 1, "none": 2}
    for numbers in summary.values():
        map_key = numbers["unit"].split("_")[0]
        by_map[map_key].append((numbers["bonus"], numbers["intrinsic_weight"], numbers))
    for rows in by_map.values():
        rows.sort(key=lambda r: (order[r[0]], r[1]))
    return by_map


def screening_verdicts(by_map):
    """The three rules of the screening per map: (rule 1, rule 2, rule 3, numbers used)."""
    verdicts = {}
    for map_key, rows in by_map.items():
        none = next(n for b, _, n in rows if b == "none")
        rnd = [n for b, _, n in rows if b == "rnd_next_state"]
        best = max(rnd, key=lambda n: n["whole_run"])
        oracle = next(n for b, _, n in rows if b == "gt_position_velocity_sqrt")
        r1 = none["ever_reached"] <= 0.25
        r2 = best["holds_peak"] <= 0.6 and best["peak_fraction_of_run"] <= 0.5
        r3 = oracle["holds_peak"] <= 0.6
        verdicts[map_key] = (r1, r2, r3, none, best, oracle)
    return verdicts


def screening_table() -> str:
    """Per map and setting: peak, final, how much of the peak is held, and goal-reaching shares."""
    by_map = screening_rows()
    lines = [r"\begin{tabular}{@{}>{\raggedright\arraybackslash}p{4.6cm} r r r r r r@{}}",
             r"\toprule",
             r"\textbf{bonus and weight} & \textbf{\shortstack[c]{peak\\return}} & "
             r"\textbf{\shortstack[c]{steps at\\peak (M)}} & "
             r"\textbf{\shortstack[c]{final-10\%\\return}} & "
             r"\textbf{\shortstack[c]{final\\/ peak}} & "
             r"\textbf{\shortstack[c]{copies ever\\at the goal}} & "
             r"\textbf{\shortstack[c]{final share of\\episodes at goal}} \\",
             r"\midrule"]
    for k, (map_key, title) in enumerate(SCREEN_MAPS):
        rows = by_map[map_key]
        finals = [round(n["final"], 1) for _, _, n in rows]
        holds = [round(n["holds_peak"], 2) for _, _, n in rows]
        if k:
            lines.append(r"\midrule")
        lines.append(r"\multicolumn{7}{@{}l}{\textit{" + title + "}} \\\\")
        for (bonus, weight, n), f, h in zip(rows, finals, holds):
            name = BONUS_NAMES[bonus] + ("" if bonus == "none" else f", $\\beta={weight:g}$")
            lines.append(
                f"{name} & {n['peak']:.1f} & {n['peak_step'] / 1e6:.1f} & "
                f"{mark(finals, f, f'{f:.1f} $\\pm$ {n['final_se']:.1f}')} & "
                f"{mark(holds, h, f'{h:.2f}')} & {n['ever_reached']:.2f} & "
                f"{n['reach_share_final']:.2f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def screening_verdict_table() -> str:
    """Pass or fail of each screening rule per map, with the number each rule reads."""
    verdicts = screening_verdicts(screening_rows())
    word = lambda ok: "pass" if ok else r"\textbf{fail}"
    lines = [r"\begin{tabular}{@{}l >{\raggedright\arraybackslash}p{3.3cm} "
             r">{\raggedright\arraybackslash}p{4.1cm} >{\raggedright\arraybackslash}p{3.3cm}@{}}",
             r"\toprule",
             r"\textbf{map} & \textbf{1. no bonus at the goal in at most 25\% of copies} & "
             r"\textbf{2. best RND setting holds at most 60\% of its peak, peak in the first half} & "
             r"\textbf{3. oracle $1/\sqrt{n}$ holds at most 60\% of its peak} \\",
             r"\midrule"]
    for map_key, title in SCREEN_MAPS:
        r1, r2, r3, none, best, oracle = verdicts[map_key]
        lines.append(
            f"{title} & {word(r1)} ({none['ever_reached'] * 100:.0f}\\%) & "
            f"{word(r2)} ($\\beta={best['intrinsic_weight']:g}$: {best['holds_peak']:.2f}, "
            f"peak at {best['peak_fraction_of_run'] * 100:.0f}\\% of the run) & "
            f"{word(r3)} ({oracle['holds_peak']:.2f}) \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


TABLES = {"pm-jax-speed-jaguar03": speed_table,
          "pm-jax-screening-results": screening_table,
          "pm-jax-screening-verdicts": screening_verdict_table}


def splice(text: str, name: str, body: str) -> str:
    """Replace what lies between the named table's two marker comments."""
    start = f"% >>> AUTO-GENERATED TABLE START: {name}"
    end = f"% <<< AUTO-GENERATED TABLE END: {name}"
    pattern = re.compile(re.escape(start) + r"\n(?:.*?\n)?" + re.escape(end), re.S)
    if len(pattern.findall(text)) != 1:
        raise SystemExit(f"expected exactly one marker pair for {name}")
    return pattern.sub(lambda _: f"{start}\n{body}\n{end}", text)


def main():
    """Rewrite every table (or the one named) in the document."""
    parser = argparse.ArgumentParser()
    parser.add_argument("document", type=Path)
    parser.add_argument("--only")
    args = parser.parse_args()
    text = args.document.read_text()
    for name, build in TABLES.items():
        if args.only and name != args.only:
            continue
        text = splice(text, name, build())
    args.document.write_text(text)


if __name__ == "__main__":
    main()
