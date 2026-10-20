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
    lines = [r"\begin{tabular}{@{}>{\raggedright\arraybackslash}p{4.3cm} r r r r r r@{}}",
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
    lines = [r"\begin{tabular}{@{}>{\raggedright\arraybackslash}p{2.4cm} "
             r">{\raggedright\arraybackslash}p{3.1cm} "
             r">{\raggedright\arraybackslash}p{4.4cm} >{\raggedright\arraybackslash}p{3.1cm}@{}}",
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


BATCH_NAMES = {
    "control": "control", "e1_terminate-at-goal": "E1 end at the goal",
    "e2_terminate-at-goal_reward-shift-minus-1": "E2 end at the goal, step cost",
    "e3_terminate-at-goal_goal-reward-remaining-steps": "E3 end at the goal, lump sum",
    "a1_intrinsic-weight-linear-decay": "A1 weight decay",
    "a2_intrinsic-weight-goal-rate-gate-0.05": "A2 goal-rate gate",
    "a3_clipped-ppo_epochs-4_minibatches-4": "A3 clipped PPO",
    "a3_clipped-ppo_epochs-4_minibatches-4_second-attempt": "A3 clipped PPO",
    "a4_entropy-coefficient-0.01": "A4 entropy 0.01",
    "a5_normalize-extrinsic-reward": "A5 task-reward scaling",
    "a6_extrinsic-discount-0.99": "A6 discount 0.99",
    "a7_gradient-clip-per-network": "A7 clip per network",
    "a8_learning-rate-annealed-over-the-run": "A8 rate annealed within the run",
    "b1_logstd-floor-minus-0.5": "B1 log std floor $-0.5$",
    "b2_logstd-floor-minus-0.2": "B2 log std floor $-0.2$",
    "b3_logstd-floor-0": "B3 log std floor $0$",
    "b4_critic-step-0.1": "B4 critic step $0.1$",
    "b5_terminate-at-goal_reward-shift-minus-1_gradient-clip-per-network": "B5 E2 with A7",
}


def batch_summary(run: Path) -> dict:
    """{unit: {weight: numbers}} for a batch, keeping one attempt per idea (complete ones first).

    Paired rows from other runs (same program shape, seeds and weights; the screening's large-maze
    RND unit is batch 1's control with the rate annealed over the run) come from the batch's
    analysis/data/paired_summary.json when it exists.
    """
    summary = json.loads((run / "analysis" / "data" / "summary.json").read_text())
    paired = run / "analysis" / "data" / "paired_summary.json"
    if paired.exists():
        summary.update(json.loads(paired.read_text()))
    out = {}
    for numbers in summary.values():
        unit = numbers["unit"].replace("_second-attempt", "")
        slot = out.setdefault(unit, {})
        w = numbers["intrinsic_weight"]
        if w not in slot or (numbers["complete"] and not slot[w]["complete"]):
            slot[w] = numbers
    return out


def fixed(unit: str, n: dict, control_peak: dict) -> bool:
    """The study's test: holds 90% of its own peak and still learns (see the subsection text).

    control_peak is the control setting with the highest peak over all weights.
    """
    holds = n["holds_peak"] >= 0.9
    if unit.startswith("e"):
        return holds and n["reach_share_final"] >= control_peak["reach_share_peak"]
    return holds and n["final"] > control_peak["peak"] - control_peak["peak_se"]


def batch_results_table(run: Path) -> str:
    """One row per idea at its best weight (highest final-10% return), sorted by that return."""
    data = batch_summary(run)
    best = {u: max(ws.values(), key=lambda n: n["final"]) for u, ws in data.items()}
    control = max(data["control"].values(), key=lambda n: n["peak"])
    rows = sorted(best.items(), key=lambda kv: -kv[1]["final"])
    finals = [round(n["final"], 2) for _, n in rows]
    holds = [round(n["holds_peak"], 2) for _, n in rows]
    reach = [round(n["reach_share_final"], 2) for _, n in rows]
    lines = [r"\begin{tabular}{@{}>{\raggedright\arraybackslash}p{3.9cm} r r r r r r l@{}}",
             r"\toprule",
             r"\textbf{idea} & \textbf{$\beta$} & \textbf{\shortstack[c]{peak\\return}} & "
             r"\textbf{\shortstack[c]{steps at\\peak (M)}} & "
             r"\textbf{\shortstack[c]{final-10\%\\return $\uparrow$}} & "
             r"\textbf{\shortstack[c]{final\\/ peak}} & "
             r"\textbf{\shortstack[c]{final share of\\episodes at goal}} & \textbf{fixed} \\",
             r"\midrule"]
    for (unit, n), f, h, g in zip(rows, finals, holds, reach):
        done = "" if n["complete"] else r"$^{\ast}$"
        lines.append(
            f"{BATCH_NAMES.get(unit, unit)}{done} & {n['intrinsic_weight']:g} & {n['peak']:.2f} & "
            f"{n['peak_step'] / 1e6:.1f} & {mark(finals, f, f'{f:.2f} $\\pm$ {n['final_se']:.2f}')} & "
            f"{mark(holds, h, f'{h:.2f}')} & {mark(reach, g, f'{g:.2f}')} & "
            f"{'yes' if fixed(unit, n, control) else 'no'} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def batch_holds_grid(run: Path) -> str:
    """final / peak for every idea and weight, and the final-10% return beside it."""
    data = batch_summary(run)
    weights = sorted({w for ws in data.values() for w in ws})
    order = [u for u in BATCH_NAMES if u in data]
    lines = [r"\begin{tabular}{@{}>{\raggedright\arraybackslash}p{3.9cm} " + "r " * len(weights)
             + "@{}}", r"\toprule",
             r"\textbf{idea} & " + " & ".join(rf"\textbf{{$\beta={w:g}$}}" for w in weights)
             + r" \\", r"\midrule"]
    for unit in order:
        cells = []
        for w in weights:
            n = data[unit].get(w)
            cells.append("--" if n is None else f"{n['holds_peak']:.2f} ({n['final']:.1f})")
        lines.append(f"{BATCH_NAMES.get(unit, unit)} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


BATCH1 = ROOT / ("runs/2026-10-19-16-46_reward-decline-batch-1_large-cap400_rnd-weights-1-10-100-"
                 "1000_learning-rate-1e-3-annealed-over-1000M_steps-50M_seeds-0-127_control-e1-e2-"
                 "e3-a1-a2-a3-a4-a5-a6-a7")


TABLES = {"pm-jax-speed-jaguar03": speed_table,
          "pm-jax-batch1-results": lambda: batch_results_table(BATCH1),
          "pm-jax-batch1-holds": lambda: batch_holds_grid(BATCH1),
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
        # a table whose markers are not in the document yet is skipped, not built
        if (args.only and name != args.only) or f"TABLE START: {name}" not in text:
            continue
        text = splice(text, name, build())
    args.document.write_text(text)


if __name__ == "__main__":
    main()
