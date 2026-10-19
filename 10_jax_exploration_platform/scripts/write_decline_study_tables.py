"""Regenerate the reward-decline study's tables inside the development document.

Each table's body sits between a pair of marker comments inside its table float; this script
rewrites only what lies between the markers, so captions and prose are never touched.

Tables:
- pm-jax-speed-jaguar03: the jaguar03 speed ideas, from the benchmark's analysis_rows.json.

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


TABLES = {"pm-jax-speed-jaguar03": speed_table}


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
