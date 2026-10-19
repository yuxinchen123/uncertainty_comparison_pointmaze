"""Turn the jaguar03 speed measurements into analysis.md, a LaTeX table body and a two-panel plot.

Per idea: the mean over its replicates (a multi-program idea first sums its programs' rates on one
card). Its speed relative to the baseline divides by the mean of the baselines that ran in the same
rounds, so node load is matched. The agreement column compares setting 0's parameters after ten
iterations with the first baseline's: "identical" when every byte matches, else the largest
absolute difference.

Usage: python analyze.py <benchmark folder>
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

LABELS = {
    "baseline": "baseline: 1 setting x 128 copies, 1 program per card",
    "01_four_settings_one_program": "01 four settings in one program",
    "02_eight_settings_one_program": "02 eight settings in one program",
    "03_two_programs_per_card": "03 two programs per card",
    "04_four_programs_per_card": "04 four programs per card",
    "05_command_buffers_off": "05 XLA command buffers off",
    "06_command_buffers_with_loops": "06 XLA command buffers incl. loops",
    "07_latency_hiding_scheduler": "07 latency-hiding scheduler",
    "08_compilation_cache_cold": "08 compilation cache, cold",
    "08_compilation_cache_warm": "08 compilation cache, warm",
    "09_two_processors": "09 two processors instead of 8",
    "10_diagnostics_on": "10 study diagnostics on",
}


def load(folder: Path):
    """Group result files by idea and replicate; return {idea: [ {per-replicate numbers} ]}."""
    plan = json.loads((folder / "data" / "plan.json").read_text())
    by = defaultdict(lambda: defaultdict(list))
    for f in sorted((folder / "data").glob("*__rep*__card*__proc*.json")):
        idea, rep, card, proc = f.stem.split("__")
        by[idea][(rep, card)].append(json.loads(f.read_text()))
    out = {}
    for idea, reps in by.items():
        rows = []
        for (rep, card), procs in reps.items():
            total = sum(p["steps_per_second_total"] for p in procs)
            rows.append({"rep": rep, "card": card, "programs": len(procs),
                         "copies": sum(p["copies"] for p in procs),
                         "settings": sum(p["settings"] for p in procs),
                         "seconds_per_iteration": float(np.mean([p["seconds_per_iteration"]
                                                                 for p in procs])),
                         "total": total, "per_copy": total / sum(p["copies"] for p in procs),
                         "setup": float(np.mean([p["build_seconds"] + p["compile_seconds"]
                                                 for p in procs])),
                         "memory_mb": sum(p["peak_device_memory_mb"] for p in procs)})
        out[idea] = rows
    return plan, out


def agreement(folder: Path, idea: str, base_npz: Path) -> str:
    """How many of setting 0's 128 copies hold bit-identical parameters to the baseline's after ten
    iterations, and the largest absolute difference among the rest.

    For the baseline itself the comparison is between two of its own runs on different cards,
    which measures the card's run-to-run variation between compilations of one program.
    """
    files = [f for f in sorted((folder / "data").glob(f"{idea}__rep*__proc0.npz")) if f != base_npz]
    if not files:
        return "missing"
    a, b = np.load(base_npz), np.load(files[0])
    if set(a.files) != set(b.files):
        return "different tensors"
    # before: tensors [128, ...] per run; after: one flag per copy, True when every byte matches
    same = np.ones(a[a.files[0]].shape[0], bool)
    worst = 0.0
    for k in a.files:
        x, y = a[k].reshape(len(same), -1), b[k].reshape(len(same), -1)
        same &= (x == y).all(axis=1)
        worst = max(worst, float(np.max(np.abs(x.astype(float) - y.astype(float)))))
    return f"{int(same.sum())} of {len(same)} identical; largest difference {worst:.1e}"


def main():
    """Write analysis.md, the LaTeX rows and the plot from the data folder."""
    folder = Path(sys.argv[1]).resolve()
    plan, results = load(folder)
    base = results["baseline"]
    base_total = float(np.mean([r["total"] for r in base]))
    base_npz = sorted((folder / "data").glob("baseline__rep*__proc0.npz"))[0]
    rows = []
    for idea in LABELS:
        if idea not in results:
            continue
        r = results[idea]
        total = float(np.mean([x["total"] for x in r]))
        copies = r[0]["copies"]
        per_copy = total / copies
        rows.append({
            "idea": idea, "label": LABELS[idea], "replicates": len(r),
            "copies": copies, "settings": r[0]["settings"], "programs": r[0]["programs"],
            "seconds_per_iteration": float(np.mean([x["seconds_per_iteration"] for x in r])),
            "total_millions": total / 1e6, "per_copy_thousands": per_copy / 1e3,
            "hours_per_million_per_copy": 1e6 / (3600 * per_copy),
            "relative_total": total / base_total,
            "setup_seconds": float(np.mean([x["setup"] for x in r])),
            "memory_mb": float(np.mean([x["memory_mb"] for x in r])),
            "agreement": agreement(folder, idea, base_npz)})
    (folder / "analysis_rows.json").write_text(json.dumps(rows, indent=1) + "\n")

    # the Markdown table: the five throughput columns plus what the study needs to decide
    lines = ["| idea | copies (settings x copies) | seconds per iteration | total steps per second "
             "(millions) | steps per second per copy (thousands) | hours per million steps per copy "
             "| total relative to baseline | setup seconds | setting 0's copies after 10 iterations, against the baseline |",
             "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['label']} | {r['copies']} ({r['settings']} x 128) | "
                     f"{r['seconds_per_iteration']:.4f} | {r['total_millions']:.2f} | "
                     f"{r['per_copy_thousands']:.1f} | {r['hours_per_million_per_copy']:.4f} | "
                     f"{r['relative_total']:.2f} | {r['setup_seconds']:.0f} | {r['agreement']} |")
    (folder / "analysis_table.md").write_text("\n".join(lines) + "\n")

    # the two-panel plot: total and per-copy rate against copies on the card, for the two ways of
    # putting more copies on one card (the flag ideas all run 128 copies and are in the table)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    by = {r["idea"]: r for r in rows}
    series = {"settings packed in one program": ["baseline", "01_four_settings_one_program",
                                                 "02_eight_settings_one_program"],
              "several programs on one card": ["baseline", "03_two_programs_per_card",
                                               "04_four_programs_per_card"]}
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for (name, ideas), marker in zip(series.items(), ("o", "s")):
        x = [by[i]["copies"] for i in ideas]
        axes[0].plot(x, [by[i]["total_millions"] for i in ideas], marker=marker, label=name)
        axes[1].plot(x, [by[i]["per_copy_thousands"] for i in ideas], marker=marker, label=name)
    axes[0].set(xscale="log", xlabel="copies on one RTX A4500",
                ylabel="total steps per second (millions)")
    axes[1].set(xscale="log", xlabel="copies on one RTX A4500",
                ylabel="steps per second per copy (thousands)")
    for ax in axes:
        ax.set_xticks([128, 256, 512, 1024], ["128", "256", "512", "1024"])
        ax.minorticks_off()
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    (folder / "plots").mkdir(exist_ok=True)
    fig.savefig(folder / "plots" / "speed_ideas.pdf")
    fig.savefig(folder / "plots" / "speed_ideas.png", dpi=150)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
