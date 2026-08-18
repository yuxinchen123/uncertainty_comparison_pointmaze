"""Answer one question about this sweep: does every configuration's curve rise and then fall?

Aggregated over a configuration's 128 copies, a learning curve can end in several ways, and the
difference matters for what a long budget is for: a curve still climbing at 1000M steps has not
been given enough steps, while one that peaks and comes back down has been given too many and the
budget itself is what chose the score.

The classifier is `curve_shape.classify`; this module applies it to every configuration of the run
through `code/aggregate.curve_of`, so the shapes and the results table read exactly the same pooled
numbers. `--run` points it at another run folder, which is how it was checked against the completed
10M-step batch before this run's own data existed.

Writes `analysis/shape_classification.md` and `analysis/shape_classification.json`.

Run:
  PYTHONNOUSERSITE=1 /p/rlprojects/RND/.venvs/platform_jax/bin/python analysis/code/shape_report.py
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

RUN_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(RUN_DIR / "code"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from aggregate import cell_table, curve_of                              # noqa: E402
from curve_shape import DEFAULTS, classify, sensitivity                 # noqa: E402

SCORE = "whole_run_reward"      # the column the results table sorts on; shared so ranks agree


def classify_run(run_dir: Path) -> list:
    """One record per configuration: its table row, its shape, and how stable that shape is."""
    out = []
    for row in cell_table(run_dir):
        curve = curve_of(row["bonus"], row["learning_rate"], row["intrinsic_weight"], run_dir)
        means, errors = curve["mean_episode_return"], curve["standard_error"]
        if len(means) < 10:
            continue
        shape = classify(means, errors)
        out.append({**{k: row[k] for k in ("bonus", "learning_rate", "intrinsic_weight",
                                           "copies", SCORE, "last_window_reward")},
                    **shape, "sensitivity": sensitivity(means, errors)})
    out.sort(key=lambda r: -(r[SCORE] or 0.0))
    return out


def report(records: list) -> str:
    """The Markdown answer: the distribution, then the best configuration of each shape per arm."""
    total = len(records)
    counts = Counter(r["shape"] for r in records)
    stable = sum(r["sensitivity"]["stable"] for r in records)
    all_rise_fall = counts.get("rise_then_fall", 0) == total

    lines = [f"# Curve shape, all {total} configurations of this sweep", "",
             f"**Does every configuration rise then fall? {'Yes' if all_rise_fall else 'No'}.**", "",
             "| shape | configurations | share |", "|---|---:|---:|"]
    for shape, n in counts.most_common():
        lines.append(f"| `{shape}` | {n} | {100 * n / total:.0f}% |")
    lines += ["", f"Label unchanged under all {len(records[0]['sensitivity']['labels'])} threshold "
                  f"variants: {stable} of {total}.", ""]

    if not all_rise_fall:
        lines += ["## Best configuration that does NOT rise then fall, one per algorithm", "",
                  "| algorithm | learning rate | intrinsic weight | shape | score | rank overall |",
                  "|---|---:|---:|---|---:|---:|"]
        seen = {}
        for rank, r in enumerate(records, start=1):
            if r["shape"] != "rise_then_fall" and r["bonus"] not in seen:
                seen[r["bonus"]] = (rank, r)
        for arm, (rank, r) in sorted(seen.items()):
            lines.append(f"| {arm} | {r['learning_rate']:.0e} | {r['intrinsic_weight']:.0e} | "
                         f"`{r['shape']}` | {r[SCORE]:.4f} | {rank} |")
        lines += ["", "The rank column is over every configuration of the sweep. A rank of 1 means "
                      "this arm's best configuration overall is already the one named here, so the "
                      "results table and the curve figure carry it and no separate table or plot "
                      "repeats it.", ""]
    lines += ["## Thresholds", "", "| setting | value |", "|---|---:|"]
    lines += [f"| `{k}` | {v} |" for k, v in DEFAULTS.items()]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, default=RUN_DIR)
    arguments = parser.parse_args()
    records = classify_run(arguments.run)
    if not records:
        print("no completed configuration yet — the table rule keeps unfinished arms out")
        return
    (arguments.run / "analysis" / "shape_classification.json").write_text(
        json.dumps(records, indent=1))
    text = report(records)
    (arguments.run / "analysis" / "shape_classification.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
