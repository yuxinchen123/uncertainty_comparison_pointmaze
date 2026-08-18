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

from aggregate import cell_table                                        # noqa: E402
from curves_all import curves_all, verify_against_curve_of              # noqa: E402
from curve_shape import DEFAULTS, classify, sensitivity                 # noqa: E402

SCORE = "whole_run_reward"      # the column the results table sorts on; shared so ranks agree


def classify_run(run_dir: Path) -> list:
    """One record per configuration: its table row, its shape, and how stable that shape is.

    Every curve comes from ONE pass over the shards (`curves_all`), which the gate above has just
    shown produces the same numbers as `aggregate.curve_of` — sixty-six separate passes over about
    4 GB would otherwise cost an hour at exactly the moment the run finishes.
    """
    everything = curves_all(run_dir)
    out = []
    for row in cell_table(run_dir):
        key = (row["bonus"], row["learning_rate"], row["intrinsic_weight"])
        curve = everything.get(key)
        if curve is None:
            continue
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


def plain_python(value):
    """Turn a numpy scalar into the python one json can write; raise on anything else.

    The classifier works in numpy, so `interior_peak`, `rise_significant` and their neighbours come
    back as `numpy.bool_` rather than `bool`, which `json.dumps` refuses. Converting through
    `.item()` keeps the value exactly and leaves any genuinely unserialisable object raising, which
    is what a `default=` hook should do.

    before: numpy.bool_(True) ; after: True
    """
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(f"cannot write {type(value).__name__} to json")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, default=RUN_DIR)
    arguments = parser.parse_args()
    gate = verify_against_curve_of(arguments.run)
    if not gate["identical"]:
        raise SystemExit(f"the one-pass reader disagrees with aggregate.curve_of: {gate}")
    print(f"reader gate: {gate['configurations_checked']} configurations, "
          f"{gate['windows_compared']} windows, identical to aggregate.curve_of\n")
    records = classify_run(arguments.run)
    if not records:
        print("no completed configuration yet — the table rule keeps unfinished arms out")
        return
    (arguments.run / "analysis" / "shape_classification.json").write_text(
        json.dumps(records, indent=1, default=plain_python))
    text = report(records)
    (arguments.run / "analysis" / "shape_classification.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
