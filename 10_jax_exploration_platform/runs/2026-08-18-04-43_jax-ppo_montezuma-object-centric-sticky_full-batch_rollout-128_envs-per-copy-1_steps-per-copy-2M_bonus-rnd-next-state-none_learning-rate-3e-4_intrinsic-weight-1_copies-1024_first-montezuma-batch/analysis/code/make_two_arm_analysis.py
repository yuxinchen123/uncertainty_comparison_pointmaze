"""Analysis for a two-arm (rnd_next_state vs none) single-setting run: coverage and reward
curves per window, and the run's results table as markdown + a LaTeX tabular block.

Copied into <run>/analysis/code/ at run completion and executed there; reads the run's own
data/ shards, writes plots/ and tables/ beside itself, per the analysis-folder rule.

Usage: python make_two_arm_analysis.py <run_dir> <coverage_label>
  coverage_label: e.g. "maze-cell coverage" or "rooms visited"
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ARMS = [("unit-rnd_next_state", "rnd_next_state", "black", "-"),
        ("unit-none", "no_exploration", "tab:red", "--")]


def load_unit(run_dir: Path, unit_id: str):
    """One unit's episode-window records, in window order.

    before: data/unit-rnd_next_state.jsonl, one json per line
    after:  list of dicts with reward_ext_sum_per_copy [C], coverage_per_copy [C], env_steps
    """
    rows = []
    with open(run_dir / "data" / f"{unit_id}.jsonl") as f:
        for line in f:
            r = json.loads(line)
            if r.get("record") == "episode_window":
                rows.append(r)
    rows.sort(key=lambda r: r["window_index"])
    return rows


def curves(run_dir: Path, coverage_label: str):
    """The two-panel figure: mean coverage and mean window reward against env steps per copy."""
    fig, (ax_cov, ax_rew) = plt.subplots(1, 2, figsize=(11, 4))
    for unit_id, label, color, ls in ARMS:
        rows = load_unit(run_dir, unit_id)
        steps = np.array([r["env_steps_per_copy"] for r in rows])
        cov = np.array([r["coverage_per_copy"] for r in rows])          # [W, C]
        rew = np.array([r["reward_ext_sum_per_copy"] for r in rows])    # [W, C]
        for ax, y in ((ax_cov, cov), (ax_rew, rew)):
            m = y.mean(axis=1)
            se = y.std(axis=1, ddof=1) / np.sqrt(y.shape[1])
            ax.plot(steps, m, color=color, linestyle=ls, label=label)
            ax.fill_between(steps, m - se, m + se, color=color, alpha=0.2)
    ax_cov.set_ylabel(f"{coverage_label} (fraction, mean over copies)")
    ax_rew.set_ylabel("extrinsic reward per copy summed over the window")
    for ax in (ax_cov, ax_rew):
        ax.set_xlabel("environment steps per copy")
        ax.legend(frameon=False)
    fig.tight_layout()
    out = run_dir / "analysis" / "plots"
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "two_arm_curves.pdf")
    fig.savefig(out / "two_arm_curves.png", dpi=150)
    return out / "two_arm_curves.pdf"


def table(run_dir: Path, coverage_label: str):
    """Markdown + LaTeX rows: per arm, whole-run reward, last-window reward, success fraction,
    final coverage; mean +- standard error over copies where a mean applies."""
    md = [f"| arm | whole-run reward per copy | last-window reward | success rate | final {coverage_label} % | N |",
          "|---|---|---|---|---|---|"]
    tex = []
    for unit_id, label, _, _ in ARMS:
        rows = load_unit(run_dir, unit_id)
        rew = np.array([r["reward_ext_sum_per_copy"] for r in rows])   # [W, C]
        total = rew.sum(axis=0)                                        # [C]
        last = rew[-1]
        cov = np.array(rows[-1]["coverage_per_copy"]) * 100.0
        C = total.shape[0]
        succ = float((total > 0).mean())
        def pm(x):
            return f"{x.mean():.3f} ± {x.std(ddof=1) / np.sqrt(len(x)):.3f}"
        md.append(f"| `{label}` | {pm(total)} | {pm(last)} | {succ:.3f} | "
                  f"{cov.mean():.2f} ± {cov.std(ddof=1) / np.sqrt(C):.2f} | {C} |")
        tex.append(
            f"\\texttt{{{label.replace('_', chr(92) + '_')}}} & "
            f"${total.mean():.3f} \\pm {total.std(ddof=1) / np.sqrt(C):.3f}$ & "
            f"${last.mean():.3f} \\pm {last.std(ddof=1) / np.sqrt(C):.3f}$ & "
            f"${succ:.3f}$ & ${cov.mean():.2f} \\pm {cov.std(ddof=1) / np.sqrt(C):.2f}$ & "
            f"${C}$ \\\\")
    out = run_dir / "analysis" / "tables"
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.md").write_text("\n".join(md) + "\n")
    (out / "results_rows.tex").write_text("\n".join(tex) + "\n")
    print("\n".join(md))
    return out


if __name__ == "__main__":
    run_dir = Path(sys.argv[1]).resolve()
    coverage_label = sys.argv[2] if len(sys.argv) > 2 else "coverage"
    curves(run_dir, coverage_label)
    table(run_dir, coverage_label)
    print("analysis written under", run_dir / "analysis")
