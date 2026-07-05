#!/usr/bin/env python3
"""Follow-up sweep (unit norm, ridge 1e-8, no clip) results: per-beta table vs the replica and run-2.

Same definitions as make_results.py:
- Per run, R_i = final eval extrinsic reward (100-episode deterministic eval at 1e6 steps).
- Per beta: Rbar = mean(R_i), SE = std(ddof=1)/sqrt(n), success = fraction of seeds with R_i > 5,
  rho = mean measured effective ridge ratio from intrinsic_diagnostics.
- Only completed=true records count (checkpoint partials from killed/in-flight attempts are excluded).

Reference rows (recomputed live, not hardcoded): the run-2 replica cell from the 3.1.2 arm sweeps
(raw features, lambda=1e-6, no clip, beta=0.01) and run-2 itself (rnd_elliptical, beta=0.01).

Outputs: <plots>/followup_reward_table.tex (input by main.tex) and a Markdown table + coverage counts
on stdout (pasted into analysis.md). Rerun any time; the table refreshes as more seeds complete.
Usage: make_followup_interim.py [plots_dir]
"""
from __future__ import annotations

import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parent
RUN_DIR = CODE_DIR.parent.parent
FOLLOWUP_SWEEP = "2026-07-03-21-22_unit_ridge-1e-8_clip-inf_beta-1e-4-1e-3-1e-2-1e-1-1e0_50seed_16x1"
ARM_SWEEPS = ["2026-07-02-03-52_armA-8tasks-2cpu_replicate-ablate",
              "2026-07-02-03-52_armB-16tasks-1cpu_replicate-ablate"]
RUN2_DATA = RUN_DIR.parent / "2026-06-24-21-24_run_2_after_reorganization" / "data" / "local"
BETAS = [0.0001, 0.001, 0.01, 0.1, 1.0]
SUCCESS_THRESHOLD = 5.0
EVAL_KEY = "eval/mean_extrinsic_reward"
TRAIN_KEY = "train/mean_extrinsic_reward"


def _finite(x) -> bool:
    """True for a real finite number."""
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def _load_completed(data_dir: Path) -> list[dict]:
    """Load every completed run JSON in one sweep's local/ dir into tidy dicts.

    Before: per-run JSON with eval_history rows like {"step": 1000000, "eval/mean_extrinsic_reward": 81.65}
    After:  {"beta": 0.0001, "seed": 0, "final_eval": 81.65, "final_train": 75.2, "rho": 1.28e-06,
             "norm": "unit", "ridge": 1e-08, "clip": inf}
    """
    runs = []
    for jf in sorted(data_dir.glob("*.json")):
        try:
            r = json.loads(jf.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        if r.get("completed", True) is False:
            continue
        ev = [(int(e["step"]), float(e[EVAL_KEY])) for e in r.get("eval_history", [])
              if EVAL_KEY in e and _finite(e.get(EVAL_KEY))]
        tr = [(int(e["step"]), float(e[TRAIN_KEY])) for e in r.get("train_history", [])
              if TRAIN_KEY in e and _finite(e.get(TRAIN_KEY))]
        if not ev:
            continue
        d = r.get("intrinsic_diagnostics") or {}
        runs.append({
            "algorithm": r.get("algorithm"),
            "beta": float(r["beta"]),
            "seed": int(r["a_seed"]),
            "final_eval": sorted(ev)[-1][1],
            "final_train": sorted(tr)[-1][1] if tr else None,
            "rho": d.get("effective_ridge_ratio"),
            "norm": r.get("elliptical_feature_normalization"),
            "ridge": float(r["elliptical_regularization"]) if "elliptical_regularization" in r else None,
            "clip": float(r["elliptical_bonus_clip"]) if "elliptical_bonus_clip" in r else None,
        })
    return runs


def _stats(group: list[dict]) -> dict:
    """One table row's numbers from a group of runs (n, mean, SE, success rate, mean rho, mean train R)."""
    R = [g["final_eval"] for g in group]
    trains = [g["final_train"] for g in group if g["final_train"] is not None]
    rhos = [g["rho"] for g in group if _finite(g.get("rho"))]
    return {
        "n": len(R),
        "mean": statistics.mean(R),
        "se": statistics.stdev(R) / math.sqrt(len(R)) if len(R) > 1 else float("nan"),
        "success": sum(1 for x in R if x > SUCCESS_THRESHOLD) / len(R),
        "train_mean": statistics.mean(trains) if trains else None,
        "rho": statistics.mean(rhos) if rhos else None,
    }


def _sci(x) -> str:
    """Compact scientific LaTeX for a measured value (1.28e-06 -> $1.3\\times10^{-6}$); --- when absent."""
    if x is None or x == 0 or not _finite(x):
        return "---"
    k = math.floor(math.log10(abs(x)))
    m = x / 10 ** k
    return f"${m:.1f}\\times10^{{{k}}}$"


def _pow10(x: float) -> str:
    """LaTeX power-of-ten for a grid value (0.0001 -> $10^{-4}$)."""
    return f"$10^{{{round(math.log10(x))}}}$"


def _mark(values: list, formatted: list[str]) -> list[str]:
    """Bold the best and underline the second-best of one metric column (higher is better; None skipped)."""
    idx = [i for i, v in enumerate(values) if v is not None and _finite(v)]
    order = sorted(idx, key=lambda i: values[i], reverse=True)
    out = list(formatted)
    if order:
        out[order[0]] = f"\\textbf{{{out[order[0]]}}}"
    if len(order) > 1:
        out[order[1]] = f"\\underline{{{out[order[1]]}}}"
    return out


def main() -> None:
    """Build the per-beta rows + two reference rows, write the LaTeX tabular, print the Markdown table."""
    plots = Path(sys.argv[1]) if len(sys.argv) > 1 else RUN_DIR / "analysis" / "plots"
    plots.mkdir(parents=True, exist_ok=True)

    # trained block: the follow-up sweep's completed runs grouped by beta (ascending)
    fu = _load_completed(RUN_DIR / "data" / FOLLOWUP_SWEEP / "local")
    by_beta = defaultdict(list)
    for r in fu:
        by_beta[r["beta"]].append(r)
    rows = []
    for b in BETAS:
        grp = by_beta.get(b, [])
        if len(grp) < 2:
            continue
        rows.append({"label_tex": f"unit, $\\lambda{{=}}10^{{-8}}$, no clip, $\\beta{{=}}{_pow10(b)[1:-1]}$",
                     "label_md": f"unit, ridge 1e-8, no clip, beta={b:g}", **_stats(grp)})

    # reference block: the replica cell from the 3.1.2 arm sweeps, and run-2 itself
    arms = []
    for sw in ARM_SWEEPS:
        arms.extend(_load_completed(RUN_DIR / "data" / sw / "local"))
    replica = [r for r in arms if r["norm"] == "none" and r["ridge"] == 1e-06
               and r["clip"] is not None and math.isinf(r["clip"]) and r["beta"] == 0.01]
    run2 = [r for r in _load_completed(RUN2_DATA)
            if r["algorithm"] == "rnd_elliptical" and abs(r["beta"] - 0.01) < 1e-9]
    refs = [{"label_tex": "replica (raw, $\\lambda{=}10^{-6}$, no clip, $\\beta{=}10^{-2}$), run 3.1.2",
             "label_md": "replica (raw, ridge 1e-6, no clip, beta=0.01), run 3.1.2", **_stats(replica)},
            {"label_tex": "run 2 (raw, $\\lambda{=}10^{-6}$, no clip, $\\beta{=}10^{-2}$)",
             "label_md": "run 2 (raw, ridge 1e-6, no clip, beta=0.01)", **_stats(run2)}]
    refs.sort(key=lambda r: r["mean"], reverse=True)  # other-models block sorted by the primary metric

    # per-column best/second marks across ALL rows (trained + references)
    allr = rows + refs
    mean_s = _mark([r["mean"] for r in allr], [f"{r['mean']:.2f}" for r in allr])
    succ_s = _mark([r["success"] for r in allr], [f"{r['success']:.2f}" for r in allr])
    train_s = _mark([r["train_mean"] for r in allr],
                    ["---" if r["train_mean"] is None else f"{r['train_mean']:.2f}" for r in allr])

    # LaTeX tabular: trained block, midrule, reference block; single down-arrow on the sort column Rbar
    lines = []
    for i, r in enumerate(allr):
        lines.append(f"{r['label_tex']} & {_sci(r['rho'])} & {mean_s[i]} & "
                     f"{r['se']:.2f} & {succ_s[i]} & {int(r['n'])} & {train_s[i]} \\\\")
        if i == len(rows) - 1:
            lines.append("\\midrule")
    (plots / "followup_reward_table.tex").write_text(
        "% Follow-up sweep table (unit norm, ridge 1e-8, no clip; one row per beta) vs the replica cell\n"
        "% and run-2. Regenerated by analysis/code/make_followup_interim.py; refreshes as seeds finish.\n"
        "\\begin{tabular}{@{}l c r r r r r@{}}\n"
        "\\toprule\n"
        "\\textbf{Configuration} & \\textbf{$\\rho$ (meas.)} & \\textbf{$\\bar{R}$ ($\\downarrow$)} & "
        "\\textbf{SE} & \\textbf{succ.} & \\textbf{$n$} & \\textbf{$\\bar{R}_{\\text{train}}$} \\\\\n"
        "\\midrule\n"
        + "\n".join(lines) + "\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
    )

    # Markdown table for analysis.md (same rows; ** best / <u> second marks transcribed from the tex marks)
    def md(s: str) -> str:
        return s.replace("\\textbf{", "**").replace("\\underline{", "<u>").replace("}", "**", 1) \
            if s.startswith("\\textbf{") else (s.replace("\\underline{", "<u>")[:-1] + "</u>"
                                               if s.startswith("\\underline{") else s)
    print("| configuration | rho measured | mean final eval reward | SE | success rate | n | mean final train reward |")
    print("|---|---|---|---|---|---|---|")
    for i, r in enumerate(allr):
        rho_md = "—" if r["rho"] is None else f"{r['rho']:.2e}"
        print(f"| {r['label_md']} | {rho_md} | {md(mean_s[i])} | {r['se']:.2f} | {md(succ_s[i])} | "
              f"{int(r['n'])} | {md(train_s[i])} |")
    done = len(fu)
    per_beta = "  ".join(f"beta={b:g}: {len(by_beta.get(b, []))}" for b in BETAS)
    print(f"\ncompleted runs loaded: {done}/250  ({per_beta})", file=sys.stderr)


if __name__ == "__main__":
    main()
