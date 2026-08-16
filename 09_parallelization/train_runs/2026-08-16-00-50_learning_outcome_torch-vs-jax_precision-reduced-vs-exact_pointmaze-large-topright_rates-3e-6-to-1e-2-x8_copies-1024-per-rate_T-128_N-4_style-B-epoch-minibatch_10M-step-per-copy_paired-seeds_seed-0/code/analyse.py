"""The learning-outcome campaign, read out of its four run records.

Two questions, one module, so the run folder's `analysis/analysis.md` and the project report
cannot come to disagree: `make_figures(directory)` draws every figure wherever it is told to, and
`summary()` returns every number the tables are built from.

The score a copy is judged by is its **mean extrinsic reward per iteration over the last ten
recorded iterations** — the last ten percent of training, by which point the annealed learning
rate is near zero and the policy has stopped moving. One iteration is 512 environment steps, and
the reward is 1 for every step spent inside the goal radius, so the score reads as "steps in the
goal per 512 steps of behaviour". Averaging ten records rather than reading the last one cuts the
per-copy sampling noise without smearing the learning curve.

Run directly to write `analysis/analysis.md`, `analysis/plots/` and `data/summary.json`:
  /p/rlprojects/RND/.venvs/exploration/bin/python analyse.py
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
from scipy import stats

# the run folder this reads. The override exists so the analysis can be exercised on the pilot's
# short runs before the campaign's own records land, rather than first being tried on the data it
# has to be right about.
RUN = Path(os.environ.get("LEARNING_OUTCOME_RUN", Path(__file__).resolve().parent.parent))
TAGS = ["torch_reduced", "torch_exact", "jax_reduced", "jax_exact"]
LABEL = {"torch_reduced": "PyTorch, reduced precision",
         "torch_exact": "PyTorch, exact single precision",
         "jax_reduced": "JAX, reduced precision",
         "jax_exact": "JAX, exact single precision"}
FINAL_RECORDS = 10          # the last ten recorded iterations define a copy's final score
STEPS_PER_COPY_PER_ITER = 128 * 4

# colour is the learning rate, an ordered quantity, so the three rates take one sequential ramp;
# line style separates the two things being compared, per the plotting brief
RAMP = ["#86b6ef", "#2a78d6", "#0d3a72"]


def load(tag):
    """One configuration: its record, and its history as arrays of shape [records, copies]."""
    d = RUN / "data" / tag
    record = json.loads((d / "record.json").read_text())
    rows = [json.loads(ln) for ln in (d / "history.jsonl").read_text().splitlines() if ln.strip()]
    rows.sort(key=lambda r: r["iteration"])
    # before: one dict per recorded iteration, each with three lists of 8,192 numbers
    # after:  three [records, copies] arrays plus the iteration and step axes
    return {
        "record": record,
        "iteration": np.array([r["iteration"] for r in rows]),
        "steps_per_copy": np.array([r["global_step"] / record["n_copies"] for r in rows]),
        "reward": np.array([r["reward_ext_sum_per_copy"] for r in rows], dtype=float),
        "rint": np.array([r["rint_mean_per_copy"] for r in rows], dtype=float),
        "coverage": np.array([r["coverage_per_copy"] for r in rows], dtype=float),
        "group": np.array(record["group_index"]),
        "rates": list(record["rates"]),
    }


def load_all():
    """Every configuration that has finished, keyed by tag."""
    return {t: load(t) for t in TAGS if (RUN / "data" / t / "record.json").exists()}


def group_mask(data, rate_index):
    """The copies belonging to one learning-rate group."""
    return data["group"] == rate_index


def final_scores(data, rate_index):
    """Per-copy final score for one learning-rate group, shape [copies in the group]."""
    return data["reward"][-FINAL_RECORDS:, group_mask(data, rate_index)].mean(axis=0)


def final_coverage(data, rate_index):
    """Per-copy final maze coverage for one learning-rate group."""
    return data["coverage"][-1, group_mask(data, rate_index)]


def mean_ci(x):
    """Mean and its 95% interval across copies (normal approximation; n is 1,024)."""
    x = np.asarray(x, dtype=float)
    half = 1.959964 * x.std(ddof=1) / np.sqrt(x.size)
    return float(x.mean()), float(x.mean() - half), float(x.mean() + half)


def compare(a, b):
    """Two independent sets of per-copy scores: is the difference between them resolvable?

    Reports the difference of means with a Welch interval, the rank test's p-value, and the
    probability that a copy drawn from the first set scores above one drawn from the second
    (0.5 means the two distributions are interchangeable).
    """
    a, b = np.asarray(a, float), np.asarray(b, float)
    diff = a.mean() - b.mean()
    se = np.sqrt(a.var(ddof=1) / a.size + b.var(ddof=1) / b.size)
    u, p = stats.mannwhitneyu(a, b, alternative="two-sided")
    return {"mean_a": float(a.mean()), "mean_b": float(b.mean()),
            "difference": float(diff), "difference_lo": float(diff - 1.959964 * se),
            "difference_hi": float(diff + 1.959964 * se),
            "relative_difference": float(diff / b.mean()) if b.mean() else float("nan"),
            "rank_test_p": float(p),
            "probability_a_above_b": float(u / (a.size * b.size)),
            "n_a": int(a.size), "n_b": int(b.size)}


def pooled_over_rates(a_groups, b_groups):
    """One verdict over all eight learning rates, blocked BY rate rather than concatenated.

    Concatenating would pool distributions whose means differ by a factor of a hundred, so the
    spread of the pooled sample would be dominated by which rate a copy belongs to and every
    difference would drown in it. Blocking asks the same question inside each rate and then
    combines: the difference is the mean of the eight per-rate differences, and the rank statistic
    is the sum of the eight rank sums against its own null variance (the stratified rank test with
    equal weights). No tie correction is applied, which makes the p-value conservative — the
    scores are integer counts, so ties are common and correcting for them would only shrink the
    null variance.
    """
    diffs, variances, u_total, e_total, var_total, pairs = [], [], 0.0, 0.0, 0.0, 0
    for a, b in zip(a_groups, b_groups):
        a, b = np.asarray(a, float), np.asarray(b, float)
        diffs.append(a.mean() - b.mean())
        variances.append(a.var(ddof=1) / a.size + b.var(ddof=1) / b.size)
        u, _ = stats.mannwhitneyu(a, b, alternative="two-sided")
        n1, n2 = a.size, b.size
        u_total += u
        e_total += n1 * n2 / 2
        var_total += n1 * n2 * (n1 + n2 + 1) / 12
        pairs += n1 * n2
    k = len(diffs)
    diff = float(np.mean(diffs))
    se = float(np.sqrt(np.sum(variances)) / k)
    z = (u_total - e_total) / np.sqrt(var_total)
    mean_b = float(np.mean([np.mean(b) for b in b_groups]))
    return {"mean_a": float(np.mean([np.mean(a) for a in a_groups])), "mean_b": mean_b,
            "difference": diff, "difference_lo": diff - 1.959964 * se,
            "difference_hi": diff + 1.959964 * se,
            "relative_difference": diff / mean_b if mean_b else float("nan"),
            "rank_test_p": float(2 * stats.norm.sf(abs(z))),
            "probability_a_above_b": float(u_total / pairs),
            "n_a": int(sum(len(a) for a in a_groups)),
            "n_b": int(sum(len(b) for b in b_groups))}


def chosen_rates(all_data):
    """Three of the eight rates, spanning the useful range: the best and its two neighbours.

    Chosen from the data rather than by hand, and from ALL configurations pooled, so the same
    three appear in every figure and no configuration gets a rate picked to flatter it.
    """
    rates = next(iter(all_data.values()))["rates"]
    pooled = [np.mean([final_scores(d, i).mean() for d in all_data.values()])
              for i in range(len(rates))]
    best = int(np.argmax(pooled))
    # before: best = 4 of 0..7; after: the window [3, 4, 5], clamped at the ends of the ladder
    lo = min(max(best - 1, 0), len(rates) - 3)
    return [lo, lo + 1, lo + 2]


# ---------------- figures ----------------

def _style(ax):
    """Recessive grid and spines, matching the project report's figures."""
    ax.grid(True, which="major", color="#d9d9d9", linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def seed_band(ax, x, y, color, linestyle, label):
    """One curve: the mean over copies, with the middle half of the copies shaded behind it.

    The band is the spread ACROSS SEEDS, not the uncertainty of the mean. The question these
    figures are for is whether two seed distributions agree, and at a thousand copies the mean's
    own interval is about a sixteenth as wide as the spread — a band of that width would be a
    line. The intervals for the means are in the tables, where they can be read as numbers.

    before: y is [records, copies] of one rate group's per-iteration reward
    after:  a mean line and a fill between the 25th and 75th percentile at every record
    """
    ax.plot(x, y.mean(axis=1), linestyle, color=color, linewidth=1.7, label=label)
    ax.fill_between(x, np.percentile(y, 25, axis=1), np.percentile(y, 75, axis=1),
                    color=color, alpha=0.13, linewidth=0)


def fig_frameworks(all_data, figdir, name="learning_outcome_frameworks.png"):
    """PyTorch against JAX at three learning rates, both at reduced precision."""
    import matplotlib.pyplot as plt
    if not {"torch_reduced", "jax_reduced"} <= set(all_data):
        return "learning-outcome figure: waiting on the two reduced-precision runs"
    rates = all_data["torch_reduced"]["rates"]
    picks = chosen_rates(all_data)
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.3), dpi=160)
    for slot, ri in enumerate(picks):
        c = RAMP[slot]
        for ax, key in zip(axes, ["reward", "coverage"]):
            for tag, ls, side in [("torch_reduced", "-", "PyTorch"), ("jax_reduced", "--", "JAX")]:
                d = all_data[tag]
                m = group_mask(d, ri)
                x = d["steps_per_copy"] / 1e6
                seed_band(ax, x, d[key][:, m], c, ls, f"{side}, rate {rates[ri]:g}")
    axes[0].set_ylabel("extrinsic reward per copy per iteration")
    axes[1].set_ylabel("maze coverage (fraction of open cells)")
    for ax in axes:
        ax.set_xlabel("environment steps per copy [millions]")
        _style(ax)
    axes[0].legend(frameon=False, fontsize=8, ncol=2)
    per_rate = int(group_mask(all_data["torch_reduced"], picks[0]).sum())
    fig.suptitle(f"PyTorch (solid) and JAX (dashed), {per_rate:,} copies per learning rate, "
                 "line = mean, band = middle half of the copies", fontsize=11)
    fig.tight_layout()
    fig.savefig(Path(figdir) / name)
    plt.close(fig)
    return None


def fig_precision(all_data, figdir, framework, name=None):
    """Reduced against exact single precision inside one framework, at three learning rates."""
    import matplotlib.pyplot as plt
    name = name or f"learning_outcome_precision_{framework}.png"
    need = {f"{framework}_reduced", f"{framework}_exact"}
    if not need <= set(all_data):
        return f"learning-outcome figure: waiting on the two {framework} runs"
    rates = all_data[f"{framework}_reduced"]["rates"]
    picks = chosen_rates(all_data)
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.3), dpi=160)
    for slot, ri in enumerate(picks):
        c = RAMP[slot]
        for ax, key in zip(axes, ["reward", "coverage"]):
            for suffix, ls, side in [("reduced", "-", "reduced"), ("exact", "--", "exact")]:
                d = all_data[f"{framework}_{suffix}"]
                m = group_mask(d, ri)
                x = d["steps_per_copy"] / 1e6
                seed_band(ax, x, d[key][:, m], c, ls, f"{side}, rate {rates[ri]:g}")
    axes[0].set_ylabel("extrinsic reward per copy per iteration")
    axes[1].set_ylabel("maze coverage (fraction of open cells)")
    for ax in axes:
        ax.set_xlabel("environment steps per copy [millions]")
        _style(ax)
    axes[0].legend(frameon=False, fontsize=8, ncol=2)
    title = {"torch": "PyTorch", "jax": "JAX"}[framework]
    fig.suptitle(f"{title}: reduced precision (solid) against exact single precision (dashed), "
                 "line = mean, band = middle half of the copies", fontsize=11)
    fig.tight_layout()
    fig.savefig(Path(figdir) / name)
    plt.close(fig)
    return None


def fig_rate_curve(all_data, figdir, name="learning_outcome_by_rate.png"):
    """Final score against learning rate for all four configurations, and the seed spread."""
    import matplotlib.pyplot as plt
    if not all_data:
        return "learning-outcome figure: waiting on the campaign records"
    rates = next(iter(all_data.values()))["rates"]
    colors = {"torch_reduced": "#2a78d6", "torch_exact": "#2a78d6",
              "jax_reduced": "#1baf7a", "jax_exact": "#1baf7a"}
    styles = {"torch_reduced": "-", "torch_exact": "--", "jax_reduced": "-", "jax_exact": "--"}
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.3), dpi=160)
    for tag, d in all_data.items():
        means, los, his = [], [], []
        for ri in range(len(rates)):
            m, lo, hi = mean_ci(final_scores(d, ri))
            means.append(m); los.append(lo); his.append(hi)
        axes[0].errorbar(rates, means, yerr=[np.array(means) - los, np.array(his) - np.array(means)],
                         fmt=styles[tag] + "o", color=colors[tag], linewidth=1.7, markersize=4,
                         capsize=3, label=LABEL[tag])
    axes[0].set_xscale("log")
    axes[0].set_xlabel("learning rate (log scale)")
    axes[0].set_ylabel("final extrinsic reward per copy per iteration")
    axes[0].legend(frameon=False, fontsize=8)

    # the seed distribution at the best rate: the question is whether these overlap, so they are
    # drawn as cumulative distributions rather than summarised into a bar
    best = chosen_rates(all_data)[1]
    for tag, d in all_data.items():
        s = np.sort(final_scores(d, best))
        axes[1].plot(s, np.arange(1, s.size + 1) / s.size, styles[tag], color=colors[tag],
                     linewidth=1.7, label=LABEL[tag])
    axes[1].set_xlabel(f"final extrinsic reward per copy per iteration (rate {rates[best]:g})")
    axes[1].set_ylabel("fraction of copies at or below")
    axes[1].legend(frameon=False, fontsize=8)
    for ax in axes:
        _style(ax)
    per_rate = int(group_mask(next(iter(all_data.values())), best).sum())
    fig.suptitle("Every learning rate, and the seed distribution at the best one "
                 f"({per_rate:,} copies per rate)", fontsize=11)
    fig.tight_layout()
    fig.savefig(Path(figdir) / name)
    plt.close(fig)
    return None


def make_figures(figdir):
    """Every figure of this analysis, drawn into the given directory. Returns missing-data notes."""
    Path(figdir).mkdir(parents=True, exist_ok=True)
    all_data = load_all()
    notes = [fig_frameworks(all_data, figdir), fig_precision(all_data, figdir, "torch"),
             fig_precision(all_data, figdir, "jax"), fig_rate_curve(all_data, figdir)]
    return [n for n in notes if n]


# ---------------- the numbers every table is built from ----------------

def summary():
    """Every number the tables need, computed once."""
    all_data = load_all()
    if not all_data:
        return {}
    rates = next(iter(all_data.values()))["rates"]
    out = {"rates": rates, "chosen_rate_indices": chosen_rates(all_data),
           "final_records": FINAL_RECORDS, "configurations": {}, "comparisons": {}}

    for tag, d in all_data.items():
        r = d["record"]
        secs = r["sec_per_iteration"]
        copies = r["n_copies"]
        out["configurations"][tag] = {
            "label": LABEL[tag], "n_copies": copies,
            "iterations": r["iterations"], "steps_per_copy": r["steps_per_copy"],
            "train_seconds": r["train_seconds"], "sec_per_iteration": secs,
            "million_steps_per_second": copies * STEPS_PER_COPY_PER_ITER / secs / 1e6,
            "thousand_steps_per_second_per_copy": STEPS_PER_COPY_PER_ITER / secs / 1e3,
            "hours_per_million_steps_per_copy": 1e6 / (3600 * STEPS_PER_COPY_PER_ITER / secs),
            "peak_vram_mb": r["peak_vram_mb"],
            "precision_probe": r["precision_probe"],
            "per_rate": [dict(zip(("mean", "lo", "hi"), mean_ci(final_scores(d, i))),
                              rate=rates[i],
                              median=float(np.median(final_scores(d, i))),
                              q1=float(np.percentile(final_scores(d, i), 25)),
                              q3=float(np.percentile(final_scores(d, i), 75)),
                              coverage_mean=float(final_coverage(d, i).mean()),
                              solved_fraction=float((final_scores(d, i) > 0).mean()))
                         for i in range(len(rates))],
        }

    # the three comparisons the campaign was built to make
    pairs = [("frameworks_reduced", "torch_reduced", "jax_reduced"),
             ("precision_torch", "torch_reduced", "torch_exact"),
             ("precision_jax", "jax_reduced", "jax_exact"),
             ("frameworks_exact", "torch_exact", "jax_exact")]
    for name, ta, tb in pairs:
        if ta not in all_data or tb not in all_data:
            continue
        out["comparisons"][name] = {
            "a": LABEL[ta], "b": LABEL[tb],
            "per_rate": [dict(compare(final_scores(all_data[ta], i),
                                      final_scores(all_data[tb], i)), rate=rates[i])
                         for i in range(len(rates))],
            "pooled": pooled_over_rates(
                [final_scores(all_data[ta], i) for i in range(len(rates))],
                [final_scores(all_data[tb], i) for i in range(len(rates))]),
        }
    return out


# ---------------- the run folder's own written analysis ----------------

def _fmt(x, digits=3):
    """A number for a table cell."""
    return f"{x:.{digits}f}"


def wrap_label(label):
    """A configuration name folded at its comma, so a table column stays narrow enough to print."""
    # before: "PyTorch, exact single precision"; after: "PyTorch<br>exact single precision"
    return label.replace(", ", "<br>")


def comparison_table(cmp):
    """One difference table: per learning rate, the two means and whether they can be told apart."""
    head = (f"| learning<br>rate | A:<br>{wrap_label(cmp['a'])} | B:<br>{wrap_label(cmp['b'])} | "
            "difference<br>A − B | 95% interval<br>for the difference | rank<br>test p | "
            "P(A copy<br>above B copy) |\n"
            "|---|---|---|---|---|---|---|\n")
    rows = []
    for r in cmp["per_rate"]:
        rows.append(f"| {r['rate']:g} | {_fmt(r['mean_a'])} | {_fmt(r['mean_b'])} | "
                    f"{_fmt(r['difference'])} | [{_fmt(r['difference_lo'])}, "
                    f"{_fmt(r['difference_hi'])}] | {r['rank_test_p']:.3g} | "
                    f"{_fmt(r['probability_a_above_b'])} |")
    p = cmp["pooled"]
    rows.append(f"| all rates, blocked | {_fmt(p['mean_a'])} | {_fmt(p['mean_b'])} | "
                f"{_fmt(p['difference'])} | [{_fmt(p['difference_lo'])}, "
                f"{_fmt(p['difference_hi'])}] | {p['rank_test_p']:.3g} | "
                f"{_fmt(p['probability_a_above_b'])} |")
    return head + "\n".join(rows) + "\n"


def throughput_table(s):
    """Wall-clock cost of the four runs, in the project's throughput-table columns."""
    head = ("| configuration | copies | seconds per<br>iteration | total million<br>steps per second | "
            "thousand steps per<br>second per copy | hours per million<br>steps per copy | "
            "peak memory<br>(MB) | whole run<br>(hours) |\n|---|---|---|---|---|---|---|---|\n")
    rows = []
    for tag in TAGS:
        c = s["configurations"].get(tag)
        if not c:
            continue
        rows.append(f"| {wrap_label(c['label'])} | {c['n_copies']:,} | "
                    f"{c['sec_per_iteration']:.3f} | "
                    f"{c['million_steps_per_second']:.2f} | "
                    f"{c['thousand_steps_per_second_per_copy']:.2f} | "
                    f"{c['hours_per_million_steps_per_copy']:.2f} | "
                    f"{c['peak_vram_mb']:,.0f} | {c['train_seconds'] / 3600:.2f} |")
    return head + "\n".join(rows) + "\n"


def per_rate_table(s):
    """What each configuration reached at each learning rate."""
    rates = s["rates"]
    head = "| configuration | " + " | ".join(f"{r:g}" for r in rates) + " |\n"
    head += "|---" * (len(rates) + 1) + "|\n"
    rows = []
    for tag in TAGS:
        c = s["configurations"].get(tag)
        if not c:
            continue
        rows.append("| " + wrap_label(c["label"]) + " | "
                    + " | ".join(_fmt(x["mean"], 2) for x in c["per_rate"]) + " |")
    return head + "\n".join(rows) + "\n"


def main():
    """Write the run folder's analysis: figures, analysis.md, and the summary the report reads."""
    notes = make_figures(RUN / "analysis" / "plots")
    s = summary()
    (RUN / "data" / "summary.json").write_text(json.dumps(s, indent=1))
    if not s:
        print("no finished runs yet:", notes)
        return
    rates = s["rates"]
    picks = [rates[i] for i in s["chosen_rate_indices"]]
    first = next(iter(s["configurations"].values()))
    per_rate_copies = first["n_copies"] // len(rates)
    md = [f"""# Do the two implementations learn the same thing, and does reduced precision change it?

Four runs, each {len(rates)} learning rates x {per_rate_copies:,} independent copies x {first['steps_per_copy']:,} environment
steps per copy: {{PyTorch, JAX}} x {{reduced precision, exact single precision}}. A copy's score is
its mean extrinsic reward per iteration over the last {s['final_records']} recorded iterations, which is the
number of steps out of {STEPS_PER_COPY_PER_ITER} it spends inside the goal radius. Parity between the two
implementations was checked first and is written down in `../parity_check.md`.

## What each configuration reached, per learning rate

Mean over the {per_rate_copies:,} copies of a rate group.

{per_rate_table(s)}
The three rates carried through the figures are {', '.join(f'{r:g}' for r in picks)} — the best of the
eight and its two neighbours, chosen from all four configurations pooled.

## PyTorch against JAX, both at reduced precision

{comparison_table(s['comparisons']['frameworks_reduced']) if 'frameworks_reduced' in s['comparisons'] else '*pending*'}
![PyTorch against JAX](plots/learning_outcome_frameworks.png)

## Reduced against exact single precision, inside PyTorch

{comparison_table(s['comparisons']['precision_torch']) if 'precision_torch' in s['comparisons'] else '*pending*'}
![PyTorch precision](plots/learning_outcome_precision_torch.png)

## Reduced against exact single precision, inside JAX

{comparison_table(s['comparisons']['precision_jax']) if 'precision_jax' in s['comparisons'] else '*pending*'}
![JAX precision](plots/learning_outcome_precision_jax.png)

## Every rate, and the seed distribution at the best one

![every rate](plots/learning_outcome_by_rate.png)

## What the four runs cost

{throughput_table(s)}
## The precision each run actually used

| configuration | declared setting | largest relative error of a float32 matrix product against float64 |
|---|---|---|
""" ]
    for tag in TAGS:
        c = s["configurations"].get(tag)
        if c:
            p = c["precision_probe"]
            md.append(f"| {wrap_label(c['label'])} | {p['declared_setting']} | "
                      f"{p['relative_error_against_float64']:.2e} |")
    md.append("")
    (RUN / "analysis" / "analysis.md").write_text("\n".join(md))
    print("wrote analysis.md, plots and summary.json")
    for n in notes:
        print("PENDING:", n)


if __name__ == "__main__":
    main()
