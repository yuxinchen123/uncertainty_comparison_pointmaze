"""The PointMaze fresh-seed comparison: the method as it stands against CleanRL PPO-RND on seeds 30000 to 31023.

Writes `fresh_seeds/pointmaze_fresh_seeds.md` (and the numbers in `fresh_seeds/pointmaze_fresh_seeds.json`) of
the auto research folder. Every difference is paired by seed and gets a 95% interval from 10,000 resamples of the
seeds. Uses numpy: run it through Slurm (`fresh_seeds/code/pointmaze_fresh_seeds_job.sh`), never on a login node.

    python pointmaze_fresh_seeds.py
"""
import glob
import hashlib
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

HERE = Path(__file__).resolve().parent
FOLDER = HERE.parent.parent
RUNS = FOLDER.parent
RND_FOLDER = Path(glob.glob(str(RUNS / "2026-11-12-02-24_*"))[0])
CASES = [("pointmaze-medium", "arrival-plus-one", "Medium, reward 1 on arrival"),
         ("pointmaze-medium", "step-cost", "Medium, step cost"),
         ("pointmaze-large", "arrival-plus-one", "Large, reward 1 on arrival"),
         ("pointmaze-large", "step-cost", "Large, step cost")]
# The method's arm per case for the whole-curve mean: the divisor held from each policy's first arrival plus the
# fast last layer at the step with the highest whole-curve mean on the first validation block among 0, 0.01, 0.03 and
# 0.1 (the rule fixed before the second blocks were read; `development_document/code/beat_rnd_tables.py`, CASES)
METHOD_ARM = {("pointmaze-medium", "arrival-plus-one"): "held-from-first-arrival",
              ("pointmaze-medium", "step-cost"): "fast-step-0-point-01-with-hold",
              ("pointmaze-large", "arrival-plus-one"): "fast-step-0-point-03-with-hold",
              ("pointmaze-large", "step-cost"): "fast-step-0-point-03-with-hold"}
# RND's choice for the first-goal time on the first validation block, per case (the standing page's discovery check);
# the other configurations RND has on the fresh seeds for that measure are the baseline sweep's choices
RND_VALIDATION_CHOICE_M = {("pointmaze-medium", "arrival-plus-one"): "learning-rate-0.02_entropy-coefficient-0.1_intrinsic-weight-4.0",
                           ("pointmaze-medium", "step-cost"): "learning-rate-0.01_entropy-coefficient-0.03_intrinsic-weight-4.0",
                           ("pointmaze-large", "arrival-plus-one"): "learning-rate-0.006_entropy-coefficient-0.01_intrinsic-weight-64.0",
                           ("pointmaze-large", "step-cost"): "learning-rate-0.006_entropy-coefficient-0.01_intrinsic-weight-8.0"}
RESAMPLES = 10_000
WINDOWS = 256


def per_seed(paths):
    """seed -> (W, L, M in thousand steps) over the given unit files; every unit must have all its windows."""
    # before: unit files of 256 window records with per-policy sums, e.g. return_sum_per_copy [256 numbers]
    # after:  {30000: (0.9312, 0.9990, 18.4), 30001: (...), ...}, one entry per policy (= per seed)
    out = {}
    for path in paths:
        # read the last start record of the unit and the windows written after it
        start, windows = None, []
        for line in open(path):
            if '"unit_start"' in line:
                start, windows = json.loads(line), []
            elif '"return_sum_per_copy"' in line:
                windows.append(json.loads(line))
        if len(windows) != WINDOWS:
            raise ValueError(f"{path} has {len(windows)} of {WINDOWS} windows")
        steps_per_window = start["rollout_steps"] * start["envs_per_copy"] * start["window_iterations"]
        tail = WINDOWS // 10
        for c, seed in enumerate(start["copy_seed_index"]):
            # the mean return of the window's finished episodes, carried through windows with none
            last, series = 0.0, []
            for w in windows:
                if w["episodes_per_copy"][c] > 0:
                    last = w["return_sum_per_copy"][c] / w["episodes_per_copy"][c]
                series.append(last)
            # the first-goal step, capped at the budget for a policy that never arrived
            first = windows[-1]["first_goal_step_per_copy"][c]
            capped = first if first is not None and first >= 0 else steps_per_window * WINDOWS
            out[seed] = (float(np.mean(series)), float(np.mean(series[-tail:])), capped / 1e3)
    return out


def units(folder, pattern):
    """The unit files of one configuration: every seed range of the name pattern, at least one."""
    paths = sorted(glob.glob(str(folder / "data" / pattern)))
    if not paths:
        raise FileNotFoundError(f"no unit matches {pattern} in {folder.name}")
    return paths


def configurations(folder, pattern):
    """One name per configuration matching the pattern: the unit name without its seed range."""
    # before: [..._intrinsic-weight-2.0_clip-coefficient-0.1_seeds-30000-to-30255.jsonl, ...seeds-30256-...]
    # after:  [..._intrinsic-weight-2.0_clip-coefficient-0.1]
    return sorted({Path(p).name.rsplit("_seeds-", 1)[0] for p in glob.glob(str(folder / "data" / pattern))})


def interval(method, rnd, k, label):
    """Paired mean difference (method minus RND) of measure k with its 95% resampling interval and seed count."""
    # before: two {seed: (W, L, M)} maps over the same 1,024 seeds; after: (+0.0065, +0.0030, +0.0101, 1024)
    seeds = sorted(set(method) & set(rnd))
    d = np.array([method[s][k] - rnd[s][k] for s in seeds])
    # one random stream per comparison and measure, keyed by its name (rng-seeding rule)
    key = int(hashlib.sha256(f"pointmaze-fresh-seeds::{label}::{k}".encode()).hexdigest(), 16) & 0xFFFFFFFF
    rng = np.random.default_rng(key)
    means = d[rng.integers(0, len(d), size=(RESAMPLES, len(d)))].mean(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return float(d.mean()), float(low), float(high), len(d)


def verdict_w(w, level):
    """Whole-curve mean: ahead when its interval is above zero and the final level's is not below zero."""
    if w[1] > 0 and level[2] >= 0:
        return "ahead"
    if w[2] < 0:
        return "behind"
    return "level"


def verdict_m(m):
    """Mean capped first-goal time: ahead when the interval of the difference is below zero."""
    return "ahead" if m[2] < 0 else "behind" if m[1] > 0 else "level"


def combined(words):
    """One verdict over RND's configurations: ahead only if ahead of each, behind if behind any, else level."""
    return "ahead" if all(w == "ahead" for w in words) else "behind" if "behind" in words else "level"


def words(name):
    """The configuration of a unit name in words: learning rate, entropy coefficient, intrinsic weight."""
    # before: confirmation_pointmaze-medium_reward-step-cost_cleanrl-ppo-rnd_learning-rate-0.006_entropy-coefficient-0.01_intrinsic-weight-0.001953125_clip-coefficient-0.1
    # after:  learning rate 0.006, entropy 0.01, weight 1/512
    parts = dict(p.split("-", 1) for p in name.replace("learning-rate", "lr").replace("entropy-coefficient", "ent")
                 .replace("intrinsic-weight", "w").split("_") if p.split("-", 1)[0] in ("lr", "ent", "w"))
    text = f"learning rate {parts['lr']}, entropy {parts['ent']}"
    if "w" in parts:
        weight = float(parts["w"])
        text += f", weight {weight:g}" if weight >= 1 else f", weight 1/{round(1 / weight)}"
    return text


def fmt(x, digits):
    """A difference with its interval, e.g. +0.0065 [+0.0030, +0.0101]."""
    return f"{x[0]:+.{digits}f} [{x[1]:+.{digits}f}, {x[2]:+.{digits}f}]"


def main():
    """Read every case, compute the paired differences and verdicts, and write the page and the numbers."""
    rows_w, rows_m, rows_ref, rows_other, record = [], [], [], [], {}
    for env, setting, name in CASES:
        prefix = f"{env}_reward-{setting}"
        # the method for the whole-curve mean and RND's fresh-seed runs for it (the baseline folder)
        method_name = configurations(FOLDER, f"fresh-seeds_{prefix}_{METHOD_ARM[(env, setting)]}_*.jsonl")
        assert len(method_name) == 1, method_name
        # arms run on these seeds that are not the method (Medium with reward 1 on arrival: step 0.01, run there by
        # mistake on 2026-11-15 before the hold alone); shown for the record, never part of a verdict
        for other in configurations(FOLDER, f"fresh-seeds_{prefix}_*-with-hold_*.jsonl") + \
                configurations(FOLDER, f"fresh-seeds_{prefix}_held-from-first-arrival_*.jsonl"):
            if other != method_name[0]:
                rows_other.append((name, other.split(f"{prefix}_")[1].split("_learning-rate")[0], other))
        method_w = per_seed(units(FOLDER, method_name[0] + "_seeds-*.jsonl"))
        case_w = []
        for rnd_name in configurations(RND_FOLDER, f"confirmation_{prefix}_cleanrl-ppo-rnd_learning-rate-*.jsonl"):
            rnd = per_seed(units(RND_FOLDER, rnd_name + "_seeds-*.jsonl"))
            w, level = interval(method_w, rnd, 0, rnd_name), interval(method_w, rnd, 1, rnd_name)
            case_w.append(verdict_w(w, level))
            rows_w.append((name, words(rnd_name), np.mean([v[0] for v in method_w.values()]),
                           np.mean([v[0] for v in rnd.values()]), w, level, case_w[-1], w[3]))
        # the reference rows of the record: RND at CleanRL's defaults and PPO without a bonus
        for ref in configurations(RND_FOLDER, f"confirmation_{prefix}_*.jsonl"):
            if "cleanrl-defaults" in ref or "without-a-bonus" in ref:
                rnd = per_seed(units(RND_FOLDER, ref + "_seeds-*.jsonl"))
                rows_ref.append((name, "RND at CleanRL's defaults" if "defaults" in ref else "PPO without a bonus",
                                 np.mean([v[0] for v in rnd.values()]), interval(method_w, rnd, 0, ref)))
        # the method for the first-goal time and RND at each of its configurations for it (both folders)
        method_m_name = configurations(FOLDER, f"fresh-seeds_{prefix}_fast-layer-step-0-point-1_*.jsonl")
        assert len(method_m_name) == 1, method_m_name
        method_m = per_seed(units(FOLDER, method_m_name[0] + "_seeds-*.jsonl"))
        case_m = []
        # label each RND row by the rule that chose it, wherever it was run
        names = [(RND_FOLDER, n) for n in configurations(RND_FOLDER, f"confirmation-for-discovery_{prefix}_cleanrl-ppo-rnd_*.jsonl")]
        names += [(FOLDER, n) for n in configurations(FOLDER, f"fresh-seeds_{prefix}_control-cleanrl-ppo-rnd_*.jsonl")]
        sources = [(f, n, "its validation-block choice" if RND_VALIDATION_CHOICE_M[(env, setting)] in n
                    else "the baseline sweep's choice") for f, n in names]
        for folder, rnd_name, why in sources:
            rnd = per_seed(units(folder, rnd_name + "_seeds-*.jsonl"))
            m = interval(method_m, rnd, 2, rnd_name)
            case_m.append(verdict_m(m))
            rows_m.append((name, f"{words(rnd_name)} ({why})", np.mean([v[2] for v in method_m.values()]),
                           np.mean([v[2] for v in rnd.values()]), m, case_m[-1], m[3]))
        record[name] = {"whole_curve_mean": combined(case_w), "first_goal_time": combined(case_m),
                        "method_for_whole_curve_mean": method_name[0], "method_for_first_goal_time": method_m_name[0]}
    # the page
    now = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d %H:%M PT")
    out = ["# PointMaze fresh seeds: the method against CleanRL PPO-RND", "",
           f"Generated by `fresh_seeds/code/pointmaze_fresh_seeds.py` at {now} (times are Pacific, converted from "
           "machines running Eastern). Seeds 30000 to 31023, 1,024 per configuration, the same seeds for the method "
           "and for RND; every difference is method minus RND, paired by seed, with its 95% interval from "
           f"{RESAMPLES:,} resamples of the seeds.", "",
           "Define $K$ as the number of seeds, $B$ as the budget in environment steps per policy, $R_i(s)$ as the "
           "mean extrinsic return of the training episodes of policy $i$ that end in the logging window containing "
           "step $s$ (a window with none takes the last value), and $T_i$ as the training step of policy $i$'s first "
           "arrival at the goal ($T_i = B$ if it never arrives).", "",
           "- **Whole-curve mean:** $W = \\dfrac{1}{K}\\sum_{i=1}^{K} \\dfrac{1}{B}\\sum_{s \\le B} R_i(s)$, the area "
           "under the training curve divided by the budget; larger is better.",
           "- **Final level:** $L$, the same mean over the last 25 of the 256 logging windows (9.8% of the budget; "
           "the development document's tables use exactly the last 10%, which moves $L$ by at most 0.0001 here).",
           "- **Mean capped first-goal time:** $M = \\dfrac{1}{K}\\sum_{i=1}^{K} \\min(T_i, B)$, in thousand steps; "
           "smaller is better.",
           "- **Verdicts:** for $W$, ahead when the interval of the difference is above zero and the interval of the "
           "difference in $L$ is not below zero; for $M$, ahead when the interval is below zero. A case is ahead "
           "only when it is ahead of RND at each configuration RND has on these seeds.", "",
           "The method: RND with the divisor held from each policy's first arrival and the fast last layer at the "
           "step each case fixed on its first validation block (0 on Medium with reward 1 on arrival, that is the "
           "hold alone; 0.01 on Medium with the step cost; 0.03 on both Large cases), at that block's "
           "configuration; for $M$, the fast last layer at step 0.1 at its own configuration for discovery.", "",
           "## Verdicts", "", "| case | whole-curve mean $W$ | first-goal time $M$ |", "|---|---|---|"]
    out += [f"| {name} | {v['whole_curve_mean']} | {v['first_goal_time']} |" for name, v in record.items()]
    out += ["", "## Whole-curve mean", "",
            "| case | RND's configuration | $W$, method | $W$, RND | difference in $W$ [95%] | difference in $L$ [95%] | verdict | seeds |",
            "|---|---|---:|---:|---|---|---|---:|"]
    out += [f"| {n} | {c} | {mw:.4f} | {rw:.4f} | {fmt(w, 4)} | {fmt(l, 4)} | {v} | {k} |"
            for n, c, mw, rw, w, l, v, k in rows_w]
    out += ["", "## Mean capped first-goal time", "",
            "| case | RND's configuration | $M$, method | $M$, RND | difference in $M$ [95%] | verdict | seeds |",
            "|---|---|---:|---:|---|---|---:|"]
    out += [f"| {n} | {c} | {mm:.1f} | {rm:.1f} | {fmt(m, 1)} | {v} | {k} |" for n, c, mm, rm, m, v, k in rows_m]
    out += ["", "## Reference rows of the record (never used for a verdict)", "",
            "| case | row | $W$ of the row | the method's $W$ minus the row's [95%] |", "|---|---|---:|---|"]
    out += [f"| {n} | {r} | {w:.4f} | {fmt(d, 4)} |" for n, r, w, d in rows_ref]
    # arms that are not the method, against the same RND runs (first RND configuration of the case)
    out += ["", "## Arms on these seeds that are not the method (never used for a verdict)", "",
            "| case | arm | $W$ of the arm | the arm's $W$ minus RND's [95%] | RND's configuration |", "|---|---|---:|---|---|"]
    for name, arm, other in rows_other:
        env, setting = next((e, s) for e, s, n in CASES if n == name)
        prefix = f"{env}_reward-{setting}"
        arm_w = per_seed(units(FOLDER, other + "_seeds-*.jsonl"))
        rnd_name = configurations(RND_FOLDER, f"confirmation_{prefix}_cleanrl-ppo-rnd_learning-rate-*.jsonl")[0]
        rnd = per_seed(units(RND_FOLDER, rnd_name + "_seeds-*.jsonl"))
        out.append(f"| {name} | {arm.replace('-', ' ')} | {np.mean([v[0] for v in arm_w.values()]):.4f} | "
                   f"{fmt(interval(arm_w, rnd, 0, other), 4)} | {words(rnd_name)} |")
    (FOLDER / "fresh_seeds" / "pointmaze_fresh_seeds.md").write_text("\n".join(out) + "\n")
    (FOLDER / "fresh_seeds" / "pointmaze_fresh_seeds.json").write_text(json.dumps(record, indent=1) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    main()
