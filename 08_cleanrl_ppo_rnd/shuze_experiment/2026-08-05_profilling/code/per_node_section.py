"""The per-node results section: one row per node class, at the configuration the run really uses.

Two independent sources, kept separate on purpose:

1. **Isolated profiling** — one training process alone on a node, 8 cpus, the run's exact flags.
   Clean, comparable, no contention. Files: `data/per_node/*.json`.
2. **The live 30-seed run** — the same configuration under real conditions: several runs per node,
   other users' jobs on the same hardware. Read from the run folder's records.

They answer different questions. The isolated number says what a card can do; the live number says
what it is doing. Where they disagree, the gap is contention, and that is worth seeing.

The architecture and compute capability come from the cards themselves, read by nvidia-smi during
each measurement, so this table depends on no external catalog — one of which was already wrong
about titanx03.
"""

import glob
import json
import os
import statistics

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
# Compute capability to architecture family. Read off the cards themselves via nvidia-smi, so this
# document does not depend on any external catalog — one of which was already wrong about titanx03.
# The live run whose records give the "under real conditions" column.
RUN_DIR = os.path.join(
    HERE, "..", "..", "..", "train_runs",
    "2026-08-05-17-45_cleanrl_train_run_1_ppo-rnd_montezuma-revenge-v5__envpool-autoreset-fixed__"
    "128env-128step-2e9step__int-coef1-ext-coef2-updateproportion0.25__seed1-30__"
    "checkpoint-8h-resumable__gpu-gnolim")

BATCH = 128 * 128
TOTAL_STEPS = 2_000_000_000

ARCH_BY_CC = {
    5.2: ("Maxwell", 2015), 6.0: ("Pascal", 2016), 6.1: ("Pascal", 2016),
    7.0: ("Volta", 2017), 7.5: ("Turing", 2018), 8.0: ("Ampere", 2020),
    8.6: ("Ampere", 2020), 8.9: ("Ada", 2022), 9.0: ("Hopper", 2022),
    10.0: ("Blackwell", 2024), 12.0: ("Blackwell", 2025),
}


def parse_gpu_smi(smi):
    """Pull (model, memory MiB, compute capability) out of an nvidia-smi csv line.

    before: "NVIDIA A40, 46068 MiB, 8.6"
    after:  ("NVIDIA A40", 46068.0, 8.6)
    """
    parts = [p.strip() for p in smi.split(",")]
    if len(parts) < 3:
        return smi, None, None
    mem = float(parts[1].split()[0]) if parts[1].split()[0].replace(".", "").isdigit() else None
    try:
        cc = float(parts[2])
    except ValueError:
        cc = None
    return parts[0], mem, cc


def node_catalog():
    """Build the node description from the measurements themselves, not an external catalog."""
    out = {}
    # The canary jobs read the card directly and are the most trustworthy source.
    for f in sorted(glob.glob(os.path.join(DATA, "canary", "*.json"))):
        d = json.load(open(f))
        if d.get("gpu_name"):
            cc = float(d["compute_capability"])
            fam, year = ARCH_BY_CC.get(cc, ("unknown", "?"))
            out[d["node"]] = {"gpu": d["gpu_name"], "cc": cc, "arch": fam, "year": year}
    # Every profiling job also records an nvidia-smi line, which covers the nodes with no canary.
    for pattern in ("per_node/*.json", "highend/*.json", "profile_*.json"):
        for f in sorted(glob.glob(os.path.join(DATA, pattern))):
            d = json.load(open(f))
            if d["node"] in out or not d.get("gpu_smi"):
                continue
            model, _, cc = parse_gpu_smi(d["gpu_smi"])
            fam, year = ARCH_BY_CC.get(cc, ("unknown", "?")) if cc else ("unknown", "?")
            out[d["node"]] = {"gpu": model, "cc": cc, "arch": fam, "year": year}
    # Finally the live run's records, which name the GPU for every node hosting the run — those
    # nodes are deliberately absent from the profiling wave, because profiling them would compete
    # with the training they are running.
    for f in sorted(glob.glob(os.path.join(RUN_DIR, "data", "local", "*.json"))):
        try:
            d = json.load(open(f))
        except (json.JSONDecodeError, OSError):
            continue
        node = d.get("hostname")
        if node and node not in out and d.get("gpu_name"):
            out[node] = {"gpu": d["gpu_name"], "cc": None, "arch": "unknown", "year": "?"}
    return out


def isolated_results():
    """Per-node throughput from the isolated profiling wave, plus any failures."""
    rows = {}
    for f in sorted(glob.glob(os.path.join(DATA, "per_node", "*.json"))):
        d = json.load(open(f))
        ok = [v for v in d["variants"] if "steps_per_second" in v]
        if ok:
            v = ok[0]
            rows[d["node"]] = {
                "steps_per_second": v["steps_per_second"],
                "rollout": v["rollout_seconds_mean"], "update": v["update_seconds_mean"],
                "iteration": v["iteration_seconds_mean"], "gpu_mem_mb": v["gpu_memory_peak_mb"],
                "cpus": d["cpus_per_task"], "gpu_name": v.get("gpu_name", ""),
            }
        else:
            err = d["variants"][0].get("error", "no result") if d["variants"] else "no result"
            rows[d["node"]] = {"error": err, "cpus": d["cpus_per_task"]}
    return rows


def live_results():
    """Per-node throughput from the live 30-seed run's own records."""
    by_node = {}
    for f in sorted(glob.glob(os.path.join(RUN_DIR, "data", "local", "*.json"))):
        try:
            d = json.load(open(f))
        except (json.JSONDecodeError, OSError):
            continue          # a record being rewritten right now; it will be there next time
        eh = d.get("eval_history") or []
        rows = [r["charts/iteration_seconds"] for r in eh[-5:] if "charts/iteration_seconds" in r]
        if not rows:
            continue
        by_node.setdefault(d.get("hostname", "?"), []).append({
            "rate": BATCH / statistics.fmean(rows),
            "step": eh[-1]["step"], "seed": d["a_seed"], "threads": d.get("env_threads"),
        })
    out = {}
    for node, rs in by_node.items():
        out[node] = {
            "runs": len(rs),
            "rate_mean": statistics.fmean(r["rate"] for r in rs),
            "rate_min": min(r["rate"] for r in rs),
            "step_mean": statistics.fmean(r["step"] for r in rs),
            "cpus_per_run": rs[0]["threads"],
        }
    return out


def render(table):
    """Build the per-node section. `table` is the shared markdown-table renderer."""
    cat, iso, live = node_catalog(), isolated_results(), live_results()
    L = [
        "## Results per node",
        "",
        "One row per node class, all at the configuration the 30-seed run uses: the auto-reset fix,",
        "every same-numerics option, a constant minibatch shape, **8 cpus per run and one run per",
        "GPU**. This is the table to read when deciding where to put work.",
        "",
        "`days per seed` is the time one seed needs for its 2 billion steps at that rate, ignoring",
        "resumes. `GPU-bound share` is the update phase over the whole iteration — it is the part a",
        "faster card can shorten, and it bounds what a card upgrade can buy.",
        "",
    ]

    if iso:
        rows = []
        for node in sorted(iso):
            c, r = cat.get(node, {}), iso[node]
            arch = f"{c.get('arch', '?')} ({c.get('year', '?')})"
            if "error" in r:
                rows.append([f"`{node}`", c.get("gpu", "?"), arch, str(c.get("cc", "?")),
                             "**cannot run**", "—", "—", "—", "—"])
                continue
            gpu_share = r["update"] / r["iteration"] * 100
            days = TOTAL_STEPS / r["steps_per_second"] / 86400
            rows.append([f"`{node}`", c.get("gpu", "?"), arch, str(c.get("cc", "?")),
                         f"{r['steps_per_second']:,.0f}", f"{r['rollout']:.2f} / {r['update']:.2f}",
                         f"{gpu_share:.0f}%", f"{r['gpu_mem_mb']:,.0f}", f"{days:.1f}"])
        # Fastest first: that is the order a placement decision is made in.
        rows.sort(key=lambda x: -float(x[4].replace(",", "")) if x[4][0].isdigit() else 1)
        L += ["### Measured alone on the node, 8 cpus, one run", "",
              table(["node", "GPU", "architecture", "compute capability", "steps/s",
                     "rollout / update (s)", "GPU-bound share", "GPU memory (MB)",
                     "days per seed"], rows), ""]
        failed = [n for n in iso if "error" in iso[n]]
        if failed:
            L += [f"`{'`, `'.join(failed)}` could not run this stack at all — see the compute-capability",
                  "section above for why.", ""]
    else:
        L += ["The per-node wave had not produced results when this document was last generated.", ""]

    if live:
        rows = []
        for node in sorted(live, key=lambda n: -live[n]["rate_mean"]):
            c, r = cat.get(node, {}), live[node]
            iso_rate = iso.get(node, {}).get("steps_per_second")
            gap = f"{r['rate_mean'] / iso_rate * 100:.0f}%" if iso_rate else "N/A"
            days = TOTAL_STEPS / r["rate_mean"] / 86400
            rows.append([f"`{node}`", c.get("gpu", "?"), str(r["runs"]), str(r["cpus_per_run"]),
                         f"{r['rate_mean']:,.0f}", f"{r['rate_min']:,.0f}", gap,
                         f"{r['step_mean']:,.0f}", f"{days:.1f}"])
        L += ["### The same configuration in the live 30-seed run", "",
              "Several runs share each node here, and other users' jobs share the hardware, so these",
              "are the rates the campaign actually gets. `vs alone` compares to the isolated",
              "measurement above; it reads N/A for every node hosting the run, because those nodes",
              "were deliberately left out of the profiling wave rather than have a benchmark compete",
              "with the training on the same hardware.",
              "",
              table(["node", "GPU", "runs on it", "cpus per run", "mean steps/s", "slowest run",
                     "vs alone", "mean step now", "days per seed"], rows), "",
              "The campaign finishes when the **last** seed finishes, so the largest `days per seed`",
              "is the completion estimate, not the average.", ""]
    return L
