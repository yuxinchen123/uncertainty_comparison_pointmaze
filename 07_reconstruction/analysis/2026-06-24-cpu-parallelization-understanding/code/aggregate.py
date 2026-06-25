"""
Aggregate the per-node CPU-profiling sweep into CSV tables + plots.
Robust to partial data; safe to re-run while jobs are still finishing.

Reads:  <OUT>/logs/<node>/microbench_dev-*_threads-*.json
        <OUT>/logs/<node>/e2e_<algo>_dev-*_threads-*.json
        <OUT>/logs/<node>/crosscheck_real04_threads-*.log   (/usr/bin/time -v)
Writes: <OUT>/results/*.csv, <OUT>/results/summary.json, <OUT>/plots/*.png
"""
import csv
import glob
import json
import os
import re
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.dirname(HERE)
LOGS = os.path.join(OUT, "logs")
RES = os.path.join(OUT, "results")
PLOTS = os.path.join(OUT, "plots")
os.makedirs(RES, exist_ok=True)
os.makedirs(PLOTS, exist_ok=True)

THREADS = [1, 4, 8, 16]


def load_microbench():
    rows = []
    for f in glob.glob(os.path.join(LOGS, "*", "microbench_dev-*_threads-*.json")):
        node = os.path.basename(os.path.dirname(f))
        try:
            data = json.load(open(f))
        except Exception:
            continue
        for r in data:
            if "error" in r:
                continue
            rows.append({"node": node, "device": r["device"], "op": r["op"],
                         "backend": r.get("backend", ""), "n_threads": r["n_threads"],
                         "ops_per_sec": r["ops_per_sec"], "effective_cores": r["effective_cores"]})
    return rows


def load_e2e():
    rows = []
    for f in glob.glob(os.path.join(LOGS, "*", "e2e_*_dev-*_threads-*.json")):
        node = os.path.basename(os.path.dirname(f))
        try:
            r = json.load(open(f))
        except Exception:
            continue
        if "steady_steps_per_sec" not in r:
            continue
        cw = r.get("cpu_window", {}) or {}
        ta = r.get("thread_activity", {}) or {}
        rows.append({
            "node": node, "algorithm": r["algorithm"], "device": r["device"],
            "n_threads": r["n_threads"],
            "steady_steps_per_sec": r["steady_steps_per_sec"],
            "steady_effective_cores": r["steady_effective_cores"],
            "steady_ms_per_step": r.get("steady_ms_per_step"),
            "mean_cpu_percent": cw.get("mean_cpu_percent"),
            "max_num_threads": cw.get("max_num_threads"),
            "n_threads_with_cpu_time": ta.get("n_threads_with_cpu_time"),
        })
    return rows


def load_crosscheck():
    rows = []
    for f in glob.glob(os.path.join(LOGS, "*", "crosscheck_real04_threads-*.log")):
        node = os.path.basename(os.path.dirname(f))
        if node == "cheetah01":  # GPU node's CPU baseline, excluded for consistency with Table B
            continue
        m = re.search(r"threads-(\d+)\.log", f)
        n = int(m.group(1)) if m else None
        txt = open(f, errors="ignore").read()
        def grab(pat):
            mm = re.search(pat, txt)
            return mm.group(1) if mm else None
        elapsed = grab(r"Elapsed \(wall clock\) time.*?:\s*([\d:.]+)")
        pct = grab(r"Percent of CPU this job got:\s*(\d+)%")
        user = grab(r"User time \(seconds\):\s*([\d.]+)")
        system = grab(r"System time \(seconds\):\s*([\d.]+)")
        # parse elapsed m:ss or h:mm:ss
        ev = None
        if elapsed:
            parts = [float(x) for x in elapsed.split(":")]
            ev = parts[0]*60+parts[1] if len(parts) == 2 else parts[0]*3600+parts[1]*60+parts[2]
        eff = None
        if user and system and ev:
            eff = round((float(user)+float(system))/ev, 3)
        rows.append({"node": node, "n_threads": n, "elapsed_s": ev,
                     "percent_cpu": int(pct) if pct else None,
                     "user_s": float(user) if user else None,
                     "system_s": float(system) if system else None,
                     "effective_cores": eff})
    return rows


def write_csv(path, rows, fields):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def summarize(rows, key_fields, value, group_extra=()):
    """Node-paired aggregation: for each (key, node) compute the per-node value at
    each thread count, then the per-node speedup = value(N)/value(1) *within that
    node* (isolates scaling from hardware differences). Aggregate value and
    speedup across nodes (mean/std)."""
    cell = defaultdict(list)   # (keytuple, node, n_threads) -> [values]
    meta = {}
    for r in rows:
        if r.get(value) is None:
            continue
        k = tuple(r[f] for f in key_fields)
        cell[(k, r["node"], r["n_threads"])].append(r[value])
        meta[k] = {e: r.get(e) for e in group_extra}
    cellm = {kk: float(np.mean(v)) for kk, v in cell.items()}  # per-node mean
    keys = sorted(set(k for (k, _, _) in cellm))
    out = []
    for k in keys:
        nodes = sorted(set(nd for (kk, nd, _) in cellm if kk == k))
        for n in THREADS:
            vals, sps = [], []
            for nd in nodes:
                v = cellm.get((k, nd, n))
                b = cellm.get((k, nd, 1))
                if v is None:
                    continue
                vals.append(v)
                if b:
                    sps.append(v / b)
            if not vals:
                continue
            row = {f: kv for f, kv in zip(key_fields, k)}
            row.update(meta[k])
            row["n_threads"] = n
            row[f"mean_{value}"] = round(float(np.mean(vals)), 3)
            row[f"std_{value}"] = round(float(np.std(vals)), 3)
            row["n_nodes"] = len(vals)
            if sps:
                sp = float(np.mean(sps))
                row["speedup_vs1"] = round(sp, 3)
                row["speedup_std"] = round(float(np.std(sps)), 3)
                row["par_efficiency"] = round(sp / n, 3)  # speedup / cores
            out.append(row)
    return out


def plot_lines(summary, key, value_key, title, ylab, fname, ref_y=None):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print("matplotlib unavailable:", e)
        return
    err_key = "speedup_std" if value_key == "speedup_vs1" else value_key.replace("mean_", "std_")
    groups = defaultdict(lambda: {"x": [], "y": [], "e": []})
    for r in summary:
        if r.get(value_key) is None:
            continue
        groups[r[key]]["x"].append(r["n_threads"])
        groups[r[key]]["y"].append(r[value_key])
        groups[r[key]]["e"].append(r.get(err_key, 0) or 0)
    plt.figure(figsize=(8, 5.5))
    for g, d in sorted(groups.items()):
        order = np.argsort(d["x"])
        x = np.array(d["x"])[order]; y = np.array(d["y"])[order]; e = np.array(d["e"])[order]
        plt.errorbar(x, y, yerr=e, marker="o", capsize=3, label=str(g))
    if ref_y == "ideal":
        plt.plot(THREADS, THREADS, "k--", alpha=0.5, label="ideal linear (y=x)")
    plt.xticks(THREADS); plt.xlabel("CPU thread cap (cores allowed)")
    plt.ylabel(ylab); plt.title(title); plt.grid(alpha=0.3)
    plt.legend(fontsize=8, ncol=2); plt.tight_layout()
    plt.savefig(os.path.join(PLOTS, fname), dpi=130)
    plt.close()
    print("wrote plot", fname)


def main():
    mb = load_microbench()
    e2e = load_e2e()
    cc = load_crosscheck()
    nodes_mb = sorted(set(r["node"] for r in mb))
    nodes_e2e = sorted(set(r["node"] for r in e2e))
    print(f"microbench rows={len(mb)} nodes={len(nodes_mb)}: {nodes_mb}")
    print(f"e2e rows={len(e2e)} nodes={len(nodes_e2e)}: {nodes_e2e}")
    print(f"crosscheck rows={len(cc)}")

    write_csv(os.path.join(RES, "microbench_long.csv"), mb,
              ["node", "device", "op", "backend", "n_threads", "ops_per_sec", "effective_cores"])
    write_csv(os.path.join(RES, "e2e_long.csv"), e2e,
              ["node", "algorithm", "device", "n_threads", "steady_steps_per_sec",
               "steady_effective_cores", "steady_ms_per_step", "mean_cpu_percent",
               "max_num_threads", "n_threads_with_cpu_time"])
    if cc:
        write_csv(os.path.join(RES, "crosscheck.csv"), cc,
                  ["node", "n_threads", "elapsed_s", "percent_cpu", "user_s", "system_s", "effective_cores"])

    # exclude cheetah01 (the GPU node's CPU baseline, added for the §3 same-node
    # comparison) so Table A stays the 29 core-op nodes (17 main + 12 mb2_*).
    mb_cpu = [r for r in mb if r["device"] == "cpu" and r["node"] != "cheetah01"]
    mb_speed = summarize(mb_cpu, ["op"], "ops_per_sec", group_extra=("backend",))
    mb_eff = summarize(mb_cpu, ["op"], "effective_cores", group_extra=("backend",))
    # merge speedup into eff table for one combined summary
    speed_idx = {(r["op"], r["n_threads"]): r for r in mb_speed}
    for r in mb_eff:
        s = speed_idx.get((r["op"], r["n_threads"]), {})
        r["mean_ops_per_sec"] = s.get("mean_ops_per_sec")
        r["speedup_vs1"] = s.get("speedup_vs1")
        r["par_efficiency"] = s.get("par_efficiency")
    write_csv(os.path.join(RES, "microbench_summary.csv"), mb_eff,
              ["op", "backend", "n_threads", "mean_ops_per_sec", "speedup_vs1", "speedup_std",
               "par_efficiency", "mean_effective_cores", "std_effective_cores", "n_nodes"])

    # Table B = the 17 main-sweep CPU nodes that ran the full 1/4/8/16 sweep.
    # Supplementary nodes are reported separately: followup_* (with-eval/uncapped),
    # twocheck_* (2-core probe), cheetah01 (the GPU node's CPU baseline, used for
    # the §3 same-node comparison via e2e_long.csv).
    def _supplementary(n):
        return n.startswith("followup_") or n.startswith("twocheck_") or n == "cheetah01"
    e2e_cpu = [r for r in e2e if r["device"] == "cpu" and not _supplementary(r["node"])]
    e2e_speed = summarize(e2e_cpu, ["algorithm"], "steady_steps_per_sec")
    e2e_eff = summarize(e2e_cpu, ["algorithm"], "steady_effective_cores")
    eidx = {(r["algorithm"], r["n_threads"]): r for r in e2e_eff}
    for r in e2e_speed:
        ee = eidx.get((r["algorithm"], r["n_threads"]), {})
        r["mean_effective_cores"] = ee.get("mean_steady_effective_cores")
    write_csv(os.path.join(RES, "e2e_summary.csv"), e2e_speed,
              ["algorithm", "n_threads", "mean_steady_steps_per_sec", "speedup_vs1", "speedup_std",
               "par_efficiency", "std_steady_steps_per_sec", "mean_effective_cores", "n_nodes"])

    # cuda summary (if any)
    e2e_cuda = [r for r in e2e if r["device"] == "cuda"]
    if e2e_cuda:
        cu = summarize(e2e_cuda, ["algorithm"], "steady_steps_per_sec")
        cu_e = summarize(e2e_cuda, ["algorithm"], "steady_effective_cores")
        ci = {(r["algorithm"], r["n_threads"]): r for r in cu_e}
        for r in cu:
            r["mean_effective_cores"] = ci.get((r["algorithm"], r["n_threads"]), {}).get("mean_steady_effective_cores")
        write_csv(os.path.join(RES, "e2e_cuda_summary.csv"), cu,
                  ["algorithm", "n_threads", "mean_steady_steps_per_sec", "speedup_vs1",
                   "mean_effective_cores", "n_nodes"])

    # plots
    if mb_eff:
        plot_lines([r for r in mb_speed], "op", "speedup_vs1",
                   "Microbench: wall-clock speedup vs 1 thread (mean over nodes)",
                   "speedup (ops/s relative to 1 thread)", "microbench_speedup.png", ref_y="ideal")
        plot_lines(mb_eff, "op", "mean_effective_cores",
                   "Microbench: effective cores used (CPU-seconds / wall-second)",
                   "effective cores", "microbench_effcores.png", ref_y="ideal")
    if e2e_speed:
        plot_lines(e2e_speed, "algorithm", "speedup_vs1",
                   "End-to-end training: speedup vs 1 thread (mean over nodes)",
                   "speedup (steps/s relative to 1 thread)", "e2e_speedup.png", ref_y="ideal")
        plot_lines(e2e_speed, "algorithm", "mean_steady_steps_per_sec",
                   "End-to-end training: steady-state steps/sec (mean over nodes)",
                   "steps / sec", "e2e_steps_per_sec.png")
        plot_lines(e2e_eff, "algorithm", "mean_steady_effective_cores",
                   "End-to-end training: effective cores used (mean over nodes)",
                   "effective cores", "e2e_effcores.png", ref_y="ideal")

    summary = {
        "nodes_microbench": nodes_mb, "nodes_e2e": nodes_e2e,
        "n_microbench_rows": len(mb), "n_e2e_rows": len(e2e), "n_crosscheck": len(cc),
    }
    json.dump(summary, open(os.path.join(RES, "summary.json"), "w"), indent=2)
    print("done.")


if __name__ == "__main__":
    main()
