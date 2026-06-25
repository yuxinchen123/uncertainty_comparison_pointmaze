"""
Aggregate the ntasks/cpus-per-task concurrency + memory experiments.
Reads logs/array_<node>/{colocated,isolated}_cpt<N>_proc<P>.json and bw_K<K>_proc<P>.json.
Writes results/array_concurrency.csv, results/array_bandwidth.csv and prints markdown.
"""
import csv, glob, json, os
from collections import defaultdict
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.dirname(HERE)
LOGS = os.path.join(OUT, "logs")
RES = os.path.join(OUT, "results")
os.makedirs(RES, exist_ok=True)
CPTS = [1, 2, 4]


def load_e2e():
    rows = []
    for f in glob.glob(os.path.join(LOGS, "array_*", "*_cpt*_proc*.json")):
        try:
            d = json.load(open(f))
        except Exception:
            continue
        if "steady_steps_per_sec" not in d:
            continue
        base = os.path.basename(f)
        tag = base.split("_cpt")[0]  # colocated | isolated
        cpt = int(base.split("_cpt")[1].split("_")[0])  # from filename, NOT SLURM_CPUS_PER_TASK
        proc = base.split("_proc")[1].split(".")[0]  # taskset worker index from filename
        node = os.path.basename(os.path.dirname(f)).replace("array_", "")
        rows.append({"node": node, "mode": tag,
                     "cpt": cpt,
                     "procid": proc,
                     "steps_per_sec": d["steady_steps_per_sec"],
                     "eff_cores": d.get("steady_effective_cores"),
                     "torch_threads": d.get("torch_threads_effective"),
                     "affinity_ncpus": d.get("affinity_ncpus"),
                     "rss_mb": d.get("peak_rss_mb")})
    return rows


def load_bw():
    rows = []
    for f in glob.glob(os.path.join(LOGS, "array_*", "bw_K*_proc*.json")):
        node = os.path.basename(os.path.dirname(f)).replace("array_", "")
        K = int(os.path.basename(f).split("_K")[1].split("_")[0])
        try:
            data = json.load(open(f))
        except Exception:
            continue
        for r in data:
            if r.get("op") == "mem_triad":
                rows.append({"node": node, "K": K, "ops_per_sec": r["ops_per_sec"],
                             "gb_per_sec": round(r["ops_per_sec"] * 0.192, 2)})
    return rows


def main():
    e = load_e2e()
    bw = load_bw()
    nodes = sorted(set(r["node"] for r in e))
    print(f"e2e rows={len(e)} nodes={len(nodes)}: {nodes}")
    print(f"bandwidth rows={len(bw)}")

    with open(os.path.join(RES, "array_concurrency_long.csv"), "w", newline="") as fo:
        w = csv.DictWriter(fo, fieldnames=["node", "mode", "cpt", "procid", "steps_per_sec",
                                           "eff_cores", "torch_threads", "affinity_ncpus", "rss_mb"])
        w.writeheader(); [w.writerow(r) for r in e]

    # per (node, cpt): isolated steps/s (1 task) and co-located per-task mean (4 tasks)
    iso = defaultdict(list); col = defaultdict(list); rss = defaultdict(list)
    for r in e:
        key = (r["node"], r["cpt"])
        if r["mode"] == "isolated":
            iso[key].append(r["steps_per_sec"])
        else:
            col[key].append(r["steps_per_sec"]); rss[key].append(r["rss_mb"] or 0)

    summ = []
    for cpt in CPTS:
        iso_v, col_v, pen, tot, rssv = [], [], [], [], []
        for nd in nodes:
            k = (nd, cpt)
            if k in iso and k in col:
                i = np.mean(iso[k]); c = np.mean(col[k])
                iso_v.append(i); col_v.append(c); pen.append(c / i); tot.append(c * 4)
                rssv.append(np.mean(rss[k]))
        if not col_v:
            continue
        summ.append({"cpt": cpt,
                     "isolated_steps_s": round(np.mean(iso_v), 1) if iso_v else None,
                     "colocated_pertask_steps_s": round(np.mean(col_v), 1),
                     "concurrency_penalty": round(np.mean(pen), 3) if pen else None,
                     "colocated_total_steps_s": round(np.mean(tot), 1),
                     "per_task_rss_mb": round(np.mean(rssv), 0) if rssv else None,
                     "n_nodes": len(col_v)})
    with open(os.path.join(RES, "array_concurrency.csv"), "w", newline="") as fo:
        w = csv.DictWriter(fo, fieldnames=["cpt", "isolated_steps_s", "colocated_pertask_steps_s",
                                           "concurrency_penalty", "colocated_total_steps_s",
                                           "per_task_rss_mb", "n_nodes"])
        w.writeheader(); [w.writerow(r) for r in summ]

    # bandwidth: per-copy throughput at K=1 vs K=4
    bw_by = defaultdict(lambda: defaultdict(list))
    for r in bw:
        bw_by[r["node"]][r["K"]].append(r["gb_per_sec"])
    k1 = [np.mean(d[1]) for d in bw_by.values() if 1 in d]
    k4_per = [np.mean(d[4]) for d in bw_by.values() if 4 in d]  # per-copy at K=4
    with open(os.path.join(RES, "array_bandwidth.csv"), "w", newline="") as fo:
        w = csv.writer(fo); w.writerow(["node", "percopy_gb_s_K1", "percopy_gb_s_K4",
                                        "aggregate_gb_s_K4", "contention_ratio"])
        for nd, d in sorted(bw_by.items()):
            a = np.mean(d.get(1, [np.nan])); b = np.mean(d.get(4, [np.nan]))
            w.writerow([nd, round(a, 2), round(b, 2), round(b * 4, 2),
                        round(b / a, 3) if a else None])

    print("\n### Concurrency: co-located (ntasks=4) vs isolated (1 task), per cpus-per-task\n")
    print("| cpus/task | isolated 1-task steps/s | co-located per-task steps/s | penalty | co-located total (×4) | per-run RSS |")
    print("|--:|--:|--:|--:|--:|--:|")
    for r in summ:
        print(f"| {r['cpt']} | {r['isolated_steps_s']} | {r['colocated_pertask_steps_s']} | "
              f"{r['concurrency_penalty']}× | {r['colocated_total_steps_s']} | {r['per_task_rss_mb']:.0f} MB |")
    if k1 and k4_per:
        print(f"\n### Memory bandwidth (triad): per-copy {np.mean(k1):.1f} GB/s alone (K=1) -> "
              f"{np.mean(k4_per):.1f} GB/s each when 4 run concurrently (K=4); "
              f"aggregate {np.mean(k4_per)*4:.1f} GB/s; contention {np.mean(k4_per)/np.mean(k1):.2f}× per copy")


if __name__ == "__main__":
    main()
