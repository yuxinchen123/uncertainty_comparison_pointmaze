"""Aggregate the repeated 1/2/4-thread sweep into the 2-core column for Table B.
Per node: mean steps/s over reps at each thread; per-node speedup vs that node's
1-thread mean; aggregate across nodes (mean +/- std). Drops nodes lacking 2-thread
data. Safe to re-run on partial data."""
import csv, glob, json, os, statistics as st
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.dirname(HERE)
LOGS = os.path.join(OUT, "logs")
RES = os.path.join(OUT, "results")
ALGOS = ["no_exploration", "rnd_linear_next_state", "gt_position", "rnd_elliptical"]
THREADS = [1, 2, 4]

# node -> algo -> thread -> [steps/s over reps]
data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
for f in glob.glob(os.path.join(LOGS, "twocore_*", "e2e_*_threads-*_rep-*.json")):
    node = os.path.basename(os.path.dirname(f)).replace("twocore_", "")
    try:
        r = json.load(open(f))
    except Exception:
        continue
    if "steady_steps_per_sec" not in r:
        continue
    data[node][r["algorithm"]][r["n_threads"]].append(r["steady_steps_per_sec"])

# per-node mean over reps
mean_sps = defaultdict(lambda: defaultdict(dict))   # node->algo->thread->mean
reps = defaultdict(lambda: defaultdict(dict))
for node in data:
    for algo in data[node]:
        for n in data[node][algo]:
            vals = data[node][algo][n]
            mean_sps[node][algo][n] = st.mean(vals)
            reps[node][algo][n] = len(vals)

# long per-node CSV
os.makedirs(RES, exist_ok=True)
with open(os.path.join(RES, "twocore_long.csv"), "w", newline="") as fo:
    w = csv.writer(fo); w.writerow(["node", "algorithm", "n_threads", "mean_steps_per_sec", "n_reps"])
    for node in sorted(mean_sps):
        for algo in ALGOS:
            for n in THREADS:
                if n in mean_sps[node].get(algo, {}):
                    w.writerow([node, algo, n, round(mean_sps[node][algo][n], 2), reps[node][algo][n]])

# aggregate: per-algo, node-paired speedup@2 and @4 (nodes with that algo's 1 AND n)
print(f"{'algorithm':22s} {'n':>3} {'s/s@1':>7} {'s/s@2':>7} {'s/s@4':>7} "
      f"{'sp@2(±std)':>14} {'sp@4(±std)':>14}")
rows = {}
for algo in ALGOS:
    s1, s2, s4, sp2, sp4 = [], [], [], [], []
    nodes2 = []
    for node in sorted(mean_sps):
        m = mean_sps[node].get(algo, {})
        if 1 in m and 2 in m:
            s1.append(m[1]); s2.append(m[2]); sp2.append(m[2] / m[1]); nodes2.append(node)
            if 4 in m:
                s4.append(m[4]); sp4.append(m[4] / m[1])
    if not sp2:
        print(f"{algo:22s}  (no 2-thread data yet)"); continue
    rows[algo] = {
        "n2": len(sp2),
        "steps1": round(st.mean(s1), 1), "steps2": round(st.mean(s2), 1),
        "steps4": round(st.mean(s4), 1) if s4 else None,
        "sp2": round(st.mean(sp2), 3), "sp2_std": round(st.pstdev(sp2) if len(sp2) > 1 else 0, 3),
        "sp4": round(st.mean(sp4), 3) if sp4 else None,
        "sp4_std": round(st.pstdev(sp4) if len(sp4) > 1 else 0, 3) if sp4 else None,
        "nodes": nodes2,
    }
    r = rows[algo]
    print(f"{algo:22s} {r['n2']:>3} {r['steps1']:>7.1f} {r['steps2']:>7.1f} "
          f"{(r['steps4'] or 0):>7.1f} {r['sp2']:>6.2f}±{r['sp2_std']:<7} "
          f"{(r['sp4'] or 0):>6.2f}±{(r['sp4_std'] or 0):<7}")

json.dump(rows, open(os.path.join(RES, "twocore_summary.json"), "w"), indent=2)
allnodes = sorted(set(n for algo in rows for n in rows[algo]["nodes"]))
print(f"\nnodes with 2-thread data: {len(allnodes)} -> {allnodes}")
