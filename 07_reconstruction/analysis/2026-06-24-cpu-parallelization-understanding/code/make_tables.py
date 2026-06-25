"""Emit markdown tables from results/*.csv for analysis.md. Re-runnable on final data."""
import csv, os
HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(os.path.dirname(HERE), "results")


def load(name):
    p = os.path.join(RES, name)
    return list(csv.DictReader(open(p))) if os.path.exists(p) else []


def fnum(x, d=2):
    try:
        return f"{float(x):.{d}f}"
    except (TypeError, ValueError):
        return "—"


def pivot(rows, rowkey, valkey):
    """rowkey -> {n_threads: value}"""
    out = {}
    for r in rows:
        out.setdefault(r[rowkey], {})[int(r["n_threads"])] = r[valkey]
    return out


def verdict_micro(eff, sp):
    """eff/sp are dicts n_threads->float."""
    e8 = eff.get(8, 1.0)
    peak = max(sp.get(n, 1.0) for n in (1, 4, 8, 16) if n in sp)
    sp16 = sp.get(16, None)
    if e8 <= 1.2:
        return "serial — locked to 1 core"
    if peak < 1.25:
        return f"spins up to {e8:.0f} cores, ~no speedup (peak {peak:.2f}x)"
    tail = f", then {sp16:.2f}x at 16" if sp16 is not None else ""
    if peak >= 3:
        return f"scales with size (peak {peak:.2f}x{tail})"
    return f"limited: peak {peak:.2f}x{tail}"


# ---- microbench which-part table ----
mb = load("microbench_summary.csv")
eff = {}
sp = {}
backend = {}
for r in mb:
    op = r["op"]; n = int(r["n_threads"])
    eff.setdefault(op, {})[n] = float(r["mean_effective_cores"])
    sp.setdefault(op, {})[n] = float(r["speedup_vs1"]) if r["speedup_vs1"] else 1.0
    backend[op] = r["backend"]

# display order + human labels (maps each microbench op to the code component)
ORDER = [
    ("mujoco_step", "MuJoCo env.step physics"),
    ("python_loop_256", "VisitCount compute loop (gt_position)"),
    ("np_runningstd", "RND obs RunningMeanStd.update"),
    ("adam_step_134k", "Adam optimizer .step (SAC actor+critics)"),
    ("rnd_mlp_batch1", "Intrinsic fwd per env-step (batch 1)"),
    ("sac_critic_mlp", "SAC critic MLP fwd+bwd (6->256->256->1)"),
    ("sac_actor_mlp", "SAC actor MLP fwd+bwd + squashed-Gaussian"),
    ("rnd_mlp", "RND predictor MLP fwd+bwd (4->256->128)"),
    ("torch_inv_128", "EllipticalBonus inv(128x128) (torch)"),
    ("np_inv_128", "matrix inverse 128x128 (numpy LAPACK)"),
    ("torch_mahalanobis", "EllipticalBonus phi@Lambda^-1 (256x128@128x128)"),
    ("torch_phiT_phi", "EllipticalBonus phi^T@phi (128x256@256x128)"),
    ("np_matmul_small", "small matmul 256x128@128x128 (numpy)"),
    ("torch_mlp_big", "[contrast] big MLP 1024-wide, batch 4096"),
    ("np_matmul_big", "[contrast] big matmul 2048x2048 (numpy)"),
]

print("### Table A — per-primitive thread scaling (microbenchmark, mean over nodes)\n")
print("| Component (code) | Library | eff. cores @8 | speedup @4 | @8 | @16 | verdict |")
print("|---|---|--:|--:|--:|--:|---|")
for op, label in ORDER:
    if op not in sp:
        continue
    e = eff[op]; s = sp[op]
    print(f"| {label} | {backend[op]} | {e.get(8,float('nan')):.1f} | "
          f"{s.get(4,float('nan')):.2f} | {s.get(8,float('nan')):.2f} | "
          f"{s.get(16,float('nan')) if 16 in s else float('nan'):.2f} | {verdict_micro(e,s)} |")

# ---- e2e table ----
e = load("e2e_summary.csv")
print("\n### Table B — end-to-end training, steps/sec scaling (node-paired, mean±std over nodes)\n")
print("| algorithm | 1 thr | 4 thr | 8 thr | 16 thr | speedup @4 | @8 | @16 | eff.cores @16 |")
print("|---|--:|--:|--:|--:|--:|--:|--:|--:|")
ss = pivot(e, "algorithm", "mean_steady_steps_per_sec")
spv = pivot(e, "algorithm", "speedup_vs1")
sps = pivot(e, "algorithm", "speedup_std")
ec = pivot(e, "algorithm", "mean_effective_cores")
for a in ["no_exploration", "rnd_linear_next_state", "gt_position", "rnd_elliptical"]:
    if a not in ss:
        continue
    s = ss[a]; v = spv[a]; sd = sps.get(a, {}); c = ec[a]
    def sv(n):
        return f"{float(v[n]):.2f}±{float(sd.get(n,0)):.2f}" if n in v else "—"
    print(f"| {a} | {fnum(s.get(1),0)} | {fnum(s.get(4),0)} | {fnum(s.get(8),0)} | "
          f"{fnum(s.get(16),0)} | {sv(4)} | {sv(8)} | {sv(16)} | {fnum(c.get(16),1)} |")

# ---- cuda table ----
cu = load("e2e_cuda_summary.csv")
if cu:
    print("\n### Table C — device=cuda (one GPU node)\n")
    print("| algorithm | 1 thr | 4 thr | 8 thr | 16 thr | eff. CPU cores |")
    print("|---|--:|--:|--:|--:|--:|")
    css = pivot(cu, "algorithm", "mean_steady_steps_per_sec")
    cec = pivot(cu, "algorithm", "mean_effective_cores")
    for a, s in css.items():
        c = cec[a]
        print(f"| {a} | {fnum(s.get(1),0)} | {fnum(s.get(4),0)} | {fnum(s.get(8),0)} | "
              f"{fnum(s.get(16),0)} | {fnum(list(c.values())[0],2)} |")
