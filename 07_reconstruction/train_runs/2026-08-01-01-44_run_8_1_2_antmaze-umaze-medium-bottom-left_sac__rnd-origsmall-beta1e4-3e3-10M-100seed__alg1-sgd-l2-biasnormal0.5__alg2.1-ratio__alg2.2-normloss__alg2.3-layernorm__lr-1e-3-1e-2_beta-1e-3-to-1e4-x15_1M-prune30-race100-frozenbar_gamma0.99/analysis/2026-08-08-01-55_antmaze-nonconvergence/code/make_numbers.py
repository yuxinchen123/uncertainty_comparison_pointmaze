#!/usr/bin/env python
"""Every number the analysis and the tex subsubsection quote, printed with its verification.

Sections: (V) the cross-checks against already-published values — these gate everything;
(N) the new derived numbers. Run after extract_summaries.py."""
import math

import common as C


def check(name, got, want, tol):
    """Print one verification line and hard-fail on disagreement (no silent drift)."""
    ok = abs(got - want) <= tol
    print(f"  [{'OK ' if ok else 'FAIL'}] {name}: got {got:.4f}, published {want}, tol {tol}")
    if not ok:
        raise SystemExit(f"verification failed: {name}")


def main():
    print("== V: cross-checks against published values ==")
    # V1: run-1.1 last-100 success rates of the RND winner configurations (writeup table)
    for env, beta, want in ((C.UMAZE, "10000", 0.061), (C.MEDIUM, "3000", 0.015)):
        rows = C.load(source="run11", arm="baseline", env_setup=env, beta=beta)
        m, se = C.mean_se([r["succ_last100"] for r in rows])
        print(f"  run-1.1 RND {C.ENV_SHORT[env]} beta={beta}: n={len(rows)}")
        check(f"V1 last-100 success {C.ENV_SHORT[env]}", m, want, 0.01)
    # V2: run-1.2 task-R whole-run rewards (Table 74)
    for env, want in ((C.UMAZE, -695.29), (C.MEDIUM, -996.75)):
        rows = C.load(source="run12", arm="baseline", env_setup=env,
                      total_timesteps=10_000_000)
        m, se = C.mean_se([r["R_whole"] for r in rows])
        print(f"  task-R {C.ENV_SHORT[env]}: n={len(rows)}")
        check(f"V2 task-R reward {C.ENV_SHORT[env]}", m, want, 0.02)
    # V5: reward identity violations over the reward-shift -1 sweeps (runs 1.1 and 1.2). The
    # addendum's PointMaze env uses the RAW sparse reward with no shift, so the -1/step identity
    # does not apply there and its records are excluded from this check.
    viol = sum(r["ident_viol"] for r in C.load(source=["run11", "run12"]))
    n = len(C.load(source=["run11", "run12"]))
    print(f"  [{'OK ' if viol == 0 else 'FAIL'}] V5 reward-identity violations over "
          f"{n} shift(-1) records: {viol}")
    if viol:
        raise SystemExit("V5 failed")

    print("\n== N: the numbers the subsubsection quotes ==")
    # N1: per-1M-block median success rates, task R
    for env in (C.UMAZE, C.MEDIUM):
        rows = C.load(source="run12", arm="baseline", env_setup=env,
                      total_timesteps=10_000_000)
        per_seed = [C.bin_success(r, agg=20)[1] for r in rows]
        med = [sorted(pr[i] for pr in per_seed)[len(per_seed) // 2] for i in range(10)]
        print(f"  N1 {C.ENV_SHORT[env]} per-1M-block median success: "
              + ", ".join(f"{x:.4f}" for x in med))
        last = [r["last_success_step"] for r in rows if r["last_success_step"]]
        print(f"     median last-success step: {sorted(last)[len(last)//2]/1e6:.2f}M "
              f"(n={len(last)} of {len(rows)} seeds ever succeed)")
    # N2: median steps-to-goal + the discounted advantage per environment (+ PointMaze contrast)
    for label, src, arm, env, cap, tts in (
            ("AntMaze UMaze", "run12", "baseline", C.UMAZE, 700, 10_000_000),
            ("AntMaze Medium", "run12", "baseline", C.MEDIUM, 1000, 10_000_000),
            ("PointMaze UMaze", "run11", "baseline",
             "PointMaze_UMaze-v3_start_bottom_left", 300, 1_000_000)):
        rows = C.load(source=src, arm=arm, env_setup=env, total_timesteps=tts)
        lens = sorted(k for r in rows for k in r["succ_lens"])
        if not lens:
            print(f"  N2 {label}: no successes")
            continue
        med = lens[len(lens) // 2]
        adv = C.delta_gamma(med, cap)
        scale = (1 - C.GAMMA ** cap) / (1 - C.GAMMA)
        print(f"  N2 {label}: median k={med} of cap {cap}; "
              f"advantage {adv:.3f} = {100*adv/scale:.2f}% of the {scale:.0f} return scale; "
              f"successes n={len(lens)}")
    # N3: the intrinsic:extrinsic ratio trajectory, task R (medians per 1M block)
    for env in (C.UMAZE, C.MEDIUM):
        rows = C.load(source="run12", arm="baseline", env_setup=env,
                      total_timesteps=10_000_000)
        meds = []
        for i in range(0, 200, 20):
            ratios = [sum(r["sum_int"][i:i+20]) / sum(abs(x) for x in r["sum_ext"][i:i+20])
                      for r in rows if sum(abs(x) for x in r["sum_ext"][i:i+20]) > 0]
            meds.append(sorted(ratios)[len(ratios) // 2])
        print(f"  N3 {C.ENV_SHORT[env]} intrinsic:extrinsic per 1M block: "
              + ", ".join(f"{x:.2f}" for x in meds))
    # N4: steps-to-goal trend over training (stumbled vs learned), task-R UMaze
    rows = C.load(source="run12", arm="baseline", env_setup=C.UMAZE,
                  total_timesteps=10_000_000)
    pairs = sorted((s, k) for r in rows for s, k in zip(r["succ_steps"], r["succ_lens"]))
    half = len(pairs) // 2
    k_first = sorted(k for _, k in pairs[:half])
    k_last = sorted(k for _, k in pairs[half:])
    print(f"  N4 UMaze steps-to-goal: first-half median {k_first[len(k_first)//2]}, "
          f"second-half median {k_last[len(k_last)//2]} "
          f"(a converging agent's should FALL); p10 {pairs and sorted(k for _,k in pairs)[len(pairs)//10]}, "
          f"p90 {sorted(k for _,k in pairs)[9*len(pairs)//10]}; memoryless reference ≈ cap/2 = 350")
    # N5: coverage saturation step (first eval at 100% cell coverage), task R
    for env in (C.UMAZE, C.MEDIUM):
        rows = C.load(source="run12", arm="baseline", env_setup=env,
                      total_timesteps=10_000_000)
        firsts = []
        for r in rows:
            hit = [s for s, c in zip(r["eval_steps"], r["cov_cell"]) if c is not None and c >= 100]
            if hit:
                firsts.append(hit[0])
        print(f"  N5 {C.ENV_SHORT[env]} first eval at 100% cell coverage: median "
              f"{sorted(firsts)[len(firsts)//2]/1e6:.2f}M over n={len(firsts)} seeds")
    # N6: the oracle-vs-RND 1M contrast (coverage AND success), run 1.1, both envs
    print("  N6 run-1.1 1M-step arms (mean last-100 success / mean final cell coverage):")
    for env in (C.UMAZE, C.MEDIUM):
        for arm in ("baseline", "gt_position_1m", "gt_position_maze_cell",
                    "no_exploration"):
            rows = C.load(source="run11", arm=arm, env_setup=env)
            if not rows:
                continue
            ms, _ = C.mean_se([r["succ_last100"] for r in rows])
            cov = [r["cov_cell"][-1] for r in rows if r["cov_cell"] and r["cov_cell"][-1] is not None]
            mc, _ = C.mean_se(cov)
            print(f"     {C.ENV_SHORT[env]:15s} {arm:22s} succ {ms:.3f}  cov {mc:6.1f}%  n={len(rows)}")
    # N7: the addendum lr tie-in on AntMaze (success rate + steps-to-goal by lr, best config)
    print("  N7 addendum arms, best config per lr (whole-run success / median k):")
    for env in (C.UMAZE, C.MEDIUM):
        for arm in ("origsmall_lr0.001", "origsmall_lr0.01"):
            rows = C.load(source="addendum", arm=arm, env_setup=env)
            beta, best = C.best_config(rows)
            if not best:
                continue
            ms, _ = C.mean_se([r["succ_whole"] for r in best])
            lens = sorted(k for r in best for k in r["succ_lens"])
            medk = lens[len(lens) // 2] if lens else None
            print(f"     {C.ENV_SHORT[env]:15s} {arm:20s} beta={beta:>6s} succ {ms:.3f} "
                  f"median k {medk}  n={len(best)}")


if __name__ == "__main__":
    main()
