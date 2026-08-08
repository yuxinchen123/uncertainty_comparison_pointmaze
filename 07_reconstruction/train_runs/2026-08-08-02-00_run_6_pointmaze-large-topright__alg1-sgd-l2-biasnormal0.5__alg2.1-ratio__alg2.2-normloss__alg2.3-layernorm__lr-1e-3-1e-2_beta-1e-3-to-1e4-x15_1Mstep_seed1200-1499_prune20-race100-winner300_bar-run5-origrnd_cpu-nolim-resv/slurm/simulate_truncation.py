#!/usr/bin/env python
"""Launch gate: the whole two-phase machinery at sweep scale on synthetic data.

The unit tests cover the rules one at a time. This runs all 120 configurations through seed waves
exactly as the queue delivers them, with the controller deciding once per wave and the invariant
checker verifying after every wave. Per arm, the synthetic truth plants: most configurations far
below the bar (phase-1 truncations), a few above it (candidates), and ONE clearly best (the
intended winner). The gate passes only if every arm's intended winner is crowned, every planted
loser is truncated, every non-winner candidate stops at 100, winners complete at 300, and the
checker reports zero violations at every wave.

Usage:  python simulate_truncation.py [--full]     (--full = run the checker after every wave)
"""
import argparse
import json
import os
import random
import shutil
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
RUN_DIR = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(RUN_DIR, "20_mins_monitoring"))
import build_queue as bq            # noqa: E402
import truncation_controller as tc  # noqa: E402
import truncation_check as chk      # noqa: E402

SWEEP = "simulated-sweep"
N_REQUIRED, N_RACE, N_FINAL = 20, 100, 300
BAR = 38.6412


def synthetic_score(kind, rng):
    """One synthetic final reward: 'loser' sits far below the bar, 'candidate' above it,
    'winner' clearly best. Spreads are small relative to the offsets, so phase 1 decides at the
    20-seed floor and the winner ordering is unambiguous."""
    base = {"loser": BAR - 25, "candidate": BAR + 6, "winner": BAR + 14}[kind]
    return base + rng.gauss(0, 3.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true", help="check the invariants after every wave")
    args = ap.parse_args()
    rng = random.Random(20260808)
    root = tempfile.mkdtemp(prefix="simulate_truncation_")
    try:
        # a temporary run tree that mirrors the real one, with the REAL bars file copied in
        slurm = os.path.join(root, "slurm")
        os.makedirs(slurm)
        shutil.copy(os.path.join(HERE, "FROZEN_BARS.json"), slurm)
        local = os.path.join(root, "data", SWEEP, "local")
        os.makedirs(local)
        for sub in ("pending", "running", "done", "failed", "pruned"):
            os.makedirs(os.path.join(root, "queue", SWEEP, sub))
        tc.RUN_DIR, tc.HERE = root, slurm
        chk.RUN_DIR, chk.SLURM = root, slurm

        # the planted truth per arm: 2 candidates + 1 winner per arm, everything else a loser
        # before: 30 configurations per arm; after: kind["...|alg1|lr0.01|b30"] == "winner" etc.
        kind = {}
        for arm in bq.ARMS:
            arm_cfgs = [c for c in bq.CONFIGS if c["arm"] == arm]
            for j, cfg in enumerate(arm_cfgs):
                kind[bq.config_key(cfg)] = ("winner" if j == 7 else
                                            "candidate" if j in (3, 12) else "loser")

        # every configuration starts with its full 300 pending markers
        width = len(str(bq.RUN_TOTAL))
        for seed_index in bq.SEED_INDICES:
            for cfg_index, cfg in enumerate(bq.CONFIGS):
                run_id = seed_index * len(bq.CONFIGS) + cfg_index
                name = (f"{run_id:0{width}d}_of_{bq.RUN_TOTAL}_{bq.label(cfg)}_"
                        f"seed{bq.a_seed_of(seed_index)}.json")
                with open(os.path.join(root, "queue", SWEEP, "pending", name), "w") as fh:
                    json.dump({"config_key": bq.config_key(cfg), "seed_index": seed_index}, fh)

        def complete_seed(cfg, seed_index):
            # a wave "runs" a seed: its pending marker leaves the queue and a completed record
            # appears, exactly what the controller sees in production
            cfg_index = bq.CONFIGS.index(cfg)
            run_id = seed_index * len(bq.CONFIGS) + cfg_index
            name = (f"{run_id:0{width}d}_of_{bq.RUN_TOTAL}_{bq.label(cfg)}_"
                    f"seed{bq.a_seed_of(seed_index)}.json")
            src = os.path.join(root, "queue", SWEEP, "pending", name)
            if os.path.exists(src):
                os.rename(src, os.path.join(root, "queue", SWEEP, "done", name))
            rec = {"completed": True, "total_timesteps": bq.STEPS,
                   "env_setup": cfg["env_setup"], "beta": float(cfg["beta"]),
                   "rnd_lr": float(cfg["lr"]),
                   **{k: v for k, v in cfg["params"].items()
                      if k in ("rnd_readout_norm_init", "rnd_predictor_loss", "rnd_layer_norm")},
                   "train_history": [{"step": bq.STEPS, "train/mean_extrinsic_reward":
                                      synthetic_score(kind[bq.config_key(cfg)], rng)}]}
            with open(os.path.join(local, f"{run_id}_of_{bq.RUN_TOTAL}.json"), "w") as fh:
                json.dump(rec, fh)

        # seeds arrive in waves of 10, in seed order, exactly as the claim window delivers them;
        # a decided-away configuration receives no further seeds (its markers are gone)
        done_to = {bq.config_key(c): 0 for c in bq.CONFIGS}
        for wave_end in range(10, N_FINAL + 1, 10):
            for cfg in bq.CONFIGS:
                k = bq.config_key(cfg)
                for i in range(done_to[k], wave_end):
                    name_prefix = f"{i * len(bq.CONFIGS) + bq.CONFIGS.index(cfg):0{width}d}_"
                    marker = os.path.join(root, "queue", SWEEP, "pending",
                                          f"{name_prefix}of_{bq.RUN_TOTAL}_{bq.label(cfg)}_"
                                          f"seed{bq.a_seed_of(i)}.json")
                    if os.path.exists(marker):
                        complete_seed(cfg, i)
                done_to[k] = wave_end
            lines = tc.run_cycle(SWEEP, N_REQUIRED, N_RACE, N_FINAL)
            if lines:
                print(f"[wave to n={wave_end}] {len(lines)} new decisions")
            if args.full:
                violations, _ = chk.check(SWEEP, N_REQUIRED, N_RACE, N_FINAL)
                assert not violations, f"violations after wave {wave_end}: {violations[:5]}"

        # final assertions: the outcome equals the planted truth exactly
        decided = tc.already_decided(SWEEP)
        losers = [k for k, v in kind.items() if v == "loser"]
        cands = [k for k, v in kind.items() if v == "candidate"]
        winners = [k for k, v in kind.items() if v == "winner"]
        assert all(decided.get(k) == "truncated" for k in losers), "a planted loser survived"
        assert all(decided.get(k) == "stopped_at_100" for k in cands), \
            f"a planted candidate did not stop at 100: { {k: decided.get(k) for k in cands} }"
        assert all(decided.get(k) == "winner_complete" for k in winners), \
            f"a planted winner did not complete: { {k: decided.get(k) for k in winners} }"
        violations, _ = chk.check(SWEEP, N_REQUIRED, N_RACE, N_FINAL)
        assert not violations, f"final violations: {violations[:5]}"
        n_run = len(os.listdir(os.path.join(root, "queue", SWEEP, "done")))
        print(f"\nPASS: 120 configurations -> {len(losers)} truncated, {len(cands)} stopped at "
              f"100, {len(winners)} winners complete at 300; {n_run} seed-runs consumed; "
              f"0 invariant violations")
    finally:
        shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    main()
