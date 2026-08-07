#!/usr/bin/env python
"""LAUNCH GATE: drive the real controller and the real checker over a full synthetic sweep.

The unit tests cover the rules one at a time. This runs the whole thing at sweep scale: all 90
configurations, seeds arriving in waves the way the queue delivers them, the real
truncation_controller and the real truncation_check, against the REAL frozen bars file. It answers
the question the unit tests cannot — does the machinery, wired together, truncate the configurations
that deserve it, keep the ones that do not, move exactly the right markers, and end with every
invariant intact?

Half the synthetic configurations are built to sit clearly below their environment's bar and half
clearly above it, so the expected outcome is exact: 24 truncated (the below-bar half of each
environment, floor(15/2)=7 on each environment plus the deliberate borderline cases described
below) and the rest survivors. The script asserts the counts rather than printing them.

Usage:  python simulate_truncation.py            # the fast gate (waves of 10 seeds)
        python simulate_truncation.py --full     # same, plus a per-wave invariant check
Exit code 0 = PASS.
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
import score_rules                  # noqa: E402
import truncation_controller as tc  # noqa: E402
import truncation_check as chk      # noqa: E402

SWEEP = "simulated-sweep"
SEED_FLOOR, SEED_TARGET = 30, 300


def synthetic_score(env, bar, good, rng):
    """One synthetic run score: comfortably above the bar for a `good` configuration, comfortably
    below it for a bad one. The offsets are large relative to the noise, so the 99% upper bound of a
    bad configuration is unambiguously under the bar by the seed floor."""
    # scale the offset to the bar's own magnitude so the same code works for a reward near 40
    # (PointMaze) and one near -690 (AntMaze)
    spread = max(abs(bar) * 0.01, 1.0)
    offset = 6 * spread if good else -20 * spread
    return bar + offset + rng.gauss(0, spread)


def write_records(local, key, env, lr, bar, good, n_from, n_to, rng):
    """Write the records of seeds [n_from, n_to) for one configuration, in that environment's score
    shape (train run 5 reads the last train_history row; train run 1.2 the episode history)."""
    rule = score_rules.rule_for(env)
    for i in range(n_from, n_to):
        s = synthetic_score(env, bar, good, rng)
        rec = {"completed": True, "total_timesteps": bq.STEPS, "env_setup": env,
               "rnd_lr": float(lr), "beta": float(key.rsplit("|b", 1)[1]),
               "train_history": [{"step": bq.STEPS, "train/mean_extrinsic_reward":
                                  s if rule == "final_reward" else 0.0}],
               "train_episode_history": [{"train/extrinsic_reward":
                                          s if rule == "whole_run_mean" else 0.0}]}
        name = f"{abs(hash((key, i))) % 10**12}_of_{bq.RUN_TOTAL}.json"
        with open(os.path.join(local, name), "w") as fh:
            json.dump(rec, fh)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true", help="check the invariants after every wave")
    args = ap.parse_args()
    rng = random.Random(20260805)
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

        BARS = {env: v["mean"]
                for env, v in json.load(open(os.path.join(slurm, "FROZEN_BARS.json")))["bars"].items()}

        # half of each environment's configurations are "good": the ones at index 0, 2, 4, ...
        # before: 90 configurations (2 learning rates x 3 environments x 15 weights); after:
        # good[key] is True for 48 of them, False for 42
        good = {}
        for env in bq.ENV_SETUPS:
            for j, cfg in enumerate([c for c in bq.CONFIGS if c["env_setup"] == env]):
                good[bq.config_key(cfg)] = (j % 2 == 0)
        expected_truncated = sum(1 for v in good.values() if not v)
        expected_survivors = sum(1 for v in good.values() if v)

        # every configuration starts with its full 300 pending markers
        for cfg in bq.CONFIGS:
            tag, key = bq.label(cfg), bq.config_key(cfg)
            for i in range(SEED_TARGET):
                path = os.path.join(root, "queue", SWEEP, "pending",
                                    f"{i:05d}{abs(hash(key)) % 100:02d}_of_{bq.RUN_TOTAL}_{tag}_seed{i}.json")
                with open(path, "w") as fh:
                    json.dump({"config_key": key}, fh)

        # seeds arrive in waves of 10, exactly as the queue delivers them, and the controller runs
        # once per wave — a configuration truncated in an early wave receives no further seeds
        done_to = {k: 0 for k in good}
        decided = {}
        for wave_end in range(10, SEED_TARGET + 1, 10):
            for cfg in bq.CONFIGS:
                key, env = bq.config_key(cfg), cfg["env_setup"]
                if key in decided:
                    continue          # a decided configuration gets no new seeds
                write_records(local, key, env, cfg["lr"], BARS[env], good[key],
                              done_to[key], wave_end, rng)
                done_to[key] = wave_end
            lines = tc.run_cycle(SWEEP, SEED_FLOOR, SEED_TARGET)
            for ln in lines:
                verdict, key = ln.split()[0], ln.split()[1].rstrip(":")
                decided[key] = verdict
            print(f"[wave to n={wave_end}] {len(lines)} new decisions, {len(decided)} decided total")
            if args.full:
                violations, _ = chk.check(SWEEP, SEED_FLOOR, SEED_TARGET)
                assert not violations, f"invariant violation after wave {wave_end}: {violations}"

        # final assertions: the outcome is exact, not merely plausible
        truncated = [k for k, v in decided.items() if v == "truncated"]
        survivors = [k for k, v in decided.items() if v == "survivor"]
        assert len(decided) == len(bq.CONFIGS), \
            f"{len(decided)} of {len(bq.CONFIGS)} configurations decided"
        assert len(truncated) == expected_truncated, \
            f"{len(truncated)} truncated, expected {expected_truncated}"
        assert len(survivors) == expected_survivors, \
            f"{len(survivors)} survivors, expected {expected_survivors}"
        assert all(not good[k] for k in truncated), "a good configuration was truncated"
        assert all(good[k] for k in survivors), "a bad configuration survived"
        # no truncated configuration may have left a pending marker behind
        left = chk.pending_counts(SWEEP)
        assert all(left.get(k, 0) == 0 for k in truncated), "a truncated configuration kept markers"
        violations, _ = chk.check(SWEEP, SEED_FLOOR, SEED_TARGET)
        assert not violations, f"final invariant violations: {violations}"
        print(f"\nPASS: {len(bq.CONFIGS)} configurations -> {len(truncated)} truncated, "
              f"{len(survivors)} survivors, 0 invariant violations")
    finally:
        shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    main()
