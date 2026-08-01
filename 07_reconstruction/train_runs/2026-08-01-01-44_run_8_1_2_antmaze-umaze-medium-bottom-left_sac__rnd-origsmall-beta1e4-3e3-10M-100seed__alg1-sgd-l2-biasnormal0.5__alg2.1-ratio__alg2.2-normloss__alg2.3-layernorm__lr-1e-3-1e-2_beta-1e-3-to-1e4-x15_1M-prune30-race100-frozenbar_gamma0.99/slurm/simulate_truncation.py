#!/usr/bin/env python
"""Truncation simulation for train run 8.1.2 — the REQUIRED launch gate (user rule): the REAL
stage-1 controller code is driven against a synthetic universe with known true config means, and
the simulation asserts the truncation behaves correctly before the full sweep may start.

What it does:
1. Builds a temporary run-folder tree (queue markers via build_queue's own label/key functions, a
   synthetic FROZEN_BARS.json, an empty decisions log) under a local tmp dir.
2. Draws each sampled config a TRUE mean at a known offset from its env's bar (clearly-below /
   near-bar / clearly-above tiers) with a run-8.1-like per-seed spread, using per-config keyed
   streams (the rng-seeding rule: adding a config never shifts another config's draws).
3. Feeds completed 1M records to the controller in seed WAVES with random per-wave arrival order
   (seed-outermost, like the real queue) and calls stage1_controller.run_cycle after every wave —
   each call is also a controller "restart" because run_cycle rebuilds all state from the log.
4. Asserts, at the end:
   - every clearly-below config was pruned, and none of them kept a pending_1m marker;
   - every clearly-above config reached the survivor verdict at the seed target;
   - no decision was made below the n_required floor; exactly one decision per decided config;
   - every sampled config is decided by the end (near-bar ones either way);
   - re-running run_cycle after completion makes no new decision (idempotence);
   - the final decision log passes stage1_check.check_decisions with zero violations.

Usage:  python simulate_truncation.py [--configs 60] [--seed 0] [--full]
        (pytest wrapper: test_simulate_truncation.py; --full simulates all 240 configs)
"""
import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "20_mins_monitoring"))
import build_queue        # noqa: E402
import stage1_controller  # noqa: E402
import stage1_check       # noqa: E402

N_REQUIRED = 30
N_TARGET = 100
SWEEP = "sim-sweep"
# per-seed score spread, roughly the run-8.1 UMaze scale (sd ~ 6); the tier offsets are in
# per-seed-sd units and chosen so the 99% rule is statistically certain at the floor:
# clearly-below = -6 sd (upper bound ~ mean + 2.9 sd at n=30 -> far under the bar),
# clearly-above = +6 sd, near-bar = {-0.5, 0, +0.5} sd (either verdict is legitimate).
SD = 6.0
TIER_OFFSETS = {"below": -6.0, "near": 0.0, "near_lo": -0.5, "near_hi": 0.5, "above": 6.0}


def keyed_rng(*parts):
    """One numpy Generator per named quantity (rng-seeding rule), keyed by stable strings."""
    key = "::".join(str(p) for p in parts)
    return np.random.default_rng(int(hashlib.sha256(key.encode()).hexdigest(), 16) & 0xFFFFFFFF)


def synth_record(cfg_spec, seed, score):
    """A minimal completed 1M record carrying exactly the fields the controller reads.
    before: cfg_spec {'env_setup':..., 'arm':'alg2.2', 'lr':'0.01', 'beta':'3'}, score -685.2
    after:  a dict whose key_from_record() == build_queue.config_key(cfg_spec)"""
    extras = build_queue.ARM_EXTRAS[cfg_spec["arm"]]
    return {
        "completed": True, "total_timesteps": build_queue.STEPS_S,
        "env_setup": cfg_spec["env_setup"], "algorithm": "rnd_next_state",
        "beta": cfg_spec["beta"], "a_seed": seed, "rnd_lr": cfg_spec["lr"],
        "rnd_optimizer": "sgd",
        "rnd_readout_norm_init": extras.get("rnd_readout_norm_init") == "True",
        "rnd_predictor_loss": extras.get("rnd_predictor_loss", "mse"),
        "rnd_layer_norm": extras.get("rnd_layer_norm") == "True",
        "train_episode_history": [{"train/extrinsic_reward": score}],
    }


def run_simulation(n_configs, base_seed, verbose=True):
    """Run the whole simulation; return the list of failure strings (empty = pass)."""
    failures = []
    tmp = tempfile.mkdtemp(prefix="run812_truncation_sim_", dir="/tmp")
    try:
        # ---- synthetic run tree: queue dirs, bars file, and the controller redirected into it ----
        for sub in ("pending_1m", "pending_10m", "running", "done", "failed", "pruned"):
            os.makedirs(os.path.join(tmp, "queue", SWEEP, sub), exist_ok=True)
        local = os.path.join(tmp, "data", SWEEP, "local")
        os.makedirs(local, exist_ok=True)
        slurm_dir = os.path.join(tmp, "slurm")
        os.makedirs(slurm_dir, exist_ok=True)
        bars = {env: {"beta": "1", "mean": -700.0 + 10.0 * i, "sd": SD, "n": 77, "se": 0.7}
                for i, env in enumerate(build_queue.ENV_SETUPS_RUN12)}
        bars_path = os.path.join(slurm_dir, "FROZEN_BARS.json")
        with open(bars_path, "w") as fh:
            json.dump({"schema": 1, "source_sweep": "sim", "bars": bars}, fh)
        # redirect the REAL controller's module paths into the synthetic tree (the production file
        # derives them from its own location; the simulation is the one sanctioned override)
        stage1_controller.RUN_DIR = tmp
        stage1_controller.HERE = slurm_dir

        # ---- sample configs and assign known true means by tier (round-robin over tiers) ----
        sample = build_queue.CONFIGS[:n_configs]
        tiers = ["below", "above", "near", "below", "above", "near_lo", "below", "near_hi"]
        truth = {}
        for i, spec in enumerate(sample):
            tier = tiers[i % len(tiers)]
            key = build_queue.config_key(spec)
            truth[key] = (tier, bars[spec["env_setup"]]["mean"] + TIER_OFFSETS[tier] * SD)
        # every sampled config gets its full 100 seed markers in pending_1m (the real layout)
        width = len(str(build_queue.RUN_TOTAL))
        for s in range(N_TARGET):
            for j, spec in enumerate(sample):
                name = f"{(s * len(sample) + j):0{width}d}_of_{build_queue.RUN_TOTAL}_" \
                       f"{build_queue.label(spec)}_seed{s}.json"
                with open(os.path.join(tmp, "queue", SWEEP, "pending_1m", name), "w") as fh:
                    json.dump({"config_key": build_queue.config_key(spec)}, fh)

        # ---- seed waves: records arrive in randomized within-wave order; controller runs per wave.
        # A pruned config stops producing records (its markers were archived = seeds never ran). ----
        pruned_live = set()
        for wave in range(N_TARGET):
            order = keyed_rng(base_seed, "wave-order", wave).permutation(len(sample))
            for j in order:
                spec = sample[j]
                key = build_queue.config_key(spec)
                if key in pruned_live:
                    continue
                tier, true_mean = truth[key]
                score = float(true_mean + SD * keyed_rng(base_seed, "score", key, wave).standard_normal())
                rec = synth_record(spec, wave, score)
                with open(os.path.join(local, f"sim_{key.replace('|', '_')}_s{wave}.json"), "w") as fh:
                    json.dump(rec, fh)
            # run the REAL controller (every call rebuilds state from the log = a restart each wave)
            stage1_controller.run_cycle(SWEEP, N_REQUIRED, N_TARGET)
            for k, v in stage1_controller.already_decided(SWEEP).items():
                if v == "pruned":
                    pruned_live.add(k)

        # one extra cycle after completion must be a no-op (idempotence)
        n_before = len(stage1_controller.already_decided(SWEEP))
        lines_extra = stage1_controller.run_cycle(SWEEP, N_REQUIRED, N_TARGET)
        decided = stage1_controller.already_decided(SWEEP)
        if lines_extra or len(decided) != n_before:
            failures.append(f"idempotence: extra cycle produced {len(lines_extra)} new decisions")

        # ---- assertions against the known truth ----
        for key, (tier, _) in truth.items():
            verdict = decided.get(key)
            if tier == "below" and verdict != "pruned":
                failures.append(f"clearly-below config not pruned: {key} verdict={verdict}")
            if tier == "above" and verdict != "survivor":
                failures.append(f"clearly-above config not a survivor: {key} verdict={verdict}")
            if verdict is None:
                failures.append(f"undecided config at the end: {key} (tier {tier})")
        # decision-log discipline: floor respected, one decision per config, checker passes
        log_path = os.path.join(slurm_dir, f"stage1_decisions_{SWEEP}.jsonl")
        lines = [json.loads(l) for l in open(log_path)]
        per_key = {}
        for d in lines:
            per_key[d["config_key"]] = per_key.get(d["config_key"], 0) + 1
            if d["n"] < N_REQUIRED:
                failures.append(f"decision below the floor: {d['config_key']} n={d['n']}")
        for k, c in per_key.items():
            if c != 1:
                failures.append(f"{c} decisions for {k}")
        raw = open(bars_path, "rb").read()
        checker_violations = stage1_check.check_decisions(
            lines, N_REQUIRED, N_TARGET,
            {env: v["mean"] for env, v in json.loads(raw)["bars"].items()},
            hashlib.sha256(raw).hexdigest(),
            {build_queue.config_key(c) for c in build_queue.CONFIGS})
        failures.extend(checker_violations)
        # pruned configs keep no pending markers; their archived markers exist in pruned/
        pending_left = os.listdir(os.path.join(tmp, "queue", SWEEP, "pending_1m"))
        for key, verdict in decided.items():
            if verdict != "pruned":
                continue
            spec = next(c for c in build_queue.CONFIGS if build_queue.config_key(c) == key)
            tag = build_queue.label(spec)
            if any(f"_{tag}_seed" in n for n in pending_left):
                failures.append(f"pruned config still pending: {key}")
        n_pruned = sum(1 for v in decided.values() if v == "pruned")
        n_surv = sum(1 for v in decided.values() if v == "survivor")
        if verbose:
            print(f"simulation: {len(truth)} configs -> {n_pruned} pruned, {n_surv} survivors, "
                  f"{len(failures)} failures")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return failures


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--configs", type=int, default=60)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--full", action="store_true", help="simulate all 240 configs")
    args = p.parse_args()
    n = len(build_queue.CONFIGS) if args.full else args.configs
    failures = run_simulation(n, args.seed)
    for f in failures:
        print(f"FAILURE: {f}")
    if failures:
        sys.exit(1)
    print("truncation simulation PASS")


if __name__ == "__main__":
    main()
