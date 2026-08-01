"""Tests for stage1_controller (frozen-bar decisions), stage1_check (invariants on deliberately
wrong logs), and worker.py's pool routing + 10M walltime guard."""
import hashlib
import importlib
import json
import math
import os
import sys
import time

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
RUN_DIR = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(RUN_DIR, "20_mins_monitoring"))
import build_queue        # noqa: E402
import stage1_controller  # noqa: E402
import stage1_check       # noqa: E402

Z99 = 2.576
SWEEP = "t"
CFG_BELOW = build_queue.CONFIGS[0]    # UMaze alg1 lr0.001 b0.001
CFG_ABOVE = build_queue.CONFIGS[1]    # UMaze alg1 lr0.001 b0.003


def make_tree(tmp_path, monkeypatch, bar_mean=-690.0):
    """Synthetic run tree + redirected controller paths; returns (bars_sha, queue_dir, local_dir)."""
    for sub in ("pending_1m", "pending_10m", "running", "done", "failed", "pruned"):
        os.makedirs(tmp_path / "queue" / SWEEP / sub, exist_ok=True)
    local = tmp_path / "data" / SWEEP / "local"
    os.makedirs(local, exist_ok=True)
    slurm = tmp_path / "slurm"
    os.makedirs(slurm, exist_ok=True)
    bars = {"schema": 1, "bars": {env: {"beta": "1", "mean": bar_mean, "sd": 6.0, "n": 77,
                                        "se": 0.7}
                                  for env in build_queue.ENV_SETUPS_RUN12}}
    (slurm / "FROZEN_BARS.json").write_text(json.dumps(bars))
    monkeypatch.setattr(stage1_controller, "RUN_DIR", str(tmp_path))
    monkeypatch.setattr(stage1_controller, "HERE", str(slurm))
    sha = hashlib.sha256((slurm / "FROZEN_BARS.json").read_bytes()).hexdigest()
    return sha, tmp_path / "queue" / SWEEP, local


def write_records(local, spec, scores):
    """One completed 1M record per score for the given config spec."""
    extras = build_queue.ARM_EXTRAS[spec["arm"]]
    for i, s in enumerate(scores):
        rec = {"completed": True, "total_timesteps": build_queue.STEPS_S,
               "env_setup": spec["env_setup"], "beta": spec["beta"], "rnd_lr": spec["lr"],
               "rnd_optimizer": "sgd",
               "rnd_readout_norm_init": extras.get("rnd_readout_norm_init") == "True",
               "rnd_predictor_loss": extras.get("rnd_predictor_loss", "mse"),
               "rnd_layer_norm": extras.get("rnd_layer_norm") == "True",
               "train_episode_history": [{"train/extrinsic_reward": s}]}
        (local / f"{build_queue.label(spec)}_s{i}.json").write_text(json.dumps(rec))


def write_markers(queue_dir, spec, seeds):
    """Pending markers for the given seeds of one config."""
    for s in seeds:
        name = f"{s:05d}_of_{build_queue.RUN_TOTAL}_{build_queue.label(spec)}_seed{s}.json"
        (queue_dir / "pending_1m" / name).write_text("{}")


def decisions(tmp_path):
    path = tmp_path / "slurm" / f"stage1_decisions_{SWEEP}.jsonl"
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text().splitlines()]


def test_prune_and_survivor_paths(tmp_path, monkeypatch):
    """Golden path: a clearly-below config prunes (markers archived, one decision line); a config
    reaching 100 seeds above the bar gets the survivor verdict; a second cycle is a no-op."""
    make_tree(tmp_path, monkeypatch)
    _, q, local = None, tmp_path / "queue" / SWEEP, tmp_path / "data" / SWEEP / "local"
    write_records(local, CFG_BELOW, [-700.0 + 0.1 * i for i in range(30)])   # far below -690
    write_records(local, CFG_ABOVE, [-680.0 + 0.1 * i for i in range(100)])  # far above, full target
    write_markers(q, CFG_BELOW, range(30, 100))
    lines = stage1_controller.run_cycle(SWEEP, 30, 100)
    assert len(lines) == 2
    dec = {d["config_key"]: d for d in decisions(tmp_path)}
    assert dec[build_queue.config_key(CFG_BELOW)]["verdict"] == "pruned"
    assert dec[build_queue.config_key(CFG_BELOW)]["pending_moved"] == 70
    assert dec[build_queue.config_key(CFG_ABOVE)]["verdict"] == "survivor"
    assert len(os.listdir(q / "pending_1m")) == 0
    assert len(os.listdir(q / "pruned")) == 70
    # idempotence: a rerun makes no new decisions and moves nothing
    assert stage1_controller.run_cycle(SWEEP, 30, 100) == []
    assert len(decisions(tmp_path)) == 2


def test_no_decision_below_floor_and_baseline_exempt(tmp_path, monkeypatch):
    """Edge: n=29 -> no decision; task-R baseline records (10M / adam) never enter scoring."""
    make_tree(tmp_path, monkeypatch)
    local = tmp_path / "data" / SWEEP / "local"
    write_records(local, CFG_BELOW, [-700.0] * 29)
    # a baseline 10M record that would be far below the bar if (wrongly) counted
    base = build_queue.BASELINE_CONFIGS[0]
    rec = {"completed": True, "total_timesteps": build_queue.STEPS_R,
           "env_setup": base["env_setup"], "beta": base["beta"], "rnd_lr": base["lr"],
           "rnd_optimizer": "adam", "rnd_readout_norm_init": False,
           "rnd_predictor_loss": "mse", "rnd_layer_norm": False,
           "train_episode_history": [{"train/extrinsic_reward": -10000.0}]}
    (local / "baseline.json").write_text(json.dumps(rec))
    assert stage1_controller.run_cycle(SWEEP, 30, 100) == []
    assert decisions(tmp_path) == []
    scores = stage1_controller.load_scores(SWEEP)
    assert not any("baseline" in k for k in scores)


def test_crash_between_side_effect_and_log(tmp_path, monkeypatch):
    """Crash-safety: markers moved but the log line never written (simulated crash) -> the next
    cycle re-runs the idempotent move (0 files) and logs EXACTLY once."""
    make_tree(tmp_path, monkeypatch)
    q, local = tmp_path / "queue" / SWEEP, tmp_path / "data" / SWEEP / "local"
    write_records(local, CFG_BELOW, [-700.0 + 0.1 * i for i in range(30)])
    write_markers(q, CFG_BELOW, range(30, 100))
    # the crash: side effect executed, process dies before log_decision
    moved = stage1_controller.move_pending(SWEEP, build_queue.config_key(CFG_BELOW))
    assert moved == 70
    assert decisions(tmp_path) == []
    # restart: the cycle decides again from records; the move is a no-op; one log line total
    stage1_controller.run_cycle(SWEEP, 30, 100)
    dec = decisions(tmp_path)
    assert len(dec) == 1 and dec[0]["verdict"] == "pruned" and dec[0]["pending_moved"] == 0


def test_partial_and_mid_flush_records_ignored(tmp_path, monkeypatch):
    """Edge: completed=false records and unparseable JSON never enter a mean."""
    make_tree(tmp_path, monkeypatch)
    local = tmp_path / "data" / SWEEP / "local"
    write_records(local, CFG_BELOW, [-700.0] * 30)
    partial = json.loads((local / f"{build_queue.label(CFG_BELOW)}_s0.json").read_text())
    partial["completed"] = False
    (local / "partial.json").write_text(json.dumps(partial))
    (local / "mid_flush.json").write_text('{"completed": tru')
    scores = stage1_controller.load_scores(SWEEP)
    assert len(scores[build_queue.config_key(CFG_BELOW)]) == 30


def good_line(key, env, n, mean, std, bar, sha, verdict="pruned"):
    """A well-formed decision line for the checker tests."""
    upper = mean + Z99 * std / math.sqrt(n)
    d = {"verdict": verdict, "config_key": key, "env_setup": env, "n": n, "mean": mean,
         "std": std, "upper_99": upper, "bar": bar, "bars_sha256": sha,
         "time": time.strftime("%Y-%m-%dT%H:%M:%S")}
    if verdict == "pruned":
        d.update({"pending_moved": 0, "n_required": 30})
    else:
        d.update({"n_target": 100})
    return d


def test_checker_correct_and_deliberately_wrong_logs():
    """The pure checker passes a correct log and flags each deliberately-wrong log (sweep_prune
    skill rule 6: unit-tested on correct AND wrong synthetic logs)."""
    key = build_queue.config_key(CFG_BELOW)
    key2 = build_queue.config_key(CFG_ABOVE)
    env = CFG_BELOW["env_setup"]
    bars = {e: -690.0 for e in build_queue.ENV_SETUPS_RUN12}
    sha = "s" * 64
    valid = {build_queue.config_key(c) for c in build_queue.CONFIGS}
    ok = [good_line(key, env, 30, -700.0, 3.0, -690.0, sha),
          good_line(key2, env, 100, -680.0, 3.0, -690.0, sha, verdict="survivor")]
    assert stage1_check.check_decisions(ok, 30, 100, bars, sha, valid) == []
    wrong_cases = {
        "prune below floor": [good_line(key, env, 29, -700.0, 3.0, -690.0, sha)],
        "upper not below bar": [good_line(key, env, 30, -690.5, 3.0, -690.0, sha)],
        "upper_99 inconsistent": [dict(good_line(key, env, 30, -700.0, 3.0, -690.0, sha),
                                       upper_99=-600.0)],
        "bar not frozen value": [good_line(key, env, 30, -700.0, 3.0, -650.0, sha)],
        "sha mismatch": [good_line(key, env, 30, -700.0, 3.0, -690.0, "x" * 64)],
        "double decision": [good_line(key, env, 30, -700.0, 3.0, -690.0, sha)] * 2,
        "unknown / baseline key": [good_line(env + "|baseline|lr0.0001|b10000", env, 30,
                                             -700.0, 3.0, -690.0, sha)],
        "survivor below target": [good_line(key2, env, 99, -680.0, 3.0, -690.0, sha,
                                            verdict="survivor")],
        "survivor that should have pruned": [good_line(key2, env, 100, -700.0, 3.0, -690.0, sha,
                                                       verdict="survivor")],
        "env not matching key": [good_line(key, "AntMaze_Medium-v5_start_bottom_left", 30,
                                           -700.0, 3.0, -690.0, sha)],
    }
    for name, lines in wrong_cases.items():
        got = stage1_check.check_decisions(lines, 30, 100, bars, sha, valid)
        assert got, f"deliberately-wrong log not flagged: {name}"


def load_worker(monkeypatch, tmp_path, pools, device, job_end):
    """Import a fresh worker module under controlled env (RUN_DIR/SWEEP_ID/pools/device)."""
    monkeypatch.setenv("RUN_DIR", str(tmp_path))
    monkeypatch.setenv("SWEEP_ID", SWEEP)
    monkeypatch.setenv("WORKER_POOLS", pools)
    if device:
        monkeypatch.setenv("WORKER_DEVICE", device)
    else:
        monkeypatch.delenv("WORKER_DEVICE", raising=False)
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    if "worker" in sys.modules:
        del sys.modules["worker"]
    import worker
    worker.JOB_END = job_end
    return worker


def test_worker_pool_routing_and_walltime_guard(tmp_path, monkeypatch):
    """gnolim-style worker (pools '10m 1m', cpu): with plenty of walltime it claims the 10M item
    first; with too little remaining it falls back to the 1M item; a 10m-only worker with unknown
    job end never claims."""
    for sub in ("pending_1m", "pending_10m", "running", "done", "failed", "pruned"):
        os.makedirs(tmp_path / "queue" / SWEEP / sub, exist_ok=True)
    (tmp_path / "queue" / SWEEP / "pending_10m" / "24000_of_24200_x_seed0.json").write_text("{}")
    (tmp_path / "queue" / SWEEP / "pending_1m" / "00000_of_24200_y_seed0.json").write_text("{}")
    # golden path: 20 days of walltime -> the 10M item is claimed first
    w = load_worker(monkeypatch, tmp_path, "pending_10m pending_1m", "cpu",
                    time.time() + 20 * 86400)
    name, path, blocked = w.claim()
    assert name.startswith("24000") and not blocked
    os.rename(path, tmp_path / "queue" / SWEEP / "pending_10m" / name)  # put it back
    # edge: 100 h left on a cpu worker (< 170 h) -> falls back to the 1M item
    w = load_worker(monkeypatch, tmp_path, "pending_10m pending_1m", "cpu",
                    time.time() + 100 * 3600)
    name, path, blocked = w.claim()
    assert name.startswith("00000")
    os.rename(path, tmp_path / "queue" / SWEEP / "pending_1m" / name)
    # edge: cuda worker with 100 h (>= 90 h) MAY claim the 10M item
    w = load_worker(monkeypatch, tmp_path, "pending_10m", "cuda", time.time() + 100 * 3600)
    name, path, blocked = w.claim()
    assert name.startswith("24000")
    os.rename(path, tmp_path / "queue" / SWEEP / "pending_10m" / name)
    # edge: unknown job end -> the 10m-only worker claims nothing and reports blocked
    w = load_worker(monkeypatch, tmp_path, "pending_10m", "cuda", None)
    name, path, blocked = w.claim()
    assert name is None and blocked


if __name__ == "__main__":
    import subprocess
    sys.exit(subprocess.call([sys.executable, "-m", "pytest", "-q", __file__]))
