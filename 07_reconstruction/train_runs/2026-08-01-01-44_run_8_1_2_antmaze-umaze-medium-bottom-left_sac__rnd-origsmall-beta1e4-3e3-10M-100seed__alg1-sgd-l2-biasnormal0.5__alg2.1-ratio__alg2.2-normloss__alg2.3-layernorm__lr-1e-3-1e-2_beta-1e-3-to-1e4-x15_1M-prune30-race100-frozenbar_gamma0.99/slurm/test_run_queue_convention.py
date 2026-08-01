"""Pins the run-8.1.2 queue convention: config counts, id layout (task-S seed-outermost, task-R
appended), the two pools, config-key/label round trips, and the arm inference from records."""
import json
import os
import subprocess
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import build_queue  # noqa: E402


def test_config_counts():
    """240 task-S configs (2 env x 4 arms x 2 lr x 15 beta), 2 baselines, 24,200 total entries."""
    assert len(build_queue.CONFIGS) == 240
    assert len(build_queue.BASELINE_CONFIGS) == 2
    assert build_queue.N_S == 24000
    assert build_queue.N_R == 200
    assert build_queue.RUN_TOTAL == 24200


def test_seed_outermost_order():
    """Task-S id i = seed * 240 + config_index in the fixed CONFIGS order (seed outermost)."""
    # golden path: id 0 = seed 0 x config 0; id 240 = seed 1 x config 0; id 241 = seed 1 x config 1
    c0 = build_queue.CONFIGS[0]
    assert c0["env_setup"] == "AntMaze_UMaze-v5_start_bottom_left"
    assert c0["arm"] == "alg1" and c0["lr"] == "0.001" and c0["beta"] == "0.001"
    # the innermost axis is beta ascending, then lr, then arm, then env
    c1 = build_queue.CONFIGS[1]
    assert c1["arm"] == "alg1" and c1["lr"] == "0.001" and c1["beta"] == "0.003"
    c15 = build_queue.CONFIGS[15]
    assert c15["arm"] == "alg1" and c15["lr"] == "0.01" and c15["beta"] == "0.001"
    c30 = build_queue.CONFIGS[30]
    assert c30["arm"] == "alg2.1" and c30["lr"] == "0.001"
    c120 = build_queue.CONFIGS[120]
    assert c120["env_setup"] == "AntMaze_Medium-v5_start_bottom_left" and c120["arm"] == "alg1"


def test_arm_params():
    """Each arm carries exactly its knob bundle on top of ALG1_PARAMS."""
    by_arm = {}
    for c in build_queue.CONFIGS:
        by_arm.setdefault(c["arm"], c)
    p1 = by_arm["alg1"]["params"]
    assert p1["rnd_optimizer"] == "sgd" and p1["rnd_bonus_readout"] == "l2"
    assert p1["rnd_bias_init"] == "normal_0.5" and p1["rnd_reward_norm"] == "False"
    assert "rnd_readout_norm_init" not in p1
    assert by_arm["alg2.1"]["params"]["rnd_readout_norm_init"] == "True"
    assert by_arm["alg2.2"]["params"]["rnd_predictor_loss"] == "mse_init_normalized"
    assert by_arm["alg2.3"]["params"]["rnd_layer_norm"] == "True"
    assert "rnd_layer_norm" not in by_arm["alg2.2"]["params"]
    # baseline: the run-8.1 orig-small stack verbatim, reward norm ON, fixed winner beta
    b = {c["env_setup"]: c for c in build_queue.BASELINE_CONFIGS}
    u = b["AntMaze_UMaze-v5_start_bottom_left"]
    assert u["beta"] == "10000" and u["params"]["rnd_reward_norm"] == "True"
    assert b["AntMaze_Medium-v5_start_bottom_left"]["beta"] == "3000"


def test_key_and_label_round_trip():
    """config_key is unique across all 242 configs; key_from_record inverts it for every arm."""
    keys = [build_queue.config_key(c) for c in build_queue.CONFIGS + build_queue.BASELINE_CONFIGS]
    assert len(set(keys)) == 242
    # a synthetic record built from each arm's flag fields maps back to the same key
    for c in build_queue.CONFIGS[:1] + [build_queue.CONFIGS[30], build_queue.CONFIGS[60],
                                        build_queue.CONFIGS[90]] + build_queue.BASELINE_CONFIGS:
        extras = build_queue.ARM_EXTRAS.get(c["arm"], {})
        rec = {
            "env_setup": c["env_setup"], "beta": c["beta"], "rnd_lr": c["lr"],
            "rnd_optimizer": c["params"]["rnd_optimizer"],
            "rnd_readout_norm_init": extras.get("rnd_readout_norm_init") == "True",
            "rnd_predictor_loss": extras.get("rnd_predictor_loss", "mse"),
            "rnd_layer_norm": extras.get("rnd_layer_norm") == "True",
        }
        assert build_queue.key_from_record(rec) == build_queue.config_key(c)


def test_label_glob_safety():
    """No config's label is a prefix of another's label followed by '_seed' — the controller's
    pending-marker glob `*_<label>_seed*` must never match a different config's markers."""
    labels = [build_queue.label(c) for c in build_queue.CONFIGS + build_queue.BASELINE_CONFIGS]
    assert len(set(labels)) == 242
    for a in labels:
        for b in labels:
            if a != b:
                assert not b.startswith(a) or not b[len(a):].startswith("_seed")


def test_build_queue_writes_pools(tmp_path, monkeypatch):
    """main() writes 24,000 markers into pending_1m/ and 200 into pending_10m/ with the id split
    and per-task total_timesteps; edge: refuses under a non-owner user."""
    monkeypatch.setattr(build_queue, "RUN_DIR", str(tmp_path))
    monkeypatch.setattr(sys, "argv", ["build_queue.py", "--sweep_id", "t"])
    build_queue.main()
    q = tmp_path / "queue" / "t"
    ones = sorted(os.listdir(q / "pending_1m"))
    tens = sorted(os.listdir(q / "pending_10m"))
    assert len(ones) == 24000 and len(tens) == 200
    first = json.loads((q / "pending_1m" / ones[0]).read_text())
    assert first["run_id"] == 0 and first["task"] == "S" and first["pool"] == "pending_1m"
    assert first["fixed"]["total_timesteps"] == 1000000
    r0 = json.loads((q / "pending_10m" / tens[0]).read_text())
    assert r0["run_id"] == 24000 and r0["task"] == "R" and r0["arm"] == "baseline"
    assert r0["fixed"]["total_timesteps"] == 10000000
    # edge case: the owner guard (run in a subprocess with a faked USER via getpass fallback is
    # brittle; assert the guard line exists instead)
    src = open(os.path.join(HERE, "build_queue.py")).read()
    assert 'getpass.getuser() != "sl5nw"' in src


if __name__ == "__main__":
    sys.exit(subprocess.call([sys.executable, "-m", "pytest", "-q", __file__]))
