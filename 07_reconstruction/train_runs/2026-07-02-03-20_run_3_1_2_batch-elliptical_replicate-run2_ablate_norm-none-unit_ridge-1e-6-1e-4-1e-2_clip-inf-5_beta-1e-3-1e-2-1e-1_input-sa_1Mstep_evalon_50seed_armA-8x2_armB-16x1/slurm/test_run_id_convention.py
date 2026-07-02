"""Unit tests for Train-run-3.1.2's two-arm queue convention: 36-config inner grid (normalization outer,
ridge, clip, beta inner), parity seed split (arm A even / arm B odd, 25 each), seed-outermost gap-free ids,
run_total=900 per arm. Run: `pytest test_run_id_convention.py`."""
import os
import sys
import json
import glob
import shutil
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))


def _build(tmp_path, sweep_id, arm):
    """Run build_queue.py in a throwaway run dir and return its configs sorted by run_id."""
    slurm_dir = tmp_path / "slurm"
    slurm_dir.mkdir(exist_ok=True)
    shutil.copy(os.path.join(HERE, "build_queue.py"), str(slurm_dir / "build_queue.py"))
    subprocess.check_call([sys.executable, str(slurm_dir / "build_queue.py"), "--sweep_id", sweep_id, "--arm", arm])
    pending = tmp_path / "queue" / sweep_id / "pending"
    return sorted((json.load(open(f)) for f in glob.glob(str(pending / "*.json"))), key=lambda c: c["run_id"])


def test_arm_a_grid_seed_outermost_and_axes(tmp_path):
    """Arm A: 900 configs = 36/seed x 25 even seeds; ids gap-free; the 36-config order and axes are pinned."""
    cfgs = _build(tmp_path, "sweep-a", "A")
    assert len(cfgs) == 900 and all(c["run_total"] == 900 for c in cfgs)
    assert [c["run_id"] for c in cfgs] == list(range(900))
    assert all(c["arm"] == "A" and c["sweep_id"] == "sweep-a" for c in cfgs)
    # even seeds only, seed-outermost: first 36 ids are seed 0, next 36 are seed 2
    assert sorted({c["a_seed"] for c in cfgs}) == list(range(0, 50, 2))
    assert all(c["a_seed"] == 0 for c in cfgs[:36]) and cfgs[36]["a_seed"] == 2
    # config 0 = the run-2 replica axes start: raw features, ridge 1e-6, clip inf, beta 0.001
    p0 = cfgs[0]["params"]
    assert cfgs[0]["algorithm"] == "rnd_elliptical" and cfgs[0]["beta"] == "0.001"
    assert p0 == {"elliptical_feature_normalization": "none", "elliptical_regularization": "1e-06",
                  "elliptical_bonus_clip": "inf", "elliptical_update_timing": "sample",
                  "elliptical_feature_input": "state_action"}
    # the exact run-2 replica cell (none, 1e-6, inf, beta=0.01) is config index 1 within each seed block
    assert cfgs[1]["beta"] == "0.01" and cfgs[1]["params"]["elliptical_bonus_clip"] == "inf"
    # the 36 per-seed configs cover the full factorial: 2 norms x 3 ridges x 2 clips x 3 betas
    seed0 = cfgs[:36]
    combos = {(c["params"]["elliptical_feature_normalization"], c["params"]["elliptical_regularization"],
               c["params"]["elliptical_bonus_clip"], c["beta"]) for c in seed0}
    assert len(combos) == 36
    assert {c["params"]["elliptical_feature_normalization"] for c in seed0} == {"none", "unit"}
    assert {c["params"]["elliptical_regularization"] for c in seed0} == {"1e-06", "0.0001", "0.01"}
    assert {c["params"]["elliptical_bonus_clip"] for c in seed0} == {"inf", "5"}
    # eval is ON for every config (the run-3.1.2 standard: numbers compare to run-2's eval 44.49)
    assert all(c["fixed"]["eval_standalone"] == "True" and c["fixed"]["total_timesteps"] == 1000000 for c in cfgs)


def test_arm_b_gets_odd_seeds_same_grid(tmp_path):
    """Arm B: same 36-config grid, odd seeds 1..49; the two arms' seed sets partition 0..49."""
    cfgs = _build(tmp_path, "sweep-b", "B")
    assert len(cfgs) == 900
    assert sorted({c["a_seed"] for c in cfgs}) == list(range(1, 50, 2))
    assert all(c["arm"] == "B" for c in cfgs)
    # both arms are catalogued in the manifest
    manifest = (tmp_path / "data" / "SWEEPS.md").read_text()
    assert "sweep-b" in manifest and "| 900 |" in manifest


def test_per_run_json_name_id_based():
    """train.py names the JSON 'NNN_of_900.json' (id zero-padded to 900's width = 3)."""
    for run_id, expected in [(0, "000_of_900.json"), (36, "036_of_900.json"), (899, "899_of_900.json")]:
        assert f"{run_id:0{len(str(900))}d}_of_900.json" == expected
