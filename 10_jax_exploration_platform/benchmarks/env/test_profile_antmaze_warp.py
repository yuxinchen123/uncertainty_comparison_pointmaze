"""Tests of profile_antmaze_warp.py: the Markdown rendering, and the measuring functions at tiny sizes.

The rendering tests use the standard library only. The measuring tests need warp-lang and run MuJoCo
Warp on processors (a compute node, never a login node); they check that every kind of cell fills its
record, not how fast it is.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import profile_antmaze_warp as profile  # noqa: E402

POOL = {"nacon_peak": 9000, "ncollision_peak": 20000, "naconmax": 196608, "nefc_peak": 52, "njmax": 160,
        "overflow_bits": 0, "dropped": False, "reasons": []}


def timed(ms_per_env_step, first_call_seconds=8.0, rounds=15, **extra):
    """A timed-function record whose every round took the given milliseconds per environment step."""
    seconds = ms_per_env_step / 1000.0 * profile.ENV_STEPS_PER_CALL
    return dict(first_call_seconds=first_call_seconds, call_seconds=[seconds] * rounds, **extra)


def cell(kind, name, others=(), **fields):
    """A measured cell of one kind."""
    return {"kind": kind, "name": name, "status": "measured", "measured_pacific": "2026-11-12 02:30 PT",
            "env_steps_per_call": profile.ENV_STEPS_PER_CALL, "rounds_done": 15, "functions": {},
            "other_processes_on_card": list(others), "wall_seconds": 30.0,
            "memory": {"jax_peak_bytes": 2 ** 30, "warp_pool_peak_bytes": 2 ** 29, "process_mib_on_card": 2048},
            **fields}


def speed_cell(implementation, n_envs, ms, **fields):
    """A measured speed cell."""
    extra = {"pool": POOL} if implementation == "warp" else {}
    return cell("speed", profile.speed_name(implementation, n_envs), implementation=implementation,
                environments=n_envs, **extra, **fields) | {"functions": {profile.ENVIRONMENT_STEP: timed(ms)}}


def run_record(cells, unit="unit 1", discarded=None):
    """A finished run holding the given cells."""
    run = {"unit": unit, "machine": "serval02 card 1", "host": "serval02", "parts": ["speed-warp"],
           "card": {"name": "NVIDIA H100 NVL", "driver": "590.48.01"}, "started_pacific": "2026-11-12 02:30 PT",
           "versions": {"jax": "0.11.0", "jaxlib": "0.11.0", "mujoco": "3.11.0", "mujoco-mjx": "3.11.0",
                        "warp-lang": "1.14.0"},
           "warp": {"kernel_cache_dir": "/localtmp/sl5nw/warp_kernel_cache/1.14.0", "kernel_folders_at_start": 0,
                    "kernel_folders_at_end": 40, "cuda_toolkit": "12.9", "cuda_driver": "13.1"},
           "python": "3.12.13", "cells": cells, "finished": True, "wall_seconds": 600.0}
    if discarded:
        run["discarded"] = discarded
    return run


def test_render_speed_table_best_rows_and_ratio():
    """The speed table carries the five throughput columns, the best row per implementation and the ratio."""
    record = {"runs": [run_record([speed_cell("warp", 1024, 2.0), speed_cell("warp", 4096, 4.0),
                                   speed_cell("jax", 1024, 8.0), speed_cell("jax", 4096, 16.0)])]}
    text = profile.render(record)
    # 4,096 environments at 4 ms: 1.024 million steps per second, 250 steps per second per environment,
    # 10^6 / (3600 x 250) = 1.111 hours per million steps per environment
    assert "| MuJoCo Warp | 4,096 | 0.00400 | 1.024 (+100.0%) | 250.0 | 1.111 | 8.0 | 2.00 | 4.6%, 10.2%; 52 |" in text
    assert "| unit 1 | `/localtmp/sl5nw/warp_kernel_cache/1.14.0` | 0 | 40 | speed, warp, 1,024 environments | 8.0 |" in text
    assert "Best total of MuJoCo Warp: 1.024 million steps per second at 4,096 environments." in text
    assert "Best total of MJX JAX: 0.256 million steps per second at 4,096 environments." in text
    assert "| 4,096 | 16.000 | 4.000 | 25.0% | 4.00 times |" in text


def test_render_skips_shared_card_and_discarded_runs_and_lists_them():
    """A cell measured beside another process, a discarded run and an unmeasured cell are listed, not used."""
    shared = speed_cell("warp", 1024, 1.0, others=["pid 7 user other 100 MiB"])
    unmeasured = {"kind": "speed", "name": profile.speed_name("warp", 8192), "status": "not measured",
                  "reason": "the card's memory ran out"}
    record = {"runs": [run_record([shared, unmeasured]),
                       run_record([speed_cell("warp", 4096, 3.0)], unit="unit 2", discarded="another process appeared"),
                       run_record([speed_cell("warp", 1024, 2.0)], unit="unit 3")]}
    cells = profile.usable_cells(record)
    assert list(cells) == [profile.speed_name("warp", 1024)] and cells[profile.speed_name("warp", 1024)]["unit"] == "unit 3"
    text = profile.render(record)
    assert "another process was on the card: ['pid 7 user other 100 MiB']" in text
    assert "run discarded: another process appeared" in text and "the card's memory ran out" in text


def test_render_every_kind_of_cell():
    """Settings, the trainer's program, the second process, the soak and a check all reach the report."""
    settings = cell("settings", "settings, warp, 4,096 environments", environments=4096) | {"functions": {
        profile.DEFAULT_WARP: timed(4.0, pool=POOL, non_finite_resets=0, mean_solver_iterations=1.5),
        "solver iterations 2": timed(3.0, pool=POOL, non_finite_resets=2, mean_solver_iterations=1.2)}}
    trainer = cell("trainer", "trainer-like scan, 1,024 x 4, graph mode WARP", graph_mode="WARP", result={
        "copies": 1024, "environments_per_copy": 4, "rollout_steps": 128, "calls": 4, "scan_unroll": 16,
        "seconds_per_call": [20.0, 5.0, 0.512, 0.512], "graphs_captured_after_each_call": [2, 2, 2, 2],
        "resets_counted": 40960, "resets_expected_from_the_step_limit": 40960,
        "dtypes": {"qpos": "float32", "statistic": "float64"}})
    soak = cell("soak", "soak, warp, 4,096 environments", environments=4096, marks=[
        {"environment_steps": 2048000, "seconds": 9.0, "non_finite_resets": 0, "pool": POOL}])
    second = cell("second process", "second process on the same card", result={
        "child_exit_code": 0, "parent_calls_of_10_steps_while_the_child_ran": 856, "seconds": 26.5,
        "parent_state_is_finite": True, "parent_pool": POOL,
        "child": {"environments": 256, "environment_steps": 100, "seconds": 18.4, "first_call_seconds": 3.4,
                  "state_is_finite": True, "pool": POOL}})
    resting = {"steps": 1200, "reference_height": 0.38248,
               "c_mujoco": {"height": 0.382481, "largest_speed": 1e-9},
               "warp": {"height": 0.382481, "largest_speed": 3e-6, "c_settled_state_moved_by": 1.6e-8},
               "jax": {"height": 0.382481, "largest_speed": 1e-5, "c_settled_state_moved_by": 5e-6}}
    check = cell("check", "check b: resting height after 1,200 passive steps", result=resting)
    processors = {"host": "puma01", "measured_pacific": "2026-11-12 02:10 PT", "slurm_job": "1", "checks": {
        "check c: a pool below the need is reported": dict(POOL, dropped=True, reasons=["candidate pairs over the shared pool"]),
        "check e: semantics against JAX, continuing task": {"mismatches": [], "largest_observation_difference": 1.2e-5}}}
    maps = cell("maps", "maps, warp, 4,096 environments", environments=4096) | {"functions": {
        "umaze map": timed(4.0, pool=POOL), "large map": timed(5.0, pool=POOL)}}
    platform = cell("platform trainer", "platform training iteration, warp, 512 x 4", result={
        "implementation": "warp", "copies": 512, "environments_per_copy": 4, "rollout_steps": 128,
        "seconds_per_iteration": [60.0, 50.0, 0.512, 0.512, 0.512], "loss_finite": True, "episodes_ended": 0,
        "non_finite_resets": 0})
    text = profile.render({"runs": [run_record([settings, maps, trainer, platform, soak, second, check])]}, processors)
    # 512 x 4 x 128 steps in 0.512 s: 0.512 million steps per second, 1,000 per copy, 0.278 hours per million
    assert "| MuJoCo Warp | 512 x 4 | 0.5120 | 0.512 | 1000.0 | 0.278 | 60, 50 | True | 0; 0 | 2.00 |" in text
    assert "| 4,096 | large map | 5.000 | 125.0% | 0.819 | 8.0 | 4.6%, 10.2%; 52 |" in text
    assert "| solver iterations 2 | 3.000 | 75.0% | 1.365 | 8.0 | 2 | 4.6%, 10.2%; 52 | 1.20 |" in text
    # 0.512 s per 128 steps = 4 ms per environment step; 4,096 environments -> 1.024 million steps per second
    assert "| 1,024 x 4 | WARP | 16 | 4 | 4.000 | 1.024 | 20.0 | 40,960 / 40,960 | 2, 2, 2, 2 | float32 / float64 |" in text
    assert ("| 4 solver iterations, 8 line-search iterations | 4,096 | 2,048,000 | 0 | 9,000; 20,000; 196,608 |"
            " 52 (160) | no | 9 |") in text
    assert "stepped it 100 times: exit code 0, 18.4 s in all, first call 3.4 s" in text
    assert "| MuJoCo Warp, float32 | 0.382481 | 1.00e-06 | 3.00e-06 | 1.60e-08 |" in text
    assert "DROPPED: candidate pairs over the shared pool" in text
    assert "Fields that differ from the JAX environment over two steps of five prepared environments: 0." in text
    assert profile.render({"runs": []}).count("Not measured.") == 6


def test_is_out_of_memory():
    """Only messages that say the card's memory ran out count."""
    assert profile.is_out_of_memory(RuntimeError("RESOURCE_EXHAUSTED: Out of memory while trying to allocate"))
    assert not profile.is_out_of_memory(RuntimeError("MuJoCo Warp dropped contacts"))


def test_measuring_functions_fill_their_cells_on_processors(monkeypatch):
    """Every measuring function, at tiny sizes with MuJoCo Warp on processors."""
    pytest.importorskip("warp")
    import time
    save = lambda: None
    new_cell = lambda: {"functions": {}, "env_steps_per_call": profile.ENV_STEPS_PER_CALL, "rounds_done": 0}
    # GOAL: the speed cell of each implementation.
    for implementation in ("warp", "jax"):
        c = new_cell()
        profile.measure_speed(c, implementation, 4, rounds=11, save=save)
        assert c["rounds_done"] == 11 and profile.median_ms(c, profile.ENVIRONMENT_STEP) > 0
        assert ("pool" in c) == (implementation == "warp")
    # GOAL: two settings in turn.
    monkeypatch.setattr(profile, "SETTINGS", profile.SETTINGS[:2])
    c = new_cell()
    profile.measure_settings(c, 4, rounds=11, save=save)
    assert set(c["functions"]) == {profile.DEFAULT_WARP, "solver iterations 2"}
    assert all(not f["pool"]["dropped"] and len(f["call_seconds"]) == 11 for f in c["functions"].values())
    # GOAL: two maps in turn.
    monkeypatch.setattr(profile, "MAPS", ("umaze", "large"))
    c = new_cell()
    profile.measure_maps(c, 4, rounds=11, save=save)
    assert set(c["functions"]) == {"umaze map", "large map"}
    assert all(not f["pool"]["dropped"] and len(f["call_seconds"]) == 11 for f in c["functions"].values())
    # GOAL: the trainer's program, shortened.
    monkeypatch.setattr(profile, "TRAINER_ROLLOUT_STEPS", 8)
    monkeypatch.setattr(profile, "TRAINER_CALLS", 3)
    monkeypatch.setattr(profile, "TRAINER_EPISODE_STEPS", 6)
    c = new_cell()
    profile.measure_trainer(c, 2, 2, "WARP", 4, rounds=11, save=save)
    assert c["result"]["resets_counted"] == c["result"]["resets_expected_from_the_step_limit"] == 4 * 4
    # GOAL: the soak, shortened to 800 steps of 4 environments.
    monkeypatch.setattr(profile, "SOAK_ENVIRONMENTS", 4)
    monkeypatch.setattr(profile, "SOAK_REQUIRED_STEPS", 800)
    c = new_cell()
    profile.measure_soak(c, deadline=time.time(), soak_seconds=0.0, rounds=11, save=save)
    assert c["marks"][0]["environment_steps"] == 800 and not c["marks"][0]["pool"]["dropped"]
    # GOAL: a second process beside this one.
    monkeypatch.setattr(profile, "SECOND_PROCESS_ENVIRONMENTS", (4, 4))
    c = new_cell()
    profile.measure_second_process(c, rounds=11, save=save)
    assert c["result"]["child_exit_code"] == 0 and c["result"]["child"]["state_is_finite"]
    assert c["result"]["parent_state_is_finite"]
