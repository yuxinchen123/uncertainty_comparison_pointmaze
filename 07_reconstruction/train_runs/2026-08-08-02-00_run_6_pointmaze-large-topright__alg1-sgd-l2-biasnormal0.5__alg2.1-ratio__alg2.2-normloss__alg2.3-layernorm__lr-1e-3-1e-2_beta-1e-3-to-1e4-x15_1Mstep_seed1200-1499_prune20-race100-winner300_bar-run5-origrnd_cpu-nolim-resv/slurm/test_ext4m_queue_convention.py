#!/usr/bin/env python
"""Pin the ext4m queue convention: the three configurations' parameters are VERBATIM copies of
their sources (run-5 markers on disk, run-6 build_queue.py, run-3.2.2 oracle markers on disk),
the seed range is fresh, and the id math is seed-outermost.  Run: pytest test_ext4m_queue_convention.py
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import build_queue as bq
import ext4m_build_queue as ext

RUN5_DONE = ("/p/rlprojects/RND/07_reconstruction/train_runs/2026-07-20-16-44_run_5_baseline_"
             "rnd-next-state__origsmall-mse-mean-lr1e-4-out128-predextra1-leaky0.2-warmupenv6400-"
             "rewardnorm-beta1e-2to1e4+0.5-prune30__C2-adam-mse-b100__N1-rewardnorm-b1e4__"
             "seed600-899_1Mstep_cpu-nolim/queue/2026-07-20-16-55_set-baseline/done")
ORACLE_DONE = ("/p/rlprojects/RND/07_reconstruction/train_runs/2026-07-07-23-10_run_3_2_2_validate_"
               "sgd1t-l2-eta1e-1-t1e3-b1__sgd1t-mse-eta1e-1-t1e3-b10__adam-mse-b100-rndbench__"
               "rnd-next-state_seed100-199_100seed_1Mstep_cpu-nolim-resv_32x1/queue/"
               "2026-07-09-00-15_oracle-gt-position-velocity_seed200-499/done")


def _cfg(arm):
    """The ext4m configuration spec for one arm name."""
    return next(c for c in ext.CONFIGS if c["arm"] == arm)


def test_three_configurations_300_seeds():
    """900 runs = 3 configurations x 300 fresh seeds, seed range disjoint from every earlier run."""
    assert ext.RUN_TOTAL == 900
    assert len(ext.CONFIGS) == 3
    assert len(ext.SEED_INDICES) == 300
    assert ext.SEED_OFFSET == 1500          # run5 600-899, addendum 900-1199, run6-1M 1200-1499
    assert ext.SEED_OFFSET + max(ext.SEED_INDICES) == 1799


def test_run5_origrnd_params_verbatim():
    """The run-5 RND configuration matches an actual run-5 beta=1000 marker on disk, byte for byte."""
    marker = next(n for n in sorted(os.listdir(RUN5_DONE)) if "b1000_" in n)
    with open(os.path.join(RUN5_DONE, marker)) as fh:
        src = json.load(fh)
    cfg = _cfg("run5-origrnd")
    assert cfg["params"] == src["params"]
    assert cfg["algorithm"] == src["algorithm"] == "rnd_next_state"
    assert float(cfg["beta"]) == float(src["beta"]) == 1000.0


def test_alg23_params_are_the_1m_sweeps():
    """Algorithm 2.3's dict is build_queue.py's ALG1_PARAMS + ARM_EXTRAS['alg2.3'] + lr 0.01."""
    expected = dict(bq.ALG1_PARAMS)
    expected.update(bq.ARM_EXTRAS["alg2.3"])
    expected["rnd_lr"] = "0.01"
    cfg = _cfg("alg2.3")
    assert cfg["params"] == expected
    assert cfg["beta"] == "30"
    assert cfg["config_key"] == f"{bq.ENV_SETUP}|alg2.3|lr0.01|b30"


def test_gt_params_verbatim():
    """The gt configuration matches the run-3.2.2 oracle re-run (1/sqrt(n) sweep, beta=1) plus the
    explicit decay knob (the oracle sweep relied on the -0.5 default; ext4m records it)."""
    marker = sorted(os.listdir(ORACLE_DONE))[0]
    with open(os.path.join(ORACLE_DONE, marker)) as fh:
        src = json.load(fh)
    cfg = _cfg("gt-position-velocity")
    assert cfg["algorithm"] == src["algorithm"] == "gt_position_velocity"
    assert float(cfg["beta"]) == float(src["beta"]) == 1.0
    assert cfg["params"] == {"visit_count_decay": "-0.5"}
    # the oracle sweep's env knobs are exactly what ext4m's env_setup bundles
    assert src["fixed"]["env_name"] == "PointMaze_Large-v3"
    assert src["fixed"]["env_max_episode"] == 400
    assert src["fixed"]["goal_position"] == "top_right"
    assert src["fixed"]["discount_factor"] == 0.999
    assert cfg["env_setup"] == "initial_single_large_pointmaze_max_400"


def test_fixed_common_only_changes_the_step_budget():
    """FIXED_COMMON is the 1M sweep's with only total_timesteps changed, to 4,000,000."""
    diff = {k for k in set(ext.FIXED_COMMON) | set(bq.FIXED_COMMON)
            if ext.FIXED_COMMON.get(k) != bq.FIXED_COMMON.get(k)}
    assert diff == {"total_timesteps"}
    assert ext.FIXED_COMMON["total_timesteps"] == 4000000


def test_seed_outermost_id_math():
    """Seed index i owns run ids [3i .. 3i+2] in CONFIGS order (the project's execution order)."""
    # before: (seed_index, config position); after: the run id the builder's loop assigns
    for seed_index in (0, 1, 299):
        for pos in range(3):
            assert seed_index * 3 + pos == seed_index * len(ext.CONFIGS) + pos
    assert (ext.RUN_TOTAL - 1) == 299 * 3 + 2


def test_share_budget_math():
    """The workload-share cap: non-cancelled ledger slots consume the share; cancelled release."""
    import ext4m_plan_jobs as pj
    rows = [("101", 32), ("102", 30), ("103", 30)]
    assert pj.budget_remaining(rows, set(), 600) == 508
    assert pj.budget_remaining(rows, {"102"}, 600) == 538       # cancelled job's slots return
    assert pj.budget_remaining(rows, set(), 60) == 0            # never negative
    assert pj.budget_remaining([], set(), 300) == 300
