#!/usr/bin/env python
"""Unit tests for the run-8.1.2 monitoring report and the two CLI tables behind it: the running-status
counts (both pending pools summed, baseline cells N/A), the per-env interim metrics (best config per
arm, verdict column, frozen-bar line), the marking rule, the worker-log parsing, the memory-bounded
record loading, and the snapshot file the report writes.

Every test builds a synthetic run tree under tmp_path and redirects the report's queue/, data/,
slurm/ and outputs/ to it with the RUN_DIR_OVERRIDE env var (monkeypatch); build_queue is always the
real one, so the tests run against the real 240 + 2 config list.

Run:  /p/rlprojects/RND/.venvs/exploration/bin/python -m pytest 20_mins_monitoring/test_monitoring_report.py -q
"""
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
RUN_DIR = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(RUN_DIR, "slurm"))
import build_queue          # noqa: E402
import status_table         # noqa: E402
import env_metrics_tables   # noqa: E402
import monitoring_report as mr  # noqa: E402

SWEEP = "t"
UMAZE, MEDIUM = build_queue.ENV_SETUPS_RUN12
BARS = {UMAZE: (-690.0, "10000"), MEDIUM: (-998.0, "3000")}


def make_tree(tmp_path, monkeypatch):
    """Synthetic run tree (both queue pools + data/local + slurm with frozen bars) and the
    RUN_DIR_OVERRIDE redirection; returns (queue_dir, local_dir)."""
    for sub in ("pending_1m", "pending_10m", "running", "done", "failed", "pruned"):
        os.makedirs(tmp_path / "queue" / SWEEP / sub, exist_ok=True)
    local = tmp_path / "data" / SWEEP / "local"
    os.makedirs(local, exist_ok=True)
    slurm = tmp_path / "slurm"
    os.makedirs(slurm, exist_ok=True)
    # the frozen bars the report reads (same file layout as the committed slurm/FROZEN_BARS.json)
    bars = {"schema": 1, "bars": {env: {"beta": beta, "mean": mean, "sd": 6.0, "n": 80, "se": 0.7}
                                 for env, (mean, beta) in BARS.items()}}
    (slurm / "FROZEN_BARS.json").write_text(json.dumps(bars))
    monkeypatch.setenv("RUN_DIR_OVERRIDE", str(tmp_path))
    return tmp_path / "queue" / SWEEP, local


def config_of(env_setup, arm, lr, beta):
    """The task-S config spec with these knobs."""
    return next(c for c in build_queue.CONFIGS if c["env_setup"] == env_setup and c["arm"] == arm
                and c["lr"] == lr and c["beta"] == beta)


def baseline_of(env_setup):
    """The env's task-R baseline config spec."""
    return next(c for c in build_queue.BASELINE_CONFIGS if c["env_setup"] == env_setup)


def write_records(local, spec, scores):
    """One completed per-run record per score, carrying the flag fields build_queue.key_from_record
    reads back (adam + 10M for the baseline arm, sgd + 1M for a task-S arm)."""
    for i, score in enumerate(scores):
        if spec["arm"] == "baseline":
            flags = {"rnd_optimizer": "adam", "rnd_readout_norm_init": False,
                     "rnd_predictor_loss": "mse", "rnd_layer_norm": False}
            steps = build_queue.STEPS_R
        else:
            extras = build_queue.ARM_EXTRAS[spec["arm"]]
            flags = {"rnd_optimizer": "sgd",
                     "rnd_readout_norm_init": extras.get("rnd_readout_norm_init") == "True",
                     "rnd_predictor_loss": extras.get("rnd_predictor_loss", "mse"),
                     "rnd_layer_norm": extras.get("rnd_layer_norm") == "True"}
            steps = build_queue.STEPS_S
        rec = dict({"completed": True, "total_timesteps": steps, "env_setup": spec["env_setup"],
                    "beta": spec["beta"], "rnd_lr": spec["lr"],
                    "train_episode_history": [{"train/extrinsic_reward": score}],
                    "eval_history": [{"eval/mean_extrinsic_reward": score}]}, **flags)
        (local / f"{build_queue.label(spec)}_s{i}.json").write_text(json.dumps(rec))


def write_markers(queue_dir, pool, spec, seeds):
    """Queue markers for the given seeds of one config, in the named pool."""
    for s in seeds:
        name = f"{s:05d}_of_{build_queue.RUN_TOTAL}_{build_queue.label(spec)}_seed{s}.json"
        (queue_dir / pool / name).write_text("{}")


def write_decision(tmp_path, spec, verdict):
    """One stage-1 decision line for a config (the fields the tables read)."""
    path = tmp_path / "slurm" / f"stage1_decisions_{SWEEP}.jsonl"
    line = {"verdict": verdict, "config_key": build_queue.config_key(spec),
            "env_setup": spec["env_setup"], "n": 30, "mean": -700.0, "std": 3.0,
            "upper_99": -698.6, "bar": BARS[spec["env_setup"]][0], "bars_sha256": "s" * 64}
    with open(path, "a") as fh:
        fh.write(json.dumps(line) + "\n")


def row_of(rows, env_setup, arm):
    """The status row of one (env_setup, arm) cell."""
    short = env_setup.replace("_start_bottom_left", "")
    return next(r for r in rows if r["env"] == short and r["arm"] == arm)


def test_status_counts_sum_both_pools_and_baseline_is_na(tmp_path, monkeypatch):
    """Golden path: markers and records land in the right cells, `pending` sums pending_1m and
    pending_10m, the decision columns count verdicts, and the baseline rows show N/A."""
    q, local = make_tree(tmp_path, monkeypatch)
    alg1 = config_of(UMAZE, "alg1", "0.001", "0.001")
    alg22 = config_of(UMAZE, "alg2.2", "0.01", "3")
    base = baseline_of(UMAZE)
    # 5 + 2 task-S markers pending, 3 baseline markers in the other pool, plus running/done/failed
    write_markers(q, "pending_1m", alg1, range(5))
    write_markers(q, "pending_1m", alg22, range(5, 7))
    write_markers(q, "pending_10m", base, range(7, 10))
    write_markers(q, "running", alg1, range(10, 11))
    write_markers(q, "done", alg1, range(11, 13))
    write_markers(q, "failed", alg22, range(13, 14))
    write_records(local, alg1, [-700.0, -701.0, -702.0])
    write_records(local, base, [-600.0])
    write_decision(tmp_path, alg1, "pruned")
    rows, total = status_table.build_rows(SWEEP)

    assert len(rows) == 10                                   # 2 env setups x 5 arms
    r = row_of(rows, UMAZE, "alg1")
    assert (r["cfgs"], r["undecided"], r["pruned"], r["survivors"]) == (30, 29, 1, 0)
    assert (r["pending"], r["running"], r["done"], r["failed"], r["compl"]) == (5, 1, 2, 0, 3)
    b = row_of(rows, UMAZE, "baseline")
    assert (b["cfgs"], b["pending"], b["compl"]) == (1, 3, 1)
    assert (b["undecided"], b["pruned"], b["survivors"]) == ("N/A", "N/A", "N/A")
    # TOTAL: 240 task-S + 2 baseline configs; pending sums both pools; N/A cells are not summed
    assert total["cfgs"] == len(build_queue.CONFIGS) + len(build_queue.BASELINE_CONFIGS)
    assert total["pending"] == 10 and total["undecided"] == 239 and total["pruned"] == 1


def test_status_ignores_pruned_pool_partial_and_unreadable_records(tmp_path, monkeypatch):
    """Edge: markers archived under pruned/ are in no state column, a completed=false checkpoint and
    a mid-flush unparseable JSON are not counted as completed records."""
    q, local = make_tree(tmp_path, monkeypatch)
    alg1 = config_of(UMAZE, "alg1", "0.001", "0.001")
    write_markers(q, "pending_1m", alg1, range(2))
    write_markers(q, "pruned", alg1, range(2, 9))            # archived: never runs, never pending
    write_records(local, alg1, [-700.0])
    partial = json.loads((local / f"{build_queue.label(alg1)}_s0.json").read_text())
    partial["completed"] = False
    (local / "partial.json").write_text(json.dumps(partial))
    (local / "mid_flush.json").write_text('{"completed": tru')
    rows, total = status_table.build_rows(SWEEP)
    r = row_of(rows, UMAZE, "alg1")
    assert r["pending"] == 2 and r["compl"] == 1
    assert total["pending"] == 2 and total["compl"] == 1


def test_env_metrics_pick_best_config_verdicts_and_bar_line(tmp_path, monkeypatch):
    """Golden path: each arm's row is its highest-mean config, rows rank by reward, the verdict column
    reads the decision log (N/A for the baseline), and the bar line counts configs at or above the bar."""
    _, local = make_tree(tmp_path, monkeypatch)
    alg1_low = config_of(UMAZE, "alg1", "0.001", "0.001")
    alg1_high = config_of(UMAZE, "alg1", "0.01", "10")
    alg22 = config_of(UMAZE, "alg2.2", "0.01", "3")
    base = baseline_of(UMAZE)
    write_records(local, alg1_low, [-710.0, -710.0])          # worse alg1 config
    write_records(local, alg1_high, [-696.0, -694.0])         # better alg1 config -> the row
    write_records(local, alg22, [-686.0, -684.0])             # above the -690 bar
    write_records(local, base, [-600.0])                      # 10M baseline
    write_decision(tmp_path, alg1_low, "pruned")
    write_decision(tmp_path, alg22, "survivor")
    by_key = env_metrics_tables.load_completed(SWEEP)
    verdicts = status_table.decision_verdicts(SWEEP)
    rows, awaiting = env_metrics_tables.env_rows(UMAZE, by_key, verdicts)

    # three arms have records; the two untouched arms await their first record
    assert awaiting == ["alg2.1", "alg2.3"]
    assert [r["label"] for r in rows] == [
        "baseline — lr 0.0001, bonus-weight 10000",
        "alg2.2 — lr 0.01, bonus-weight 3",
        "alg1 — lr 0.01, bonus-weight 10",
    ]
    assert [r["verdict"] for r in rows] == ["N/A", "survivor", "undecided"]
    assert [r["N"] for r in rows] == ["1", "2", "2"]
    assert rows[2]["reward"] == "-695.00 ± 1.00"              # the better alg1 config, mean ± se
    # marking: best bold, second best underlined, third unmarked
    mr.mark_rows(rows)
    assert rows[0]["reward"].startswith("**") and rows[1]["reward"].startswith("<u>")
    assert not rows[2]["reward"].startswith(("**", "<u>"))
    # bar line: only the alg2.2 config sits at or above the -690 bar, out of the env's 120 configs
    assert env_metrics_tables.bar_line(UMAZE, by_key) == (
        "frozen bar: -690.0000 (beta 10000); configs above bar so far: 1 of 120")


def test_env_metrics_edge_no_records_and_partial_record(tmp_path, monkeypatch):
    """Edge: an env with no completed record yields no rows, every arm awaiting, and a bar line with
    zero configs above the bar; a completed=false record never creates a row."""
    _, local = make_tree(tmp_path, monkeypatch)
    alg23 = config_of(MEDIUM, "alg2.3", "0.001", "1")
    write_records(local, alg23, [-900.0])
    partial = json.loads((local / f"{build_queue.label(alg23)}_s0.json").read_text())
    partial["completed"] = False
    (local / f"{build_queue.label(alg23)}_s0.json").write_text(json.dumps(partial))
    by_key = env_metrics_tables.load_completed(SWEEP)
    rows, awaiting = env_metrics_tables.env_rows(MEDIUM, by_key, {})
    assert rows == [] and awaiting == ["baseline", "alg1", "alg2.1", "alg2.2", "alg2.3"]
    assert env_metrics_tables.bar_line(MEDIUM, by_key) == (
        "frozen bar: -998.0000 (beta 3000); configs above bar so far: 0 of 120")


def test_compact_record_keeps_only_scalars(tmp_path, monkeypatch):
    """The loader is memory-bounded: a loaded record carries the four scalars only — never the
    per-episode history lists (task-R records are ~10x longer, which is why this matters)."""
    _, local = make_tree(tmp_path, monkeypatch)
    base = baseline_of(UMAZE)
    write_records(local, base, [-600.0])
    by_key = env_metrics_tables.load_completed(SWEEP)
    rec = by_key[build_queue.config_key(base)][0]
    assert set(rec) == {"key", "score", "completed", "total_timesteps"}
    assert rec["score"] == -600.0 and rec["total_timesteps"] == build_queue.STEPS_R


def test_report_writes_snapshot(tmp_path, monkeypatch):
    """End to end: the report script runs against the synthetic tree and writes ONE timestamped
    Markdown snapshot carrying every section."""
    q, local = make_tree(tmp_path, monkeypatch)
    alg22 = config_of(UMAZE, "alg2.2", "0.01", "3")
    write_markers(q, "pending_1m", alg22, range(3))
    write_records(local, alg22, [-686.0, -684.0])
    env = dict(os.environ, RUN_DIR_OVERRIDE=str(tmp_path))
    res = subprocess.run([sys.executable, os.path.join(HERE, "monitoring_report.py"),
                          "--sweep_id", SWEEP, "--snapshot"],
                         capture_output=True, text=True, env=env)
    assert res.returncode == 0, res.stderr
    out_files = os.listdir(tmp_path / "20_mins_monitoring" / "outputs")
    assert len(out_files) == 1 and out_files[0].endswith("_monitoring.md")
    text = (tmp_path / "20_mins_monitoring" / "outputs" / out_files[0]).read_text()
    for needle in ("## Running status", "## Per-node run counts", "## Per-env interim metrics",
                   "## Stage-1 decision summary", f"### {UMAZE}", "frozen bar: -690.0000",
                   "alg2.2 — lr 0.01, bonus-weight 3", "N/A"):
        assert needle in text, f"missing from the snapshot: {needle}"


def test_parse_worker_log(tmp_path):
    """parse_worker_log counts disjoint claim / done / failed lines and captures the done seconds
    (the run-8.1.2 worker prints the same line formats as run 8.1)."""
    # a log with 3 claims, 2 rc=0 finishes (45273s, 46247s), 1 rc=1 finish -> 2 done, 1 failed
    log = tmp_path / "r812-w30-1_9999999.log"
    log.write_text(
        "[2026-08-01T03:08:56 worker 9.1.2 sweep=s] claimed 00031_of_24200_x_seed0.json :: task=S\n"
        "[2026-08-01T03:08:57 worker 9.2.3 sweep=s] claimed 00032_of_24200_y_seed0.json :: task=S\n"
        "[2026-08-01T03:08:58 worker 9.3.4 sweep=s] claimed 24000_of_24200_z_seed0.json :: task=R\n"
        "[2026-08-01T15:43:35 worker 9.1.2 sweep=s] 00031_of_24200_x_seed0.json rc=0 in 45273s -> done "
        "(running totals: 1 done, 0 failed)\n"
        "[2026-08-01T15:59:43 worker 9.2.3 sweep=s] 00032_of_24200_y_seed0.json rc=0 in 46247s -> done "
        "(running totals: 2 done, 0 failed)\n"
        "[2026-08-01T06:36:50 worker 9.3.4 sweep=s] 24000_of_24200_z_seed0.json rc=1 in 12457s -> "
        "failed (running totals: 2 done, 1 failed)\n")
    claims, done, failed, fin = mr.parse_worker_log(str(log))
    assert (claims, done, failed) == (3, 2, 1)
    assert fin == [45273, 46247]
    # running_now = claims - done - failed = 3 - 2 - 1 = 0
    assert max(0, claims - done - failed) == 0


def test_mark_rows_ties_and_missing_values():
    """Edge: tied best values are all bolded (no underline for the tie), the next distinct value is
    underlined, and a row with no value for the column is skipped."""
    rows = [{"reward": "-679.99", "_raw": {"reward": -679.99}},
            {"reward": "-679.99", "_raw": {"reward": -679.994}},   # ties at 2 decimals
            {"reward": "-681.01", "_raw": {"reward": -681.01}},
            {"reward": "—", "_raw": {"reward": None}}]
    mr.mark_rows(rows)
    assert rows[0]["reward"] == "**-679.99**" and rows[1]["reward"] == "**-679.99**"
    assert rows[2]["reward"] == "<u>-681.01</u>"
    assert rows[3]["reward"] == "—"


if __name__ == "__main__":
    sys.exit(subprocess.call([sys.executable, "-m", "pytest", "-q", __file__]))
