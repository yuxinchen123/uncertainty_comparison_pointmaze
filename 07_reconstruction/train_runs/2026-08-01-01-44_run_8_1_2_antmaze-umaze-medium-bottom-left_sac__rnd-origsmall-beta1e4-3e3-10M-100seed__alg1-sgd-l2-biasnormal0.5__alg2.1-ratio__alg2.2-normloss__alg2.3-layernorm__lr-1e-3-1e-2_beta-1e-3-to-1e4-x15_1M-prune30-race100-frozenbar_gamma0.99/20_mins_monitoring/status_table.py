#!/usr/bin/env python
"""Running-status table for the train run 8.1.2 monitoring loop (one of the two CLI table kinds of
the shared sweep-monitoring skill). Prints ONE table: one row per (env_setup, arm) cell — the 2 env
setups x 5 arms (baseline + alg1 + alg2.1 + alg2.2 + alg2.3) = 10 rows — plus a TOTAL row.
Repeatedly callable: reads the current queue / records / decisions each call and keeps no state.

Columns per cell:
- configs           : configs in the cell (30 per task-S arm = 2 learning rates x 15 bonus weights;
                      1 per baseline arm)
- undecided         : task-S configs with no stage-1 verdict yet. "N/A" on the baseline rows — the
                      task-R baseline arm is exempt from every decision, so it is never undecided.
- pruned            : configs the stage-1 controller pruned (verdict "pruned"); "N/A" for baseline
- survivors         : configs that reached the seed target unpruned (verdict "survivor"); "N/A" for
                      baseline
- pending           : queue markers still waiting — the SUM of pending_1m/ and pending_10m/ (task-S
                      arms fill pending_1m, the baseline arm fills pending_10m)
- running/done/failed : markers under queue/<sweep_id>/<state>/
- completed         : completed per-run JSON records in data/<sweep_id>/local/ grouped to the cell
                      via build_queue.key_from_record (its first two key fields are env_setup|arm)

Markers of a pruned config are archived under queue/<sweep_id>/pruned/ and are counted in NO state
column (they will never run); the pruned CONFIG count is the "pruned" column.

Usage:  python status_table.py --sweep_id <id> [--snapshot]
  --snapshot also writes the printed text to 20_mins_monitoring/outputs/<YYYY-MM-DD-HH-MM>_status.txt

RUN_DIR_OVERRIDE (env var, testing only): when set, queue/, data/, slurm/ (the decisions log and
FROZEN_BARS.json) and outputs/ are read/written under that directory instead of the real run folder.
build_queue is ALWAYS imported from the real run folder's slurm/ (it is the config source of truth);
only the per-sweep data location is redirected.
"""
import argparse
import glob
import json
import os
import re
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REAL_RUN_DIR = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(REAL_RUN_DIR, "slurm"))
import build_queue  # noqa: E402  (CONFIGS / BASELINE_CONFIGS / config_key / label / key_from_record)

NA = "N/A"
# the two pending pools, summed into the single "pending" column, and the other marker states
POOL_DIRS = ["pending_1m", "pending_10m"]
STATE_DIRS = ["running", "done", "failed"]
# filename convention: {id}_of_{total}_{label}_seed{seed}.json ; capture the label between them
NAME_RE = re.compile(r"^\d+_of_\d+_(.+)_seed\d+\.json$")


def run_dir():
    """The run folder whose queue/, data/, slurm/ and outputs/ this table reads (RUN_DIR_OVERRIDE
    redirects it for tests; read per call so a test can set it after import)."""
    return os.environ.get("RUN_DIR_OVERRIDE", REAL_RUN_DIR)


def ordered_cells():
    """The 10 (env_setup, arm) cells in report order: per env setup the baseline arm first, then the
    four task-S arms in build_queue.ARMS order."""
    cells = []
    for env_setup in build_queue.ENV_SETUPS_RUN12:
        cells.append((env_setup, "baseline"))
        for arm in build_queue.ARMS:
            cells.append((env_setup, arm))
    return cells


def cell_index():
    """Return (cell_config_keys, label_to_cell): every cell's config_keys and the reverse map from a
    queue-filename label to its cell, over BOTH task lists (task-S CONFIGS + task-R BASELINE_CONFIGS)."""
    cell_config_keys = {cell: [] for cell in ordered_cells()}
    label_to_cell = {}
    for spec in list(build_queue.CONFIGS) + list(build_queue.BASELINE_CONFIGS):
        cell = (spec["env_setup"], spec["arm"])
        cell_config_keys[cell].append(build_queue.config_key(spec))
        label_to_cell[build_queue.label(spec)] = cell
    return cell_config_keys, label_to_cell


def queue_counts(sweep_id, label_to_cell):
    """Per-cell marker counts {pending, running, done, failed}, matched to a cell by the label
    embedded in each filename. `pending` sums BOTH pending pools; the archived pruned/ pool is not
    counted (those markers will never run)."""
    counts = {cell: {"pending": 0, "running": 0, "done": 0, "failed": 0} for cell in ordered_cells()}
    # (directory on disk -> the column it feeds): the two pending pools both feed "pending"
    dirs = [(p, "pending") for p in POOL_DIRS] + [(s, s) for s in STATE_DIRS]
    for sub, column in dirs:
        d = os.path.join(run_dir(), "queue", sweep_id, sub)
        if not os.path.isdir(d):
            continue
        for name in os.listdir(d):
            m = NAME_RE.match(name)
            if not m:
                continue
            cell = label_to_cell.get(m.group(1))
            if cell is not None:
                counts[cell][column] += 1
    return counts


def completed_counts(sweep_id):
    """Per-cell count of completed per-run JSON records in data/<sweep_id>/local/, grouped to a cell
    via build_queue.key_from_record (first two key fields = env_setup|arm). A record whose
    "completed" field is absent counts as complete (the run-id-and-logging convention: a missing flag
    is an old write-once record); a completed=false checkpoint of a killed attempt does not count."""
    counts = {cell: 0 for cell in ordered_cells()}
    local = os.path.join(run_dir(), "data", sweep_id, "local")
    for path in glob.glob(os.path.join(local, "*.json")):
        try:
            with open(path) as fh:
                d = json.load(fh)
        except (json.JSONDecodeError, OSError):
            continue  # a record mid-flush this cycle; picked up next tick
        if not d.get("completed", True):
            continue
        cell = tuple(build_queue.key_from_record(d).split("|")[:2])   # (env_setup, arm)
        if cell in counts:
            counts[cell] += 1
    return counts


def decision_verdicts(sweep_id):
    """{config_key: verdict} from slurm/stage1_decisions_<sweep_id>.jsonl (empty when the controller
    has not decided anything yet). Shared with env_metrics_tables so both tables read one log."""
    verdicts = {}
    path = os.path.join(run_dir(), "slurm", f"stage1_decisions_{sweep_id}.jsonl")
    if os.path.exists(path):
        with open(path) as fh:
            for line in fh:
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if d.get("verdict") in ("pruned", "survivor"):
                    verdicts[d["config_key"]] = d["verdict"]
    return verdicts


def build_rows(sweep_id):
    """Assemble the 10 per-cell row dicts plus a TOTAL row. Baseline cells carry "N/A" in the three
    decision columns; the TOTAL row sums only the numeric cells."""
    cell_config_keys, label_to_cell = cell_index()
    qc = queue_counts(sweep_id, label_to_cell)
    cc = completed_counts(sweep_id)
    verdicts = decision_verdicts(sweep_id)
    rows = []
    for cell in ordered_cells():
        env_setup, arm = cell
        keys = cell_config_keys[cell]
        n_pruned = sum(1 for k in keys if verdicts.get(k) == "pruned")
        n_surv = sum(1 for k in keys if verdicts.get(k) == "survivor")
        # the baseline arm is exempt from stage-1 decisions -> N/A, never a count and never a dash
        decided_cols = ({"undecided": NA, "pruned": NA, "survivors": NA} if arm == "baseline" else
                        {"undecided": len(keys) - n_pruned - n_surv, "pruned": n_pruned,
                         "survivors": n_surv})
        rows.append(dict({
            "env": env_setup.replace("_start_bottom_left", ""),
            "arm": arm,
            "cfgs": len(keys),
            "pending": qc[cell]["pending"],
            "running": qc[cell]["running"],
            "done": qc[cell]["done"],
            "failed": qc[cell]["failed"],
            "compl": cc[cell],
        }, **decided_cols))
    # TOTAL row: sum each numeric column, skipping the baseline rows' "N/A" decision cells
    total = {"env": "TOTAL", "arm": ""}
    for col in ("cfgs", "undecided", "pruned", "survivors", "pending", "running", "done", "failed",
                "compl"):
        total[col] = sum(r[col] for r in rows if r[col] != NA)
    return rows, total


COLS = [("env", "env", "l"), ("arm", "arm", "l"), ("cfgs", "configs", "r"),
        ("undecided", "undecided", "r"), ("pruned", "pruned", "r"),
        ("survivors", "survivors", "r"), ("pending", "pending", "r"),
        ("running", "running", "r"), ("done", "done", "r"), ("failed", "failed", "r"),
        ("compl", "completed", "r")]


def render(sweep_id, rows, total):
    """Format the rows + TOTAL into one aligned plain-text table (string)."""
    # column width = max over the header and every cell (TOTAL included)
    widths = {}
    for key, header, _ in COLS:
        cells = [header] + [str(r[key]) for r in rows] + [str(total[key])]
        widths[key] = max(len(x) for x in cells)

    def fmt_row(r):
        # left-justify text columns, right-justify numeric columns
        parts = []
        for key, _, align in COLS:
            s = str(r[key])
            parts.append(s.ljust(widths[key]) if align == "l" else s.rjust(widths[key]))
        return "  ".join(parts)

    header = fmt_row({k: h for k, h, _ in COLS})
    sep = "  ".join("-" * widths[k] for k, _, _ in COLS)
    lines = [
        f"# run 8.1.2 running-status  sweep_id={sweep_id}  {time.strftime('%Y-%m-%dT%H:%M:%S')}",
        header, sep,
    ]
    lines += [fmt_row(r) for r in rows]
    lines.append(sep)
    lines.append(fmt_row(total))
    return "\n".join(lines)


def main():
    """Print the running-status table; with --snapshot also save a timestamped copy."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep_id", required=True)
    ap.add_argument("--snapshot", action="store_true")
    args = ap.parse_args()
    rows, total = build_rows(args.sweep_id)
    text = render(args.sweep_id, rows, total)
    print(text)
    if args.snapshot:
        out_dir = os.path.join(run_dir(), "20_mins_monitoring", "outputs")
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, f"{time.strftime('%Y-%m-%d-%H-%M')}_status.txt")
        with open(path, "w") as fh:
            fh.write(text + "\n")
        print(f"\n[snapshot] {path}")


if __name__ == "__main__":
    main()
