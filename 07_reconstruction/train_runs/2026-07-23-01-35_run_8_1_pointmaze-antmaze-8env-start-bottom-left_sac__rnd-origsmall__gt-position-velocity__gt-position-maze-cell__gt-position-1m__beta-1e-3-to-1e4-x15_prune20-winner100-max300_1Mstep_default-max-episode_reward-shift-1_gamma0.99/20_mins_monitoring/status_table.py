#!/usr/bin/env python
"""Running-status table for the point maze + ant maze train run 1 monitoring loop (one of the two
CLI table kinds of the shared sweep-monitoring skill). Prints ONE table: one row per
(env_setup, algorithm) racing cell (28 rows, in build_queue.CONFIGS order), a totals row at the
bottom. Repeatedly callable — reads the current queue/data/decisions each call, keeps no state.

Columns (per cell):
- configs_total       : configs in the cell (1 for a SAC cell, 15 for a bonus cell)
- configs_racing      : configs with no prune / winner-only decision yet
- configs_pruned      : configs the prune controller pruned (verdict "pruned")
- configs_winner_cut  : configs cut at winner-only continuation (verdict "winner_only")
- pending/running/done/failed : queue markers under queue/<sid>/<state>/ whose embedded label maps
                                to this cell (summed over the cell's configs)
- completed_records   : completed per-run JSONs in data/<sid>/local/ grouped to the cell via
                        prune_controller.key_from_record

Usage:  python status_table.py --sweep_id <id> [--snapshot]
  --snapshot also writes the printed text to 20_mins_monitoring/outputs/<YYYY-MM-DD-HH-MM>_status.txt

RUN_DIR_OVERRIDE (env var, testing only): when set, the queue/, data/, slurm/prune_decisions and
outputs/ are read/written under that directory instead of the real run folder. build_queue and
prune_controller are ALWAYS imported from the real run folder's slurm/ (they are the config source of
truth); only the per-sweep data location is redirected. Used by the machinery_test harness.
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
import build_queue        # noqa: E402  (CONFIGS / config_key / label — the config source of truth)
import prune_controller   # noqa: E402  (key_from_record — the record->cell grouping)

# where per-sweep queue/data/decisions live: the real run folder, or a test override
DATA_RUN_DIR = os.environ.get("RUN_DIR_OVERRIDE", REAL_RUN_DIR)

STATES = ["pending", "running", "done", "failed"]
# filename convention: {id}_of_{total}_{label}_seed{seed}.json ; capture the label between them
NAME_RE = re.compile(r"^\d+_of_\d+_(.+)_seed\d+\.json$")


def ordered_cells():
    """The 28 (env_setup, algorithm) cells in build_queue.CONFIGS first-seen order."""
    cells, seen = [], set()
    for c in build_queue.CONFIGS:
        cell = (c["env_setup"], c["algorithm"])
        if cell not in seen:
            seen.add(cell)
            cells.append(cell)
    return cells


def cell_index():
    """Return (cell_config_keys, label_to_cell): the config_keys of each cell and the reverse map
    from a queue-filename label to its cell."""
    cell_config_keys, label_to_cell = {}, {}
    for c in build_queue.CONFIGS:
        cell = (c["env_setup"], c["algorithm"])
        cell_config_keys.setdefault(cell, []).append(build_queue.config_key(c))
        label_to_cell[build_queue.label(c)] = cell
    return cell_config_keys, label_to_cell


def queue_counts(sweep_id, label_to_cell):
    """Per-cell counts of pending/running/done/failed queue markers, matched to a cell by the label
    embedded in each filename."""
    counts = {cell: {s: 0 for s in STATES} for cell in ordered_cells()}
    for state in STATES:
        d = os.path.join(DATA_RUN_DIR, "queue", sweep_id, state)
        if not os.path.isdir(d):
            continue
        for name in os.listdir(d):
            m = NAME_RE.match(name)
            if not m:
                continue
            cell = label_to_cell.get(m.group(1))
            if cell is not None:
                counts[cell][state] += 1
    return counts


def completed_counts(sweep_id):
    """Per-cell count of completed per-run JSON records in data/<sid>/local/, grouped to a cell via
    prune_controller.key_from_record (its first two fields are env_setup|algorithm). A record whose
    "completed" field is absent counts as complete, matching prune_controller.load_scores and the
    run-id-and-logging convention (a missing flag is an old write-once record); a completed=false
    checkpoint of a killed attempt does not count."""
    counts = {cell: 0 for cell in ordered_cells()}
    local = os.path.join(DATA_RUN_DIR, "data", sweep_id, "local")
    for path in glob.glob(os.path.join(local, "*.json")):
        try:
            with open(path) as fh:
                d = json.load(fh)
        except (json.JSONDecodeError, OSError):
            continue  # mid-flush this cycle; picked up next tick
        if not d.get("completed", True):
            continue
        key = prune_controller.key_from_record(d)          # env_setup|algorithm|beta
        cell = tuple(key.split("|")[:2])                   # (env_setup, algorithm)
        if cell in counts:
            counts[cell] += 1
    return counts


def decision_sets(sweep_id):
    """({pruned config_keys}, {winner_only config_keys}) from slurm/prune_decisions_<sid>.jsonl."""
    pruned, winner = set(), set()
    path = os.path.join(DATA_RUN_DIR, "slurm", f"prune_decisions_{sweep_id}.jsonl")
    if os.path.exists(path):
        with open(path) as fh:
            for line in fh:
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if d.get("verdict") == "pruned":
                    pruned.add(d.get("config_key"))
                elif d.get("verdict") == "winner_only":
                    winner.add(d.get("config_key"))
    return pruned, winner


def build_rows(sweep_id):
    """Assemble the per-cell row dicts (28 rows) plus a totals row."""
    cell_config_keys, label_to_cell = cell_index()
    qc = queue_counts(sweep_id, label_to_cell)
    cc = completed_counts(sweep_id)
    pruned_set, winner_set = decision_sets(sweep_id)
    rows = []
    for env_setup, algorithm in ordered_cells():
        cell = (env_setup, algorithm)
        keys = cell_config_keys[cell]
        n_pruned = sum(1 for k in keys if k in pruned_set)
        n_winner = sum(1 for k in keys if k in winner_set)
        rows.append({
            "env": env_setup.replace("_start_bottom_left", ""),
            "algorithm": algorithm,
            "cfgs": len(keys),
            "racing": len(keys) - n_pruned - n_winner,
            "pruned": n_pruned,
            "wcut": n_winner,
            "pending": qc[cell]["pending"],
            "running": qc[cell]["running"],
            "done": qc[cell]["done"],
            "failed": qc[cell]["failed"],
            "compl": cc[cell],
        })
    # totals row (sum every numeric column; env/algorithm carry the TOTAL label)
    total = {"env": "TOTAL", "algorithm": ""}
    for col in ("cfgs", "racing", "pruned", "wcut", "pending", "running", "done", "failed", "compl"):
        total[col] = sum(r[col] for r in rows)
    return rows, total


COLS = [("env", "env", "l"), ("algorithm", "algorithm", "l"), ("cfgs", "cfgs", "r"),
        ("racing", "racing", "r"), ("pruned", "pruned", "r"), ("wcut", "winner_cut", "r"),
        ("pending", "pending", "r"), ("running", "running", "r"), ("done", "done", "r"),
        ("failed", "failed", "r"), ("compl", "completed", "r")]


def render(sweep_id, rows, total):
    """Format the rows + totals into one aligned plain-text table (string)."""
    # column width = max over header and every cell (totals included)
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
        f"# run 8.1 running-status  sweep_id={sweep_id}  {time.strftime('%Y-%m-%dT%H:%M:%S')}",
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
        out_dir = os.path.join(DATA_RUN_DIR, "20_mins_monitoring", "outputs")
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, f"{time.strftime('%Y-%m-%d-%H-%M')}_status.txt")
        with open(path, "w") as fh:
            fh.write(text + "\n")
        print(f"\n[snapshot] {path}")


if __name__ == "__main__":
    main()
