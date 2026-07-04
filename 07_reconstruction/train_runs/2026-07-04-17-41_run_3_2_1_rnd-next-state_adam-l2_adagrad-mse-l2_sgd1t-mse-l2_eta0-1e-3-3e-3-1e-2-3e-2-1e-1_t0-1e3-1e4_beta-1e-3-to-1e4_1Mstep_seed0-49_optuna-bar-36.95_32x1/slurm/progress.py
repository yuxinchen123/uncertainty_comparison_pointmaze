#!/usr/bin/env python
"""Print the run-3.2.1 progress snapshot without touching the optuna journal (safe to run anywhere).

Decided configurations are read from optuna/decisions.jsonl (the controller's manifest), queue and
result counts from the filesystem — so this works even while the controller holds the journal lock.

Usage:  python progress.py --sweep_id <id>
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import optuna_controller as ctl  # reuses scan_queue / scan_results / write_progress


def main():
    """Assemble the same snapshot the controller writes, from the manifest + filesystem only."""
    p = argparse.ArgumentParser()
    p.add_argument("--sweep_id", required=True)
    args = p.parse_args()
    # decided keys from the manifest (stop/complete/exhausted lines each end one configuration)
    decided = set()
    if os.path.exists(ctl.DECISIONS):
        with open(ctl.DECISIONS) as fh:
            for line in fh:
                entry = json.loads(line)
                if entry.get("action") in ("stop", "complete", "exhausted"):
                    decided.add(entry["config_key"])
    # frozen bar if the controller froze it already, else the 36.95 reference
    bar = ctl.BAR_REFERENCE
    if os.path.exists(ctl.FROZEN_BAR):
        with open(ctl.FROZEN_BAR) as fh:
            bar = json.load(fh)["bar"]
    queue_counts = ctl.scan_queue(args.sweep_id)
    rewards_by_key, n_partial, _ = ctl.scan_results(args.sweep_id)
    lines = ctl.write_progress(args.sweep_id, rewards_by_key, queue_counts, decided, bar, n_partial)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
