#!/usr/bin/env python
"""Turn the applied, record-identical switches on for a work-queue sweep.

The switches (`opt_polyak_foreach`, `opt_torch_reward`) already ship in `train.py`; a run uses them
only if its queue marker asks for them, because the worker builds the trainer's argv from the
marker's `fixed` dict. This script adds them there.

Safe to run on a LIVE sweep, by construction:

- it rewrites each marker **in place through the same file descriptor** (`open(path, "r+")`), so it
  never creates a directory entry. A worker claims a run with `os.rename(pending -> running)`; since
  no new entry is ever made, this script can never resurrect a claimed marker into `pending/` and
  can never cause the same run to execute twice — the failure mode that a tmp+rename edit would
  have;
- it skips anything that vanished (already claimed) between listing and opening;
- the whole record is written in a single `write()` of a small JSON;
- and above all, the flags it adds are **record-identical**: a run started before the edit and a run
  started after produce the same numbers, so a sweep may be half-flagged without becoming
  non-uniform. That is the property `run_conditions.py` verifies, and the only reason a live edit is
  acceptable at all.

Usage:
  python apply_optflags.py --queue <run>/queue/<sweep_id>            # dry run: report only
  python apply_optflags.py --queue <run>/queue/<sweep_id> --apply    # write the flags
  python apply_optflags.py --queue ... --pools pending,failed        # which pools to touch
"""
import argparse
import json
import os

FLAGS = {"opt_polyak_foreach": "True", "opt_torch_reward": "True"}


def update_marker(path, apply):
    """Add the flags to one marker's `fixed` dict. Returns "added" | "already" | "gone".
    before: fixed = {"eval_freq": 50000, ..., "total_timesteps": 4000000}
    after:  fixed = {..., "total_timesteps": 4000000, "opt_polyak_foreach": "True",
                     "opt_torch_reward": "True"}"""
    try:
        with open(path, "r+") as fh:
            cfg = json.load(fh)
            fixed = cfg.setdefault("fixed", {})
            if all(fixed.get(k) == v for k, v in FLAGS.items()):
                return "already"
            if not apply:
                return "added"
            fixed.update(FLAGS)
            blob = json.dumps(cfg)
            fh.seek(0)
            fh.write(blob)       # one write, same inode: a concurrent claim just moves the entry
            fh.truncate()
            return "added"
    except FileNotFoundError:
        return "gone"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--queue", required=True, help="the sweep's queue dir (holding pending/ running/ ...)")
    p.add_argument("--pools", default="pending", help="comma-separated pools to update")
    p.add_argument("--apply", action="store_true", help="write; without it, only report")
    args = p.parse_args()

    totals = {"added": 0, "already": 0, "gone": 0}
    for pool in args.pools.split(","):
        d = os.path.join(args.queue, pool.strip())
        if not os.path.isdir(d):
            continue
        for name in sorted(os.listdir(d)):
            if name.endswith(".json"):
                totals[update_marker(os.path.join(d, name), args.apply)] += 1
        print(f"[{pool}] {totals}")
    verb = "updated" if args.apply else "would update"
    print(f"{verb} {totals['added']} marker(s); {totals['already']} already had the flags; "
          f"{totals['gone']} were claimed while scanning")
    if not args.apply:
        print("dry run — pass --apply to write")


if __name__ == "__main__":
    main()
