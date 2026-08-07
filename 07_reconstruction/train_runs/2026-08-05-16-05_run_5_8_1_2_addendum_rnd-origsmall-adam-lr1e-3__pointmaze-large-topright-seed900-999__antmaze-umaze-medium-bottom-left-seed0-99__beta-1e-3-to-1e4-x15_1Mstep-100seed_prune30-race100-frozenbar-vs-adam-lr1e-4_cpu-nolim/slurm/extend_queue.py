#!/usr/bin/env python
"""Grow the LIVE work queue to the configurations and seeds build_queue.py now describes.

Written 2026-08-06 to make two changes to a sweep that was already running with ~926 workers, with
nothing stopped and no completed run repeated:
  1. the seed target per configuration goes from 100 to 300;
  2. a second predictor learning rate, Adam 1e-2, joins the same sweep as 45 more configurations.

WHY THIS WORKS WITHOUT STOPPING ANYTHING. A worker calls claim() in a loop, and claim() re-lists
queue/<sweep_id>/pending/ every time. A marker dropped into pending/ is therefore picked up by the
workers that are already running, with no restart and no code reaching them. The only thing this
script must not do is disturb work that is already claimed or finished, so it only ever:
  - CREATES a pending marker for a (configuration, seed) the queue has never held, and
  - RENAMES a marker that is still in pending/, which no worker has claimed.
Markers in running/, done/, failed/ and pruned/ are never touched, so a completed run is never
repeated and an in-flight run is never disturbed.

WHY THE RENAMES ARE NEEDED. claim() takes the 32 lexically smallest names in pending/ and picks one
at random, which is a seed-ordered window ONLY because the ids are zero-padded to a fixed width.
The old queue padded to 4 digits (run_total 4,500); the extended one pads to 5 (run_total 27,000),
and "00000..." sorts before "1676...". Leaving the two widths mixed would put every new marker ahead
of every old one and starve the Adam 1e-3 arm, so every surviving pending marker is renamed into the
new scheme. After the rename the whole queue is one width again, lexical order is numeric order
again, and the two arms advance together seed by seed: the new arm's low seeds are the lexically
smallest, so it catches up to the running arm's frontier first and then they interleave.

RENAME SAFETY. A pending marker is moved OUT of pending/ into staging/, rewritten there, and moved
back under its new name — so at no instant does the same (configuration, seed) exist under two names
in pending/, which is the only way a unit could be claimed twice. Markers are processed one at a
time, so pending/ never dips by more than one and no worker can see an empty queue and exit. If this
script dies mid-rename the marker is left in staging/ and the next invocation returns it to pending/
before doing anything else, so a crash costs nothing and re-running is always the repair.

IDEMPOTENT. Re-running finds every unit present and every pending name already correct, and reports
0 created / 0 renamed. That is also how it is tested.

Usage:  python extend_queue.py --sweep_id <id> [--apply]      (default is a dry run)
"""
import argparse
import getpass
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
RUN_DIR = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import build_queue as bq   # noqa: E402  (CONFIGS / config_key / label / a_seed_of / RUN_TOTAL)

POOLS = ("pending", "running", "done", "failed", "pruned")
STAGING = "staging"   # a marker lives here only during its own rename, never between invocations


def marker_unit(path):
    """The (config_key, seed_index) a marker file stands for, or None when it cannot be read.

    Read from the marker's JSON, never from its filename, so a marker written under either id scheme
    identifies the same unit of work.
    """
    try:
        with open(path) as fh:
            d = json.load(fh)
    except (json.JSONDecodeError, OSError):
        return None
    if "config_key" not in d or "seed_index" not in d:
        return None
    return (d["config_key"], int(d["seed_index"]))


def scan_queue(queue_dir):
    """{(config_key, seed_index): (pool, filename)} over every marker currently in the queue.

    before: queue/<id>/{pending,running,done,...}/ holding 4,500 markers of one learning rate
    after:  {("...|lr0.001|b300", 37): ("pending", "1676_of_4500_..._seed937.json"), ...}
    """
    units = {}
    for pool in POOLS:
        d = os.path.join(queue_dir, pool)
        if not os.path.isdir(d):
            continue
        for name in os.listdir(d):
            if not name.endswith(".json"):
                continue
            unit = marker_unit(os.path.join(d, name))
            if unit is not None:
                units[unit] = (pool, name)
    return units


def drain_staging(queue_dir):
    """Return any marker stranded in staging/ by a crashed run to pending/. Returns the count."""
    staging = os.path.join(queue_dir, STAGING)
    if not os.path.isdir(staging):
        return 0
    n = 0
    for name in os.listdir(staging):
        os.rename(os.path.join(staging, name), os.path.join(queue_dir, "pending", name))
        n += 1
    return n


def target_units():
    """{(config_key, seed_index): (marker_name, marker_json)} for the full extended sweep.

    One entry per (configuration, seed) the sweep should hold: 90 configurations x 300 seeds.
    """
    width = len(str(bq.RUN_TOTAL))
    out = {}
    for seed_index in bq.SEED_INDICES:
        for cfg_index, cfg_spec in enumerate(bq.CONFIGS):
            # seed index OUTERMOST, configuration inner: index i owns ids [90*i .. 90*i+89]
            run_id = seed_index * len(bq.CONFIGS) + cfg_index
            seed = bq.a_seed_of(cfg_spec["env_setup"], seed_index)
            key = bq.config_key(cfg_spec)
            cfg = {
                "sweep_id": None,   # filled in by the caller, which knows the sweep id
                "run_id": run_id, "run_total": bq.RUN_TOTAL, "pool": "pending",
                "env_setup": cfg_spec["env_setup"], "algorithm": cfg_spec["algorithm"],
                "arm": cfg_spec["arm"], "beta": cfg_spec["beta"],
                "a_seed": seed, "seed_index": seed_index,
                "config_key": key,
                "params": cfg_spec["params"],
                "fixed": dict(bq.FIXED_COMMON),
            }
            name = (f"{run_id:0{width}d}_of_{bq.RUN_TOTAL}_"
                    f"{bq.label(cfg_spec)}_seed{seed}.json")
            out[(key, seed_index)] = (name, cfg)
    return out


def create_marker(queue_dir, name, cfg):
    """Write one new pending marker, via a temp file so a worker never reads a half-written JSON."""
    pending = os.path.join(queue_dir, "pending")
    tmp = os.path.join(queue_dir, STAGING, name)
    with open(tmp, "w") as fh:
        json.dump(cfg, fh)
    os.rename(tmp, os.path.join(pending, name))


def rename_marker(queue_dir, old_name, new_name, cfg):
    """Rename one PENDING marker into the new id scheme, rewriting its run_id/run_total to match.

    Returns True when the marker was renamed, False when a worker claimed it first (in which case it
    runs to completion under its old identity, which is correct and needs no repair).
    """
    pending = os.path.join(queue_dir, "pending")
    staged = os.path.join(queue_dir, STAGING, new_name)
    # move OUT of pending first: after this the unit exists under exactly one name and no worker can
    # claim it, which is what makes a double claim impossible
    try:
        os.rename(os.path.join(pending, old_name), staged)
    except OSError:
        return False   # claimed by a worker in this instant; leave it alone
    with open(staged, "w") as fh:
        json.dump(cfg, fh)
    os.rename(staged, os.path.join(pending, new_name))
    return True


def extend(sweep_id, apply_changes):
    """Bring the live queue up to the full extended sweep. Returns a counts dict."""
    queue_dir = os.path.join(RUN_DIR, "queue", sweep_id)
    for sub in POOLS + (STAGING,):
        os.makedirs(os.path.join(queue_dir, sub), exist_ok=True)
    drained = drain_staging(queue_dir) if apply_changes else 0

    have = scan_queue(queue_dir)
    want = target_units()
    counts = {"already": 0, "created": 0, "renamed": 0, "claimed_mid_rename": 0,
              "untouched_non_pending": 0, "staging_drained": drained,
              "unknown_in_queue": len(set(have) - set(want))}

    # A unit already in running/done/failed/pruned is finished or in flight: never touch it. A unit
    # in pending/ is unclaimed, so it is safe to rename into the new id scheme. A unit the queue has
    # never held is new work.
    for unit, (name, cfg) in sorted(want.items(), key=lambda kv: kv[1][0]):
        cfg = dict(cfg, sweep_id=sweep_id)
        if unit not in have:
            if apply_changes:
                create_marker(queue_dir, name, cfg)
            counts["created"] += 1
            continue
        pool, old_name = have[unit]
        if pool != "pending":
            counts["untouched_non_pending"] += 1
        elif old_name == name:
            counts["already"] += 1
        elif apply_changes:
            if rename_marker(queue_dir, old_name, name, cfg):
                counts["renamed"] += 1
            else:
                counts["claimed_mid_rename"] += 1
        else:
            counts["renamed"] += 1
    return counts


def main():
    if getpass.getuser() != "sl5nw":
        sys.exit("owner-only script; collaborators never build or extend the queue")
    p = argparse.ArgumentParser()
    p.add_argument("--sweep_id", required=True)
    p.add_argument("--apply", action="store_true",
                   help="actually change the queue (default: report what would change)")
    args = p.parse_args()
    counts = extend(args.sweep_id, args.apply)
    mode = "APPLIED" if args.apply else "DRY RUN (nothing changed)"
    print(f"[extend_queue] {mode} sweep={args.sweep_id} "
          f"target={len(bq.CONFIGS)} configurations x {len(bq.SEED_INDICES)} seeds "
          f"= {bq.RUN_TOTAL} runs")
    for k in ("already", "created", "renamed", "claimed_mid_rename", "untouched_non_pending",
              "staging_drained", "unknown_in_queue"):
        print(f"[extend_queue]   {k:24s} {counts[k]}")
    if counts["unknown_in_queue"]:
        print(f"[extend_queue] WARNING: {counts['unknown_in_queue']} markers in the queue do not "
              f"belong to the extended sweep; they were left alone")


if __name__ == "__main__":
    main()
