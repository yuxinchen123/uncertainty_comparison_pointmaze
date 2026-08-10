"""Keep the sweep's pending queue stocked with runs a freed worker slot can actually finish.

The work queue claims at random among the 32 lowest-id pending markers, so it has no notion of how
far a run has got. Left alone, a slot freed by a four-day walltime kill is as likely to start a run
at zero as to resume one at 60% -- and a run resumed at 60% finishes inside one more segment while a
run started at zero cannot. This pass sorts the queue by that criterion instead.

It is idempotent and meant to run at the END OF EVERY REQUEUE, not once. Requeue is a continuing
process: each wave returns whatever markers its dead jobs held, so a one-shot filter is correct for
about twenty minutes and then quietly wrong. (Reported by the collaborator 2026-08-10 04:35, after
ten near-zero runs re-entered pending behind the first pass and left a freed slot with a 10-in-13
chance of starting one of them.)
"""

import argparse
import glob
import json
import os
import shutil

HOLD = "hold_fresh"


def resumable_steps(sweep_dir, run_id):
    """How many steps of this run would survive a requeue, by the checkpoint first.

    The checkpoint is the authority, not the record: a requeue archives the record out of
    `local/` and leaves the checkpoint behind, so a record-only reading of `local/` reports zero for
    exactly the runs worth protecting. The record is consulted only when no checkpoint exists, and
    archived records count -- a run requeued twice has its history under `killed_attempts_*`.
    """
    # before: run_id = 48 -> data/<sweep>/local/048_of_150.checkpoint.pt holding global_step 1.88e9
    # after:  1884815360
    for pattern in (f"{run_id:03d}_of_*.checkpoint.pt", f"{run_id}_of_*.checkpoint.pt"):
        for path in glob.glob(os.path.join(sweep_dir, "data", "*", "local", pattern)):
            try:
                import torch
                return int(torch.load(path, map_location="cpu", weights_only=False)["global_step"])
            except Exception:
                pass   # a checkpoint mid-write is unreadable; the records below still answer

    # No checkpoint: fall back to the furthest step any record for this run reached, live or archived.
    best = 0
    for path in glob.glob(os.path.join(sweep_dir, "data", "**", "*.json"), recursive=True):
        try:
            record = json.load(open(path))
        except Exception:
            continue
        if record.get("run_id") != run_id:
            continue
        history = record.get("train_history") or []
        best = max(best, history[-1]["step"] if history else 0)
    return best


def marker_run_id(path):
    """Read the run id out of a marker filename such as `048_of_150.json`."""
    return int(os.path.basename(path).split("_")[0])


def run_pass(sweep_dir, target_steps, threshold, pending_floor):
    """Move never-started runs out of pending, and release some back if the queue would run dry.

    Returns (held, released, pending_after) so the caller can log what moved.
    """
    queue = os.path.join(sweep_dir, "queue")
    os.makedirs(os.path.join(queue, HOLD), exist_ok=True)
    cut = threshold * target_steps

    # Classify every unclaimed marker, wherever it currently sits, so the pass is idempotent: a
    # marker that requeue put back into pending gets re-held, and one wrongly held gets released.
    unclaimed = {}
    for state in ("pending", HOLD):
        for path in glob.glob(os.path.join(queue, state, "*.json")):
            unclaimed[marker_run_id(path)] = (path, state)
    progress = {rid: resumable_steps(sweep_dir, rid) for rid in unclaimed}

    held, released = [], []
    for rid, (path, state) in sorted(unclaimed.items()):
        want = "pending" if progress[rid] >= cut else HOLD
        if want != state:
            shutil.move(path, os.path.join(queue, want, os.path.basename(path)))
            (held if want == HOLD else released).append(rid)

    # Starvation insurance. A slot that looks for work and finds none exits for the rest of its
    # job's four days -- about 1e9 steps of a GPU -- which costs far more than letting one
    # near-zero run start. So keep a floor in pending, releasing the most-progressed held runs first.
    pending = sorted(marker_run_id(p) for p in glob.glob(os.path.join(queue, "pending", "*.json")))
    holding = sorted(((progress[marker_run_id(p)], p)
                      for p in glob.glob(os.path.join(queue, HOLD, "*.json"))), reverse=True)
    while len(pending) < pending_floor and holding:
        _, path = holding.pop(0)
        shutil.move(path, os.path.join(queue, "pending", os.path.basename(path)))
        rid = marker_run_id(path)
        released.append(rid)
        pending.append(rid)
    return held, released, sorted(pending)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep_dir", required=True)
    parser.add_argument("--target_steps", type=int, default=2_000_000_000)
    parser.add_argument("--threshold", type=float, default=0.05,
                        help="a run below this fraction of target is held back")
    parser.add_argument("--pending_floor", type=int, default=8,
                        help="never leave fewer than this many claimable markers")
    args = parser.parse_args()

    held, released, pending = run_pass(args.sweep_dir, args.target_steps,
                                       args.threshold, args.pending_floor)
    print(f"held back {len(held)}: {held}")
    print(f"released {len(released)}: {released}")
    print(f"pending now {len(pending)}: {pending}")


if __name__ == "__main__":
    main()
