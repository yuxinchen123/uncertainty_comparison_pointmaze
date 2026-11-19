"""Switch MIT of the stage-1 run from four-copy packs to eight-copy packs, in one hold of the remote fleet's lock.

    python3 switch_mit_to_eight_copies.py --run-dir RUN [--cancel JOB ...] [--remove-pool POOL ...] [--dry-run]

User decision of 2026-11-18 about 21:50 PT ("8 montezuma per h200"). In one hold of RUN/slurm/remote_fleet.lock, so
no hand-over tick runs between the steps:
  1. the server table: MIT's pool advanced-h200 and its held-back pool preemptable-h200 run the "eight copies"
     profile (hand_over then gives MIT eight-copy packs only), and the named check pools are removed;
  2. the four-copy packs that wait on MIT (no job has claimed them) come back to UVA (the grid campaign's
     take-back) and are taken apart into their singles (build_units.unpack_pending, by the pack's exact name);
     a pack a job has claimed on MIT is not touched, and finishes there;
  3. the named MIT jobs are cancelled, each only when it is in this run's MIT ledger, carries this run's job name
     prefix and still waits (PENDING): the copy-2 Advanced worker jobs, whose worker has no eight-copy profile, and
     check jobs that still wait.
Standard library only; it runs on the UVA login node beside the hand-over loop, through the shared MIT connection.
"""
import argparse
import json
from pathlib import Path
import shlex
import sys

sys.path.insert(0, "/p/rlprojects/RND/10_jax_exploration_platform/scripts/pixel_sweep_campaign")
from campaign_imports import common, grid, own  # noqa: E402

remote = own("remote_fleet")
fleet = grid("remote_fleet")
build_units = own("build_units")
PACK_WORD = "four-copies"
EIGHT = "eight copies"


def switch_table(table, pools_to_remove):
    """Set MIT's two worker pools to the eight-copy profile and drop the check pools; returns what changed."""
    # before: pools {"advanced-h200": {"execution_profile": "four copies", ...}, "eight-copy-check": {...}},
    #         pools_held_back {"preemptable-h200": {"execution_profile": "four copies", ...}, "note": "..."}
    # after:  pools {"advanced-h200": {"execution_profile": "eight copies", ...}},
    #         pools_held_back {"preemptable-h200": {"execution_profile": "eight copies", ...}, "note": "..."}
    mit = table["mit"]
    changed = []
    # held_back_pools gives the table's own pool records (without the free-text note), so a change goes into the table
    for holder in (mit["pools"], remote.held_back_pools(mit)):
        for name in ("advanced-h200", "preemptable-h200"):
            if name in holder and holder[name]["execution_profile"] != EIGHT:
                holder[name]["execution_profile"] = EIGHT
                changed.append(f"{name}: eight copies")
    for name in pools_to_remove:
        if mit["pools"].pop(name, None) is not None:
            changed.append(f"{name}: removed")
    mit["hand_over"]["note"] = (
        "Eight-copy packs only since 2026-11-18 PT (user decision of about 21:50 PT, \"8 montezuma per h200\"; "
        "switch_mit_to_eight_copies.py). An H200's worker of code copy 3 also continues four-copy packs that "
        "started there. hand_over counts every eight-copy unit the server holds against a pool's limit, so the "
        "server-wide limit stands in the first pool (2 packs per card; mit-share of the hand-over loop sets it).")
    return changed


def cancel_script(jobs):
    """The bash lines that cancel each named job only while it waits, and say which were cancelled."""
    lines = [f"for id in {' '.join(shlex.quote(job) for job in jobs)}; do",
             '  state=$(squeue -h -j "$id" -o %T 2> /dev/null)',
             f'  if [ "$state" = PENDING ] && scancel "$id"; then echo "{fleet.MARK}cancelled $id"; '
             f'else echo "{fleet.MARK}left $id ${{state:-ended}}"; fi', "done", f'echo "{fleet.MARK}end"', "exit 0"]
    return "\n".join(lines) + "\n"


def main(argv=None):
    """Run the three steps under the fleet lock and print what each did."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--cancel", action="append", default=[], help="an MIT job id of this run to cancel if it waits")
    parser.add_argument("--remove-pool", action="append", default=[], help="a check pool of MIT to remove")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    run_dir = args.run_dir.resolve()
    remote.adapt(run_dir)
    with fleet.fleet_lock(run_dir):
        table = common.read_json(fleet.table_path(run_dir))
        # every job to cancel must be this run's: in its MIT ledger, with its job name prefix, waiting
        assigned, status = fleet.read_status(run_dir, table, "mit", args.dry_run)
        ledger = set(fleet.ledger_ids(run_dir, "mit"))
        prefix = f"{remote.job_prefix(run_dir)}-mit-"
        for job in args.cancel:
            row = status["jobs"].get(job)
            if job not in ledger or row is None or not row["name"].startswith(prefix):
                raise SystemExit(f"job {job} is not a live job of this run's MIT ledger ({row}); nothing was changed")
        waiting = sorted(unit for unit in set(assigned) & set(status["queue"]["pending"]) if PACK_WORD in unit)
        print(f"four-copy packs waiting on MIT: {len(waiting)}: {', '.join(waiting) or 'none'}")
        print("jobs named: " + ", ".join(f"{job} {status['jobs'][job]['state']} {status['jobs'][job]['name']}"
                                         for job in args.cancel))
        if args.dry_run:
            print("dry run: nothing changed")
            return
        # 1. the table first, so a pack step between the hand-over loop's passes counts MIT as an eight-copy server
        changed = switch_table(table, args.remove_pool)
        common.write_json(fleet.table_path(run_dir), table)
        fleet.log(run_dir, f"mit | switch | table: {'; '.join(changed) or 'already eight copies'}")
        # 2. the waiting four-copy packs back to UVA, then apart into singles
        if waiting:
            fleet.take_back(run_dir, table, argparse.Namespace(server="mit", units=len(waiting), match=PACK_WORD,
                                                               from_end=False, hold=False))
        back = [unit for unit in waiting if (run_dir / "queue/pending" / f"{unit}.json").exists()]
        for unit in back:
            build_units.unpack_pending(run_dir, match=unit)
        fleet.log(run_dir, f"mit | switch | {len(back)} of {len(waiting)} waiting four-copy pack(s) back on UVA and "
                  "taken apart into singles")
        # 3. the named jobs, each only while it waits
        if args.cancel:
            answer = fleet.remote(run_dir, table, "mit", cancel_script(args.cancel))
            for line in answer.splitlines():
                if line.startswith(fleet.MARK) and not line.endswith("end"):
                    fleet.log(run_dir, f"mit | switch | {line[len(fleet.MARK):]}")
        common.write_json(fleet.table_path(run_dir), table)
    print(json.dumps({"table": changed, "packs_back_and_apart": back}, indent=2))


if __name__ == "__main__":
    main()
