"""Cancel the jobs of ONE ledger file of a run folder, by exact id. The only cancel path of the maze campaign.

    python cancel_ledger.py --ledger RUN/slurm/submitted_jobids_<phase>_<account>.txt [--dry-run]
    python cancel_ledger.py --ledger RUN/slurm/submitted_jobids_<phase>_<account>.txt --job-id <id> [--job-id <id>]
    python cancel_ledger.py --ledger RUN/slurm/submitted_jobids_<server>.txt --server <server>

The first form cancels the UVA jobs of one phase and one account, running and waiting. The second
cancels only the named ids of that ledger (a job pinned to a node that still waits after a
monitoring pass is cancelled this way and submitted again). The third cancels the jobs of one
remote server that still wait (the grid campaign's rule: a running remote job is never cancelled;
it ends when its queue is empty).

A whole ledger is cancelled only when RUN/STOP exists: the queue must stop refilling first (rule
submit-the-whole-queue-up-front.md, "Cancelling a whole queue"). Nothing is cancelled by user,
by state or by name. On UVA the shared wrapper cancel_as.sh checks every id against the ledger and
against the account that owns the live job. A complete stop runs this once per ledger file.
"""
import argparse
from pathlib import Path
import subprocess

from grid_campaign import common, load

ACCOUNTS = ("sl5nw", "yuxinchen")


def main(argv=None):
    """Check the ledger and the STOP file, then cancel that ledger's jobs by id."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ledger", type=Path, required=True, help="one submitted_jobids_*.txt of a run folder")
    parser.add_argument("--server", help="the remote server this ledger belongs to (waiting jobs only)")
    parser.add_argument("--job-id", action="append", default=[],
                        help="UVA only: cancel this id of the ledger instead of the whole ledger (repeatable)")
    parser.add_argument("--dry-run", action="store_true", help="UVA only: print the command and send nothing")
    args = parser.parse_args(argv)
    ledger = args.ledger.resolve()
    run_dir = ledger.parents[1]
    # the file must be a ledger of a run folder; a whole ledger is cancelled only for a stopped run
    if ledger.parent.name != "slurm" or not ledger.name.startswith("submitted_jobids_") or not ledger.exists():
        raise SystemExit(f"{ledger} is not an existing job-id ledger (RUN/slurm/submitted_jobids_*.txt)")
    if not args.job_id and not (run_dir / "STOP").exists():
        raise SystemExit(f"{run_dir / 'STOP'} does not exist. Create it first, so that no worker claims a unit "
                         "and no monitor submits a job while this ledger is cancelled.")
    if args.job_id and args.server:
        raise SystemExit("single ids are cancelled on UVA only; a remote server's waiting jobs go together")

    if args.server:
        # a remote server: only its own ledger, only ids that still wait there
        fleet = load("remote_fleet")
        if ledger != fleet.ledger_path(run_dir, args.server).resolve():
            raise SystemExit(f"{ledger.name} is not the ledger of server {args.server}")
        with fleet.fleet_lock(run_dir):
            fleet.cancel_pending(run_dir, common.read_json(fleet.table_path(run_dir)),
                                 argparse.Namespace(server=args.server))
        return

    # UVA: the account is the last word of the file's name; the wrapper checks every id twice
    account = ledger.stem.rsplit("_", 1)[1]
    if account not in ACCOUNTS:
        raise SystemExit(f"{ledger.name} names no UVA account ({', '.join(ACCOUNTS)}); a remote server's "
                         "ledger needs --server")
    command = [str(common.ACCOUNTS / "cancel_as.sh"), f"uva-{account}", "--idfile", str(ledger),
               *(args.job_id or ["--all"])]
    print(" ".join(command))
    if not args.dry_run:
        subprocess.run(command, check=True, stdin=subprocess.DEVNULL)


if __name__ == "__main__":
    main()
