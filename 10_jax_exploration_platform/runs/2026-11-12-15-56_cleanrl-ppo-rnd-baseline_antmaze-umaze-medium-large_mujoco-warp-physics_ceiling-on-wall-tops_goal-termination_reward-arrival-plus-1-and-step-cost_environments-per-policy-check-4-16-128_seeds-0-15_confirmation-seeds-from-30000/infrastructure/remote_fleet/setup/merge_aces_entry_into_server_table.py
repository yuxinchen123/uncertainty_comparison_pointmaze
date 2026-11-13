"""Add the TAMU ACES entry to this run's server table, under the remote fleet's lock.

    python merge_aces_entry_into_server_table.py

The entry is read from RUN/infrastructure/remote_fleet_servers_aces_entry.json. Another process rewrites the
table at its ticks, so the table is read, changed and written while this script holds the fleet's lock. A table
that already has an `aces` entry is left as it is. Standard library only; nothing is sent to any server.
"""
from pathlib import Path
import sys

RUN = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(RUN.parents[1] / "scripts/maze_campaign"))
from grid_campaign import common  # noqa: E402
import remote_fleet as maze  # noqa: E402


def merge(run_dir):
    """Write the server table with the ACES entry added; returns the table's server names afterwards."""
    fleet = maze.fleet
    entry = common.read_json(run_dir / "infrastructure/remote_fleet_servers_aces_entry.json")
    with fleet.fleet_lock(run_dir):
        # the table is read inside the lock, so whatever the last tick wrote is kept
        # before: {"gilbreth": {...}}; after: {"gilbreth": {...}, "aces": {...}}
        table = common.read_json(fleet.table_path(run_dir))
        if "aces" in table:
            raise SystemExit("the server table already has an aces entry; nothing was changed")
        table.update(entry)
        common.write_json(fleet.table_path(run_dir), table)
    return sorted(table)


if __name__ == "__main__":
    print("servers in the table now:", ", ".join(merge(RUN)))
