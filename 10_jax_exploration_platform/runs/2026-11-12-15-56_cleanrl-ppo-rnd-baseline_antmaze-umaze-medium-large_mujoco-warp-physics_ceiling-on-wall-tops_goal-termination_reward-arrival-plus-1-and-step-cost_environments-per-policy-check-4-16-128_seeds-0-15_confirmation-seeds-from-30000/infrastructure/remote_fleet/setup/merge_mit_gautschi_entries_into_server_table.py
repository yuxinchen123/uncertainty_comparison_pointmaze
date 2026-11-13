"""Add the MIT and Purdue Gautschi entries to this run's server table; run it while holding the fleet's lock.

    flock RUN/slurm/remote_fleet.lock /usr/bin/python3 merge_mit_gautschi_entries_into_server_table.py

The entries are read from RUN/infrastructure/remote_fleet_servers_mit_gautschi_entries.json. This script does not
take the lock itself: flock(1) holds it, and a second flock on the same file from this process would wait for ever.
Only the two entries are added; every other entry (gilbreth) is written back as it was read. A table that already
has either entry is left as it is. Standard library only; nothing is sent to any server.
"""
import json
import os
from pathlib import Path
import tempfile

RUN = Path(__file__).resolve().parents[3]
TABLE = RUN / "infrastructure/remote_fleet_servers.json"
ENTRIES = RUN / "infrastructure/remote_fleet_servers_mit_gautschi_entries.json"


def merged(table, entries):
    """Return the table with the new entries added after the existing ones."""
    # before: {"gilbreth": {...}}, entries {"mit": {...}, "gautschi": {...}}
    # after:  {"gilbreth": {...}, "mit": {...}, "gautschi": {...}}  (gilbreth unchanged)
    present = sorted(set(table) & set(entries))
    if present:
        raise SystemExit(f"the server table already has {', '.join(present)}; nothing was changed")
    return {**table, **entries}


def write_atomically(path, value):
    """Replace the table in one rename, as the fleet's own writer does."""
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    with os.fdopen(descriptor, "w") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.chmod(temporary, 0o664)
    os.replace(temporary, path)


if __name__ == "__main__":
    # the table is read here, inside the lock flock(1) holds, so whatever the last tick wrote is kept
    table = merged(json.loads(TABLE.read_text()), json.loads(ENTRIES.read_text()))
    write_atomically(TABLE, table)
    print("servers in the table now:", ", ".join(table))
