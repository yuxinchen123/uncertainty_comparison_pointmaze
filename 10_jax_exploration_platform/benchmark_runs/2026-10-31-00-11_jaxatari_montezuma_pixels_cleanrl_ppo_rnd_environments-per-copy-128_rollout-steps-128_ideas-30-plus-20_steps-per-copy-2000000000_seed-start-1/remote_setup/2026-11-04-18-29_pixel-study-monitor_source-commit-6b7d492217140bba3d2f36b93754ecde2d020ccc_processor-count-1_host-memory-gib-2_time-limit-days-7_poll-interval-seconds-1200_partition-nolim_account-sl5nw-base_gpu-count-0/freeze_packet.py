"""Freeze the complete new monitor packet and its five standard unit records."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import stat

PACKET = Path(__file__).resolve().parent
UNIT = PACKET.parent.parent / "data" / PACKET.name


def main():
    """Record exact prepared source, archive, metadata, modes and existing full-candidate proofs."""
    paths = sorted(path for directory in (PACKET, UNIT) for path in directory.rglob("*") if path.is_file())
    files = {}
    for path in paths:
        if path.is_symlink() or path.resolve(strict=True) != path:
            raise ValueError("a prepared monitor artifact is not an exact ordinary file")
        raw = path.read_bytes()
        files[str(path)] = {"sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw),
                            "mode": oct(stat.S_IMODE(path.stat().st_mode))}
    value = {"schema_version": 1, "kind": "prepared_cpu_monitor_complete_source_and_records_freeze",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": "6b7d492217140bba3d2f36b93754ecde2d020ccc", "files": files,
        "old_monitor_job_id": "6783003", "new_monitor_job_id": None, "new_jobs_submitted": 0,
        "scientific_credit": 0, "canonical_actions_and_provenance_not_included": True}
    path = PACKET / "static_source_freeze.json"
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
    print(json.dumps({"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                      "files": len(files), "total_bytes": sum(item["bytes"] for item in files.values())}))


if __name__ == "__main__":
    main()

