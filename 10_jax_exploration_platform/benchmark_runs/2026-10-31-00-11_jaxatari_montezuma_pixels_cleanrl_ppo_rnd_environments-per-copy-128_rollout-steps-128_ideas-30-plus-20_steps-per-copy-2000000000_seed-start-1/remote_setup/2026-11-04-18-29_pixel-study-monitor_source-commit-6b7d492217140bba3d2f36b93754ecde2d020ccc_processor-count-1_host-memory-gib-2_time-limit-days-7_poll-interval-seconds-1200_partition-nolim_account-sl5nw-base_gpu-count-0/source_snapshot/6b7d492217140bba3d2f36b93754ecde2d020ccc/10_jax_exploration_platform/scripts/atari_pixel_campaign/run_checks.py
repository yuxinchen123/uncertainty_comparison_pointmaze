"""Record the identity and outcome of a compute-node correctness command."""

import argparse
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import sys

from run_unit import atomic_json, hardware_record


def main():
    """Run the exact recorded command and preserve its exit status and hardware identity."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--unit-id", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        raise ValueError("a correctness command is required")
    directory = args.run_dir / "data" / args.unit_id
    directory.mkdir(parents=True, exist_ok=True)
    hardware = hardware_record()
    atomic_json(directory / "hardware.json", hardware)
    start = datetime.now(timezone.utc).isoformat()
    identity = {"source_commit": args.source_commit, "command": command,
                "started_at": start, "hardware": hardware}
    atomic_json(directory / "progress.json", {
        **identity, "state": "running", "pid": hardware["pid"], "phase": "correctness"})
    print("Correctness command: " + repr(command), flush=True)
    result = subprocess.run(command)
    record = {**identity, "passed": result.returncode == 0, "exit_code": result.returncode,
              "finished_at": datetime.now(timezone.utc).isoformat()}
    atomic_json(directory / "result.json", record)
    atomic_json(directory / "progress.json", {
        **record, "state": "completed" if record["passed"] else "failed",
        "pid": hardware["pid"], "completed": record["passed"]})
    if result.returncode:
        atomic_json(directory / "failure.json", {"error": "correctness command exited "
                                                + str(result.returncode)})
        raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
