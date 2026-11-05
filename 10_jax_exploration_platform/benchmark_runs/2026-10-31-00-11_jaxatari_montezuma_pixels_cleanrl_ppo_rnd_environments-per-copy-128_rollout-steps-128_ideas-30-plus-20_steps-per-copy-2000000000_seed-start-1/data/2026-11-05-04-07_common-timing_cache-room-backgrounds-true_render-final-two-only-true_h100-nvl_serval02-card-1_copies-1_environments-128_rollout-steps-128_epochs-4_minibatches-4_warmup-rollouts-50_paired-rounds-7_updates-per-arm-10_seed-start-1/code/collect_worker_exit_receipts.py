"""Observe exact old study processes and jobs without stopping or changing them.

Prepare freezes the campaign metadata. Collect makes one bounded observation of
the owned jobs and direct serval02 process identities. A live or unreported item
produces a waiting report, never a worker-stop receipt. No array library is used.
"""

import argparse
from datetime import datetime, timezone
import getpass
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess


TERMINAL_STATES = {"COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY",
                   "NODE_FAIL", "PREEMPTED", "BOOT_FAIL", "DEADLINE", "REVOKED"}
DIRECT_HOSTS = {"serval02"}


def utc_now():
    """Record the actual observation time with its explicit timezone."""
    return datetime.now(timezone.utc).isoformat()


def digest(contents):
    """Identify exact metadata bytes without importing numerical code."""
    return hashlib.sha256(contents).hexdigest()


def content_digest(value):
    """Match the controller's stable complete-record identity."""
    return digest(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode())


def write_once(path, value):
    """Publish a new metadata artifact durably without replacing prior observations."""
    # Immutable bytes make a later collection distinguishable from an older one.
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write((json.dumps(value, indent=2, allow_nan=False) + "\n").encode())
        stream.flush()
        os.fsync(stream.fileno())
    descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def input_paths(run_dir):
    """List only the exact campaign ownership and process metadata used by the guard."""
    # Before: two metadata directories; after: stable exact relative file names.
    fixed = ["processes.json", "slurm/submissions.tsv", "slurm/submitted_jobids_uva.txt"]
    return fixed + sorted(str(path.relative_to(run_dir)) for group in ("workers", "leases")
                          for path in (run_dir / "queue" / group).glob("*.json"))


def required_jobs(submissions, workers, runners, owned):
    """Include pending ordinary launches and every recorded retry job from the exact UVA ledger."""
    # Before: an ordinary launch plus a retry runner; after: both exact job rows.
    rows = [line for line in submissions.splitlines() if line]
    if any(len(line.split("\t")) != 4 for line in rows):
        raise ValueError("each submission row must contain its four recorded fields")
    identifiers = {line.split("\t")[0] for line in rows if line.split("\t")[1] == "queue-worker"}
    identifiers.update(str(row["slurm_job_id"]) for row in workers + runners if row.get("slurm_job_id"))
    selected = {}
    for identifier in sorted(identifiers, key=int):
        matches = [line for line in rows if line.split("\t")[0] == identifier]
        if not identifier.isdecimal() or identifier not in owned or len(matches) != 1:
            raise ValueError("each old worker job needs one exact submission row and exact UVA ledger ownership")
        selected[identifier] = matches[0]
    return selected


def snapshot(run_dir):
    """Freeze complete input bytes and derive the exact observations they require."""
    # A missing or changing record is an error, never evidence that a process exited.
    files = {name: (run_dir / name).read_bytes() for name in input_paths(run_dir)}
    launches = [row for row in json.loads(files["processes.json"]) if row.get("kind") == "queue_worker"]
    workers = [json.loads(contents) for name, contents in files.items() if name.startswith("queue/workers/")]
    runners = [json.loads(contents) for name, contents in files.items() if name.startswith("queue/leases/")]
    runners = [row for row in runners if row.get("runner_pid") is not None]
    owned = files["slurm/submitted_jobids_uva.txt"].decode().splitlines()
    jobs = required_jobs(files["slurm/submissions.tsv"].decode(), workers, runners, owned)
    return {"files": {name: {"sha256": digest(contents), "text": contents.decode()} for name, contents in files.items()},
            "launches": launches, "workers": workers, "runners": runners, "jobs": jobs}


def direct_identities(inputs):
    """Collect each exact host, process identifier and birth counter once."""
    # Before: a launch and worker naming the same process; after: one observation.
    identities = {(row["hostname"], row["pid"], row["start_time"]) for row in inputs["launches"]}
    for group, pid_key, birth_key in (("workers", "worker_pid", "worker_start_time"),
                                     ("runners", "runner_pid", "runner_start_time")):
        identities.update((row["hostname"], row[pid_key], row[birth_key]) for row in inputs[group]
                          if not row.get("slurm_job_id"))
    for host, pid, birth in identities:
        if host not in DIRECT_HOSTS or type(pid) is not int or pid < 1 or not str(birth).isdigit():
            raise ValueError("direct observations require authorized serval02 and exact positive process identities")
    return sorted(identities)


def prepare(run_dir, output):
    """Prepare an immutable metadata snapshot without querying jobs, GPUs or processes."""
    # Keep operational preparation under this campaign; no experiment is launched.
    run_dir, output = Path(run_dir).resolve(strict=True), Path(output).absolute()
    if not output.is_relative_to(run_dir / "remote_setup"):
        raise ValueError("prepared exit metadata must live under this campaign's remote_setup directory")
    inputs = snapshot(run_dir)
    direct_identities(inputs)
    plan = {"schema_version": 1, "kind": "worker_exit_observation_plan", "run_dir": str(run_dir),
            "prepared_at": utc_now(), "owner": getpass.getuser(), "owner_uid": os.getuid(),
            "collector_sha256": digest(Path(__file__).read_bytes()), "inputs": inputs,
            "scope": "metadata observations only; no signals, cancellation, admission or experiment credit"}
    write_once(output / "exit_observation_plan.json", plan)
    return output / "exit_observation_plan.json"


def verify_inputs(plan):
    """Reject an added, removed or changed old process record before or after observation."""
    # Freeze coverage as well as file bytes so a later launch cannot be omitted.
    run_dir = Path(plan["run_dir"])
    if snapshot(run_dir) != plan["inputs"]:
        raise ValueError("campaign ownership or process metadata changed; prepare a new snapshot")


def command_output(command, *, timeout=30):
    """Execute one bounded read-only command and preserve failures without retrying."""
    result = subprocess.run(command, capture_output=True, text=True, timeout=timeout, check=False)
    if result.returncode:
        raise RuntimeError("read-only observation failed: " + shlex.join(command[:5]) + ": " + result.stderr.strip())
    return result.stdout


def observe_jobs(jobs, owner):
    """Read only exact owned allocation identifiers and retain raw accounting rows."""
    # No jobs means no scheduler query; no accounting row means waiting.
    if not jobs:
        return {}, {"command": None, "stdout": "", "observed_at": utc_now()}
    command = ["sacct", "-X", "--jobs=" + ",".join(jobs), "--noheader", "--parsable2",
               "--format=JobIDRaw,User,State,ExitCode,NodeList,Start,End"]
    raw = command_output(command)
    queue_command = ["squeue", "--jobs=" + ",".join(jobs), "--noheader", "--format=%i|%u|%T"]
    queue_raw, observed = command_output(queue_command), utc_now()
    records = {}
    for line in raw.splitlines():
        fields = line.split("|")
        if len(fields) != 7 or fields[0] not in jobs or fields[1] != owner or fields[0] in records:
            raise ValueError("accounting returned an ambiguous, foreign or unrequested allocation")
        identifier, user, state, exit_code, nodes, start, end = fields
        records[identifier] = {"job_id": identifier, "owner": user, "state": state.split()[0].rstrip("+"),
            "state_raw": state, "exit_code": exit_code, "node_list": nodes, "start": start, "end": end,
            "submission_row": jobs[identifier], "observed_at": observed}
    # Before: accounting says PREEMPTED but squeue says PENDING; after: waiting.
    queued = set()
    for line in queue_raw.splitlines():
        fields = line.strip().split("|")
        if len(fields) != 3 or fields[0] not in jobs or fields[1] != owner or fields[0] in queued:
            raise ValueError("queue returned an ambiguous, foreign or unrequested allocation")
        identifier, _, state = fields
        queued.add(identifier)
        records.setdefault(identifier, {"job_id": identifier, "owner": owner, "state": "not_reported",
            "submission_row": jobs[identifier], "observed_at": observed})
        records[identifier].update(active_in_scheduler=True, queue_state=state)
    return records, {"command": command, "stdout": raw, "squeue_command": queue_command,
                     "squeue_stdout": queue_raw, "observed_at": observed}


def observe_direct(identities):
    """Make one key-only read-only SSH call per authorized direct host, with no retry."""
    # The remote code reads only the listed process birth counters and numeric owners.
    records, raw_records = {}, []
    for host in sorted({row[0] for row in identities}):
        requested = [[pid, str(birth)] for name, pid, birth in identities if name == host]
        script = ("import datetime,json,os,socket\nfrom pathlib import Path\n"
            "rows=[]\nfor pid,birth in " + repr(requested) + ":\n"
            " p=Path('/proc')/str(pid)\n"
            " try:\n  current=(p/'stat').read_text().rsplit(')',1)[1].split()[19]; uid=p.stat().st_uid\n"
            " except FileNotFoundError:\n  current=None; uid=None\n"
            " rows.append({'pid':pid,'expected_birth':birth,'observed_birth':current,'observed_uid':uid,"
            "'original_process_alive':current==birth})\n"
            "print(json.dumps({'hostname':socket.gethostname().split('.')[0],'observations':rows,"
            "'observed_at':datetime.datetime.now(datetime.timezone.utc).isoformat()}))\n")
        remote = "python3 -c " + shlex.quote(script)
        command = ["ssh", "-o", "BatchMode=yes", "-o", "ControlMaster=no", "-o", "ConnectTimeout=15", host, remote]
        raw = command_output(command, timeout=180)
        value = json.loads(raw)
        expected = {(pid, birth) for pid, birth in requested}
        actual = {(row["pid"], row["expected_birth"]) for row in value["observations"]}
        if value["hostname"] != host or actual != expected or len(actual) != len(value["observations"]):
            raise ValueError("direct observation returned another host or incomplete process identities")
        for row in value["observations"]:
            if row["original_process_alive"] is not (row["observed_birth"] == row["expected_birth"]):
                raise ValueError("direct observation liveness differs from its actual process birth counter")
            if row["original_process_alive"] and row.get("observed_uid") != os.getuid():
                raise ValueError("a matching original process birth counter belongs to a different user")
            records[host, row["pid"], row["expected_birth"]] = {**row, "hostname": host, "observed_at": value["observed_at"]}
        raw_records.append({"hostname": host, "stdout": raw, "observed_at": value["observed_at"]})
    return records, raw_records


def make_receipt(plan, jobs, direct):
    """Build complete guard-compatible receipts only from actual terminal observations."""
    # Waiting records remain observations; they can never become stop proofs.
    waiting = [{"job_id": identifier, "state": jobs.get(identifier, {}).get("queue_state",
                jobs.get(identifier, {}).get("state", "not_reported"))}
               for identifier in plan["inputs"]["jobs"] if jobs.get(identifier, {}).get("state") not in TERMINAL_STATES
               or jobs[identifier].get("active_in_scheduler") is True]
    waiting += [row for row in direct.values() if row["original_process_alive"]]
    if waiting:
        return None, waiting
    receipt = {"schema_version": 1, "workers": [], "jobs": list(jobs.values()), "processes": [], "runners": []}
    ledger = str(Path(plan["run_dir"]) / "slurm/submitted_jobids_uva.txt")
    for row in plan["inputs"]["launches"]:
        proof = direct[row["hostname"], row["pid"], str(row["start_time"]) ]
        receipt["processes"].append({**row, **proof, "pid": row["pid"], "start_time": row["start_time"],
                                     "launch_record_digest": content_digest(row)})
    for group, pid_key, birth_key in (("workers", "worker_pid", "worker_start_time"),
                                     ("runners", "runner_pid", "runner_start_time")):
        for row in plan["inputs"][group]:
            if row.get("slurm_job_id"):
                job = jobs[str(row["slurm_job_id"])]
                proof = {"kind": "terminal_slurm_job", "state": job["state"], "ledger_path": ledger,
                         "observed_at": job["observed_at"]}
            else:
                observed = direct[row["hostname"], row[pid_key], str(row[birth_key])]
                proof = {**observed, "kind": "dead_original_process"}
            receipt[group].append({**row, **proof})
    return receipt, []


def collect(prepared, output):
    """Observe one frozen snapshot, write status, and publish a stop receipt only after every exit."""
    # Collection is read-only toward original campaign metadata and all processes.
    prepared, output = Path(prepared).resolve(strict=True), Path(output).absolute()
    plan = json.loads(prepared.read_text())
    if (plan.get("kind") != "worker_exit_observation_plan" or plan.get("schema_version") != 1
            or plan["owner"] != getpass.getuser() or plan["owner_uid"] != os.getuid()
            or plan["collector_sha256"] != digest(Path(__file__).read_bytes())
            or not output.is_relative_to(prepared.parent)):
        raise ValueError("exit collection owner, helper or output differs from its prepared snapshot")
    verify_inputs(plan)
    jobs, scheduler = observe_jobs(plan["inputs"]["jobs"], plan["owner"])
    direct, processes = observe_direct(direct_identities(plan["inputs"]))
    verify_inputs(plan)
    receipt, waiting = make_receipt(plan, jobs, direct)
    status = {"schema_version": 1, "state": "waiting" if waiting else "complete", "waiting": waiting,
              "observed_at": utc_now(), "prepared_plan": str(prepared), "prepared_plan_sha256": digest(prepared.read_bytes()),
              "counts": {key: len(plan["inputs"][key]) for key in ("workers", "jobs", "launches", "runners")}}
    write_once(output / "observations.json", {"scheduler": scheduler, "direct_processes": processes})
    if receipt is not None:
        write_once(output / "worker_stop.json", receipt)
    write_once(output / "status.json", status)
    return status


def main():
    """Keep immutable preparation and one actual observation as separate commands."""
    parser = argparse.ArgumentParser(description=__doc__)
    operations = parser.add_subparsers(dest="operation", required=True)
    preparation = operations.add_parser("prepare")
    preparation.add_argument("--run-dir", type=Path, required=True)
    preparation.add_argument("--output", type=Path, required=True)
    collection = operations.add_parser("collect")
    collection.add_argument("--prepared", type=Path, required=True)
    collection.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = prepare(args.run_dir, args.output) if args.operation == "prepare" else collect(args.prepared, args.output)
    print(json.dumps(str(result) if isinstance(result, Path) else result, indent=2), flush=True)


if __name__ == "__main__":
    main()
