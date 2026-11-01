"""Refresh one owned ACES metadata observation through an already open portal05 connection.

This program submits, releases and cancels no jobs. It copies only allowlisted
metadata, preserves original remote identities and publishes the status snapshot
last. A failed refresh retains the prior snapshot and records a separate error.
"""

import argparse
import base64
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import pwd
import shlex
import socket
import subprocess
import tempfile
import uuid


REMOTE_PYTHON = "/scratch/user/u.sl359075/envs/rnd_atari_pixels_jax/bin/python"
DATA_FILES = {"progress.json", "hardware.json", "owned_allocation.json", "hardware_observations.json",
              "nvidia_hardware_queries.json", "result.json", "failure.json"}
MAX_FILE_BYTES = 2 * 1024 ** 2
MAX_RESPONSE_BYTES = 16 * 1024 ** 2
SPECIFICATION = importlib.util.spec_from_file_location("aces_observation_frozen_monitor", Path(__file__).with_name("monitor.py"))
MONITOR = importlib.util.module_from_spec(SPECIFICATION)
SPECIFICATION.loader.exec_module(MONITOR)

REMOTE_PROGRAM = r"""
import base64,datetime,hashlib,json,os,pwd,socket,subprocess,sys
from pathlib import Path
request=json.loads(sys.argv[1])
if any(name in sys.modules for name in ('jax','numpy','torch')):raise RuntimeError('metadata interpreter loaded a numerical library')
if pwd.getpwuid(os.getuid()).pw_name!='u.sl359075' or socket.gethostname()!='alogin1.cluster':raise ValueError('unexpected ACES owner or existing-connection host')
run=Path(request['remote_campaign_run_dir']); packet=run/'remote_setup'/request['packet_name']; unit=run/'data'/packet.name
if run!=Path('/scratch/user/u.sl359075/rnd_atari_pixels/runs')/request['campaign_name']:raise ValueError('unexpected remote campaign')
if packet.is_symlink() or packet.resolve().parent!=(run/'remote_setup').resolve() or unit.is_symlink() or unit.resolve().parent!=(run/'data').resolve():raise ValueError('remote metadata directory escaped its campaign')
def read_file(path,root):
 '''Read bounded regular metadata without following a leaf or parent escape.'''
 if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()) or not path.is_file() or path.stat().st_size>2097152:raise ValueError('unsafe or oversized metadata file')
 contents=path.read_bytes()
 return {'path':str(path),'sha256':hashlib.sha256(contents).hexdigest(),'base64':base64.b64encode(contents).decode('ascii')}
packets={}
for name,expected in request['packet_hashes'].items():
 record=read_file(packet/name,packet)
 if record['sha256']!=expected:raise ValueError('immutable remote preparation changed: '+name)
 packets[name]=record
receipt=json.loads(base64.b64decode(packets['submission_receipt.json']['base64']))
plan=json.loads(base64.b64decode(packets['allocation_observation_plan.json']['base64']))
identifier=request['job_id']
if receipt['cluster']!='aces' or receipt['job_id']!=identifier or plan['owner']!={'username':'u.sl359075','uid':os.getuid()}:raise ValueError('remote observation owner or job differs')
ledger=run/'slurm/submitted_jobids_aces.txt'; rows=(run/'slurm/submissions_aces.tsv').read_text().splitlines()
matches=[line for line in rows if line.split('\t',1)[0]==identifier]
if ledger.is_symlink() or identifier not in ledger.read_text().splitlines() or len(matches)!=1 or matches[0]!='\t'.join(receipt['submission_row']):raise ValueError('remote exact ACES ledger or submission row differs')
if (packet/'release_receipt.json').exists():packets['release_receipt.json']=read_file(packet/'release_receipt.json',packet)
def query(command):
 '''Query one exact owned identifier without retrying a failed command.'''
 result=subprocess.run(command,capture_output=True,text=True,timeout=20)
 return {'command':command,'returncode':result.returncode,'stdout':result.stdout,'stderr':result.stderr}
queue=query(['squeue','-h','-j',identifier,'-o','%i|%u|%T|%N|%R'])
accounting=query(['sacct','-X','-n','-P','-j',identifier,'--format=JobIDRaw,User%64,State%64,ExitCode,NodeList'])
files={}
for name in sorted(request['data_filenames']):
 path=unit/name
 if path.exists():files[name]=read_file(path,unit)
payload={'schema_version':1,'kind':'aces_allocation_refresh_payload','query_host':socket.gethostname(),'user':pwd.getpwuid(os.getuid()).pw_name,
 'job_id':identifier,'remote_campaign_run_dir':str(run),'packet_name':packet.name,'packet_files':packets,'data_files':files,
 'squeue':queue,'sacct':accounting,'observed_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
serialized=json.dumps(payload)
if len(serialized.encode())>16777216:raise ValueError('metadata response exceeds its bound')
print(serialized)
"""


def digest(contents):
    """Hash exact metadata bytes rather than a reserialized approximation."""
    return hashlib.sha256(contents).hexdigest()


def utc_now():
    """Record actual refresh times with an explicit UTC offset."""
    return datetime.now(timezone.utc).isoformat()


def atomic_bytes(path, contents, *, replace=True):
    """Publish complete durable bytes while refusing symlink destinations."""
    # A status file changes once, after its complete replacement reaches disk.
    path = Path(path)
    if path.is_symlink():
        raise ValueError("refusing a symlink metadata destination: " + str(path))
    descriptor, temporary = tempfile.mkstemp(prefix=".aces-refresh-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            os.fchmod(stream.fileno(), 0o644)
            stream.write(contents)
            stream.flush()
            os.fsync(stream.fileno())
        if replace:
            os.replace(temporary, path)
        else:
            try:
                os.link(temporary, path)
            except FileExistsError as error:
                if path.is_symlink() or path.read_bytes() != contents:
                    raise ValueError("immutable metadata destination already contains different bytes: " + str(path)) from error
        descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        if Path(temporary).exists():
            Path(temporary).unlink()


def local_inputs(run_dir, packet):
    """Validate exact local ownership and path bounds before any master or network command."""
    # Equal job numbers in UVA and ACES remain separate ledger entries.
    if socket.gethostname().split(".")[0] != "portal05" or pwd.getpwuid(os.getuid()).pw_name != "sl5nw":
        raise ValueError("ACES observation refresh is restricted to sl5nw on portal05")
    run_dir, packet = Path(run_dir).resolve(strict=True), Path(packet).absolute()
    namespace = packet.name.replace("-", " ").replace("_", " ").lower()
    if (packet.parent != run_dir / "remote_setup" or packet.is_symlink()
            or packet.resolve().parent != run_dir / "remote_setup" or "aces" not in namespace.split()
            or "allocation observation" not in namespace):
        raise ValueError("refresh requires the exact recognizable direct-child ACES observation packet")
    ledger = run_dir / "slurm/submitted_jobids_aces.txt"
    if ledger.is_symlink() or (packet / "submission_receipt.json").is_symlink():
        raise ValueError("ACES ownership records cannot be symlinks")
    receipt = MONITOR.read_json(packet / "submission_receipt.json")
    if not isinstance(receipt, dict) or receipt.get("kind") != "owned_aces_allocation_observation_submission":
        raise ValueError("refresh requires the exact ACES allocation-observation submission kind")
    identifier, plan, unit, remote, _ = MONITOR.aces_observation_packet(
        run_dir, packet, receipt, set(ledger.read_text().splitlines()))
    manifest = MONITOR.read_json(packet / "manifest.yaml")
    names = set(manifest["artifacts"]) | {"manifest.yaml", "submission_receipt.json"}
    if (packet / "release_receipt.json").exists():
        names.add("release_receipt.json")
    hashes = {name: digest((packet / name).read_bytes()) for name in sorted(names)}
    protected = {str(ledger): digest(ledger.read_bytes()),
                 str(run_dir / "slurm/submissions_aces.tsv"): digest((run_dir / "slurm/submissions_aces.tsv").read_bytes())}
    protected.update({str(packet / name): value for name, value in hashes.items()})
    return {"run": run_dir, "packet": packet, "unit": unit, "receipt": receipt, "plan": plan,
            "remote": remote, "identifier": identifier, "packet_hashes": hashes, "protected": protected}


def command_result(command, timeout):
    """Run one bounded command without retries, interactive authentication or shell interpolation."""
    result = subprocess.run(command, capture_output=True, text=True, timeout=timeout, check=False)
    return {"command": command, "returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr}


def receive(inputs):
    """Reuse an existing ACES control master for one bounded metadata-only remote command."""
    # Reuse the configured socket; the false proxy prevents a new connection if it is absent.
    # Explicit master checks belong to the separate hourly resource review.
    request = {"remote_campaign_run_dir": str(inputs["remote"]), "campaign_name": inputs["run"].name,
               "packet_name": inputs["packet"].name, "job_id": inputs["identifier"],
               "packet_hashes": inputs["packet_hashes"], "data_filenames": sorted(DATA_FILES)}
    remote_command = shlex.join([REMOTE_PYTHON, "-I", "-c", REMOTE_PROGRAM, json.dumps(request)])
    command = ["ssh", "-o", "ControlMaster=no", "-o", "ProxyCommand=false", "-o", "BatchMode=yes",
               "-o", "ConnectTimeout=10", "aces", remote_command]
    received = command_result(command, 120)
    if received["returncode"] != 0 or len(received["stdout"].encode()) > MAX_RESPONSE_BYTES:
        raise RuntimeError("existing-master ACES metadata command failed: " + received["stderr"][:4000])
    return json.loads(received["stdout"]), {"master_check": {"performed": False,
        "reason": "not separately checked; explicit checks belong to the hourly resource review"},
        "remote_returncode": received["returncode"], "remote_stderr": received["stderr"], "remote_python": REMOTE_PYTHON}


def decoded_file(record, expected_path):
    """Require one bounded unchanged file at its exact original remote path."""
    contents = base64.b64decode(record["base64"], validate=True)
    if (record.get("path") != str(expected_path) or len(contents) > MAX_FILE_BYTES
            or digest(contents) != record.get("sha256")):
        raise ValueError("received metadata file changed its path, size or bytes")
    return contents


def proposed_snapshot(inputs, payload):
    """Validate every transported file and derive the monitor snapshot from original raw query output."""
    # No independent normalized status is trusted or supplied to the monitor.
    expected = {"schema_version": 1, "kind": "aces_allocation_refresh_payload", "query_host": "alogin1.cluster",
                "user": "u.sl359075", "job_id": inputs["identifier"], "remote_campaign_run_dir": str(inputs["remote"]),
                "packet_name": inputs["packet"].name}
    if any(type(payload.get(key)) is not type(value) or payload.get(key) != value for key, value in expected.items()):
        raise ValueError("remote metadata payload identifies another host, owner, campaign or job")
    packet_files, data_files, writes = payload["packet_files"], payload["data_files"], {}
    if not set(inputs["packet_hashes"]) <= set(packet_files) or set(packet_files) - set(inputs["packet_hashes"]) - {"release_receipt.json"}:
        raise ValueError("remote packet file roster differs from the immutable request")
    remote_packet = inputs["remote"] / "remote_setup" / inputs["packet"].name
    for name, record in packet_files.items():
        contents = decoded_file(record, remote_packet / name)
        destination = inputs["packet"] / name
        if name in inputs["packet_hashes"] and digest(contents) != inputs["packet_hashes"][name]:
            raise ValueError("immutable preparation bytes changed: " + name)
        if destination.exists() and destination.read_bytes() != contents:
            raise ValueError("refusing conflicting local packet bytes: " + name)
        writes[destination] = contents
    if set(data_files) - DATA_FILES:
        raise ValueError("remote response included a data file outside the allowlist")
    mirrored = {}
    for name, record in data_files.items():
        contents = decoded_file(record, inputs["remote"] / "data" / inputs["unit"].name / name)
        destination = inputs["unit"] / name
        if destination.is_symlink() or destination.resolve().parent != inputs["unit"]:
            raise ValueError("local data destination escaped its exact observation shard")
        if name != "progress.json" and destination.exists() and destination.read_bytes() != contents:
            raise ValueError("refusing conflicting immutable observation bytes: " + name)
        writes[destination] = contents
        mirrored[str(destination.relative_to(inputs["run"]))] = {"source_path": record["path"], "sha256": record["sha256"]}
    release = writes.get(inputs["packet"] / "release_receipt.json")
    snapshot = {"schema_version": 1, "kind": "aces_allocation_monitor_snapshot", "cluster": "aces",
        "user": "u.sl359075", "job_id": inputs["identifier"], "origin_host": "portal05", "query_host": payload["query_host"],
        "local_campaign_run_dir": str(inputs["run"]), "remote_campaign_run_dir": str(inputs["remote"]),
        "plan_sha256": inputs["receipt"]["plan_sha256"], "job_script_sha256": inputs["receipt"]["job_script_sha256"],
        "submission_receipt_sha256": inputs["packet_hashes"]["submission_receipt.json"],
        "release_receipt_sha256": digest(release) if release is not None else None,
        "observed_at_utc": payload["observed_at_utc"], "squeue": payload["squeue"], "sacct": payload["sacct"], "mirrored_files": mirrored}
    return snapshot, writes


def validate_proposal(inputs, snapshot, writes):
    """Exercise the exact monitor against a temporary complete mirror before changing current files."""
    # Original remote records remain byte-for-byte copies; only this temporary snapshot names the temporary local root.
    with tempfile.TemporaryDirectory(prefix="aces-observation-review-") as temporary:
        staged = Path(temporary) / inputs["run"].name
        for path, contents in writes.items():
            target = staged / path.relative_to(inputs["run"])
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(contents)
        for relative in ("slurm/submitted_jobids_aces.txt", "slurm/submissions_aces.tsv"):
            target = staged / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes((inputs["run"] / relative).read_bytes())
        packet = staged / "remote_setup" / inputs["packet"].name
        document = {**snapshot, "local_campaign_run_dir": str(staged)}
        (packet / "remote_monitor_snapshot.json").write_text(json.dumps(document))
        rows, _ = MONITOR.aces_allocation_observation_rows(staged)
        if len(rows) != 1 or rows[0]["state"] == "unknown":
            raise ValueError("received metadata cannot produce a valid owned monitor observation: " + str(rows))
        return rows[0]


def publish(inputs, snapshot, writes):
    """Save complete files first and the snapshot last, restoring prior mutable bytes if publication fails."""
    # Immutable preparation and observation files may be added, never replaced with conflicting bytes.
    for path, expected in inputs["protected"].items():
        if digest(Path(path).read_bytes()) != expected:
            raise ValueError("local ownership or immutable preparation changed during refresh")
    if inputs["unit"].is_symlink() or inputs["unit"].resolve().parent != inputs["run"] / "data":
        raise ValueError("local observation directory changed during refresh")
    progress = inputs["unit"] / "progress.json"
    previous_progress = progress.read_bytes() if progress.exists() else None
    snapshot_path = inputs["packet"] / "remote_monitor_snapshot.json"
    previous_snapshot = snapshot_path.read_bytes() if snapshot_path.exists() else None
    try:
        for path, contents in writes.items():
            if path == progress:
                continue
            if path.exists() and path.read_bytes() == contents:
                continue
            if path.exists():
                raise ValueError("immutable local observation changed during refresh: " + str(path))
            atomic_bytes(path, contents, replace=False)
        if progress in writes:
            atomic_bytes(progress, writes[progress])
        atomic_bytes(snapshot_path, (json.dumps(snapshot, indent=2) + "\n").encode())
    except BaseException:
        # Restore only files this refresh can replace; newly copied immutable artifacts remain unclaimed by the old snapshot.
        if previous_progress is not None:
            atomic_bytes(progress, previous_progress)
        if previous_snapshot is not None:
            atomic_bytes(snapshot_path, previous_snapshot)
        raise


def refresh(run_dir, packet):
    """Perform one owned refresh or retain the prior snapshot and save a separate explicit error."""
    run_dir = Path(run_dir).resolve(strict=True)
    packet = Path(packet).absolute()
    previous = packet / "remote_monitor_snapshot.json"
    safe_packet = (packet.parent == run_dir / "remote_setup" and not packet.is_symlink()
                   and packet.resolve().parent == run_dir / "remote_setup")
    previous_bytes = previous.read_bytes() if safe_packet and previous.is_file() and not previous.is_symlink() else None
    stage = "local ownership validation"
    try:
        inputs = local_inputs(run_dir, packet)
        if (packet / ".refresh.lock").is_symlink():
            raise ValueError("the observation refresh lock cannot be a symlink")
        with (packet / ".refresh.lock").open("a+") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            stage = "existing-master metadata observation"
            payload, transport = receive(inputs)
            stage = "transported byte and monitor validation"
            snapshot, writes = proposed_snapshot(inputs, payload)
            row = validate_proposal(inputs, snapshot, writes)
            stage = "atomic publication"
            publish(inputs, snapshot, writes)
            return {"state": "refreshed", "job_id": inputs["identifier"], "snapshot": str(previous),
                    "snapshot_sha256": digest(previous.read_bytes()), "observed_state": row["state"],
                    "transport": transport, "no_job_action": True}
    except Exception as error:
        # Error records are separate from the last successful snapshot and cannot manufacture a scheduler state.
        errors = run_dir / "20_mins_monitoring" / "aces_refresh_errors"
        errors.mkdir(parents=True, exist_ok=True)
        path = errors / (datetime.now(timezone.utc).strftime("%Y-%m-%d-%H-%M-%S") + "_" + uuid.uuid4().hex + ".json")
        retained = previous_bytes is None or previous.is_file() and previous.read_bytes() == previous_bytes
        record = {"schema_version": 1, "kind": "aces_allocation_refresh_error", "at_utc": utc_now(), "stage": stage,
                  "packet": str(packet), "error": repr(error), "prior_snapshot_retained": retained,
                  "prior_snapshot_sha256": digest(previous_bytes) if previous_bytes is not None else None, "no_job_action": True}
        atomic_bytes(path, (json.dumps(record, indent=2) + "\n").encode())
        raise RuntimeError("ACES metadata refresh failed; prior snapshot retained=" + str(retained) + "; error record " + str(path)) from error


def main():
    """Refresh one explicit packet using only its own recorded ACES identifier."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--packet", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(refresh(args.run_dir, args.packet), indent=2), flush=True)


if __name__ == "__main__":
    main()
