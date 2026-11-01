"""Mirror one owned ACES full-site check through an existing SSH master, without job actions."""

import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import pwd
import shlex
import socket
import tempfile
import uuid

import aces_site_monitor as adapter
import refresh_aces_allocation_observation as transport


REMOTE_PROGRAM = r"""
import base64,datetime,hashlib,json,os,pathlib,pwd,socket,subprocess,sys
if any(name.split('.')[0] in {'jax','jaxlib','numpy','torch','flax','optax'} for name in sys.modules):raise RuntimeError('remote site metadata process imported a numerical runtime')
request=json.loads(sys.argv[1]);run=pathlib.Path(request['remote_campaign_run_dir'])
allowed_data={'progress.json','hardware.json','result.json','failure.json','parent_owner.json','parent_allocation_observation.json','site_assignment.json','identity_result.json','probe_result.json'}|{kind+'_'+suffix+'.json' for kind in ('identity','learner','renderer','probe') for suffix in ('owner','process','terminal','allocation_observation')}
if not set(request['data_filenames'])<=allowed_data:raise ValueError('requested data exceeds the fixed site metadata allowlist')
if run!=pathlib.Path('/scratch/user/u.sl359075/rnd_atari_pixels/runs')/request['campaign_name'] or run!=run.resolve():raise ValueError('wrong native ACES campaign')
if pwd.getpwuid(os.getuid()).pw_name!='u.sl359075' or os.getuid()!=56626:raise ValueError('wrong actual ACES owner')
packet=run/'remote_setup'/request['packet_name'];unit=run/'data'/request['packet_name']
if pathlib.Path(request['packet_name']).name!=request['packet_name'] or any(path!=path.resolve() for path in (packet,unit)):raise ValueError('packet or unit escapes its native campaign')
def read_file(path):
    '''Read one bounded ordinary file and retain its exact original path and bytes.'''
    if path!=path.resolve() or not path.is_file():raise ValueError('metadata path is not an exact ordinary file')
    with path.open('rb') as stream:data=stream.read(2097153)
    if len(data)>2097152:raise ValueError('metadata file exceeds its bound')
    return {'path':str(path),'sha256':hashlib.sha256(data).hexdigest(),'base64':base64.b64encode(data).decode('ascii')}
packets={}
for name,expected in request['packet_hashes'].items():
    if pathlib.Path(name).name!=name:raise ValueError('packet file must be a direct child')
    item=read_file(packet/name)
    if item['sha256']!=expected:raise ValueError('native immutable packet bytes changed')
    packets[name]=item
submission=json.loads(base64.b64decode(packets['submission_receipt.json']['base64']))
plan=json.loads(base64.b64decode(packets['site_validation_plan.json']['base64']))
identifier=request['job_id']
if submission.get('kind')!='owned_aces_full_site_qualification_submission' or submission.get('cluster')!='aces' or submission.get('job_id')!=identifier or plan.get('owner')!={'username':'u.sl359075','uid':56626}:raise ValueError('native job owner or submission changed')
ledger=run/'slurm/submitted_jobids_aces.txt';rows=run/'slurm/submissions_aces.tsv'
if ledger!=ledger.resolve() or rows!=rows.resolve() or ledger.read_text().splitlines().count(identifier)!=1:raise ValueError('native ACES ledger does not own the exact ID')
matches=[line for line in rows.read_text().splitlines() if line.split('\t',1)[0]==identifier]
if matches!=['\t'.join(submission['submission_row'])]:raise ValueError('native exact ACES submission row changed')
if (packet/'release_receipt.json').exists():packets['release_receipt.json']=read_file(packet/'release_receipt.json')
def query(command):
    '''Query only this already validated owned ACES identifier, without retries.'''
    result=subprocess.run(command,capture_output=True,text=True,timeout=20,env={**os.environ,'TZ':'UTC'})
    return {'command':command,'returncode':result.returncode,'stdout':result.stdout,'stderr':result.stderr}
queue=query(['squeue','-h','-j',identifier,'-o','%i|%u|%T|%N|%R'])
accounting=query(['sacct','-X','-n','-P','-j',identifier,'--format=JobIDRaw,User%64,State%64,ExitCode,NodeList'])
files={name:read_file(unit/name) for name in request['data_filenames'] if (unit/name).exists()}
payload={'schema_version':1,'kind':'aces_full_site_refresh_payload','query_host':socket.gethostname(),'user':pwd.getpwuid(os.getuid()).pw_name,
    'job_id':identifier,'remote_campaign_run_dir':str(run),'packet_name':packet.name,'packet_files':packets,'data_files':files,
    'squeue':queue,'sacct':accounting,'observed_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
serialized=json.dumps(payload)
if len(serialized.encode())>16777216:raise ValueError('metadata response exceeds its bound')
print(serialized)
"""


def local_inputs(run, packet):
    """Validate all local ACES ownership before permitting any transport command."""
    adapter.require(socket.gethostname().split(".")[0] == "portal05"
        and pwd.getpwuid(os.getuid()).pw_name == "sl5nw", "ACES site refresh requires sl5nw on portal05")
    inputs = adapter.packet_inputs(run, packet)
    names = adapter.ARTIFACTS | {"manifest.yaml", "submission_receipt.json"}
    if inputs["release"] is not None:
        names.add("release_receipt.json")
    inputs["packet_hashes"] = {name: adapter.digest(adapter.contents(inputs["packet"] / name)) for name in sorted(names)}
    protected = [inputs["run"] / "slurm" / name for name in ("submitted_jobids_aces.txt", "submissions_aces.tsv")]
    inputs["protected"] = {str(path): adapter.digest(adapter.contents(path)) for path in protected}
    inputs["protected"].update({str(inputs["packet"] / name): value for name, value in inputs["packet_hashes"].items()})
    return inputs


def receive(inputs):
    """Reuse the existing ACES socket once; a missing master cannot trigger new authentication."""
    request = {"remote_campaign_run_dir": str(inputs["remote"]), "campaign_name": inputs["run"].name,
        "packet_name": inputs["packet"].name, "job_id": inputs["identifier"],
        "packet_hashes": inputs["packet_hashes"], "data_filenames": sorted(adapter.DATA_FILES)}
    remote = shlex.join([transport.REMOTE_PYTHON, "-I", "-c", REMOTE_PROGRAM, json.dumps(request)])
    command = ["ssh", "-o", "ControlMaster=no", "-o", "ProxyCommand=false", "-o", "BatchMode=yes",
        "-o", "ConnectTimeout=10", "aces", remote]
    result = transport.command_result(command, 120)
    adapter.require(result["returncode"] == 0 and len(result["stdout"].encode()) <= transport.MAX_RESPONSE_BYTES,
        "existing-master ACES site metadata command failed: " + result["stderr"][:4000])
    return json.loads(result["stdout"]), {"master_check": {"performed": False,
        "reason": "not separately checked; explicit checks belong to the hourly resource review"},
        "remote_returncode": result["returncode"], "remote_stderr": result["stderr"], "remote_python": transport.REMOTE_PYTHON}


def proposed_snapshot(inputs, payload):
    """Bind every received byte to its native path and retain raw scheduler output unchanged."""
    expected = {"schema_version": 1, "kind": "aces_full_site_refresh_payload", "query_host": "alogin1.cluster",
        "user": adapter.OWNER, "job_id": inputs["identifier"], "remote_campaign_run_dir": str(inputs["remote"]),
        "packet_name": inputs["packet"].name}
    adapter.require(all(type(payload.get(key)) is type(value) and payload[key] == value for key, value in expected.items()),
        "ACES site payload identifies another host, owner, packet or job")
    packet_files, data_files, writes = payload["packet_files"], payload["data_files"], {}
    adapter.require(set(inputs["packet_hashes"]) <= set(packet_files)
        and not set(packet_files) - set(inputs["packet_hashes"]) - {"release_receipt.json"}, "native packet roster changed")
    for name, record in packet_files.items():
        contents = transport.decoded_file(record, inputs["remote"] / "remote_setup" / inputs["packet"].name / name)
        destination = inputs["packet"] / name
        adapter.require(name not in inputs["packet_hashes"] or adapter.digest(contents) == inputs["packet_hashes"][name],
            "immutable native site preparation changed: " + name)
        adapter.require(not destination.exists() or adapter.contents(destination) == contents, "local packet bytes conflict: " + name)
        writes[destination] = contents
    adapter.require(not set(data_files) - adapter.DATA_FILES, "native site payload exceeded the data allowlist")
    mirrored = {}
    for name, record in data_files.items():
        contents = transport.decoded_file(record, inputs["remote"] / "data" / inputs["unit"].name / name)
        destination = inputs["unit"] / name
        adapter.require(not destination.is_symlink() and destination.resolve().parent == inputs["unit"], "site data destination escaped")
        adapter.require(name == "progress.json" or not destination.exists() or adapter.contents(destination) == contents,
            "immutable local site data conflicts: " + name)
        writes[destination] = contents
        mirrored[str(destination.relative_to(inputs["run"]))] = {"source_path": record["path"], "sha256": record["sha256"]}
    release = writes.get(inputs["packet"] / "release_receipt.json")
    snapshot = {"schema_version": 1, "kind": "aces_full_site_monitor_snapshot", "cluster": "aces", "user": adapter.OWNER,
        "job_id": inputs["identifier"], "origin_host": "portal05", "query_host": payload["query_host"],
        "local_campaign_run_dir": str(inputs["run"]), "remote_campaign_run_dir": str(inputs["remote"]),
        **{key: inputs["receipt"][key] for key in ("plan_sha256", "manifest_sha256", "job_script_sha256")},
        "submission_receipt_sha256": inputs["packet_hashes"]["submission_receipt.json"],
        "release_receipt_sha256": adapter.digest(release) if release is not None else None,
        "observed_at_utc": payload["observed_at_utc"], "squeue": payload["squeue"], "sacct": payload["sacct"], "mirrored_files": mirrored}
    return snapshot, writes


def validate_proposal(inputs, snapshot, writes):
    """Exercise the actual monitor adapter on a complete temporary mirror before publication."""
    with tempfile.TemporaryDirectory(prefix="aces-site-metadata-review-") as temporary:
        staged = Path(temporary) / inputs["run"].name
        (staged / "data" / inputs["unit"].name).mkdir(parents=True)
        for path, value in writes.items():
            destination = staged / path.relative_to(inputs["run"])
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(value)
        for name in ("submitted_jobids_aces.txt", "submissions_aces.tsv"):
            path = staged / "slurm" / name
            path.parent.mkdir(exist_ok=True)
            path.write_bytes(adapter.contents(inputs["run"] / "slurm" / name))
        packet = staged / "remote_setup" / inputs["packet"].name
        (packet / "remote_monitor_snapshot.json").write_text(json.dumps({**snapshot, "local_campaign_run_dir": str(staged)}))
        row, _ = adapter.site_row(adapter.packet_inputs(staged, packet))
        return row


def refresh(run, packet):
    """Publish a complete owned snapshot last, or retain it and save a separate refresh error."""
    run, packet = Path(run).resolve(strict=True), Path(packet).absolute()
    stage, inputs = "local ownership", None
    try:
        inputs = local_inputs(run, packet)
        lock_path = packet / ".site-refresh.lock"
        adapter.require(not lock_path.is_symlink(), "ACES site refresh lock cannot be a symlink")
        with lock_path.open("a+") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            stage = "existing-master metadata observation"
            payload, details = receive(inputs)
            stage = "bounded byte and monitor validation"
            snapshot, writes = proposed_snapshot(inputs, payload)
            row = validate_proposal(inputs, snapshot, writes)
            stage = "atomic publication"
            transport.publish(inputs, snapshot, writes)
            return {"state": "refreshed", "job_id": inputs["identifier"], "observed_state": row["state"],
                "snapshot": str(packet / "remote_monitor_snapshot.json"), "transport": details, "no_job_action": True}
    except Exception as error:
        # The previous snapshot remains authoritative; a refresh failure never supplies a scheduler state.
        directory = run / "20_mins_monitoring/aces_site_refresh_errors"
        directory.mkdir(parents=True, exist_ok=True)
        adapter.ordinary(directory, True)
        path = directory / (datetime.now(timezone.utc).strftime("%Y-%m-%d-%H-%M-%S") + "_" + uuid.uuid4().hex + ".json")
        record = {"schema_version": 1, "kind": "aces_full_site_refresh_error", "at_utc": transport.utc_now(),
            "packet": str(packet), "stage": stage, "error": repr(error), "no_job_action": True}
        transport.atomic_bytes(path, (json.dumps(record, indent=2) + "\n").encode())
        raise RuntimeError("ACES site refresh failed; inspect " + str(path)) from error


def main():
    """Refresh one explicit submitted site packet; never submit, release or cancel a job."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--packet", type=Path, required=True)
    arguments = parser.parse_args()
    print(json.dumps(refresh(arguments.run_dir, arguments.packet), indent=2), flush=True)


if __name__ == "__main__":
    main()
