"""Submit one reviewed ACES site packet held, retaining durable evidence and never retrying.

The default action validates local metadata. Execution is a separate root action
after publication. The returned remote receipt is kept below execution/received;
the root must append the local ACES ledgers before publishing that receipt.
There is no release, cancellation, authentication or scheduler-query action.
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
import re
import shlex
import socket
import subprocess
import sys


PLAN_SHA = "50d36dc21fe5842ddace8d84b4d7d91d584e0013b9aaaf4f19e2da778e7c6bae"
BASE_SHA = "95e99f9c213b23c6f8f82c8e8123d25d2a71acf3f4166217a872b0d1126e410a"
BOUNDARY = {"workers_permitted": False, "campaign_claims_permitted": False,
    "scientific_hypothesis_credit": 0, "comparison_credit": 0, "idea_credit": 0, "admission_credit": 0}
NUMERICAL = ("jax", "jaxlib", "numpy", "torch", "flax", "optax")
MAX_BYTES = 2 * 1024 * 1024
BOOTSTRAP = '''import hashlib,json,sys
raw=sys.stdin.buffer.readline(2097153)
if len(raw)>2097152 or not raw.endswith(b"\\n"): raise ValueError("bounded submission metadata required")
payload=json.loads(raw)
if hashlib.sha256(payload["driver_source"].encode()).hexdigest()!=sys.argv[1]: raise ValueError("submission source changed")
space={"__name__":"reviewed_aces_held_submission","__file__":"<reviewed held submission>"}
exec(compile(payload["driver_source"],"<reviewed held submission>","exec"),space)
try: space["remote_submit"](payload)
except Exception as error:
 failure={"schema_version":1,"kind":"aces_held_submission_failure","error_type":type(error).__name__,"error":str(error),"automatic_retry_permitted":False}
 sys.stdout.buffer.write((json.dumps(failure)+"\\n").encode());sys.stdout.buffer.flush()
 raise
'''


def digest(raw):
    """Identify exact original file bytes."""
    return hashlib.sha256(raw).hexdigest()


def ordinary(path, directory=False):
    """Reject links, changed ancestors and unexpected ordinary file types."""
    path = Path(path)
    if not path.is_absolute() or path != path.resolve(strict=True) or not (path.is_dir() if directory else path.is_file()):
        raise ValueError("an exact ordinary absolute submission path is required")
    return path


def record(path):
    """Hash exact ordinary inputs incrementally, including the preserved CleanRL bundle."""
    path = ordinary(path)
    hasher, size = hashlib.sha256(), 0
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
            size += len(block)
            if size > 200 * 1024 * 1024:
                raise ValueError("submission input exceeded its explicit size bound")
    return {"path": str(path), "sha256": hasher.hexdigest(), "bytes": size}


def require_record(row):
    """Reject changed bytes before a submission command can run."""
    if set(row) != {"path", "sha256", "bytes"} or type(row["bytes"]) is not int or record(row["path"]) != row:
        raise ValueError("a reviewed submission input changed its exact bytes")


def boundary(value):
    """Require literal false permissions and integer zero credit."""
    if any(type(value.get(key)) is not type(expected) or value[key] != expected for key, expected in BOUNDARY.items()):
        raise ValueError("held site preparation grants no admission or study credit")


def fsync_directory(path):
    """Persist newly created metadata names before continuing."""
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def exclusive_bytes(path, raw):
    """Create and persist one new artifact without replacing any prior evidence."""
    path = Path(path)
    ordinary(path.parent, True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    fsync_directory(path.parent)


def exclusive_json(path, value):
    """Publish exact JSON bytes exclusively and durably."""
    exclusive_bytes(path, (json.dumps(value, indent=2, allow_nan=False) + "\n").encode())


def checked_payload(payload):
    """Bind this one packet, raw preparation receipt and published metadata utilities."""
    if set(payload) != {"plan", "base_source", "driver_source", "publication_commit"} or digest(payload["plan"].encode()) != PLAN_SHA:
        raise ValueError("the exact reviewed held submission payload is required")
    plan = json.loads(payload["plan"])
    boundary(plan)
    if (type(plan.get("schema_version")) is not int or plan["schema_version"] != 1
            or plan.get("kind") != "aces_exact_full_site_held_submission_plan"
            or not re.fullmatch(r"[0-9a-f]{40}", payload["publication_commit"])
            or digest(payload["base_source"].encode()) != BASE_SHA or plan["base_driver"]["sha256"] != BASE_SHA
            or plan["no_release_or_retry"] is not True):
        raise ValueError("held submission changed its exact scope or publication")
    return plan


def validate_packet(plan, packet, unit, *, remote=False):
    """Bind all standard records, raw admission files and the exact native held command."""
    packet, unit = ordinary(packet, True), ordinary(unit, True)
    for row in plan["native_packet_files"]:
        require_record({"path": str(packet / row["relative_path"]), "sha256": row["sha256"], "bytes": row["bytes"]})
    for row in plan["native_data_files"]:
        require_record({"path": str(unit / row["relative_path"]), "sha256": row["sha256"], "bytes": row["bytes"]})
    native = json.loads((packet / "site_validation_plan.json").read_bytes())
    manifest = json.loads((packet / "manifest.yaml").read_bytes())
    for value in (native, manifest):
        boundary(value)
    if (native["python"] != plan["private_python"] or native["helpers"] != plan["helpers"]
            or native["numerical_source"] != plan["numerical_source"] or native["data_unit"] != plan["data_unit"]
            or native["campaign_run_dir"] != plan["native_run"] or native["owner"] != {"username": "u.sl359075", "uid": 56626}
            or manifest["submission_command"] != plan["held_command"]
            or shlex.split((packet / "command.txt").read_text()) != plan["held_command"]
            or plan["held_command"][0:3] != ["sbatch", "--parsable", "--hold"]
            or native["source_commit"] != "d28e6735e79788f618acb0ebfde8609db1f776af"
            or native["helpers"]["source_commit"] != "bad2ee3736a1451094d5d1644b36b8cfc2d91c38"
            or native.get("qualification_profile") != plan.get("qualification_profile")
            or native["qualification_profile"].get("name") != "aces-h100-pcie-driver-580.178.04-autodetect-nvml-configured-isolation"
            or native["qualification_profile"].get("physical_gres_mapping_verified") is not False
            or native["qualification_profile"].get("kernel_device_enforcement_verified") is not False):
        raise ValueError("the held packet changed its source, interpreter, owner or exact command")
    for name in ("held_submission_intent.json", "submission_receipt.json", "release_receipt.json", "held_submission_failure.json"):
        if (packet / name).exists() or (packet / name).is_symlink():
            raise ValueError("a held submission was already attempted; automatic retry is forbidden")
    return native


def local_payload(path, publication_commit=None):
    """Validate local bytes; optionally prove the exact files were published before creating any intent."""
    path = ordinary(path)
    raw = path.read_bytes()
    if digest(raw) != PLAN_SHA:
        raise ValueError("the local held submission plan changed")
    plan = json.loads(raw)
    source = Path(__file__).read_text()
    payload = {"plan": raw.decode(), "driver_source": source,
        "base_source": ordinary(plan["base_driver"]["path"]).read_text(),
        "publication_commit": publication_commit or "0" * 40}
    checked_payload(payload)
    for row in plan["local_files"]:
        require_record(row)
    validate_packet(plan, plan["local_packet"], Path(plan["local_campaign_run_dir"]) / "data" / Path(plan["local_packet"]).name)
    if publication_commit is not None:
        repository = plan["local_repository"]
        def git(arguments):
            result = subprocess.run(["git", "-C", repository, *arguments], capture_output=True, timeout=15)
            if result.returncode:
                raise ValueError("the exact published submission evidence is unavailable")
            return result.stdout
        git(["merge-base", "--is-ancestor", publication_commit, plan["publication_ref"]])
        rows = [(str(Path(row["path"]).relative_to(repository)), Path(row["path"]).read_bytes()) for row in plan["local_files"]]
        rows += [(plan["driver_repository_path"], source.encode()), (plan["plan_repository_path"], raw)]
        for relative, expected in rows:
            if git(["cat-file", "blob", publication_commit + ":" + relative]) != expected:
                raise ValueError("submission evidence differs from its published Git bytes")
    return payload, plan


def submit_once(plan, packet, publication_commit, source_hash):
    """Submit once under both ACES ledger locks, persisting raw return before parsing or appending."""
    packet = ordinary(packet, True)
    run = ordinary(plan["native_run"], True)
    ledger, table = run / "slurm/submitted_jobids_aces.txt", run / "slurm/submissions_aces.tsv"
    streams = []
    try:
        for path in (ledger, table):
            ordinary(path)
            descriptor = os.open(path, os.O_RDWR | os.O_APPEND | os.O_NOFOLLOW)
            stream = os.fdopen(descriptor, "r+", encoding="utf-8")
            streams.append(stream)
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        previous = []
        for stream in streams:
            stream.seek(0)
            value = stream.read()
            if value and not value.endswith("\n"):
                raise ValueError("an existing ACES ledger has an incomplete final line")
            previous.append(value.splitlines())
        if (any(not re.fullmatch(r"[1-9][0-9]*", item) for item in previous[0])
                or len(previous[0]) != len(set(previous[0]))
                or any(not re.fullmatch(r"[1-9][0-9]*", line.split("\t", 1)[0]) for line in previous[1])
                or len(previous[1]) != len({line.split("\t", 1)[0] for line in previous[1]})):
            raise ValueError("the existing ACES ledgers contain ambiguous identifiers")
        intent = {"schema_version": 1, "kind": "aces_full_site_held_submission_intent",
            "at_utc": datetime.now(timezone.utc).isoformat(), "published_preparation_commit": publication_commit,
            "driver_sha256": source_hash, "submission_plan_sha256": PLAN_SHA, "held_command": plan["held_command"],
            "automatic_retry_permitted": False, **BOUNDARY}
        exclusive_json(packet / "held_submission_intent.json", intent)
        stdout_path, stderr_path = packet / "held_submission_stdout.raw", packet / "held_submission_stderr.raw"
        with stdout_path.open("xb") as stdout, stderr_path.open("xb") as stderr:
            fsync_directory(packet)
            try:
                result = subprocess.run(plan["held_command"], stdout=stdout, stderr=stderr, timeout=90)
            except BaseException as error:
                stdout.flush(); os.fsync(stdout.fileno())
                stderr.flush(); os.fsync(stderr.fileno())
                exclusive_json(packet / "held_submission_return.json", {"returncode": None,
                    "error_type": type(error).__name__, "error": str(error), "stdout": record(stdout_path),
                    "stderr": record(stderr_path), "automatic_retry_permitted": False})
                raise
            stdout.flush(); os.fsync(stdout.fileno())
            stderr.flush(); os.fsync(stderr.fileno())
        returned = {"returncode": result.returncode, "stdout": record(stdout_path), "stderr": record(stderr_path),
            "automatic_retry_permitted": False}
        exclusive_json(packet / "held_submission_return.json", returned)
        raw = stdout_path.read_bytes()
        match = re.fullmatch(rb"([1-9][0-9]*)\n?", raw)
        if result.returncode != 0 or match is None:
            raise ValueError("held submission failed or returned an ambiguous ID; raw evidence is preserved")
        identifier = match.group(1).decode()
        row = [identifier, packet.name, plan["helpers"]["source_commit"], str(run)]
        streams[0].seek(0); identifiers = streams[0].read().splitlines()
        streams[1].seek(0); rows = streams[1].read().splitlines()
        if (identifier in identifiers or any(line.split("\t", 1)[0] == identifier for line in rows)
                or any(not re.fullmatch(r"[1-9][0-9]*", item) for item in identifiers)):
            raise ValueError("the new held ID conflicts with the existing exact ACES ledgers")
        for stream, line in zip(streams, (identifier, "\t".join(row))):
            stream.write(line + "\n"); stream.flush(); os.fsync(stream.fileno())
        fsync_directory(ledger.parent)
        receipt = submission_receipt(plan, packet, identifier, publication_commit, source_hash, raw, stderr_path.read_bytes())
        exclusive_json(packet / "submission_receipt.json", receipt)
        return receipt, [packet / name for name in ("submission_receipt.json", "held_submission_intent.json",
            "held_submission_stdout.raw", "held_submission_stderr.raw", "held_submission_return.json")] + [ledger, table]
    finally:
        for stream in reversed(streams):
            stream.close()


def submission_receipt(plan, packet, identifier, publication_commit, source_hash, stdout, stderr):
    """Use the exact native helper and existing monitor's shared held-submission schema."""
    return {"schema_version": 1, "kind": "owned_aces_full_site_qualification_submission", "cluster": "aces",
        "job_id": identifier, "held_command": plan["held_command"],
        "submission_row": [identifier, Path(packet).name, plan["helpers"]["source_commit"], plan["native_run"]],
        "ledger_path": str(Path(plan["native_run"]) / "slurm/submitted_jobids_aces.txt"),
        "data_unit": plan["data_unit"], "plan_sha256": plan["plan_sha256"],
        "manifest_sha256": plan["manifest_sha256"], "job_script_sha256": plan["job_script_sha256"],
        "submitted_at_utc": datetime.now(timezone.utc).isoformat(), "published_preparation_commit": publication_commit,
        "submission_plan_sha256": PLAN_SHA, "submission_driver_sha256": source_hash,
        "stdout": stdout.decode(), "stderr": stderr.decode(), "automatic_retry_permitted": False, **BOUNDARY}


def remote_submit(payload):
    """Verify native metadata with the private Python, then perform the single held submission."""
    plan = checked_payload(payload)
    if (os.getuid() != 56626 or pwd.getpwuid(os.getuid()).pw_name != "u.sl359075"
            or socket.gethostname().split(".")[0] != "alogin1" or str(Path(sys.executable).absolute()) != plan["private_python"]
            or not sys.dont_write_bytecode or any(name in sys.modules for name in NUMERICAL)):
        raise ValueError("held submission requires actual ACES owner/private Python and metadata-only execution")
    packet = Path(plan["native_packet"])
    native = validate_packet(plan, packet, plan["data_unit"]["path"], remote=True)
    for row in [*plan["native_proofs"], *plan["original_inputs"]]:
        require_record(row)
    for relative, expected in plan["helpers"]["files"].items():
        if record(Path(plan["helpers"]["source_root"]) / relative)["sha256"] != expected:
            raise ValueError("a pinned native helper file changed")
    helper_path = Path(plan["helpers"]["source_root"]) / "10_jax_exploration_platform/scripts/atari_pixel_campaign/aces_slurm_site_validation.py"
    specification = importlib.util.spec_from_file_location("native_owned_aces_submission_validation", helper_path)
    helper = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(helper)
    directory, unit, checked = helper.prepared(packet)
    if checked != native:
        raise ValueError("native packet validation returned another preparation")
    helper.checked_software(checked)
    if any(name in sys.modules for name in NUMERICAL):
        raise ValueError("native held submission validation imported a numerical module")
    try:
        receipt, paths = submit_once(plan, packet, payload["publication_commit"], digest(payload["driver_source"].encode()))
    except Exception as error:
        if (packet / "held_submission_intent.json").exists():
            exclusive_json(packet / "held_submission_failure.json", {"schema_version": 1,
                "kind": "aces_held_submission_failure", "error_type": type(error).__name__, "error": str(error),
                "automatic_retry_permitted": False, **BOUNDARY})
        raise
    files = [{**record(path), "name": path.name, "contents_base64": base64.b64encode(path.read_bytes()).decode()} for path in paths]
    sys.stdout.buffer.write((json.dumps({"receipt": receipt, "files": files}) + "\n").encode())
    sys.stdout.buffer.flush()


def accept_response(raw, directory, plan, source_hash, publication_commit):
    """Keep exact returned bytes separate until the root fsyncs the local ACES ledgers."""
    if len(raw) > MAX_BYTES:
        raise ValueError("held submission response exceeds its explicit metadata bound")
    response = json.loads(raw)
    receipt, files = response["receipt"], response["files"]
    boundary(receipt)
    identifier = receipt.get("job_id")
    row = [identifier, Path(plan["native_packet"]).name, plan["helpers"]["source_commit"], plan["native_run"]]
    if (type(receipt.get("schema_version")) is not int or receipt["schema_version"] != 1
            or receipt.get("kind") != "owned_aces_full_site_qualification_submission" or receipt.get("cluster") != "aces"
            or not isinstance(identifier, str) or not re.fullmatch(r"[1-9][0-9]*", identifier)
            or receipt.get("held_command") != plan["held_command"] or receipt.get("submission_row") != row
            or receipt.get("data_unit") != plan["data_unit"] or receipt.get("ledger_path") != str(Path(plan["native_run"]) / "slurm/submitted_jobids_aces.txt")
            or receipt.get("submission_plan_sha256") != PLAN_SHA or receipt.get("submission_driver_sha256") != source_hash
            or receipt.get("published_preparation_commit") != publication_commit
            or any(receipt.get(key) != plan[key] for key in ("plan_sha256", "manifest_sha256", "job_script_sha256"))):
        raise ValueError("the held submission response identifies another packet or source")
    names = {"submission_receipt.json", "held_submission_intent.json", "held_submission_stdout.raw",
        "held_submission_stderr.raw", "held_submission_return.json", "submitted_jobids_aces.txt", "submissions_aces.tsv"}
    if len(files) != len(names) or {item["name"] for item in files} != names:
        raise ValueError("the held submission return omitted or duplicated its exact artifact roster")
    received = directory / "received"
    received.mkdir()
    fsync_directory(directory)
    values = {}
    for item in files:
        expected_parent = Path(plan["native_run"]) / "slurm" if item["name"] in {"submitted_jobids_aces.txt", "submissions_aces.tsv"} else Path(plan["native_packet"])
        value = base64.b64decode(item["contents_base64"], validate=True)
        if (item["path"] != str(expected_parent / item["name"]) or type(item["bytes"]) is not int
                or len(value) != item["bytes"] or digest(value) != item["sha256"]):
            raise ValueError("returned held submission bytes changed their exact remote identity")
        exclusive_bytes(received / item["name"], value)
        values[item["name"]] = value
    if (json.loads(values["submission_receipt.json"]) != receipt
            or values["submitted_jobids_aces.txt"].decode().splitlines().count(identifier) != 1
            or [line for line in values["submissions_aces.tsv"].decode().splitlines() if line.split("\t", 1)[0] == identifier] != ["\t".join(row)]
            or values["held_submission_stdout.raw"] != receipt["stdout"].encode()):
        raise ValueError("the remote held ID is not durable in both exact returned ACES ledgers")
    return receipt


def local_execute(payload, plan):
    """Create one local intent and use the existing master once; never update ledgers or release locally."""
    if socket.gethostname().split(".")[0] != "portal05" or pwd.getpwuid(os.getuid()).pw_name != "sl5nw":
        raise ValueError("held submission may start only from the owned portal05 session")
    directory = Path(plan["receipt_directory"])
    ordinary(directory.parent, True)
    directory.mkdir()
    fsync_directory(directory.parent)
    source_hash = digest(payload["driver_source"].encode())
    intent = {"schema_version": 1, "kind": "aces_full_site_held_submission_local_intent",
        "at_utc": datetime.now(timezone.utc).isoformat(), "published_preparation_commit": payload["publication_commit"],
        "driver_sha256": source_hash, "submission_plan_sha256": PLAN_SHA, "native_packet": plan["native_packet"],
        "held_command": plan["held_command"], "automatic_retry_permitted": False, **BOUNDARY}
    exclusive_json(Path(plan["local_packet"]) / "held_submission_intent.json", intent)
    exclusive_json(directory / "local_intent.json", intent)
    command = ["ssh", "-T", "-o", "ControlMaster=no", "-o", "ProxyCommand=false", "-o", "BatchMode=yes",
        "-o", "NumberOfPasswordPrompts=0", "-o", "ConnectionAttempts=1", "-o", "ConnectTimeout=10", "aces",
        "exec " + shlex.join([plan["private_python"], "-I", "-B", "-c", BOOTSTRAP, source_hash])]
    raw_payload = (json.dumps(payload) + "\n").encode()
    if len(raw_payload) > MAX_BYTES:
        raise ValueError("the fixed held submission header exceeds its metadata bound")
    exclusive_json(directory / "transport_invocation.json", {"command": command, "master_separately_checked": False,
        "authentication_retry_permitted": False, "source_sha256": source_hash})
    with (directory / "stdout.raw").open("xb") as stdout, (directory / "stderr.raw").open("xb") as stderr:
        try:
            result = subprocess.run(command, input=raw_payload, stdout=stdout, stderr=stderr, timeout=600)
        except BaseException as error:
            stdout.flush(); os.fsync(stdout.fileno())
            stderr.flush(); os.fsync(stderr.fileno())
            exclusive_json(directory / "transport_result.json", {"returncode": None,
                "error_type": type(error).__name__, "error": str(error), "stdout": record(directory / "stdout.raw"),
                "stderr": record(directory / "stderr.raw"), "automatic_retry_permitted": False})
            raise
        finally:
            stdout.flush(); os.fsync(stdout.fileno())
            stderr.flush(); os.fsync(stderr.fileno())
    exclusive_json(directory / "transport_result.json", {"returncode": result.returncode,
        "stdout": record(directory / "stdout.raw"), "stderr": record(directory / "stderr.raw"), "automatic_retry_permitted": False})
    if result.returncode:
        raise ValueError("held submission transport failed; preserve intent and raw evidence, never retry")
    return accept_response((directory / "stdout.raw").read_bytes(), directory, plan, source_hash, payload["publication_commit"])


def main():
    """Describe validated metadata, or let the root execute the published driver once."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--publication-commit")
    parser.add_argument("--execute", action="store_true")
    arguments = parser.parse_args()
    if arguments.execute and arguments.publication_commit is None:
        raise ValueError("execution requires the exact already published preparation commit")
    payload, plan = local_payload(arguments.plan, arguments.publication_commit)
    if arguments.execute:
        receipt = local_execute(payload, plan)
        print(json.dumps({"job_id": receipt["job_id"], "state": "held submission recorded remotely",
            "root_must_append_local_aces_ledgers_before_receipt_publication": True, "release_performed": False}))
    else:
        print(json.dumps({"plan_sha256": PLAN_SHA, "held_command": plan["held_command"],
            "native_python": plan["private_python"], "network_calls": 0, "submissions": 0}, indent=2))


if __name__ == "__main__":
    main()
