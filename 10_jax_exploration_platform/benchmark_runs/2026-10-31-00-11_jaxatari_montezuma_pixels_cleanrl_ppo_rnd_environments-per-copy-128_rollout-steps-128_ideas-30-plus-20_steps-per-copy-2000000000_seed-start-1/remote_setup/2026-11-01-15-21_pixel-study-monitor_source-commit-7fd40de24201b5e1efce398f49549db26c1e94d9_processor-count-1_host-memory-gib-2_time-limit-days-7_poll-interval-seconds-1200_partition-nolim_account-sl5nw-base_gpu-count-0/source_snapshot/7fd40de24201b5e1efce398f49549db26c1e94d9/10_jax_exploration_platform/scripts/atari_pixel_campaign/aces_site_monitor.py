"""Read bounded ACES full-site mirrors without querying a scheduler or granting admission."""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re


OWNER = "u.sl359075"
REMOTE_ROOT = Path("/scratch/user/u.sl359075/rnd_atari_pixels/runs")
SOURCE_COMMIT = "d28e6735e79788f618acb0ebfde8609db1f776af"
KINDS = ("identity", "learner", "renderer", "probe")
BOUNDARY = {"workers_permitted": False, "campaign_claims_permitted": False,
    "scientific_hypothesis_credit": 0, "comparison_credit": 0, "idea_credit": 0, "admission_credit": 0}
ARTIFACTS = {"site_validation_plan.json", "job.sbatch", "command.txt", "config_resolved.yaml",
    "experiment_background.md", "software_at_prepare.json"}
DATA_FILES = {"progress.json", "hardware.json", "result.json", "failure.json", "parent_owner.json",
    "parent_allocation_observation.json", "site_assignment.json", "identity_result.json", "probe_result.json"}
DATA_FILES |= {kind + "_" + suffix + ".json" for kind in KINDS
    for suffix in ("owner", "process", "terminal", "allocation_observation")}
MAX_FILE_BYTES = 2 * 1024 * 1024
TERMINAL = {"COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL",
    "PREEMPTED", "BOOT_FAIL", "DEADLINE", "REVOKED"}


def require(condition, message):
    """Reject missing or contradictory metadata instead of inventing a scheduler state."""
    if not condition:
        raise ValueError(message)


def ordinary(path, directory=False):
    """Keep every local metadata path and its ancestors at the exact recorded location."""
    path = Path(path).absolute()
    require(path == path.resolve(strict=True) and (path.is_dir() if directory else path.is_file()),
        "ACES full-site metadata must use an ordinary exact path: " + str(path))
    return path


def contents(path):
    """Read at most the explicit metadata bound, without loading driver images or logs."""
    with ordinary(path).open("rb") as stream:
        value = stream.read(MAX_FILE_BYTES + 1)
    require(len(value) <= MAX_FILE_BYTES, "ACES full-site metadata exceeds its byte bound")
    return value


def read(path):
    """Read one complete bounded JSON metadata record."""
    return json.loads(contents(path))


def digest(value):
    """Hash exact transported bytes without reserializing their original JSON."""
    return hashlib.sha256(value).hexdigest()


def boundary(value):
    """Require literal false permissions and integer zero credit in native site records."""
    require(isinstance(value, dict) and all(type(value.get(key)) is type(expected)
        and value[key] == expected for key, expected in BOUNDARY.items()),
        "ACES full-site evidence cannot grant worker admission or study credit")


def query_commands(identifier):
    """Name only the exact owned ACES identifier in both original raw scheduler queries."""
    return (["squeue", "-h", "-j", identifier, "-o", "%i|%u|%T|%N|%R"],
        ["sacct", "-X", "-n", "-P", "-j", identifier,
         "--format=JobIDRaw,User%64,State%64,ExitCode,NodeList"])


def packet_inputs(run, packet):
    """Bind an actual native packet and submission to this campaign's separate ACES ledger."""
    run, packet = ordinary(run, True), ordinary(packet, True)
    require(packet.parent == run / "remote_setup", "ACES site packet must be a direct campaign child")
    submission = read(packet / "submission_receipt.json")
    identifier = submission.get("job_id")
    require(type(submission.get("schema_version")) is int and submission["schema_version"] == 1
        and submission.get("kind") == "owned_aces_full_site_qualification_submission"
        and submission.get("cluster") == "aces" and isinstance(identifier, str)
        and re.fullmatch(r"[1-9][0-9]*", identifier), "invalid owned ACES full-site submission")
    ledger = run / "slurm/submitted_jobids_aces.txt"
    require(contents(ledger).decode().splitlines().count(identifier) == 1,
        "ACES site job is absent or duplicated in its exact ACES ledger")
    remote = REMOTE_ROOT / run.name
    remote_packet = remote / "remote_setup" / packet.name
    for name, key in (("site_validation_plan.json", "plan_sha256"), ("manifest.yaml", "manifest_sha256"),
                      ("job.sbatch", "job_script_sha256")):
        require(digest(contents(packet / name)) == submission.get(key), "ACES prepared byte receipt changed: " + name)
    plan, manifest = read(packet / "site_validation_plan.json"), read(packet / "manifest.yaml")
    for record in (plan, manifest, submission):
        boundary(record)
    require(type(plan.get("schema_version")) is int and plan["schema_version"] == 1
        and plan.get("kind") == "aces_full_site_qualification" and plan.get("cluster") == "aces"
        and plan.get("campaign_run_dir") == str(remote) and plan.get("source_commit") == SOURCE_COMMIT
        and plan.get("owner") == {"username": OWNER, "uid": 56626}
        and plan.get("data_unit") == {"path": str(remote / "data" / packet.name), "name": packet.name}
        and submission.get("data_unit") == plan["data_unit"], "ACES site plan changed its native owner, source or shard")
    require(type(manifest.get("schema_version")) is int and manifest["schema_version"] == 1
        and manifest.get("kind") == "aces_full_site_qualification_preparation" and manifest.get("cluster") == "aces"
        and set(manifest.get("artifacts", {})) == ARTIFACTS, "ACES site preparation is incomplete")
    shape = {"account": "158590754315", "partition": "gpu", "qos": "normal", "nodes": 1,
        "ntasks": 1, "gres": "gpu:h100:1", "cpus_per_task": 8, "memory": "32G", "time": "03:00:00"}
    require(isinstance(plan.get("job_shape"), dict) and set(plan["job_shape"]) == set(shape)
        and all(type(plan["job_shape"].get(key)) is type(value) and plan["job_shape"][key] == value
                for key, value in shape.items()), "ACES site request differs from the reviewed one-H100 request")
    for name, item in manifest["artifacts"].items():
        require(item == {"path": str(remote_packet / name), "sha256": digest(contents(packet / name))},
            "ACES prepared artifact changed its original path or bytes: " + name)
    require(plan.get("software_at_prepare") == manifest["artifacts"]["software_at_prepare.json"],
        "ACES site software must retain its actual native prepared artifact")
    command = submission.get("held_command")
    required = {"--hold", "--parsable", "--account=158590754315", "--partition=gpu", "--qos=normal",
        "--nodes=1", "--ntasks=1", "--gres=gpu:h100:1", "--cpus-per-task=8", "--mem=32G", "--time=03:00:00"}
    require(isinstance(command, list) and bool(command) and command == manifest.get("submission_command")
        and command[0] == "sbatch" and command[-1] == str(remote_packet / "job.sbatch")
        and required <= set(command), "ACES site must retain its exact held command")
    commit = plan["helpers"].get("source_commit")
    require(isinstance(commit, str) and re.fullmatch(r"[0-9a-f]{40}", commit), "ACES helper needs its full source commit")
    row = [identifier, packet.name, commit, str(remote)]
    matches = [line for line in contents(run / "slurm/submissions_aces.tsv").decode().splitlines()
        if line.split("\t", 1)[0] == identifier]
    require(submission.get("submission_row") == row and matches == ["\t".join(row)]
        and submission.get("ledger_path") == str(remote / "slurm/submitted_jobids_aces.txt"),
        "ACES site submission row identifies another cluster, source or campaign")
    release = None
    if (packet / "release_receipt.json").exists() or (packet / "release_receipt.json").is_symlink():
        release = read(packet / "release_receipt.json")
        require(type(release.get("schema_version")) is int and release["schema_version"] == 1
            and release.get("kind") == "owned_aces_full_site_qualification_release" and release.get("cluster") == "aces"
            and release.get("job_id") == identifier and release.get("command") == ["scontrol", "release", identifier]
            and type(release.get("returncode")) is int and release["returncode"] == 0
            and release.get("submission_receipt_sha256") == digest(contents(packet / "submission_receipt.json")),
            "ACES site release differs from its exact held submission")
    unit = ordinary(run / "data" / packet.name, True)
    return {"run": run, "packet": packet, "unit": unit, "remote": remote, "plan": plan,
        "receipt": submission, "identifier": identifier, "release": release}


def snapshot_inputs(inputs):
    """Validate fresh UTC scheduler observations and every bounded copied data artifact."""
    run, packet, identifier = inputs["run"], inputs["packet"], inputs["identifier"]
    snapshot = read(packet / "remote_monitor_snapshot.json")
    expected = {"schema_version": 1, "kind": "aces_full_site_monitor_snapshot", "cluster": "aces", "user": OWNER,
        "job_id": identifier, "origin_host": "portal05", "query_host": "alogin1.cluster",
        "local_campaign_run_dir": str(run), "remote_campaign_run_dir": str(inputs["remote"]),
        "plan_sha256": inputs["receipt"]["plan_sha256"], "manifest_sha256": inputs["receipt"]["manifest_sha256"],
        "job_script_sha256": inputs["receipt"]["job_script_sha256"],
        "submission_receipt_sha256": digest(contents(packet / "submission_receipt.json")),
        "release_receipt_sha256": digest(contents(packet / "release_receipt.json")) if inputs["release"] else None}
    require(all(type(snapshot.get(key)) is type(value) and snapshot[key] == value for key, value in expected.items()),
        "ACES full-site snapshot identifies another cluster, source or submission")
    observed = datetime.fromisoformat(snapshot["observed_at_utc"].replace("Z", "+00:00"))
    require(observed.tzinfo is not None and observed.utcoffset().total_seconds() == 0,
        "ACES site observation must have a timezone-aware UTC timestamp")
    require(0 <= (datetime.now(timezone.utc) - observed).total_seconds() <= 2400,
        "ACES full-site status is stale or from the future; refresh its exact owned ID")
    parsed = []
    for name, command in zip(("squeue", "sacct"), query_commands(identifier)):
        record = snapshot[name]
        absent = (name == "squeue" and record.get("returncode") == 1 and not record.get("stdout")
            and record.get("stderr", "").strip() == "slurm_load_jobs error: Invalid job id specified")
        require(record.get("command") == command and type(record.get("returncode")) is int
            and (record["returncode"] == 0 or absent), "ACES site scheduler query failed or named another ID")
        rows = [line.split("|") for line in record.get("stdout", "").splitlines() if line]
        require(len(rows) <= 1 and all(len(row) == 5 and row[:2] == [identifier, OWNER] for row in rows),
            "ACES site scheduler records are duplicate, foreign or malformed")
        parsed.append(rows[0] if rows else None)
    if inputs["release"] is None:
        require(parsed[0] and parsed[0][2] == "PENDING" and parsed[0][4] in {"JobHeldUser", "(JobHeldUser)"},
            "unreleased ACES site job needs an actual current JobHeldUser record")
    mirrored = snapshot.get("mirrored_files", {})
    require(isinstance(mirrored, dict), "ACES site mirror needs an exact file roster")
    for relative, item in mirrored.items():
        path = Path(relative)
        require(path.parts[:2] == ("data", packet.name) and len(path.parts) == 3 and path.name in DATA_FILES,
            "ACES site mirror escaped its bounded data shard")
        require(item == {"source_path": str(inputs["remote"] / path), "sha256": digest(contents(run / path))},
            "ACES site copied bytes or original path changed")
    return snapshot, *parsed


def mirrored_record(inputs, snapshot, name):
    """Read only data whose exact original bytes are included in this scheduler snapshot."""
    relative = str(Path("data") / inputs["unit"].name / name)
    return read(inputs["unit"] / name) if relative in snapshot["mirrored_files"] else None


def site_row(inputs):
    """Display actual owned state while reserving full qualification for independent driver verification."""
    snapshot, queued, accounting = snapshot_inputs(inputs)
    row = {"kind": "aces_full_site", "cluster": "aces", "run": "ACES original pixel full-site check",
        "pool": "ACES gpu/" + OWNER, "gpu": ["H100 PCIe requested"], "slurm_job_id": inputs["identifier"],
        "data_unit_name": inputs["unit"].name, "state": "unknown", "qualification_verified": False,
        "initial_case_credit": 0, "scientific_case_credit": 0, "final_training_credit": 0, **BOUNDARY}
    if queued:
        _, _, state, node, reason = queued
        require(state in {"PENDING", "RUNNING", "COMPLETING", "CONFIGURING", "SUSPENDED", "REQUEUED"},
            "ACES site current scheduler state is unsupported")
        row.update(state="held" if reason in {"JobHeldUser", "(JobHeldUser)"} else "waiting" if state in
            {"PENDING", "SUSPENDED", "REQUEUED"} else "running", progress=reason or state)
        if state in {"RUNNING", "COMPLETING", "CONFIGURING"}:
            completed = []
            for kind in KINDS:
                terminal = mirrored_record(inputs, snapshot, kind + "_terminal.json")
                if terminal is not None:
                    require(terminal.get("job_id") == inputs["identifier"] and terminal.get("kind") == kind
                        and type(terminal.get("exit_code")) is int, "ACES stage status belongs to another job")
                    completed.append(kind + (" passed" if terminal["exit_code"] == 0 else " failed"))
            row["progress"] = "; ".join(completed) or "owned site preflight or first stage running"
        row["observed_host"] = node
    else:
        require(accounting is not None, "ACES site has no actual queue or accounting record")
        _, _, state, exit_code, node = accounting
        require(bool(state.split()), "ACES terminal scheduler state is empty")
        state = state.split()[0].removesuffix("+")
        require(state in TERMINAL, "ACES site absent from queue has no terminal accounting proof")
        failure = mirrored_record(inputs, snapshot, "failure.json")
        result = mirrored_record(inputs, snapshot, "result.json")
        require(not (failure is not None and result is not None), "ACES site has conflicting result and failure artifacts")
        if failure is not None:
            boundary(failure)
            require(failure.get("kind") == "aces_full_site_failure" and failure.get("passed") is False,
                "ACES site failure record changed its kind or outcome")
            row.update(state="failed", progress="site failed: " + str(failure.get("error", "inspect raw failure")))
        elif state == "COMPLETED" and exit_code == "0:0" and result is not None:
            boundary(result)
            require(type(result.get("schema_version")) is int and result["schema_version"] == 1
                and result.get("kind") == "aces_full_site_evidence" and result.get("passed") is True
                and result.get("plan") == {"path": str(inputs["remote"] / "remote_setup" / inputs["packet"].name /
                    "site_validation_plan.json"), "sha256": inputs["receipt"]["plan_sha256"]},
                "ACES saved site result is not bound to this exact native plan")
            parent = mirrored_record(inputs, snapshot, "parent_allocation_observation.json")
            assignment = mirrored_record(inputs, snapshot, "site_assignment.json")
            require(parent is not None and assignment is not None and parent["owned_allocation"]["job_id"] == inputs["identifier"]
                and assignment.get("job_id") == inputs["identifier"] and assignment.get("hostname") == node,
                "ACES saved site result lacks its actual owned allocation records")
            for key, name in (("parent_observation", "parent_allocation_observation.json"), ("assignment", "site_assignment.json")):
                require(result.get(key) == {"path": str(inputs["remote"] / "data" / inputs["unit"].name / name),
                    "sha256": digest(contents(inputs["unit"] / name))}, "ACES site result substituted an allocation artifact")
            row.update(state="completed", site_result_reported=True,
                progress="site result saved; independent verification required; zero study credit")
        else:
            row.update(state="failed", progress="site job ended " + state + "; exit " + exit_code + "; no successful bound result")
        row["observed_host"] = node
    row["observed_at_utc"] = snapshot["observed_at_utc"]
    return row, {"packet": str(inputs["packet"]), "snapshot_sha256": digest(contents(inputs["packet"] /
        "remote_monitor_snapshot.json")), "row": row}


def rows(run):
    """Contain missing or invalid optional site mirrors so existing study reports remain available."""
    output, records = [], []
    for path in sorted((Path(run) / "remote_setup").glob("*/submission_receipt.json")):
        words = path.parent.name.lower().replace("_", " ").replace("-", " ")
        if "aces" not in words.split() or "full site" not in words:
            continue
        try:
            row, record = site_row(packet_inputs(run, path.parent))
        except (OSError, ValueError, KeyError, TypeError, AttributeError, IndexError, OverflowError) as error:
            row = {"kind": "aces_full_site", "cluster": "aces", "run": "ACES original pixel full-site check",
                "pool": "ACES gpu/" + OWNER, "gpu": ["H100 PCIe requested"], "state": "unknown",
                "slurm_job_id": "", "data_unit_name": path.parent.name, "qualification_verified": False,
                "scientific_case_credit": 0, "initial_case_credit": 0, "final_training_credit": 0,
                "progress": "H100 PCIe requested; remote site status unknown: " + str(error), **BOUNDARY}
            record = {"packet": str(path.parent), "row": row}
        output.append(row)
        records.append(record)
    return output, records
