"""Run one explicitly prepared ACES H100 final queue with an owned CPU monitor.

All commands execute locally on ACES; this module has no SSH or authentication
code. Preparation checks existing metadata only. Real admission and checkpoint
validation run inside the campaign's recorded CPU/GPU allocations. Numerical
work belongs entirely to the pinned train_deployed.py subprocess.
"""

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import signal
import socket
import subprocess
import sys
import time
import uuid

import launch_final_training as metadata
import portable_evidence
import prepare_portable_evidence as portable_builder
import post_comparison_validation as validation
import train_deployed

CLUSTER = "aces"
POLICY = "aces_monitor_policy.json"
INFRASTRUCTURE = {"TIMEOUT", "PREEMPTED", "NODE_FAIL"}
TERMINAL_COLUMNS = "JobIDRaw,State,ExitCode,End"
RESOURCE_KEYS = {"account", "partition", "qos", "gres", "cpus_per_task", "host_memory_gib",
                 "walltime_seconds", "checkpoint_reserve_seconds"}


def receipt(path):
    """Record exact small or streaming file identity without loading arrays."""
    path = Path(path).absolute()
    return {"path": str(path), "sha256": metadata.file_digest(path)}


def resolver_for(binding):
    """Use only the explicitly bound transport manifest for original logical evidence paths."""
    item = binding.get("evidence_manifest")
    if item is None:
        return None
    metadata.verify_receipt(item)
    return portable_evidence.ExactPathResolver(item["path"], item["sha256"])


def verify_original(item, resolver):
    """Hash exact physical bytes while retaining the original logical receipt path."""
    evidence = validation.Evidence(resolver)
    if metadata.file_digest(evidence.physical_file(item["path"])) != item["sha256"]:
        raise ValueError("an original deployed evidence file changed: " + item["path"])


def read_original(path, resolver):
    """Read one explicit logical JSON file through its pinned resolver."""
    return validation.Evidence(resolver).read(path)


def verify_binding(run, binding):
    """Bind an explicitly reserved H100 unit, its raw admission and unchanged detached helper source."""
    metadata.verify_receipt(binding["global_descriptor"])
    metadata.verify_receipt(binding["admission_receipt"])
    unit, admitted = metadata.read(binding["global_descriptor"]["path"]), metadata.read(binding["admission_receipt"]["path"])
    resolver = resolver_for(binding)
    spec = read_original(unit["final_specification"]["path"], resolver)
    verify_original(unit["final_specification"], resolver)
    proof = admitted.get("validation", {})
    if (type(unit.get("schema_version")) is not int or unit.get("schema_version") != 1
            or not re.fullmatch(r"[A-Za-z0-9_-]{1,180}", unit.get("unit_id", "")) or unit.get("kind") != "final_training_unit"
            or unit.get("assigned_server") != CLUSTER or unit.get("hardware_class") != "h100"
            or proof.get("kind") != "deployed_training_admission" or admitted.get("unit_id") != unit["unit_id"]
            or proof.get("specification_digest") != metadata.digest(spec)
            or any(admitted.get(name) != value for name, value in spec.items())
            or unit["copy_seeds"] != admitted["copy_seeds"]
            or unit["configuration_digest"] != metadata.digest(admitted["configuration"])
            or unit["runtime_environment_digest"] != metadata.digest(admitted["runtime_environment"])
            or unit["numerical_source"] != {"root": admitted["source_root"], "commit": admitted["source_commit"]}
            or type(unit["checkpoint_seconds"]) is not int or not 0 < unit["checkpoint_seconds"] <= 1200):
        raise ValueError("explicit H100 unit differs from its globally reserved deployed admission")
    for item in proof["admission_records"] + proof["evidence"]:
        verify_original(item, resolver)
    deployment = read_original(spec["deployment_verification"], resolver)
    if (deployment["target"]["server"] != CLUSTER or not deployment["target"]["hostnames"]
            or any(not re.fullmatch(r"ac\d+", host) for host in deployment["target"]["hostnames"])
            or metadata.digest(deployment) != proof["deployment_digest"]):
        raise ValueError("deployed unit names a different target proof")
    seeds = admitted["copy_seeds"]
    if (not seeds or any(type(seed) is not int for seed in seeds)
            or seeds != list(range(seeds[0], seeds[0] + len(seeds)))):
        raise ValueError("each H100 unit requires its one explicit independent seed block")
    root, commit = Path(unit["helpers"]["source_root"]), unit["helpers"]["source_commit"]
    physical = train_deployed.deployment.git_root(str(root), commit, validation.Evidence())
    verified = {}
    for relative, expected in unit["helpers"]["files"].items():
        if not relative.startswith(metadata.PREFIX) or ".." in Path(relative).parts:
            raise ValueError("the pinned deployed helper path is invalid")
        blob = subprocess.check_output(["git", "-C", str(physical), "show", commit + ":" + relative])
        if hashlib.sha256(blob).hexdigest() != expected or metadata.file_digest(root / relative) != expected:
            raise ValueError("an original deployed helper differs from its committed bytes")
        verified[relative] = blob
    required = validation.helper_requirements({metadata.PREFIX + name for name in train_deployed.HELPERS}, verified)
    if not required <= set(unit["helpers"]["files"]):
        raise ValueError("the explicit deployed unit omitted a required original runner helper")
    validator = root / metadata.PREFIX / "validate_deployed_training_completion.py"
    if validator.read_bytes() != subprocess.check_output(["git", "-C", str(root), "show", commit + ":" + metadata.PREFIX + validator.name]):
        raise ValueError("the pinned deployed checkpoint validator changed")
    python = Path(unit["python"])
    if not python.is_absolute() or not python.is_file() or not os.access(python, os.X_OK):
        raise ValueError("the explicit original Python environment is unavailable")
    return unit, admitted, deployment


def resource_shape(run, policy):
    """Require recorded ACES resources and the selected packing memory plus at least 25 percent."""
    shape = policy["resources"]
    if not isinstance(shape, dict) or set(shape) != RESOURCE_KEYS:
        raise ValueError("supply every explicit ACES resource field")
    proof = policy["resource_proof"]
    metadata.verify_receipt(proof["scheduler_template"])
    metadata.verify_receipt(proof["usage_record"])
    directives = {}
    for line in Path(proof["scheduler_template"]["path"]).read_text().splitlines():
        match = re.fullmatch(r"#SBATCH --(account|partition|qos|gres|cpus-per-task)=(\S+)", line)
        if match:
            directives[match.group(1).replace("-", "_")] = match.group(2)
    if ({key: str(shape[key]) for key in directives} != directives
            or {"account", "partition", "qos", "gres", "cpus_per_task"} != set(directives)
            or shape["account"] != "158590754315" or shape["partition"] != "gpu"
            or shape["qos"] != "normal" or shape["gres"] != "gpu:h100:1"
            or type(shape["cpus_per_task"]) is not int or shape["cpus_per_task"] != 8
            or any(type(shape[name]) is not int for name in ("host_memory_gib", "walltime_seconds", "checkpoint_reserve_seconds"))
            or shape["checkpoint_reserve_seconds"] < 1800
            or not 2 * shape["checkpoint_reserve_seconds"] < shape["walltime_seconds"] <= 172800):
        raise ValueError("resources differ from the recorded one-H100 ACES account/partition/QOS and bounded walltime")
    owned = set((run / "slurm/submitted_jobids_aces.txt").read_text().splitlines())
    peak, usage_jobs = 0, set()
    for line in Path(proof["usage_record"]["path"]).read_text().splitlines():
        job, rss = line.rstrip("|").split("|", 1)
        if job.split(".")[0] not in owned:
            raise ValueError("resource usage belongs to a job outside the exact ACES ledger")
        if rss:
            usage_jobs.add(job.split(".")[0])
            match = re.fullmatch(r"(\d+)([KMG])", rss)
            if not match:
                raise ValueError("MaxRSS proof requires the recorded integer Slurm memory units")
            peak = max(peak, int(match.group(1)) * 1024 ** ("KMG".index(match.group(2)) + 1))
    minimum = math.ceil(peak * 5 / (4 * 2 ** 30))
    # Slurm checks the explicit request against the actual resources of the selected node.
    if not peak or shape["host_memory_gib"] < minimum:
        raise ValueError("host memory must cover recorded MaxRSS plus at least 25 percent")
    records = proof.get("measurement_records")
    if not isinstance(records, list) or not records:
        raise ValueError("MaxRSS proof must name the selected target packing measurement results")
    bindings = policy["bindings"].values() if isinstance(policy["bindings"], dict) else policy["bindings"]
    admitted_records = [(*verify_binding(run, binding), resolver_for(binding)) for binding in bindings]
    selected_jobs = set()
    for item in records:
        matched = False
        for unit, admitted, deployed, resolver in admitted_records:
            paths = {row["path"] for row in deployed["executions"] if row["mode"] == "packing-point"}
            if item["path"] not in paths:
                continue
            verify_original(item, resolver)
            measured = read_original(item["path"], resolver)
            if (measured.get("completed") is not True or measured.get("source_commit") != unit["numerical_source"]["commit"]
                    or metadata.digest(measured.get("configuration")) != unit["configuration_digest"]):
                raise ValueError("MaxRSS measurement differs from the selected final source/copy count/configuration")
            selected_jobs.add(measured["hardware"]["slurm_job_id"])
            matched = True
        if not matched:
            raise ValueError("MaxRSS measurement is outside the validated target packing evidence")
    if not usage_jobs <= selected_jobs:
        raise ValueError("MaxRSS rows must belong to those exact selected target packing jobs")
    return dict(shape)


def prepare_local(run, proposal_path):
    """Stage metadata for caller-provided reserved units, leaving all model output directories untouched."""
    run = Path(run).resolve(strict=True)
    proposal = metadata.read(proposal_path)
    if (set(proposal) != {"schema_version", "bindings", "resources", "resource_proof", "cpu_monitor", "evidence_root"}
            or type(proposal["schema_version"]) is not int or proposal["schema_version"] != 1 or not proposal["bindings"]):
        raise ValueError("prepare-local needs explicit reserved bindings, resource evidence and CPU-monitor shape")
    for name in ("experiment_background.md", "manifest.yaml", "config_resolved.yaml", "command.txt"):
        if not (run / name).is_file():
            raise ValueError("the deployed campaign lacks its original launch record: " + name)
    evidence_root = Path(proposal["evidence_root"]).resolve(strict=True)
    if not run.is_relative_to(evidence_root):
        raise ValueError("the explicit common evidence root must contain the local campaign")
    resource_shape(run, proposal)
    monitor = proposal["cpu_monitor"]
    if (set(monitor) != {"partition", "qos", "cpus", "memory_gib", "walltime_seconds"}
            or monitor["partition"] != "cpu" or monitor["qos"] != proposal["resources"]["qos"]
            or type(monitor["cpus"]) is not int or not 1 <= monitor["cpus"] <= 2
            or type(monitor["memory_gib"]) is not int or not 1 <= monitor["memory_gib"] <= 4
            or type(monitor["walltime_seconds"]) is not int or not 3600 <= monitor["walltime_seconds"] <= 259200):
        raise ValueError("use an explicit small ACES CPU monitor with at most three days walltime")
    policy = {"schema_version": 1, "kind": "aces_final_training_monitor_policy", "run_dir": str(run),
              "bindings": {}, **{name: proposal[name] for name in ("resources", "resource_proof", "cpu_monitor", "evidence_root")}}
    seeds, units, cohorts = [], [], []
    with metadata.queue_lock(run) as queue:
        for binding in proposal["bindings"]:
            unit, admitted, deployment = verify_binding(run, binding)
            name = unit["unit_id"]
            if name in policy["bindings"]:
                raise ValueError("an explicit deployed unit was listed twice")
            policy["bindings"][name] = binding
            seeds.extend(unit["copy_seeds"])
            units.append(unit)
            cohorts.append({"helpers": unit["helpers"], "numerical_source": unit["numerical_source"], "configuration": admitted["configuration"],
                "runtime_environment": deployment["actual_runtime_environment"], "backend": deployment["backend"], "target": deployment["target"]})
            existing = queue / "units" / (name + ".json")
            if existing.exists() and metadata.read(existing) != unit:
                raise ValueError("an existing deployed unit cannot change its identity")
            if (run / "data" / name).exists() and not (queue / "status" / (name + ".json")).is_file():
                raise ValueError("existing deployed outputs require their original registered continuation history")
        if len(seeds) != len(set(seeds)):
            raise ValueError("explicit deployed units repeat an independent policy seed")
        if any(metadata.digest(cohort) != metadata.digest(cohorts[0]) for cohort in cohorts):
            raise ValueError("one generic ACES queue requires one exact selected source/helper/backend/target cohort")
        policy["helpers"], policy["target"] = cohorts[0]["helpers"], cohorts[0]["target"]
        if (queue / POLICY).exists() and metadata.read(queue / POLICY) != policy:
            raise ValueError("the explicit ACES policy is immutable; no automatic revision is permitted")
        for unit in units:
            metadata.atomic(queue / "units" / (unit["unit_id"] + ".json"), unit, immutable=True)
            status = queue / "status" / (unit["unit_id"] + ".json")
            if not status.exists():
                metadata.atomic(status, {"state": "ready", "unit_id": unit["unit_id"], "last_job_cluster": CLUSTER, "last_job_id": None})
        metadata.atomic(queue / POLICY, policy, immutable=True)
    return policy



def verify_policy(run, policy):
    """Reject an empty, changed or differently registered local H100 policy before scheduler actions."""
    required = {"schema_version", "kind", "run_dir", "bindings", "resources", "resource_proof", "cpu_monitor", "evidence_root", "helpers", "target"}
    if (not isinstance(policy, dict) or set(policy) != required or type(policy["schema_version"]) is not int
            or policy["schema_version"] != 1 or policy["kind"] != "aces_final_training_monitor_policy"
            or policy["run_dir"] != str(run) or not isinstance(policy["bindings"], dict) or not policy["bindings"]):
        raise ValueError("a complete explicit prepared ACES monitor policy is required")
    if metadata.digest(metadata.read(run / "final_training_queue" / POLICY)) != metadata.digest(policy):
        raise ValueError("the submission policy differs from its immutable prepared on-disk policy")
    root = Path(policy["evidence_root"]).resolve(strict=True)
    if not run.is_relative_to(root):
        raise ValueError("the local campaign is outside its explicit common evidence root")
    resource_shape(run, policy)
    for name, binding in policy["bindings"].items():
        unit, _, deployment = verify_binding(run, binding)
        registered = metadata.read(run / "final_training_queue/units" / (name + ".json"))
        status = metadata.read(run / "final_training_queue/status" / (name + ".json"))
        if (status.get("unit_id") != name or status.get("state") not in {"ready", "running", "checkpointed", "completed", "held"}
                or status.get("last_job_id") and status.get("last_job_cluster") != CLUSTER):
            raise ValueError("a deployed status lacks its exact original ACES unit/cluster identity")
        if (name != unit["unit_id"] or metadata.digest(registered) != metadata.digest(unit)
                or unit["helpers"] != policy["helpers"] or deployment["target"] != policy["target"]):
            raise ValueError("registered H100 unit/source/helper/target differs from its explicit policy")


def owned_compute(run):
    """Require this campaign's exact ACES ledger job, cluster receipt and actual compute cgroup."""
    job = os.environ.get("SLURM_JOB_ID", "")
    owned = set((run / "slurm/submitted_jobids_aces.txt").read_text().splitlines())
    host = socket.gethostname().split(".")[0]
    if (not job.isdecimal() or job not in owned or not re.fullmatch(r"ac\d+", host)
            or not re.search(r"(?:^|/)job_" + re.escape(job) + r"(?:/|$)", Path("/proc/self/cgroup").read_text(), re.MULTILINE)):
        raise ValueError("real deployed validation requires this campaign's owned ACES compute allocation")
    return job


def accounting(run, identifiers):
    """Query only exact ACES IDs and retain the complete scheduler terminal-state rows."""
    identifiers = sorted(set(identifiers))
    if not identifiers:
        return {}, None
    owned = set((run / "slurm/submitted_jobids_aces.txt").read_text().splitlines())
    if any(not identifier.isdecimal() or identifier not in owned for identifier in identifiers):
        raise ValueError("refusing a scheduler query outside the exact ACES job ledger")
    command = ["sacct", "-X", "-j", ",".join(identifiers), "--format=" + TERMINAL_COLUMNS, "-P", "-n"]
    result = subprocess.run(command, capture_output=True, text=True, check=True, env={**os.environ, "TZ": "UTC"})
    rows = {}
    for line in result.stdout.splitlines():
        job, state, code, end = line.rstrip("|").split("|")
        state = state.split()[0].rstrip("+")
        if job not in identifiers or job in rows or state not in metadata.ACTIVE | metadata.TERMINAL:
            raise ValueError("scheduler returned a duplicate, foreign or unknown ACES state")
        rows[job] = {"state": state, "exit_code": code, "end": end, "raw": line}
    current, current_proof = metadata.active_owned_jobs(run, identifiers, CLUSTER)
    for job, state in current.items():
        rows.setdefault(job, {"exit_code": None, "end": None, "raw": None})
        rows[job].update(state=state, active_in_scheduler=True)
    if set(rows) != set(identifiers):
        raise ValueError("an exact ACES scheduler state is unknown; diagnosis is required")
    return rows, {"command": command, "returncode": result.returncode, "stdout": result.stdout,
                  "current_queue": current_proof}


def record_submission(run, job, kind, record):
    """Capture every exact sbatch ID durably before releasing any new monitor or GPU worker."""
    directory = run / "slurm"
    with (directory / "submissions.lock").open("a+") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        for name in ("submitted_jobids_aces.txt", "submitted_jobids.txt", "deployed_training_jobids_aces.txt"):
            with (directory / name).open("a") as ledger:
                ledger.write(job + "\n"); ledger.flush(); os.fsync(ledger.fileno())
        with (directory / "submissions.tsv").open("a") as ledger:
            ledger.write("\t".join((job, kind, record["metadata_commit"], record["script"])) + "\n")
            ledger.flush(); os.fsync(ledger.fileno())



def live_monitor(run, queue):
    """Require a recent heartbeat from an exact running owned ACES CPU-monitor allocation."""
    records = [metadata.read(path) for path in (queue / "jobs").glob("aces_job-*.json")
               if metadata.read(path)["kind"] == "aces_final_cpu_monitor"]
    rows, _ = accounting(run, [item["job_id"] for item in records])
    heartbeat = metadata.read(queue / "aces_monitor_heartbeat.json")
    observed = datetime.fromisoformat(heartbeat["time"]).timestamp()
    if (heartbeat.get("job_id") not in rows or rows[heartbeat["job_id"]]["state"] != "RUNNING"
            or not 0 <= time.time() - observed <= 2400):
        raise ValueError("a live recorded ACES CPU monitor is required before GPU submission")
    return heartbeat



def metadata_source():
    """Pin the complete committed metadata import closure independently from each unit's training identity."""
    path = Path(__file__).resolve()
    root = path.parents[3]
    commit = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    train_deployed.deployment.git_root(str(root), commit, validation.Evidence())
    names = set(train_deployed.HELPERS) | {path.name, "launch_final_training.py", "launch_queue.py", "prepare_portable_evidence.py"}
    return train_deployed.training.source_receipt(root, commit, [metadata.PREFIX + name for name in sorted(names)])


def submit(run, policy, *, monitor=False):
    """Submit one recorded local job; unresolved intents and any active same-kind job prevent repeats."""
    if not re.fullmatch(r"(?:alogin|ac)\d+", socket.gethostname().split(".")[0]):
        raise ValueError("ACES submissions must execute locally on its existing login or compute session")
    if not monitor:
        owned_compute(run)
    with metadata.queue_lock(run) as queue:
        verify_policy(run, policy)
        if not monitor:
            live_monitor(run, queue)
            statuses = [metadata.read(queue / "status" / (name + ".json")) for name in policy["bindings"]]
            if any(row["state"] == "held" for row in statuses) or not any(row["state"] == "ready" for row in statuses):
                raise ValueError("only explicit ready H100 units with no application hold may be submitted")
        kind = "aces_final_cpu_monitor" if monitor else "aces_final_h100_worker"
        if any(metadata.read(path).get("state") == "submitting" for path in (queue / "aces_submissions").glob("*.json")):
            raise ValueError("an unresolved ACES sbatch intent requires exact-ID diagnosis")
        old = [metadata.read(path) for path in (queue / "jobs").glob("aces_job-*.json") if metadata.read(path)["kind"] == kind]
        rows, _ = accounting(run, [record["job_id"] for record in old])
        if any(row["state"] in metadata.ACTIVE for row in rows.values()):
            raise ValueError("an exact owned ACES job of this kind is already active or pending")
        for binding in policy["bindings"].values():
            verify_binding(run, binding)
        shape = resource_shape(run, policy)
        identifier = kind + "_" + uuid.uuid4().hex
        directory = run / "slurm/deployed_training" / identifier
        directory.mkdir(parents=True)
        script_path = Path(__file__).resolve()
        source = metadata_source()
        commit = source["source_commit"]
        plan = {"schema_version": 1, "kind": kind, "run_dir": str(run), "policy": receipt(queue / POLICY),
                "metadata_launcher": receipt(script_path), "metadata_source": source, "metadata_commit": commit, "helpers": policy["helpers"]}
        metadata.atomic(directory / "job_plan.json", plan, immutable=True)
        script = directory / "submit.sh"
        script.write_text("#!/bin/bash\nset -euo pipefail\nexport PYTHONNOUSERSITE=1\nexport PYTHONUNBUFFERED=1\nexec /usr/bin/python3 " + shlex.join([
            str(script_path), "monitor" if monitor else "job", "--plan", str(directory / "job_plan.json")]) + "\n")
        intent = queue / "aces_submissions" / (identifier + ".json")
        metadata.atomic(intent, {"state": "submitting", "kind": kind, "script": str(script)})
        if monitor:
            cpu = policy["cpu_monitor"]
            args = ["--partition=" + cpu["partition"], "--qos=" + cpu["qos"], "--cpus-per-task=" + str(cpu["cpus"]),
                    "--mem=" + str(cpu["memory_gib"]) + "G", "--time=" + str(math.ceil(cpu["walltime_seconds"] / 60))]
        else:
            args = ["--partition=" + shape["partition"], "--qos=" + shape["qos"], "--gres=" + shape["gres"],
                    "--cpus-per-task=" + str(shape["cpus_per_task"]), "--mem=" + str(shape["host_memory_gib"]) + "G",
                    "--time=" + str(math.ceil(shape["walltime_seconds"] / 60)), "--signal=B:USR1@" + str(shape["checkpoint_reserve_seconds"]),
                    "--nodelist=" + ",".join(policy["target"]["hostnames"])]
        command = ["sbatch", "--parsable", "--account=" + shape["account"], "--nodes=1", "--ntasks=1", "--no-requeue",
                   "--job-name=" + kind, "--output=" + str(directory / "job_%j.log"), *args, str(script)]
        output = subprocess.run(command, capture_output=True, text=True, check=True).stdout.strip()
        job = output.split(";", 1)[0]
        if not job.isdecimal():
            raise ValueError("sbatch returned no exact ACES ID; the intent remains held for diagnosis")
        record = {"schema_version": 1, "cluster": CLUSTER, "job_id": job, "kind": kind,
                  "job_plan": receipt(directory / "job_plan.json"), "script": str(script), "metadata_commit": commit}
        print("SUBMITTED_ACES_JOB_ID=" + job, flush=True)
        metadata.atomic(intent, {"state": "submitting", "returned_job_id": job, "kind": kind, "script": str(script)})
        record_submission(run, job, kind, record)
        metadata.atomic(queue / "jobs" / ("aces_job-" + job + ".json"), record, immutable=True)
        metadata.atomic(intent, {**record, "state": "submitted"})
    return record



def unit_snapshot(run, unit):
    """Record every actual output byte and explicit current-attempt absences before strict transport copying."""
    directory = run / "data" / unit["unit_id"]
    progress = metadata.read(directory / "progress.json")
    attempt = progress.get("attempt")
    if type(attempt) is not int or attempt < 1:
        raise ValueError("a deployed output snapshot requires its exact current attempt")
    files = {}
    for path in sorted(directory.rglob("*")):
        if path.is_symlink():
            raise ValueError("deployed output snapshots cannot follow symlinked files")
        if path.is_file():
            files[str(path.relative_to(directory))] = receipt(path)
    candidates = ["result.json", "failure.json", f"attempts/attempt-{attempt:04d}/result.json", f"attempts/attempt-{attempt:04d}/failure.json"]
    destination = run / "final_training_queue/output_snapshots" / (unit["unit_id"] + "_" + uuid.uuid4().hex + ".json")
    metadata.atomic(destination, {"schema_version": 1, "kind": "deployed_training_unit_snapshot", "unit_directory": str(directory),
        "observed_at": datetime.now(timezone.utc).isoformat(), "attempt": attempt, "files": files,
        "absent": [str(directory / name) for name in candidates if name not in files]}, immutable=True)
    return destination


def validation_transport(run, unit, binding, terminal=None):
    """Extend a pinned base capsule with immutable streaming copies of explicit current output receipts."""
    policy = metadata.read(run / "final_training_queue" / POLICY)
    root = Path(policy["evidence_root"]).resolve(strict=True)
    base = resolver_for(binding)
    base_path = Path(binding["evidence_manifest"]["path"])
    manifest = metadata.read(base_path)
    files, git_roots = {}, []
    for item in manifest["files"]:
        physical = base.file_path(item["original_path"]).resolve(strict=True)
        relative = physical.relative_to(root).as_posix()
        portable_builder.target_path(root, relative, exists=True)
        files[item["original_path"]] = {**item, "relative_path": relative}
    for item in manifest["git_roots"]:
        physical = base.git_root(item["original_root"], item["commit"]).resolve(strict=True)
        translated = {**item, "relative_path": physical.relative_to(root).as_posix()}
        portable_builder.verify_git_root(root, translated)
        git_roots.append(translated)
    helper = unit["helpers"]
    if not any(item["original_root"] == helper["source_root"] and item["commit"] == helper["source_commit"] for item in git_roots):
        item = {"original_root": helper["source_root"], "relative_path": Path(helper["source_root"]).relative_to(root).as_posix(),
                "commit": helper["source_commit"]}
        portable_builder.verify_git_root(root, item)
        git_roots.append(item)
    snapshot = unit_snapshot(run, unit)
    snapshots = metadata.read(snapshot)
    pending = [receipt(snapshot), *snapshots["files"].values()]
    if terminal is not None:
        record = metadata.read(terminal)
        claim = metadata.read(record["submission_record"]["path"])
        pending += [receipt(terminal), record["ledger_snapshot"], record["submission_record"], record["terminal_unit_snapshot"],
                    record["accounting"]["stdout"], receipt(record["ledger_path"]), claim["job_plan"], claim["generic_submission"]]
    for item in pending:
        original, expected = portable_evidence.original_path(item["path"]), item["sha256"]
        if original in files:
            if files[original]["sha256"] != expected:
                raise ValueError("live output would replace a pinned original/global evidence identity")
            continue
        target = {"original_path": original, "relative_path": "files/" + expected, "sha256": expected}
        portable_builder.copy_file(root, target)
        files[original] = target
    extended = {"schema_version": 1, "kind": "portable_raw_evidence", "files": list(files.values()), "git_roots": git_roots}
    path = root / ("live_output_manifest_" + uuid.uuid4().hex + ".json")
    metadata.atomic(path, extended, immutable=True)
    portable_evidence.ExactPathResolver(path, metadata.file_digest(path))
    record = {"base_manifest": binding["evidence_manifest"], "extended_manifest": receipt(path), "unit_snapshot": receipt(snapshot)}
    metadata.atomic(snapshot.with_suffix(".transport.json"), record, immutable=True)
    return record


def validation_command(unit, binding, run, *, checkpoint=False, terminal=None, proof_attempt=None, admission_only=False):
    """Invoke the unit's original pinned metadata validator in a fresh process, without numerical imports."""
    helper = "deployed_training_admission.py" if admission_only else "validate_deployed_training_completion.py"
    command = [unit["python"], str(Path(unit["helpers"]["source_root"]) / metadata.PREFIX / helper),
               "--specification", unit["final_specification"]["path"]]
    if not admission_only:
        command += ["--unit-directory", str(run / "data" / unit["unit_id"])]
    if checkpoint:
        command.append("--checkpoint")
    if terminal is not None:
        command += ["--terminal-receipt", str(terminal)]
    if proof_attempt is not None:
        command += ["--checkpoint-proof-attempt", str(proof_attempt)]
    if binding.get("evidence_manifest"):
        if admission_only:
            item = binding["evidence_manifest"]
        else:
            transport = validation_transport(run, unit, binding, terminal)
            item = transport["extended_manifest"]
            if terminal is None:
                command += ["--unit-snapshot", transport["unit_snapshot"]["path"]]
        command += ["--evidence-manifest", item["path"], "--evidence-manifest-sha256", item["sha256"]]
    output = subprocess.run(command, capture_output=True, text=True, check=True,
                            env={**os.environ, "PYTHONNOUSERSITE": "1", "PYTHONUNBUFFERED": "1"})
    return json.loads(output.stdout)


def terminal_receipt(run, unit, job):
    """Record exact owned accounting plus a complete unit-file snapshot after an infrastructure terminal state."""
    rows, raw = accounting(run, [job])
    if rows[job]["state"] not in INFRASTRUCTURE:
        raise ValueError("only exact TIMEOUT/PREEMPTED/NODE_FAIL can use periodic-checkpoint recovery")
    directory = run / "data" / unit["unit_id"]
    progress = metadata.read(directory / "progress.json")
    attempt = progress.get("attempt")
    if type(attempt) is not int or attempt < 1:
        raise ValueError("terminal recovery needs the actual current attempt")
    stamp = datetime.now(timezone.utc).isoformat()
    destination = run / "final_training_queue/terminal_receipts" / (unit["unit_id"] + "_job-" + job + "_" + uuid.uuid4().hex)
    destination.mkdir(parents=True)
    ledger = run / "slurm/submitted_jobids_aces.txt"
    snapshot = destination / "ledger.txt"
    snapshot.write_bytes(ledger.read_bytes())
    stdout = destination / "accounting.txt"
    stdout.write_text(raw["stdout"])
    files = {}
    for path in sorted(directory.rglob("*")):
        if path.is_symlink():
            raise ValueError("terminal unit snapshot cannot follow symlinked training files")
        if path.is_file():
            files[str(path.relative_to(directory))] = receipt(path)
    candidates = ["result.json", "failure.json", f"attempts/attempt-{attempt:04d}/result.json", f"attempts/attempt-{attempt:04d}/failure.json"]
    unit_snapshot = {"schema_version": 1, "kind": "deployed_training_terminal_unit_snapshot", "unit_directory": str(directory),
                     "observed_at": stamp, "attempt": attempt, "files": files, "absent": [str(directory / name) for name in candidates if name not in files]}
    metadata.atomic(destination / "unit_snapshot.json", unit_snapshot, immutable=True)
    claim = run / "final_training_queue/claims" / unit["unit_id"] / ("aces_job-" + job + ".json")
    record = {"schema_version": 1, "kind": "owned_slurm_terminal_receipt", "cluster": CLUSTER, "campaign_run_dir": str(run),
              "job_id": job, "observed_at": stamp, "ledger_path": str(ledger), "ledger_snapshot": receipt(snapshot),
              "submission_record": receipt(claim), "terminal_unit_snapshot": receipt(destination / "unit_snapshot.json"),
              "accounting": {"command": raw["command"], "returncode": raw["returncode"], "stdout": receipt(stdout)}}
    metadata.atomic(destination / "terminal_receipt.json", record, immutable=True)
    return destination / "terminal_receipt.json"


def checkpoint_selector(run, unit, status):
    """Reuse only an explicit prior validated attempt when a resumed process has not replaced its mapped proof."""
    directory = run / "data" / unit["unit_id"]
    checkpoint = metadata.read(directory / "checkpoints/latest.metadata.json")
    if checkpoint.get("iteration") == 0:
        return None
    progress = metadata.read(directory / "progress.json")
    placement = metadata.read(directory / "attempts" / f"attempt-{progress['attempt']:04d}" / "placement_hardware.json")
    if checkpoint.get("loaded_gpu_libraries", {}).get("pid") == placement.get("pid"):
        return None
    prior = status.get("checkpoint_proof_attempt")
    if type(prior) is not int or not 0 < prior < progress["attempt"]:
        raise ValueError("a saved prior-process mapped proof needs its explicit previously validated attempt")
    return prior



def verify_checkpoint_receipt(run, unit, status):
    """Recheck the saved compute validator receipt and every recorded actual unit file before reuse."""
    metadata.verify_receipt(status["checkpoint_validation"])
    proof = metadata.read(status["checkpoint_validation"]["path"])
    directory = run / "data" / unit["unit_id"]
    identity = metadata.read(directory / "training_identity.json")
    if (proof.get("unit_id") != unit["unit_id"] or proof.get("source_commit") != unit["numerical_source"]["commit"]
            or proof.get("configuration_digest") != unit["configuration_digest"] or proof.get("copy_seeds") != unit["copy_seeds"]
            or proof.get("identity_digest") != metadata.digest(identity)
            or proof.get("checkpoint_sha256") != metadata.file_digest(directory / "checkpoints/latest.pkl")):
        raise ValueError("a previously validated deployed checkpoint changed before reuse")
    for item in proof["evidence"]:
        if Path(item["path"]).is_relative_to(directory):
            metadata.verify_receipt(item)
    return proof


def refresh(run, policy):
    """Continue only exact terminal owned jobs through the shared strict checkpoint validator."""
    owned_compute(run)
    queue = run / "final_training_queue"
    jobs = [metadata.read(path) for path in (queue / "jobs").glob("aces_job-*.json")
            if metadata.read(path)["kind"] == "aces_final_h100_worker"]
    states, _ = accounting(run, [item["job_id"] for item in jobs])
    result = []
    for name, binding in policy["bindings"].items():
        unit, _, _ = verify_binding(run, binding)
        path = queue / "status" / (name + ".json")
        status = metadata.read(path)
        if status["state"] in {"ready", "held", "completed"}:
            if status["state"] != "held" and status.get("checkpoint_validation"):
                try:
                    proof = verify_checkpoint_receipt(run, unit, status)
                    if status["state"] == "completed" and proof.get("completed") is not True:
                        raise ValueError("completed deployed status lacks its full checkpoint validation")
                except (ValueError, KeyError, TypeError, OSError) as error:
                    status.update(state="held", reason=f"{type(error).__name__}: {error}")
                    metadata.atomic(path, status)
            elif status["state"] == "completed":
                status.update(state="held", reason="completed deployed status lacks its compute validation receipt")
                metadata.atomic(path, status)
            result.append(status)
            continue
        job = status.get("last_job_id")
        if status.get("last_job_cluster") != CLUSTER or job not in states:
            raise ValueError("a deployed unit lacks its exact original ACES scheduler state")
        if states[job]["state"] in metadata.ACTIVE:
            result.append(status)
            continue
        try:
            terminal = terminal_receipt(run, unit, job) if states[job]["state"] in INFRASTRUCTURE else None
            if states[job]["state"] not in INFRASTRUCTURE | {"COMPLETED"}:
                raise ValueError("application failure, OOM or other terminal state requires diagnosis")
            proof = validation_command(unit, binding, run, checkpoint=True, terminal=terminal,
                                       proof_attempt=checkpoint_selector(run, unit, status))
            proof_path = queue / "validation" / (name + "_job-" + job + "_" + uuid.uuid4().hex + ".json")
            metadata.atomic(proof_path, proof, immutable=True)
            status.update(state="completed" if proof["completed"] else "ready", checkpoint_validation=receipt(proof_path),
                          checkpoint_proof_attempt=proof.get("checkpoint_proof_attempt"))
        except (ValueError, KeyError, TypeError, OSError, subprocess.SubprocessError) as error:
            status.update(state="held", reason=f"{type(error).__name__}: {error}")
        metadata.atomic(path, status)
        result.append(status)
    return {"cluster": CLUSTER, "jobs": {job: row["state"] for job, row in states.items()}, "units": result,
            "counts": {state: sum(row["state"] == state for row in result) for state in ("ready", "running", "checkpointed", "completed", "held")}}



def status_report(run, policy, result):
    """Print the required five status groups from this local queue's exact owned records."""
    lines = ["run or job | pool | gpu | state | progress"]
    groups = {"TRAINING": {"running"}, "WAITING": {"ready", "checkpointed"}, "HELD": {"held"}, "FINISHED": {"completed"}, "PLANNED": set()}
    model = (policy or {}).get("target", {}).get("gpu_model", "unknown").removeprefix("NVIDIA ")
    for section, states in groups.items():
        lines.append(section)
        for unit in result.get("units", []):
            if unit["state"] not in states:
                continue
            path = run / "data" / unit["unit_id"] / "progress.json"
            value = metadata.read(path) if path.exists() else {}
            progress = str(value.get("update", 0)) + "/" + str(metadata.TARGET_UPDATES) + " updates"
            if unit["state"] == "held":
                progress = unit.get("reason", result["reason"])
            lines.append(unit["unit_id"].replace("_", " ") + " | ACES | 1 " + model + " | " + unit["state"].upper() + " | " + progress)
        if section == "HELD" and result["state"] == "held":
            lines.append("ACES continuation | ACES | 1 " + model + " | HELD | " + result["reason"])
        if section == "PLANNED" and result["state"] not in {"held", "finished"}:
            lines.append("ACES continuation | ACES | 1 " + model + " | " + result["state"].upper() + " | " + result["reason"])
    return "\n".join(lines) + "\n"


def tick(run):
    """Perform one owned-compute decision and report holds without repeating failed applications."""
    run = Path(run).resolve(strict=True)
    queue = run / "final_training_queue"
    if not (queue / POLICY).exists():
        return {"state": "disabled", "reason": "no explicit local ACES final policy"}
    owned_compute(run)
    with (queue / "aces_monitor.lock").open("a+") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return {"state": "waiting", "reason": "another owned ACES tick holds the decision lock"}
        policy = None
        try:
            policy = metadata.read(queue / POLICY)
            with metadata.queue_lock(run):
                verify_policy(run, policy)
                snapshot = refresh(run, policy)
                if not snapshot["counts"]["held"]:
                    for status in snapshot["units"]:
                        if status["state"] == "ready":
                            binding = policy["bindings"][status["unit_id"]]
                            unit, admitted, _ = verify_binding(run, binding)
                            if validation_command(unit, binding, run, admission_only=True) != admitted:
                                raise ValueError("ready H100 raw admission changed before allocation submission")
            if snapshot["counts"]["held"]:
                result = {**snapshot, "state": "held", "reason": "a final H100 unit requires diagnosis"}
            elif snapshot["counts"]["completed"] == len(policy["bindings"]):
                result = {**snapshot, "state": "finished", "reason": "all explicit ACES units have validated completion"}
            elif any(state in metadata.ACTIVE for state in snapshot["jobs"].values()):
                result = {**snapshot, "state": "waiting", "reason": "an exact owned ACES H100 job is active or pending"}
            elif snapshot["counts"]["ready"]:
                submission = submit(run, policy)
                result = {**snapshot, "state": "submitted", "reason": "one generic H100 job durably recorded", "submission": submission}
            else:
                raise ValueError("no explicit unit is ready and no owned job explains its state")
        except (ValueError, KeyError, TypeError, OSError, subprocess.SubprocessError) as error:
            result = {"state": "held", "reason": f"{type(error).__name__}: {error}"}
        result["time"] = datetime.now(timezone.utc).isoformat()
        result["owned_job_ids"] = (run / "slurm/submitted_jobids_aces.txt").read_text().splitlines()
        metadata.atomic(queue / "aces_monitor_latest.json", result)
        report = status_report(run, policy, result)
        reports = run / "20_mins_monitoring"
        reports.mkdir(exist_ok=True)
        (reports / "aces_latest_report.md").write_text("# ACES final training status\n\n```text\n" + report + "```\n")
        print(report, flush=True)
        return result


def admitted_job(plan_path):
    """Wait only for this allocation's durable submission record, then prove its exact owned compute identity."""
    plan_path = Path(plan_path).resolve(strict=True)
    plan = metadata.read(plan_path)
    run = Path(plan["run_dir"])
    job = os.environ.get("SLURM_JOB_ID", "")
    record_path = run / "final_training_queue/jobs" / ("aces_job-" + job + ".json")
    deadline = time.monotonic() + 120
    while not record_path.exists() and time.monotonic() < deadline:
        time.sleep(.1)
    record = metadata.read(record_path)
    owned_compute(run)
    if record["job_plan"] != receipt(plan_path):
        raise ValueError("the ACES job differs from its durable prepared plan")
    metadata.verify_receipt(plan["policy"])
    metadata.verify_receipt(plan["metadata_launcher"])
    for relative, expected in plan["metadata_source"]["files"].items():
        if metadata.file_digest(Path(plan["metadata_source"]["source_root"]) / relative) != expected:
            raise ValueError("the frozen ACES metadata helper closure changed")
    return run, job, plan, metadata.read(plan["policy"]["path"]), record_path


def scheduler_end(job):
    """Read this exact owned allocation's end time for startup and checkpoint reserve accounting."""
    text = subprocess.check_output(["scontrol", "show", "job", job, "-o"], text=True, env={**os.environ, "TZ": "UTC"})
    match = re.search(r"(?:^| )EndTime=(\S+)", text)
    if not match or match.group(1) in {"Unknown", "N/A"}:
        raise ValueError("the exact ACES allocation has no usable scheduler end time")
    return datetime.fromisoformat(match.group(1)).replace(tzinfo=timezone.utc).timestamp()


def run_job(plan_path):
    """Claim explicit units under one H100 allocation and delegate every update to its unchanged runner."""
    run, job, plan, policy, record_path = admitted_job(plan_path)
    deadline = scheduler_end(job)
    stopping, child = {"requested": False, "signal": None, "forwarded": False}, None
    def request_stop(signum, frame):
        """Forward checkpoint requests only to this supervisor's actual child process."""
        stopping.update(requested=True, signal=signum)
        if child is not None and child.poll() is None:
            child.send_signal(signum)
            stopping["forwarded"] = True
    for number in (signal.SIGUSR1, signal.SIGTERM, signal.SIGINT):
        signal.signal(number, request_stop)
    while not stopping["requested"]:
        shape = resource_shape(run, policy)
        if deadline - time.time() <= 2 * shape["checkpoint_reserve_seconds"]:
            return
        with metadata.queue_lock(run) as queue:
            paths = [queue / "status" / (name + ".json") for name in policy["bindings"]]
            available = [path for path in paths if metadata.read(path)["state"] == "ready"]
            if not available:
                return
            status_path = available[0]
            status = metadata.read(status_path)
            unstarted = dict(status)
            name, binding = status["unit_id"], policy["bindings"][status["unit_id"]]
            unit, admitted, deployment = verify_binding(run, binding)
            if status.get("checkpoint_validation"):
                verify_checkpoint_receipt(run, unit, status)
            if status.get("last_job_id"):
                if status.get("last_job_cluster") != CLUSTER:
                    raise ValueError("the explicit deployed unit was previously owned by another cluster")
                previous, _ = accounting(run, [status["last_job_id"]])
                if previous[status["last_job_id"]]["state"] not in metadata.TERMINAL:
                    raise ValueError("a deployed unit's previous exact job remains active")
            claim = {"schema_version": 1, "cluster": CLUSTER, "job_id": job, "unit_id": name,
                "training_specification": unit["final_specification"], "helpers": unit["helpers"],
                "job_plan": receipt(plan_path), "generic_submission": receipt(record_path),
                "supervisor_pid": os.getpid(), "supervisor_process_start_time": metadata.process_birth(os.getpid())}
            metadata.atomic(queue / "claims" / name / ("aces_job-" + job + ".json"), claim, immutable=True)
            status.update(state="running", last_job_cluster=CLUSTER, last_job_id=job)
            metadata.atomic(status_path, status)
        try:
            actual = validation_command(unit, binding, run, admission_only=True)
            if actual != admitted:
                raise ValueError("the full deployed raw admission changed since explicit preparation")
            remaining = deadline - time.time() - shape["checkpoint_reserve_seconds"]
            if stopping["requested"] or remaining < shape["checkpoint_reserve_seconds"]:
                with metadata.queue_lock(run):
                    metadata.atomic(status_path, unstarted)
                return
            command = [unit["python"], str(Path(unit["helpers"]["source_root"]) / metadata.PREFIX / "train_deployed.py"),
                "--final-spec", unit["final_specification"]["path"], "--run-dir", str(run), "--unit-id", name,
                "--server", CLUSTER, "--checkpoint-seconds", str(unit["checkpoint_seconds"]), "--maximum-seconds", str(remaining)]
            if binding.get("evidence_manifest"):
                item = binding["evidence_manifest"]
                command += ["--evidence-manifest", item["path"], "--evidence-manifest-sha256", item["sha256"]]
            log = queue / "logs" / (name + "_aces_job-" + job + ".log")
            log.parent.mkdir(exist_ok=True)
            with log.open("ab") as stream:
                environment = {**os.environ, "PYTHONNOUSERSITE": "1", "PYTHONUNBUFFERED": "1"}
                for runtime_name, value in deployment["actual_runtime_environment"].items():
                    if runtime_name in environment and environment[runtime_name] != value:
                        raise ValueError("inherited runtime differs from the explicitly verified target: " + runtime_name)
                    environment[runtime_name] = value
                if stopping["requested"]:
                    with metadata.queue_lock(run):
                        metadata.atomic(status_path, unstarted)
                    return
                child = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT, env=environment)
                status.update(child_pid=child.pid, child_start_time=metadata.process_birth(child.pid))
                with metadata.queue_lock(run):
                    metadata.atomic(status_path, status)
                if stopping["requested"] and not stopping["forwarded"] and child.poll() is None:
                    child.send_signal(stopping["signal"])
                    stopping["forwarded"] = True
                try:
                    code = child.wait(timeout=max(1., deadline - time.time() - shape["checkpoint_reserve_seconds"]))
                except subprocess.TimeoutExpired:
                    # The absolute allocation stop includes raw admission and asset/startup time before _train begins.
                    request_stop(signal.SIGUSR1, None)
                    code = child.wait(timeout=max(1., deadline - time.time() - 60))
            if code:
                raise ValueError("the final deployed application failed; diagnosis is required")
            proof = validation_command(unit, binding, run, checkpoint=True, proof_attempt=checkpoint_selector(run, unit, status))
            proof_path = queue / "validation" / (name + "_job-" + job + "_" + uuid.uuid4().hex + ".json")
            metadata.atomic(proof_path, proof, immutable=True)
            status.update(state="completed" if proof["completed"] else "checkpointed", checkpoint_validation=receipt(proof_path),
                          checkpoint_proof_attempt=proof.get("checkpoint_proof_attempt"))
            with metadata.queue_lock(run):
                metadata.atomic(status_path, status)
            if not proof["completed"]:
                return
        except BaseException as error:
            if child is not None and child.poll() is None:
                metadata.cleanup_children([child], deadline)
            with metadata.queue_lock(run):
                metadata.atomic(status_path, {**status, "state": "held", "reason": f"{type(error).__name__}: {error}"})
            raise
        finally:
            child = None


def run_monitor(plan_path):
    """Run serialized owned-compute ticks every twenty minutes and stop only after this explicit roster finishes."""
    run, job, _, _, _ = admitted_job(plan_path)
    end = scheduler_end(job)
    while True:
        heartbeat = {"job_id": job, "pid": os.getpid(), "process_start_time": metadata.process_birth(os.getpid()),
            "time": datetime.now(timezone.utc).isoformat(), "interval_seconds": 1200, "state": "checking"}
        metadata.atomic(run / "final_training_queue/aces_monitor_heartbeat.json", heartbeat)
        result = tick(run)
        metadata.atomic(run / "final_training_queue/aces_monitor_heartbeat.json", {**heartbeat, "state": result["state"]})
        if result["state"] == "finished":
            return
        if end - time.time() <= 1800:
            metadata.atomic(run / "final_training_queue/aces_monitor_restart_required.json", {
                "job_id": job, "reason": "CPU monitor allocation approaching its end; root observer must start its continuation"})
            return
        time.sleep(1200)


def main():
    """Expose explicit preparation, one local decision and separately recorded CPU/GPU allocation entrypoints."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    preparation = commands.add_parser("prepare-local")
    preparation.add_argument("--run-dir", type=Path, required=True)
    preparation.add_argument("--input", type=Path, required=True)
    for name in ("tick", "start-monitor"):
        command = commands.add_parser(name)
        command.add_argument("--run-dir", type=Path, required=True)
    for name in ("job", "monitor"):
        command = commands.add_parser(name)
        command.add_argument("--plan", type=Path, required=True)
    arguments = parser.parse_args()
    if arguments.command == "prepare-local":
        result = prepare_local(arguments.run_dir, arguments.input)
    elif arguments.command == "tick":
        tick(arguments.run_dir)
        return
    elif arguments.command == "start-monitor":
        run = arguments.run_dir.resolve(strict=True)
        result = submit(run, metadata.read(run / "final_training_queue" / POLICY), monitor=True)
    else:
        (run_job if arguments.command == "job" else run_monitor)(arguments.plan)
        return
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
