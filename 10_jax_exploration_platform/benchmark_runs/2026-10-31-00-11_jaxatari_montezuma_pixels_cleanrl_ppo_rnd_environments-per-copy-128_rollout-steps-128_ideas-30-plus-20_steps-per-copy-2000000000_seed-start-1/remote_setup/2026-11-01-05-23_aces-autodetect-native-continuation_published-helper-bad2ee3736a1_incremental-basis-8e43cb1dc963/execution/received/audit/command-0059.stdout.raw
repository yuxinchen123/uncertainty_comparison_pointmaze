"""Prepare and collect owned ACES allocation metadata without accepting a GPU mapping."""

import argparse
import csv
from datetime import datetime, timezone
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
import time


PREFIX = "10_jax_exploration_platform/scripts/atari_pixel_campaign/"
HELPERS = ("aces_allocation_observation.py", "h100_nvl_allocation_observation.py")
ACCOUNT = "158590754315"
USERNAME = "u.sl359075"
BASIS_UNIT = "aces_reference_copies_1_source_d28e6735e797"
BOUNDARY = {"mapping_accepted": False, "compatibility_verified": False, "workers_permitted": False,
            "campaign_claims_permitted": False, "scientific_hypothesis_credit": 0}
GPU_FIELDS = ("index", "uuid", "name", "pci.bus_id", "memory.total", "driver_version")
STANDARD_FILES = ("experiment_background.md", "manifest.yaml", "config_resolved.yaml", "command.txt")
XML_LIMITATION = {"status": "not_collected", "reason": "Full NVIDIA XML includes process information; no process query is permitted.",
    "constructed_substitute": False, "documentation": "https://docs.nvidia.com/deploy/nvidia-smi/"}


def read(path):
    """Read one explicitly named JSON metadata artifact."""
    return json.loads(Path(path).read_text())


def file_record(path):
    """Hash one explicit small metadata artifact without numerical imports."""
    path = Path(path)
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def write_once(path, contents):
    """Publish new complete bytes without replacing another preparation or observation."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(contents)
        stream.flush()
        os.fsync(stream.fileno())


def json_bytes(value):
    """Use finite JSON for both JSON artifacts and the standard YAML-named records."""
    return (json.dumps(value, indent=2, allow_nan=False) + "\n").encode()


def generic_collector():
    """Load only the adjacent unchanged standard-library hardware/configuration collector."""
    path = Path(__file__).resolve().with_name(HELPERS[1])
    specification = importlib.util.spec_from_file_location("aces_generic_allocation_observations", path)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def allocation_fields(generic):
    """Add only the configured Slurm path to the existing explicit allocation-field list."""
    return (*generic.ALLOCATION_FIELDS, "SLURM_CONF")


def helper_source(root):
    """Require the exact two helper files in one clean detached Git checkout."""
    root = Path(root).resolve(strict=True)
    def git(*arguments):
        """Read only this selected checkout's local Git metadata."""
        return subprocess.check_output(["git", "-C", str(root), *arguments])
    if (Path(git("rev-parse", "--show-toplevel").decode().strip()).resolve() != root
            or git("rev-parse", "--abbrev-ref", "HEAD").decode().strip() != "HEAD"):
        raise ValueError("ACES observation helpers require their exact detached Git root")
    commit = git("rev-parse", "HEAD").decode().strip()
    files = {}
    for name in HELPERS:
        relative = PREFIX + name
        path = root / relative
        if git("status", "--porcelain", "--", relative) or git("show", commit + ":" + relative) != path.read_bytes():
            raise ValueError("ACES observation helper differs from committed bytes: " + name)
        files[relative] = file_record(path)["sha256"]
    return {"source_root": str(root), "source_commit": commit, "files": files}


def submission_rows(path, identifier):
    """Require one exact submission row for an identifier, retaining its original text."""
    rows = [line for line in Path(path).read_text().splitlines() if line.split("\t", 1)[0] == identifier]
    if len(rows) != 1:
        raise ValueError("the exact ACES submission identifier needs one unambiguous recorded row")
    return rows[0]


def resource_basis(run, identifier):
    """Bind the old owned reference's resource records without granting qualification or precision credit."""
    ledger = run / "slurm/submitted_jobids_aces.txt"
    identifiers = ledger.read_text().splitlines()
    if not re.fullmatch(r"[1-9][0-9]*", identifier) or identifier not in identifiers:
        raise ValueError("resource basis must be in this campaign's exact ACES ledger")
    row = submission_rows(run / "slurm/submissions_aces.tsv", identifier)
    fields = row.split("\t")
    if (len(fields) not in (3, 4) or fields[:3] != [identifier, BASIS_UNIT, "d28e6735e79788f618acb0ebfde8609db1f776af"]
            or len(fields) == 4 and fields[3] != str(run)):
        raise ValueError("resource basis must name the original ACES reference submission")
    hardware = run / "data" / BASIS_UNIT / "hardware.json"
    progress = run / "data" / BASIS_UNIT / "progress.json"
    script = run / "slurm/aces_reference_d28e673.sbatch"
    if (read(hardware).get("slurm_job_id") != identifier or read(progress).get("completed") is not True
            or read(progress).get("rounds") != 7):
        raise ValueError("resource basis lacks the original completed owned reference records")
    expected = ("--account=" + ACCOUNT, "--cpus-per-task=8", "--mem=32G", "--gres=gpu:h100:1")
    lines = script.read_text().splitlines()
    if any(sum(line.strip() == "#SBATCH " + item for line in lines) != 1 for item in expected):
        raise ValueError("resource basis changed its original one-H100 eight-CPU 32-GiB request")
    return {"job_id": identifier, "ledger_path": str(ledger), "submission_row": row,
            "hardware": file_record(hardware), "progress": file_record(progress), "job_script": file_record(script),
            "qualification_credit": 0, "precision_verified": False}, ledger.read_bytes()


def prepare(directory, run, source_root, python, basis_job_id="2271806", partition="gpu"):
    """Create a held five-minute metadata command and all standard records without submitting it."""
    directory, run = Path(directory).absolute(), Path(run).resolve(strict=True)
    source = helper_source(source_root)
    if Path(__file__).resolve() != Path(source["source_root"]) / (PREFIX + HELPERS[0]):
        raise ValueError("prepare must run the exact selected committed ACES helper")
    if (directory.parent != run / "remote_setup" or directory.is_symlink() or partition not in ("gpu", "gpu_debug")
            or "allocation observation" not in directory.name.replace("-", " ").replace("_", " ").lower()):
        raise ValueError("use a new named ACES allocation observation directly under campaign remote_setup")
    unit = run / "data" / directory.name
    if directory.exists() or unit.exists():
        raise ValueError("both ACES observation packet and data shard must be new")
    python = Path(python).absolute()
    if not python.is_file() or pwd.getpwuid(os.getuid()).pw_name != USERNAME:
        raise ValueError("prepare requires the existing project Python and actual ACES owner")
    basis, ledger_bytes = resource_basis(run, basis_job_id)
    shape = {"account": ACCOUNT, "partition": partition, "qos": "normal", "nodes": 1, "ntasks": 1,
             "gres": "gpu:h100:1", "cpus_per_task": 8, "memory": "32G", "time": "00:05:00"}
    snapshot = directory / "resource_basis/aces_submitted_jobids.txt"
    basis["ledger_snapshot"] = {"path": str(snapshot), "sha256": hashlib.sha256(ledger_bytes).hexdigest()}
    generic = generic_collector()
    plan = {"schema_version": 1, "kind": "aces_allocation_observation", "cluster": "aces", "campaign_run_dir": str(run),
        "data_unit": {"name": directory.name, "path": str(unit)}, "helpers": source, "python": str(python),
        "owner": {"username": USERNAME, "uid": os.getuid()}, "resource_basis": basis, "job_shape": shape,
        "allocation_fields": list(allocation_fields(generic)), "gpu_xml": XML_LIMITATION, **BOUNDARY}
    command = ["sbatch", "--parsable", "--hold", "--account=" + ACCOUNT, "--partition=" + partition,
        "--qos=normal", "--nodes=1", "--ntasks=1", "--gres=gpu:h100:1", "--cpus-per-task=8", "--mem=32G", "--time=00:05:00",
        "--job-name=ACES_H100_allocation_observation", "--output=" + str(run / "slurm/jobs/aces_allocation_observation_%j.log"),
        str(directory / "job.sbatch")]
    capture = [str(python), "-I", str(Path(source["source_root"]) / (PREFIX + HELPERS[0])), "capture", "--directory", str(directory)]
    script = "#!/bin/bash\nset -euo pipefail\nexport TZ=UTC\nexec " + shlex.join([
        "srun", "--unbuffered", "--ntasks=1", "--cpus-per-task=8", "--cpu-bind=threads", *capture]) + "\n"
    background = ("## Purpose\n\nCollect actual owned ACES scheduler, cgroup and NVIDIA hardware metadata. "
        "Collection does not select a physical card, qualify precision, admit a worker or start training.\n\n"
        "## Key hyperparameters\n\nOne typed H100, eight CPUs, 32 GiB host memory and five minutes. "
        "No learner, policy, seed, rollout or numerical computation is configured.\n\n"
        "## Code and config changes\n\nThe separate ACES guard reads only this campaign's ACES ledger and exact submission/release records. "
        "It retains all hardware CSV rows without equating CUDA visibility, scheduler numbers or device minors. "
        "XML is not collected because full NVIDIA XML includes process information. Missing configuration remains explicit.\n\n"
        f"## Git state\n\nExact helper commit: {source['source_commit']}. Both helper files are bound in the plan. "
        "The unchanged adjacent collector supplies only generic hardware and configured-isolation observations.\n")
    artifacts = {"allocation_observation_plan.json": json_bytes(plan), "config_resolved.yaml": json_bytes(plan),
        "command.txt": (shlex.join(command) + "\n").encode(), "job.sbatch": script.encode(),
        "experiment_background.md": background.encode(), "resource_basis/aces_submitted_jobids.txt": ledger_bytes}
    # Before: metadata values; after: immutable files and exact byte receipts for every prepared input.
    for name, contents in artifacts.items():
        write_once(directory / name, contents)
    manifest = {"schema_version": 1, "kind": "aces_allocation_observation_preparation", "cluster": "aces",
        "submission_command": command, "data_unit": plan["data_unit"], "artifacts": {
            name: file_record(directory / name) for name in artifacts}, **BOUNDARY}
    write_once(directory / "manifest.yaml", json_bytes(manifest))
    for name in STANDARD_FILES:
        write_once(unit / name, (directory / name).read_bytes())
    generic.atomic_json(unit / "progress.json", {"state": "waiting_for_allocation_observation", "kind": "allocation observation",
        "cluster": "aces", "completed": False, **BOUNDARY})
    return {"directory": str(directory), "data_unit": plan["data_unit"], "submitted": False,
            "submission_command": command, **BOUNDARY}


def field(text, name):
    """Require one unambiguous field in the exact scheduler response."""
    values = re.findall(r"(?:^|\s)" + re.escape(name) + r"=([^\s]+)", text)
    if len(values) != 1:
        raise ValueError("missing or ambiguous owned ACES scheduler field: " + name)
    return values[0]


def run_query(command):
    """Retain the exact read-only command and output, including failed metadata queries."""
    result = subprocess.run(command, capture_output=True, text=True, timeout=20)
    return {"command": command, "returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr}


def owned_allocation(plan, directory, environment, cgroup, hostname, *, release_wait_seconds=30):
    """Verify this exact owned ACES job and numeric step before any NVIDIA or hardware query."""
    run = Path(plan["campaign_run_dir"])
    ledger = run / "slurm/submitted_jobids_aces.txt"
    job, step = environment.get("SLURM_JOB_ID", ""), environment.get("SLURM_STEP_ID", "")
    if (not re.fullmatch(r"ac[0-9]+", hostname) or not re.fullmatch(r"[1-9][0-9]*", job)
            or not re.fullmatch(r"[0-9]+", step) or job not in ledger.read_text().splitlines()
            or not all(re.search(r"(?:^|/)" + re.escape(part) + r"(?:/|$)", cgroup, re.MULTILINE)
                       for part in ("job_" + job, "step_" + step))
            or plan["owner"] != {"username": pwd.getpwuid(os.getuid()).pw_name, "uid": os.getuid()}
            or plan["owner"]["username"] != USERNAME):
        raise ValueError("collection requires the exact ACES owner, own ledger, job/step cgroup and compute host")
    manifest = read(directory / "manifest.yaml")
    submission_path = directory / "submission_receipt.json"
    submission = read(submission_path)
    expected_row = [job, directory.name, plan["helpers"]["source_commit"], str(run)]
    if (type(submission.get("schema_version")) is not int or submission.get("schema_version") != 1
            or submission.get("kind") != "owned_aces_allocation_observation_submission"
            or submission.get("cluster") != "aces" or submission.get("job_id") != job
            or submission.get("plan_sha256") != file_record(directory / "allocation_observation_plan.json")["sha256"]
            or submission.get("manifest_sha256") != file_record(directory / "manifest.yaml")["sha256"]
            or submission.get("job_script_sha256") != file_record(directory / "job.sbatch")["sha256"]
            or submission.get("held_command") != manifest["submission_command"] or submission.get("data_unit") != plan["data_unit"]
            or submission.get("ledger_path") != str(ledger) or submission.get("submission_row") != expected_row
            or submission_rows(run / "slurm/submissions_aces.tsv", job) != "\t".join(expected_row)):
        raise ValueError("the exact ACES held submission, preparation and durable ledger row must agree")
    # The release command can start Slurm before its successful return receipt reaches shared storage.
    release_path = directory / "release_receipt.json"
    deadline = time.monotonic() + release_wait_seconds
    while not release_path.exists() and time.monotonic() < deadline:
        time.sleep(.2)
    release = read(release_path)
    if (type(release.get("schema_version")) is not int or release.get("schema_version") != 1
            or release.get("kind") != "owned_aces_allocation_observation_release"
            or release.get("cluster") != "aces" or release.get("job_id") != job
            or release.get("command") != ["scontrol", "release", job]
            or type(release.get("returncode")) is not int or release["returncode"] != 0
            or release.get("submission_receipt_sha256") != file_record(submission_path)["sha256"]):
        raise ValueError("the exact ACES release must succeed and bind its durable submission receipt")
    if (environment.get("SLURM_GPUS_ON_NODE") != "1" or environment.get("SLURM_CPUS_PER_TASK") != "8"
            or environment.get("SLURM_LOCALID") != "0" or len(os.sched_getaffinity(0)) != 8):
        raise ValueError("collection needs the one-GPU eight-CPU task and its actual eight-CPU affinity")
    queries = {"job": run_query(["scontrol", "show", "job", "-d", job]),
               "step": run_query(["scontrol", "show", "step", "-d", job + "." + step]),
               "node": run_query(["scontrol", "show", "node", "-d", hostname])}
    raw = {"schema_version": 1, "kind": "aces_owned_allocation", "cluster": "aces", "job_id": job, "step_id": step,
        "hostname": hostname, "uid": os.getuid(), "pid": os.getpid(), "cgroup": cgroup,
        "process_start_time": Path("/proc/self/stat").read_text().rsplit(")", 1)[1].split()[19],
        "cpu_affinity": sorted(os.sched_getaffinity(0)), "ledger": file_record(ledger),
        "submission": file_record(submission_path), "release": file_record(release_path), "scheduler": queries,
        "allocation_environment": {name: environment.get(name) for name in plan["allocation_fields"]},
        "absent_allocation_fields": [name for name in plan["allocation_fields"] if name not in environment], **BOUNDARY}
    write_once(Path(plan["data_unit"]["path"]) / "owned_allocation.json", json_bytes(raw))
    if any(item["returncode"] != 0 for item in queries.values()):
        raise ValueError("owned ACES scheduler metadata query failed; original responses are preserved")
    job_text, step_text = queries["job"]["stdout"], queries["step"]["stdout"]
    uid = str(os.getuid())
    if (field(job_text, "JobId") != job or field(job_text, "JobState") != "RUNNING"
            or field(job_text, "UserId") != USERNAME + "(" + uid + ")" or field(job_text, "Account") != ACCOUNT
            or field(job_text, "Partition") != plan["job_shape"]["partition"] or field(job_text, "QOS") != "normal"
            or field(job_text, "NodeList") != hostname or field(job_text, "NumNodes") != "1" or field(job_text, "NumCPUs") != "8"
            or field(job_text, "JOB_GRES") != "gpu:h100:1" or field(job_text, "TresPerNode") != "gres/gpu:h100:1"
            or field(step_text, "StepId") != job + "." + step or field(step_text, "State") != "RUNNING"
            or field(step_text, "UserId") not in (uid, USERNAME + "(" + uid + ")")
            or field(step_text, "NodeList") != hostname or field(step_text, "Nodes") != "1"
            or field(step_text, "CPUs") != "8" or field(step_text, "Tasks") != "1"):
        raise ValueError("actual ACES job/step must identify this owner/account/partition and one eight-CPU H100 task")
    resources = dict(item.split("=", 1) for item in field(job_text, "AllocTRES").split(","))
    step_resources = dict(item.split("=", 1) for item in field(step_text, "TRES").split(","))
    if (resources.get("cpu") != "8" or resources.get("node") != "1" or resources.get("mem") not in ("32G", "32768M")
            or any(values.get("gres/gpu") != "1" or values.get("gres/gpu:h100") != "1"
                   or any(value != "1" for name, value in values.items() if name.startswith("gres/gpu:"))
                   for values in (resources, step_resources))):
        raise ValueError("actual ACES job/step resource records must each allocate one typed H100 and the requested job memory")
    return raw


def configured_observations(generic, environment, cgroup):
    """Observe selected running settings and configuration-path candidates without assuming /etc/slurm is active."""
    command = ["scontrol", "show", "config"]
    try:
        query = run_query(command)
    except (OSError, subprocess.SubprocessError) as error:
        query = {"command": command, "returncode": None, "stdout": "", "stderr": "selected configuration query unavailable: " + type(error).__name__}
    keys = (*generic.SLURM_CONFIG_FIELDS, "SLURM_CONF")
    running = {"command": command, "returncode": query["returncode"], "stderr": query["stderr"],
        "status": "readable" if query["returncode"] == 0 else "unavailable",
        **generic.selected_config(query["stdout"] if query["returncode"] == 0 else "", keys)}
    sources = [{"source": name, "path": value} for name, value in (
        ("allowlisted environment SLURM_CONF", environment.get("SLURM_CONF")),
        ("running Slurm SLURM_CONF", running["fields"].get("SLURM_CONF"))) if value]
    directories = sorted({str(Path(row["path"]).parent) for row in sources if Path(row["path"]).is_absolute()})
    # Before: observed slurm.conf paths; after: explicitly labeled adjacent cgroup.conf candidates, never inferred proof.
    candidates = [generic.selected_config_file(Path(directory) / "cgroup.conf", generic.CGROUP_CONFIG_FIELDS)
                  for directory in directories]
    if len(candidates) == 1:
        selected = {**candidates[0], "path_basis": "adjacent to the sole observed absolute SLURM_CONF directory; not independent proof the plugin read this file"}
    else:
        selected = {"status": "not_located" if not candidates else "ambiguous", "file": {"path": None, "status": "not_located"},
                    **generic.selected_config("", generic.CGROUP_CONFIG_FIELDS)}
    return {"node_cgroup_configuration": selected, "node_cgroup_configuration_candidates": candidates,
        "configuration_path_observations": sources, "observed_configuration_directories": directories,
        "running_slurm_configuration": running, "own_placement": generic.own_placement(environment, cgroup)}


def collect(directory, plan):
    """Collect hardware rows without selecting a GPU or equating scheduler and device numbers."""
    generic = generic_collector()
    unit = Path(plan["data_unit"]["path"])
    names = allocation_fields(generic)
    environment = {name: os.environ[name] for name in names if name in os.environ}
    owned = owned_allocation(plan, directory, environment, Path("/proc/self/cgroup").read_text(), socket.gethostname().split(".")[0])
    generic.atomic_json(unit / "progress.json", {"state": "running", "kind": "allocation observation", "cluster": "aces", **BOUNDARY})
    help_record = run_query(["nvidia-smi", "--help-query-gpu"])
    minor_supported = help_record["returncode"] == 0 and bool(re.search(r'(?m)^\s*"?minor_number"?\s*$', help_record["stdout"]))
    fields = list(GPU_FIELDS) + (["minor_number"] if minor_supported else [])
    query = run_query(["nvidia-smi", "--query-gpu=" + ",".join(fields), "--format=csv,noheader,nounits"])
    queries = {"gpu_csv": {**query, "fields": fields}, "minor_number_field": {
        "command": help_record["command"], "returncode": help_record["returncode"], "supported": minor_supported,
        "help_stdout_sha256": hashlib.sha256(help_record["stdout"].encode()).hexdigest(),
        "reason": "explicit installed query field" if minor_supported else "installed query help did not declare minor_number",
        "stderr": help_record["stderr"]}, "gpu_xml": XML_LIMITATION}
    write_once(unit / "nvidia_hardware_queries.json", json_bytes(queries))
    configuration = configured_observations(generic, environment, owned["cgroup"])
    directories = configuration["observed_configuration_directories"]
    inventory = generic.host_inventory(**({"gres_path": Path(directories[0]) / "gres.conf"} if len(directories) == 1 else {}))
    inventory["gpu_gres_mapping"]["path_basis"] = ("adjacent to observed absolute SLURM_CONF directory"
        if len(directories) == 1 else "default inventory candidate only; the active configuration location is unknown or ambiguous")
    observations = {"schema_version": 1, "kind": "aces_allocation_hardware_observations", "cluster": "aces",
        "observed_at_utc": datetime.now(timezone.utc).isoformat(), "plan": file_record(directory / "allocation_observation_plan.json"),
        "owned_allocation": file_record(unit / "owned_allocation.json"), "hostname": owned["hostname"],
        "job_id": owned["job_id"], "step_id": owned["step_id"], "allocation_environment": {
            name: environment.get(name) for name in names},
        "absent_allocation_fields": [name for name in names if name not in environment],
        "hardware_queries": queries, "hardware_query_receipt": file_record(unit / "nvidia_hardware_queries.json"),
        "host_gpu_inventory": inventory, "device_isolation_observations": configuration, **BOUNDARY}
    write_once(unit / "hardware_observations.json", json_bytes(observations))
    rows = list(csv.reader(query["stdout"].splitlines()))
    if query["returncode"] != 0 or not rows or any(len(row) != len(fields) for row in rows):
        raise ValueError("hardware-only NVIDIA CSV query failed or changed its declared field count; raw response is preserved")
    # Before: every actual CSV row; after: named hardware-only fields, with no allocated-card inference.
    hardware = {"cluster": "aces", "hostname": owned["hostname"], "slurm_job_id": owned["job_id"], "slurm_step_id": owned["step_id"],
        "pid": owned["pid"], "process_start_time": owned["process_start_time"], "requested_gpu_count": 1,
        "observed_hardware_rows": [{name: value.strip() for name, value in zip(fields, row)} for row in rows],
        "gpu_query": "", "gpu_uuid_selected": False, **BOUNDARY}
    write_once(unit / "hardware.json", json_bytes(hardware))
    result = {"schema_version": 1, "kind": "aces_allocation_observation_result", "cluster": "aces", "passed": True,
        "job_id": owned["job_id"], "step_id": owned["step_id"], "data_unit": plan["data_unit"],
        "plan": file_record(directory / "allocation_observation_plan.json"), "observations": file_record(unit / "hardware_observations.json"),
        "observations_completed": True, "meaning_of_passed": "metadata collection only", **BOUNDARY}
    write_once(unit / "result.json", json_bytes(result))
    generic.atomic_json(unit / "progress.json", {"state": "completed", "kind": "allocation observation", "cluster": "aces",
        "completed": True, "passed": True, **BOUNDARY})
    return result


def capture(directory):
    """Verify every immutable preparation byte and retain terminal collection failures."""
    directory = Path(directory).resolve(strict=True)
    plan = read(directory / "allocation_observation_plan.json")
    run, unit = Path(plan["campaign_run_dir"]), Path(plan["data_unit"]["path"])
    if (directory.parent != run / "remote_setup" or unit != run / "data" / directory.name or unit.is_symlink() or not unit.is_dir()):
        raise ValueError("ACES observation packet and data shard must remain at their exact prepared paths")
    generic = generic_collector()
    try:
        if (type(plan.get("schema_version")) is not int or plan.get("schema_version") != 1
                or plan.get("kind") != "aces_allocation_observation" or plan.get("cluster") != "aces"
                or any(type(plan.get(name)) is not type(value) or plan.get(name) != value for name, value in BOUNDARY.items())
                or plan.get("allocation_fields") != list(allocation_fields(generic)) or plan.get("gpu_xml") != XML_LIMITATION
                or helper_source(plan["helpers"]["source_root"]) != plan["helpers"]
                or Path(__file__).resolve() != Path(plan["helpers"]["source_root"]) / (PREFIX + HELPERS[0])
                or str(Path(sys.executable).absolute()) != plan["python"]):
            raise ValueError("ACES collection changed its explicit boundary, interpreter or committed helpers")
        manifest = read(directory / "manifest.yaml")
        if (type(manifest.get("schema_version")) is not int or manifest.get("schema_version") != 1
                or manifest.get("kind") != "aces_allocation_observation_preparation" or manifest.get("cluster") != "aces"
                or manifest.get("data_unit") != plan["data_unit"]
                or any(type(manifest.get(name)) is not type(value) or manifest.get(name) != value for name, value in BOUNDARY.items())):
            raise ValueError("ACES preparation manifest changed its exact schema, cluster or collection-only boundary")
        expected = {"allocation_observation_plan.json", "config_resolved.yaml", "command.txt", "job.sbatch",
                    "experiment_background.md", "resource_basis/aces_submitted_jobids.txt"}
        if set(manifest["artifacts"]) != expected:
            raise ValueError("ACES preparation manifest changed its exact artifact roster")
        for name, receipt in manifest["artifacts"].items():
            if receipt != file_record(directory / name):
                raise ValueError("ACES prepared artifact bytes changed: " + name)
        basis = plan["resource_basis"]
        if (basis["ledger_snapshot"] != file_record(directory / "resource_basis/aces_submitted_jobids.txt")
                or basis["job_id"] not in (directory / "resource_basis/aces_submitted_jobids.txt").read_text().splitlines()):
            raise ValueError("ACES resource basis changed its original owned-ledger snapshot")
        for name in ("hardware", "progress", "job_script"):
            if basis[name] != file_record(basis[name]["path"]):
                raise ValueError("ACES original reference resource basis changed: " + name)
        for name in STANDARD_FILES:
            if (unit / name).read_bytes() != (directory / name).read_bytes():
                raise ValueError("ACES data shard preparation differs: " + name)
        return collect(directory, plan)
    except BaseException as error:
        write_once(unit / "failure.json", json_bytes({"schema_version": 1, "kind": "aces_allocation_observation_failure", "cluster": "aces",
            "passed": False, "error": str(error), "exception_type": type(error).__name__,
            "failed_at_utc": datetime.now(timezone.utc).isoformat(), **BOUNDARY}))
        generic.atomic_json(unit / "progress.json", {"state": "failed", "kind": "allocation observation", "cluster": "aces", **BOUNDARY})
        raise


def main():
    """Expose prepare-only metadata and capture inside an already submitted owned allocation."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="mode", required=True)
    preparation = commands.add_parser("prepare")
    preparation.add_argument("--directory", required=True, type=Path)
    preparation.add_argument("--campaign", required=True, type=Path)
    preparation.add_argument("--helper-root", required=True, type=Path)
    preparation.add_argument("--python", required=True, type=Path)
    preparation.add_argument("--basis-job-id", default="2271806")
    preparation.add_argument("--partition", choices=("gpu", "gpu_debug"), default="gpu")
    execution = commands.add_parser("capture")
    execution.add_argument("--directory", required=True, type=Path)
    args = parser.parse_args()
    if args.mode == "prepare":
        result = prepare(args.directory, args.campaign, args.helper_root, args.python, args.basis_job_id, args.partition)
    else:
        result = capture(args.directory)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
