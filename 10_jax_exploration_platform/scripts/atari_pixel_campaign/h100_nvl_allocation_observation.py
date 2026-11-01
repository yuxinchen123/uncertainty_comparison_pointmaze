"""Prepare or collect owned H100 NVL allocation metadata without accepting a GPU mapping."""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import socket
import stat
import subprocess
import tempfile

RELATIVE_PATH = "10_jax_exploration_platform/scripts/atari_pixel_campaign/h100_nvl_allocation_observation.py"
HOSTS = ("serval06", "serval07", "serval08", "serval09")
ALLOCATION_FIELDS = ("SLURM_JOB_ID", "SLURM_STEP_ID", "SLURM_JOB_GPUS", "SLURM_STEP_GPUS",
    "SLURM_GPUS_ON_NODE", "SLURM_CPUS_PER_TASK", "SLURM_CPU_BIND_TYPE", "SLURM_CPU_BIND_LIST",
    "SLURM_CPU_BIND", "SLURM_JOB_NODELIST", "SLURM_NODELIST", "SLURM_STEP_NODELIST", "SLURMD_NODENAME",
    "SLURM_NTASKS", "SLURM_PROCID", "SLURM_LOCALID", "SLURM_NODEID", "SLURM_CPUS_ON_NODE",
    "SLURM_GPUS", "SLURM_GPUS_PER_NODE", "SLURM_GPUS_PER_TASK", "SLURM_GPU_BIND", "SLURM_TRES_BIND",
    "CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER", "NVIDIA_VISIBLE_DEVICES")
CGROUP_CONFIG_FIELDS = ("CgroupPlugin", "ConstrainDevices", "ConstrainCores", "CgroupMountpoint",
                        "IgnoreSystemd", "IgnoreSystemdOnFailure")
SLURM_CONFIG_FIELDS = ("TaskPlugin", "ProctrackType")
BOUNDARY = {"mapping_accepted": False, "compatibility_verified": False, "workers_permitted": False,
            "campaign_claims_permitted": False, "scientific_hypothesis_credit": 0}
PCI_ADDRESS = re.compile(r"[0-9a-fA-F]{4,8}:[0-9a-fA-F]{2}:[0-9a-fA-F]{2}\.[0-7]")


def read(path):
    """Read one explicitly named metadata record."""
    return json.loads(Path(path).read_text())


def digest(path):
    """Hash one small metadata or helper file without importing experiment packages."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def file_record(path):
    """Bind an exact physical metadata path and its original bytes."""
    return {"path": str(path), "sha256": digest(path)}


def write_once(path, value):
    """Publish a new metadata file without replacing an earlier observation."""
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def atomic_json(path, value):
    """Replace only this observation unit's current progress or hardware record atomically."""
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, prefix=path.name + ".", delete=False) as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(stream.name, path)


def helper_source(root):
    """Require the exact committed helper in a detached checkout before staging a command."""
    root = Path(root).resolve()
    def git(*arguments):
        """Read only the explicitly chosen helper checkout."""
        return subprocess.check_output(["git", "-C", str(root), *arguments])
    if Path(git("rev-parse", "--show-toplevel").decode().strip()).resolve() != root:
        raise ValueError("helper root must be the exact Git checkout root")
    commit = git("rev-parse", "HEAD").decode().strip()
    if git("rev-parse", "--abbrev-ref", "HEAD").decode().strip() != "HEAD":
        raise ValueError("helper checkout must be detached")
    path = root / RELATIVE_PATH
    if git("status", "--porcelain=v1", "--", RELATIVE_PATH) or git("show", commit + ":" + RELATIVE_PATH) != path.read_bytes():
        raise ValueError("allocation observation helper differs from its committed bytes")
    return {"source_root": str(root), "source_commit": commit, "files": {RELATIVE_PATH: digest(path)}}


def resource_basis(run, submission_path):
    """Use an existing owned site receipt as the explicit resource source, without admitting its results."""
    submission_path = Path(submission_path).resolve()
    submission = read(submission_path)
    identifiers = (run / "slurm/submitted_jobids_uva.txt").read_text().splitlines()
    job = submission.get("job_id")
    if (submission.get("kind") != "owned_h100_nvl_site_validation_submission" or submission.get("cluster") != "uva"
            or not isinstance(job, str) or not re.fullmatch(r"[1-9][0-9]*", job) or job not in identifiers):
        raise ValueError("resource source must be an exact campaign-owned UVA site submission")
    plan_path = submission_path.parent / "site_validation_plan.json"
    script_path = submission_path.parent / "job.sbatch"
    if digest(plan_path) != submission["plan_sha256"] or digest(script_path) != submission["job_script_sha256"]:
        raise ValueError("resource source plan or script changed")
    source_plan = read(plan_path)
    if source_plan["campaign_run_dir"] != str(run):
        raise ValueError("resource source belongs to another campaign")
    shape = source_plan["job_shape"]
    if (shape["nodes"] != 1 or shape["ntasks"] != 1 or shape["gres"] != "gpu:nvidia_h100_nvl:1"
            or shape["cpus_per_task"] != 8 or shape["memory"] != "64G" or shape["eligible_nodes"] != list(HOSTS)):
        raise ValueError("resource source must request one H100 NVL, eight CPUs and 64 GiB on eligible hosts")
    constraints = [argument for argument in submission["held_command"] if argument.startswith("--constraint=")]
    if len(constraints) != 1:
        raise ValueError("resource source must record its exact hardware constraint")
    return {"submission": file_record(submission_path), "plan": file_record(plan_path),
            "job_script": file_record(script_path), "job_id": job}, {
        **shape, "time": "00:05:00", "constraint": constraints[0].split("=", 1)[1]}


def prepare(directory, run, source_root, submission_path, python):
    """Write immutable prepare-only artifacts for one held five-minute metadata allocation."""
    directory, run = Path(directory).resolve(), Path(run).resolve()
    source = helper_source(source_root)
    if Path(__file__).resolve() != Path(source["source_root"]) / RELATIVE_PATH:
        raise ValueError("prepare must execute the exact committed selected helper")
    basis, shape = resource_basis(run, submission_path)
    if not Path(python).is_absolute() or not Path(python).is_file():
        raise ValueError("metadata interpreter must be an existing absolute path")
    if directory.parent != run / "remote_setup":
        raise ValueError("prepared observations must be a direct child of campaign remote_setup")
    if "allocation observation" not in directory.name.replace("-", " ").replace("_", " ").lower():
        raise ValueError("the observation unit name must explicitly say allocation observation")
    unit = run / "data" / directory.name
    if directory.exists() or unit.exists():
        raise ValueError("prepared observations and their data shard must both be new")
    plan = {"schema_version": 1, "kind": "h100_nvl_allocation_observation", "campaign_run_dir": str(run),
        "data_unit": {"path": str(unit), "name": directory.name},
        "helpers": source, "python": str(python), "resource_basis": basis, "job_shape": shape,
        "allocation_fields": list(ALLOCATION_FIELDS), "purpose": "hardware observations only; no alternative mapping is accepted",
        **BOUNDARY}
    command = ["sbatch", "--parsable", "--hold", "--account=" + shape["account"], "--partition=" + shape["partition"],
        "--qos=" + shape["qos"], "--nodes=1", "--ntasks=1", "--constraint=" + shape["constraint"],
        "--gres=gpu:nvidia_h100_nvl:1", "--cpus-per-task=8", "--mem=64G", "--time=00:05:00",
        "--job-name=H100_NVL_allocation_observation",
        "--output=" + str(run / "slurm/jobs/h100_nvl_allocation_observation_%j.log"), str(directory / "job.sbatch")]
    capture_command = [str(python), "-I", str(Path(source["source_root"]) / RELATIVE_PATH), "capture", "--directory", str(directory)]
    script = "#!/bin/bash\nset -euo pipefail\nexport TZ=UTC\nexec " + shlex.join([
        "srun", "--unbuffered", "--ntasks=1", "--cpus-per-task=8", "--cpu-bind=threads", *capture_command]) + "\n"
    background = ("## Purpose\n\nRecord actual Slurm allocation identifiers and NVIDIA GPU hardware inventory. "
        "No GPU mapping, site compatibility or source admission is accepted by this collection.\n\n"
        "## Key hyperparameters\n\nOne H100 NVL, eight CPUs, 64 GiB host memory and five minutes. "
        "No learner, policy, seed, rollout or numerical computation is configured.\n\n"
        "## Code and config changes\n\nA standalone standard-library helper records allowlisted allocation and hardware fields. "
        "The learner, failed site packets, queues, gates and policies remain unchanged.\n\n"
        "## Git state\n\nDetached helper commit: " + source["source_commit"] + ". "
        "The executed helper file matches its committed bytes.\n")
    directory.mkdir()
    write_once(directory / "allocation_observation_plan.json", plan)
    write_once(directory / "config_resolved.yaml", plan)
    (directory / "job.sbatch").write_text(script)
    (directory / "command.txt").write_text(shlex.join(command) + "\n")
    (directory / "experiment_background.md").write_text(background)
    manifest = {"schema_version": 1, "kind": "prepared_h100_nvl_allocation_observation", "state": "prepared not submitted",
        "helpers": source, "submission_command": command,
        "artifacts": {name: file_record(directory / name) for name in (
            "allocation_observation_plan.json", "config_resolved.yaml", "job.sbatch", "command.txt", "experiment_background.md")}, **BOUNDARY}
    write_once(directory / "manifest.yaml", manifest)
    unit.mkdir()
    for filename in ("experiment_background.md", "manifest.yaml", "config_resolved.yaml", "command.txt"):
        (unit / filename).write_bytes((directory / filename).read_bytes())
    atomic_json(unit / "progress.json", {"state": "waiting_for_allocation_observation", "kind": "allocation observation", **BOUNDARY})
    return {"directory": str(directory), "plan": file_record(directory / "allocation_observation_plan.json"),
            "manifest": file_record(directory / "manifest.yaml"), "data_unit": str(unit), "submitted": False, **BOUNDARY}


def owned_allocation(plan, directory, environment, cgroup, hostname):
    """Reject foreign jobs, extra GPUs and ineligible hosts before any NVIDIA or host hardware query."""
    identifier = environment.get("SLURM_JOB_ID", "")
    run = Path(plan["campaign_run_dir"])
    ledger = run / "slurm/submitted_jobids_uva.txt"
    identifiers = ledger.read_text().splitlines()
    if (hostname not in HOSTS or not re.fullmatch(r"[1-9][0-9]*", identifier) or identifier not in identifiers
            or not re.search(r"(?:^|/)job_" + re.escape(identifier) + r"(?:/|$)", cgroup, re.MULTILINE)):
        raise ValueError("collection requires this campaign's exact UVA job, cgroup and eligible compute host")
    submission = read(directory / "submission_receipt.json")
    if (submission.get("kind") != "owned_h100_nvl_allocation_observation_submission" or submission.get("cluster") != "uva"
            or submission.get("job_id") != identifier or submission.get("plan_sha256") != digest(directory / "allocation_observation_plan.json")
            or submission.get("job_script_sha256") != digest(directory / "job.sbatch")):
        raise ValueError("the current allocation is not this exact prepared observation submission")
    selector = environment.get("CUDA_VISIBLE_DEVICES", "")
    if (not re.fullmatch(r"(?:[0-9]+|GPU-[0-9a-fA-F-]+)", selector)
            or environment.get("SLURM_GPUS_ON_NODE") != "1" or environment.get("SLURM_CPUS_PER_TASK") != "8"):
        raise ValueError("collection requires one explicit visible GPU and its eight-CPU task")
    for name in ("SLURM_JOB_GPUS", "SLURM_STEP_GPUS"):
        if name in environment and not re.fullmatch(r"(?:[0-9]+|GPU-[0-9a-fA-F-]+)", environment[name]):
            raise ValueError("allocation must not contain a GPU list or range: " + name)
    command = ["scontrol", "show", "job", "-d", identifier]
    result = subprocess.run(command, capture_output=True, text=True, check=True, timeout=20)
    text = result.stdout
    def field(name):
        """Require a unique scheduler field rather than accepting another job's row."""
        values = re.findall(r"(?:^|\s)" + re.escape(name) + r"=([^\s]+)", text)
        if len(values) != 1:
            raise ValueError("missing or ambiguous owned scheduler field: " + name)
        return values[0]
    if (field("JobId") != identifier or field("JobState") != "RUNNING" or field("NodeList") != hostname
            or field("NumNodes") != "1" or field("NumCPUs") != "8" or field("JOB_GRES") != "gpu:nvidia_h100_nvl:1"
            or field("TresPerNode") != "gres/gpu:nvidia_h100_nvl:1"):
        raise ValueError("owned scheduler row must allocate one typed H100 NVL on this exact host")
    allocated = dict(item.split("=", 1) for item in field("AllocTRES").split(","))
    if allocated.get("gres/gpu") != "1" or any(value != "1" for name, value in allocated.items() if name.startswith("gres/gpu:")):
        raise ValueError("owned allocation must contain exactly one GPU")
    lines = [line.strip() for line in text.splitlines() if re.search(r"\bGRES=", line)]
    if len(lines) != 1 or not re.search(r"\bNodes=" + re.escape(hostname) + r"\s", lines[0]) or not re.search(
            r"\bGRES=gpu:nvidia_h100_nvl:1\(IDX:[0-9]+\)(?:\s|$)", lines[0]):
        raise ValueError("owned detailed allocation must contain one exact typed GPU index")
    return {"job_id": identifier, "hostname": hostname, "cgroup": cgroup, "ledger": file_record(ledger),
            "ledger_identifiers": identifiers, "scheduler": {"command": command, "stdout": text, "returncode": result.returncode},
            "visible_selector": selector}


def read_hardware_file(path):
    """Record unavailable hardware metadata explicitly instead of inventing missing values."""
    try:
        return {"status": "readable", "text": path.read_text()}
    except (FileNotFoundError, PermissionError) as error:
        return {"status": "absent" if isinstance(error, FileNotFoundError) else "permission denied"}


def path_metadata(path):
    """Record metadata for an explicitly selected configuration or own-cgroup path."""
    try:
        information = path.stat()
        return {"path": str(path), "status": "readable", "owner_uid": information.st_uid,
                "owner_gid": information.st_gid, "mode": oct(stat.S_IMODE(information.st_mode)),
                "mtime_ns": information.st_mtime_ns, "inode": information.st_ino, "device": information.st_dev}
    except (FileNotFoundError, PermissionError) as error:
        return {"path": str(path), "status": "absent" if isinstance(error, FileNotFoundError) else "permission denied",
                "error": "cannot inspect the explicitly selected path: " + type(error).__name__}


def selected_config(text, names):
    """Hash and save only explicit allowed configuration entries, never the remaining configuration."""
    fields = {}
    for line in text.splitlines():
        key, separator, value = line.partition("=")
        key = key.strip()
        if separator and key in names:
            if key in fields:
                raise ValueError("duplicate selected configuration field: " + key)
            fields[key] = value.split("#", 1)[0].strip()
    filtered = "".join(name + "=" + fields[name] + "\n" for name in names if name in fields)
    return {"selected_keys": list(names), "fields": fields, "missing_fields": [name for name in names if name not in fields],
            "filtered_bytes": filtered, "sha256": hashlib.sha256(filtered.encode()).hexdigest(),
            "hash_scope": "only the selected entries, in the listed key order, encoded as UTF-8 key=value lines"}


def selected_config_file(path, names):
    """Describe unavailable node-local configuration explicitly and retain only selected readable bytes."""
    contents = read_hardware_file(path)
    record = {"file": path_metadata(path), "status": contents["status"], **selected_config(contents.get("text", ""), names)}
    if contents["status"] != "readable":
        record["error"] = "cannot read the selected node-local configuration: " + contents["status"]
    return record


def running_slurm_config():
    """Read public running Slurm settings and discard every field except the two explicit plugin names."""
    command = ["scontrol", "show", "config"]
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True, timeout=20)
        return {"command": command, "status": "readable", "returncode": result.returncode,
                **selected_config(result.stdout, SLURM_CONFIG_FIELDS)}
    except (OSError, subprocess.SubprocessError) as error:
        return {"command": command, "status": "unavailable", "error": "public selected Slurm configuration query failed: " + type(error).__name__,
                **selected_config("", SLURM_CONFIG_FIELDS)}


def mount_path(value):
    """Decode only the documented escaped characters in an explicit mountinfo path."""
    return re.sub(r"\\(040|011|012|134)", lambda match: chr(int(match[1], 8)), value)


def own_cgroup_mounts(text):
    """Retain only cgroup mountinfo entries rather than unrelated filesystem mounts."""
    entries = []
    for line in text.splitlines():
        left, separator, right = line.partition(" - ")
        tail = right.split()
        if not separator or len(tail) < 3 or tail[0] not in {"cgroup", "cgroup2"}:
            continue
        fields = left.split()
        if len(fields) < 6:
            raise ValueError("invalid selected cgroup mountinfo entry")
        entries.append({"filesystem": tail[0], "root": mount_path(fields[3]), "mountpoint": mount_path(fields[4]),
                        "controllers": tail[2].split(",") if tail[0] == "cgroup" else [], "original_line": line})
    filtered = "".join(entry["original_line"] + "\n" for entry in entries)
    return {"entries": entries, "filtered_bytes": filtered, "sha256": hashlib.sha256(filtered.encode()).hexdigest(),
            "hash_scope": "only the original cgroup and cgroup2 mountinfo lines, each followed by one newline"}


def scoped_cgroup_levels(cgroup_path, environment):
    """Select only the current process's exact job, step and task ancestors, with no sibling inspection."""
    path = PurePosixPath(cgroup_path)
    if not path.is_absolute() or ".." in path.parts:
        raise ValueError("own cgroup path must be absolute and must not contain parent traversal")
    parts = path.parts
    expected = ("job_" + environment["SLURM_JOB_ID"], "step_" + environment.get("SLURM_STEP_ID", ""))
    positions = []
    for component in expected:
        matches = [index for index, value in enumerate(parts) if value == component]
        if len(matches) != 1:
            raise ValueError("own cgroup path does not identify the exact current job and step")
        positions.append(matches[0])
    if positions[1] <= positions[0]:
        raise ValueError("own step cgroup must descend from its exact job")
    task = "task_" + environment.get("SLURM_LOCALID", "")
    task_positions = [index for index, value in enumerate(parts) if value == task]
    levels = [("job", positions[0]), ("step", positions[1])]
    if len(task_positions) == 1 and task_positions[0] > positions[1]:
        levels.append(("task", task_positions[0]))
    return [{"level": name, "cgroup_path": str(PurePosixPath(*parts[:position + 1]))} for name, position in levels]


def own_placement(environment, cgroup, proc_self=Path("/proc/self")):
    """Inspect only this process's cgroup paths and mark device-rule evidence that was not inspected."""
    process_stat = (proc_self / "stat").read_text()
    fields = process_stat.rsplit(")", 1)[1].split()
    if int(process_stat.split(" ", 1)[0]) != os.getpid() or len(fields) < 20 or not fields[19].isdigit():
        raise ValueError("own process stat does not identify the current PID and birth counter")
    mounts = own_cgroup_mounts((proc_self / "mountinfo").read_text())
    records = []
    for line in cgroup.splitlines():
        hierarchy, controllers, cgroup_path = line.split(":", 2)
        controller_names = controllers.split(",") if controllers else []
        record = {"hierarchy": hierarchy, "controllers": controller_names, "cgroup_path": cgroup_path,
                  "mounts": [], "status": "not independently inspected"}
        try:
            levels = scoped_cgroup_levels(cgroup_path, environment)
        except ValueError as error:
            record["error"] = str(error)
            records.append(record)
            continue
        for mount in mounts["entries"]:
            if (mount["filesystem"] == "cgroup" and not set(controller_names).intersection(mount["controllers"])) or (
                    mount["filesystem"] == "cgroup2" and (hierarchy != "0" or controllers)):
                continue
            root = PurePosixPath(mount["root"])
            mountpoint = Path(mount["mountpoint"])
            evidence = {"mountpoint": str(mountpoint), "filesystem": mount["filesystem"], "levels": [],
                "bpf_device_rules_independently_inspected": False,
                "device_rule_scope": "v1 devices.list only" if mount["filesystem"] == "cgroup" else "v2 BPF device rules were not independently inspected"}
            for level in levels:
                try:
                    relative = PurePosixPath(level["cgroup_path"]).relative_to(root)
                except ValueError:
                    evidence["levels"].append({**level, "status": "own ancestor is outside the recorded mount root"})
                    continue
                target = mountpoint.joinpath(*relative.parts)
                if not target.resolve().is_relative_to(mountpoint.resolve()):
                    raise ValueError("own cgroup path escaped its recorded mountpoint")
                entry = {**level, "directory": path_metadata(target)}
                if mount["filesystem"] == "cgroup" and "devices" in controller_names:
                    entry["devices_list"] = {"path": str(target / "devices.list"), **read_hardware_file(target / "devices.list")}
                cpuset_name = "cpuset.cpus.effective" if mount["filesystem"] == "cgroup2" else "cpuset.effective_cpus"
                if mount["filesystem"] == "cgroup2" or "cpuset" in controller_names:
                    entry["effective_cpuset"] = {"path": str(target / cpuset_name), **read_hardware_file(target / cpuset_name)}
                evidence["levels"].append(entry)
            record["mounts"].append(evidence)
        record["status"] = "own paths observed" if record["mounts"] else "matching cgroup mount not available"
        record["task_path_observed"] = any(level["level"] == "task" for level in levels)
        records.append(record)
    return {"pid": os.getpid(), "uid": os.getuid(), "process_start_time": fields[19],
            "cpu_affinity": sorted(os.sched_getaffinity(0)),
            "process_stat_file": path_metadata(proc_self / "stat"),
            "own_cgroup_file": {"file": path_metadata(proc_self / "cgroup"), "text": cgroup,
                "sha256": hashlib.sha256(cgroup.encode()).hexdigest(),
                "hash_scope": "only the UTF-8 encoded current-process entries read from /proc/self/cgroup"},
            "cgroup_namespace": path_metadata(proc_self / "ns/cgroup"),
            "mountinfo_file": path_metadata(proc_self / "mountinfo"), "cgroup_mountinfo": mounts, "cgroups": records,
            "bpf_device_rules_independently_inspected": False}


def device_isolation_observations(environment, cgroup, cgroup_config=Path("/etc/slurm/cgroup.conf"), proc_self=Path("/proc/self")):
    """Collect explicit configuration and own-placement observations without claiming device isolation is verified."""
    return {"node_cgroup_configuration": selected_config_file(cgroup_config, CGROUP_CONFIG_FIELDS),
            "running_slurm_configuration": running_slurm_config(), "own_placement": own_placement(environment, cgroup, proc_self),
            "device_isolation_verified": False, **BOUNDARY}


def host_inventory(proc_root=Path("/proc/driver/nvidia/gpus"), pci_root=Path("/sys/bus/pci/devices"),
                   device_root=Path("/dev"), gres_path=Path("/etc/slurm/gres.conf")):
    """Read only full-host NVIDIA GPU hardware identifiers, device numbers and GPU GRES declarations."""
    # For example, an information file becomes {Model, GPU UUID, Bus Location, Device Minor}; other fields are omitted.
    proc_cards, pci_cards, devices = [], [], []
    for path in sorted(proc_root.glob("*/information")):
        if not PCI_ADDRESS.fullmatch(path.parent.name):
            continue
        result = read_hardware_file(path)
        fields = {}
        for line in result.get("text", "").splitlines():
            key, separator, value = line.partition(":")
            if separator and key.strip() in {"Model", "GPU UUID", "Bus Location", "Device Minor"}:
                fields[key.strip()] = value.strip()
        proc_cards.append({"path": str(path), "pci_directory": path.parent.name, "status": result["status"], "fields": fields})
    for path in sorted(pci_root.glob("*")):
        if not PCI_ADDRESS.fullmatch(path.name):
            continue
        vendor, device_class = read_hardware_file(path / "vendor"), read_hardware_file(path / "class")
        if vendor.get("text", "").strip().lower() != "0x10de" or not re.fullmatch(r"0x03[0-9a-fA-F]{4}", device_class.get("text", "").strip()):
            continue
        fields = {name: read_hardware_file(path / name) for name in ("vendor", "device", "class", "subsystem_vendor", "subsystem_device")}
        pci_cards.append({"path": str(path), "pci_address": path.name,
                          "fields": {name: {**value, **({"text": value["text"].strip()} if "text" in value else {})} for name, value in fields.items()}})
    for path in sorted(device_root.glob("nvidia[0-9]*")):
        if not re.fullmatch(r"nvidia[0-9]+", path.name):
            continue
        information = path.stat()
        character = stat.S_ISCHR(information.st_mode)
        devices.append({"path": str(path), "character_device": character,
                        "major": os.major(information.st_rdev) if character else None,
                        "minor": os.minor(information.st_rdev) if character else None})
    mapping = read_hardware_file(gres_path)
    lines = [line.strip() for line in mapping.get("text", "").splitlines() if line.strip() and not line.lstrip().startswith("#")
             and (re.search(r"\bName=gpu(?:\s|$)", line) or ("AutoDetect=nvml" in line and not re.search(r"\bName=(?!gpu(?:\s|$))", line)))]
    return {"proc_gpu_inventory": {"path": str(proc_root), "status": "present" if proc_root.is_dir() else "absent", "cards": proc_cards},
        "pci_gpu_inventory": {"path": str(pci_root), "status": "present" if pci_root.is_dir() else "absent", "cards": pci_cards},
        "device_files": devices, "gpu_gres_mapping": {"path": str(gres_path), "status": mapping["status"], "lines": lines}}


def collect(directory, unit, plan):
    """Read hardware only after exact owned-allocation checks, retaining the two identifier systems separately."""
    environment = {name: os.environ[name] for name in ALLOCATION_FIELDS if name in os.environ}
    owned = owned_allocation(plan, directory, environment, Path("/proc/self/cgroup").read_text(), socket.gethostname().split(".")[0])
    write_once(unit / "owned_allocation.json", owned)
    atomic_json(unit / "progress.json", {"state": "running", "kind": "allocation observation", **BOUNDARY})
    hardware = {"hostname": owned["hostname"], "slurm_job_id": owned["job_id"],
                "cuda_visible_devices": owned["visible_selector"], "gpu_query": "", **BOUNDARY}
    atomic_json(unit / "hardware.json", hardware)
    observations = {"schema_version": 1, "kind": "h100_nvl_allocation_hardware_observations", "observed_at_utc": datetime.now(timezone.utc).isoformat(),
        "plan": file_record(directory / "allocation_observation_plan.json"), "owned_allocation": file_record(unit / "owned_allocation.json"),
        "hostname": owned["hostname"], "job_id": owned["job_id"],
        "allocation_environment": {name: environment.get(name) for name in ALLOCATION_FIELDS},
        "absent_allocation_fields": [name for name in ALLOCATION_FIELDS if name not in environment], **BOUNDARY}
    command = ["nvidia-smi", "-i", owned["visible_selector"],
               "--query-gpu=index,uuid,name,pci.bus_id,memory.total,driver_version", "--format=csv,noheader,nounits"]
    result = subprocess.run(command, capture_output=True, text=True, check=True, timeout=20)
    queries = {"gpu_csv": {"command": command, "returncode": result.returncode, "stdout": result.stdout}}
    cards = list(csv.reader(result.stdout.splitlines(), skipinitialspace=True))
    if len(cards) != 1 or len(cards[0]) != 6:
        raise ValueError("the explicit visible selector did not return one GPU hardware row")
    index, gpu, model, pci, memory, driver = [value.strip() for value in cards[0]]
    hardware.update(gpu_query=", ".join((index, model, memory + " MiB", driver, gpu)),
                    observed_gpu_uuid=gpu, observed_pci_bus_id=pci)
    atomic_json(unit / "hardware.json", hardware)
    observations.update(visible_gpu_queries=queries, host_gpu_inventory=host_inventory(),
                        device_isolation_observations=device_isolation_observations(environment, owned["cgroup"]))
    write_once(unit / "hardware_observations.json", observations)
    result = {"schema_version": 1, "kind": "h100_nvl_allocation_observation_result", "passed": True,
              "observations": file_record(unit / "hardware_observations.json"), "observations_completed": True, **BOUNDARY}
    write_once(unit / "result.json", result)
    atomic_json(unit / "progress.json", {"state": "completed", "passed": True, "kind": "allocation observation", **BOUNDARY})
    return result


def capture(directory):
    """Validate the prepared metadata boundary and publish monitored terminal outcome records."""
    directory = Path(directory).resolve()
    plan = read(directory / "allocation_observation_plan.json")
    manifest = read(directory / "manifest.yaml")
    run = Path(plan["campaign_run_dir"])
    unit = run / "data" / directory.name
    if (directory.parent != run / "remote_setup" or plan.get("data_unit") != {"path": str(unit), "name": directory.name}
            or unit.is_symlink() or not unit.is_dir()):
        raise ValueError("prepared observation data shard differs from its exact campaign path")
    try:
        if (plan.get("kind") != "h100_nvl_allocation_observation" or plan.get("schema_version") != 1
                or any(plan.get(name) != value or type(plan.get(name)) is not type(value) for name, value in BOUNDARY.items())
                or plan.get("allocation_fields") != list(ALLOCATION_FIELDS)):
            raise ValueError("prepared observation must preserve its metadata-only boundary")
        if helper_source(plan["helpers"]["source_root"]) != plan["helpers"] or Path(__file__).resolve() != Path(plan["helpers"]["source_root"]) / RELATIVE_PATH:
            raise ValueError("capture must execute the exact committed prepared helper")
        required = {"allocation_observation_plan.json", "config_resolved.yaml", "job.sbatch", "command.txt", "experiment_background.md"}
        if set(manifest["artifacts"]) != required:
            raise ValueError("prepared observation manifest must retain all five immutable artifacts")
        for name, proof in manifest["artifacts"].items():
            if proof != file_record(directory / name):
                raise ValueError("prepared observation artifact changed: " + name)
        for filename in ("experiment_background.md", "manifest.yaml", "config_resolved.yaml", "command.txt"):
            if (unit / filename).read_bytes() != (directory / filename).read_bytes():
                raise ValueError("data shard preparation bytes differ: " + filename)
        return collect(directory, unit, plan)
    except BaseException as error:
        write_once(unit / "failure.json", {"kind": "h100_nvl_allocation_observation_failure", "passed": False,
            "error": str(error), "exception_type": type(error).__name__, "failed_at_utc": datetime.now(timezone.utc).isoformat(), **BOUNDARY})
        atomic_json(unit / "progress.json", {"state": "failed", "kind": "allocation observation", **BOUNDARY})
        raise


def main():
    """Prepare a held metadata command or collect inside its already recorded owned allocation."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="mode", required=True)
    staging = commands.add_parser("prepare")
    staging.add_argument("--directory", type=Path, required=True)
    staging.add_argument("--campaign", type=Path, required=True)
    staging.add_argument("--helper-root", type=Path, required=True)
    staging.add_argument("--resource-submission", type=Path, required=True)
    staging.add_argument("--python", type=Path, default=Path("/usr/bin/python3"))
    collection = commands.add_parser("capture")
    collection.add_argument("--directory", type=Path, required=True)
    args = parser.parse_args()
    if args.mode == "prepare":
        result = prepare(args.directory, args.campaign, args.helper_root, args.resource_submission, args.python)
    else:
        result = capture(args.directory)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
