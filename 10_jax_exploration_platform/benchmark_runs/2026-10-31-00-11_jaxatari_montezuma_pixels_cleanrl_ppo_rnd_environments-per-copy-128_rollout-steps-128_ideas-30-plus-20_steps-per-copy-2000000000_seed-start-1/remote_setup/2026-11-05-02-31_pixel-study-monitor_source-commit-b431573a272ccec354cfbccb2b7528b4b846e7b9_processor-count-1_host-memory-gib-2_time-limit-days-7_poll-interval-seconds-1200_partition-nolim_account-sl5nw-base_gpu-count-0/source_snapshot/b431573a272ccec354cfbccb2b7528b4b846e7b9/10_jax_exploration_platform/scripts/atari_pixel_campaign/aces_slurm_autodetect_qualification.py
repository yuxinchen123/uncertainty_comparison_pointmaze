"""Verify a separate ACES NVML AutoDetect profile without inventing a physical GRES mapping.

The owned scheduler indices agree logically. A sole visible NVML UUID and PCI
address identify the physical GPU independently through proc metadata. The
subsequent same-process CUDA check remains mandatory. Configured cgroup
isolation is trusted explicitly; neither kernel enforcement nor the Slurm
index-to-physical-device relation is independently proved.
"""

import csv
import hashlib
import importlib.util
import os
from pathlib import Path
import re
import stat


def adjacent(name):
    """Load only the exact adjacent metadata helper without numerical imports."""
    path = Path(__file__).resolve().with_name(name + ".py")
    specification = importlib.util.spec_from_file_location("aces_autodetect_" + name, path)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


legacy = adjacent("aces_slurm_qualification")
require, field, pci_address = legacy.require, legacy.field, legacy.pci_address
BOUNDARY, UPSTREAM_DEVICE_SOURCES = legacy.BOUNDARY, legacy.UPSTREAM_DEVICE_SOURCES
PROFILE = {**legacy.PROFILE,
    "name": "aces-h100-pcie-driver-580.178.04-autodetect-nvml-configured-isolation",
    "gres_configuration": "AutoDetect=nvml", "controller_configuration_path": "/etc/slurm/slurm.conf",
    "physical_gres_mapping_verified": False,
    "isolation_basis": "trusted root-owned NVML AutoDetect and task/cgroup configuration; logical GRES mapping and kernel device enforcement not independently verified",
    "configuration_sources": [*legacy.PROFILE["configuration_sources"],
        "https://raw.githubusercontent.com/SchedMD/slurm/slurm-25-11-7-1/src/common/read_config.c"]}
DIRECTORIES = ("/", "/var", "/var/spool", "/var/spool/slurmd", "/var/spool/slurmd/conf-cache")


def require_profile(value):
    """Require the complete distinct profile with exact scalar types and values."""
    require(isinstance(value, dict) and set(value) == set(PROFILE)
        and all(type(value[k]) is type(v) and value[k] == v for k, v in PROFILE.items()),
        "the complete separate ACES NVML AutoDetect configured-isolation profile is required")
    return value


def failed_metadata(path, error):
    """Retain an explicit metadata failure for later rejection after the own allocation record is saved."""
    status = "absent" if isinstance(error, FileNotFoundError) else "unreadable" if isinstance(error, PermissionError) else "error"
    return {"path": str(path), "status": status, "exception_type": type(error).__name__,
        "errno": error.errno, "error": str(error)}


def directory_chain():
    """Capture each ordinary cache ancestor and preserve absent or unreadable metadata explicitly."""
    records = []
    for value in DIRECTORIES:
        try:
            information = Path(value).lstat()
        except OSError as error:
            records.append(failed_metadata(value, error))
            continue
        records.append({"path": value, "status": "readable", "owner_uid": information.st_uid, "owner_gid": information.st_gid,
            "mode": oct(stat.S_IMODE(information.st_mode)), "directory": stat.S_ISDIR(information.st_mode),
            "symlink": stat.S_ISLNK(information.st_mode), "inode": information.st_ino,
            "device": information.st_dev, "mtime_ns": information.st_mtime_ns})
    return records


def configuration_observations(environment, running):
    """Capture only the six reviewed paths and retain read failures before any trust validation."""
    before, candidates = directory_chain(), []
    paths = {"environment": environment.get("SLURM_CONF"),
        "running_controller": running.get("fields", {}).get("SLURM_CONF")}
    # Before: a file read may fail. After: all candidate paths and that exact failure remain visible.
    for root in ("/etc/slurm", PROFILE["configuration_directory"]):
        for name in ("slurm", "gres", "cgroup"):
            path = Path(root) / (name + ".conf")
            try:
                record = legacy.cached_file(path, name)
            except OSError as error:
                record = {**failed_metadata(path, error), "kind": name}
            candidates.append(record)
    return {"observed_paths": paths, "candidates": candidates,
        "cache_directory_chain": {"before": before, "after": directory_chain()}}


def task_plugins(value):
    """Normalize only the two exact nonduplicate plugins that Slurm 25.11.7 sorts itself."""
    require(isinstance(value, str), "both exact task plugins must be recorded")
    names = value.split(",")
    require(len(names) == 2 and set(names) == {"task/cgroup", "task/affinity"},
        "only the two exact distinct task/cgroup and task/affinity plugins are supported")
    return tuple(sorted(names, reverse=True))


def autodetect_configuration(text, hostname):
    """Require one effective host-specific NVML AutoDetect line without physical File claims."""
    matching = []
    global_setting = None
    for line in text.splitlines():
        tokens = line.split("#", 1)[0].split()
        if not tokens:
            continue
        pairs = [token.split("=", 1) for token in tokens]
        require(all(len(pair) == 2 and pair[0] and pair[1] for pair in pairs)
            and len({pair[0].lower() for pair in pairs}) == len(pairs), "invalid or duplicate GRES keys")
        values = {key.lower(): value for key, value in pairs}
        if "nodename" in values and hostname not in legacy.expand_list(values["nodename"]):
            continue
        if "nodename" not in values:
            require(set(values) == {"autodetect"} and global_setting is None,
                "unreviewed or duplicate global AutoDetect configuration")
            global_setting = values["autodetect"]
        else:
            require(set(values) == {"nodename", "autodetect"},
                "host-specific File, shared GPU, MIG or other GRES overrides need a separate profile")
            matching.append(values)
    require(len(matching) == 1 and matching[0]["autodetect"] == "nvml"
        and global_setting in (None, "nvml"), "one unambiguous host-specific NVML AutoDetect setting is required")
    return {"host_expression": matching[0]["nodename"], "autodetect": "nvml",
        "physical_gres_mapping_verified": False}


def validate_configuration(configuration, isolation, hostname):
    """Bind trusted cached configuration and normalized running plugins without physical-index inference."""
    expected = PROFILE["configuration_directory"]
    chain = configuration["cache_directory_chain"]
    require(set(chain) == {"before", "after"} and chain["before"] == chain["after"]
        and [row["path"] for row in chain["before"]] == list(DIRECTORIES),
        "the complete cache directory chain must remain unchanged during capture")
    require(all(row.get("status") == "readable" and type(row.get("owner_uid")) is int and row["owner_uid"] == 0 and row.get("directory") is True and row.get("symlink") is False
        and int(row["mode"], 8) & 0o022 == 0 for row in chain["before"]),
        "every cache directory ancestor must be root-owned, ordinary and not group or other writable")
    require(configuration["observed_paths"]["environment"] == expected + "/slurm.conf"
        and configuration["observed_paths"].get("running_controller") == PROFILE["controller_configuration_path"],
        "the actual owned step must identify the explicit root-owned node cache")
    candidates = {row["path"]: row for row in configuration["candidates"]}
    expected_paths = {root + "/" + name + ".conf" for root in ("/etc/slurm", expected) for name in ("slurm", "gres", "cgroup")}
    require(len(candidates) == len(configuration["candidates"]) and set(candidates) == expected_paths,
        "all six distinct configuration candidates must be retained")
    selected = {name: candidates[expected + "/" + name + ".conf"] for name in ("slurm", "gres", "cgroup")}
    require(all(type(row.get("owner_uid")) is int for row in selected.values()),
        "cached configuration file owners must retain integer UIDs")
    values = {name: legacy.trusted_file(row) for name, row in selected.items()}
    require(all(row.get("status") == "absent" for path, row in candidates.items() if not path.startswith(expected + "/")),
        "competing node-local configuration candidates are ambiguous")
    slurm, cgroup = legacy.configuration_values(values["slurm"]), legacy.configuration_values(values["cgroup"])
    running = isolation["running_slurm_configuration"]
    require(running.get("command") == ["scontrol", "show", "config"] and type(running.get("returncode")) is int
        and running["returncode"] == 0, "the successful exact running configuration query is required")
    selected_keys = ["TaskPlugin", "ProctrackType", "SLURM_CONF"]
    rebuilt = adjacent("h100_nvl_allocation_observation").selected_config(running["filtered_bytes"], selected_keys)
    require(running.get("selected_keys") == selected_keys
        and all(running.get(key) == rebuilt[key] for key in ("fields", "missing_fields", "filtered_bytes", "sha256", "hash_scope"))
        and configuration["observed_paths"].get("running_controller") == running["fields"].get("SLURM_CONF")
        and legacy.configuration_values(running["filtered_bytes"]) == running["fields"],
        "running selected fields must agree with their retained raw bytes and SHA")
    require(task_plugins(slurm.get("TaskPlugin")) == task_plugins(running["fields"].get("TaskPlugin"))
        and slurm.get("ProctrackType") == running["fields"].get("ProctrackType") == "proctrack/cgroup",
        "cached and running plugin identities must agree after Slurm's documented normalization")
    require(slurm.get("GresTypes") == "gpu"
        and cgroup.get("ConstrainDevices", "").lower() == cgroup.get("ConstrainCores", "").lower() == "yes"
        and cgroup.get("CgroupPlugin", "autodetect") in ("autodetect", "cgroup/v1")
        and all(cgroup.get(name, "no").lower() == "no" for name in ("IgnoreSystemd", "IgnoreSystemdOnFailure")),
        "configured version-one CPU and device restrictions remain mandatory")
    return {"configuration_files": selected, "cache_directory_chain": chain,
        "autodetect_configuration": autodetect_configuration(values["gres"], hostname),
        "isolation_basis": PROFILE["isolation_basis"], "physical_gres_mapping_verified": False,
        "kernel_device_enforcement_verified": False}


def validate_allocation(observation):
    """Check logical allocation separately from visible physical UUID/PCI and configured isolation trust."""
    require_profile(observation["qualification_profile"])
    owned, inventory = observation["owned_allocation"], observation["host_gpu_inventory"]
    environment, host = owned["allocation_environment"], owned["hostname"]
    require(type(owned.get("uid")) is int and owned["uid"] == PROFILE["uid"]
        and type(owned.get("pid")) is int and owned["pid"] > 0
        and isinstance(owned.get("process_start_time"), str)
        and re.fullmatch(r"[1-9][0-9]*", owned["process_start_time"]) and re.fullmatch(r"ac[0-9]+", host), "wrong ACES owner or compute host")
    job, step = owned["job_id"], owned["step_id"]
    require(re.fullmatch(r"[1-9][0-9]*", job) and re.fullmatch(r"[0-9]+", step), "exact numeric owned job and step are required")
    scheduler = owned["scheduler"]
    for kind, identity in (("job", job), ("step", job + "." + step), ("node", host)):
        query = scheduler[kind]
        require(query["command"] == ["scontrol", "show", kind, "-d", identity]
            and type(query.get("returncode")) is int and query["returncode"] == 0,
            "all scheduler observations must retain the exact owned query")
    job_text, step_text, node_text = (scheduler[kind]["stdout"] for kind in ("job", "step", "node"))
    require(field(job_text, "JobId") == job and field(job_text, "JobState") == "RUNNING"
        and field(job_text, "UserId") == PROFILE["username"] + "(" + str(PROFILE["uid"]) + ")"
        and field(job_text, "Account") == "158590754315" and field(job_text, "Partition") == "gpu"
        and field(job_text, "QOS") == "normal" and field(job_text, "NodeList") == host
        and field(step_text, "StepId") == job + "." + step and field(step_text, "State") == "RUNNING"
        and field(step_text, "UserId") in (str(PROFILE["uid"]), PROFILE["username"] + "(" + str(PROFILE["uid"]) + ")")
        and field(step_text, "NodeList") == host and field(node_text, "NodeName") == host
        and field(node_text, "Version") == PROFILE["scheduler_version"], "actual owner, scheduler or allocation differs from the ACES profile")
    for text, key in ((job_text, "AllocTRES"), (step_text, "TRES")):
        pairs = [pair.split("=", 1) for pair in field(text, key).split(",")]
        require(all(len(pair) == 2 for pair in pairs) and len({pair[0] for pair in pairs}) == len(pairs),
            "duplicate or invalid resource names in actual TRES are ambiguous")
        resources = dict(pairs)
        require(all(resources.get(k) == v for k, v in {"cpu": "8", "node": "1", "gres/gpu": "1", "gres/gpu:h100": "1"}.items())
            and resources.get("mem") in ("32G", "32768M"), "the site qualification requires exactly one typed H100 and eight CPUs/32G")
    index = re.findall(r"GRES=gpu:h100:1\(IDX:([0-9]+)\)", job_text)
    require(len(index) == 1 and environment.get("SLURM_JOB_GPUS") == environment.get("SLURM_STEP_GPUS") == index[0]
        and environment.get("SLURM_JOB_ID") == job and environment.get("SLURM_STEP_ID") == step,
        "the owned job's explicit GRES index and original step fields must agree")
    configuration = validate_configuration(observation["cached_configuration"], observation["device_isolation_observations"], host)
    node_gres = re.fullmatch(r"gpu:h100:([0-9]+)(?:\(S:[0-9,-]+\))?", field(node_text, "Gres"))
    require(node_gres is not None and 0 <= int(index[0]) < int(node_gres[1]),
        "the logical GRES index must be inside this node's reported typed GPU count")
    devices = {row["path"]: row for row in inventory["device_files"]}
    require(len(devices) == len(inventory["device_files"]) and devices
        and all(re.fullmatch(r"/dev/nvidia(?:0|[1-9][0-9]*)", name) and row.get("character_device") is True
            and type(row.get("major")) is int and row["major"] == 195
            and type(row.get("minor")) is int and row["minor"] >= 0 for name, row in devices.items()),
        "the observed whole-GPU character-device roster must be complete and unambiguous")
    require(inventory["proc_gpu_inventory"]["status"] == inventory["pci_gpu_inventory"]["status"] == "present",
        "complete proc and PCI metadata are required")
    proc = inventory["proc_gpu_inventory"]["cards"]
    require(len(proc) == len(devices) == int(node_gres[1])
        and len({row["fields"].get("GPU UUID") for row in proc}) == len(proc)
        and len({row["fields"].get("Device Minor") for row in proc}) == len(proc)
        and len({row["minor"] for row in devices.values()}) == len(devices)
        and len({pci_address(row["fields"].get("Bus Location", "")) for row in proc}) == len(proc)
        and len({pci_address(row["pci_address"]) for row in inventory["pci_gpu_inventory"]["cards"]})
            == len(inventory["pci_gpu_inventory"]["cards"]),
        "the node count, proc identities and device minors must each describe one complete roster")
    require({str(row["minor"]) for row in devices.values()} == {row["fields"].get("Device Minor") for row in proc}
        and {pci_address(row["fields"].get("Bus Location", "")) for row in proc}
        == {pci_address(row["pci_address"]) for row in inventory["pci_gpu_inventory"]["cards"]}
        and len(inventory["pci_gpu_inventory"]["cards"]) == len(proc),
        "proc minors, physical PCI addresses and character devices must agree without ordinal inference")
    hardware = observation["hardware_queries"]["gpu_csv"]
    fields = ["index", "uuid", "name", "pci.bus_id", "memory.total", "driver_version"]
    require(hardware.get("command") == ["nvidia-smi", "--query-gpu=" + ",".join(fields), "--format=csv,noheader,nounits"]
        and hardware.get("fields") == fields and type(hardware.get("returncode")) is int,
        "the hardware observation must retain the exact allowlisted CSV command and field roster")
    rows = list(csv.reader(hardware["stdout"].splitlines()))
    require(hardware["returncode"] == 0 and len(rows) == 1 and len(rows[0]) == 6,
        "the recorded hardware-only NVML query must expose exactly one whole GPU")
    visible_index, gpu, model, pci, memory, driver = [value.strip() for value in rows[0]]
    # Before: one visible UUID/PCI plus an unordered host inventory. After: one unique physical minor.
    cards = [row for row in proc if row["status"] == "readable" and row["fields"].get("GPU UUID") == gpu
        and pci_address(row["fields"].get("Bus Location", "")) == pci_address(pci)]
    require(len(cards) == 1, "visible UUID and PCI must identify one proc GPU independently of every ordinal")
    physical = cards[0]["fields"]
    files = [name for name, row in devices.items() if str(row["minor"]) == physical.get("Device Minor")]
    require(len(files) == 1, "the visible proc GPU must identify exactly one observed character device")
    selected_file, = files
    require(physical.get("Model") == model == PROFILE["model"]
        and memory == str(PROFILE["memory_mib"]) and driver == PROFILE["driver"],
        "the visible physical model, memory and driver differ from the separate ACES profile")
    placement = observation["device_isolation_observations"]["own_placement"]
    require(placement["pid"] == owned["pid"] and placement["uid"] == owned["uid"]
        and placement["process_start_time"] == owned["process_start_time"]
        and placement["cpu_affinity"] == owned["cpu_affinity"] and len(set(owned["cpu_affinity"])) == 8,
        "own task metadata must identify the exact eight-CPU process")
    require(environment.get("SLURM_GPUS_ON_NODE") == "1" and environment.get("SLURM_CPUS_PER_TASK") == "8"
        and environment.get("SLURM_LOCALID") == "0" and all(type(cpu) is int and cpu >= 0 for cpu in owned["cpu_affinity"]),
        "the actual environment and affinity must retain one GPU and eight CPUs for task zero")
    expected = "/slurm/uid_" + str(owned["uid"]) + "/job_" + job + "/step_" + step + "/task_0"
    membership = placement["own_cgroup_file"]
    require(membership["sha256"] == hashlib.sha256(membership["text"].encode()).hexdigest()
        and membership["text"] == owned["cgroup"], "own cgroup bytes changed")
    groups = [row for row in placement["cgroups"] if row["controllers"] == ["devices"] and row["cgroup_path"] == expected]
    require(len(groups) == 1 and groups[0]["task_path_observed"] is True,
        "the exact owned version-one job/step/task device hierarchy is required")
    mounts = groups[0]["mounts"]
    require(len(mounts) == 1 and mounts[0]["filesystem"] == "cgroup"
        and mounts[0]["mountpoint"] == "/sys/fs/cgroup/devices"
        and [row["level"] for row in mounts[0]["levels"]] == ["job", "step", "task"],
        "the owned version-one device controller must retain all exact mounted levels")
    expected_paths = [expected.rsplit("/step_", 1)[0], expected.rsplit("/task_", 1)[0], expected]
    for level, expected_path in zip(mounts[0]["levels"], expected_paths):
        require(level["cgroup_path"] == expected_path and level["directory"].get("status") == "readable"
            and level["directory"].get("owner_uid") == 0
            and level["directory"].get("path") == "/sys/fs/cgroup/devices" + expected_path
            and level.get("devices_list", {}).get("status") == "readable",
            "all owned device cgroup levels and literal rule observations must be present")
    cpu_groups = [row for row in placement["cgroups"] if row["controllers"] == ["cpuset"] and row["cgroup_path"] == expected]
    require(len(cpu_groups) == 1 and cpu_groups[0]["task_path_observed"] is True,
        "the owned version-one task CPU hierarchy is required")
    cpu_mounts = cpu_groups[0]["mounts"]
    require(len(cpu_mounts) == 1 and cpu_mounts[0]["filesystem"] == "cgroup"
        and cpu_mounts[0]["mountpoint"] == "/sys/fs/cgroup/cpuset"
        and [row["level"] for row in cpu_mounts[0]["levels"]] == ["job", "step", "task"],
        "all owned CPU hierarchy levels are required")
    for level, expected_path in zip(cpu_mounts[0]["levels"], expected_paths):
        require(level["cgroup_path"] == expected_path and level["directory"].get("status") == "readable"
            and level["directory"].get("owner_uid") == 0
            and level["directory"].get("path") == "/sys/fs/cgroup/cpuset" + expected_path,
            "every owned CPU hierarchy level must retain its exact root-owned directory")
    parse_cpus = adjacent("h100_nvl_slurm_qualification").cpu_set
    require(all(level.get("effective_cpuset", {}).get("status") == "readable"
        and set(owned["cpu_affinity"]) <= parse_cpus(level["effective_cpuset"]["text"])
        for level in cpu_mounts[0]["levels"]), "eight actual CPUs must remain within every owned effective cpuset")
    # Retain each devices.list literally; default-allow output cannot attest deny exceptions.
    return {"job_id": job, "step_id": step, "hostname": host, "gpu_uuid": gpu,
        "pci_address": pci_address(pci), "physical_device_file": selected_file,
        "physical_minor": devices[selected_file]["minor"], "device_major": devices[selected_file]["major"],
        "slurm_gres_index": index[0], "visible_nvml_index": visible_index,
        "cuda_selector_original": environment.get("CUDA_VISIBLE_DEVICES"),
        "model": model, "memory_mib": int(memory), "driver": driver,
        "cpu_affinity": owned["cpu_affinity"], "configuration": configuration,
        "qualification_profile": PROFILE, "physical_gres_mapping_verified": False,
        "logical_allocation_agreement_verified": True, "kernel_device_enforcement_verified": False, **BOUNDARY}


def cuda_device_identity(device):
    """Reuse the unchanged same-process CUDA query for this identical model and driver."""
    return legacy.cuda_device_identity(device)


def validate_cuda_identity(record, allocation, pid):
    """Require this distinct profile and unchanged same-process UUID/PCI/driver proof."""
    require_profile(allocation["qualification_profile"])
    require(allocation.get("physical_gres_mapping_verified") is False
        and allocation.get("kernel_device_enforcement_verified") is False,
        "same-process physical identity cannot imply an independently verified GRES mapping or kernel isolation")
    return legacy.validate_cuda_identity(record, allocation, pid)
