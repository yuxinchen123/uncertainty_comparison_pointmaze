"""Check an explicit ACES allocation without equating its GPU numbering systems.

Cached root-owned configuration is scheduler configuration evidence. It does not
attest actual kernel device enforcement. A wildcard-looking version-one
devices.list proves neither presence nor absence of deny exceptions.
"""

import csv
import ctypes
import hashlib
import importlib.util
import os
from pathlib import Path
import re
import stat
import uuid


PROFILE = {"schema_version": 1, "name": "aces-h100-pcie-driver-580.178.04",
    "cluster": "aces", "username": "u.sl359075", "uid": 56626,
    "model": "NVIDIA H100 PCIe", "memory_mib": 81559, "driver": "580.178.04",
    "scheduler_version": "25.11.7", "configuration_directory": "/var/spool/slurmd/conf-cache",
    "configuration_sources": ["https://slurm.schedmd.com/gres.conf.html", "https://slurm.schedmd.com/gres.html"],
    "isolation_basis": "recorded root-owned scheduler task/cgroup configuration; kernel enforcement not independently verified",
    "kernel_device_enforcement_verified": False, "workers_permitted": False,
    "campaign_claims_permitted": False, "scientific_hypothesis_credit": 0}
BOUNDARY = {"workers_permitted": False, "campaign_claims_permitted": False,
    "scientific_hypothesis_credit": 0, "comparison_credit": 0, "idea_credit": 0, "admission_credit": 0}
SLURM_KEYS = ("TaskPlugin", "ProctrackType", "GresTypes", "SlurmdSpoolDir", "SlurmctldParameters")
CGROUP_KEYS = ("CgroupPlugin", "ConstrainDevices", "ConstrainCores", "CgroupMountpoint",
    "IgnoreSystemd", "IgnoreSystemdOnFailure")
UPSTREAM_DEVICE_SOURCES = (
    "https://linux.googlesource.com/linux/kernel/git/torvalds/linux/+/c1481c94e74c955e0448ddf46b8615a44d840c1e/security/device_cgroup.c",
    "https://docs.kernel.org/admin-guide/cgroup-v1/devices.html")


def require(condition, message):
    """Reject missing or ambiguous evidence before selecting a physical device."""
    if not condition:
        raise ValueError(message)


def adjacent(name):
    """Load one adjacent standard-library helper without importing numerical packages."""
    path = Path(__file__).resolve().with_name(name + ".py")
    specification = importlib.util.spec_from_file_location("aces_site_" + name, path)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def require_profile(value):
    """Require the exact profile, including strict boolean and integer types."""
    require(isinstance(value, dict) and set(value) == set(PROFILE)
        and all(type(value[k]) is type(v) and value[k] == v for k, v in PROFILE.items()),
        "the complete explicit ACES profile is required")
    return value


def pci_address(value):
    """Normalize the PCI domain width without treating PCI order as a GPU number."""
    match = re.fullmatch(r"([0-9a-fA-F]{4,8}):([0-9a-fA-F]{2}):([0-9a-fA-F]{2})\.([0-7])", value)
    require(match is not None, "a complete PCI address is required")
    domain, bus, device, function = match.groups()
    return f"{int(domain, 16):04x}:{bus.lower()}:{device.lower()}.{function}"


def field(text, name):
    """Read one exact field from an original scheduler response."""
    values = re.findall(r"(?:^|\s)" + re.escape(name) + r"=([^\s]+)", text)
    require(len(values) == 1, "missing or ambiguous ACES scheduler field: " + name)
    return values[0]


def cached_file(path, kind):
    """Record only allowlisted configuration entries and a hash of the complete original file."""
    path = Path(path)
    record = {"path": str(path), "kind": kind, "status": "absent"}
    try:
        information = path.lstat()
    except FileNotFoundError:
        return record
    # Keep failed trust checks as evidence; do not follow a configurable symlink.
    record.update(owner_uid=information.st_uid, mode=oct(stat.S_IMODE(information.st_mode)),
        inode=information.st_ino, device=information.st_dev, mtime_ns=information.st_mtime_ns,
        regular_file=stat.S_ISREG(information.st_mode), symlink=stat.S_ISLNK(information.st_mode))
    if not stat.S_ISREG(information.st_mode):
        record["status"] = "not_regular_file"
        return record
    try:
        original = path.read_bytes()
    except PermissionError:
        record["status"] = "unreadable"
        return record
    # Before: complete file bytes. After: exact relevant lines plus the complete-file SHA.
    record.update(bytes=len(original), sha256=hashlib.sha256(original).hexdigest())
    try:
        text = original.decode("utf-8")
    except UnicodeDecodeError:
        record["status"] = "not_utf8"
        return record
    after = path.lstat()
    if (after.st_ino, after.st_dev, after.st_mtime_ns, after.st_size) != (information.st_ino, information.st_dev, information.st_mtime_ns, information.st_size):
        record["status"] = "changed_during_read"
        return record
    rows = [line for line in text.splitlines(keepends=True)
        if line.strip() and not line.lstrip().startswith("#")]
    names = SLURM_KEYS if kind == "slurm" else CGROUP_KEYS
    selected = [line for line in rows if (kind == "gres" and
        (re.search(r"(?:^|\s)(?:Name|AutoDetect|NodeName)=", line, re.I) or re.match(r"\s*Include\b", line, re.I)))
        or (kind != "gres" and (any(re.match(r"\s*" + re.escape(name) + r"\s*=", line, re.I) for name in names)
            or re.match(r"\s*Include\b", line, re.I)))]
    filtered = "".join(selected)
    record.update(status="readable", bytes=len(original), sha256=hashlib.sha256(original).hexdigest(),
        filtered_bytes=filtered, filtered_sha256=hashlib.sha256(filtered.encode()).hexdigest())
    return record


def configuration_observations(environment, running):
    """Inspect only fixed and explicitly observed Slurm configuration directories."""
    directories = {"/etc/slurm", PROFILE["configuration_directory"]}
    paths = {"environment": environment.get("SLURM_CONF"),
        "running_controller": running.get("fields", {}).get("SLURM_CONF")}
    for value in paths.values():
        if value:
            require(Path(value).is_absolute() and Path(value).name == "slurm.conf",
                "only an observed absolute slurm.conf path is allowed")
            require(str(Path(value).parent) in directories, "observed Slurm path is outside the explicit ACES directories")
    # Before: two known directories. After: six individually recorded candidate files.
    return {"observed_paths": paths, "candidates": [cached_file(Path(directory) / filename, kind)
        for directory in sorted(directories) for filename, kind in
        (("slurm.conf", "slurm"), ("gres.conf", "gres"), ("cgroup.conf", "cgroup"))]}


def trusted_file(record):
    """Require root-owned ordinary immutable configuration bytes and reject omitted includes."""
    require(record.get("status") == "readable" and record.get("owner_uid") == 0
        and record.get("regular_file") is True and record.get("symlink") is False
        and int(record["mode"], 8) & 0o022 == 0,
        "active ACES configuration must be root-owned ordinary files without group or other write access")
    require(hashlib.sha256(record["filtered_bytes"].encode()).hexdigest() == record["filtered_sha256"]
        and re.fullmatch(r"[0-9a-f]{64}", record["sha256"]), "cached configuration byte hashes changed")
    require(not re.search(r"(?im)^\s*Include\b", record["filtered_bytes"]),
        "included ACES configuration needs a separate complete reviewed capture")
    return record["filtered_bytes"]


def configuration_values(text):
    """Parse one value per selected key without silently resolving duplicate assignments."""
    values = {}
    for line in text.splitlines():
        match = re.fullmatch(r"\s*([A-Za-z][A-Za-z0-9_]*)\s*=\s*([^#]*?)\s*(?:#.*)?", line)
        require(match is not None and match[1] in (*SLURM_KEYS, *CGROUP_KEYS, "SLURM_CONF")
            and match[1].lower() not in {key.lower() for key in values},
            "invalid, noncanonical or duplicate selected configuration key")
        values[match[1]] = match[2]
    return values


def expand_list(value):
    """Expand explicit bounded decimal ranges without guessing unspecified host or file order."""
    match = re.fullmatch(r"([^\[\],]*)\[([0-9,\-]+)\]([^\[\],]*)", value)
    if match is None:
        require(re.fullmatch(r"[^\[\],]+", value) is not None, "unsupported Slurm list expression")
        return [value]
    prefix, groups, suffix = match.groups()
    result = []
    for group in groups.split(","):
        endpoints = group.split("-")
        require(len(endpoints) in (1, 2) and all(item.isdigit() for item in endpoints), "unsupported decimal range")
        first, last = int(endpoints[0]), int(endpoints[-1])
        require(first <= last and last - first < 256, "configuration range is reversed or too large")
        width = len(endpoints[0]) if endpoints[0].startswith("0") else 0
        result.extend(prefix + str(number).zfill(width) + suffix for number in range(first, last + 1))
    require(len(result) == len(set(result)), "duplicate expanded configuration entry")
    return result


def gres_devices(text, hostname):
    """Require explicit ordered device files for this host instead of inferring an AutoDetect order."""
    devices = []
    for line in text.splitlines():
        tokens = line.split("#", 1)[0].split()
        pairs = [token.split("=", 1) for token in tokens]
        require(all(len(pair) == 2 for pair in pairs) and len({pair[0] for pair in pairs}) == len(pairs),
            "invalid or duplicate GRES line keys")
        values = dict(pairs)
        if "NodeName" in values and hostname not in expand_list(values["NodeName"]):
            continue
        if values.get("Name") != "gpu":
            require(set(values) <= {"AutoDetect", "NodeName"}, "unreviewed non-GPU GRES configuration")
            continue
        require(values.get("Type") == "h100" and "File" in values
            and set(values) <= {"Name", "Type", "File", "NodeName", "AutoDetect", "CPUs", "Cores", "Links", "Count"},
            "each ACES GPU must have an explicit typed physical File mapping")
        files = expand_list(values["File"])
        require(all(re.fullmatch(r"/dev/nvidia[0-9]+", path) for path in files), "GRES File must name whole NVIDIA character devices")
        require(values.get("Count", str(len(files))) == str(len(files)), "GRES Count differs from explicit device files")
        devices.extend(files)
    require(devices and len(devices) == len(set(devices)), "missing or duplicate physical GRES device files")
    require(devices == sorted(devices, key=lambda path: int(path.removeprefix("/dev/nvidia"))),
        "configured GPU File entries must retain the documented increasing numeric order")
    return devices


def validate_configuration(configuration, isolation, hostname):
    """Bind the node's selected cache and matching running plugins without claiming kernel attestation."""
    expected = PROFILE["configuration_directory"]
    require(configuration["observed_paths"]["environment"] == expected + "/slurm.conf",
        "the actual step must identify the explicit root-owned node cache")
    candidates = {row["path"]: row for row in configuration["candidates"]}
    require(len(candidates) == len(configuration["candidates"]), "duplicate cached configuration candidate paths are ambiguous")
    selected = {name: candidates[expected + "/" + name + ".conf"] for name in ("slurm", "gres", "cgroup")}
    values = {name: trusted_file(row) for name, row in selected.items()}
    # A competing present node-local configuration is ambiguous even if the controller reports it.
    require(all(row.get("status") == "absent" for path, row in candidates.items() if not path.startswith(expected + "/")),
        "multiple readable or uninspectable node configuration candidates are ambiguous")
    slurm, cgroup = configuration_values(values["slurm"]), configuration_values(values["cgroup"])
    running = isolation["running_slurm_configuration"]
    require(running.get("command") == ["scontrol", "show", "config"] and type(running.get("returncode")) is int and running["returncode"] == 0,
        "running Slurm settings must retain their original successful query")
    selected_keys = ["TaskPlugin", "ProctrackType", "SLURM_CONF"]
    rebuilt = adjacent("h100_nvl_allocation_observation").selected_config(running["filtered_bytes"], selected_keys)
    require(running.get("selected_keys") == selected_keys
        and all(running.get(key) == rebuilt[key] for key in ("fields", "missing_fields", "filtered_bytes", "sha256", "hash_scope"))
        and configuration["observed_paths"].get("running_controller") == running["fields"].get("SLURM_CONF")
        and configuration_values(running["filtered_bytes"]) == running["fields"],
        "running Slurm selected fields, original filtered bytes and SHA must agree")
    require(slurm.get("TaskPlugin") == running["fields"].get("TaskPlugin") == "task/cgroup,task/affinity"
        and slurm.get("ProctrackType") == running["fields"].get("ProctrackType") == "proctrack/cgroup",
        "root-owned cache and running task/cgroup settings must agree")
    require("gpu" in slurm.get("GresTypes", "").split(",")
        and cgroup.get("ConstrainDevices", "").lower() == cgroup.get("ConstrainCores", "").lower() == "yes"
        and cgroup.get("CgroupPlugin", "autodetect") in ("autodetect", "cgroup/v1")
        and all(cgroup.get(name, "no").lower() == "no" for name in ("IgnoreSystemd", "IgnoreSystemdOnFailure")),
        "the explicit ACES profile requires configured version-one task device and CPU restrictions")
    return {"configuration_files": selected, "ordered_gres_device_files": gres_devices(values["gres"], hostname),
        "isolation_basis": PROFILE["isolation_basis"], "kernel_device_enforcement_verified": False}


def validate_allocation(observation):
    """Join recorded Slurm GRES files to proc UUID/PCI data and this exact owned task."""
    require_profile(observation["qualification_profile"])
    owned, inventory = observation["owned_allocation"], observation["host_gpu_inventory"]
    environment, host = owned["allocation_environment"], owned["hostname"]
    require(owned.get("uid") == PROFILE["uid"] and re.fullmatch(r"ac[0-9]+", host), "wrong ACES owner or compute host")
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
    files = configuration["ordered_gres_device_files"]
    node_gres = re.fullmatch(r"gpu:h100:([0-9]+)(?:\(S:[0-9,-]+\))?", field(node_text, "Gres"))
    require(node_gres is not None and int(node_gres[1]) == len(files),
        "the explicit root-owned File roster must match the actual node's complete typed GPU count")
    require(int(index[0]) < len(files), "allocated GRES index is outside the explicit configured device roster")
    selected_file = files[int(index[0])]
    devices = {row["path"]: row for row in inventory["device_files"]}
    require(len(devices) == len(inventory["device_files"]), "duplicate observed device-file paths are ambiguous")
    require(set(files) == set(devices) and all(row.get("character_device") is True for row in devices.values()),
        "the configured whole-GPU roster must exactly match observed NVIDIA character devices")
    require(inventory["proc_gpu_inventory"]["status"] == inventory["pci_gpu_inventory"]["status"] == "present",
        "complete proc and PCI metadata are required")
    proc = inventory["proc_gpu_inventory"]["cards"]
    require(len(proc) == len(files) and len({row["fields"].get("Device Minor") for row in proc}) == len(proc),
        "proc GPU minor metadata must be complete and unambiguous")
    require({str(row["minor"]) for row in devices.values()} == {row["fields"].get("Device Minor") for row in proc}
        and {pci_address(row["fields"].get("Bus Location", "")) for row in proc}
        == {pci_address(row["pci_address"]) for row in inventory["pci_gpu_inventory"]["cards"]},
        "configured devices, proc minors and PCI inventory must describe one complete roster")
    card, = [row for row in proc if row["status"] == "readable"
        and row["fields"].get("Device Minor") == str(devices[selected_file]["minor"])]
    physical = card["fields"]
    hardware = observation["hardware_queries"]["gpu_csv"]
    fields = ["index", "uuid", "name", "pci.bus_id", "memory.total", "driver_version"]
    require(hardware.get("command") == ["nvidia-smi", "--query-gpu=" + ",".join(fields), "--format=csv,noheader,nounits"]
        and hardware.get("fields") == fields and type(hardware.get("returncode")) is int,
        "the hardware observation must retain the exact allowlisted CSV command and field roster")
    rows = list(csv.reader(hardware["stdout"].splitlines()))
    require(hardware["returncode"] == 0 and len(rows) == 1 and len(rows[0]) == 6,
        "the recorded hardware-only NVML query must expose exactly one whole GPU")
    visible_index, gpu, model, pci, memory, driver = [value.strip() for value in rows[0]]
    require(physical.get("GPU UUID") == gpu and physical.get("Model") == model == PROFILE["model"]
        and pci_address(physical.get("Bus Location", "")) == pci_address(pci)
        and memory == str(PROFILE["memory_mib"]) and driver == PROFILE["driver"],
        "configured physical device, proc identity and visible hardware disagree")
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
        "kernel_device_enforcement_verified": False, **BOUNDARY}


def cuda_device_identity(device):
    """Resolve only the already initialized JAX device through the already mapped CUDA driver."""
    require(device.platform == "gpu" and device.device_kind == PROFILE["model"] and type(device.local_hardware_id) is int,
        "JAX must identify the actual ACES whole-GPU ordinal")
    paths = {line.split()[-1] for line in Path("/proc/self/maps").read_text().splitlines()
        if line.split() and Path(line.split()[-1]).name.startswith("libcuda.so")}
    require(len(paths) == 1, "one already mapped CUDA driver image is required")
    path, = paths
    require(path.endswith("." + PROFILE["driver"]), "the loaded CUDA driver differs from the explicit ACES profile")
    driver = ctypes.CDLL(path, mode=os.RTLD_LOCAL | os.RTLD_NOLOAD)
    count, handle = ctypes.c_int(), ctypes.c_int()
    driver.cuDeviceGetCount.argtypes, driver.cuDeviceGetCount.restype = [ctypes.POINTER(ctypes.c_int)], ctypes.c_int
    driver.cuDeviceGet.argtypes, driver.cuDeviceGet.restype = [ctypes.POINTER(ctypes.c_int), ctypes.c_int], ctypes.c_int
    codes = {"cuDeviceGetCount": driver.cuDeviceGetCount(ctypes.byref(count))}
    require(codes["cuDeviceGetCount"] == 0 and count.value == 1, "CUDA must expose exactly the assigned whole GPU")
    codes["cuDeviceGet"] = driver.cuDeviceGet(ctypes.byref(handle), device.local_hardware_id)
    require(codes["cuDeviceGet"] == 0, "the actual JAX ordinal is not a valid CUDA device")
    raw_uuid, pci = (ctypes.c_ubyte * 16)(), ctypes.create_string_buffer(32)
    driver.cuDeviceGetUuid_v2.argtypes, driver.cuDeviceGetUuid_v2.restype = [ctypes.c_void_p, ctypes.c_int], ctypes.c_int
    driver.cuDeviceGetPCIBusId.argtypes, driver.cuDeviceGetPCIBusId.restype = [ctypes.c_char_p, ctypes.c_int, ctypes.c_int], ctypes.c_int
    codes["cuDeviceGetUuid_v2"] = driver.cuDeviceGetUuid_v2(ctypes.byref(raw_uuid), handle.value)
    codes["cuDeviceGetPCIBusId"] = driver.cuDeviceGetPCIBusId(pci, len(pci), handle.value)
    require(all(code == 0 for code in codes.values()), "same-process CUDA UUID and PCI queries must succeed")
    return {"schema_version": 1, "kind": "same_process_aces_jax_cuda_identity", "pid": os.getpid(),
        "jax_local_hardware_id": device.local_hardware_id, "cuda_device_count": count.value,
        "cuda_device_handle": handle.value, "api_return_codes": codes,
        "gpu_uuid": "GPU-" + str(uuid.UUID(bytes=bytes(raw_uuid))), "pci_address": pci_address(pci.value.decode("ascii")),
        "driver_library": path, "driver_library_sha256": adjacent("h100_nvl_slurm_site_validation").sha256(path)}


def validate_cuda_identity(record, allocation, pid):
    """Bind same-process CUDA identity to configured physical UUID and PCI before any selector change."""
    require(type(record.get("schema_version")) is int and record["schema_version"] == 1
        and record.get("kind") == "same_process_aces_jax_cuda_identity"
        and type(record.get("pid")) is int and record["pid"] == pid
        and type(record.get("cuda_device_count")) is int and record["cuda_device_count"] == 1
        and type(record.get("jax_local_hardware_id")) is int and record["jax_local_hardware_id"] == 0
        and type(record.get("cuda_device_handle")) is int and record["cuda_device_handle"] >= 0
        and record.get("api_return_codes") == {name: 0 for name in
            ("cuDeviceGetCount", "cuDeviceGet", "cuDeviceGetUuid_v2", "cuDeviceGetPCIBusId")}
        and all(type(code) is int for code in record["api_return_codes"].values())
        and record.get("gpu_uuid") == allocation["gpu_uuid"]
        and pci_address(record.get("pci_address", "")) == allocation["pci_address"]
        and record.get("driver_library", "").endswith("." + PROFILE["driver"])
        and re.fullmatch(r"[0-9a-f]{64}", record.get("driver_library_sha256", "")),
        "the initialized JAX ordinal does not prove this owned physical CUDA UUID/PCI/driver")
    return record
