"""Validate an explicit Slurm device-isolation profile without equating GPU numbering systems.

Slurm 23.11.6 applies GRES character-device rules in task_cgroup_devices.c.
Its cgroup/v2 plugin attaches device BPF programs; this profile relies on the
recorded scheduler configuration, not independent inspection of those programs.
"""

import csv
import ctypes
import hashlib
import os
from pathlib import Path, PurePosixPath
import re
import uuid
import xml.etree.ElementTree as ET


NAME = "slurm-configured-device-isolation-driver-580.159.04"
PROFILE = {"schema_version": 1, "name": NAME, "expected_driver": "580.159.04",
    "scheduler_version": "23.11.6", "eligible_hosts": ["serval06", "serval07", "serval08", "serval09"],
    "model": "NVIDIA H100 NVL", "memory_mib": 95830,
    "isolation_basis": "scheduler-trusted configured task/cgroup ConstrainDevices device restrictions",
    "numeric_gpu_mapping_reconstructed": False, "device_bpf_rules_independently_inspected": False,
    "workers_permitted": False, "campaign_claims_permitted": False, "scientific_hypothesis_credit": 0,
    "slurm_sources": ["https://github.com/SchedMD/slurm/blob/slurm-23-11-6-1/src/plugins/task/cgroup/task_cgroup_devices.c",
        "https://github.com/SchedMD/slurm/blob/slurm-23-11-6-1/src/plugins/cgroup/v2/cgroup_v2.c",
        "https://github.com/SchedMD/slurm/blob/slurm-23-11-6-1/src/plugins/gres/common/gres_common.c"]}
CONFIG_KEYS = ("CgroupPlugin", "ConstrainDevices", "ConstrainCores", "CgroupMountpoint", "IgnoreSystemd", "IgnoreSystemdOnFailure")
PLUGIN_KEYS = ("TaskPlugin", "ProctrackType")


def require(condition, message):
    """Reject incomplete or contradictory raw evidence with a literal description."""
    if not condition:
        raise ValueError(message)


def require_profile(value):
    """Require exact profile values and types so numeric zero cannot replace a false permission flag."""
    require(isinstance(value, dict) and set(value) == set(PROFILE)
        and all(type(value[name]) is type(expected) and value[name] == expected for name, expected in PROFILE.items()),
        "a complete explicit configured-isolation profile is required")
    return value


def pci_address(value):
    """Normalize only the PCI domain width; never use PCI order as a scheduler GPU number."""
    match = re.fullmatch(r"([0-9a-fA-F]{4,8}):([0-9a-fA-F]{2}):([0-9a-fA-F]{2})\.([0-7])", value)
    require(match is not None, "a complete GPU PCI address is required")
    domain, bus, device, function = match.groups()
    return f"{int(domain, 16):04x}:{bus.lower()}:{device.lower()}.{function}"


def field(text, name):
    """Read one unambiguous original scheduler field without accepting another record."""
    values = re.findall(r"(?:^|\s)" + re.escape(name) + r"=([^\s]+)", text)
    require(len(values) == 1, "missing or ambiguous scheduler field: " + name)
    return values[0]


def scheduler_identity(observation):
    """Compare scheduler identifiers only with the owned scheduler job and step records."""
    # Compare scheduler strings with their own records, never with a GPU minor or NVML index.
    environment, host = observation["allocation_environment"], observation["hostname"]
    job, step = environment.get("SLURM_JOB_ID") or "", environment.get("SLURM_STEP_ID") or ""
    require(host in PROFILE["eligible_hosts"] and re.fullmatch(r"[1-9][0-9]*", job)
        and re.fullmatch(r"[0-9]+", step), "qualification requires one exact eligible Slurm job and numerical step")
    require(environment.get("SLURM_GPUS_ON_NODE") == "1" and environment.get("SLURM_CPUS_PER_TASK") == "8"
        and environment.get("SLURM_LOCALID") == "0", "qualification requires one GPU and its eight-CPU single task")
    rows = {}
    for kind, identifier in (("job", job), ("step", job + "." + step)):
        record = observation["scheduler_" + kind]
        require(record.get("command") == ["scontrol", "show", kind, "-d", identifier]
            and record.get("returncode") == 0, "qualification lacks its exact successful owned scheduler query")
        rows[kind] = record["stdout"]
    job_text, step_text = rows["job"], rows["step"]
    node = observation["scheduler_node"]
    require(node.get("command") == ["scontrol", "show", "node", "-d", host] and node.get("returncode") == 0
        and field(node["stdout"], "NodeName") == host and field(node["stdout"], "Version") == PROFILE["scheduler_version"],
        "the actual allocated node must record the qualified Slurm version")
    require(field(job_text, "JobId") == job and field(job_text, "JobState") == "RUNNING"
        and field(job_text, "NodeList") == host and field(job_text, "NumNodes") == "1"
        and field(job_text, "NumCPUs") == "8" and field(job_text, "JOB_GRES") == "gpu:nvidia_h100_nvl:1"
        and field(job_text, "TresPerNode") == "gres/gpu:nvidia_h100_nvl:1", "owned job must allocate one typed H100 NVL and eight CPUs")
    indices = re.findall(r"\bNodes=" + re.escape(host) + r"\s+[^\n]*?\bGRES=gpu:nvidia_h100_nvl:1\(IDX:([0-9]+)\)(?:\s|$)", job_text)
    require(len(indices) == 1, "owned detailed job must record one exact typed GPU scheduler index")
    require(all(environment.get(name) == indices[0] for name in ("SLURM_JOB_GPUS", "SLURM_STEP_GPUS")),
        "job GPU, step GPU and detailed GRES scheduler identifiers disagree")
    require(field(step_text, "StepId") == job + "." + step and field(step_text, "State") == "RUNNING"
        and field(step_text, "NodeList") == host and field(step_text, "Nodes") == "1"
        and field(step_text, "CPUs") == "8" and field(step_text, "Tasks") == "1", "owned step must identify this exact one-node eight-CPU task")
    for text, name in ((job_text, "AllocTRES"), (step_text, "TRES")):
        resources = dict(item.split("=", 1) for item in field(text, name).split(","))
        require(resources.get("gres/gpu") == "1" and all(value == "1" for key, value in resources.items()
            if key.startswith("gres/gpu:")), "owned job and step must each allocate exactly one GPU")
    uid = observation["device_isolation_observations"]["own_placement"]["uid"]
    require(field(step_text, "UserId") == str(uid) and re.fullmatch(r"[^\s()]+\(" + str(uid) + r"\)", field(job_text, "UserId")),
        "owned scheduler job and step must belong to the actual process UID")
    return {"job_id": job, "step_id": step, "hostname": host, "scheduler_gpu_id": indices[0],
        "numeric_gpu_mapping_reconstructed": False}


def filtered_configuration(record, names):
    """Rebuild only the recorded allowlisted configuration bytes and reject missing inputs."""
    require(record.get("status") == "readable" and record.get("selected_keys") == list(names)
        and isinstance(record.get("fields"), dict) and set(record["fields"]) <= set(names), "required selected configuration is unavailable")
    text = "".join(name + "=" + record["fields"][name] + "\n" for name in names if name in record["fields"])
    require(record.get("filtered_bytes") == text and record.get("sha256") == hashlib.sha256(text.encode()).hexdigest(),
        "selected configuration bytes or their hash changed")
    return record["fields"]


def cpu_set(text):
    """Expand a Linux CPU list and reject ambiguous or reversed ranges."""
    result = set()
    for item in text.strip().split(","):
        require(re.fullmatch(r"[0-9]+(?:-[0-9]+)?", item), "effective CPU list is absent or invalid")
        bounds = [int(value) for value in item.split("-")]
        first, last = bounds[0], bounds[-1]
        require(first <= last, "effective CPU range is reversed")
        result.update(range(first, last + 1))
    return result


def configured_isolation(observation, scheduler, physical):
    """Require actual own task placement and explicitly configured scheduler device restrictions."""
    # Configuration supports the stated scheduler trust; it does not reveal a live version2 BPF program.
    raw = observation["device_isolation_observations"]
    node, running, placement = raw["node_cgroup_configuration"], raw["running_slurm_configuration"], raw["own_placement"]
    config, plugins = filtered_configuration(node, CONFIG_KEYS), filtered_configuration(running, PLUGIN_KEYS)
    require("task/cgroup" in plugins.get("TaskPlugin", "").split(",")
        and plugins.get("ProctrackType") == "proctrack/cgroup" and config.get("ConstrainDevices", "").lower() == "yes",
        "qualification requires configured task/cgroup, proctrack/cgroup and ConstrainDevices=yes")
    metadata = node["file"]
    require(metadata.get("path") == "/etc/slurm/cgroup.conf" and metadata.get("status") == "readable" and metadata.get("owner_uid") == 0
        and int(metadata.get("mode", "0"), 8) & 0o022 == 0, "node cgroup configuration must be root-owned and not writable by group or others")
    require(running.get("command") == ["scontrol", "show", "config"] and running.get("returncode") == 0,
        "task plugin settings must come from the recorded running Slurm configuration")
    require(config.get("IgnoreSystemd", "no").lower() == "no" and config.get("IgnoreSystemdOnFailure", "no").lower() == "no",
        "qualification does not accept the development systemd bypass settings")
    # Bind the exact current process and CPU affinity to its recorded job/step/task path.
    pid, birth = observation["pid"], observation["process_start_time"]
    require(type(pid) is int and pid > 0 and str(birth).isdigit() and placement.get("pid") == pid
        and placement.get("process_start_time") == birth and type(placement.get("uid")) is int,
        "own cgroup placement must identify this exact process and birth counter")
    cpus = placement.get("cpu_affinity", [])
    require(len(cpus) == 8 and len(set(cpus)) == 8 and all(type(value) is int and value >= 0 for value in cpus),
        "own task must retain eight actual distinct CPUs")
    membership = placement["own_cgroup_file"]["text"]
    require(placement["own_cgroup_file"].get("sha256") == hashlib.sha256(membership.encode()).hexdigest(), "own cgroup membership bytes changed")
    expected_parts = ["job_" + scheduler["job_id"], "step_" + scheduler["step_id"], "task_0"]
    candidates = []
    mountinfo = placement["cgroup_mountinfo"]
    mount_bytes = "".join(entry["original_line"] + "\n" for entry in mountinfo["entries"])
    require(mountinfo.get("filtered_bytes") == mount_bytes
        and mountinfo.get("sha256") == hashlib.sha256(mount_bytes.encode()).hexdigest(), "actual cgroup mountinfo bytes changed")
    # Rebuild the actual mounted task hierarchy before interpreting its version-specific evidence.
    for entry in placement["cgroups"]:
        parts = PurePosixPath(entry["cgroup_path"]).parts
        if not all(parts.count(part) == 1 for part in expected_parts):
            continue
        positions = [parts.index(part) for part in expected_parts]
        if positions != sorted(positions) or parts[-1] != "task_0" or entry.get("task_path_observed") is not True:
            continue
        exact_line = entry["hierarchy"] + ":" + ",".join(entry["controllers"]) + ":" + entry["cgroup_path"]
        require(exact_line in membership.splitlines(), "recorded task path is not in actual cgroup membership")
        for mount in entry["mounts"]:
            filesystem = mount["filesystem"]
            if filesystem == "cgroup" and "devices" not in entry["controllers"]:
                continue
            require(filesystem in ("cgroup", "cgroup2") and [level["level"] for level in mount["levels"]] == ["job", "step", "task"],
                "own device hierarchy must include exact job, step and task levels")
            matches = [item for item in mountinfo["entries"] if item["filesystem"] == filesystem and item["mountpoint"] == mount["mountpoint"]]
            require(len(matches) == 1, "own device hierarchy lacks its original cgroup mount")
            mounted, = matches
            require(config.get("CgroupMountpoint", "/sys/fs/cgroup") == mounted["mountpoint"],
                "configured cgroup mountpoint differs from the actual task device hierarchy")
            for level, position in zip(mount["levels"], positions):
                expected_path = str(Path(mounted["mountpoint"]) / PurePosixPath(level["cgroup_path"]).relative_to(mounted["root"]))
                require(level["cgroup_path"] == str(PurePosixPath(*parts[:position + 1]))
                    and level["directory"].get("path") == expected_path and level["directory"].get("status") == "readable"
                    and type(level["directory"].get("inode")) is int,
                    "exact own cgroup directory evidence is unavailable")
            task = mount["levels"][-1]
            if filesystem == "cgroup2":
                require(entry["hierarchy"] == "0" and entry["controllers"] == []
                    and config.get("CgroupPlugin", "autodetect") in ("autodetect", "cgroup/v2"), "configured cgroup plugin contradicts the actual unified hierarchy")
                effective = task.get("effective_cpuset", {})
                require(effective.get("status") == "readable" and set(cpus) <= cpu_set(effective["text"]), "actual eight-CPU affinity is outside the effective task cpuset")
            else:
                require(config.get("CgroupPlugin", "autodetect") in ("autodetect", "cgroup/v1"), "configured cgroup plugin contradicts the actual device hierarchy")
                rules = task.get("devices_list", {})
                require(rules.get("status") == "readable", "actual version1 task device rules are unavailable")
                validate_device_rules(rules["text"], complete_host_devices(observation["host_gpu_inventory"]), physical)
            candidates.append({"filesystem": filesystem, "cgroup_path": entry["cgroup_path"], "cpu_affinity": cpus})
    require(len(candidates) == 1, "qualification requires one unambiguous actual job/step/task device hierarchy")
    return {**candidates[0], "basis": PROFILE["isolation_basis"], "device_bpf_rules_independently_inspected": False,
        "configuration_observed": True, "independent_kernel_rule_verification": candidates[0]["filesystem"] == "cgroup"}


def validate_device_rules(text, devices, physical):
    """Require a version1 task rule set to allow only the selected physical GPU among host GPU devices."""
    rules = []
    for line in text.splitlines():
        match = re.fullmatch(r"([acb]) ([0-9]+|\*):([0-9]+|\*) ([rwm]+)", line.strip())
        require(match is not None, "invalid version1 task device rule")
        rules.append(match.groups())
    for device in devices:
        require(device.get("character_device") is True, "host GPU device entry is not a character device")
        permissions = set().union(*(set(access) for kind, major, minor, access in rules if kind in ("a", "c")
            and major in ("*", str(device["major"])) and minor in ("*", str(device["minor"]))))
        selected = device["minor"] == int(physical["physical_minor"]) and device["major"] == physical["device_major"]
        require({"r", "w"} <= permissions if selected else not permissions.intersection("rw"),
            "version1 task device rules do not isolate the selected physical GPU")


def complete_host_devices(inventory):
    """Require complete matching proc, PCI and character-device GPU rosters before a version1 rule audit."""
    require(inventory["proc_gpu_inventory"].get("status") == "present" and inventory["pci_gpu_inventory"].get("status") == "present",
        "version1 device-rule verification requires complete host GPU inventory")
    proc, pci, minors, identifiers = set(), set(), set(), set()
    for card in inventory["proc_gpu_inventory"]["cards"]:
        fields = card.get("fields", {})
        minor, gpu = fields.get("Device Minor", ""), fields.get("GPU UUID", "")
        address = pci_address(fields.get("Bus Location", ""))
        require(card.get("status") == "readable" and minor.isdecimal() and gpu.startswith("GPU-")
            and pci_address(card["pci_directory"]) == address and address not in proc and minor not in minors and gpu not in identifiers,
            "version1 host proc GPU inventory is unreadable, incomplete or ambiguous")
        proc.add(address)
        minors.add(minor)
        identifiers.add(gpu)
    for card in inventory["pci_gpu_inventory"]["cards"]:
        address = pci_address(card["pci_address"])
        require(address not in pci, "version1 host PCI GPU roster is ambiguous")
        pci.add(address)
    devices = inventory["device_files"]
    require(proc and proc == pci and len(devices) == len(minors)
        and {device.get("path") for device in devices} == {"/dev/nvidia" + minor for minor in minors}
        and all(device.get("path") == "/dev/nvidia" + str(device.get("minor")) and device.get("character_device") is True
                and type(device.get("major")) is int and device["major"] > 0 for device in devices),
        "version1 host GPU inventory omits or substitutes a physical device")
    return devices


def physical_identity(observation):
    """Bind one visible whole GPU to UUID, PCI and device metadata without using its scheduler number."""
    # For example, scheduler ID1 and physical minor0 remain two independent identifiers.
    rows, devices = list(csv.reader(observation["gpu_csv"].splitlines())), ET.fromstring(observation["gpu_xml"]).findall("gpu")
    require(len(rows) == 1 and len(rows[0]) == 5 and len(devices) == 1, "qualification requires exactly one visible whole physical GPU")
    index, gpu, model, memory, driver = [value.strip() for value in rows[0]]
    require(re.fullmatch(r"GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}", gpu) and index.isdecimal() and model == PROFILE["model"]
        and memory == str(PROFILE["memory_mib"]) and driver == PROFILE["expected_driver"], "visible physical GPU differs from the explicit qualification model, memory or driver")
    xml = devices[0]
    minor = xml.findtext("minor_number", "")
    require(xml.findtext("uuid") == gpu and xml.findtext("product_name") == model and minor.isdecimal(), "visible CSV and XML physical identities disagree")
    pci = pci_address(xml.findtext("pci/pci_bus_id", ""))
    # Match the selected UUID to its proc/sysfs PCI identity and exact character-device metadata.
    inventory = observation["host_gpu_inventory"]
    matches = [card for card in inventory["proc_gpu_inventory"]["cards"] if card.get("fields", {}).get("GPU UUID") == gpu]
    require(inventory["proc_gpu_inventory"].get("status") == "present" and len(matches) == 1, "selected UUID needs one actual proc GPU identity")
    card, = matches
    values = card["fields"]
    require(card.get("status") == "readable" and values.get("Model") == model and values.get("Device Minor") == minor
        and pci_address(values.get("Bus Location", "")) == pci and pci_address(card["pci_directory"]) == pci,
        "selected UUID, proc PCI address and physical device minor disagree")
    cards = [card for card in inventory["pci_gpu_inventory"]["cards"] if pci_address(card["pci_address"]) == pci]
    require(inventory["pci_gpu_inventory"].get("status") == "present" and len(cards) == 1, "selected GPU needs its actual NVIDIA PCI sysfs entry")
    fields = cards[0]["fields"]
    require(fields["vendor"].get("status") == "readable" and fields["vendor"].get("text", "").lower() == "0x10de"
        and fields["class"].get("status") == "readable" and re.fullmatch(r"0x03[0-9a-fA-F]{4}", fields["class"].get("text", "")),
        "selected PCI entry is not an observed NVIDIA GPU")
    nodes = [device for device in inventory["device_files"] if device.get("path") == "/dev/nvidia" + minor]
    require(len(nodes) == 1 and nodes[0].get("character_device") is True and nodes[0].get("minor") == int(minor)
        and type(nodes[0].get("major")) is int and nodes[0]["major"] > 0, "selected GPU lacks its matching character-device metadata")
    selector = observation["allocation_environment"].get("CUDA_VISIBLE_DEVICES", "")
    require(selector in ("0", index, gpu), "original CUDA visibility must select the sole visible GPU")
    return {"gpu_uuid": gpu, "pci_address": pci, "model": model, "memory_mib": int(memory), "driver": driver,
        "physical_minor": minor, "device_major": nodes[0]["major"], "visible_index": index,
        "cuda_visible_devices_original": selector}


def validate_allocation(observation):
    """Rebuild explicit configured-isolation evidence and refuse absent inputs before numerical work."""
    require(observation.get("schema_version") == 1 and observation.get("kind") == "h100_nvl_slurm_configured_isolation_observation"
        and observation.get("qualification_profile") is not None, "a complete explicit configured-isolation observation is required")
    require_profile(observation["qualification_profile"])
    scheduler, physical = scheduler_identity(observation), physical_identity(observation)
    isolation = configured_isolation(observation, scheduler, physical)
    return {**scheduler, **physical, "qualification_profile": NAME, "isolation": isolation,
        "slurm_allocations": {name: observation["allocation_environment"][name] for name in ("SLURM_JOB_GPUS", "SLURM_STEP_GPUS")},
        "gpu_csv": observation["gpu_csv"], "gpu_xml": observation["gpu_xml"]}


def capture(observer, environment, hostname, gpu_csv, gpu_xml, query, birth):
    """Collect only owned scheduler and allowlisted host metadata; do not create a GPU context."""
    selected = {name: environment.get(name) for name in observer.ALLOCATION_FIELDS}
    job, step = selected.get("SLURM_JOB_ID", ""), selected.get("SLURM_STEP_ID", "")
    require(re.fullmatch(r"[1-9][0-9]*", job or "") and re.fullmatch(r"[0-9]+", step or ""), "capture requires an exact current numerical Slurm step")
    records = {}
    for kind, identifier in (("job", job), ("step", job + "." + step)):
        command = ["scontrol", "show", kind, "-d", identifier]
        records["scheduler_" + kind] = {"command": command, "returncode": 0, "stdout": query(command)}
    command = ["scontrol", "show", "node", "-d", hostname]
    records["scheduler_node"] = {"command": command, "returncode": 0, "stdout": query(command)}
    cgroup = Path("/proc/self/cgroup").read_text()
    return {"schema_version": 1, "kind": "h100_nvl_slurm_configured_isolation_observation", "qualification_profile": PROFILE,
        "hostname": hostname, "allocation_environment": selected, "gpu_csv": gpu_csv, "gpu_xml": gpu_xml,
        "pid": os.getpid(), "process_start_time": birth, **records, "host_gpu_inventory": observer.host_inventory(),
        "device_isolation_observations": observer.device_isolation_observations(environment, cgroup)}


def cuda_device_identity(device):
    """Query the already initialized JAX CUDA ordinal through its already loaded driver image."""
    require(device.platform == "gpu" and device.device_kind == PROFILE["model"] and type(device.local_hardware_id) is int,
        "actual JAX device must identify one H100 NVL CUDA hardware ordinal")
    # Inspect only the already mapped driver; this function does not initialize or enumerate another GPU.
    paths = sorted({line.split()[-1] for line in Path("/proc/self/maps").read_text().splitlines()
        if line.split() and Path(line.split()[-1]).name.startswith("libcuda.so")})
    require(len(paths) == 1 and paths[0].endswith("." + PROFILE["expected_driver"]), "one already loaded qualified CUDA driver image is required")
    driver = ctypes.CDLL(paths[0], mode=os.RTLD_LOCAL | os.RTLD_NOLOAD)
    count, ordinal = ctypes.c_int(), ctypes.c_int()
    driver.cuDeviceGetCount.argtypes, driver.cuDeviceGetCount.restype = [ctypes.POINTER(ctypes.c_int)], ctypes.c_int
    driver.cuDeviceGet.argtypes, driver.cuDeviceGet.restype = [ctypes.POINTER(ctypes.c_int), ctypes.c_int], ctypes.c_int
    codes = {"cuDeviceGetCount": driver.cuDeviceGetCount(ctypes.byref(count))}
    require(codes["cuDeviceGetCount"] == 0 and count.value == 1, "CUDA must expose exactly the assigned whole GPU")
    codes["cuDeviceGet"] = driver.cuDeviceGet(ctypes.byref(ordinal), device.local_hardware_id)
    require(codes["cuDeviceGet"] == 0, "JAX hardware ordinal is not a valid actual CUDA device")
    # Use the actual JAX CUDA ordinal for both identity queries and retain every returned status.
    raw_uuid, pci = (ctypes.c_ubyte * 16)(), ctypes.create_string_buffer(32)
    driver.cuDeviceGetUuid_v2.argtypes, driver.cuDeviceGetUuid_v2.restype = [ctypes.c_void_p, ctypes.c_int], ctypes.c_int
    driver.cuDeviceGetPCIBusId.argtypes, driver.cuDeviceGetPCIBusId.restype = [ctypes.c_char_p, ctypes.c_int, ctypes.c_int], ctypes.c_int
    codes["cuDeviceGetUuid_v2"] = driver.cuDeviceGetUuid_v2(ctypes.byref(raw_uuid), ordinal.value)
    codes["cuDeviceGetPCIBusId"] = driver.cuDeviceGetPCIBusId(pci, len(pci), ordinal.value)
    require(all(value == 0 for value in codes.values()), "actual CUDA UUID and PCI queries must both succeed")
    digest = hashlib.sha256()
    with Path(paths[0]).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return {"schema_version": 1, "kind": "same_process_jax_cuda_device_identity", "pid": os.getpid(),
        "jax_local_hardware_id": device.local_hardware_id, "cuda_device_count": count.value,
        "cuda_device_handle": ordinal.value, "api_return_codes": codes,
        "gpu_uuid": "GPU-" + str(uuid.UUID(bytes=bytes(raw_uuid))), "pci_address": pci_address(pci.value.decode("ascii")),
        "driver_library": paths[0], "driver_library_sha256": digest.hexdigest()}


def validate_cuda_identity(record, allocation, pid):
    """Require the actual JAX CUDA ordinal to resolve to the qualified physical UUID and PCI address."""
    require(record.get("schema_version") == 1 and record.get("kind") == "same_process_jax_cuda_device_identity"
        and type(record.get("pid")) is int and record.get("pid") == pid and type(record.get("cuda_device_count")) is int
        and record.get("cuda_device_count") == 1 and type(record.get("jax_local_hardware_id")) is int and record.get("jax_local_hardware_id") == 0
        and type(record.get("cuda_device_handle")) is int and record.get("cuda_device_handle") >= 0
        and record.get("api_return_codes") == {name: 0 for name in ("cuDeviceGetCount", "cuDeviceGet", "cuDeviceGetUuid_v2", "cuDeviceGetPCIBusId")}
        and all(type(value) is int for value in record["api_return_codes"].values())
        and record.get("gpu_uuid", "").lower() == allocation["gpu_uuid"].lower()
        and pci_address(record.get("pci_address", "")) == allocation["pci_address"]
        and record.get("driver_library", "").endswith("." + PROFILE["expected_driver"])
        and re.fullmatch(r"[0-9a-f]{64}", record.get("driver_library_sha256", "")),
        "same-process JAX CUDA UUID, PCI, ordinal or qualified driver proof disagrees")
    return record
