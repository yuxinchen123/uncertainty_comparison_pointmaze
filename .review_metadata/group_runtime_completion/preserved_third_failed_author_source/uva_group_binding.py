"""Bind one owned numeric coordinator step and one owned step per A4500.

This is a separate group interface. It never interprets logical Slurm selectors
as physical minors, and it does not change the existing single-card interface.
The coordinator has zero GPUs; each numerical worker has exactly one GPU.
"""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import pwd
import re
import sys

PREFIX = "10_jax_exploration_platform/scripts/atari_pixel_campaign/"
CAPABILITY = "owned_numeric_coordinator_and_single_gpu_steps_current_context_version_1"
ENVIRONMENT = ("CUDA_VISIBLE_DEVICES", "SLURM_CONF", "SLURM_JOB_ID", "SLURM_STEP_ID",
    "SLURM_JOB_GPUS", "SLURM_STEP_GPUS", "SLURM_GPUS_ON_NODE", "SLURM_CPUS_PER_TASK")
NEW_HELPERS = ("uva_group_binding.py", "configured_gpu_scaling.py", "configured_scaling_evidence.py",
    "group_backend_package.py", "direct_backend_image.py", "group_runtime.py", "group_runtime_evidence.py",
    "group_original_qualification.py", "group_runtime_launch.py")

REQUIRED_HELPERS = ('measure_selected.py', 'diagnose_gpu_execution.py', 'post_comparison_validation.py', 'infrastructure_retry.py', 'pregather_async_retry.py', 'h100_nvl_slurm_queue.py', 'h100_nvl_slurm_case.py', 'h100_nvl_slurm_evidence.py', 'h100_nvl_slurm_site_validation.py', 'h100_nvl_slurm_qualification.py', 'h100_nvl_allocation_observation.py', 'train_optimized.py', 'short_learning.py', 'device_memory_limit.py', 'validate_training_deployment.py', 'portable_evidence.py', 'configured_measurement.py', 'uva_slurm_configured_isolation.py', 'uva_active_configuration_observation.py', 'passive_installed_slurm_observation.py', 'uva_group_binding.py', 'configured_gpu_scaling.py', 'configured_scaling_evidence.py')


def adjacent(name):
    """Load only an adjacent, explicitly staged standard-library helper."""
    path = Path(__file__).with_name(name + ".py")
    specification = importlib.util.spec_from_file_location("uva_group_" + name, path)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


single = adjacent("uva_slurm_configured_isolation")
PROFILE = single.PROFILE
BOUNDARY = {**single.BOUNDARY, "scientific_credit": 0, "packing_point_credit": 0,
    "throughput_credit": 0, "final_training_credit": 0}
GROUP_PROFILE = {"schema_version": 1,
    "name": "uva-jaguar03-owned-numeric-group-configured-isolation-version-1",
    "member_profile": PROFILE, "supported_gpu_counts": [1, 8],
    "isolation_basis": PROFILE["isolation_basis"], **BOUNDARY}


def require(value, message):
    """Reject incomplete or contradictory group evidence."""
    if not value:
        raise ValueError(message)


def exact(value, expected):
    """Keep Boolean, integer and floating-point fields distinct."""
    return single.exact(value, expected)


def encoded(value):
    """Use one stable JSON encoding for hashes and durable records."""
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def digest(raw):
    """Hash the exact supplied bytes."""
    return hashlib.sha256(raw).hexdigest()


def receipt(path):
    """Bind an exact ordinary file to its current bytes."""
    return single.receipt(path)


def read_receipt(value):
    """Read a duplicate-key-checked record from its exact hashed file."""
    return single.read_receipt(value)


def publish(path, value):
    """Expose only complete fsynced records, without overwriting a previous or partial attempt."""
    path = Path(path).absolute()
    temporary = path.with_name(path.name + ".prepared")
    require(path.parent.resolve(strict=True) == path.parent and not os.path.lexists(path)
        and not os.path.lexists(temporary), "preserve the prior immutable group publication attempt")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(encoded(value)); stream.flush(); os.fsync(stream.fileno())
    os.link(temporary, path, follow_symlinks=False)
    descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(descriptor)
        temporary.unlink()
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return receipt(path)


def unit_directory(plan):
    """Name the one group unit without resolving a different path."""
    return Path(plan["campaign_run_dir"]) / "data" / plan["unit_id"]


def worker_directory(plan, index):
    """Name one stable worker index independently of its eventual device."""
    require(type(index) is int and 0 <= index < plan["gpu_count"], "invalid group worker index")
    return unit_directory(plan) / "workers" / ("worker_" + str(index + 1))


def validate_plan(plan):
    """Require an explicit unresolved one- or eight-device group plan."""
    require(type(plan.get("schema_version")) is int and plan["schema_version"] == 2
        and plan.get("kind") == "configured_a4500_scaling_group"
        and plan.get("hardware_class") == "jaguar03_rtx_a4500",
        "an explicit new A4500 group plan is required")
    require(plan.get("mode") in {"qualification", "correctness", "measurement", "exact_eight_eligibility"},
        "new group plans require an explicit qualification, correctness or complete measurement mode")
    require(type(plan.get("gpu_count")) is int and plan["gpu_count"] in {1, 8}
        and type(plan.get("unit_id")) is str and Path(plan["unit_id"]).name == plan["unit_id"]
        and plan["unit_id"] not in {"", ".", ".."}, "invalid group shape or unit name")
    options = plan.get("configured_group")
    require(isinstance(options, dict) and set(options) == {"schema_version", "capability", "profile",
        "physical_identities", "resources", "owned_inputs_path"}
        and type(options["schema_version"]) is int and options["schema_version"] == 1
        and options["capability"] == CAPABILITY and exact(options["profile"], GROUP_PROFILE),
        "unsupported group configured-isolation declaration")
    require(exact(options["physical_identities"], [single.unresolved_identity() for _ in range(plan["gpu_count"])])
        and plan.get("gpu_uuids") == [None] * plan["gpu_count"],
        "the frozen group plan must leave all physical identities unresolved")
    resources = options["resources"]
    require(isinstance(resources, dict) and set(resources) == {"job", "coordinator", "worker"},
        "whole-job, coordinator and worker resources must be separate")
    for role, fields in (("job", {"requested_cpus", "allocated_cpus", "host_memory_mib", "gpus"}),
            ("coordinator", {"cpus", "host_memory_mib", "gpus"}), ("worker", {"cpus", "host_memory_mib", "gpus"})):
        row = resources[role]
        require(isinstance(row, dict) and set(row) == fields
            and all(type(value) is int and value > 0 for key, value in row.items() if key != "gpus")
            and type(row["gpus"]) is int, "exact positive resource counts are required")
    job, coordinator, worker = (resources[key] for key in ("job", "coordinator", "worker"))
    require(job["gpus"] == plan["gpu_count"] and coordinator["gpus"] == 0 and worker["gpus"] == 1
        and job["requested_cpus"] >= coordinator["cpus"] + plan["gpu_count"] * worker["cpus"]
        and job["allocated_cpus"] >= job["requested_cpus"]
        and job["host_memory_mib"] >= coordinator["host_memory_mib"] + plan["gpu_count"] * worker["host_memory_mib"],
        "whole-job resources cannot omit coordinator resources or independent worker steps")
    require(options["owned_inputs_path"] == str(unit_directory(plan) / "owned_group_inputs.json")
        and not any(key in plan for key in ("configured_isolation", "direct_binding", "final_protocol",
            "final_confirmation_protocol", "device_memory_limit_protocol")),
        "default, direct, capacity, final and portable protocols remain separate")
    return options


def source_identity(plan):
    """Recheck every staged helper against its declared committed Git byte sequence."""
    value = plan["helpers"]
    require(re.fullmatch(r"[0-9a-f]{40}", value["source_commit"]), "a full helper source commit is required")
    root = Path(value["source_root"])
    require(root.resolve(strict=True) == root, "helper source root must be exact")
    import subprocess
    names = set()
    for relative, expected in value["files"].items():
        require(relative.startswith(PREFIX) and Path(relative).parent.as_posix() == PREFIX.rstrip("/"),
            "group helper closure cannot use another source directory")
        name = Path(relative).name
        require(name not in names, "duplicate staged helper basename")
        names.add(name)
        raw = subprocess.check_output(["git", "-C", str(root), "show", value["source_commit"] + ":" + relative])
        require(expected == digest(raw) == receipt(root / relative)["sha256"]
            == receipt(Path(__file__).with_name(name))["sha256"], "a group helper source changed")
    require(set(REQUIRED_HELPERS) | set(NEW_HELPERS) <= names, "incomplete explicit group helper closure")
    return value


def owned_inputs(plan):
    """Prove one already-submitted owned job before any scheduler query."""
    options = validate_plan(plan)
    unit = unit_directory(plan)
    value = read_receipt(receipt(options["owned_inputs_path"]))
    require(set(value) == {"schema_version", "kind", "job_id", "plan", "ledger", "submissions",
        "submission", "release", "submission_row"}
        and type(value["schema_version"]) is int and value["schema_version"] == 1
        and value["kind"] == "uva_configured_group_owned_inputs",
        "unsupported exact group ownership record")
    job = value["job_id"]
    require(type(job) is str and re.fullmatch(r"[1-9][0-9]*", job)
        and value["plan"] == receipt(unit / "scaling_plan.json")
        and exact(read_receipt(value["plan"]), plan), "owned job or frozen group plan changed")
    script = str(Path(plan["campaign_run_dir"]) / "remote_setup" / plan["unit_id"] / "job.sbatch")
    row = [job, plan["kind"], plan["helpers"]["source_commit"], script]
    require(value["submission_row"] == row, "exact group submission row changed")
    for key, filename, member in (("ledger", "submitted_jobids_uva.txt", job),
            ("submissions", "submissions.tsv", "\t".join(row))):
        item = value[key]
        snapshot = unit / ("group_owned_job_ids.txt" if key == "ledger" else "group_owned_submissions.tsv")
        require(set(item) == {"snapshot", "source_path"} and item["source_path"] ==
            str(Path(plan["campaign_run_dir"]) / "slurm" / filename)
            and item["snapshot"] == receipt(snapshot), "group ledger snapshot changed")
        for candidate in (snapshot, Path(item["source_path"])):
            lines = single.ordinary(candidate).read_text().splitlines()
            require(lines.count(member) == 1, "group ownership is absent or duplicated")
            if key == "ledger":
                require(len(lines) == len(set(lines)), "owned job ledger contains duplicates")
            else:
                require([line for line in lines if line.split("\t", 1)[0] == job] == [member],
                    "another row uses this exact owned job")
    submission, release = read_receipt(value["submission"]), read_receipt(value["release"])
    for document, kind in ((submission, "owned_uva_configured_group_submission"),
            (release, "owned_uva_configured_group_release")):
        require(type(document.get("schema_version")) is int and document["schema_version"] == 1
            and document.get("kind") == kind and document.get("job_id") == job
            and document.get("cluster") == "uva" and document.get("data_unit") == str(unit)
            and document.get("plan_sha256") == value["plan"]["sha256"]
            and type(document.get("returncode")) is int and document["returncode"] == 0,
            "group submission/release receipt changed")
    command = submission.get("held_command")
    require(isinstance(command, list) and command[0] == "sbatch" and "--hold" in command
        and "--no-requeue" in command and command[-1] == script and submission.get("submission_row") == row
        and release.get("command") == ["scontrol", "release", job]
        and release.get("submission_receipt_sha256") == value["submission"]["sha256"],
        "the real held-submission/release chain is incomplete")
    return value


def selectors(environment, count, role):
    """Preserve logical scheduler selectors without mapping them to physical devices."""
    job = environment.get("SLURM_JOB_GPUS", "")
    require(re.fullmatch(r"(?:0|[1-9][0-9]*)(?:,(?:0|[1-9][0-9]*))*", job), "unsupported original whole-job GPU selector")
    job_tokens = job.split(",")
    require(len(job_tokens) == count and len(set(job_tokens)) == count, "wrong whole-job logical GPU count")
    step = environment.get("SLURM_STEP_GPUS", "")
    visible = environment.get("CUDA_VISIBLE_DEVICES")
    if role == "coordinator":
        require(step in {"", "(null)"} and visible in {None, "", "NoDevFiles"},
            "zero-GPU coordinator inherited a GPU selector")
    else:
        require(re.fullmatch(r"(?:0|[1-9][0-9]*)", step) and step in job_tokens
            and type(visible) is str and re.fullmatch(r"(?:0|[1-9][0-9]*|GPU-[0-9a-fA-F-]{36})", visible),
            "worker must retain one original step and CUDA selector")
    return {"job": job, "step": step, "cuda": visible, "physical_mapping_inferred": False}


def memory_mib(value):
    """Convert only explicit scheduler MiB or GiB values without rounding."""
    match = re.fullmatch(r"([1-9][0-9]*)([MG])", value)
    require(match is not None, "unsupported scheduler host-memory unit")
    return int(match[1]) * (1024 if match[2] == "G" else 1)


def resource_row(text, key, cpus, memory, gpus):
    """Validate GPU counts independently from the CPU and host-memory request."""
    helper = adjacent("device_memory_limit")
    values = single.resources(helper.scheduler_field(text, key))
    require(values.get("cpu") == str(cpus) and values.get("node") == "1"
        and memory_mib(values["mem"]) == memory, "scheduler CPU or host memory changed")
    require(values.get("gres/gpu", "0") == str(gpus)
        and all(name in {"gres/gpu", "gres/gpu:nvidia_rtx_a4500"} and amount == str(gpus)
            for name, amount in values.items() if name.startswith("gres/"))
        and (gpus != 0 or not any(name.startswith("gres/") for name in values)),
        "actual step/job GPU resources changed")


def validate_metadata(plan, value):
    """Rebuild exact group ownership and the independent zero/one-GPU numeric step."""
    options = validate_plan(plan)
    require(type(value.get("schema_version")) is int and value["schema_version"] == 1
        and value.get("kind") == "uva_owned_group_step_metadata"
        and value.get("capability") == CAPABILITY and "capture_error" not in value
        and exact(value.get("profile"), GROUP_PROFILE)
        and all(exact(value.get(key), expected) for key, expected in BOUNDARY.items()),
        "incomplete or unsupported group-step metadata")
    role = value.get("role")
    allowed_directories = {str(unit_directory(plan))} if role == "coordinator" else {
        str(worker_directory(plan, index)) for index in range(plan["gpu_count"])}
    require(value.get("directory") in allowed_directories, "metadata belongs to another group or worker directory")
    require(role in {"coordinator", "worker"} and value["source"] == source_identity(plan)
        and value["inputs"] == owned_inputs(plan)
        and value["inputs_receipt"] == receipt(options["owned_inputs_path"]), "group source or launch custody changed")
    process, environment = value["process"], value["environment"]
    require(exact(value.get("uids"), [2618919] * 4),
        "actual real, effective, saved and filesystem process UIDs must all be the campaign owner")
    job, step = environment.get("SLURM_JOB_ID", ""), environment.get("SLURM_STEP_ID", "")
    helper = adjacent("device_memory_limit")
    helper.v2_paths(job, step)
    shape = options["resources"][role]
    require(process["hostname"] == "jaguar03" and type(process["uid"]) is int
        and process["uid"] == 2618919 and job == value["inputs"]["job_id"]
        and environment.get("SLURM_CONF") == "/etc/slurm/slurm.conf"
        and environment.get("SLURM_CPUS_PER_TASK") == str(shape["cpus"])
        and len(process["cpu_affinity"]) == len(set(process["cpu_affinity"])) == shape["cpus"],
        "actual own group process, numeric step or CPU affinity changed")
    selectors(environment, plan["gpu_count"], role)
    helper.validate_v2_ownership(value["cgroup"], process, job, step)
    for key, command in (("job", ["scontrol", "show", "job", "-d", job]),
            ("step", ["scontrol", "show", "step", "-d", job + "." + step]),
            ("running_configuration", ["scontrol", "show", "config"])):
        single.validate_query(value[key], command, configuration=key == "running_configuration")
    field = helper.scheduler_field
    row, task, whole = value["job"]["stdout"], value["step"]["stdout"], options["resources"]["job"]
    require(field(row, "JobId") == job and field(row, "UserId") == "sl5nw(2618919)"
        and field(row, "JobState") == "RUNNING" and field(row, "Account") == "sl5nw_base"
        and field(row, "Partition") == "gpu" and field(row, "QOS") == plan["placement"]["qos"]
        and field(row, "Reservation") == (plan["placement"]["reservation"] or "(null)")
        and field(row, "NodeList") == "jaguar03" and field(row, "NumNodes") == "1"
        and field(row, "NumCPUs") == str(whole["allocated_cpus"])
        and field(row, "TresPerNode") == "gres/gpu:nvidia_rtx_a4500:" + str(plan["gpu_count"])
        and field(row, "Requeue") == "0" and field(row, "Restarts") == "0",
        "actual whole-job owner or typed resource allocation changed")
    resource_row(row, "ReqTRES", whole["requested_cpus"], whole["host_memory_mib"], plan["gpu_count"])
    resource_row(row, "AllocTRES", whole["allocated_cpus"], whole["host_memory_mib"], plan["gpu_count"])
    require(field(task, "StepId") == job + "." + step and field(task, "State") == "RUNNING"
        and field(task, "UserId") in {"2618919", "sl5nw(2618919)"}
        and field(task, "NodeList") == "jaguar03" and field(task, "Tasks") == "1",
        "actual one-task numeric step ownership changed")
    resource_row(task, "TRES", shape["cpus"], shape["host_memory_mib"], shape["gpus"])
    running = value["running_configuration"]["stdout"]
    for key, expected in (("SLURM_CONF", "/etc/slurm/slurm.conf"), ("SlurmUser", "slurm(64030)"),
            ("ProctrackType", "proctrack/cgroup"), ("GresTypes", "gpu")):
        require(re.findall(r"^\s*" + re.escape(key) + r"\s*=\s*(.*?)\s*$", running, re.M) == [expected],
            "running controller configuration changed")
    plugins = re.findall(r"^\s*TaskPlugin\s*=\s*(.*?)\s*$", running, re.M)
    require(len(plugins) == 1 and len(plugins[0].split(",")) == 2
        and set(plugins[0].split(",")) == {"task/cgroup", "task/affinity"}, "task plugins changed")
    single.validate_configuration(value["configuration"])
    return value


def collect_metadata(plan, directory, role):
    """Capture only this owned job and numeric step before any numerical import."""
    directory = Path(directory)
    result = {"schema_version": 1, "kind": "uva_owned_group_step_metadata",
        "capability": CAPABILITY, "profile": GROUP_PROFILE, "role": role, "directory": str(directory), **BOUNDARY}
    try:
        # Ownership precedes scheduler queries; failures retain their bounded partial record.
        require(not ({"jax", "jaxlib", "numpy", "torch"} & {name.split(".")[0] for name in sys.modules}),
            "group metadata must precede numerical imports")
        validate_plan(plan)
        result["source"] = source_identity(plan)
        result["inputs"] = owned_inputs(plan)
        result["inputs_receipt"] = receipt(plan["configured_group"]["owned_inputs_path"])
        helper = adjacent("device_memory_limit")
        result["process"] = helper.process_identity()
        uids = re.findall(r"^Uid:\s*([0-9]+)\s+([0-9]+)\s+([0-9]+)\s+([0-9]+)\s*$",
            Path("/proc/self/status").read_text(), re.M)
        require(len(uids) == 1 and [int(value) for value in uids[0]] == [2618919] * 4,
            "all process UIDs must prove campaign ownership before any scheduler query")
        result["uids"] = [int(value) for value in uids[0]]
        result["environment"] = {key: os.environ[key] for key in ENVIRONMENT if key in os.environ}
        job, step = result["environment"].get("SLURM_JOB_ID", ""), result["environment"].get("SLURM_STEP_ID", "")
        require(role in {"coordinator", "worker"} and job == result["inputs"]["job_id"]
            and result["process"]["hostname"] == "jaguar03" and result["process"]["uid"] == 2618919,
            "foreign job/owner/host refused before queries")
        helper.v2_paths(job, step)
        selectors(result["environment"], plan["gpu_count"], role)
        wrapped = {"ownership_profile": helper.UVA_CGROUP_V2_OWNERSHIP_PROFILE,
            "device_memory_limit_protocol": {"allocation_kind": "slurm", "cluster": "uva"}}
        result["cgroup"] = helper.capture_v2_ownership(wrapped, result["process"], job, step,
            directory / "group_process_cgroup.json")
        for key, command in (("job", ["scontrol", "show", "job", "-d", job]),
                ("step", ["scontrol", "show", "step", "-d", job + "." + step]),
                ("running_configuration", ["scontrol", "show", "config"])):
            result[key] = single.bounded_query(command, configuration=key == "running_configuration")
        account = pwd.getpwnam("slurm")
        result["configuration"] = {"service": {"name": account.pw_name, "uid": account.pw_uid, "gid": account.pw_gid},
            "cache_absent": not os.path.lexists("/var/spool/slurmd/conf-cache"), "files": {}}
        for path in single.CONFIG_FILES:
            result["configuration"]["files"][path] = {}
            single.config_file(path, result["configuration"]["files"][path])
        validate_metadata(plan, result)
    except BaseException as error:
        result["capture_error"] = {"type": type(error).__name__, "error": str(error)}
        raise
    finally:
        publish(directory / "group_step_metadata.json", result)
    return result


def current_process(plan, metadata):
    """Refuse a different current process or changed original step/CUDA selector."""
    helper = adjacent("device_memory_limit")
    require(exact(helper.process_identity(), metadata["process"])
        and all(os.environ.get(key) == value for key, value in metadata["environment"].items()),
        "group process birth, placement or original selectors changed")
    return metadata["process"]


def resolve_group(plan, coordinator, members):
    """Publish a physical group only after distinct same-process bindings all pass."""
    validate_metadata(plan, coordinator)
    require(coordinator["role"] == "coordinator" and len(members) == plan["gpu_count"],
        "a complete owned coordinator/member group is required")
    seen = {name: set() for name in ("gpu_uuid", "pci_address", "physical_minor", "step", "pid")}
    result = []
    for index, member in enumerate(members):
        directory = worker_directory(plan, index)
        metadata = read_receipt(member["metadata"])
        binding = read_receipt(member["binding"])
        require(member["metadata"] == receipt(directory / "group_step_metadata.json")
            and member["binding"] == receipt(directory / "resolved_device_binding.json")
            and metadata["role"] == "worker", "worker binding was substituted or reordered")
        validate_binding(plan, metadata, binding, directory)
        fields = {key: binding[key] for key in ("gpu_uuid", "pci_address", "physical_minor")}
        fields.update(step=metadata["environment"]["SLURM_STEP_ID"], pid=metadata["process"]["pid"])
        require(fields["step"] != coordinator["environment"]["SLURM_STEP_ID"]
            and fields["pid"] != coordinator["process"]["pid"], "coordinator was reused as a numerical worker")
        for name, value in fields.items():
            require(value not in seen[name], "duplicate physical device, process or numeric worker step")
            seen[name].add(value)
        result.append({"index": index, **member, **fields,
            "process_start_time": metadata["process"]["process_start_time"]})
    return {"schema_version": 1, "kind": "uva_resolved_distinct_a4500_group", "capability": CAPABILITY,
        "plan": receipt(unit_directory(plan) / "scaling_plan.json"),
        "coordinator": receipt(unit_directory(plan) / "group_step_metadata.json"),
        "members": result, "profile": GROUP_PROFILE, **BOUNDARY}


import csv
import stat


def bind_process(plan, metadata, jax_device, directory):
    """Preserve one attempted resolution, including failures, before learner work."""
    directory = Path(directory)
    require(not any(((directory / name).exists() for name in ('resolved_device_binding.json', 'resolved_device_binding_attempt.json'))), 'preserve the prior physical binding attempt; do not retry this immutable unit')
    attempt = {'schema_version': 1, 'kind': 'uva_group_current_process_device_binding_attempt', 'profile': PROFILE, 'capability': CAPABILITY, 'native': {}, **BOUNDARY}
    try:
        value = bind_process_attempt(plan, metadata, jax_device, directory, attempt)
        publish(directory / 'resolved_device_binding.json', value)
        attempt.update(passed=True, binding=receipt(directory / 'resolved_device_binding.json'))
        return value
    except BaseException as error:
        attempt.update(passed=False, error_type=type(error).__name__, error=str(error))
        raise
    finally:
        publish(directory / 'resolved_device_binding_attempt.json', attempt)


def bind_process_attempt(plan, metadata, jax_device, directory, attempt):
    """Resolve only after current metadata and the sole JAX device have passed."""
    directory = Path(directory)
    validate_metadata(plan, metadata)
    require(metadata['role'] == 'worker', 'physical binding is forbidden in the zero-GPU coordinator')
    helper = adjacent('device_memory_limit')
    before = helper.process_identity()
    require(before == metadata['process'] and all((os.environ.get(k) == v for (k, v) in metadata['environment'].items())), 'process or original selector changed before native CUDA resolution')
    require('jax' in sys.modules and sys.modules['jax'].local_devices(backend='gpu') == [jax_device], 'the actual process must expose exactly this one local JAX GPU')
    attempt.update(plan_sha256=metadata['inputs']['plan']['sha256'], metadata_sha256=digest(encoded(metadata)), process=before)
    native = single.native_identity(jax_device, attempt['native'])
    (gpu, bus) = (native['gpu_uuid'], native['pci_address'])
    command = ['nvidia-smi', '-i', gpu, '--query-gpu=uuid,name,pci.bus_id,memory.total,driver_version', '--format=csv,noheader,nounits']
    attempt['hardware'] = single.bounded_query(command)
    query = attempt['hardware']
    require(query['capture_status'] == 'completed' and type(query['returncode']) is int and (query['returncode'] == 0) and (query['stderr'] == ''), 'assigned-device hardware query failed')
    proc_path = Path('/proc/driver/nvidia/gpus') / bus / 'information'
    attempt['proc'] = {'path': str(proc_path), 'read_status': 'started'}
    with proc_path.open() as stream:
        proc_raw = stream.read(32769)
    attempt['proc'] = {'path': str(proc_path), 'raw': proc_raw, 'sha256': digest(proc_raw.encode())}
    require(len(proc_raw.encode()) <= 32768, 'assigned-device proc metadata exceeds its read limit')
    minors = re.findall('^Device Minor:\\s*([0-9]+)\\s*$', proc_raw, re.M)
    require(len(minors) == 1, 'current CUDA PCI address must identify one physical device minor')
    device_path = Path('/dev/nvidia' + minors[0])
    information = device_path.lstat()
    attempt['device'] = {'path': str(device_path), 'character_device': stat.S_ISCHR(information.st_mode), 'symlink': stat.S_ISLNK(information.st_mode), 'major': os.major(information.st_rdev), 'minor': os.minor(information.st_rdev)}
    attempt['driver_capture_expected_path'] = str(directory / 'captured_driver' / Path(native['driver_library']).name)
    capture = adjacent('h100_nvl_slurm_site_validation').capture_driver_image(native, directory)
    attempt['driver_capture'] = capture
    value = {'schema_version': 1, 'kind': 'uva_group_current_process_resolved_device_binding', 'profile': PROFILE, 'capability': CAPABILITY, 'plan_sha256': metadata['inputs']['plan']['sha256'], 'metadata_sha256': digest(encoded(metadata)), 'process': before, 'original_environment': metadata['environment'], 'native': native, 'gpu_uuid': gpu, 'pci_address': bus, 'physical_minor': int(minors[0]), 'driver_capture': capture, 'hardware': query, 'proc': attempt['proc'], 'device': attempt['device'], 'process_after': helper.process_identity(), 'environment_after': {k: os.environ.get(k) for k in metadata['environment']}, 'startup_memory_observed': False, 'other_gpu_processes_observed': False, **BOUNDARY}
    validate_binding(plan, metadata, value, directory)
    return value


def validate_binding(plan, metadata, value, directory=None):
    """Rebuild one binding from saved bytes; this grants no portable or formal admission."""
    validate_metadata(plan, metadata)
    require(metadata['role'] == 'worker', 'physical binding is forbidden in the zero-GPU coordinator')
    require(type(value.get('schema_version')) is int and value['schema_version'] == 1 and (value.get('kind') == 'uva_group_current_process_resolved_device_binding') and (value.get('capability') == CAPABILITY), 'unsupported resolved physical binding')
    single.require_profile(value['profile'])
    require(all((exact(value.get(k), v) for (k, v) in BOUNDARY.items())) and value.get('startup_memory_observed') is False and (value.get('other_gpu_processes_observed') is False), 'the binding cannot invent mapping, startup capture, process visibility or admission')
    require(value['plan_sha256'] == metadata['inputs']['plan']['sha256'] and value['metadata_sha256'] == digest(encoded(metadata)) and (value['process'] == value['process_after'] == metadata['process']) and (value['original_environment'] == value['environment_after'] == metadata['environment']), 'resolved process, immutable input or original selector changed')
    native = value['native']
    require(type(native['pid']) is int and native['pid'] == metadata['process']['pid'] and (type(native['cuda_device_count']) is int) and (native['cuda_device_count'] == 1) and (type(native['jax_local_hardware_id']) is int) and (native['jax_local_hardware_id'] == 0) and (type(native['cuda_device_handle']) is int) and (native['cuda_device_handle'] >= 0) and (type(native.get('current_context')) is int) and (native['current_context'] > 0) and (type(native.get('current_context_device')) is int) and (native['current_context_device'] == native['cuda_device_handle']) and (native.get('jax_identity_basis') == "sole local JAX GPU, sole CUDA-visible ordinal, and that device's current CUDA context") and (native['jax_device_kind'] == PROFILE['model']) and exact(native['api_return_codes'], {k: 0 for k in ('cuDeviceGetCount', 'cuDeviceGet', 'cuDeviceGetUuid_v2', 'cuDeviceGetPCIBusId', 'cuCtxGetCurrent', 'cuCtxGetDevice')}) and re.fullmatch('GPU-[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}', native['gpu_uuid']) and (native['gpu_uuid'] == value['gpu_uuid']) and (single.pci(native['pci_address']) == value['pci_address']), 'same-process native CUDA/JAX identity changed')
    require(native['driver_library'].endswith('.' + PROFILE['driver']) and digest(native['mapped_driver_bytes'].encode()) == native['mapped_driver_sha256'] and ({line.split()[-1] for line in native['mapped_driver_bytes'].splitlines()} == {native['driver_library']}), 'the actual mapped driver path or bytes changed')
    query = value['hardware']
    single.validate_query(query, ['nvidia-smi', '-i', value['gpu_uuid'], '--query-gpu=uuid,name,pci.bus_id,memory.total,driver_version', '--format=csv,noheader,nounits'])
    rows = list(csv.reader(query['stdout'].splitlines()))
    require(len(rows) == 1 and len(rows[0]) == 5, 'one current physical hardware row is required')
    (gpu, model, bus, memory, driver) = [x.strip() for x in rows[0]]
    require(gpu == value['gpu_uuid'] and model == PROFILE['model'] and (single.pci(bus) == value['pci_address']) and (memory == str(PROFILE['memory_mib'])) and (driver == PROFILE['driver']), 'actual UUID/model/PCI/memory/driver changed')
    proc = value['proc']
    require(proc['path'] == '/proc/driver/nvidia/gpus/' + value['pci_address'] + '/information' and digest(proc['raw'].encode()) == proc['sha256'], 'physical proc metadata changed')
    for (key, expected) in (('GPU UUID', gpu), ('Model', model), ('Bus Location', value['pci_address']), ('Device Minor', str(value['physical_minor']))):
        found = re.findall('^' + re.escape(key) + ':\\s*(.*?)\\s*$', proc['raw'], re.M)
        require(found == [expected], 'native CUDA UUID/PCI does not join exact physical proc/minor metadata')
    device = value['device']
    require(type(value['physical_minor']) is int and value['physical_minor'] >= 0 and exact(device, {'path': '/dev/nvidia' + str(value['physical_minor']), 'character_device': True, 'symlink': False, 'major': 195, 'minor': value['physical_minor']}), 'physical Linux device metadata changed')
    require(directory is not None and str(Path(directory)) == metadata['directory']
        and Path(directory) in {worker_directory(plan, index) for index in range(plan['gpu_count'])},
        'group binding requires its exact separately owned worker directory')
    directory = Path(directory)
    capture = value['driver_capture']
    target = directory / 'captured_driver' / Path(native['driver_library']).name
    require(type(capture.get('schema_version')) is int and capture['schema_version'] == 1 and (capture.get('kind') == 'captured_mapped_cuda_driver_image') and (type(capture.get('pid')) is int) and (capture['pid'] == native['pid']) and (capture['original'] == {'path': native['driver_library'], 'sha256': native['driver_library_sha256']}) and (capture['captured'] == {'path': str(target), 'sha256': native['driver_library_sha256']}) and (type(capture['bytes']) is int) and (capture['bytes'] > 0) and (receipt(target) == capture['captured']) and (target.stat().st_size == capture['bytes']), 'immutable captured mapped driver changed')
    return value
