"""Prepare and execute explicitly reviewed one- or eight-A4500 group operations.

Preparation does not submit jobs. The default command does not execute a plan.
The batch controller and zero-GPU coordinator use system Python; each numerical
worker uses the exact selected training Python and retains its Slurm selectors.
"""
import argparse
import copy
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import signal
import stat
import subprocess
import sys
import time
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parent))
import uva_group_binding as group
import configured_gpu_scaling as producer
import configured_scaling_evidence as aggregation
import group_backend_package as package
import group_runtime_evidence as evidence

OWNER = 2618919
METADATA_PYTHON = "/usr/bin/python3"
MODES = {"qualification", "correctness", "measurement", "exact_eight_eligibility"}
RUNTIME_ADMISSION_KIND = "independent_configured_group_runtime_launch_review"


def helper_names():
    """Keep every shared helper required by either group metadata, package or numerical execution."""
    return tuple(dict.fromkeys((*producer.helper_names(), *package.helper_names(), *group.NEW_HELPERS)))


def validate_plan(plan):
    """Validate a new complete plan without choosing a future implementation or packing count."""
    group.validate_plan(plan)
    configuration = producer.selected.validation.configuration(plan["configuration"])
    reference = producer.selected.validation.configuration(plan["reference_configuration"])
    producer.selected.validation.runtime_environment(plan["runtime_environment"])
    group.require(plan["mode"] in MODES and plan.get("metadata_python") == METADATA_PYTHON
        and type(plan.get("python")) is str and Path(plan["python"]).is_absolute()
        and Path(plan["python"]).parent.name == "bin", "exact metadata and numerical Python programs are required")
    group.require(configuration["learner"]["n_copies"] == reference["learner"]["n_copies"]
        and all(configuration["learner"][key] == expected for key, expected in {
            "n_envs": 128, "num_steps": 128, "update_epochs": 4, "num_minibatches": 4,
            "observation_warmup_iterations": 50, "total_timesteps": 2000000000}.items()),
        "group work must retain the full original environment, rollout, optimizer and annealing configuration")
    copies = configuration["learner"]["n_copies"]
    seeds = plan.get("copy_seeds")
    group.require(type(copies) is int and copies > 0 and isinstance(seeds, list) and len(seeds) == plan["gpu_count"]
        and all(isinstance(row, list) and len(row) == copies and all(type(seed) is int and 0 <= seed < 2 ** 32 for seed in row) for row in seeds)
        and len({seed for row in seeds for seed in row}) == copies * plan["gpu_count"],
        "each numerical copy needs its own explicit stable global seed")
    group.require(type(plan.get("barrier_timeout_seconds")) in {int, float}
        and math.isfinite(plan["barrier_timeout_seconds"]) and plan["barrier_timeout_seconds"] > 0
        and plan.get("compilation_cache_policy") == "disabled", "explicit bounded barriers and disabled compilation cache are required")
    placement = plan.get("placement")
    group.require(type(placement) is dict and set(placement) == {"node", "partition", "account", "reservation", "qos"}
        and placement["node"] == "jaguar03" and placement["partition"] == "gpu" and placement["account"] == "sl5nw_base"
        and ((placement["reservation"] is None and placement["qos"] == "normal")
            or type(placement["reservation"]) is str and re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.-]*", placement["reservation"])
            and placement["qos"] == "csresnolim"), "reservation and quality of service must be explicitly paired")
    group.require(type(plan.get("wall_seconds")) is int and plan["wall_seconds"] > 0
        and "job_id" in plan and plan["job_id"] is None and plan.get("gpu_uuids") == [None] * plan["gpu_count"],
        "frozen preparation cannot guess a future scheduler identifier or physical GPU")
    if plan["mode"] in {"measurement", "exact_eight_eligibility"}:
        aggregation.validate_protocol(plan)
    elif plan["mode"] == "qualification":
        group.adjacent("group_original_qualification").validate_configuration(plan)
    else:
        group.require(configuration != reference, "correctness requires genuinely distinct implementations")
        producer.selected.validate_correctness_protocol([reference, configuration])
    return configuration


def literal_command(parts):
    """Quote a literal argument vector for the generated batch script."""
    return " ".join(shlex.quote(value) for value in parts)


def numeric_command(plan, role, index=None):
    """Keep the numerical interpreter separate from the standard-library coordinator."""
    group.require(role in {"coordinator", "worker"}, "numeric role is not a coordinator or worker")
    unit = group.unit_directory(plan)
    python = plan["metadata_python"] if role == "coordinator" else plan["python"]
    flags = ["-I", "-S", "-B"] if role == "coordinator" else ["-I", "-B"]
    command = [python, *flags, str(unit / "code/group_runtime.py"), "--execute-owned-plan", str(unit), "--role", role]
    if role == "worker":
        group.require(type(index) is int and 0 <= index < plan["gpu_count"], "an exact worker index is required")
        command += ["--worker-index", str(index)]
    else:
        group.require(index is None, "the coordinator has no numerical worker index")
    return command


def held_submission_command(plan, script):
    """Return one held whole-job resource request without submitting or choosing devices."""
    validate_plan(plan)
    shape, place = plan["configured_group"]["resources"]["job"], plan["placement"]
    command = ["sbatch", "--hold", "--parsable", "--no-requeue", "--partition=" + place["partition"],
        "--account=" + place["account"], "--qos=" + place["qos"], "--nodelist=jaguar03", "--nodes=1", "--ntasks=1",
        "--cpus-per-task=" + str(shape["requested_cpus"]), "--mem=" + str(shape["host_memory_mib"]) + "M",
        "--gres=gpu:nvidia_rtx_a4500:" + str(plan["gpu_count"]), "--time=" +
        str(plan["wall_seconds"] // 3600) + ":" + str(plan["wall_seconds"] // 60 % 60).zfill(2) + ":" + str(plan["wall_seconds"] % 60).zfill(2)]
    if place["reservation"] is not None:
        command.append("--reservation=" + place["reservation"])
    return [*command, str(script)]


def prepare(plan, backend_image_record):
    """Create an immutable staged group and standard records without actual job or GPU actions."""
    plan = copy.deepcopy(plan)
    validate_plan(plan)
    unit, run = group.unit_directory(plan), Path(plan["campaign_run_dir"])
    packet = run / "remote_setup" / plan["unit_id"]
    group.require(not os.path.lexists(unit) and not os.path.lexists(packet), "preserve every prior attempted group unit")
    expected = {group.PREFIX + name for name in helper_names()}
    group.require(set(plan["helpers"]["files"]) == expected, "the exact complete current group helper union is required")
    source_root, commit = Path(plan["helpers"]["source_root"]), plan["helpers"]["source_commit"]
    group.require(re.fullmatch(r"[0-9a-f]{40}", commit) and subprocess.check_output(
        ["git", "-C", str(source_root), "rev-parse", "HEAD"], text=True).strip() == commit,
        "stage a clean published helper commit")
    blobs = {}
    for relative, digest in plan["helpers"]["files"].items():
        raw = subprocess.check_output(["git", "-C", str(source_root), "show", commit + ":" + relative])
        group.require(hashlib.sha256(raw).hexdigest() == digest == group.receipt(source_root / relative)["sha256"],
            "helper source differs from its exact committed byte sequence")
        blobs[Path(relative).name] = raw
    unit.mkdir(parents=True, mode=0o700)
    packet.mkdir(parents=True, mode=0o700)
    package.configure_plan(plan, backend_image_record)
    group.publish(unit / "scaling_plan.json", plan)
    (unit / "code").mkdir(mode=0o700)
    for name, raw in blobs.items():
        with (unit / "code" / name).open("xb") as stream:
            stream.write(raw); stream.flush(); os.fsync(stream.fileno())
    group.publish(unit / "scaling_prepared.json", {"plan_sha256": group.receipt(unit / "scaling_plan.json")["sha256"]})
    job_command = [plan["metadata_python"], "-I", "-S", "-B", str(unit / "code/group_runtime.py"),
        "--execute-owned-plan", str(unit), "--role", "job-controller"]
    script = "#!/bin/bash\n#SBATCH --output=" + str(run / "slurm/jobs/group-operation-%j.out") + "\n"
    script += "#SBATCH --error=" + str(run / "slurm/jobs/group-operation-%j.err") + "\n"
    script += "set -euo pipefail\numask 077\nunset PYTHONPATH PYTHONHOME\n"
    script += "export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1\n"
    script += "export XLA_PYTHON_CLIENT_PREALLOCATE=false JAXTARI_SKIP_SPRITE_UPDATE=1\n"
    for key, value in plan["runtime_environment"].items():
        group.require(re.fullmatch(r"[A-Z_][A-Z0-9_]*", key) and type(value) is str, "runtime variables must be literal strings")
        script += "export " + key + "=" + shlex.quote(value) + "\n"
    script += "cd " + shlex.quote(str(run)) + "\nexec " + literal_command(job_command) + "\n"
    with (packet / "job.sbatch").open("x") as stream:
        stream.write(script); stream.flush(); os.fsync(stream.fileno())
    held = held_submission_command(plan, packet / "job.sbatch")
    group.publish(packet / "held_submission_command.json", held)
    background = ("# Explicit configured A4500 group operation\n\nPurpose: " + plan["mode"] +
        ". Each independent copy retains128environments,128rollout steps,4epochs,4minibatches and50normalization warmup rollouts. "
        "The batch controller and numeric coordinator use no GPUs; each numerical worker receives one original Slurm GPU selector. "
        "Physical UUID/PCI/minor are unresolved until same-process binding. No scientific or admission credit is implied.\n\n"
        "Git state: helper" + commit + "; numerical" + plan["source_commit"] + ".\n")
    command_text = literal_command(held) + "\n"
    for destination in (unit, packet):
        records = {"experiment_background.md": background.encode(), "manifest.yaml": group.encoded({
            "kind": plan["kind"], "mode": plan["mode"], "source_commit": plan["source_commit"],
            "helper_commit": commit, "gpu_count": plan["gpu_count"], "copy_seeds": plan["copy_seeds"]}),
            "config_resolved.yaml": group.encoded(plan), "command.txt": command_text.encode()}
        for name, raw in records.items():
            with (destination / name).open("xb") as stream:
                stream.write(raw); stream.flush(); os.fsync(stream.fileno())
        group.publish(destination / "progress.json", {"state": "prepared_not_submitted", "job_id": None,
            "scientific_credit": 0, "worker_admitted": False})
    return {"plan": group.receipt(unit / "scaling_plan.json"), "held_command": held, "submitted": False}


def read_plan(directory):
    """Execute only the exact staged source and frozen plan for this one immutable group."""
    directory = Path(directory).resolve(strict=True)
    plan = group.read_receipt(group.receipt(directory / "scaling_plan.json"))
    validate_plan(plan)
    group.require(directory == group.unit_directory(plan)
        and Path(__file__).resolve(strict=True) == directory / "code/group_runtime.py"
        and group.read_receipt(group.receipt(directory / "scaling_prepared.json")) == {
            "plan_sha256": group.receipt(directory / "scaling_plan.json")["sha256"]}, "execute the exact staged group runtime")
    group.source_identity(plan)
    package.expected_for_plan(plan)
    group.owned_inputs(plan)
    return directory, plan


def require_launch_review(plan):
    """Keep every operational entry disabled without a separate exact source and phase review."""
    value = plan.get("runtime_launch_review")
    group.require(type(value) is dict, "this new runtime has no reviewed operational launch admission")
    review = group.read_receipt(value)
    group.require(type(review.get("schema_version")) is int and review["schema_version"] == 2
        and review.get("kind") == RUNTIME_ADMISSION_KIND and review.get("approved") is True
        and review.get("helpers") == plan["helpers"] and review.get("configuration") == plan["configuration"]
        and review.get("source_commit") == plan["source_commit"] and review.get("mode") == plan["mode"]
        and review.get("gpu_count") == plan["gpu_count"] and type(review.get("gpu_count")) is int
        and review.get("admission_granted") is False and type(review.get("scientific_credit")) is int
        and review["scientific_credit"] == 0, "the exact group source, mode, configuration and resource shape were not reviewed")
    return review


def process_identity(pid=None):
    """Read only a current own process and retain PID/parent/birth, all four UIDs, argv and cgroup."""
    pid = os.getpid() if pid is None else pid
    group.require(type(pid) is int and pid > 0, "a positive exact process identifier is required")
    directory = Path("/proc") / str(pid)
    info = directory.stat()
    group.require(info.st_uid == OWNER, "refuse process metadata belonging to another user")
    before = (directory / "stat").read_text()
    fields = before[before.rfind(")") + 2:].split()
    status = (directory / "status").read_text()
    rows = re.findall(r"^Uid:\s*([0-9]+)\s+([0-9]+)\s+([0-9]+)\s+([0-9]+)\s*$", status, re.M)
    group.require(len(rows) == 1 and [int(x) for x in rows[0]] == [OWNER] * 4, "all real/effective/saved/filesystem UIDs must be this owner")
    argv = (directory / "cmdline").read_bytes().rstrip(b"\0").split(b"\0")
    cgroup = (directory / "cgroup").read_text()
    after = (directory / "stat").read_text()
    group.require(after[after.rfind(")") + 2:].split()[19] == fields[19]
        and directory.stat().st_ino == info.st_ino, "process identity changed during its bounded own reads")
    return {"pid": pid, "parent_pid": int(fields[1]), "process_start_time": fields[19],
        "uids": [int(x) for x in rows[0]], "argv": [x.decode("utf-8", errors="strict") for x in argv], "cgroup": cgroup}


def validate_process(value, metadata, command):
    """Keep the actual numerical process separate from its Slurm wrapper's identity."""
    saved = metadata["process"]
    group.require(type(value.get("pid")) is int and value["pid"] == saved["pid"]
        and type(value.get("parent_pid")) is int and value["parent_pid"] == saved["parent_pid"]
        and type(value.get("process_start_time")) is str and value["process_start_time"] == saved["process_start_time"]
        and group.exact(value.get("uids"), [OWNER] * 4) and value.get("cgroup") == saved["cgroup"]
        and value.get("argv") == command, "numerical argv, all-UID ownership, parent, birth or cgroup changed")
    return value


def wait_record(path, timeout):
    """Wait for one immutable phase record under a bounded monotonic deadline."""
    deadline = time.monotonic() + timeout
    while not path.exists():
        if time.monotonic() >= deadline:
            raise TimeoutError("the owned group did not publish " + path.name)
        time.sleep(min(.02, max(0, deadline - time.monotonic())))
    return group.read_receipt(group.receipt(path))


def record_progress(directory, state, **values):
    """Replace only the owned mutable progress record while preserving immutable operation artifacts."""
    value = {"state": state, "time_seconds": time.time(), **values}
    target = Path(directory) / "progress.json"
    temporary = target.with_name("progress." + str(os.getpid()) + ".prepared")
    with temporary.open("xb") as stream:
        stream.write(group.encoded(value)); stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary, target)


def metadata_step(plan, directory, role, index=None):
    """Capture the actual numeric step before any numerical import and bind its own argv separately."""
    before = process_identity()
    metadata = group.collect_metadata(plan, directory, role)
    identity = process_identity()
    group.require(identity == before, "own process or all-UID identity changed during numeric-step capture")
    validate_process(identity, metadata, numeric_command(plan, role, index))
    group.publish(directory / "group_process_identity.json", identity)
    return metadata


def publish_to_workers(plan, attempt, coordinator, phase, number, payload):
    """Send one exact immutable phase message to every stable worker index."""
    for index in range(plan["gpu_count"]):
        producer.publish_message(plan, attempt, index, "from_coordinator", phase, number, payload, coordinator["process"])


def run_coordinator(directory, plan):
    """Resolve distinct members and control their measured common wall without a numerical import."""
    require_launch_review(plan)
    coordinator = metadata_step(plan, directory, "coordinator")
    attempt = producer.begin_group(plan, coordinator)
    group.publish(directory / "group_workers_prepared.json", {"attempt": group.receipt(directory / "group_control_attempt.json"),
        "worker_count": plan["gpu_count"]})
    members = producer.collect_phase(plan, attempt, "bound", 0, plan["barrier_timeout_seconds"])
    resolved = group.resolve_group(plan, coordinator, members)
    group.publish(directory / "resolved_group_binding.json", resolved)
    evidence.verify_prerequisites(plan, resolved)
    if plan["mode"] in {"measurement", "exact_eight_eligibility"}:
        aggregation.verify_completed_phase(plan, producer.group_clock(attempt), resolved)
    publish_to_workers(plan, attempt, coordinator, "work_permitted", 0, {"mode": plan["mode"]})
    rounds = producer.collect_group_windows(plan, coordinator, resolved, attempt) if plan["mode"] in {
        "measurement", "exact_eight_eligibility"} else []
    finished = producer.collect_phase(plan, attempt, "finished", 0, plan["barrier_timeout_seconds"])
    group.require(all(row["completed"] is True for row in finished), "an actual numerical worker did not complete")
    group.publish(directory / "coordinator_complete.json", {"completed": True, "mode": plan["mode"],
        "rounds": rounds, "workers": [row["result"] for row in finished],
        "coordinator": group.receipt(directory / "group_step_metadata.json"),
        "execution_identity": {"hostname": coordinator["process"]["hostname"], "pid": coordinator["process"]["pid"],
            "process_start_time": coordinator["process"]["process_start_time"],
            "started_at_seconds": attempt["started_at_seconds"], "finished_at_seconds": time.time()}})


def numerical_modules(plan):
    """Import the exact selected original learner only after owned numeric metadata passed."""
    group.require(not ({"jax", "jaxlib", "numpy", "torch"} & {name.split(".")[0] for name in sys.modules}),
        "group numerical modules were imported before owned worker metadata")
    producer.selected.training.enforce_runtime(plan["runtime_environment"], os.environ)
    root = Path(plan["source_root"]) / "10_jax_exploration_platform"
    sys.path.insert(0, str(root / "src"))
    sys.path.insert(0, str(root / "scripts/atari_pixel_campaign"))
    import jax
    from exploration_platform.atari_pixels.config import PixelPPOConfig
    from exploration_platform.atari_pixels.environment import AtariPixelEnvironment
    from exploration_platform.atari_pixels.training import PixelPPORND
    paired = importlib.import_module("paired_benchmark")
    producer.selected.verify_imported_sources(root, paired)
    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_default_matmul_precision", "highest")
    devices = jax.local_devices(backend="gpu")
    group.require(len(devices) == 1 and "A4500" in devices[0].device_kind, "the numerical step needs exactly one actual A4500")
    return jax, PixelPPOConfig, PixelPPORND, AtariPixelEnvironment, paired, devices[0]


def run_correctness(worker, plan, index, metadata, binding, device, paired, Config, Trainer, Environment):
    """Preserve full independent initializations and unchanged fixed-input/full-iteration comparisons."""
    configuration, reference, seeds = plan["configuration"], plan["reference_configuration"], tuple(plan["copy_seeds"][index])
    memory = producer.GroupMemoryObservation(plan, metadata, binding, device, worker / "group_memory")
    memory.begin()
    try:
        started = time.time()
        trainer = Trainer(Config(**configuration["learner"]), Environment(**configuration["environment"]))
        executor = paired.Executor(trainer, configuration["execution"])
        state = trainer.warmup(trainer.initialize(seeds))
        reference_trainer = Trainer(Config(**reference["learner"]), Environment(**reference["environment"]))
        reference_executor = paired.Executor(reference_trainer, reference["execution"])
        reference_state = reference_trainer.warmup(reference_trainer.initialize(seeds))
        initialization = paired.agreement(paired.host_copy(paired.canonical_training_state(reference_executor, reference_state)),
            paired.host_copy(paired.canonical_training_state(executor, state)))
        checks, unused = paired.correctness_checks(reference_executor, executor, paired.host_copy(reference_state))
        finished = time.time()
    finally:
        observed = memory.finish()
    return {"correctness_attempted": True, "correctness_passed": True, "implementation_distinct": True,
        "measurement_valid": False, "correctness": {"initialization": initialization, **checks},
        "correctness_memory": observed, "correctness_interval": {"started_at_seconds": started, "finished_at_seconds": finished}}


def run_worker(directory, plan, index):
    """Execute one actual member's original gates, correctness or complete measured windows."""
    require_launch_review(plan)
    worker = group.worker_directory(plan, index)
    attempt = wait_record(directory / "group_control_attempt.json", plan["barrier_timeout_seconds"])
    metadata = metadata_step(plan, worker, "worker", index)
    installed = producer.selected.installed_backend_receipt(
        producer.selected.training.jaxtari_python_receipt(plan.get("jaxtari_source_root")),
        producer.selected.training.installed_gpu_library_receipts())
    assets = producer.selected.training.verify_assets(plan["asset_record"], os.environ)
    group.publish(worker / "installed_backend.json", installed)
    group.publish(worker / "asset_receipt.json", assets)
    started = time.time()
    jax, Config, Trainer, Environment, paired, device = numerical_modules(plan)
    binding = group.bind_process(plan, metadata, device, worker)
    package.capture_begin(worker, plan, metadata, binding)
    producer.publish_message(plan, attempt, index, "from_worker", "bound", 0,
        {"metadata": group.receipt(worker / "group_step_metadata.json"), "binding": group.receipt(worker / "resolved_device_binding.json")}, metadata["process"])
    group.require(producer.read_message(plan, attempt, index, "from_coordinator", "work_permitted", 0,
        plan["barrier_timeout_seconds"]) == {"mode": plan["mode"]}, "the coordinator released a different worker operation")
    resolved = group.read_receipt(group.receipt(directory / "resolved_group_binding.json"))
    evidence.verify_prerequisites(plan, resolved)
    configuration, seeds = plan["configuration"], tuple(plan["copy_seeds"][index])
    record_progress(worker, "running", mode=plan["mode"], gpu_uuid=binding["gpu_uuid"])
    if plan["mode"] == "qualification":
        original = group.adjacent("group_original_qualification")
        memory = producer.GroupMemoryObservation(plan, metadata, binding, device, worker / "group_memory")
        memory.begin()
        try:
            qualification_started = time.time()
            result = original.run(worker, {**plan, "copy_seeds": list(seeds)}, paired, jax, Config, Trainer, Environment)
            qualification_finished = time.time()
        finally:
            observed = memory.finish()
        result["qualification_memory"] = observed
        result["qualification_interval"] = {"started_at_seconds": qualification_started,
            "finished_at_seconds": qualification_finished}
    elif plan["mode"] == "correctness":
        result = run_correctness(worker, plan, index, metadata, binding, device, paired, Config, Trainer, Environment)
    else:
        trainer = Trainer(Config(**configuration["learner"]), Environment(**configuration["environment"]))
        executor = paired.Executor(trainer, configuration["execution"])
        state = trainer.warmup(trainer.initialize(seeds))
        initial = paired.host_copy(state)
        host, warmup = paired.warm_executor(executor, initial, plan["iterations"])
        evidence.validate_compilation_warmup(warmup, plan["iterations"], int(host.iteration))
        training_origin = int(host.iteration)
        host, rows = producer.measure_worker_windows(plan, index, metadata, binding, resolved, attempt,
            executor, host, paired, jax, device, trainer)
        result = {"warmup": warmup, "training_start_iteration": training_origin,
            "normalization_warmup_iterations": 50, "rounds": rows}
    result.update(schema_version=2, kind=package.MODES[plan["mode"]], completed=True,
        configuration=configuration, reference_configuration=plan["reference_configuration"], copy_seeds=list(seeds),
        source_commit=plan["source_commit"], helpers=plan["helpers"], runtime_environment=plan["runtime_environment"],
        asset_record=plan["asset_record"], worker_index=index, group_metadata=group.receipt(worker / "group_step_metadata.json"),
        resolved_device_binding=binding, installed_backend=installed,
        loaded_gpu_libraries=producer.selected.training.actual_gpu_library_receipt(device.client, installed["gpu_libraries"]),
        execution_identity={"hostname": metadata["process"]["hostname"], "pid": os.getpid(),
            "process_start_time": metadata["process"]["process_start_time"], "started_at_seconds": started, "finished_at_seconds": time.time()},
        scientific_credit=0, admission_granted=False)
    if plan["mode"] == "qualification":
        result["original_fixture_work"] = result.pop("executed_work")
        result["executed_work"] = {"normalization_warmup_transitions": 819200 * len(seeds),
            "training_transitions": 16384 * len(seeds), "complete_updates": 1, "minibatch_updates": 16}
    group.publish(worker / "loaded_gpu_libraries.json", result["loaded_gpu_libraries"])
    package.capture_complete(worker, plan, result)
    group.publish(worker / "result.json", result)
    producer.publish_message(plan, attempt, index, "from_worker", "finished", 0,
        {"completed": True, "result": group.receipt(worker / "result.json")}, metadata["process"])
    record_progress(worker, "completed", mode=plan["mode"], gpu_uuid=binding["gpu_uuid"])


def process_control_apis():
    """Refuse unavailable descriptor-control APIs before creating any Slurm wrapper."""
    group.require(callable(getattr(os, "pidfd_open", None)) and callable(getattr(signal, "pidfd_send_signal", None)),
        "the group controller's Python lacks required pidfd process-control APIs")


def start_wrapper(plan, role, index=None, *, registry):
    """Record a real own srun wrapper separately from the numerical process launched by Slurm."""
    process_control_apis()
    job = group.owned_inputs(plan)["job_id"]
    command = producer.step_command(plan, job, role, numeric_command(plan, role, index))
    directory = group.unit_directory(plan) / "wrapper_records"
    name = "coordinator" if role == "coordinator" else "worker_" + str(index + 1)
    group.publish(directory / (name + "_intent.json"), {"role": role, "index": index, "command": command,
        "controller": process_identity(), "at_seconds": time.time(), "automatic_retry_permitted": False})
    stdout, stderr = (directory / (name + "_stdout.raw")).open("xb"), (directory / (name + "_stderr.raw")).open("xb")
    process = subprocess.Popen(command, stdout=stdout, stderr=stderr, close_fds=True)
    wrapper = {"name": name, "process": process, "descriptor": None, "identity": None,
        "stdout": stdout, "stderr": stderr, "waited": False}
    registry.append(wrapper)
    descriptor = os.pidfd_open(process.pid, 0)
    wrapper["descriptor"] = descriptor
    identity = process_identity(process.pid)
    wrapper["identity"] = identity
    group.require(identity["parent_pid"] == os.getpid() and identity["argv"] == command,
        "srun wrapper is not the exact newly created own child")
    group.publish(directory / (name + "_started.json"), {"role": role, "index": index,
        "command": command, "process": identity, "started_at_seconds": time.time(),
        "is_numerical_process": False})
    return wrapper


def finish_wrapper(plan, wrapper, *, terminate=False):
    """Wait for exactly the pinned own wrapper; never signal an unverified numerical PID or group."""
    process, descriptor = wrapper["process"], wrapper["descriptor"]
    group.require(wrapper["waited"] is False, "preserve the previous exact wrapper wait")
    group.require(type(descriptor) is int and descriptor >= 0,
        "wrapper descriptor was never obtained; cleanup is unverified and no PID-signal fallback is allowed")
    if terminate and process.poll() is None:
        signal.pidfd_send_signal(descriptor, signal.SIGTERM, None, 0)
    returned = process.wait(timeout=plan["barrier_timeout_seconds"])
    for name in ("stdout", "stderr"):
        wrapper[name].flush(); os.fsync(wrapper[name].fileno()); wrapper[name].close()
    os.close(descriptor)
    wrapper["waited"] = True
    group.publish(group.unit_directory(plan) / "wrapper_records" / (wrapper["name"] + "_wait.json"),
        {"process": wrapper["identity"], "returncode": returned, "waited_at_seconds": time.time(),
            "terminated_for_failure": terminate, "is_numerical_wait": False})
    group.require(type(returned) is int and (terminate or returned == 0), "an actual owned srun wrapper did not finish normally")
    return returned


def verify_wrapper_waits(plan):
    """Rebuild every distinct wrapper's command, direct parent, all-UID identity and normal exact wait."""
    unit = group.unit_directory(plan)
    controller = group.read_receipt(group.receipt(unit / "group_job_controller.json"))
    identities = set()
    for role, index in [("coordinator", None), *[("worker", x) for x in range(plan["gpu_count"])]]:
        name = "coordinator" if role == "coordinator" else "worker_" + str(index + 1)
        started = group.read_receipt(group.receipt(unit / "wrapper_records" / (name + "_started.json")))
        wait = group.read_receipt(group.receipt(unit / "wrapper_records" / (name + "_wait.json")))
        process = started["process"]
        group.require(started["role"] == role and group.exact(started["index"], index)
            and started["command"] == producer.step_command(plan, group.owned_inputs(plan)["job_id"], role, numeric_command(plan, role, index))
            and process["argv"] == started["command"] and type(process["pid"]) is int and process["pid"] > 0
            and type(process["parent_pid"]) is int and type(controller["process"]["pid"]) is int
            and process["parent_pid"] == controller["process"]["pid"] and group.exact(process["uids"], [OWNER] * 4)
            and type(process["process_start_time"]) is str and re.fullmatch(r"[1-9][0-9]*", process["process_start_time"])
            and wait["process"] == process and type(wait["returncode"]) is int and wait["returncode"] == 0
            and wait["terminated_for_failure"] is False and started["is_numerical_process"] is False
            and wait["is_numerical_wait"] is False, "Slurm-wrapper custody or exact wait changed")
        group.require(process["pid"] not in identities, "one wrapper process was reused for different roles")
        identities.add(process["pid"])
    return controller


def run_job(directory, plan):
    """Launch only this already owned job's zero-GPU coordinator and one-GPU numerical steps."""
    require_launch_review(plan)
    process_control_apis()
    group.require(Path(sys.executable).absolute() == Path(plan["metadata_python"])
        and os.environ.get("SLURM_JOB_ID") == group.owned_inputs(plan)["job_id"]
        and not ({"jax", "jaxlib", "numpy", "torch"} & {name.split(".")[0] for name in sys.modules}),
        "the batch controller requires this exact owned allocation and standard-library Python")
    group.publish(directory / "group_job_controller.json", {"process": process_identity(),
        "job_id": group.owned_inputs(plan)["job_id"], "helpers": plan["helpers"], "plan": group.receipt(directory / "scaling_plan.json"),
        "started_at_seconds": time.time(), "gpu_operations": 0})
    (directory / "wrapper_records").mkdir(mode=0o700)
    wrappers = []
    try:
        start_wrapper(plan, "coordinator", registry=wrappers)
        prepared = wait_record(directory / "group_workers_prepared.json", plan["barrier_timeout_seconds"])
        group.require(prepared == {"attempt": group.receipt(directory / "group_control_attempt.json"),
            "worker_count": plan["gpu_count"]}, "workers cannot start before complete coordinator directory preparation")
        for index in range(plan["gpu_count"]):
            start_wrapper(plan, "worker", index, registry=wrappers)
        complete = wait_record(directory / "coordinator_complete.json", plan["wall_seconds"])
        for wrapper in wrappers:
            finish_wrapper(plan, wrapper)
        verify_wrapper_waits(plan)
        if plan["mode"] in {"measurement", "exact_eight_eligibility"}:
            member_results = [group.read_receipt(ref) for ref in complete["workers"]]
            origins = {row["training_start_iteration"] for row in member_results}
            group.require(len(origins) == 1, "workers completed different compilation training origins")
            group.publish(directory / "result.json", {"schema_version": 2,
                "kind": aggregation.ELIGIBILITY_KIND if plan["mode"] == "exact_eight_eligibility" else aggregation.RESULT_KIND,
                "completed": True, "configuration": plan["configuration"], "source_commit": plan["source_commit"],
                "helpers": plan["helpers"], "copy_seeds": plan["copy_seeds"], "runtime_environment": plan["runtime_environment"],
                "resolved_group_binding": group.receipt(directory / "resolved_group_binding.json"),
                "normalization_warmup_iterations": 50, "training_start_iteration": next(iter(origins)),
                "rounds": complete["rounds"], "worker_results": complete["workers"],
                "execution_identity": complete["execution_identity"], "scientific_credit": 0, "admission_granted": False})
        else:
            group.publish(directory / "component_complete.json", {"schema_version": 2, "mode": plan["mode"],
                "completed": True, "worker_results": complete["workers"], "scientific_credit": 0, "admission_granted": False})
        record_progress(directory, "completed", mode=plan["mode"])
    except BaseException as original:
        cleanup_errors = []
        for wrapper in wrappers:
            if not wrapper["waited"]:
                try:
                    finish_wrapper(plan, wrapper, terminate=wrapper["process"].poll() is None)
                except BaseException as error:
                    cleanup_errors.append({"wrapper": wrapper["name"], "error_type": type(error).__name__, "error": str(error)})
        group.publish(directory / "group_cleanup_after_failure.json", {"original_error": str(original),
            "all_wrappers_exactly_waited": all(item["waited"] for item in wrappers), "cleanup_errors": cleanup_errors,
            "no_pid_or_process_group_signal_fallback": True, "scientific_credit": 0})
        raise


def main(argv=None):
    """Keep defaults inactive and require the exact reviewed owned staged plan for execution."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute-owned-plan")
    parser.add_argument("--role", choices=("job-controller", "coordinator", "worker"))
    parser.add_argument("--worker-index", type=int)
    arguments = parser.parse_args(argv)
    group.require(arguments.execute_owned_plan and arguments.role,
        "group APIs remain disabled without an explicit reviewed and owned staged operation")
    directory, plan = read_plan(arguments.execute_owned_plan)
    target = group.worker_directory(plan, arguments.worker_index) if arguments.role == "worker" else directory
    try:
        if arguments.role == "job-controller":
            run_job(directory, plan)
        elif arguments.role == "coordinator":
            run_coordinator(directory, plan)
        else:
            group.require(type(arguments.worker_index) is int, "the worker requires its exact stable index")
            run_worker(directory, plan, arguments.worker_index)
    except BaseException as error:
        group.publish(target / "failure.json", {"exception_type": type(error).__name__, "error": str(error),
            "traceback": traceback.format_exc(), "role": arguments.role, "scientific_credit": 0, "admission_granted": False})
        record_progress(target, "failed", error=str(error))
        raise


if __name__ == "__main__":
    main()
