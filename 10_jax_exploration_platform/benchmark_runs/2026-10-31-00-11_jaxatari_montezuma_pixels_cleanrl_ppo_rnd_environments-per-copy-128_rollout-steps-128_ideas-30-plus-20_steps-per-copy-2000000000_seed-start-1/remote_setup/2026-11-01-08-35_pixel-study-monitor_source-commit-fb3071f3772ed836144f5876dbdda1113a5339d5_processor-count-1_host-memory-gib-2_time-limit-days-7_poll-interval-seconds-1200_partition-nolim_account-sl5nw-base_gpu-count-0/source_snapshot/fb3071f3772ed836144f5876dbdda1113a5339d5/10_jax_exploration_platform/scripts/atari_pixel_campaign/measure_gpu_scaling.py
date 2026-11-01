"""Prepare and measure synchronized one- or eight-A4500 groups on jaguar03.

The standard-library parent proves the entire Slurm allocation and starts one
fresh numerical child per recorded UUID. GPU-complete messages close the common
wall-time window before host copies and memory finalization. This program never
submits a job, selects an optimization, or edits the frozen numerical source.
"""

import argparse
from contextlib import contextmanager
import csv
import fcntl
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import selectors
import shlex
import signal
import socket
import struct
import subprocess
import sys
import time
import traceback
import xml.etree.ElementTree as ET


SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))
import measure_selected as selected

training, diagnostic, validation = selected.training, selected.diagnostic, selected.validation
HELPERS = ("measure_gpu_scaling.py", *selected.HELPERS)
ALLOCATION_FIELDS = ("SLURM_JOB_GPUS", "SLURM_STEP_GPUS")


def process_birth(pid):
    """Read a process's kernel start counter without confusing a reused numeric PID."""
    return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[19]


def helper_receipt():
    """Bind the entire staged producer/helper set to committed source bytes."""
    root = SCRIPTS.parents[2]
    commit = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    return training.source_receipt(root, commit, [str((SCRIPTS / name).relative_to(root)) for name in HELPERS])


def prepare(arguments):
    """Stage complete one-group inputs and launch records without importing numerical libraries."""
    roster = arguments.gpu_uuid
    if (len(roster) not in {1, 8} or len(set(roster)) != len(roster)
            or any(not value.startswith("GPU-") for value in roster)):
        raise ValueError("supply an ordered roster of exactly one or eight distinct physical GPU UUIDs")
    if Path(arguments.unit_id).name != arguments.unit_id or arguments.unit_id in {"", ".", ".."}:
        raise ValueError("use a plain descriptive measurement unit identifier")
    if not math.isfinite(arguments.barrier_timeout_seconds) or arguments.barrier_timeout_seconds <= 0:
        raise ValueError("barriers require a finite positive timeout")
    root, run = arguments.source_root.resolve(strict=True), arguments.run_dir.resolve(strict=True)
    configuration, config_input = selected.read_document(arguments.configuration)
    configuration = validation.configuration(configuration)
    selected.validate_complete_fields(root, configuration)
    runtime, runtime_input = selected.read_document(arguments.runtime_environment)
    runtime = validation.runtime_environment(runtime)
    assets, asset_input = selected.read_document(arguments.asset_record)
    if set(assets) != {"root", "manifest", "sha256"}:
        raise ValueError("asset_record requires an immutable asset root, manifest and manifest hash")
    copies = configuration["learner"]["n_copies"]
    mode = getattr(arguments, "scaling_mode", "same_copies_per_card")
    total = getattr(arguments, "total_copies", None)
    if mode not in {"same_copies_per_card", "fixed_total_copies"}:
        raise ValueError("scaling mode must explicitly identify per-card or fixed-total policy work")
    if (mode == "fixed_total_copies" and (type(total) is not int or total < 8 or total % 8
            or total != copies * len(roster))) or mode == "same_copies_per_card" and total is not None:
        raise ValueError("fixed-total scaling requires a positive multiple of eight equal to copies times GPU count")
    completed_study, completed_mapping = selected.completed_study_inputs(arguments, arguments.source_commit,
        runtime, configuration, required=mode == "fixed_total_copies", hardware_class="jaguar03_rtx_a4500")
    if not 0 <= arguments.seed_start <= 2 ** 32 - copies * len(roster):
        raise ValueError("the independent per-card policy seed blocks exceed unsigned 32-bit values")
    window = selected.measurement_window([configuration], arguments.minimum_updates_per_round)
    core = diagnostic.core_identity(root, arguments.source_commit)
    paths = ["10_jax_exploration_platform/scripts/atari_pixel_campaign/" + name
             for name in ("paired_benchmark.py", "ideas.py")]
    paired = training.source_receipt(root, arguments.source_commit, paths)
    helpers = helper_receipt()
    plan = {"schema_version": 1, "kind": "synchronized_a4500_scaling_measurement",
        "source_root": str(root), "source_commit": arguments.source_commit, "core": core,
        "paired_source": paired, "helpers": helpers, "configuration": configuration,
        "runtime_environment": runtime, "asset_record": assets,
        "jaxtari_source_root": str(arguments.jaxtari_source_root.resolve()) if arguments.jaxtari_source_root else None,
        "hardware_class": "jaguar03_rtx_a4500", "gpu_count": len(roster), "gpu_uuids": roster,
        "copy_seeds": [list(range(arguments.seed_start + index * copies,
                                   arguments.seed_start + (index + 1) * copies)) for index in range(len(roster))],
        "rounds": 7, "iterations": window, "minimum_updates_per_round": arguments.minimum_updates_per_round,
        "window_protocol": "complete groups/report windows; no padded training updates",
        "barrier_timeout_seconds": arguments.barrier_timeout_seconds,
        "compilation_cache_policy": "disabled", "python": str(arguments.python.absolute()),
        "campaign_run_dir": str(run), "unit_id": arguments.unit_id,
        "inputs": {"configuration": config_input, "runtime_environment": runtime_input, "asset_record": asset_input},
        "jobs_submitted_by_producer": False, "selected_by_producer": False, "hypothesis_count_contribution": 0}
    if mode == "fixed_total_copies":
        plan["scaling_protocol"] = {"kind": mode, "total_copies": total,
            "global_copy_seeds": list(range(arguments.seed_start, arguments.seed_start + total)),
            "gradient_averaging": False, "measurement_scope": "synchronized_gpu_group_wall_time"}
        plan["completed_study"], plan["completed_study_manifest"] = completed_study, completed_mapping
    directory = run / "data" / arguments.unit_id
    command = [str(arguments.python), str(directory / "code/measure_gpu_scaling.py"), "--prepared-unit", str(directory)]
    background = ("## Purpose\n\nMeasure an explicitly selected full configuration on one or eight distinct RTX A4500 GPUs. "
        "The producer selects no optimizations or copy counts and submits no jobs.\n\n"
        "## Key hyperparameters\n\nEach policy has 128 pixel environments, 128 rollout steps, four epochs, four minibatches, "
        "and fifty normalization warmup rollouts. Neural/optimizer arithmetic is float32; statistics are float64. "
        f"Each of seven synchronized rounds contains {window} complete updates per card. "
        "Every card has distinct policy seeds and the same selected per-card configuration.\n\n"
        "## Code and config changes\n\nThe standard-library parent proves the full physical Slurm allocation. Fresh children "
        "compile before the shared measurement barrier and expose only their assigned physical UUID to CUDA. Group time "
        "starts at release and ends at the final synchronized GPU completion, before host copies and memory finalization. "
        "The aggregate rate is total transitions divided by this common wall time. No checkpoint is saved.\n\n"
        f"## Git state\n\nNumerical source: `{arguments.source_commit}`. Helper source: `{helpers['source_commit']}`. "
        "Exact source, helper, configuration, runtime, asset and physical-GPU identities are retained in scaling_plan.json.\n")
    artifacts = {directory / "scaling_plan.json": selected.json_bytes(plan),
        directory / "manifest.yaml": selected.json_bytes(plan),
        directory / "config_resolved.yaml": selected.json_bytes({"configuration": configuration, "runtime_environment": runtime}),
        directory / "command.txt": (shlex.join(command) + "\n").encode(),
        directory / "experiment_background.md": background.encode()}
    artifacts.update({directory / "code" / name: (SCRIPTS / name).read_bytes() for name in HELPERS})
    for path, contents in artifacts.items():
        if path.exists() and path.read_bytes() != contents:
            raise ValueError("a prepared scaling artifact is immutable: " + str(path))
    for path, contents in artifacts.items():
        selected.write_once(path, contents)
    selected.write_once(directory / "scaling_prepared.json", selected.json_bytes({
        "plan_sha256": hashlib.sha256(selected.json_bytes(plan)).hexdigest()}))
    return directory


def load_prepared(directory):
    """Require staged committed helpers, the exact Python executable and unchanged frozen source."""
    directory = Path(directory).resolve(strict=True)
    plan, ready = (json.loads((directory / name).read_text()) for name in ("scaling_plan.json", "scaling_prepared.json"))
    if ready.get("plan_sha256") != training.file_sha256(directory / "scaling_plan.json"):
        raise ValueError("the prepared scaling plan changed")
    if Path(__file__).resolve() != directory / "code/measure_gpu_scaling.py":
        raise ValueError("execute the immutable staged scaling producer recorded in command.txt")
    if str(Path(sys.executable).absolute()) != plan["python"]:
        raise ValueError("execute the exact recorded Python environment")
    if directory != Path(plan["campaign_run_dir"]) / "data" / plan["unit_id"]:
        raise ValueError("the scaling output directory differs from its immutable plan")
    for name in HELPERS:
        relative = "10_jax_exploration_platform/scripts/atari_pixel_campaign/" + name
        if training.file_sha256(directory / "code" / name) != plan["helpers"]["files"][relative]:
            raise ValueError("a staged scaling helper changed: " + name)
    if Path(selected.__file__).resolve() != directory / "code/measure_selected.py":
        raise ValueError("the scaling process must import only its staged producer helpers")
    selected.verify_prepared(directory, plan)
    return directory, plan


def allocation_proof(plan, environment, hostname):
    """Prove every visible GPU against Slurm's physical allocation without NVML index assumptions."""
    if hostname.split(".")[0] != "jaguar03":
        raise ValueError("synchronized A4500 measurements require jaguar03")
    job = environment.get("SLURM_JOB_ID", "")
    if not job.isdecimal() or int(job) < 1:
        raise ValueError("synchronized measurements require an actual Slurm compute allocation")
    allocations = {name: environment[name] for name in ALLOCATION_FIELDS if environment.get(name)}
    if not allocations:
        raise ValueError("the full physical Slurm GPU allocation must be recorded")
    result = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid,name", "--format=csv,noheader"],
                            capture_output=True, text=True, check=True, timeout=15)
    csv_text = result.stdout
    rows = list(csv.reader(csv_text.splitlines()))
    result = subprocess.run(["nvidia-smi", "-q", "-x"], capture_output=True, text=True, check=True, timeout=15)
    xml_text = result.stdout
    devices = ET.fromstring(xml_text).findall("gpu")
    count = plan["gpu_count"]
    if len(rows) != count or len(devices) != count:
        raise ValueError("the visible physical GPU count differs from the complete requested allocation")
    xml = {device.findtext("uuid"): device for device in devices}
    cards = []
    for row in rows:
        if len(row) != 3:
            raise ValueError("a visible GPU CSV row is malformed")
        index, gpu, name = [value.strip() for value in row]
        device = xml.get(gpu)
        minor = device.findtext("minor_number", "") if device is not None else ""
        if (not index.isdecimal() or not minor.isdecimal() or not gpu.startswith("GPU-")
                or "A4500" not in name or device.findtext("product_name") != name):
            raise ValueError("visible CSV and physical-minor XML do not identify the same RTX A4500")
        cards.append({"visible_index": index, "gpu_uuid": gpu, "physical_minor": minor, "name": name})
    uuids = {card["gpu_uuid"] for card in cards}
    minors = {card["physical_minor"]: card["gpu_uuid"] for card in cards}
    indices = {card["visible_index"] for card in cards}
    if (len(uuids) != count or len(minors) != count or len(indices) != count
            or set(plan["gpu_uuids"]) != uuids):
        raise ValueError("the visible physical UUID/minor roster differs from the immutable plan")
    for value in allocations.values():
        tokens = value.split(",")
        resolved = [minors.get(str(int(token))) if token.isdecimal() else token for token in tokens]
        if len(tokens) != count or len(set(resolved)) != count or set(resolved) != uuids:
            raise ValueError("every Slurm job/step allocation must identify the full physical UUID roster")
    visible = environment.get("CUDA_VISIBLE_DEVICES", "").split(",")
    if len(visible) != count or set(visible) not in (uuids, set(minors), indices):
        raise ValueError("CUDA visibility conflicts with the full physical Slurm allocation")
    return {"hostname": "jaguar03", "slurm_job_id": job, "slurm_allocations": allocations,
            "cuda_visible_devices_original": environment["CUDA_VISIBLE_DEVICES"], "cards": cards,
            "gpu_csv": csv_text, "gpu_xml": xml_text}


def verify_assignment(directory, plan, index):
    """Preserve the full allocation fields and check the exact parent and selected card before JAX."""
    assignment = json.loads((directory / "assignment.json").read_text())
    if (assignment["plan_sha256"] != training.file_sha256(directory / "scaling_plan.json")
            or assignment["gpu_uuids"] != plan["gpu_uuids"] or socket.gethostname().split(".")[0] != "jaguar03"
            or assignment["hostname"] != "jaguar03"):
        raise ValueError("the numerical child differs from its original parent assignment")
    if process_birth(assignment["parent_pid"]) != assignment["parent_process_start_time"]:
        raise ValueError("the recorded scaling parent no longer owns this process group")
    proof = assignment["allocation"]
    if os.environ.get("SLURM_JOB_ID") != proof["slurm_job_id"]:
        raise ValueError("the child Slurm job differs from its proved allocation")
    actual = {name: os.environ[name] for name in ALLOCATION_FIELDS if os.environ.get(name)}
    if actual != proof["slurm_allocations"]:
        raise ValueError("a child must retain the original full Slurm allocation fields")
    if type(index) is not int or not 0 <= index < plan["gpu_count"]:
        raise ValueError("the child assignment index is invalid")
    gpu = plan["gpu_uuids"][index]
    if os.environ.get("CUDA_VISIBLE_DEVICES") != gpu or gpu not in {card["gpu_uuid"] for card in proof["cards"]}:
        raise ValueError("the child's exact CUDA UUID is outside the proved parent roster")
    # This query uses an already proved UUID; a Slurm physical number is never an NVML index.
    rows = selected.selected_query(gpu, "uuid,name,memory.total,driver_version,memory.used")
    if len(rows) != 1 or len(rows[0]) != 5:
        raise ValueError("the child's selected physical device query is ambiguous")
    actual_gpu, name, total, driver, used = [value.strip() for value in rows[0]]
    if actual_gpu != gpu or "A4500" not in name or int(used) >= 1024 or selected.selected_query(gpu, "pid", processes=True):
        raise ValueError("the assigned physical RTX A4500 is occupied or differs from its receipt")
    return {"hostname": "jaguar03", "gpu_uuid": gpu, "gpu_query": f"{gpu}, {name}, {total} MiB, {driver}",
            "pid": os.getpid(), "process_start_time": process_birth(os.getpid()),
            "slurm_job_id": proof["slurm_job_id"], "assignment_sha256": training.file_sha256(directory / "assignment.json")}


def send_message(channel, value):
    """Send one bounded JSON frame without partial-line ambiguity."""
    contents = json.dumps(value, allow_nan=False, separators=(",", ":")).encode()
    if len(contents) > 64 * 2 ** 20:
        raise ValueError("a scaling control message exceeded its explicit maximum size")
    channel.sendall(struct.pack("!I", len(contents)) + contents)


def receive_message(channel, deadline):
    """Read one complete JSON frame under a common monotonic deadline."""
    def receive(size):
        """Handle fragmented local sockets without extending the barrier deadline."""
        contents = bytearray()
        while len(contents) < size:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("a scaling communication barrier exceeded its timeout")
            channel.settimeout(remaining)
            part = channel.recv(size - len(contents))
            if not part:
                raise RuntimeError("a scaling child or parent closed its control channel")
            contents.extend(part)
        return bytes(contents)
    size = struct.unpack("!I", receive(4))[0]
    if size > 64 * 2 ** 20:
        raise ValueError("a scaling control message exceeded its explicit maximum size")
    return json.loads(receive(size))


def collect_messages(children, kind, timeout, round_index=None):
    """Wait for all owned children, detecting child exits and bounding the whole barrier."""
    deadline, results = time.monotonic() + timeout, {}
    with selectors.DefaultSelector() as selector:
        for child in children:
            selector.register(child["channel"], selectors.EVENT_READ, child)
        while len(results) < len(children):
            for child in children:
                if child["index"] not in results and child["process"].poll() is not None:
                    # A queued final frame is still valid after a clean process exit.
                    if child["process"].returncode != 0:
                        raise RuntimeError("a scaling child failed: " + str(child["index"]))
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("the common scaling barrier exceeded its timeout")
            for key, unused in selector.select(min(remaining, .25)):
                child = key.data
                value = receive_message(child["channel"], deadline)
                if value.get("type") == "failure":
                    raise RuntimeError("a scaling child reported failure: " + value.get("error", "unknown"))
                if value.get("type") != kind or (round_index is not None and value.get("round") != round_index):
                    raise ValueError("a scaling child sent a message outside the current barrier")
                results[child["index"]] = value
                selector.unregister(child["channel"])
    return [results[child["index"]] for child in children]


def clock_now(assignment):
    """Project the shared host monotonic clock onto the parent's one recorded epoch origin."""
    return assignment["clock_epoch_origin"] + time.monotonic()


def aggregate_round(start, end, workers, plan, round_index):
    """Validate raw worker work/bounds/memory and divide total work by the common wall time."""
    if not math.isfinite(start) or not math.isfinite(end) or end <= start or len(workers) != plan["gpu_count"]:
        raise ValueError("the synchronized group lacks a finite complete wall-time window")
    for index, worker in enumerate(workers):
        if (worker.get("gpu_uuid") != plan["gpu_uuids"][index] or worker.get("copy_seeds") != plan["copy_seeds"][index]
                or worker.get("iterations") != plan["iterations"]):
            raise ValueError("a raw worker differs from its fixed UUID, seed block or complete update window")
        validation.measured_round(worker, plan["configuration"]["learner"]["n_copies"])
        first, last = worker.get("start_time_seconds"), worker.get("end_time_seconds")
        if (type(first) not in (int, float) or type(last) not in (int, float) or not math.isfinite(first)
                or not math.isfinite(last) or first < start or last > end or last <= first
                or abs(last - first - worker["seconds"]) > max(1e-6, worker["seconds"] * 1e-6)
                or worker["memory"]["gpu_uuid"] != worker["gpu_uuid"]):
            raise ValueError("a worker's real timing bounds are inconsistent with the common wall window")
        samples = worker["memory"].get("device_memory_samples", [])
        if len(samples) < 2:
            raise ValueError("a scaling worker lacks actual device-memory observations")
        for sample in samples:
            total, used, free = (sample.get(name) for name in ("total_bytes", "used_bytes", "free_bytes"))
            if (sample.get("gpu_uuid") != worker["gpu_uuid"] or any(type(value) is not int for value in (total, used, free))
                    or total <= 0 or min(used, free) < 0 or used + free > total or free / total < .15):
                raise ValueError("a scaling worker lacks fifteen percent actual device free memory")
    if max(row["start_time_seconds"] for row in workers) >= min(row["end_time_seconds"] for row in workers):
        raise ValueError("the selected physical GPUs did not train concurrently")
    work = sum(row["training_transitions"] for row in workers)
    return {"round": round_index, "measurement_scope": "synchronized_gpu_group_wall_time",
            "start_time_seconds": start, "end_time_seconds": end, "seconds": end - start,
            "training_transitions": work, "transitions_per_second_total": work / (end - start), "workers": workers}


def stop_owned_children(children):
    """Stop only child processes whose recorded PID, birth counter and parent relationship still match."""
    for child in children:
        process = child["process"]
        if process.poll() is None:
            try:
                text = Path(f"/proc/{process.pid}/stat").read_text().rsplit(")", 1)[1].split()
                if text[19] == child["process_start_time"] and int(text[1]) == os.getpid():
                    process.terminate()
            except FileNotFoundError:
                pass
    deadline = time.monotonic() + 10
    for child in children:
        process = child["process"]
        try:
            process.wait(timeout=max(.01, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            try:
                text = Path(f"/proc/{process.pid}/stat").read_text().rsplit(")", 1)[1].split()
                if text[19] == child["process_start_time"] and int(text[1]) == os.getpid():
                    process.kill()
                    process.wait(timeout=10)
            except FileNotFoundError:
                pass


@contextmanager
def interruption_cleanup():
    """Route termination and keyboard interruption through the verified-child cleanup path."""
    def interrupted(number, frame):
        """Raise inside the active measurement so its existing exception handler stops owned children."""
        raise InterruptedError("synchronized GPU measurement received signal " + str(number))
    previous = {number: signal.getsignal(number) for number in (signal.SIGTERM, signal.SIGINT)}
    for number in previous:
        signal.signal(number, interrupted)
    try:
        yield
    finally:
        for number, handler in previous.items():
            signal.signal(number, handler)


def run_parent(directory):
    """Start recorded child processes inside the caller's allocation and measure seven common windows."""
    directory, plan = load_prepared(directory)
    with (directory / "scaling.lock").open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        if any((directory / name).exists() for name in ("result.json", "failure.json", "assignment.json")):
            raise ValueError("retain the existing scaling execution and prepare a fresh unit")
        if "jax" in sys.modules:
            raise RuntimeError("the scaling parent must not import numerical libraries")
        runtime = training.enforce_runtime(plan["runtime_environment"], os.environ)
        for name, expected in (("XLA_PYTHON_CLIENT_PREALLOCATE", "false"), ("JAXTARI_SKIP_SPRITE_UPDATE", "1")):
            if os.environ.get(name, expected) != expected:
                raise ValueError("scaling runtime differs from the frozen worker setting: " + name)
            os.environ[name] = expected
        proof = allocation_proof(plan, os.environ, socket.gethostname())
        assignment = {"schema_version": 1, "plan_sha256": training.file_sha256(directory / "scaling_plan.json"),
            "hostname": "jaguar03", "parent_pid": os.getpid(), "parent_process_start_time": process_birth(os.getpid()),
            "gpu_uuids": plan["gpu_uuids"], "allocation": proof, "runtime_environment": runtime,
            "clock_epoch_origin": time.time() - time.monotonic()}
        selected.write_once(directory / "assignment.json", selected.json_bytes(assignment))
        children, started = [], clock_now(assignment)
        with interruption_cleanup():
            _measure_group(directory, plan, proof, assignment, children, started)


def _measure_group(directory, plan, proof, assignment, children, started):
    """Keep the owned-child exception cleanup inside the temporary interruption handlers."""
    try:
        for index, gpu in enumerate(plan["gpu_uuids"]):
            parent_channel, child_channel = socket.socketpair()
            worker_dir = directory / "workers" / ("physical_gpu_" + str(index + 1))
            worker_dir.mkdir(parents=True)
            log = (worker_dir / "process.log").open("xb")
            environment = {**os.environ, "CUDA_VISIBLE_DEVICES": gpu}
            command = [plan["python"], str(directory / "code/measure_gpu_scaling.py"), "--worker-unit", str(directory),
                       "--worker-index", str(index), "--channel-fd", str(child_channel.fileno())]
            process = subprocess.Popen(command, env=environment, stdin=subprocess.DEVNULL,
                stdout=log, stderr=subprocess.STDOUT, pass_fds=(child_channel.fileno(),), start_new_session=False)
            child_channel.close()
            log.close()
            child = {"index": index, "gpu_uuid": gpu, "process": process, "channel": parent_channel,
                     "process_start_time": process_birth(process.pid), "command": command}
            children.append(child)
            diagnostic.atomic_json(directory / "children.json", [{"index": row["index"], "gpu_uuid": row["gpu_uuid"],
                "pid": row["process"].pid, "process_start_time": row["process_start_time"], "command": row["command"]} for row in children])
            # This durable PID record precedes the child's numerical release.
            send_message(parent_channel, {"type": "initialize", "assignment_sha256": training.file_sha256(directory / "assignment.json")})
        diagnostic.record_progress(directory, "normalization_and_compilation_warmup", workers=len(children))
        ready = collect_messages(children, "compiled", plan["barrier_timeout_seconds"])
        origins = {row["training_start_iteration"] for row in ready}
        if len(origins) != 1:
            raise ValueError("the GPU workers have different training-schedule origins after compilation")
        diagnostic.atomic_json(directory / "compiled_workers.json", ready)
        rounds = []
        for round_index in range(1, 8):
            collect_messages(children, "round_ready", plan["barrier_timeout_seconds"], round_index)
            start = clock_now(assignment)
            for child in children:
                send_message(child["channel"], {"type": "release", "round": round_index})
            completed = collect_messages(children, "gpu_complete", plan["barrier_timeout_seconds"], round_index)
            end = clock_now(assignment)
            for child in children:
                send_message(child["channel"], {"type": "collect", "round": round_index})
            rows = collect_messages(children, "round_record", plan["barrier_timeout_seconds"], round_index)
            workers = [row["record"] for row in rows]
            if any(row["timing"] != {name: worker[name] for name in ("start_time_seconds", "end_time_seconds", "seconds")}
                   for row, worker in zip(completed, workers)):
                raise ValueError("a worker changed its GPU-complete timing after the common measurement window")
            rounds.append(aggregate_round(start, end, workers, plan, round_index))
            diagnostic.atomic_json(directory / "rounds.json", rounds)
            diagnostic.record_progress(directory, "running", variant="synchronized GPU group", round=round_index, total_rounds=7,
                gpu_count=plan["gpu_count"], transitions_per_second_total=rounds[-1]["transitions_per_second_total"])
        finished = collect_messages(children, "finished", plan["barrier_timeout_seconds"])
        for child in children:
            if child["process"].wait(timeout=10) != 0:
                raise RuntimeError("a measured child failed after its final round")
        result = {"completed": True, "source_commit": plan["source_commit"], "source_root": plan["source_root"],
            "hardware_class": "jaguar03_rtx_a4500", "configuration": plan["configuration"],
            "runtime_environment": plan["runtime_environment"], "gpu_count": plan["gpu_count"],
            "gpu_uuids": plan["gpu_uuids"], "copy_seeds": plan["copy_seeds"], "rounds": rounds,
            "helpers": plan["helpers"], "normalization_warmup_iterations": 50,
            "training_start_iteration": next(iter(origins)), "workers": finished,
            "assignment_sha256": training.file_sha256(directory / "assignment.json"),
            "protocol": {"rounds": 7, "requested_minimum_updates_per_round": plan["minimum_updates_per_round"],
                         "actual_updates_per_round": plan["iterations"], "group_clock": "one shared host monotonic clock"},
            "execution_identity": {"hostname": "jaguar03", "pid": os.getpid(),
                "process_start_time": assignment["parent_process_start_time"],
                "started_at_seconds": started, "finished_at_seconds": clock_now(assignment)}}
        if "scaling_protocol" in plan:
            result["scaling_protocol"] = plan["scaling_protocol"]
        diagnostic.atomic_json(directory / "result.json", result)
        diagnostic.record_progress(directory, "completed", completed=True, rounds=7, gpu_count=plan["gpu_count"])
    except BaseException as error:
        diagnostic.atomic_json(directory / "failure.json", {"error": repr(error), "traceback": traceback.format_exc(),
            "source_commit": plan["source_commit"], "children": [{"pid": row["process"].pid,
                "process_start_time": row["process_start_time"], "gpu_uuid": row["gpu_uuid"]} for row in children]})
        diagnostic.record_progress(directory, "failed", error=repr(error))
        stop_owned_children(children)
        raise
    finally:
        for child in children:
            child["channel"].close()


def worker_backend_receipt(plan, jaxtari, libraries):
    """Add complete installed versions for the new fixed-total comparison while preserving old records."""
    if plan.get("scaling_protocol", {}).get("kind") == "fixed_total_copies":
        return selected.installed_backend_receipt(jaxtari, libraries)
    return {"jaxtari_source": jaxtari, "gpu_libraries": libraries}


def run_worker(directory, index, channel_fd):
    """Admit one proved assignment before importing the unchanged numerical implementation."""
    directory, plan = load_prepared(directory)
    channel = socket.socket(fileno=channel_fd)
    worker_dir = directory / "workers" / ("physical_gpu_" + str(index + 1))
    timeout = plan["barrier_timeout_seconds"]
    try:
        release = receive_message(channel, time.monotonic() + timeout)
        if release != {"type": "initialize", "assignment_sha256": training.file_sha256(directory / "assignment.json")}:
            raise ValueError("the child initialization release differs from its durable parent assignment")
        if "jax" in sys.modules:
            raise RuntimeError("a scaling child imported JAX before allocation admission")
        hardware = verify_assignment(directory, plan, index)
        runtime = training.enforce_runtime(plan["runtime_environment"], os.environ)
        assets = training.verify_assets(plan["asset_record"], os.environ)
        jaxtari = training.jaxtari_python_receipt(plan.get("jaxtari_source_root"))
        libraries = training.installed_gpu_library_receipts()
        diagnostic.atomic_json(worker_dir / "hardware.json", {**hardware, "runtime_environment": runtime})
        diagnostic.atomic_json(worker_dir / "asset_receipt.json", assets)
        diagnostic.atomic_json(worker_dir / "installed_backend.json", worker_backend_receipt(plan, jaxtari, libraries))
        compute_worker(directory, worker_dir, plan, index, channel, hardware, libraries)
    except BaseException as error:
        diagnostic.atomic_json(worker_dir / "failure.json", {"error": repr(error), "traceback": traceback.format_exc()})
        try:
            send_message(channel, {"type": "failure", "error": repr(error)})
        except OSError:
            pass
        raise
    finally:
        channel.close()


def compute_worker(directory, worker_dir, plan, index, channel, hardware, libraries):
    """Compile once, then execute exactly the released complete-update windows on one GPU."""
    root = Path(plan["source_root"]) / "10_jax_exploration_platform"
    sys.path.insert(0, str(root / "src"))
    sys.path.insert(0, str(root / "scripts/atari_pixel_campaign"))
    import jax
    import jaxtari
    from exploration_platform.atari_pixels.config import PixelPPOConfig
    from exploration_platform.atari_pixels.environment import AtariPixelEnvironment
    from exploration_platform.atari_pixels.training import PixelPPORND
    paired = importlib.import_module("paired_benchmark")
    selected.verify_imported_sources(root, paired)
    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_default_matmul_precision", "highest")
    devices = jax.devices()
    if len(devices) != 1 or devices[0].platform != "gpu" or "A4500" not in devices[0].device_kind:
        raise ValueError("a numerical scaling child must expose exactly its assigned RTX A4500")
    config = plan["configuration"]
    trainer = PixelPPORND(PixelPPOConfig(**config["learner"]), AtariPixelEnvironment(**config["environment"]))
    executor = paired.Executor(trainer, config["execution"])
    diagnostic.record_progress(worker_dir, "normalization_warmup")
    state = trainer.warmup(trainer.initialize(tuple(plan["copy_seeds"][index])))
    host = paired.host_copy(state)
    del state
    diagnostic.record_progress(worker_dir, "compile_warmup")
    host, warmup = paired.warm_executor(executor, host, plan["iterations"])
    loaded = training.actual_gpu_library_receipt(devices[0].client, libraries)
    diagnostic.atomic_json(worker_dir / "loaded_gpu_libraries.json", loaded)
    diagnostic.atomic_json(worker_dir / "warmup.json", warmup)
    assignment = json.loads((directory / "assignment.json").read_text())
    send_message(channel, {"type": "compiled", "gpu_uuid": hardware["gpu_uuid"],
        "pid": os.getpid(), "process_start_time": hardware["process_start_time"],
        "training_start_iteration": int(host.iteration), "warmup": warmup,
        "loaded_gpu_libraries_sha256": training.file_sha256(worker_dir / "loaded_gpu_libraries.json")})
    rounds, memory = [], selected.MemoryObservation(devices[0], hardware["gpu_uuid"])
    for round_index in range(1, 8):
        origin = int(host.iteration)
        state = paired.device_copy(host)
        jax.block_until_ready(state)
        memory.begin()
        try:
            send_message(channel, {"type": "round_ready", "round": round_index, "iteration": origin})
            if receive_message(channel, time.monotonic() + plan["barrier_timeout_seconds"]) != {"type": "release", "round": round_index}:
                raise ValueError("a numerical child received another round's release")
            start = clock_now(assignment)
            state, metrics = executor.run(state, plan["iterations"], initial_iteration=origin)
            jax.block_until_ready((state, metrics))
            end = clock_now(assignment)
            timing = {"start_time_seconds": start, "end_time_seconds": end, "seconds": end - start}
            send_message(channel, {"type": "gpu_complete", "round": round_index, "timing": timing})
            if receive_message(channel, time.monotonic() + plan["barrier_timeout_seconds"]) != {"type": "collect", "round": round_index}:
                raise ValueError("a numerical child received another round's collection request")
        finally:
            memory_record = memory.finish()
        host = paired.host_copy(state)
        paired.require_finite((host, jax.device_get(metrics)), "synchronized scaling round")
        transitions = plan["iterations"] * trainer.config.batch_size * trainer.config.n_copies
        row = {**timing, "gpu_uuid": hardware["gpu_uuid"], "copy_seeds": plan["copy_seeds"][index],
            "copies": trainer.config.n_copies, "iterations": plan["iterations"], "environments_per_copy": 128,
            "rollout_steps": 128, "training_transitions": transitions, "report_origin_iteration": origin,
            "transitions_per_second_total": transitions / timing["seconds"], "memory": memory_record,
            "published_reports": jax.tree.map(lambda value: value.tolist() if hasattr(value, "tolist") else value,
                                                executor.last_reports),
            "pid": os.getpid(), "process_start_time": hardware["process_start_time"]}
        rounds.append(row)
        diagnostic.atomic_json(worker_dir / "rounds.json", rounds)
        send_message(channel, {"type": "round_record", "round": round_index, "record": row})
        del state, metrics
    result = {"completed": True, "gpu_uuid": hardware["gpu_uuid"], "copy_seeds": plan["copy_seeds"][index],
        "source_commit": plan["source_commit"], "configuration": config, "runtime_environment": plan["runtime_environment"],
        "hardware": hardware, "rounds": rounds, "loaded_gpu_libraries": loaded}
    diagnostic.atomic_json(worker_dir / "result.json", result)
    send_message(channel, {"type": "finished", "gpu_uuid": hardware["gpu_uuid"],
                          "worker_result_sha256": training.file_sha256(worker_dir / "result.json")})


def parse_arguments(argv=None):
    """Separate metadata preparation, allocation parent execution and private numerical children."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-unit", type=Path)
    parser.add_argument("--worker-unit", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--worker-index", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--channel-fd", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--source-commit")
    parser.add_argument("--configuration", type=Path)
    parser.add_argument("--runtime-environment", type=Path)
    parser.add_argument("--asset-record", type=Path)
    parser.add_argument("--jaxtari-source-root", type=Path)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--unit-id")
    parser.add_argument("--gpu-uuid", action="append", default=[])
    parser.add_argument("--seed-start", type=int, default=1)
    parser.add_argument("--scaling-mode", choices=("same_copies_per_card", "fixed_total_copies"), default="same_copies_per_card")
    parser.add_argument("--total-copies", type=int)
    parser.add_argument("--completed-study-proof", type=Path)
    parser.add_argument("--completed-study-review", type=Path)
    parser.add_argument("--completed-study-manifest", type=Path)
    parser.add_argument("--completed-study-manifest-sha256")
    parser.add_argument("--minimum-updates-per-round", type=int, default=10)
    parser.add_argument("--barrier-timeout-seconds", type=float, default=3600)
    parser.add_argument("--python", type=Path, default=Path("/p/rlprojects/RND/.venvs/atari_pixels_jax/bin/python"))
    args = parser.parse_args(argv)
    if args.prepared_unit is not None or args.worker_unit is not None:
        if args.prepared_unit is not None and args.worker_unit is not None:
            parser.error("choose the parent or its private child entrypoint")
        if args.prepare_only or args.gpu_uuid or args.total_copies is not None or args.scaling_mode != "same_copies_per_card" or any(getattr(args, name) is not None for name in (
                "source_root", "source_commit", "configuration", "runtime_environment", "asset_record",
                "jaxtari_source_root", "run_dir", "unit_id", "completed_study_proof", "completed_study_review",
                "completed_study_manifest", "completed_study_manifest_sha256")):
            parser.error("prepared execution cannot take preparation overrides")
        if args.worker_unit is not None and (args.worker_index is None or args.channel_fd is None):
            parser.error("a private numerical child needs its parent channel and assignment index")
        if args.prepared_unit is not None and (args.worker_index is not None or args.channel_fd is not None):
            parser.error("the parent cannot take a child assignment/channel")
        return args
    if any(getattr(args, name) is None for name in ("source_root", "source_commit", "configuration",
            "runtime_environment", "asset_record", "run_dir", "unit_id")):
        parser.error("preparation requires explicit source/configuration/runtime/assets/output")
    return args


if __name__ == "__main__":
    arguments = parse_arguments()
    if arguments.prepared_unit is not None:
        run_parent(arguments.prepared_unit)
    elif arguments.worker_unit is not None:
        run_worker(arguments.worker_unit, arguments.worker_index, arguments.channel_fd)
    else:
        unit = prepare(arguments)
        print(json.dumps({"prepared": True, "unit_directory": str(unit)}, indent=2), flush=True)
        if not arguments.prepare_only:
            os.execv(str(arguments.python), [str(arguments.python), str(unit / "code/measure_gpu_scaling.py"), "--prepared-unit", str(unit)])
