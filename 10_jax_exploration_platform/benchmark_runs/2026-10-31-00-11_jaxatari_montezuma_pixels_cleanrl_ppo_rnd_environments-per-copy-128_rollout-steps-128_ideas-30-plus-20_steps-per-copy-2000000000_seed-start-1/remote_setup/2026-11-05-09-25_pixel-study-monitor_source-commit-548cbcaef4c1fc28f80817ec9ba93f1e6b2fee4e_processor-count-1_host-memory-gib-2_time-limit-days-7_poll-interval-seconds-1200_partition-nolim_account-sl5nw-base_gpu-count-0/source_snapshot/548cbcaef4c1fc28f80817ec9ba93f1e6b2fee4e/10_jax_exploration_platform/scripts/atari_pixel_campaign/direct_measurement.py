"""Explicit direct card-one producer support; no default or UVA admission follows."""

from contextlib import contextmanager
import csv
import importlib.util
import json
import os
from pathlib import Path
import re
import shlex
import signal
import sqlite3
import subprocess
import sys
import time


CAPABILITY = "direct_assigned_uuid_own_process_complete_window_v1"
QUALIFICATION = {"schema_version": 1, "kind": "full_original_direct_serval02_card1_qualification",
    "reference_updates": 1, "normalization_warmup_rollouts": 50,
    "scientific_credit": 0, "case_credit": 0, "idea_credit": 0, "admission_granted": False}
BOUNDARY = {"worker_admitted": False, "admission_granted": False, "scientific_credit": 0,
    "case_credit": 0, "idea_credit": 0, "packing_point_credit": 0, "throughput_credit": 0,
    "site_qualification_credit": 0, "kernel_device_enforcement_verified": False,
    "uva_configuration_service_trust_used": False}


def require(value, message):
    if not value:
        raise ValueError(message)


def adjacent(name):
    path = Path(__file__).with_name(name + ".py").resolve(strict=True)
    if name in sys.modules:
        module = sys.modules[name]
        require(getattr(module, "__file__", None) is not None and Path(module.__file__).resolve(strict=True) == path,
            "direct support cannot use an ambient or mutable helper")
        return module
    specification = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    try:
        specification.loader.exec_module(module)
    except BaseException:
        if sys.modules.get(name) is module:
            del sys.modules[name]
        raise
    return module


def core():
    return adjacent("serval02_card1_binding")


def shared():
    return adjacent("configured_measurement")


def helper_names():
    return tuple(dict.fromkeys(("direct_measurement.py", "configured_measurement.py", *core().HELPERS)))


def historical_support(plan):
    """Load only the explicit current-common read-only historical consumer source."""
    name = "direct_historical_consumer"
    path = Path(__file__).with_name(name + ".py").resolve(strict=True)
    raw = path.read_bytes()
    expected = plan["helpers"]["files"].get("10_jax_exploration_platform/scripts/atari_pixel_campaign/" + name + ".py")
    require(core().digest(raw) == expected, "the exact historical consumer dependency is missing or changed")
    if name in sys.modules:
        module = sys.modules[name]
        require(getattr(module, "__file__", None) == str(path)
            and getattr(module, "_executed_source_sha256", None) == expected,
            "an ambient historical consumer cannot replace the pinned source")
        return module
    specification = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    try:
        exec(compile(raw, str(path), "exec"), module.__dict__)
        module._executed_source_sha256 = expected
    except BaseException:
        if sys.modules.get(name) is module:
            del sys.modules[name]
        raise
    return module



def backend_support(plan):
    """Load only the exact new staged metadata helper; no legacy operational fallback."""
    require("direct_backend_image" in plan, "current direct execution requires backend package-image schema two")
    name = "direct_backend_image.py"
    path = Path(__file__).with_name(name)
    expected = plan["helpers"]["files"].get("10_jax_exploration_platform/scripts/atari_pixel_campaign/" + name)
    require(path.is_file() and core().receipt(path)["sha256"] == expected,
        "the exact new backend helper is missing from this direct source closure")
    module = adjacent("direct_backend_image")
    module.expected_for_plan(plan)
    return module


def capture_backend_completion(directory, plan, result):
    """Join post-numerical actual maps to the new immutable initial backend capture."""
    return backend_support(plan).capture_complete(directory, plan, result)

def configure_plan(plan, profile, history_inputs, consumer, *, backend_image_record=None):
    require(backend_image_record is not None, "every new direct preparation requires a pre-GPU backend package-image record")
    identity = core(); identity.require_profile(profile)
    unit = Path(plan["campaign_run_dir"]) / "data" / plan["unit_id"]
    require(Path(history_inputs) == unit / "direct_history/inputs.json",
        "prepare true frozen history at this unit's reserved input path before the producer plan")
    plan["direct_binding"] = {"schema_version": 1, "profile": profile, "capability": identity.CAPABILITY,
        "consumer": consumer, "physical_identity": identity.pending_identity(),
        "frozen_inputs": identity.receipt(history_inputs), "current_launch_inputs_path": str(unit / "direct_current_launch.json")}
    identity.validate_plan(plan, consumer)
    identity.validate_history(plan)
    adjacent("direct_backend_image").configure_plan(plan, backend_image_record)
    return plan


def support(plan):
    identity = core(); identity.validate_plan(plan)
    backend_support(plan).validate_prepared(plan)
    require("configured_isolation" not in plan and "target_qualification" not in plan
        and "target_qualification_inputs" not in plan, "direct and UVA producer policies cannot be mixed")
    if "direct_qualification" in plan:
        require(identity.exact(plan["direct_qualification"], QUALIFICATION)
            and "direct_qualification_inputs" not in plan, "standalone direct qualification cannot require itself")
        shared().validate_original_qualification(plan)
    else:
        require("direct_qualification_inputs" in plan, "ordinary direct work requires actual independently reviewed direct qualification")
    return sys.modules[__name__]


def bound_hardware(plan, metadata, binding):
    identity = core(); identity.validate_binding(plan, metadata, binding)
    process = metadata["after"]["identity"]
    row, = list(csv.reader(binding["hardware"]["stdout"].splitlines()))
    gpu, model, unused_bus, memory, driver = [part.strip() for part in row]
    return {"hostname": process["hostname"], "pid": process["pid"], "process_start_time": process["process_start_time"],
        "gpu_uuid": gpu, "gpu_query": f"{gpu}, {model}, {memory} MiB, {driver}", "slurm_job_id": "",
        "cuda_visible_devices_original": metadata["environment"]["CUDA_VISIBLE_DEVICES"],
        "cuda_visible_devices": metadata["environment"]["CUDA_VISIBLE_DEVICES"], **BOUNDARY}


def numerical_module_paths(plan):
    return shared().numerical_module_paths(plan)


def observer_inputs(binding, directory):
    """Require the current actual direct child before allowing any assigned query."""
    identity = core(); directory = Path(directory); unit = directory.parent
    require(directory.name in {"paired_memory", "qualification_memory", "profile_memory", "correctness_memory", "confirmation_memory"}
        and directory.resolve(strict=False) == directory and unit.resolve(strict=True) == unit,
        "direct observation must use the actual unit's reserved ordinary directory")
    plan = identity.read_receipt(identity.receipt(unit / "measurement_plan.json"))
    metadata = identity.read_receipt(identity.receipt(unit / "direct_process_metadata.json"))
    saved = identity.read_receipt(identity.receipt(unit / "direct_device_binding.json"))
    require(identity.exact(saved, binding), "direct sampling cannot substitute a remembered device binding")
    identity.validate_binding(plan, metadata, binding, unit)
    verify_observer_helpers(plan, unit)
    require(binding["process"]["pid"] == os.getpid()
        and binding["process"]["process_start_time"] == Path("/proc/self/stat").read_text().rsplit(")", 1)[1].split()[19],
        "direct sampling belongs only to the actual identity-bound process")
    return {"plan": plan, "metadata": metadata, "binding_ref": identity.receipt(unit / "direct_device_binding.json")}


def verify_observer_helpers(plan, unit):
    """Recheck the complete conditional helper closure at every observed window."""
    identity = core(); value = plan["helpers"]; root = Path(value["source_root"])
    names = (*helper_names(), "measure_selected.py")
    if plan["direct_binding"]["consumer"] == "component_profile":
        names += ("profile_selected.py",)
        if plan.get("capture_process_protocol") == "direct-h100-owned-nsight-and-retained-descriptor-version-1":
            names += ("direct_profile_process.py", "direct_profile_prerequisites.py", "direct_backend_image.py", "nsight_process_identity.py",
                "direct_capture_release.py", "direct_tool_descendants.py")
    for name in names:
        relative = identity.PREFIX + name
        raw = subprocess.check_output(["git", "-C", str(root), "show", value["source_commit"] + ":" + relative])
        require(value["files"].get(relative) == identity.digest(raw) == identity.receipt(root / relative)["sha256"]
            == identity.receipt(Path(__file__).with_name(name))["sha256"]
            == identity.receipt(Path(unit) / "code" / name)["sha256"],
            "the complete direct observer helper closure changed")


def window_boundary(inputs):
    identity = core()
    identity.check_live_admission(inputs["metadata"]["current_admission"])
    identity.source_identity(inputs["plan"])
    verify_observer_helpers(inputs["plan"], identity.unit_directory(inputs["plan"]))
    custody = identity.live_custody(inputs["metadata"], inputs["metadata"]["lock_after"]["fd"])
    return {"custody": custody, "current_admission": inputs["metadata"]["current_admission"],
        "binding": inputs["binding_ref"]}


def record_window_failure(directory, counter, phase, error):
    identity = core(); directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
    identity.publish(directory / ("direct_boundary_failure_" + str(counter) + "_" + phase + ".json"), {
        "schema_version": 1, "kind": "direct_window_boundary_failure", "phase": phase,
        "error_type": type(error).__name__, "error": str(error), "measurement_valid": False, **BOUNDARY})


def validate_window(proof, unit):
    """Saved evidence is local and immutable; do not query the device or current sentinel."""
    identity = core(); unit = Path(unit)
    plan = identity.read_receipt(identity.receipt(unit / "measurement_plan.json"))
    metadata = identity.read_receipt(identity.receipt(unit / "direct_process_metadata.json"))
    binding_ref = identity.receipt(unit / "direct_device_binding.json")
    binding = identity.read_receipt(binding_ref)
    identity.validate_binding(plan, metadata, binding, unit)
    verify_observer_helpers(plan, unit)
    require(proof.get("direct_binding") == binding_ref and proof.get("startup_memory_observed") is False
        and proof.get("gpu_uuid") == binding["gpu_uuid"] and proof.get("pid") == binding["process"]["pid"],
        "direct window changed its actual process or current physical binding")
    expected = {"custody": binding["custody_after"], "current_admission": metadata["current_admission"], "binding": binding_ref}
    require(identity.exact(proof.get("direct_before"), expected) and identity.exact(proof.get("direct_after"), expected),
        "direct window source or held descriptor custody changed")
    value = shared()._validate_window_for_policy(proof, "direct_serval02_card_one")
    require(all(sample["free_bytes"] * 100 >= sample["total_bytes"] * 15 for sample in proof["samples"]),
        "the direct window has less than the unchanged fifteen percent actual physical memory free")
    return value


def process_sample(gpu, pid):
    return shared().process_sample(gpu, pid)


def validate_process_sample(value, gpu, pid):
    return shared().validate_process_sample(value, gpu, pid)


def MemoryObservation(device, binding, directory):
    require(binding.get("kind") == "serval02_card1_current_process_resolved_device_binding",
        "the separate direct observer cannot accept a UVA binding")
    return shared().MemoryObservation(device, binding, directory)


@contextmanager
def paired_observer(paired, binding, unit):
    """Reuse one observer across its complete rounds so every saved window is distinct."""
    original, observer, device_identity = paired.MemoryObservation, [], []
    def factory(device):
        if not observer:
            observer.append(MemoryObservation(device, binding, Path(unit) / "paired_memory")); device_identity.append(device)
        require(device is device_identity[0], "a direct paired window changed its actual device object")
        return observer[0]
    paired.MemoryObservation = factory
    original_error = None
    try:
        yield
    except BaseException as error:
        original_error = error; raise
    finally:
        paired.MemoryObservation = original
        if observer and observer[0].thread is not None and observer[0].thread.is_alive():
            try:
                observer[0].finish()
            except BaseException:
                if original_error is None:
                    raise


@contextmanager
def observed_work(device, binding, directory):
    Path(directory).mkdir(parents=True, exist_ok=True)
    value = MemoryObservation(device, binding, directory); result = {}; original_error = None
    value.begin()
    try:
        yield result
    except BaseException as error:
        original_error = error; raise
    finally:
        try:
            result.update(value.finish())
        except BaseException as error:
            if original_error is None:
                raise
            core().publish(Path(directory) / "sampling_failure_after_numerical_failure.json", {
                "numerical_error_type": type(original_error).__name__, "sampling_error_type": type(error).__name__,
                "measurement_valid": False, **BOUNDARY})


def post_binding_checks(plan, metadata, binding, directory):
    """Retain conservative H100 checks at their explicit post-context,pre-learner stage."""
    identity = core(); identity.validate_binding(plan, metadata, binding, directory)
    identity.check_live_admission(metadata["current_admission"])
    identity.live_custody(metadata, metadata["lock_after"]["fd"])
    gpu = binding["gpu_uuid"]; queries = []
    attempt = {"schema_version": 1, "kind": "direct_bound_h100_startup_checks", "binding": identity.receipt(Path(directory) / "direct_device_binding.json"),
        "pre_context_occupancy_observed": False, "startup_memory_observed": False, **BOUNDARY}
    try:
        command = ["nvidia-smi", "-i", gpu, "--query-gpu=uuid,mig.mode.current,memory.used", "--format=csv,noheader,nounits"]
        raw = shared().bounded_query(command); queries.append((command, raw))
        require(raw["complete"] is True and raw["capture_error"] is None and type(raw["returncode"]) is int
            and raw["returncode"] == 0 and raw["stderr"] == b"", "bound H100 mode query failed")
        text = raw["stdout"].decode("utf-8", errors="strict"); rows = list(csv.reader(text.splitlines(), strict=True))
        require(len(rows) == 1 and len(rows[0]) == 3, "bound H100 mode row is incomplete")
        actual, mig, used = [part.strip() for part in rows[0]]
        require(actual == gpu and mig == "Disabled" and re.fullmatch(r"(?:0|[1-9][0-9]*)", used)
            and int(used) < 1024, "the bound H100 startup exceeds the unchanged mode or 1024 MiB threshold")
        attempt["mode_query"] = {"command": command, "stdout": text, "stdout_sha256": identity.digest(raw["stdout"]), "returncode": 0, "stderr": ""}
        command = ["nvidia-smi", "conf-compute", "-f", "-i", gpu]
        raw = shared().bounded_query(command); queries.append((command, raw))
        require(raw["complete"] is True and raw["capture_error"] is None and type(raw["returncode"]) is int
            and raw["returncode"] == 0 and raw["stderr"] == b"", "bound confidential-computing query failed")
        text = raw["stdout"].decode("utf-8", errors="strict")
        require("CC status: OFF" in text, "the bound H100 must keep confidential computing off")
        attempt["confidential_computing_query"] = {"command": command, "stdout": text, "stdout_sha256": identity.digest(raw["stdout"]), "returncode": 0, "stderr": ""}
        attempt["process_query"] = process_sample(gpu, os.getpid())
        validate_process_sample(attempt["process_query"], gpu, os.getpid())
        attempt["passed"] = True
    except BaseException as error:
        attempt.update(passed=False, error_type=type(error).__name__, error=str(error), anonymous_queries=[{
            "command": command, "stdout_sha256": identity.digest(raw["stdout"]), "stderr_sha256": identity.digest(raw["stderr"]),
            "stdout_bytes": len(raw["stdout"]), "stderr_bytes": len(raw["stderr"]), "returncode": raw["returncode"],
            "complete": raw["complete"], "capture_error": raw["capture_error"]} for command, raw in queries])
        raise
    finally:
        identity.publish(Path(directory) / "direct_startup_checks.json", attempt)
    return attempt


def capture_backend(directory, binding):
    """Capture only the new separately prepared actual backend package image."""
    identity = core()
    directory = Path(directory)
    plan = identity.read_receipt(identity.receipt(directory / "measurement_plan.json"))
    return backend_support(plan).capture_begin(directory, plan, binding)



def qualification_sources(plan, cleanrl_root, software_result):
    return shared().qualification_sources(plan, cleanrl_root, software_result)


def verify_qualification_backend(plan, backend):
    return shared().verify_qualification_backend(plan, backend)


def run_qualification(directory, plan, paired, jax, Config, Trainer, Environment):
    value = shared().run_qualification(directory, plan, paired, jax, Config, Trainer, Environment)
    return {**value, "qualification_protocol": QUALIFICATION}


def verify_saved_sources(directory, plan):
    identity = core(); identity.validate_plan(plan)
    backend_support(plan)
    directory = Path(directory)
    require(directory == identity.unit_directory(plan) and identity.read_receipt(identity.receipt(directory / "measurement_prepared.json")) == {
        "plan_sha256": identity.receipt(directory / "measurement_plan.json")["sha256"]}, "direct saved unit or frozen plan changed")
    diagnostic, training = adjacent("diagnose_gpu_execution"), adjacent("train_optimized")
    require(diagnostic.core_identity(Path(plan["source_root"]), plan["source_commit"]) == plan["core"], "the original numerical source changed")
    for name in ("paired_source", "helpers"):
        value = plan[name]
        require(training.source_receipt(Path(value["source_root"]), value["source_commit"], list(value["files"])) == value, "direct saved helper/source receipt changed")
    for path, digest in plan["helpers"]["files"].items():
        require(identity.receipt(directory / "code" / Path(path).name)["sha256"] == digest, "direct staged helper bytes changed")
    require(set(helper_names()) <= {Path(path).name for path in plan["helpers"]["files"]}, "direct producer omitted a declared dependency")
    identity.source_identity(plan)


def prepare_qualification_inputs(plan, result_path, review_path):
    require(result_path is not None and review_path is not None, "actual independent direct qualification inputs are required")
    plan["direct_qualification_inputs"] = {"result": core().receipt(result_path), "review": core().receipt(review_path)}
    verify_qualification_inputs(plan)


def terminal_process(directory, metadata):
    """Require actual own-child wait evidence rather than a Slurm or supplied success label."""
    identity = core(); directory = Path(directory); process = metadata["after"]["identity"]
    terminal = identity.read_receipt(identity.receipt(directory / "direct_terminal_process.json"))
    wait = identity.read_receipt(terminal["return_record"])
    require(type(terminal.get("schema_version")) is int and terminal["schema_version"] == 1
        and terminal.get("kind") == "direct_serval02_card1_terminal_process"
        and terminal.get("launch") == identity.receipt(directory / "direct_current_launch.json")
        and all(terminal.get(key) == process[key] for key in ("pid", "hostname", "process_start_time"))
        and type(terminal.get("uid")) is int and terminal["uid"] == 2618919,
        "the direct terminal proof does not name this actual launch and process birth")
    command = shlex.split((directory / "command.txt").read_text())
    plan = identity.read_receipt(identity.receipt(directory / "measurement_plan.json"))
    require(command == [plan["python"], str(directory / "code/measure_selected.py"), "--prepared-unit", str(directory)],
        "the actual direct command differs from its prepared producer and unit")
    require(type(wait.get("schema_version")) is int and wait["schema_version"] == 1
        and wait.get("kind") == "owned_direct_process_wait_return" and wait.get("command") == command
        and all(wait.get(key) == process[key] for key in ("pid", "hostname", "process_start_time"))
        and type(wait.get("uid")) is int and wait["uid"] == 2618919
        and type(wait.get("waited_pid")) is int and wait["waited_pid"] == process["pid"]
        and wait.get("wait_semantics") == "Popen.wait on this exact spawned child"
        and type(wait.get("returncode")) is int and wait["returncode"] == 0 and wait.get("exited_normally") is True,
        "the actual direct child did not return normally with its exact command and identity")
    parent_fields = wait["waiter_before"]["stat"].rsplit(")", 1)[1].split()
    identity.validate_owner_process(wait["waiter_before"], process["parent_pid"], parent_fields[19])
    identity.validate_owner_process(wait["waiter_after"], process["parent_pid"], parent_fields[19])
    return terminal


def wait_document(path):
    """Wait only for absent immutable own metadata; malformed present data refuses."""
    deadline = time.monotonic() + 180
    while not Path(path).exists():
        require(not Path(path).is_symlink(), "an own launch input cannot be an unresolved alias")
        if time.monotonic() >= deadline:
            raise TimeoutError("the exact own profiling launch metadata did not arrive within 180 seconds")
        time.sleep(.05)
    return core().read_receipt(core().receipt(path))


def capture_command(plan, directory, descriptor):
    return [plan["python"], str(Path(directory) / "code/profile_selected.py"), "--capture-unit", str(directory),
        "--direct-held-lock-fd", str(descriptor)]


def run_profile_waiter(plan, directory, descriptor):
    """Wait on exactly one real capture child without importing a numerical runtime."""
    identity = core(); directory = Path(directory); identity.validate_plan(plan, "component_profile")
    require(not any(name.split(".")[0] in {"jax", "jaxlib", "numpy", "torch"} for name in sys.modules)
        and Path(__file__).resolve(strict=True) == directory / "code/direct_measurement.py",
        "only the exact staged metadata-only direct waiter may launch this child")
    own = identity.current_process(); process = identity.validate_process(own)
    parent_ref = identity.receipt(directory / "supervisor/direct_process_metadata.json")
    parent = identity.read_receipt(parent_ref); identity.validate_metadata(plan, parent)
    identity.check_live_admission(parent["current_admission"])
    holder = parent["after"]["identity"]
    lock = identity.lock_metadata(descriptor); identity.validate_lock(lock, holder["pid"])
    require(descriptor == parent["lock_after"]["fd"], "the waiter lost the exact inherited owner descriptor")
    owner = wait_document(directory / "profiler_process.json")
    require(process["parent_pid"] == process["process_group"] == owner["pid"]
        and own["stat"].rsplit(")", 1)[1].split()[3] == str(owner["pid"])
        and owner["parent_pid"] == holder["pid"] and owner["parent_start_time"] == holder["process_start_time"],
        "the waiter is not this exact new Nsight process descendant")
    identity.validate_owner_process(identity.capture_owner_process(holder["pid"]), holder["pid"], holder["process_start_time"])
    identity.validate_owner_process(identity.capture_owner_process(owner["pid"]), owner["pid"], owner["process_start_time"], holder["pid"])
    waiter = {"schema_version": 1, "kind": "direct_metadata_only_profile_waiter", "process": own,
        "plan": identity.receipt(directory / "measurement_plan.json"), "parent_custody": parent_ref,
        "profiler_process": identity.receipt(directory / "profiler_process.json"), "lock": lock, **BOUNDARY}
    identity.publish(directory / "direct_profile_waiter.json", waiter)
    command = capture_command(plan, directory, descriptor)
    child = subprocess.Popen(command, pass_fds=(descriptor,))
    launched = None
    try:
        raw = identity.capture_owner_process(child.pid)
        fields = raw["stat"].rsplit(")", 1)[1].split(); birth = fields[19]
        identity.validate_owner_process(raw, child.pid, birth, process["pid"])
        require(fields[2] == fields[3] == str(owner["pid"]), "the actual capture child left the exact new process group")
        command_raw = (Path("/proc") / str(child.pid) / "cmdline").read_bytes()
        require(len(command_raw) <= 65536 and command_raw.rstrip(b"\0").split(b"\0") == [value.encode() for value in command],
            "the actual capture executable or argv changed before release")
        launched = {"schema_version": 1, "kind": "direct_actual_capture_child_launch", "command": command,
            "child": raw, "pid": child.pid, "process_start_time": birth, "parent_pid": process["pid"],
            "parent_start_time": process["process_start_time"], "uid": 2618919,
            "cmdline": command_raw.decode("utf-8", errors="strict"), "cmdline_bytes": len(command_raw),
            "cmdline_sha256": identity.digest(command_raw), "waiter": identity.receipt(directory / "direct_profile_waiter.json"), **BOUNDARY}
        identity.publish(directory / "direct_capture_launch.json", launched)
        returncode = child.wait()
        after = identity.current_process(); identity.validate_process(after)
        require(after["identity"] == process and identity.lock_metadata(descriptor) == lock,
            "the waiter process or retained descriptor changed while waiting")
        record = {"schema_version": 1, "kind": "direct_actual_capture_child_wait_return", "command": command,
            "pid": child.pid, "process_start_time": birth, "uid": 2618919, "waited_pid": child.pid,
            "returncode": returncode, "exited_normally": returncode >= 0, "wait_semantics": "Popen.wait on this exact spawned child",
            "launch": identity.receipt(directory / "direct_capture_launch.json"), "waiter_after": after,
            "lock_after": identity.lock_metadata(descriptor), **BOUNDARY}
        identity.publish(directory / "direct_capture_wait_return.json", record)
        return returncode if returncode >= 0 else 128 - returncode
    except BaseException as error:
        identity.publish(directory / "direct_capture_wait_failure.json", {"schema_version": 1,
            "kind": "direct_capture_wait_failure", "error_type": type(error).__name__, "error": str(error),
            "launched": launched, "known_pid": child.pid, **BOUNDARY})
        if child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=120)
            except subprocess.TimeoutExpired:
                child.kill(); child.wait()
        raise


def validate_waiter_chain(plan, directory, process_record, descriptor, *, live):
    identity = core(); directory = Path(directory); process = identity.validate_process(process_record)
    launch = wait_document(directory / "direct_capture_launch.json") if live else identity.read_receipt(identity.receipt(directory / "direct_capture_launch.json"))
    waiter = identity.read_receipt(launch["waiter"]); own = identity.validate_process(waiter["process"])
    parent = identity.read_receipt(waiter["parent_custody"]); identity.validate_metadata(plan, parent)
    owner = identity.read_receipt(waiter["profiler_process"]); holder = parent["after"]["identity"]
    fields = launch["child"]["stat"].rsplit(")", 1)[1].split()
    require(type(waiter.get("schema_version")) is int and waiter["schema_version"] == 1
        and waiter.get("kind") == "direct_metadata_only_profile_waiter" and waiter["plan"] == identity.receipt(directory / "measurement_plan.json")
        and waiter["parent_custody"] == identity.receipt(directory / "supervisor/direct_process_metadata.json")
        and waiter["profiler_process"] == identity.receipt(directory / "profiler_process.json")
        and own["parent_pid"] == own["process_group"] == owner["pid"] and owner["parent_pid"] == holder["pid"]
        and owner["parent_start_time"] == holder["process_start_time"]
        and type(launch.get("schema_version")) is int and launch["schema_version"] == 1
        and launch.get("kind") == "direct_actual_capture_child_launch"
        and launch["waiter"] == identity.receipt(directory / "direct_profile_waiter.json")
        and launch["command"] == capture_command(plan, directory, descriptor)
        and isinstance(launch.get("cmdline"), str) and 0 < launch.get("cmdline_bytes", 0) <= 65536
        and type(launch.get("cmdline_bytes")) is int and len(launch["cmdline"].encode()) == launch["cmdline_bytes"]
        and identity.digest(launch["cmdline"].encode()) == launch["cmdline_sha256"]
        and launch["cmdline"].encode().rstrip(b"\0").split(b"\0") == [part.encode() for part in launch["command"]]
        and launch["pid"] == process["pid"] and launch["process_start_time"] == process["process_start_time"]
        and launch["parent_pid"] == process["parent_pid"] == own["pid"] and launch["parent_start_time"] == own["process_start_time"]
        and type(launch["uid"]) is int and launch["uid"] == 2618919
        and process["process_group"] == owner["pid"] and fields[2] == fields[3] == str(owner["pid"])
        and descriptor == waiter["lock"]["fd"] == parent["lock_after"]["fd"],
        "the owner,Nsight,waiter and actual capture chain is incomplete or changed")
    identity.validate_owner_process(launch["child"], process["pid"], process["process_start_time"], own["pid"])
    identity.validate_lock(waiter["lock"], holder["pid"])
    if live:
        identity.validate_owner_process(identity.capture_owner_process(own["pid"]), own["pid"], own["process_start_time"], owner["pid"])
        identity.validate_owner_process(identity.capture_owner_process(owner["pid"]), owner["pid"], owner["process_start_time"], holder["pid"])
        identity.validate_owner_process(identity.capture_owner_process(holder["pid"]), holder["pid"], holder["process_start_time"])
        identity.validate_lock(identity.lock_metadata(descriptor), holder["pid"])
    return launch, waiter


def validate_capture_wait(directory, metadata):
    identity = core(); directory = Path(directory)
    plan = identity.read_receipt(identity.receipt(directory / "measurement_plan.json"))
    if metadata.get("role") == "producer_guarded_child":
        return adjacent("direct_profile_process").validate_saved(directory, plan)
    launch, waiter = validate_waiter_chain(plan, directory, metadata["after"], metadata["lock_after"]["fd"], live=False)
    value = identity.read_receipt(identity.receipt(directory / "direct_capture_wait_return.json"))
    require(type(value.get("schema_version")) is int and value["schema_version"] == 1
        and value.get("kind") == "direct_actual_capture_child_wait_return"
        and value.get("command") == launch["command"] and value.get("launch") == identity.receipt(directory / "direct_capture_launch.json")
        and value.get("pid") == value.get("waited_pid") == launch["pid"] and value.get("process_start_time") == launch["process_start_time"]
        and type(value.get("uid")) is int and value["uid"] == 2618919
        and type(value.get("returncode")) is int and value["returncode"] == 0 and value.get("exited_normally") is True
        and value.get("wait_semantics") == "Popen.wait on this exact spawned child"
        and identity.validate_process(value["waiter_after"]) == waiter["process"]["identity"]
        and value["lock_after"] == waiter["lock"], "the actual Python capture child lacks an exact ordinary zero wait return")
    return value


def stop_profile_descendants(directory, child, parent, descriptor, owner=None):
    """Retain the owner descriptor until this newly created process group has ended."""
    identity = core(); directory = Path(directory)
    identity.live_custody(parent, descriptor)
    if owner is None:
        owner = identity.read_receipt(identity.receipt(directory / "profiler_process.json"))
    require(owner["pid"] == child.pid and owner["parent_pid"] == os.getpid(), "refuse to stop an unrelated profiler group")
    known = [(owner["pid"], owner["process_start_time"])]
    for name, key in (("direct_profile_waiter.json", "process"), ("direct_capture_launch.json", "child")):
        path = directory / name
        if path.exists():
            raw = identity.read_receipt(identity.receipt(path))[key]
            if key == "process":
                process = identity.validate_process(raw)
                raw = {"pid": process["pid"], **{field: raw[field] for field in ("stat", "stat_sha256", "uid_bytes", "uid_sha256")}}
            fields = raw["stat"].rsplit(")", 1)[1].split(); pid = int(raw["stat"].split(" ", 1)[0])
            identity.validate_owner_process(raw, pid, fields[19])
            require(fields[2] == fields[3] == str(child.pid), "a saved direct descendant changed process group")
            known.append((pid, fields[19]))
    def remaining():
        active = []
        for pid, birth in known:
            try:
                raw = identity.capture_owner_process(pid)
            except FileNotFoundError:
                continue
            fields = raw["stat"].rsplit(")", 1)[1].split()
            if fields[19] != birth or fields[0] == "Z":
                continue
            identity.validate_owner_process(raw, pid, birth)
            require(fields[2] == str(child.pid), "the exact own descendant left its new group")
            active.append(pid)
        return active
    if remaining():
        try:
            os.killpg(child.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    deadline = time.monotonic() + 120
    while remaining() and time.monotonic() < deadline:
        time.sleep(.1)
    if remaining():
        try:
            os.killpg(child.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    # Do not close the owner descriptor while a proved own descendant is still active.
    while remaining():
        identity.live_custody(parent, descriptor); time.sleep(.1)
    child.wait()
    identity.live_custody(parent, descriptor)


def trace_process_proof(directory, metadata, *, publish_result=True):
    """Join positive recorded CUDA events to the actual waited Python capture PID."""
    identity = core(); directory = Path(directory)
    validate_capture_wait(directory, metadata)
    process = metadata["after"]["identity"]; pid = process["pid"]
    require(type(pid) is int and 0 < pid < 2 ** 24, "unsupported Nsight process identifier range")
    analysis_ref = identity.receipt(directory / "trace/analysis.json")
    analysis = identity.read_receipt(analysis_ref)
    rows = [row for row in analysis["derived_files"] if Path(row["path"]).suffix == ".sqlite"]
    require(analysis.get("completed") is True and len(rows) == 1, "the direct profile needs one complete native SQLite export")
    row, = rows; path = Path(row["path"])
    require(path.parent == directory / "trace" and identity.receipt(path) == {"path": str(path), "sha256": row["sha256"]}
        and type(row["bytes"]) is int and row["bytes"] > 0 and path.stat().st_size == row["bytes"],
        "the exact immutable trace database changed")
    before = identity.receipt(path)
    counts = {"kernels": 0, "cuda_api_calls": 0}; tables_seen = []
    allowed = {"CUPTI_ACTIVITY_KIND_KERNEL": ("globalPid", "kernels"),
        "CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL": ("globalPid", "kernels"),
        "CUPTI_ACTIVITY_KIND_RUNTIME": ("globalTid", "cuda_api_calls"),
        "CUPTI_ACTIVITY_KIND_DRIVER": ("globalTid", "cuda_api_calls")}
    with sqlite3.connect(path.as_uri() + "?mode=ro&immutable=1", uri=True) as connection:
        connection.execute("PRAGMA query_only=ON")
        tables = {name for name, in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        for name, (column, category) in allowed.items():
            if name not in tables:
                continue
            columns = {row[1] for row in connection.execute('PRAGMA table_info("' + name + '")')}
            require({column, "start", "end"} <= columns, "the native trace has an unsupported CUDA event schema")
            count = 0
            # Names come only from the literal table/column roster above.
            for actual, start, end in connection.execute('SELECT "' + column + '",start,end FROM "' + name + '"'):
                require(type(actual) is int and actual >= 0 and type(start) is int and type(end) is int
                    and end > start >= 0 and (actual // 2 ** 24) % 2 ** 24 == pid,
                    "a CUDA trace event lacks the actual capture PID or positive duration")
                count += 1
            counts[category] += count; tables_seen.append({"table": name, "identity_column": column, "rows": count})
    require(identity.receipt(path) == before and all(counts.values()), "the actual capture lacks positive CUDA API and kernel events")
    value = {"schema_version": 1, "kind": "direct_actual_capture_cuda_trace_process_proof", "analysis": analysis_ref,
        "database": before, "report": identity.receipt(directory / "trace/capture.nsys-rep"),
        "capture_wait": identity.receipt(directory / "direct_capture_wait_return.json"),
        "pid": pid, "process_start_time": process["process_start_time"], "tables": tables_seen, **counts,
        "pid_encoding": "documented Nsight process bits 24 through 47", **BOUNDARY}
    target = directory / "direct_trace_process.json"
    if publish_result:
        identity.publish(target, value)
    else:
        require(identity.exact(identity.read_receipt(identity.receipt(target)), value), "the native CUDA capture-process proof changed")
    return identity.receipt(target)


EXPECTED_DTYPES = {"parameters": ["float32"], "target": ["float32"], "optimizer_first": ["float32"],
    "optimizer_second": ["float32"], "observation_statistics": ["float64"], "reward_statistics": ["float64"]}


def validate_original_qualification(plan):
    return shared().validate_original_qualification(plan)


def verify_qualification_source_receipts(plan):
    return shared().verify_qualification_source_receipts(plan)


def renderer_contract(plan, backend):
    return shared().renderer_contract(plan, backend)


def validate_test_collection(plan, tests, stdout):
    return shared().validate_test_collection(plan, tests, stdout)


def verify_qualification_inputs(plan):
    """Refuse a missing, changed, unrelated or unreviewed actual full-original qualification."""
    if "accepted_evidence_snapshot_protocol" in plan:
        return adjacent("accepted_evidence_snapshot").qualification(plan)
    if "common_correctness_inputs" in plan:
        return historical_support(plan).request(plan, "qualification")
    identity = core()
    expected_backend = backend_support(plan).expected_for_plan(plan)
    inputs = plan["direct_qualification_inputs"]
    result = identity.read_receipt(inputs["result"])
    review = identity.read_receipt(inputs["review"])
    directory = Path(inputs["result"]["path"]).parent
    qualified_plan = identity.read_receipt(identity.receipt(directory / "measurement_plan.json"))
    qualified_backend = backend_support(qualified_plan).expected_for_plan(qualified_plan)
    require(all(identity.exact(expected_backend[key], qualified_backend[key])
        for key in ("selected_python", "site_packages", "image", "distribution")),
        "this direct consumer changed the separately qualified backend package image or owning records")
    require(Path(inputs["result"]["path"]).name == "direct_qualification_result.json"
        and identity.exact(qualified_plan.get("direct_qualification"), QUALIFICATION),
        "a generic successful measurement cannot substitute for target qualification")
    for key in ("source_commit", "hardware_class", "runtime_environment", "asset_record", "direct_binding", "copy_seeds"):
        require(key in qualified_plan and key in result and identity.exact(result[key], qualified_plan[key]),
            "the full original qualification result changed its saved " + key)
    for key in ("source_commit", "source_root", "python", "hardware_class", "runtime_environment", "asset_record"):
        require(key in plan and identity.exact(qualified_plan[key], plan[key]),
            "the outer direct consumer differs from its qualified " + key)
    require(identity.exact(qualified_plan["direct_binding"]["profile"], plan["direct_binding"]["profile"]),
        "the outer direct consumer differs from its separately qualified profile")
    validate_original_qualification(qualified_plan)
    verify_saved_sources(directory, qualified_plan)
    verify_qualification_source_receipts(qualified_plan)
    metadata = identity.read_receipt(identity.receipt(directory / "direct_process_metadata.json"))
    binding = identity.read_receipt(identity.receipt(directory / "direct_device_binding.json"))
    identity.validate_binding(qualified_plan, metadata, binding, directory)
    execution = adjacent("post_comparison_validation").execution_identity(result)
    require(execution == tuple(metadata["after"]["identity"][key] for key in ("hostname", "pid", "process_start_time")),
        "the full original qualification execution differs from its actual metadata process")
    require(result.get("kind") == "direct_serval02_card1_full_original_qualification"
        and result.get("completed") is True and identity.exact(result.get("qualification_protocol"), QUALIFICATION)
        and result.get("source_commit") == plan["source_commit"] and result.get("hardware_class") == plan["hardware_class"]
        and result.get("runtime_environment") == plan["runtime_environment"]
        and result.get("direct_binding", {}).get("profile") == plan["direct_binding"]["profile"]
        and result.get("asset_record") == plan["asset_record"] and result.get("all_original_gates_passed") is True,
        "the actual full-original qualification is incomplete or differs from this target")
    require(all(type(result.get(key)) is int and result[key] == 0 for key in (
        "scientific_credit", "case_credit", "idea_credit", "packing_point_credit", "throughput_credit", "site_qualification_credit"))
        and result.get("worker_admitted") is False and result.get("admission_granted") is False,
        "target qualification cannot grant scientific or admission credit")
    require(identity.exact(result.get("resolved_device_binding"), binding) and identity.exact(result.get("helpers"), qualified_plan["helpers"])
        and result.get("reference_iteration") == 1
        and result.get("executed_work", {}).get("full_reference_warmup_transitions") == 819200
        and result["executed_work"].get("full_reference_training_transitions") == 16384
        and result["executed_work"].get("full_reference_updates") == 1
        and result["executed_work"].get("full_reference_optimizer_minibatch_updates") == 16,
        "qualification changed its actual physical worker or controlled full-original work")
    memory = result["qualification_memory"]
    rebuilt_memory = validate_window(memory["assigned_gpu_process_observation"], directory)
    require(all(memory.get(key) == value for key, value in rebuilt_memory.items())
        and memory["gpu_uuid"] == binding["gpu_uuid"]
        and memory["assigned_gpu_process_observation"]["pid"] == metadata["after"]["identity"]["pid"],
        "qualification memory does not belong to its actual resolved worker")
    require(identity.exact(result.get("dtypes"), EXPECTED_DTYPES)
        and type(result.get("finite_state_leaves")) is int and result["finite_state_leaves"] > 0,
        "full reference dtype groups or finite-state evidence are absent")
    for key, expected in (("reference_iteration", 1), ("reference_optimizer_step", 16)):
        require(type(result.get(key)) is int and result[key] == expected, "a reference update counter changed type or value")
    for key, expected in (("full_reference_warmup_transitions", 819200), ("full_reference_training_transitions", 16384),
            ("full_reference_updates", 1), ("full_reference_optimizer_minibatch_updates", 16)):
        require(type(result["executed_work"].get(key)) is int and result["executed_work"][key] == expected,
            "controlled full-original work counters must be exact integers")
    renderer = identity.read_receipt(identity.receipt(directory / "original_renderer.json"))
    backend = identity.read_receipt(identity.receipt(directory / "installed_backend.json"))
    matrix = renderer_contract(qualified_plan, backend)
    require(identity.exact(qualified_plan.get("renderer_contract"), matrix) and identity.exact(result.get("renderer_contract"), matrix)
        and renderer == result["renderer"] and renderer.get("checks") == {"cache": "passed", "palette": "passed", "fused": "passed"}
        and all(type(renderer.get(key)) is int and renderer[key] == matrix[key] for key in ("rooms", "cases", "synthetic_pixel_cases")),
        "the complete source-bound original renderer matrix is absent, reduced or failed")
    require(result["installed_backend"] == backend, "actual installed backend changed between unit and result")
    verify_qualification_backend(qualified_plan, backend)
    import train_optimized as training
    loaded = result["loaded_gpu_libraries"]
    training.validate_gpu_library_receipt(loaded, {"installed_gpu_libraries": backend["gpu_libraries"]})
    require(type(loaded.get("pid")) is int and loaded["pid"] == metadata["after"]["identity"]["pid"]
        and loaded.get("platform") in {"gpu", "cuda"} and isinstance(loaded.get("platform_version"), str)
        and loaded["platform_version"] and binding["native"]["driver_library"] in loaded.get("loaded_shared_libraries", []),
        "the actual loaded libraries are not this native-bound GPU process")
    for path, value in loaded["verified_package_libraries"].items():
        require(identity.receipt(path)["sha256"] == value["sha256"], "an actual shared CUDA/cuDNN image changed")
    actual_assets = identity.read_receipt(identity.receipt(directory / "asset_receipt.json"))
    require(actual_assets == identity.read_receipt(qualified_plan["qualification_software"]["asset_receipt"]),
        "the actual original asset receipt changed")
    tests = identity.read_receipt(identity.receipt(directory / "original_tests.json"))
    require(tests["command"] == [qualified_plan["python"], "-m", "pytest", *shared().qualification_pytest_arguments(qualified_plan)]
        and type(tests["returncode"]) is int and tests["returncode"] == 0
        and tests["reports"] == result["original_test_reports"] and tests["reports"]
        and all(row["outcome"] == "passed" for row in tests["reports"]), "full original test evidence changed")
    validate_test_collection(qualified_plan, tests, (directory / "original_tests.stdout").read_text())
    require(tests["collected_nodeids"] == result["original_collected_test_nodeids"], "saved collected-test identities changed")
    for name, recorded in result.get("gate_artifacts", {}).items():
        require(name in {"original_tests.json", "original_tests.stdout", "original_tests.stderr", "original_renderer.json"}
            and identity.receipt(directory / name) == recorded, "actual full-gate raw artifact changed")
    require(set(result.get("gate_artifacts", {})) == {
        "original_tests.json", "original_tests.stdout", "original_tests.stderr", "original_renderer.json"}
        and result.get("reference_optimizer_step") == 16 and result.get("frozen_target_unchanged") is True
        and result.get("policy_parameters_changed") is True, "full reference learner state or gate outputs are incomplete")
    require(type(review.get("schema_version")) is int and review["schema_version"] == 1
        and review.get("kind") == "independent_full_original_direct_qualification_review"
        and review.get("result") == inputs["result"] and review.get("approved") is True
        and review.get("admission_granted") is False and type(review.get("scientific_credit")) is int and review["scientific_credit"] == 0
        and review.get("independent_of_producer") is True
        and review.get("terminal_process") == identity.receipt(directory / "direct_terminal_process.json")
        and all(review.get(key) == metadata["after"]["identity"][key] for key in ("hostname", "pid", "process_start_time"))
        and type(review.get("actual_returncode")) is int and review["actual_returncode"] == 0,
        "an exact independent actual qualification review is required")
    validate_startup_checks(directory, binding)
    validate_backend_capture(directory, binding, result["loaded_gpu_libraries"])
    terminal_process(directory, metadata)
    return result


def verify_saved_measurement(directory, result, *, historical_profile_read_only=False):
    """Validate the explicit local single-card evidence; portable, packing and final readers remain unsupported."""
    identity = core()
    directory = Path(directory)
    plan = identity.read_receipt(identity.receipt(directory / "measurement_plan.json"))
    if plan.get("kind") == "selected_configuration_profile":
        require(type(historical_profile_read_only) is bool, "historical profile access must be explicitly read-only")
        if "direct_profile_protocol" in plan:
            adapter = adjacent("direct_profile_prerequisites")
            adapter.contract(plan)
            adapter.require_admission()
        else:
            require(historical_profile_read_only,
                "historical direct profile readers require explicit read-only access; an old unit is not new admission")
    identity.validate_plan(plan)
    require(directory == Path(plan["campaign_run_dir"]) / "data" / plan["unit_id"]
        and identity.read_receipt(identity.receipt(directory / "measurement_prepared.json")) == {
            "plan_sha256": identity.receipt(directory / "measurement_plan.json")["sha256"]},
        "configured result changed its exact prepared unit")
    require("direct_qualification" not in plan and result.get("direct_binding") == plan["direct_binding"]
        and result.get("direct_binding_record") is True and all(result.get(key) == value for key, value in BOUNDARY.items()),
        "a qualification or unrecognized record cannot supply selected measurement evidence")
    verify_saved_sources(directory, plan)
    metadata = identity.read_receipt(identity.receipt(directory / "direct_process_metadata.json"))
    binding = identity.read_receipt(identity.receipt(directory / "direct_device_binding.json"))
    identity.validate_binding(plan, metadata, binding, directory)
    require(result.get("resolved_device_binding") == binding and result.get("hardware", {}).get("gpu_uuid") == binding["gpu_uuid"]
        and result.get("helpers") == plan["helpers"] and result.get("source_commit") == plan["source_commit"]
        and result.get("runtime_environment") == plan["runtime_environment"], "result changed its physical/source/runtime binding")
    execution = result.get("execution_identity", {})
    require(all(execution.get(key) == metadata["after"]["identity"][key] for key in ("pid", "hostname", "process_start_time")),
        "result substituted the identity-bound worker")
    verify_qualification_inputs(plan)
    require(result.get("direct_qualification_inputs") == plan["direct_qualification_inputs"], "actual qualification inputs changed")
    require(result.get("copy_seeds") == plan["copy_seeds"], "result changed independent policy seeds")
    if plan.get("kind") == "selected_configuration_measurement":
        require(result.get("candidate") == plan["configuration"] and result.get("reference") == plan["reference_configuration"]
            and result.get("measurement_mode") == plan["mode"] and result.get("protocol") == {
                "rounds": plan["rounds"], "requested_minimum_updates_per_round": plan["minimum_updates_per_round"],
                "actual_updates_per_round": plan["iterations"]}, "selected configurations or prepared windows changed")
    if plan.get("mode") == "confirmation":
        require(type(plan["rounds"]) is int and plan["rounds"] == 7 and len(result.get("rounds", [])) == 7,
            "intermediate confirmation must retain all seven paired rounds")
        for pair in result["rounds"]:
            for arm in ("reference", "candidate"):
                memory = pair[arm]["memory"]
                rebuilt = validate_window(memory["assigned_gpu_process_observation"], directory)
                require(all(memory.get(key) == value for key, value in rebuilt.items()), "a measured memory window changed")
                require(memory["gpu_uuid"] == binding["gpu_uuid"] and memory["assigned_gpu_process_observation"]["pid"] == metadata["after"]["identity"]["pid"],
                    "measured process memory belongs to another physical worker")
                window = memory["assigned_gpu_process_observation"]
                require(pair[arm]["iterations"] == plan["iterations"]
                    and 0 < pair[arm]["seconds"] <= window["window_finished_at_seconds"] - window["window_started_at_seconds"] + .001,
                    "the sampled window does not cover the actual timed round")
    if plan.get("kind") == "selected_configuration_profile":
        memory = result["profile_memory"]
        rebuilt = validate_window(memory["assigned_gpu_process_observation"], directory)
        require(all(memory.get(key) == value for key, value in rebuilt.items())
            and result.get("gpu_uuid") == binding["gpu_uuid"] and memory["gpu_uuid"] == binding["gpu_uuid"]
            and memory["assigned_gpu_process_observation"]["pid"] == metadata["after"]["identity"]["pid"],
            "the complete profile memory window changed its actual child")
    validate_startup_checks(directory, binding)
    validate_backend_capture(directory, binding, result["loaded_gpu_libraries"])
    if plan.get("kind") == "selected_configuration_measurement":
        memory = result["direct_whole_work_memory"]
        rebuilt = validate_window(memory["assigned_gpu_process_observation"], directory)
        require(all(memory.get(key) == value for key, value in rebuilt.items()), "direct whole-work memory changed")
        terminal_process(directory, metadata)
    else:
        validate_profile_terminal(directory, metadata, result)
    if "common_correctness_inputs" in plan:
        adjacent("direct_common_confirmation").verify_saved(directory, plan, result)
    return plan, binding


def validate_startup_checks(directory, binding):
    identity = core(); directory = Path(directory)
    value = identity.read_receipt(identity.receipt(directory / "direct_startup_checks.json"))
    require(type(value.get("schema_version")) is int and value["schema_version"] == 1
        and value.get("kind") == "direct_bound_h100_startup_checks" and value.get("passed") is True
        and value.get("binding") == identity.receipt(directory / "direct_device_binding.json")
        and value.get("pre_context_occupancy_observed") is False and value.get("startup_memory_observed") is False
        and all(identity.exact(value.get(key), expected) for key, expected in BOUNDARY.items()),
        "direct startup checks changed their target or claim scope")
    gpu = binding["gpu_uuid"]
    for name, command in (("mode_query", ["nvidia-smi", "-i", gpu, "--query-gpu=uuid,mig.mode.current,memory.used", "--format=csv,noheader,nounits"]),
            ("confidential_computing_query", ["nvidia-smi", "conf-compute", "-f", "-i", gpu])):
        row = value[name]
        require(row["command"] == command and type(row["returncode"]) is int and row["returncode"] == 0
            and row["stderr"] == "" and row["stdout_sha256"] == identity.digest(row["stdout"].encode())
            and len(row["stdout"].encode()) <= shared().QUERY_LIMIT, "a saved direct startup query changed")
    rows = list(csv.reader(value["mode_query"]["stdout"].splitlines(), strict=True))
    require(len(rows) == 1 and len(rows[0]) == 3 and rows[0][0].strip() == gpu and rows[0][1].strip() == "Disabled"
        and re.fullmatch(r"(?:0|[1-9][0-9]*)", rows[0][2].strip()) and int(rows[0][2].strip()) < 1024
        and "CC status: OFF" in value["confidential_computing_query"]["stdout"],
        "saved direct startup memory or GPU modes differ from the unchanged limits")
    validate_process_sample(value["process_query"], gpu, binding["process"]["pid"])
    return value


def validate_backend_capture(directory, binding, loaded):
    """Require new schema-two planned/actual backend proof and refuse legacy data."""
    identity = core()
    directory = Path(directory)
    plan = identity.read_receipt(identity.receipt(directory / "measurement_plan.json"))
    filename = "direct_qualification_result.json" if "direct_qualification" in plan else "result.json"
    result = identity.read_receipt(identity.receipt(directory / filename))
    require(result.get("loaded_gpu_libraries") == loaded, "actual backend validation changed its separate CUDA receipt")
    return backend_support(plan).validate_capture(directory, plan, binding, loaded, result)



def validate_profile_terminal(directory, metadata, result):
    identity = core(); directory = Path(directory); process = metadata["after"]["identity"]
    if metadata.get("role") == "producer_guarded_child":
        return adjacent("direct_profile_process").validate_profile_terminal(directory, metadata, result, sys.modules[__name__])
    value = identity.read_receipt(identity.receipt(directory / "direct_profile_terminal.json"))
    owner = identity.read_receipt(identity.receipt(directory / "profiler_process.json"))
    plan = identity.read_receipt(identity.receipt(directory / "measurement_plan.json"))
    child = identity.read_receipt(identity.receipt(directory / "capture_identity.json"))
    release = identity.read_receipt(identity.receipt(directory / "capture_release.json"))
    capture = identity.read_receipt(identity.receipt(directory / "capture_result.json"))
    validate_capture_wait(directory, metadata)
    require(type(value.get("schema_version")) is int and value["schema_version"] == 1
        and value.get("kind") == "direct_retained_descriptor_profile_terminal" and value.get("profiler_process") == owner
        and value.get("capture_identity") == child and value.get("command") == owner["command"]
        and owner["command"] == adjacent("profile_selected").child_command(directory, plan, metadata["lock_before"]["fd"])
        and type(value.get("returncode")) is int and value["returncode"] == 0
        and child == {key: process[key] for key in ("hostname", "pid", "process_start_time")}
        and release == {**owner, "capture_identity": child}
        and value["parent_custody"] == identity.receipt(directory / "supervisor/direct_process_metadata.json")
        and value["capture_release"] == identity.receipt(directory / "capture_release.json")
        and value["capture_wait"] == identity.receipt(directory / "direct_capture_wait_return.json")
        and value["capture_result"] == identity.receipt(directory / "capture_result.json")
        and metadata["role"] == "inherited" and metadata["profiler_process"] == owner
        and capture.get("capture_completed") is True and result.get("completed") is True,
        "the actual profiling wrapper,child and retained owner-custody terminal chain changed")
    require(result.get("trace_analysis_record") == str(directory / "trace/analysis.json")
        and result.get("trace_report_sha256") == identity.receipt(directory / "trace/capture.nsys-rep")["sha256"],
        "the actual positive profiling report changed")
    analysis = identity.read_receipt(identity.receipt(directory / "trace/analysis.json"))
    diagnostic = adjacent("diagnose_gpu_execution")
    for name, column in (("cuda_gpu_kern_sum", "Instances"), ("cuda_api_sum", "Num Calls")):
        summary = analysis["summaries"][name]
        require(diagnostic.summary_csv_evidence(summary["path"], column) == {key: summary[key] for key in ("rows", "columns", "sha256")},
            "a positive actual kernel or CUDA API summary changed")
    require(result.get("direct_trace_process_proof") == trace_process_proof(directory, metadata, publish_result=False),
        "the positive trace was not joined to this actual waited capture child")
    return value
