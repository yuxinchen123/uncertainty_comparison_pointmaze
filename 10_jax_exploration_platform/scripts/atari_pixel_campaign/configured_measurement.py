"""Explicit single-card producer support; unresolved plans never become ordinary evidence."""

from contextlib import contextmanager
import ast
from collections import Counter
import csv
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import selectors
import subprocess
import sys
import threading
import time


CAPABILITY = "assigned_uuid_own_process_complete_window_v1"
DIRECT_OBSERVATION_CAPABILITY = "direct_assigned_uuid_own_process_complete_window_v1"
QUERY_LIMIT = 1024 * 1024
INTERVAL = .2
BOUNDARY = {"physical_gres_mapping_verified": False, "kernel_device_enforcement_verified": False,
            "installed_binary_matches_upstream_verified": False, "worker_admitted": False}
QUALIFICATION = {"schema_version": 1, "kind": "full_original_target_qualification",
    "reference_updates": 1, "normalization_warmup_rollouts": 50,
    "scientific_credit": 0, "case_credit": 0, "idea_credit": 0, "admission_granted": False}
EXPECTED_DTYPES = {"parameters": ["float32"], "target": ["float32"], "optimizer_first": ["float32"],
    "optimizer_second": ["float32"], "observation_statistics": ["float64"], "reward_statistics": ["float64"]}


def require(condition, message):
    """Refuse incomplete or contradictory producer evidence."""
    if not condition:
        raise ValueError(message)


def sha(raw):
    """Bind a retained byte sequence to its exact digest."""
    return hashlib.sha256(raw).hexdigest()


def core():
    """Import only the adjacent, separately reviewed configuration qualifier."""
    path = Path(__file__).with_name("uva_slurm_configured_isolation.py")
    specification = importlib.util.spec_from_file_location("configured_measurement_identity", path)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def direct_support():
    """Load the separate direct policy only for an explicit direct binding."""
    path = Path(__file__).with_name("direct_measurement.py").resolve(strict=True)
    name = "direct_measurement"
    if name in sys.modules:
        module = sys.modules[name]
        require(Path(module.__file__).resolve(strict=True) == path, "the direct adapter must be this exact staged sibling")
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


def qualification_identity(plan):
    """Share original gate mechanics while keeping target policies separate."""
    if "direct_binding" in plan:
        require("configured_isolation" not in plan, "direct and UVA qualification policies cannot be mixed")
        identity = direct_support().core()
        identity.validate_plan(plan)
        return identity
    return core()


def helper_names():
    """Stage the exact direct identity closure only for an explicit opt-in plan."""
    return tuple(dict.fromkeys(("configured_measurement.py", *core().HELPERS)))


def configure_plan(plan, profile, consumer):
    """Keep a null physical identity and a future actual-submission input path in the frozen plan."""
    identity = core()
    identity.require_profile(profile)
    plan["configured_isolation"] = {"schema_version": 1, "profile": profile,
        "capability": identity.CAPABILITY, "consumer": consumer,
        "resources": {"cpus": 8, "host_memory_gib": 64},
        "physical_identity": identity.unresolved_identity(), "owned_inputs_path": str(
            Path(plan["campaign_run_dir"]) / "data" / plan["unit_id"] / "owned_allocation_inputs.json")}
    identity.validate_plan(plan, consumer)
    return plan


def bound_hardware(plan, metadata, binding):
    """Describe the actual child without changing its submitted selector or unresolved plan."""
    identity = core()
    identity.validate_binding(plan, metadata, binding)
    row = next(csv.reader(binding["hardware"]["stdout"].splitlines()))
    gpu, name, unused_bus, memory, driver = [part.strip() for part in row]
    return {"hostname": metadata["process"]["hostname"], "gpu_uuid": gpu,
        "gpu_query": f"{gpu}, {name}, {memory} MiB, {driver}", "pid": metadata["process"]["pid"],
        "process_start_time": metadata["process"]["process_start_time"],
        "slurm_job_id": metadata["environment"]["SLURM_JOB_ID"],
        "cuda_visible_devices_original": metadata["environment"]["CUDA_VISIBLE_DEVICES"],
        "cuda_visible_devices": metadata["environment"]["CUDA_VISIBLE_DEVICES"], **BOUNDARY}


def numerical_module_paths(plan):
    """Check every actually imported pixel module against the frozen core before creating a learner."""
    root = Path(plan["source_root"])
    directory = root / "10_jax_exploration_platform/src/exploration_platform/atari_pixels"
    observed = {}
    for name, module in list(sys.modules.items()):
        if name == "exploration_platform.atari_pixels" or name.startswith("exploration_platform.atari_pixels."):
            path = Path(module.__file__)
            require(path.resolve(strict=True) == path and path.is_relative_to(directory),
                "an imported numerical module came from another source")
            expected = plan["core"]["files"].get(str(path.relative_to(root)))
            require(isinstance(expected, dict) and sha(path.read_bytes()) == expected["sha256"],
                "an actually imported numerical module changed its recorded source bytes")
            observed[name] = str(path)
    require({"exploration_platform.atari_pixels.config", "exploration_platform.atari_pixels.environment",
        "exploration_platform.atari_pixels.training"} <= set(observed), "actual learner/environment imports are incomplete")
    return observed


def bounded_query(command, timeout=10):
    """Capture a bounded complete query; retain no foreign process text after parsing fails."""
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    selector = selectors.DefaultSelector()
    output = {"stdout": bytearray(), "stderr": bytearray()}
    for name in output:
        stream = getattr(process, name)
        os.set_blocking(stream.fileno(), False)
        selector.register(stream, selectors.EVENT_READ, name)
    deadline = time.monotonic() + timeout
    error = None
    try:
        # Bound the pipes while reading, rather than after an unlimited subprocess capture.
        while selector.get_map():
            if time.monotonic() >= deadline:
                raise TimeoutError("assigned-device query timed out")
            for key, unused in selector.select(min(.1, max(0, deadline - time.monotonic()))):
                raw = os.read(key.fileobj.fileno(), 65536)
                if not raw:
                    selector.unregister(key.fileobj)
                    continue
                output[key.data].extend(raw)
                if sum(map(len, output.values())) > QUERY_LIMIT:
                    raise ValueError("assigned-device query exceeded its bounded output")
        process.wait(timeout=max(.001, deadline - time.monotonic()))
    except Exception as caught:
        error = type(caught).__name__
        process.kill()
        process.wait()
    finally:
        selector.close()
        process.stdout.close()
        process.stderr.close()
    return {"returncode": process.returncode, "stdout": bytes(output["stdout"]),
        "stderr": bytes(output["stderr"]), "capture_error": error, "complete": error is None}


def process_sample(gpu, pid, query=bounded_query):
    """Accept complete own-only rows; save only anonymous failure facts for any other result."""
    require(type(pid) is int and pid > 0 and re.fullmatch(r"GPU-[0-9a-f-]{36}", gpu),
        "process sampling requires an actual bound physical identity")
    command = ["nvidia-smi", "-i", gpu, "--query-compute-apps=gpu_uuid,pid,used_gpu_memory",
               "--format=csv,noheader,nounits"]
    start = time.time()
    try:
        raw = query(command)
    except Exception as error:
        raw = {"returncode": None, "stdout": b"", "stderr": b"", "capture_error": type(error).__name__, "complete": False}
    result = {"command": command, "started_at_seconds": start, "finished_at_seconds": time.time(),
        "gpu_uuid": gpu, "own_pid": pid, "returncode": raw["returncode"],
        "stdout_sha256": sha(raw["stdout"]), "stderr_sha256": sha(raw["stderr"]),
        "stdout_bytes": len(raw["stdout"]), "stderr_bytes": len(raw["stderr"]),
        "complete": raw["complete"], "capture_error": raw["capture_error"], "valid": False}
    try:
        require(raw["complete"] is True and raw["capture_error"] is None and type(raw["returncode"]) is int
            and raw["returncode"] == 0 and raw["stderr"] == b""
            and len(raw["stdout"]) + len(raw["stderr"]) <= QUERY_LIMIT, "query_failed_or_incomplete")
        text = raw["stdout"].decode("utf-8", errors="strict")
        rows = list(csv.reader(text.splitlines(), strict=True))
        typed, seen = [], set()
        # Before: UUID, pid, MiB text; after: one typed current-PID byte count, or anonymous rejection.
        for row in rows:
            require(len(row) == 3, "malformed_process_row")
            uuid, number, memory = [part.strip() for part in row]
            require(uuid == gpu and re.fullmatch(r"[1-9][0-9]*", number)
                and re.fullmatch(r"(?:0|[1-9][0-9]*)", memory), "invalid_process_row")
            number = int(number)
            require(number not in seen, "duplicate_process_row")
            seen.add(number)
            typed.append({"pid": number, "gpu_uuid": uuid, "bytes": int(memory) * 2 ** 20})
        foreign = sum(row["pid"] != pid for row in typed)
        result["other_active_compute_rows"] = foreign
        require(foreign == 0, "other_active_compute_rows_present")
        require(len(typed) == 1 and typed[0]["pid"] == pid and typed[0]["bytes"] > 0,
            "own_process_memory_unavailable")
        result.update(valid=True, stdout=text, stderr="", own=typed[0], row_count=1)
    except (ValueError, UnicodeError, csv.Error) as error:
        result["failure_category"] = str(error) if str(error) in {
            "query_failed_or_incomplete", "malformed_process_row", "invalid_process_row", "duplicate_process_row",
            "other_active_compute_rows_present", "own_process_memory_unavailable"} else "invalid_process_capture"
    return result


def validate_process_sample(value, gpu, pid):
    """Rebuild successful raw own-only evidence without running a query."""
    require(value.get("valid") is True and value.get("gpu_uuid") == gpu and value.get("own_pid") == pid,
        "missing complete own-only process observation")
    rebuilt = process_sample(gpu, pid, lambda command: {"returncode": value["returncode"],
        "stdout": value["stdout"].encode(), "stderr": value["stderr"].encode(),
        "complete": value["complete"], "capture_error": value["capture_error"]})
    for key in rebuilt:
        if key not in {"started_at_seconds", "finished_at_seconds"}:
            require(value.get(key) == rebuilt[key], "saved process query or typed memory changed")
    require(type(value["started_at_seconds"]) in {int, float}
        and type(value["finished_at_seconds"]) in {int, float}
        and math.isfinite(value["started_at_seconds"]) and math.isfinite(value["finished_at_seconds"])
        and 0 < value["started_at_seconds"] <= value["finished_at_seconds"], "invalid query interval")
    return value["own"]["bytes"]


class MemoryObservation:
    """Retain successful boundary/intermediate assigned-GPU samples and normal sampler shutdown."""

    def __init__(self, device, binding, directory):
        """Require the current bound worker before making any process query."""
        self.policy = "uva_a4500"
        if binding.get("kind") == "serval02_card1_current_process_resolved_device_binding":
            self.direct = direct_support()
            self.direct_inputs = self.direct.observer_inputs(binding, directory)
            self.policy = "direct_serval02_card_one"
            self.device, self.binding, self.directory = device, binding, Path(directory)
            self.uuid, self.pid = binding["gpu_uuid"], os.getpid()
            self.stop, self.thread, self.counter = threading.Event(), None, 0
            return
        require(binding.get("kind") == "uva_current_process_resolved_device_binding"
            and binding["process"]["pid"] == os.getpid() and binding["gpu_uuid"] == binding["native"]["gpu_uuid"],
            "memory observation requires this actual resolved child")
        identity = core()
        directory = Path(directory)
        require(directory.name in {"paired_memory", "qualification_memory", "profile_memory"},
            "memory evidence must use its reserved child-unit directory")
        unit = directory.parent
        plan = identity.read_receipt(identity.receipt(unit / "measurement_plan.json"))
        metadata = identity.read_receipt(identity.receipt(unit / "configured_isolation_metadata.json"))
        saved = identity.read_receipt(identity.receipt(unit / "resolved_device_binding.json"))
        require(identity.exact(saved, binding), "sampling cannot substitute a different or remembered physical binding")
        identity.validate_binding(plan, metadata, binding, unit)
        self.device, self.binding, self.directory = device, binding, Path(directory)
        self.uuid, self.pid = binding["gpu_uuid"], os.getpid()
        self.stop, self.thread, self.counter = threading.Event(), None, 0

    def sample(self):
        """Preserve anonymous capture facts for every failed device query or parsing branch."""
        if getattr(self, "policy", "uva_a4500") == "direct_serval02_card_one":
            self.direct.core().live_custody(self.direct_inputs["metadata"], self.direct_inputs["metadata"]["lock_after"]["fd"])
        birth = Path("/proc/self/stat").read_text().rsplit(")", 1)[1].split()[19]
        require(os.getpid() == self.pid and birth == self.binding["process"]["process_start_time"]
            and all(os.environ.get(key) == value for key, value in self.binding["original_environment"].items()),
            "the bound process birth or original scheduler/CUDA selector changed")
        command = ["nvidia-smi", "-i", self.uuid, "--query-gpu=uuid,memory.total,memory.used,memory.free",
                   "--format=csv,noheader,nounits"]
        query_started = time.time()
        raw = {"returncode": None, "stdout": b"", "stderr": b"", "complete": False, "capture_error": "not_completed"}
        try:
            raw = bounded_query(command)
            query_finished = time.time()
            require(raw["complete"] is True and raw["capture_error"] is None
                and type(raw["returncode"]) is int and raw["returncode"] == 0 and raw["stderr"] == b""
                and len(raw["stdout"]) + len(raw["stderr"]) <= QUERY_LIMIT, "device query failed")
            text = raw["stdout"].decode("utf-8", errors="strict")
            rows = list(csv.reader(text.splitlines(), strict=True))
            require(len(rows) == 1 and len(rows[0]) == 4 and rows[0][0].strip() == self.uuid,
                "device query changed physical identity")
            require(all(re.fullmatch(r"(?:0|[1-9][0-9]*)", value.strip()) for value in rows[0][1:]),
                "device memory counters are unavailable")
            amounts = [int(value.strip()) * 2 ** 20 for value in rows[0][1:]]
            expected_total = 95830 * 2 ** 20 if getattr(self, "policy", "uva_a4500") == "direct_serval02_card_one" else 20470 * 2 ** 20
            require(amounts[0] == expected_total and all(0 <= value <= amounts[0] for value in amounts[1:]),
                "device memory counters exceed the actual physical total")
        except Exception as error:
            self.directory.mkdir(parents=True, exist_ok=True)
            path = self.directory / ("device_query_failure_" + str(time.time_ns()) + ".json")
            # Never retain a wrong-device or malformed raw row: preserve only bounded hashes/counts/times and failure type.
            with path.open("x") as stream:
                json.dump({"command": command, "returncode": raw["returncode"], "capture_error": raw["capture_error"],
                    "stdout_sha256": sha(raw["stdout"]), "stderr_sha256": sha(raw["stderr"]),
                    "stdout_bytes": len(raw["stdout"]), "stderr_bytes": len(raw["stderr"]),
                    "started_at_seconds": query_started, "finished_at_seconds": time.time(),
                    "error_type": type(error).__name__, "measurement_valid": False}, stream)
                stream.flush(); os.fsync(stream.fileno())
            raise ValueError("the assigned-device memory query or its raw counters failed validation") from None
        process = process_sample(self.uuid, self.pid)
        return {"timestamp_seconds": time.time(), "gpu_uuid": self.uuid,
            "total_bytes": amounts[0], "used_bytes": amounts[1], "free_bytes": amounts[2],
            "device_query": {"command": command, "stdout": text, "sha256": sha(raw["stdout"]),
                "returncode": raw["returncode"], "stderr": "", "complete": raw["complete"], "capture_error": raw["capture_error"],
                "started_at_seconds": query_started, "finished_at_seconds": query_finished},
            "process_query": process}

    def _sample(self):
        """Record every ordinary failure and distinguish requested stops from unexpected thread exits."""
        try:
            while not self.stop.wait(INTERVAL):
                self.samples.append(self.sample())
            self.normal_stop = True
        except Exception as error:
            self.errors.append(type(error).__name__ + ": " + str(error))

    def begin(self):
        """Capture the successful initial boundary before the measured interval begins."""
        require(self.thread is None or not self.thread.is_alive(), "previous sampler is still active")
        if getattr(self, "policy", "uva_a4500") == "direct_serval02_card_one":
            try:
                self.direct_before = self.direct.window_boundary(self.direct_inputs)
            except BaseException as error:
                self.direct.record_window_failure(self.directory, self.counter, "begin", error)
                raise
        self.samples, self.errors, self.normal_stop = [], [], False
        self.stop.clear()
        self.samples.append(self.sample())
        try:
            validate_process_sample(self.samples[0]["process_query"], self.uuid, self.pid)
        except (ValueError, KeyError):
            self.directory.mkdir(parents=True, exist_ok=True)
            with (self.directory / ("initial_process_sample_failure_" + str(self.counter) + ".json")).open("x") as stream:
                json.dump(self.samples[0], stream, indent=2, allow_nan=False)
            raise
        self.thread = threading.Thread(target=self._sample, daemon=True)
        self.thread.start()
        self.window_started = time.time()

    def finish(self):
        """Save complete raw coverage before refusing any failed or contaminated measurement."""
        numerical_error = sys.exc_info()[1]
        ended = time.time()
        alive = self.thread is not None and self.thread.is_alive()
        self.stop.set()
        if self.thread is not None:
            self.thread.join(timeout=25)
        try:
            self.samples.append(self.sample())
        except Exception as error:
            self.errors.append(type(error).__name__ + ": " + str(error))
        direct = getattr(self, "policy", "uva_a4500") == "direct_serval02_card_one"
        after = None
        if direct:
            try:
                after = self.direct.window_boundary(self.direct_inputs)
            except BaseException as error:
                self.errors.append("direct source or lock boundary failed: " + type(error).__name__)
                self.direct.record_window_failure(self.directory, self.counter, "finish", error)
        proof = {"schema_version": 1, "kind": DIRECT_OBSERVATION_CAPABILITY if direct else CAPABILITY, "gpu_uuid": self.uuid, "pid": self.pid,
            "window_started_at_seconds": self.window_started, "window_finished_at_seconds": ended,
            "interval_seconds": INTERVAL, "samples": self.samples, "errors": self.errors,
            "alive_at_requested_stop": alive, "normal_requested_stop": self.normal_stop,
            "joined": self.thread is not None and not self.thread.is_alive()}
        if direct:
            proof.update(direct_binding=self.direct_inputs["binding_ref"], direct_before=self.direct_before,
                direct_after=after, startup_memory_observed=False)
        allocator_error = None
        try:
            allocator = self.device.memory_stats()
            proof["allocator"] = {key: int(value) for key, value in allocator.items()} if allocator is not None else None
        except Exception as error:
            allocator_error = error
            proof["allocator"] = None
            proof["allocator_error"] = {"type": type(error).__name__, "observation": "unknown"}
            self.errors.append("allocator observation failed: " + type(error).__name__)
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self.directory / ("memory_window_" + str(self.counter) + ".json")
        self.counter += 1
        with path.open("x") as stream:
            json.dump(proof, stream, indent=2, allow_nan=False)
            stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
        try:
            return self.direct.validate_window(proof, self.directory.parent) if direct else validate_window(proof)
        except BaseException:
            if numerical_error is None:
                if allocator_error is not None:
                    raise RuntimeError("allocator observation failed after its raw lifecycle proof was saved") from allocator_error
                raise
            # The frozen round calls finish in finally: preserve its original failed update and retain unknown memory.
            return {"gpu_uuid": self.uuid, "measurement_valid": False, "other_gpu_processes": None,
                "other_gpu_processes_observed": False, "sampled_process_peak_bytes": None,
                "assigned_gpu_process_observation": proof}


def validate_window(proof):
    """Recompute real own-process peak and complete boundaries without weakening old round gates."""
    return _validate_window_for_policy(proof, "uva_a4500")


def _validate_window_for_policy(proof, policy):
    """Use only the two reviewed literal policies, never a caller-supplied total."""
    require(policy in {"uva_a4500", "direct_serval02_card_one"}, "unsupported assigned-device window policy")
    capability = CAPABILITY if policy == "uva_a4500" else DIRECT_OBSERVATION_CAPABILITY
    total = (20470 if policy == "uva_a4500" else 95830) * 2 ** 20
    require(proof.get("kind") == capability and proof.get("schema_version") == 1
        and type(proof["schema_version"]) is int and not proof["errors"] and proof["alive_at_requested_stop"] is True
        and proof["normal_requested_stop"] is True and proof["joined"] is True, "sampler lifecycle was incomplete")
    samples = proof["samples"]
    start, end = proof["window_started_at_seconds"], proof["window_finished_at_seconds"]
    require(proof["interval_seconds"] == INTERVAL and all(type(x) in {int, float} and math.isfinite(x)
        for x in [start, end, *(sample["timestamp_seconds"] for sample in samples)])
        and len(samples) >= 2 and samples[0]["timestamp_seconds"] <= start <= end <= samples[-1]["timestamp_seconds"]
        and all(a["timestamp_seconds"] <= b["timestamp_seconds"] for a, b in zip(samples, samples[1:])),
        "sampling did not cover both measured boundaries")
    require(samples[0]["process_query"]["finished_at_seconds"] <= start
        and samples[-1]["process_query"]["started_at_seconds"] >= end
        and samples[-1]["device_query"]["started_at_seconds"] >= end
        and all(a["process_query"]["finished_at_seconds"] <= b["process_query"]["started_at_seconds"]
                for a, b in zip(samples, samples[1:])), "raw process queries do not cover the declared window")
    peaks = []
    for sample in samples:
        # Keep physical device counters separate from the sampled own process and allocator peaks.
        require(sample["gpu_uuid"] == proof["gpu_uuid"] and sample["total_bytes"] == total
            and type(sample["used_bytes"]) is int and type(sample["free_bytes"]) is int
            and 0 <= sample["used_bytes"] <= sample["total_bytes"] and 0 <= sample["free_bytes"] <= sample["total_bytes"],
            "saved device counters changed")
        query = sample["device_query"]
        require(query["command"] == ["nvidia-smi", "-i", proof["gpu_uuid"],
            "--query-gpu=uuid,memory.total,memory.used,memory.free", "--format=csv,noheader,nounits"]
            and query["sha256"] == sha(query["stdout"].encode()) and type(query["returncode"]) is int and query["returncode"] == 0
            and query["stderr"] == "" and query["complete"] is True and query["capture_error"] is None,
            "saved physical query changed")
        require(0 < query["started_at_seconds"] <= query["finished_at_seconds"] <= sample["process_query"]["started_at_seconds"]
            <= sample["process_query"]["finished_at_seconds"] <= sample["timestamp_seconds"], "sample query intervals changed")
        row = list(csv.reader(query["stdout"].splitlines()))
        require(len(row) == 1 and len(row[0]) == 4 and row[0][0].strip() == proof["gpu_uuid"]
            and [int(x.strip()) * 2 ** 20 for x in row[0][1:]] ==
                [sample["total_bytes"], sample["used_bytes"], sample["free_bytes"]], "raw device counters differ")
        amount = validate_process_sample(sample["process_query"], proof["gpu_uuid"], proof["pid"])
        require(amount <= sample["total_bytes"], "own process memory exceeds the device total")
        peaks.append(amount)
    allocator = proof.get("allocator")
    require(isinstance(allocator, dict) and type(allocator.get("peak_bytes_in_use")) is int
        and 0 < allocator["peak_bytes_in_use"] <= total, "actual lifetime allocator peak is unavailable")
    if "bytes_in_use" in allocator:
        require(type(allocator["bytes_in_use"]) is int and 0 <= allocator["bytes_in_use"] <= allocator["peak_bytes_in_use"],
            "actual lifetime allocator counters are inconsistent")
    return {"gpu_uuid": proof["gpu_uuid"], "sampled_process_peak_bytes": max(peaks),
        "process_memory_samples": len(samples), "sampling_interval_seconds": proof["interval_seconds"],
        "other_gpu_processes": [], "other_gpu_processes_observed": True,
        "device_memory_samples": [{k: v for k, v in sample.items() if k != "process_query"} for sample in samples],
        "assigned_gpu_process_observation": proof,
        "allocator": allocator,
        "actual_sample_gaps_seconds": [b["timestamp_seconds"] - a["timestamp_seconds"] for a, b in zip(samples, samples[1:])],
        "peak_scope": "own active compute process samples on the assigned UUID; lifetime allocator peak is separate"}


@contextmanager
def paired_observer(paired, binding, directory):
    """Replace only the frozen paired observer for this explicit worker and restore it on every exit."""
    original = paired.MemoryObservation
    observers = []
    def observer(device):
        """Use the already bound UUID, without the frozen observer's global discovery query."""
        value = MemoryObservation(device, binding, Path(directory) / "paired_memory")
        observers.append(value)
        return value
    paired.MemoryObservation = observer
    original_error = None
    try:
        yield
    except BaseException as error:
        original_error = error
        raise
    finally:
        paired.MemoryObservation = original
        for value in observers:
            if value.thread is not None and value.thread.is_alive():
                try:
                    value.finish()
                except BaseException:
                    if original_error is None:
                        raise


@contextmanager
def observed_work(device, binding, directory):
    """Capture a complete qualification/profile window and preserve an original numerical failure."""
    value = MemoryObservation(device, binding, directory)
    result = {}
    original_error = None
    value.begin()
    try:
        yield result
    except BaseException as error:
        original_error = error
        raise
    finally:
        try:
            result.update(value.finish())
        except BaseException as error:
            if original_error is None:
                raise
            # Raw samples are already retained by finish; keep the original failed numerical call as the main error.
            path = Path(directory) / "sampling_failure_after_numerical_failure.json"
            with path.open("x") as stream:
                json.dump({"numerical_error_type": type(original_error).__name__, "sampling_error_type": type(error).__name__,
                    "measurement_valid": False}, stream)


def validate_original_qualification(plan):
    """Require the full original source/configuration while leaving ordinary equal-arm rejection intact."""
    import diagnose_gpu_execution as diagnostic
    import post_comparison_validation as validation
    from_source = diagnostic.default_configuration(Path(plan["source_root"]))
    value = validation.configuration(plan["configuration"])
    require(plan["source_commit"] == "d28e6735e79788f618acb0ebfde8609db1f776af"
        and plan["mode"] == "correctness" and value == plan["reference_configuration"],
        "target qualification must compare the same original source and full configuration")
    require(value["learner"] == json.loads(json.dumps(from_source["learner"])) and value["environment"] == json.loads(json.dumps(from_source["environment"]))
        and value["learner"]["n_copies"] == 1 and value["learner"]["n_envs"] == 128
        and value["learner"]["num_steps"] == 128 and value["learner"]["update_epochs"] == 4
        and value["learner"]["num_minibatches"] == 4 and value["learner"]["observation_warmup_iterations"] == 50
        and value["learner"]["total_timesteps"] == 2000000000,
        "qualification cannot use a tiny or optimized learner/environment")
    require(value["execution"] == {"compilation_mode": "whole_iteration", "iterations_per_call": 1,
        "report_readout": "none", "report_every_iterations": 10},
        "qualification uses one original ungrouped full reference update")
    return value


def qualification_source_paths(root, commit):
    """Read the complete committed original gate roster rather than trusting a recorded subset."""
    scopes = ["10_jax_exploration_platform/tests/atari_pixels",
        "10_jax_exploration_platform/tests/test_atari_pixel_environment.py",
        "10_jax_exploration_platform/scripts/atari_pixel_campaign/check_rendering.py"]
    listing = subprocess.check_output(["git", "-C", str(root), "ls-tree", "-r", "--name-only", commit, "--", *scopes], text=True)
    paths = listing.splitlines()
    require(paths and all(path.endswith(".py") for path in paths), "full original gate source roster is unavailable")
    return paths


def renderer_contract(plan, backend):
    """Derive the complete indexed-room and synthetic matrices from exact original source bytes."""
    identity = qualification_identity(plan)
    renderer_relative = "10_jax_exploration_platform/scripts/atari_pixel_campaign/check_rendering.py"
    renderer_path = Path(plan["source_root"]) / renderer_relative
    source = backend["jaxtari_source"]
    constants_relative = "jaxtari/games/montezuma_revenge/core.py"
    constants_path = Path(source["source_root"]) / "src" / constants_relative
    require(plan["source_commit"] == "d28e6735e79788f618acb0ebfde8609db1f776af"
        and source["source_commit"] == "04ab9aec318c72162e2e53503a594a489d7e162b"
        and identity.receipt(renderer_path)["sha256"] == plan["qualification_sources"]["files"][renderer_relative]
        and identity.receipt(constants_path)["sha256"] == source["python_files"][constants_relative]["sha256"],
        "the original renderer or actual JAXtari constants changed")
    constants = ast.parse(constants_path.read_text())
    classes = [node for node in constants.body if isinstance(node, ast.ClassDef) and node.name == "MontezumaRevengeConstants"]
    require(len(classes) == 1, "the original room constants class is ambiguous")
    room_fields = [node for node in classes[0].body if isinstance(node, ast.AnnAssign)
        and isinstance(node.target, ast.Name) and node.target.id == "MAX_ROOMS"]
    require(len(room_fields) == 1 and isinstance(room_fields[0].value, ast.Call), "the original indexed room count is absent")
    defaults = [keyword.value for keyword in room_fields[0].value.keywords if keyword.arg == "default"]
    require(len(defaults) == 1, "the indexed room default is ambiguous")
    rooms = ast.literal_eval(defaults[0])
    functions = {node.name: node for node in ast.parse(renderer_path.read_text()).body if isinstance(node, ast.FunctionDef)}
    def same_scope_statements(nodes):
        """Visit statement blocks in this function while excluding nested function/class scopes."""
        for node in nodes:
            yield node
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue
            for field in ("body", "orelse", "finalbody"):
                children = getattr(node, field, None)
                if isinstance(children, list) and all(isinstance(child, ast.stmt) for child in children):
                    yield from same_scope_statements(children)
            if isinstance(node, ast.Try):
                for handler in node.handlers:
                    yield from same_scope_statements(handler.body)
    def assignment(function, name):
        """Read one literal source assignment without executing JAX or selecting a smaller case matrix."""
        values = [node.value for node in same_scope_statements(functions[function].body) if isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == name for target in node.targets)]
        require(len(values) == 1 and isinstance(values[0], ast.Call), "the original renderer fixture assignment is ambiguous")
        return values[0]
    repeat = assignment("fixture_states", "room_ids")
    require(len(repeat.args) == 2, "the original renderer room variants changed")
    variants = ast.literal_eval(repeat.args[1])
    colors, patterns = assignment("check_synthetic_pixels", "colors"), assignment("check_synthetic_pixels", "patterns")
    require(colors.args and patterns.args and isinstance(colors.args[0], (ast.List, ast.Tuple))
        and isinstance(patterns.args[0], (ast.List, ast.Tuple)), "the complete original synthetic fixtures are absent")
    synthetic = len(colors.args[0].elts) + len(patterns.args[0].elts)
    require(type(rooms) is int and rooms > 0 and type(variants) is int and variants > 0 and synthetic > 0,
        "the original renderer matrices must contain every declared case")
    return {"schema_version": 1, "kind": "pinned_original_full_renderer_matrix",
        "renderer_source": identity.receipt(renderer_path), "jaxtari_constants_source": identity.receipt(constants_path),
        "jaxtari_source_commit": source["source_commit"], "rooms": rooms, "variants_per_room": variants,
        "cases": rooms * variants, "synthetic_pixel_cases": synthetic}


def qualification_sources(plan, cleanrl_root, software_result):
    """Bind every unchanged original gate and external CleanRL model before any numerical import."""
    import train_optimized as training
    identity = qualification_identity(plan).adjacent("h100_nvl_slurm_site_validation")
    root = Path(plan["source_root"])
    paths = qualification_source_paths(root, plan["source_commit"])
    plan["qualification_sources"] = training.source_receipt(root, plan["source_commit"], paths)
    require(cleanrl_root is not None, "the target qualification requires the exact original CleanRL checkout")
    plan["cleanrl_reference"] = identity.cleanrl_identity(cleanrl_root)
    require(software_result is not None, "the actual completed UVA shared-software packet is required")
    software_result = Path(software_result)
    plan["qualification_software"] = identity.software_reference(software_result,
        software_result.with_name("installed_backend.json"), software_result.with_name("asset_receipt.json"))
    plan["renderer_contract"] = renderer_contract(plan, qualification_identity(plan).read_receipt(plan["qualification_software"]["installed_backend"]))


def verify_qualification_source_receipts(plan):
    """Rebuild original test, external model and actual software receipts without changing process state."""
    import train_optimized as training
    identity = qualification_identity(plan).adjacent("h100_nvl_slurm_site_validation")
    recorded = plan["qualification_sources"]
    require(training.source_receipt(Path(plan["source_root"]), plan["source_commit"],
        qualification_source_paths(Path(plan["source_root"]), plan["source_commit"])) == recorded,
        "an original numerical or renderer gate changed")
    require(identity.cleanrl_identity(plan["cleanrl_reference"]["source_root"]) == plan["cleanrl_reference"],
        "the original CleanRL model source changed")
    software = plan["qualification_software"]
    require(identity.software_reference(software["result"]["path"], software["installed_backend"]["path"],
        software["asset_receipt"]["path"]) == software, "the original shared-software packet changed")
    return software


def verify_qualification_sources(plan):
    """Apply execution-only original test controls after the read-only source proof succeeds."""
    verify_qualification_source_receipts(plan)
    identity = qualification_identity(plan).adjacent("h100_nvl_slurm_site_validation")
    identity.require_gate_controls(os.environ, plan["cleanrl_reference"])
    require(qualification_identity(plan).exact(plan["renderer_contract"], renderer_contract(plan,
        qualification_identity(plan).read_receipt(plan["qualification_software"]["installed_backend"]))), "the frozen original renderer matrix changed")
    os.environ["CLEANRL_REFERENCE_FILE"] = plan["cleanrl_reference"]["file"]["path"]


def test_collection_contract(plan):
    """Derive exact per-function case counts from the frozen source without importing numerical tests."""
    prefix = "10_jax_exploration_platform/"
    expected = {}
    for relative in sorted(plan["qualification_sources"]["files"]):
        path = Path(relative)
        if not (relative.startswith(prefix + "tests/") and path.name.startswith("test_") and path.suffix == ".py"):
            continue
        tree = ast.parse((Path(plan["source_root"]) / relative).read_text())
        for node in tree.body:
            require(not (isinstance(node, ast.ClassDef) and node.name.startswith("Test")),
                "unsupported class-based original test collection")
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            require(node.name != "pytest_generate_tests", "dynamic original test collection is unsupported")
            for decorator in node.decorator_list:
                if isinstance(decorator, ast.Call) and isinstance(decorator.func, ast.Attribute) and decorator.func.attr == "fixture":
                    require(not any(keyword.arg == "params" for keyword in decorator.keywords),
                        "dynamic fixture parametrization is unsupported")
            if not node.name.startswith("test_"):
                continue
            count = 1
            for decorator in node.decorator_list:
                if isinstance(decorator, ast.Call) and isinstance(decorator.func, ast.Attribute) and decorator.func.attr == "parametrize":
                    require(len(decorator.args) >= 2 and isinstance(decorator.args[1], (ast.List, ast.Tuple))
                        and decorator.args[1].elts, "dynamic or empty original test parametrization is unsupported")
                    count *= len(decorator.args[1].elts)
            expected[relative[len(prefix):] + "::" + node.name] = count
    require(expected, "the complete original test-function roster is absent")
    return expected


def qualification_pytest_arguments(plan):
    """Pin the complete original test selection and collection root to the exact frozen platform source."""
    root = Path(plan["source_root"]) / "10_jax_exploration_platform"
    return ["--rootdir", str(root), "-q", str(root / "tests/atari_pixels"),
            str(root / "tests/test_atari_pixel_environment.py")]


def validate_test_collection(plan, tests, stdout):
    """Require every collected source case and every passed phase, including explicit zero deselection."""
    expected = test_collection_contract(plan)
    root = Path(plan["source_root"]) / "10_jax_exploration_platform"
    rootdir = Path(tests["collection_rootdir"])
    require(rootdir == root, "pytest collection used an unrelated root directory")
    require(tests.get("command") == [plan["python"], "-m", "pytest", *qualification_pytest_arguments(plan)],
        "full original pytest command or explicit source root changed")
    nodes = tests["collected_nodeids"]
    require(isinstance(nodes, list) and nodes and all(type(node) is str for node in nodes)
        and len(nodes) == len(set(nodes)) and tests["deselected_nodeids"] == [], "test cases were absent, duplicated or deselected")
    observed = Counter()
    for node in nodes:
        pieces = node.split("::")
        require(len(pieces) == 2 and not Path(pieces[0]).is_absolute() and ".." not in Path(pieces[0]).parts,
            "unsupported original test node identity")
        file = rootdir / pieces[0]
        require(file.is_relative_to(root), "a test node escaped its original source")
        key = str(file.relative_to(root)) + "::" + pieces[1].split("[", 1)[0]
        require(key in expected, "an original test function was substituted")
        observed[key] += 1
    require(dict(observed) == expected, "the full original test case matrix was truncated or changed")
    reports = tests["reports"]
    phases = Counter()
    for row in reports:
        require(row["nodeid"] in nodes and row["phase"] in {"setup", "call", "teardown"}
            and row["outcome"] == "passed" and type(row["duration_seconds"]) in {int, float}
            and math.isfinite(row["duration_seconds"]) and row["duration_seconds"] >= 0,
            "a full original test phase failed, skipped or has invalid duration")
        phases[(row["nodeid"], row["phase"])] += 1
    require(phases == Counter({(node, phase): 1 for node in nodes for phase in ("setup", "call", "teardown")}),
        "every original case needs exactly one passed setup, call and teardown")
    require(tests["collection_reports"] and all(row["outcome"] == "passed" for row in tests["collection_reports"]),
        "test collection failed or skipped a source module")
    clean = re.sub(r"\x1b\[[0-9;]*m", "", stdout)
    summaries = re.findall(r"(?:^|[= \n])([0-9]+) passed(?: in |,)", clean)
    require(summaries and int(summaries[-1]) == len(nodes), "the captured pytest summary does not match its complete case roster")
    return len(nodes)


def verify_qualification_backend(plan, backend):
    """Match actual installed package/source/library bytes to the completed original UVA software packet."""
    identity = qualification_identity(plan)
    reference = plan["qualification_software"]
    expected = identity.read_receipt(reference["installed_backend"])
    require(backend == expected, "actual installed packages/JAXtari/CUDA libraries differ from the original software packet")
    return expected


def prepare_qualification_inputs(plan, result_path, review_path):
    """Freeze an actual target result and independent review without requiring a common baseline or fifty ideas."""
    require(result_path is not None and review_path is not None, "actual independently reviewed target qualification required")
    identity = core()
    plan["target_qualification_inputs"] = {"result": identity.receipt(result_path), "review": identity.receipt(review_path)}
    verify_qualification_inputs(plan)


def verify_qualification_inputs(plan):
    """Refuse a missing, changed, unrelated or unreviewed actual full-original qualification."""
    identity = core()
    inputs = plan["target_qualification_inputs"]
    result = identity.read_receipt(inputs["result"])
    review = identity.read_receipt(inputs["review"])
    directory = Path(inputs["result"]["path"]).parent
    qualified_plan = identity.read_receipt(identity.receipt(directory / "measurement_plan.json"))
    require(Path(inputs["result"]["path"]).name == "target_qualification_result.json"
        and identity.exact(qualified_plan.get("target_qualification"), QUALIFICATION),
        "a generic successful measurement cannot substitute for target qualification")
    validate_original_qualification(qualified_plan)
    verify_saved_sources(directory, qualified_plan)
    verify_qualification_source_receipts(qualified_plan)
    metadata = identity.read_receipt(identity.receipt(directory / "configured_isolation_metadata.json"))
    binding = identity.read_receipt(identity.receipt(directory / "resolved_device_binding.json"))
    identity.validate_binding(qualified_plan, metadata, binding, directory)
    require(result.get("kind") == "uva_configured_isolation_full_original_qualification"
        and result.get("completed") is True and identity.exact(result.get("qualification_protocol"), QUALIFICATION)
        and result.get("source_commit") == plan["source_commit"] and result.get("hardware_class") == plan["hardware_class"]
        and result.get("runtime_environment") == plan["runtime_environment"]
        and result.get("configured_isolation", {}).get("profile") == plan["configured_isolation"]["profile"]
        and result.get("asset_record") == plan["asset_record"] and result.get("all_original_gates_passed") is True,
        "the actual full-original qualification is incomplete or differs from this target")
    require(all(type(result.get(key)) is int and result[key] == 0 for key in (
        "scientific_credit", "case_credit", "idea_credit", "packing_point_credit", "throughput_credit", "site_qualification_credit"))
        and result.get("worker_admitted") is False and result.get("admission_granted") is False,
        "target qualification cannot grant scientific or admission credit")
    require(result.get("resolved_device_binding") == binding and result.get("helpers") == qualified_plan["helpers"]
        and result.get("reference_iteration") == 1
        and result.get("executed_work", {}).get("full_reference_warmup_transitions") == 819200
        and result["executed_work"].get("full_reference_training_transitions") == 16384
        and result["executed_work"].get("full_reference_updates") == 1
        and result["executed_work"].get("full_reference_optimizer_minibatch_updates") == 16,
        "qualification changed its actual physical worker or controlled full-original work")
    memory = result["qualification_memory"]
    rebuilt_memory = validate_window(memory["assigned_gpu_process_observation"])
    require(all(memory.get(key) == value for key, value in rebuilt_memory.items())
        and memory["gpu_uuid"] == binding["gpu_uuid"]
        and memory["assigned_gpu_process_observation"]["pid"] == metadata["process"]["pid"],
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
    require(type(loaded.get("pid")) is int and loaded["pid"] == metadata["process"]["pid"]
        and loaded.get("platform") in {"gpu", "cuda"} and isinstance(loaded.get("platform_version"), str)
        and loaded["platform_version"] and binding["native"]["driver_library"] in loaded.get("loaded_shared_libraries", []),
        "the actual loaded libraries are not this native-bound GPU process")
    for path, value in loaded["verified_package_libraries"].items():
        require(identity.receipt(path)["sha256"] == value["sha256"], "an actual shared CUDA/cuDNN image changed")
    actual_assets = identity.read_receipt(identity.receipt(directory / "asset_receipt.json"))
    require(actual_assets == identity.read_receipt(qualified_plan["qualification_software"]["asset_receipt"]),
        "the actual original asset receipt changed")
    tests = identity.read_receipt(identity.receipt(directory / "original_tests.json"))
    require(tests["command"] == [qualified_plan["python"], "-m", "pytest", *qualification_pytest_arguments(qualified_plan)]
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
        and review.get("kind") == "independent_full_original_target_qualification_review"
        and review.get("result") == inputs["result"] and review.get("approved") is True
        and review.get("admission_granted") is False and type(review.get("scientific_credit")) is int and review["scientific_credit"] == 0
        and review.get("job_id") == metadata["inputs"]["job_id"]
        and review.get("actual_terminal_state") == "COMPLETED" and review.get("actual_exit_code") == "0:0",
        "an exact independent actual qualification review is required")
    return result


def verify_saved_sources(directory, plan):
    """Recheck ordinary staged artifacts and every exact committed helper/core byte."""
    import train_optimized as training
    identity = core()
    identity.validate_plan(plan)
    require(Path(directory) == Path(plan["campaign_run_dir"]) / "data" / plan["unit_id"]
        and identity.read_receipt(identity.receipt(Path(directory) / "measurement_prepared.json")) == {
            "plan_sha256": identity.receipt(Path(directory) / "measurement_plan.json")["sha256"]}, "immutable unit identity changed")
    import diagnose_gpu_execution as diagnostic
    require(diagnostic.core_identity(Path(plan["source_root"]), plan["source_commit"]) == plan["core"],
        "the exact numerical source changed")
    for name in ("paired_source", "helpers"):
        value = plan[name]
        require(training.source_receipt(Path(value["source_root"]), value["source_commit"], list(value["files"])) == value,
            "an exact producer source receipt changed")
    for path, digest in plan["helpers"]["files"].items():
        require(identity.receipt(Path(directory) / "code" / Path(path).name)["sha256"] == digest,
            "an exact staged helper changed")
    require({"configured_measurement.py", *identity.HELPERS} <= {Path(path).name for path in plan["helpers"]["files"]},
        "configured producer omitted an identity dependency")
    identity.source_identity(plan)


def verify_saved_measurement(directory, result):
    """Validate the explicit local single-card evidence; portable, packing and final readers remain unsupported."""
    identity = core()
    directory = Path(directory)
    plan = identity.read_receipt(identity.receipt(directory / "measurement_plan.json"))
    identity.validate_plan(plan)
    require(directory == Path(plan["campaign_run_dir"]) / "data" / plan["unit_id"]
        and identity.read_receipt(identity.receipt(directory / "measurement_prepared.json")) == {
            "plan_sha256": identity.receipt(directory / "measurement_plan.json")["sha256"]},
        "configured result changed its exact prepared unit")
    require("target_qualification" not in plan and result.get("configured_isolation") == plan["configured_isolation"]
        and result.get("configured_isolation_record") is True and all(result.get(key) == value for key, value in BOUNDARY.items()),
        "a qualification or unrecognized record cannot supply selected measurement evidence")
    verify_saved_sources(directory, plan)
    metadata = identity.read_receipt(identity.receipt(directory / "configured_isolation_metadata.json"))
    binding = identity.read_receipt(identity.receipt(directory / "resolved_device_binding.json"))
    identity.validate_binding(plan, metadata, binding, directory)
    require(result.get("resolved_device_binding") == binding and result.get("hardware", {}).get("gpu_uuid") == binding["gpu_uuid"]
        and result.get("helpers") == plan["helpers"] and result.get("source_commit") == plan["source_commit"]
        and result.get("runtime_environment") == plan["runtime_environment"], "result changed its physical/source/runtime binding")
    execution = result.get("execution_identity", {})
    require(all(execution.get(key) == metadata["process"][key] for key in ("pid", "hostname", "process_start_time")),
        "result substituted the identity-bound worker")
    verify_qualification_inputs(plan)
    require(result.get("target_qualification_inputs") == plan["target_qualification_inputs"], "actual qualification inputs changed")
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
                rebuilt = validate_window(memory["assigned_gpu_process_observation"])
                require(all(memory.get(key) == value for key, value in rebuilt.items()), "a measured memory window changed")
                require(memory["gpu_uuid"] == binding["gpu_uuid"] and memory["assigned_gpu_process_observation"]["pid"] == metadata["process"]["pid"],
                    "measured process memory belongs to another physical worker")
                window = memory["assigned_gpu_process_observation"]
                require(pair[arm]["iterations"] == plan["iterations"]
                    and 0 < pair[arm]["seconds"] <= window["window_finished_at_seconds"] - window["window_started_at_seconds"] + .001,
                    "the sampled window does not cover the actual timed round")
    if plan.get("kind") == "selected_configuration_profile":
        memory = result["profile_memory"]
        rebuilt = validate_window(memory["assigned_gpu_process_observation"])
        require(all(memory.get(key) == value for key, value in rebuilt.items())
            and result.get("gpu_uuid") == binding["gpu_uuid"] and memory["gpu_uuid"] == binding["gpu_uuid"]
            and memory["assigned_gpu_process_observation"]["pid"] == metadata["process"]["pid"],
            "the complete profile memory window changed its actual child")
    return plan, binding


def run_qualification(directory, plan, paired, jax, Config, Trainer, Environment):
    """Execute every original gate followed by one full-size reference update, with no automatic admission."""
    import contextlib
    import pytest
    import train_optimized as training
    value = validate_original_qualification(plan)
    verify_qualification_sources(plan)
    directory = Path(directory)
    root = Path(plan["source_root"]) / "10_jax_exploration_platform"
    arguments = qualification_pytest_arguments(plan)
    gates, collected, collection_reports, deselected, collection_rootdir = [], [], [], [], []
    class Reports:
        """Retain each real collected test and its call/fixture outcome."""
        def pytest_runtest_logreport(self, report):
            """Record each executed test phase rather than treating pytest exit zero as complete evidence."""
            gates.append({"nodeid": report.nodeid, "phase": report.when, "outcome": report.outcome,
                          "duration_seconds": report.duration})
        def pytest_collection_finish(self, session):
            """Save the actual complete collected matrix and source root without selecting cases."""
            collected.extend(item.nodeid for item in session.items)
            collection_rootdir.append(str(session.config.rootpath))
        def pytest_collectreport(self, report):
            """Keep collection failures/skips explicit rather than losing them before test calls."""
            collection_reports.append({"nodeid": report.nodeid, "outcome": report.outcome})
        def pytest_deselected(self, items):
            """Refuse any external plugin or setting that removes original cases."""
            deselected.extend(item.nodeid for item in items)
    # All tests run inside the already physically bound process; no unbound numerical subprocess is created.
    with (directory / "original_tests.stdout").open("x") as stdout, (directory / "original_tests.stderr").open("x") as stderr:
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            returncode = int(pytest.main(arguments, plugins=[Reports()]))
    test_record = {"command": [plan["python"], "-m", "pytest", *arguments],
        "execution": "same already identity-bound process via pytest.main", "returncode": returncode, "reports": gates,
        "collected_nodeids": collected, "collection_rootdir": collection_rootdir[0] if len(collection_rootdir) == 1 else None,
        "collection_reports": collection_reports, "deselected_nodeids": deselected}
    training.atomic_json(directory / "original_tests.json", test_record)
    require(returncode == 0 and gates and all(row["outcome"] == "passed" for row in gates),
        "full original learner/environment gates failed or skipped")
    validate_test_collection(plan, test_record, (directory / "original_tests.stdout").read_text())
    renderer_path = root / "scripts/atari_pixel_campaign/check_rendering.py"
    specification = importlib.util.spec_from_file_location("qualification_original_renderer", renderer_path)
    renderer = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(renderer)
    rendering = renderer.check("all")
    training.atomic_json(directory / "original_renderer.json", rendering)
    require(rendering.get("checks") == {"cache": "passed", "palette": "passed", "fused": "passed"}
        and all(type(rendering.get(key)) is int and rendering[key] == plan["renderer_contract"][key]
            for key in ("rooms", "cases", "synthetic_pixel_cases")), "full original renderer checks did not finish every source-bound case")
    # This controlled full reference work is separate from all numerical work performed by the test suite.
    trainer = Trainer(Config(**value["learner"]), Environment(**value["environment"]))
    state = trainer.initialize(tuple(plan["copy_seeds"]))
    state = trainer.warmup(state)
    before_target = paired.host_copy(state.target)
    before_policy = paired.host_copy(state.parameters["agent"])
    state, metrics = trainer.train_iteration(state)
    jax.block_until_ready((state, metrics))
    import jax.numpy as jnp
    import numpy as np
    leaves = [leaf for leaf in jax.tree.leaves(state) if hasattr(leaf, "dtype") and leaf.dtype.kind in "fc"]
    require(leaves and bool(jax.device_get(jnp.all(jnp.stack([jnp.all(jnp.isfinite(leaf)) for leaf in leaves])))),
        "the full reference state contains nonfinite values")
    groups = {"parameters": state.parameters, "target": state.target, "optimizer_first": state.optimizer.first,
        "optimizer_second": state.optimizer.second, "observation_statistics": state.observation_statistics,
        "reward_statistics": state.reward_statistics}
    dtypes = {name: sorted({str(leaf.dtype) for leaf in jax.tree.leaves(tree)}) for name, tree in groups.items()}
    require(dtypes == EXPECTED_DTYPES
        and int(state.iteration) == 1 and int(state.optimizer.step) == 16,
        "full reference precision or update count changed")
    target_equal = all(np.array_equal(a, b) for a, b in zip(jax.tree.leaves(before_target),
        jax.tree.leaves(paired.host_copy(state.target))))
    policy_changed = any(not np.array_equal(a, b) for a, b in zip(jax.tree.leaves(before_policy),
        jax.tree.leaves(paired.host_copy(state.parameters["agent"]))))
    require(target_equal and policy_changed, "full reference update changed its frozen target or left the policy unchanged")
    return {"completed": True, "qualification_protocol": QUALIFICATION, "all_original_gates_passed": True,
        "asset_record": plan["asset_record"], "original_test_reports": gates,
        "original_collected_test_nodeids": collected, "renderer": rendering, "renderer_contract": plan["renderer_contract"],
        "dtypes": dtypes, "finite_state_leaves": len(leaves), "reference_iteration": int(state.iteration),
        "reference_optimizer_step": int(state.optimizer.step), "frozen_target_unchanged": target_equal,
        "policy_parameters_changed": policy_changed,
        "gate_artifacts": {name: qualification_identity(plan).receipt(directory / name) for name in (
            "original_tests.json", "original_tests.stdout", "original_tests.stderr", "original_renderer.json")},
        "executed_work": {"full_reference_warmup_transitions": 50 * 128 * 128,
            "full_reference_training_transitions": 128 * 128, "full_reference_updates": 1,
            "full_reference_optimizer_minibatch_updates": 16,
            "controlled_transition_scope": "agent transitions; reset NOOP/FIRE actions and raw game frames are not counted",
            "original_test_and_renderer_work": "complete commands/source/test outcomes/case counts retained; synthetic/partial fixture runtime work is unquantified",
            "whole_job_transition_total_known": False},
        "test_work_accounting_complete": False}
