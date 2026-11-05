"""Run one reviewed direct Nsight process/descriptor diagnostic with zero training transitions."""
import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import importlib.abc
import json
import os
from pathlib import Path
import stat
import signal
import subprocess
import sys
import time

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))
import direct_profile_process as direct

NAME = "diagnose_direct_nsight_process.py"
HELPERS = (NAME, "direct_profile_process.py", "nsight_process_identity.py", "direct_capture_release.py", "direct_tool_descendants.py")
ZERO = {"scientific_credit": 0, "profile_credit": 0, "timing_credit": 0, "case_credit": 0, "idea_credit": 0,
    "environment_transitions": 0, "training_updates": 0, "admission_granted": False, "worker_admitted": False,
    "cuda_qualification_established": False, "physical_binding_established": False,
    "whole_tool_library_or_device_activity_quantified": False}


@contextmanager
def producer_interruptions():
    """Let the sole lock producer clean up actual owned tool processes before closing its descriptor."""
    def interrupted(number, frame):
        """Turn a termination request into the producer's ordinary owned-child cleanup path."""
        raise InterruptedError("the direct tool producer received signal " + str(number))
    previous = {number: signal.getsignal(number) for number in (signal.SIGTERM, signal.SIGINT)}
    for number in previous:
        signal.signal(number, interrupted)
    try:
        yield
    finally:
        for number, handler in previous.items():
            signal.signal(number, handler)


def metadata_guard():
    """Forbid numerical imports and Python device/library/network operations in every role."""
    direct.require(not any(name.split(".")[0] in direct.NUMERICAL for name in sys.modules),
        "a numerical package entered the direct process diagnostic")
    class NoNumerical(importlib.abc.MetaPathFinder):
        """Retain the standard-library-only payload after actual tool injection."""
        def find_spec(self, fullname, path=None, target=None):
            """Refuse a numerical runtime before it imports any source."""
            if fullname.split(".")[0] in direct.NUMERICAL:
                raise RuntimeError("numerical imports are forbidden in this zero-transition diagnostic")
    sys.meta_path.insert(0, NoNumerical())
    def audit(event, values):
        """Keep Python away from device files, library calls, networks and shell commands."""
        if event in {"socket.connect", "socket.bind", "socket.getaddrinfo", "os.system", "ctypes.dlopen", "ctypes.dlsym"}:
            raise RuntimeError("library, network and shell operations are forbidden in the process diagnostic")
        if event == "open" and isinstance(values[0], (str, bytes)) and os.fsdecode(values[0]).startswith(("/dev/nvidia", "/dev/dri")):
            raise RuntimeError("Python device opening is forbidden in the process diagnostic")
    sys.addaudithook(audit)



def required_helpers(native=False):
    """Keep the default five helpers; explicitly add only native provider and evidence helpers."""
    direct.require(type(native) is bool, 'native diagnostic selection must be explicit')
    return (*HELPERS, 'linux_pidfd_process_control.py', 'native_process_evidence.py') if native else HELPERS


def native_role(directory, plan, role):
    """Validate an actual role and source before starting its one explicit native provider."""
    current = direct.context(plan, directory)
    observer = direct.generic(plan)
    if role == 'producer':
        current = producer_release(directory, plan)
    else:
        direct.no_standing_descriptor()
        intent = direct.validate_intent(directory, plan, observer.read_json(directory / 'direct_profile_process_intent.json'))
        direct.require(current['command'] == intent['waiter_command' if role == 'waiter' else 'capture_command'],
            'native child startup requires its exact current staged source role')
    return direct.adjacent_helper(plan, 'native_process_evidence').bootstrap(directory, plan, role, observer, current)


def read_plan(directory):
    """Require the exact staged immutable diagnostic and selected private Python."""
    directory = Path(directory).resolve(strict=True)
    raw = (directory / "measurement_plan.json").read_bytes()
    plan = json.loads(raw)
    direct.require(plan.get("kind") == direct.DIAGNOSTIC and plan.get("gpu_uuid") == direct.GPU
        and plan.get("capture_process_protocol") == direct.PROTOCOL
        and plan.get("hardware_class") == "h100" and directory == Path(plan["campaign_run_dir"]) / "data" / plan["unit_id"]
        and Path(__file__).resolve(strict=True) == directory / "code" / NAME
        and str(Path(sys.executable).absolute()) == plan["python"]
        and json.loads((directory / "measurement_prepared.json").read_bytes()) == {"plan_sha256": hashlib.sha256(raw).hexdigest()}
        and plan.get("diagnostic_boundary") == ZERO, "execute only the exact staged zero-transition direct diagnostic")
    direct.require(set(plan["helpers"]["files"]) == {direct.PREFIX + name for name in required_helpers('native_process_control' in plan)},
        "the direct diagnostic requires exactly its declared process-control and release helpers")
    for name in required_helpers('native_process_control' in plan):
        path = directory / "code" / name
        direct.require(path.resolve(strict=True) == path and path.is_file()
            and hashlib.sha256(path.read_bytes()).hexdigest() == plan["helpers"]["files"][direct.PREFIX + name],
            "the direct process diagnostic source changed")
    direct.contract(plan, directory)
    return directory, plan


def launch_review(directory, plan):
    """Require root's exact source/packet approval and actual completed C2 exit review."""
    observer = direct.generic(plan)
    value = observer.read_json(Path(directory) / "diagnostic_launch_review.json")
    direct.require(value.get("kind") == "reviewed_direct_nsight_descriptor_diagnostic_launch"
        and value.get("approved") is True and value.get("passed") is True
        and value["plan"] == observer.receipt(Path(directory) / "measurement_plan.json")
        and value.get("helper_commit") == plan["helpers"]["source_commit"]
        and value.get("gpu_uuid") == direct.GPU and value.get("standing_descriptor_owner") == "tool producer only"
        and value.get("metadata_controller_owns_standing_descriptor") is False
        and value.get("actual_c2_exit_verified") is True,
        "the exact source/packet and actual C2 exit must be reviewed before any producer starts")
    exit_reference = value["c2_exit_review"]
    exit_path = Path(exit_reference["path"])
    direct.require(exit_path.name == "independent_exit_review.json"
        and exit_path.is_relative_to(Path(plan["campaign_run_dir"]) / "remote_setup")
        and observer.receipt(exit_path) == exit_reference,
        "the actual independently reviewed C2 exit bytes changed")
    exited = observer.read_json(exit_reference["path"])
    direct.require(exited.get("passed") is True and exited.get("approved") is True
        and exited.get("independent_of_producer") is True and exited.get("all_exact_processes_exited") is True
        and exited.get("gpu_uuid") == direct.GPU and exited.get("copies") == 2
        and exited.get("source_commit") == "d28e6735e79788f618acb0ebfde8609db1f776af"
        and exited.get("unresolved_anomalies") == [], "actual exact C2 exits cannot be replaced by a planned or partial result")
    return value


def append_producer(directory, plan, producer):
    """Append only this exact new producer row under a durable campaign process-ledger lock."""
    observer = direct.generic(plan)
    run = Path(plan["campaign_run_dir"])
    ledger = run / "processes.json"
    direct.require(ledger.resolve(strict=True) == ledger and ledger.is_file(), "the exact existing process ledger is required")
    row = {"kind": direct.DIAGNOSTIC, "hostname": "serval02", "pid": producer["pid"],
        "start_time": producer["process_start_time"], "unit": plan["unit_id"], "source_commit": plan["source_commit"],
        "helper_commit": plan["helpers"]["source_commit"], "gpu_uuid": direct.GPU,
        "producer_owns_standing_descriptor": True, "controller_owns_standing_descriptor": False}
    lock = ledger.with_suffix(".lock")
    with lock.open("a+") as held:
        fcntl.flock(held.fileno(), fcntl.LOCK_EX)
        before = ledger.read_bytes()
        records = json.loads(before)
        direct.require(isinstance(records, list) and not any(record.get("unit") == plan["unit_id"] for record in records),
            "this direct diagnostic producer was already registered")
        updated = json.dumps([*records, row], indent=2, allow_nan=False).encode() + b"\n"
        temporary = run / (".processes.json.pending-" + str(os.getpid()))
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, stat.S_IMODE(ledger.stat().st_mode))
        os.fchmod(descriptor, stat.S_IMODE(ledger.stat().st_mode))
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(updated); stream.flush(); os.fsync(stream.fileno())
        direct.require(ledger.read_bytes() == before, "another process changed the ledger outside the shared lock")
        temporary.replace(ledger)
        parent = os.open(run, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(parent)
        finally:
            os.close(parent)
    observer.publish(Path(directory) / "diagnostic_process_ledger_append.json", {"row": row,
        "source_path": str(ledger), "before_sha256": observer.digest(before), "after_sha256": observer.digest(updated)})
    return row


def controller(directory):
    """Launch and wait on one registered tool producer without ever holding its GPU lock."""
    directory, plan = read_plan(directory)
    observer = direct.generic(plan)
    current = direct.context(plan, directory)
    no_lock = direct.no_standing_descriptor()
    launch_review(directory, plan)
    direct.require(not any(os.path.lexists(directory / name) for name in ("controller_spawn.json", "controller_wait.json", "failure.json")),
        "this controller attempt already exists; prepare a new unit")
    command = [plan["python"], "-I", "-S", "-B", str(directory / "code" / NAME), "--producer-unit", str(directory)]
    child = None
    with (directory / "producer_stdout.raw").open("xb") as output, (directory / "producer_stderr.raw").open("xb") as errors:
        try:
            child = subprocess.Popen(command, stdout=output, stderr=errors, close_fds=True, pass_fds=())
            actual = observer.ProcReader().snapshot(child.pid, direct.OWNER, current["cgroup"])
            direct.require(actual["parent_pid"] == current["pid"] and actual["command"] == command,
                "the actual controller Popen producer changed source")
            observer.publish(directory / "controller_spawn.json", {"kind": "actual_direct_diagnostic_producer_popen",
                "controller": current, "producer": actual, "command": command, "close_fds": True, "pass_fds": [],
                "controller_no_standing_descriptor": no_lock, "plan": observer.receipt(directory / "measurement_plan.json")})
            row = append_producer(directory, plan, actual)
            observer.publish(directory / "producer_release.json", {"controller_spawn": observer.receipt(directory / "controller_spawn.json"),
                "ledger_append": observer.receipt(directory / "diagnostic_process_ledger_append.json"), "producer_row": row,
                "launch_review": observer.receipt(directory / "diagnostic_launch_review.json")})
            returned = child.wait(timeout=plan["maximum_active_seconds"] + 300)
            after = observer.ProcReader().snapshot(os.getpid(), direct.OWNER, current["cgroup"])
            direct.require(observer.stable(after) == observer.stable(current), "the actual metadata controller changed during its wait")
            observer.publish(directory / "controller_wait.json", {"kind": "actual_direct_diagnostic_producer_wait",
                "waited_pid": child.pid, "returncode": returned, "exited_normally": returned >= 0,
                "command": command, "interventions": [], "launch": observer.receipt(directory / "controller_spawn.json"),
                "controller_after": after, "controller_no_standing_descriptor_after": direct.no_standing_descriptor()})
            direct.require(returned == 0, "the actual diagnostic tool producer failed; preserve its raw output and wait")
        finally:
            if child is not None and child.poll() is None:
                child.terminate()
                try:
                    cleanup_return = child.wait(timeout=180)
                    observer.publish(directory / "controller_cleanup_wait.json", {
                        "producer_pid": child.pid, "actual_returncode": cleanup_return,
                        "interventions": ["the exact producer received SIGTERM during controller failure"],
                        "controller_owns_standing_descriptor": False, **ZERO})
                except subprocess.TimeoutExpired:
                    observer.publish(directory / "controller_cleanup_unverified.json", {
                        "producer_pid": child.pid, "producer_left_running": True,
                        "reason": "the sole descriptor owner has not finished its own tool cleanup",
                        "controller_owns_standing_descriptor": False, **ZERO})
    observer.publish(directory / "controller_output_hashes.json", {"stdout": observer.receipt(directory / "producer_stdout.raw"),
        "stderr": observer.receipt(directory / "producer_stderr.raw"), "wait": observer.receipt(directory / "controller_wait.json")})


def producer_release(directory, plan):
    """Remain descriptor-free until the actual controller durably records this producer."""
    observer = direct.generic(plan)
    current = direct.context(plan, directory)
    direct.no_standing_descriptor()
    started = time.monotonic()
    while not os.path.lexists(Path(directory) / "producer_release.json"):
        direct.require(time.monotonic() - started < 180, "the actual controller did not publish this producer release")
        time.sleep(.05)
    release = observer.read_json(Path(directory) / "producer_release.json")
    spawn = observer.read_json(Path(directory) / "controller_spawn.json")
    direct.require(release["controller_spawn"] == observer.receipt(Path(directory) / "controller_spawn.json")
        and release["ledger_append"] == observer.receipt(Path(directory) / "diagnostic_process_ledger_append.json")
        and release["launch_review"] == observer.receipt(Path(directory) / "diagnostic_launch_review.json")
        and observer.stable(spawn["producer"]) == observer.stable(current)
        and spawn["producer"]["parent_pid"] == spawn["controller"]["pid"]
        and spawn["controller_no_standing_descriptor"]["standing_descriptor_count"] == 0
        and spawn["close_fds"] is True and spawn["pass_fds"] == [], "the current producer has no exact recorded controller launch")
    observer.validate_snapshot(spawn["controller"], direct.OWNER, current["cgroup"])
    actual_parent = observer.ProcReader().snapshot(current["parent_pid"], direct.OWNER, current["cgroup"])
    direct.require(observer.stable(actual_parent) == observer.stable(spawn["controller"]), "the actual metadata controller disappeared or changed")
    row = release["producer_row"]
    ledger = json.loads((Path(plan["campaign_run_dir"]) / "processes.json").read_bytes())
    direct.require(ledger.count(row) == 1 and row["pid"] == current["pid"] and row["start_time"] == current["process_start_time"]
        and row["unit"] == plan["unit_id"] and row["gpu_uuid"] == direct.GPU, "the actual producer is not uniquely registered")
    launch_review(directory, plan)
    return current


def produce_with_owned_descriptor(directory, *, runtime=None):
    """Alone acquire the approved existing lock and retain it through every actual tool/child wait."""
    directory, plan = read_plan(directory)
    observer = direct.runtime_observer(plan, runtime.observer if runtime is not None else None)
    current = producer_release(directory, plan)
    direct.require(not any(os.path.lexists(directory / name) for name in (
        "direct_profile_process_intent.json", "result.json", "failure.json", "trace", "capture_identity.json")),
        "this actual producer attempt already exists; never reuse it")
    descriptor = None
    try:
        descriptor = os.open(direct.LOCK, os.O_RDWR | os.O_NOFOLLOW | os.O_NONBLOCK)
        information = os.fstat(descriptor)
        fields = {"device": information.st_dev, "inode": information.st_ino, "uid": information.st_uid,
            "mode": stat.S_IMODE(information.st_mode), "regular_file": stat.S_ISREG(information.st_mode)}
        direct.require(fields == direct.LOCK_IDENTITY, "the existing approved GPU lock file changed; never create a replacement")
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        direct.descriptor_record(plan, descriptor, current, local=True)
        tool = plan["profile_tool"]
        direct.require(observer.receipt(tool["path"])["sha256"] == tool["sha256"], "the recorded actual profiling tool changed")
        with (direct.tool_scope(directory, plan, descriptor) if runtime is None else
                direct.tool_scope(directory, plan, descriptor, observer=observer)) as scope:
            actual = scope.run([tool["path"], "--version"], capture_output=True, timeout=15, check=False)
            observer.publish(directory / "tool_version_return.json", {"command": [tool["path"], "--version"],
                "returncode": actual.returncode, "stdout": actual.stdout.decode(), "stderr": actual.stderr.decode()})
            direct.require(actual.returncode == 0 and actual.stderr == b"" and actual.stdout.decode().strip() == tool["version"]
                and observer.receipt(tool["path"])["sha256"] == tool["sha256"], "the actual Nsight version or bytes changed")
            direct.run_tool(directory, plan, descriptor, scope=scope)
        diagnostic = observer.read_json(directory / "probe_result.json")
        direct.require(diagnostic.get("completed") is True and all(diagnostic.get(key) == value for key, value in ZERO.items()),
            "the actual metadata payload did not complete as a zero-transition diagnostic")
        if runtime is not None:
            runtime.complete(('tool_descendant_scope_terminal.json', 'tool_descendant_scope_restored.json', 'profiler_actual_wait.json'))
        proof = direct.validate_saved(directory, plan)
        observer.publish(directory / "result.json", {"kind": direct.DIAGNOSTIC, "completed": True, **ZERO,
            "plan": observer.receipt(directory / "measurement_plan.json"), "process_proof": proof,
            "payload": observer.receipt(directory / "probe_result.json"), "tool_version": observer.receipt(directory / "tool_version_return.json")})
    except BaseException as error:
        observer.publish(directory / "failure.json", {"kind": direct.DIAGNOSTIC, "error_type": type(error).__name__,
            "error": str(error), "measurement_valid": False, **ZERO})
        raise
    finally:
        if descriptor is not None:
            os.close(descriptor)


def producer(directory, *, runtime=None):
    """Install cleanup-aware signal handling before the sole producer opens its GPU descriptor."""
    with producer_interruptions():
        return produce_with_owned_descriptor(directory, runtime=runtime)


def probe(directory, *, runtime=None):
    """Observe the actual producer and exit without importing a learner or calling CUDA."""
    directory, plan = read_plan(directory)
    observer = direct.generic(plan)
    claim = direct.wait_release(directory, plan, observer=runtime.observer if runtime is not None else None)
    started = time.time()
    observer.publish(directory / "probe_result.json", {"kind": "actual_direct_standard_library_process_payload",
        "completed": True, "pid": claim["pid"], "process_start_time": claim["process_start_time"],
        "numerical_modules": sorted(direct.NUMERICAL.intersection(sys.modules)), "cuda_profiler_api_called": False,
        "standing_descriptor_inherited": False, "payload_started_at_seconds": started,
        "payload_finished_at_seconds": time.time(), **ZERO})

    if runtime is not None:runtime.complete(('capture_release_observed.json', 'probe_result.json'))


def prepare(arguments):
    """Write only reproducible diagnostic metadata, never execute the tool or contact a host."""
    run = arguments.run_dir.resolve(strict=True)
    direct.require(Path(arguments.unit_id).name == arguments.unit_id and arguments.unit_id not in {"", ".", ".."},
        "one literal zero-transition diagnostic unit name is required")
    root = arguments.source_root.resolve(strict=True)
    commit = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    files = {}
    selected_helpers = required_helpers(getattr(arguments, 'native_process_control', False))
    for name in selected_helpers:
        relative = direct.PREFIX + name
        raw = subprocess.check_output(["git", "-C", str(root), "show", commit + ":" + relative])
        direct.require(raw == (root / relative).read_bytes() == (SCRIPTS / name).read_bytes(), "commit reviewed diagnostic source before preparation")
        files[relative] = hashlib.sha256(raw).hexdigest()
    tool = json.loads(arguments.profile_tool_record.read_bytes())
    direct.require(set(tool) == {"path", "version", "sha256"} and Path(tool["path"]).is_absolute()
        and type(tool["version"]) is str and bool(tool["version"].strip()) and type(tool["sha256"]) is str
        and len(tool["sha256"]) == 64, "use an actual target tool record, never an invented command or version")
    directory = run / "data" / arguments.unit_id
    direct.require(not directory.exists(), "preserve every previous diagnostic packet or attempt")
    plan = {"schema_version": 1, "kind": direct.DIAGNOSTIC, "capture_process_protocol": direct.PROTOCOL,
        "hardware_class": "h100", "gpu_uuid": direct.GPU, "campaign_run_dir": str(run), "unit_id": arguments.unit_id,
        "source_commit": "d28e6735e79788f618acb0ebfde8609db1f776af", "python": str(arguments.python.absolute()),
        "helpers": {"source_root": str(root), "source_commit": commit, "files": files}, "profile_tool": tool,
        "maximum_active_seconds": 300, "diagnostic_boundary": ZERO,
        "preparation_scope": "metadata only; no launch review, C2 exit evidence, process, tool run or qualification is fabricated"}
    if getattr(arguments, 'native_process_control', False):
        plan['native_process_control'] = direct.adjacent_helper(plan, 'native_process_evidence').declaration(plan)
    directory.mkdir(parents=True)
    (directory / "code").mkdir()
    observer = direct.generic(plan)
    for name in selected_helpers:
        path = directory / "code" / name
        with path.open("xb") as stream:
            stream.write((SCRIPTS / name).read_bytes()); stream.flush(); os.fsync(stream.fileno())
    observer.publish(directory / "measurement_plan.json", plan)
    observer.publish(directory / "manifest.yaml", plan)
    observer.publish(directory / "config_resolved.yaml", {"gpu_uuid": direct.GPU, **ZERO})
    observer.publish(directory / "measurement_prepared.json", {"plan_sha256": observer.receipt(directory / "measurement_plan.json")["sha256"]})
    command = [plan["python"], "-I", "-S", "-B", str(directory / "code" / NAME), "--controller-unit", str(directory)]
    with (directory / "command.txt").open("x") as stream:
        import shlex
        stream.write(shlex.join(command) + "\n"); stream.flush(); os.fsync(stream.fileno())
    with (directory / "experiment_background.md").open("x") as stream:
        stream.write("## Purpose\n\nObserve the actual direct Nsight process relationship and sole-producer standing descriptor through exact waits. "
            "The standard-library payload executes zero environment transitions and zero training updates.\n\n"
            "## Key hyperparameters\n\nOne approved serval02 card 1 target; a five-minute active tool limit. "
            "No environment, learner, policy seed, or hyperparameter experiment runs.\n\n"
            "## Code and config changes\n\nThe root transport and metadata controller hold no standing descriptor. "
            "One tool producer acquires the existing card lock. The tool, waiter and payload receive no copy. "
            "The payload records readonly metadata about its exact known producer. "
            "Actual tool library/device activity is unquantified. The diagnostic grants zero scientific, profile, timing, CUDA or admission credit.\n\n"
            f"## Git state\n\nReviewed helper source: {commit}. The original numerical source is recorded but never imported.\n")
        stream.flush(); os.fsync(stream.fileno())
    return directory


def main():
    """Separate metadata preparation, descriptor-free controller, sole producer and stdlib payload."""
    parser = argparse.ArgumentParser(description=__doc__)
    operation = parser.add_mutually_exclusive_group(required=True)
    for name in ("controller-unit", "producer-unit", "direct-owned-waiter-unit", "direct-owned-capture-unit"):
        operation.add_argument("--" + name, type=Path)
    operation.add_argument("--prepare-only", action="store_true")
    for name in ("source-root", "run-dir", "profile-tool-record", "python"):
        parser.add_argument("--" + name, type=Path)
    parser.add_argument("--unit-id")
    parser.add_argument("--native-process-control", action="store_true")
    args = parser.parse_args()
    if args.prepare_only:
        direct.require(all(getattr(args, name) is not None for name in ("source_root", "run_dir", "profile_tool_record", "python", "unit_id")),
            "diagnostic preparation needs explicit source, run, actual tool record, Python and unit identity")
        print(json.dumps({"prepared_unit": str(prepare(args)), "jobs_submitted": 0, "gpu_operations": 0}))
        return
    direct.require(not args.native_process_control and not any(getattr(args, name) is not None for name in ("source_root", "run_dir", "profile_tool_record", "python", "unit_id")),
        "execution cannot override immutable preparation inputs")
    selected = args.producer_unit or args.direct_owned_waiter_unit or args.direct_owned_capture_unit
    runtime = None
    if selected:
        directory, plan = read_plan(selected)
        if 'native_process_control' in plan:
            role = 'producer' if args.producer_unit else 'waiter' if args.direct_owned_waiter_unit else 'capture'
            runtime = native_role(directory, plan, role)
    if args.producer_unit:
        directory, plan = read_plan(args.producer_unit)
        direct.context(plan, directory)
        direct.adjacent_helper(plan, 'direct_tool_descendants').prepare_native_control()
    metadata_guard()
    try:
        if args.controller_unit:
            controller(args.controller_unit)
        elif args.producer_unit:
            producer(args.producer_unit, runtime=runtime)
        elif args.direct_owned_waiter_unit:
            directory, plan = read_plan(args.direct_owned_waiter_unit)
            returned = direct.run_waiter(directory, plan, observer=runtime.observer if runtime is not None else None)
            if runtime is not None:
                direct.require(returned == 0, 'native waiter cannot complete after an unsuccessful payload wait')
                runtime.complete(('capture_actual_launch.json', 'capture_actual_wait.json'))
            return returned
        else:
            probe(args.direct_owned_capture_unit, runtime=runtime)
    except BaseException as error:
        if runtime is not None:runtime.failure(error)
        raise


if __name__ == '__main__':
    raise SystemExit(main())
