"""Start only the separately reviewed direct qualification, once.

The default reads local metadata and prints the intended command. It does not
contact serval02, acquire its standing lock, import a numerical runtime, or start
a child. Only the producer owns the GPU lock; this controller records and waits
for its actual child. This file is separate from the committed 6066 producer.
"""

import argparse
from contextlib import contextmanager, nullcontext
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import pwd
import re
import shlex
import signal
import socket
import stat
import subprocess
import sys
import time

PACKET = Path(__file__).absolute().parent
INTEGRATION = Path("/p/rlprojects/RND_worktrees/merge-atari-cleanrl")
RUN = PACKET.parent.parent
UNIT = RUN / "data" / PACKET.name
HELPER_COMMIT = "6066d9ea35fbc9c045855144550971c2ac5895f9"
SOURCE_COMMIT = "d28e6735e79788f618acb0ebfde8609db1f776af"
GPU_UUID = "GPU-fed69954-5a71-dbb2-ee91-544e0eb6b8a3"
UID = 2618919
HOST = "serval02"
PLAN_SHA256 = "515e4089bec5fa65931b5f08fc018d716209243fd8311cc0f4740f954769472e"
PACKET_FREEZE_SHA256 = "f7b544a5f5fcf8fb2ad6b1c974d8eb553042c92291a74580921a223047c08d87"
PREPARATION_SHA256 = "e94b3ef28a4a748cd732bf79bd9981109dde0ce8c92dfcc4f0a6a8395665c5da"
SHARED_SOFTWARE_SHA256 = "bfd50821f4f861822e379f30ef0b1823bda11a990879ca255ca787cd11236cd1"
NUMERICAL_MODULES = {"jax", "jaxlib", "numpy", "torch", "cupy", "flax", "optax", "scipy", "jaxtari", "exploration_platform"}
ATTEMPT_NAMES = (
    "direct_controller_intent.json", "direct_controller_spawn.json",
    "direct_controller_failure.json", "direct_controller_outcome.json", "direct_controller_unidentified_wait.json",
    "direct_controller_stdout.raw", "direct_controller_stderr.raw",
    "direct_controller_output_hashes.json",
    "direct_controller_timeout_observation.json",
    "direct_current_launch.json", "direct_current_launch.json.prepared",
    "direct_current_process_ledger_snapshot.json",
    "direct_process_wait_return.json", "direct_terminal_process.json",
    "direct_process_metadata.json", "direct_device_binding.json",
    "direct_device_binding_attempt.json", "direct_qualification_result.json", "result.json",
    "failure.json", "direct_startup_checks.json", "captured_driver",
)
BOUNDARY = {"scientific_credit": 0, "case_credit": 0, "idea_credit": 0,
            "worker_admitted": False, "admission_granted": False,
            "controller_owns_gpu_standing_descriptor": False,
            "producer_owns_gpu_standing_descriptor": True}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def now():
    return datetime.now(timezone.utc).isoformat()


def encoded(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def fsync_directory(path):
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def ordinary_bytes(path, bound=16 * 1024 * 1024):
    """Pin every ordinary directory and the file before opening it for reading."""
    path = Path(path)
    require(path.is_absolute() and ".." not in path.parts, "an exact absolute ordinary path is required")
    descriptor = os.open("/", os.O_PATH | os.O_DIRECTORY)
    try:
        for part in path.parts[1:-1]:
            following = os.open(part, os.O_PATH | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = following
        pinned = os.open(path.name, os.O_PATH | os.O_NOFOLLOW, dir_fd=descriptor)
        try:
            before = os.fstat(pinned)
            require(stat.S_ISREG(before.st_mode) and 0 <= before.st_size <= bound,
                    "an ordinary bounded regular file is required: " + str(path))
            reader = os.open("/proc/self/fd/" + str(pinned), os.O_RDONLY)
            try:
                raw = b""
                while len(raw) <= bound:
                    block = os.read(reader, min(65536, bound + 1 - len(raw)))
                    if not block:
                        break
                    raw += block
                require(len(raw) <= bound and stable_stat(os.fstat(reader)) == stable_stat(before), "file changed during its bounded read")
            finally:
                os.close(reader)
            require(stable_stat(os.stat(path, follow_symlinks=False)) == stable_stat(before), "ordinary input changed its inode or metadata")
            return raw
        finally:
            os.close(pinned)
    finally:
        os.close(descriptor)


def document(path):
    return json.loads(ordinary_bytes(path))


def stable_stat(info):
    return tuple(getattr(info, name) for name in ("st_dev", "st_ino", "st_mode", "st_uid", "st_gid", "st_size", "st_mtime_ns", "st_ctime_ns"))


def receipt(path):
    return {"path": str(path), "sha256": digest(ordinary_bytes(path))}


def log_receipt(path):
    """Hash an actual output file in chunks without loading the complete log."""
    path = Path(path)
    require(path.parent.resolve(strict=True) == path.parent, "the actual output parent changed")
    pinned = os.open(path, os.O_PATH | os.O_NOFOLLOW)
    try:
        before = os.fstat(pinned)
        require(stat.S_ISREG(before.st_mode), "actual output is not an ordinary regular file")
        reader = os.open("/proc/self/fd/" + str(pinned), os.O_RDONLY)
        try:
            value = hashlib.sha256()
            count = 0
            while block := os.read(reader, 65536):
                value.update(block)
                count += len(block)
            require(count == before.st_size and stable_stat(os.fstat(reader)) == stable_stat(before)
                    and stable_stat(os.stat(path, follow_symlinks=False)) == stable_stat(before),
                    "actual output changed after the exact child wait")
        finally:
            os.close(reader)
        return {"path": str(path), "sha256": value.hexdigest(), "size_bytes": count}
    finally:
        os.close(pinned)


def publish(path, value):
    """Publish complete durable bytes with no replacement, retaining a partial on failure."""
    path = Path(path)
    require(path.parent.resolve(strict=True) == path.parent, "publication directory must be ordinary")
    temporary = path.with_name(path.name + ".prepared")
    raw = value if isinstance(value, bytes) else encoded(value)
    with temporary.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    os.link(temporary, path, follow_symlinks=False)
    fsync_directory(path.parent)
    temporary.unlink()
    fsync_directory(path.parent)
    return receipt(path)


@contextmanager
def metadata_lock(path, expected_uid):
    """Use the campaign's existing metadata/process lock names, with ordinary paths."""
    path = Path(path)
    require(path.parent.resolve(strict=True) == path.parent, "metadata-lock parent changed")
    descriptor = os.open(path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        info = os.fstat(descriptor)
        require(stat.S_ISREG(info.st_mode) and info.st_uid == expected_uid
                and not info.st_mode & 0o022, "metadata lock is not an ordinary owned protected file")
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        os.close(descriptor)


def no_numerical_modules():
    require(not (NUMERICAL_MODULES & {name.split(".", 1)[0] for name in sys.modules}),
            "controller must remain standard-library metadata only")


def checked_inputs():
    """Validate the fixed packet without any scheduler, host, device, or child action."""
    no_numerical_modules()
    require(PACKET.resolve(strict=True) == PACKET and UNIT.resolve(strict=True) == UNIT,
            "the controller must use its exact ordinary packet and data paths")
    require(digest(ordinary_bytes(PACKET / "packet_metadata_freeze.json")) == PACKET_FREEZE_SHA256,
            "the original 66-file packet freeze changed")
    frozen = document(PACKET / "packet_metadata_freeze.json")
    require(len(frozen["files"]) == 66, "the original metadata freeze is not the exact 66-file roster")
    for name, expected in frozen["files"].items():
        require(Path(name).is_relative_to(RUN) and digest(ordinary_bytes(name)) == expected,
                "a frozen packet input changed: " + name)
    for name, expected in (("metadata_preparation_result.json", PREPARATION_SHA256),
                           ("shared_software_assets_staged_verification.json", SHARED_SOFTWARE_SHA256)):
        require(digest(ordinary_bytes(PACKET / name)) == expected, "actual metadata preparation proof changed")
    require(digest(ordinary_bytes(UNIT / "measurement_plan.json")) == PLAN_SHA256, "the frozen unit plan changed")
    plan = document(UNIT / "measurement_plan.json")
    require(plan["campaign_run_dir"] == str(RUN) and plan["unit_id"] == UNIT.name
            and plan["helpers"]["source_commit"] == HELPER_COMMIT
            and plan["source_commit"] == SOURCE_COMMIT and plan["gpu_uuid"] == GPU_UUID
            and plan["hardware_class"] == "h100" and plan["direct_binding"]["consumer"] == "selected_measurement"
            and plan["runtime_environment"] == {"JAX_ENABLE_X64": "true",
                "JAX_DEFAULT_MATMUL_PRECISION": "highest",
                "XLA_FLAGS": "--xla_gpu_exclude_nondeterministic_ops=true"},
            "the direct qualification source, selector or runtime changed")
    command = [plan["python"], str(UNIT / "code/measure_selected.py"), "--prepared-unit", str(UNIT)]
    require(shlex.split(ordinary_bytes(UNIT / "command.txt").decode()) == command,
            "the intended Popen command differs from the frozen producer command")
    return plan, command, frozen


def check_unattempted(unit):
    for name in ATTEMPT_NAMES:
        path = Path(unit) / name
        require(not path.exists() and not path.is_symlink()
                and not path.with_name(path.name + ".prepared").exists(),
                "an earlier or partial immutable launch attempt exists: " + name)


def check_publication(commit):
    require(isinstance(commit, str) and re.fullmatch(r"[0-9a-f]{40}", commit),
            "the exact reviewed publication commit is required")
    frozen = document(PACKET / "controller_source_freeze.json")
    require(type(frozen.get("schema_version")) is int and frozen["schema_version"] == 1
            and frozen.get("kind") == "direct_qualification_controller_source_freeze"
            and isinstance(frozen.get("files"), dict)
            and str(PACKET / "direct_qualification_controller.py") in frozen["files"]
            and str(PACKET / "test_direct_qualification_controller.py") in frozen["files"],
            "the controller freeze must bind the actual controller and regression source")
    review_path = PACKET / "controller_root_independent_review.json"
    review = document(review_path)
    require(type(review.get("schema_version")) is int and review["schema_version"] == 1
            and review.get("kind") == "direct_qualification_controller_root_independent_review"
            and review.get("passed") is True and review.get("launch_approved") is True
            and review.get("controller_freeze") == receipt(PACKET / "controller_source_freeze.json")
            and review.get("packet_freeze") == receipt(PACKET / "packet_metadata_freeze.json")
            and review.get("plan") == receipt(UNIT / "measurement_plan.json")
            and review.get("controller_owns_gpu_standing_descriptor") is False,
            "the actual independent root launch review is missing or differs")
    files = dict(frozen["files"])
    files.update(document(PACKET / "packet_metadata_freeze.json")["files"])
    for path in (PACKET / "controller_source_freeze.json", PACKET / "packet_metadata_freeze.json", review_path):
        files[str(path)] = digest(ordinary_bytes(path))
    for name, expected in files.items():
        path = Path(name)
        require(path.is_relative_to(INTEGRATION) and digest(ordinary_bytes(path)) == expected,
                "a reviewed publication input changed")
        raw = subprocess.check_output(["git", "-C", str(INTEGRATION), "cat-file", "blob",
                    commit + ":" + str(path.relative_to(INTEGRATION))], timeout=30)
        require(digest(raw) == expected, "a reviewed input is not the exact published Git byte sequence")
    return receipt(review_path)


def load_core(unit):
    """Import only the exact staged standard-library identity helpers."""
    path = Path(unit) / "code/serval02_card1_binding.py"
    specification = importlib.util.spec_from_file_location("direct_qualification_controller_core", path)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    no_numerical_modules()
    return module


def read_process(pid, expected_uid):
    """Read only this actual own process; /proc kernel aliases are intentional."""
    require(type(pid) is int and pid > 0, "an actual positive returned PID is required")
    directory = Path("/proc") / str(pid)
    require(directory.stat().st_uid == expected_uid, "the actual process is not owned by this user")
    status = (directory / "status").read_bytes()
    uids = b"".join(line + b"\n" for line in status.splitlines() if line.startswith(b"Uid:"))
    require(re.fullmatch(rb"Uid:\s*" + str(expected_uid).encode() + rb"\s+"
        + str(expected_uid).encode() + rb"\s+" + str(expected_uid).encode() + rb"\s+"
        + str(expected_uid).encode() + rb"\s*", uids), "all four kernel UIDs must be the exact owner")
    raw_stat = (directory / "stat").read_bytes()
    text = raw_stat.decode("utf-8", errors="strict")
    fields = text[text.rfind(")") + 2:].split()
    require(text.split(" ", 1)[0] == str(pid) and len(fields) > 19
            and re.fullmatch(r"[1-9][0-9]*", fields[19]), "the actual process birth is invalid")
    raw_command = (directory / "cmdline").read_bytes()
    command = [part.decode("utf-8", errors="strict") for part in raw_command.rstrip(b"\0").split(b"\0")] if raw_command else []
    affinity = sorted(os.sched_getaffinity(pid))
    repeated_stat = (directory / "stat").read_text()
    repeated_fields = repeated_stat[repeated_stat.rfind(")") + 2:].split()
    require(repeated_fields[19] == fields[19] and directory.stat().st_uid == expected_uid,
            "the actual process identity changed during observation")
    return {"pid": pid, "hostname": socket.gethostname().split(".")[0],
        "process_start_time": fields[19], "parent_pid": int(fields[1]), "uid": expected_uid,
        "stat": text, "stat_sha256": digest(raw_stat), "uid_bytes": uids.decode(),
        "uid_sha256": digest(uids), "command": command,
        "command_bytes_hex": raw_command.hex(), "command_sha256": digest(raw_command),
        "affinity": affinity, "observed_at_utc": now()}


def owner_record(process):
    return {key: process[key] for key in ("pid", "stat", "stat_sha256", "uid_bytes", "uid_sha256")}


def chosen_affinity(available, count):
    require(type(count) is int and count > 0 and isinstance(available, list)
            and all(type(cpu) is int and cpu >= 0 for cpu in available)
            and len(set(available)) == len(available) and len(available) >= count,
            "the actual allowed logical-CPU mask is too small or invalid")
    return sorted(available)[:count]


def child_environment(plan, inherited):
    """Use the frozen numerical controls and remove inherited Slurm/cache settings."""
    # Only these nonsecret path/locale settings may enter the exact child environment.
    environment = {"PATH": "/usr/local/bin:/usr/bin:/bin", "LANG": "C.UTF-8", "TZ": "UTC",
        "HOME": inherited.get("HOME", pwd.getpwuid(os.getuid()).pw_dir),
        "USER": "sl5nw", "LOGNAME": "sl5nw", "LD_LIBRARY_PATH": "/usr/lib/nvidia"}
    environment.update(plan["runtime_environment"])
    environment.update(CUDA_VISIBLE_DEVICES=GPU_UUID, PYTHONNOUSERSITE="1", PYTHONDONTWRITEBYTECODE="1",
        JAX_PLATFORMS="cuda", JAX_ENABLE_COMPILATION_CACHE="false", XLA_PYTHON_CLIENT_PREALLOCATE="false",
        JAXTARI_SKIP_SPRITE_UPDATE="1", OMP_NUM_THREADS="8", OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1")
    return environment


def append_registration(unit, plan, child, observed_argv, expected_uid):
    """Fsync the actual ledger and copied snapshot before exposing the launch path."""
    unit = Path(unit)
    run = Path(plan["campaign_run_dir"])
    require(child["command"] == observed_argv, "actual kernel argv differs from the exact Popen command")
    row = {"hostname": plan["direct_binding"]["profile"]["hostname"], "gpu_uuid": plan["gpu_uuid"],
           "kind": plan["kind"], "unit": plan["unit_id"], "source_commit": plan["source_commit"],
           "source_root": plan["source_root"], "helper_commit": plan["helpers"]["source_commit"],
           "pid": child["pid"], "start_time": child["process_start_time"]}
    ledger = run / "processes.json"
    with metadata_lock(run / "processes.lock", expected_uid):
        prior_raw = ordinary_bytes(ledger)
        records = json.loads(prior_raw)
        require(isinstance(records, list) and all(isinstance(item, dict) for item in records),
                "the live campaign process ledger is not a record list")
        require(not any(item.get("unit") == plan["unit_id"] or (
            item.get("hostname") == row["hostname"] and item.get("pid") == row["pid"]
            and item.get("start_time") == row["start_time"]) for item in records),
                "the current launch is already recorded or duplicated")
        records.append(row)
        temporary = ledger.with_name("processes.json.direct-" + str(child["pid"]) + ".prepared")
        with temporary.open("xb") as stream:
            stream.write(encoded(records))
            stream.flush()
            os.fsync(stream.fileno())
        require(ordinary_bytes(ledger) == prior_raw, "the live ledger changed outside its exact lock")
        os.replace(temporary, ledger)
        fsync_directory(run)
        publish(unit / "direct_current_process_ledger_snapshot.json", records)
        launch = {"schema_version": 1, "kind": "serval02_card1_current_launch",
            "plan": receipt(unit / "measurement_plan.json"), "process_row": row,
            "process_ledger": {"source_path": str(ledger),
                "snapshot": receipt(unit / "direct_current_process_ledger_snapshot.json")}}
        publish(unit / "direct_current_launch.json", launch)
    return launch


def wait_exact(child, identity, expected_uid, seconds):
    """Bound only our actual spawned PID after its own UID and birth are rechecked."""
    timeouts = []
    try:
        return child.wait(timeout=seconds), timeouts
    except subprocess.TimeoutExpired:
        require(identity is not None, "unknown child birth cannot be signaled; retain the actual child and no-successor rule")
        for action, wait_seconds in ((signal.SIGTERM, 120), (signal.SIGKILL, None)):
            observed = read_process(child.pid, expected_uid)
            require(observed["process_start_time"] == identity["process_start_time"],
                    "timeout child PID was reused; no signal was sent")
            timeouts.append({"signal": int(action), "observed_identity": observed, "at_utc": now()})
            os.kill(child.pid, action)
            try:
                return child.wait(timeout=wait_seconds), timeouts
            except subprocess.TimeoutExpired:
                continue
    raise RuntimeError("the exact child wait produced no result")


def supervise(plan, command, unit, environment, *, expected_uid, cpu_count,
              registration_hostname, seconds=10800, after_spawn=None, launch_lock=None, prelaunch_check=None):
    """Common real-process mechanism; tests use only private CPU fixture paths.

    The production entry has fixed paths/identity/publication checks. No command,
    hostname, CPU count, timeout or environment override is exposed by the CLI.
    """
    unit = Path(unit)
    no_numerical_modules()
    check_unattempted(unit)
    require(registration_hostname == plan["direct_binding"]["profile"]["hostname"], "registration host changed")
    parent = read_process(os.getpid(), expected_uid)
    original_affinity = parent["affinity"]
    selected_affinity = chosen_affinity(original_affinity, cpu_count)
    intent = {"schema_version": 1, "kind": "one_direct_qualification_controller_attempt",
        "at_utc": now(), "plan": receipt(unit / "measurement_plan.json"), "command": command,
        "intended_environment": environment, "controller": parent,
        "available_cpu_affinity": original_affinity, "selected_cpu_affinity": selected_affinity,
        "cpu_affinity_is_exclusive": False, "slurm_cpu_allocation_claimed": False,
        "actual_child_pid": None, **BOUNDARY}
    publish(unit / "direct_controller_intent.json", intent)
    child = None
    child_identity = None
    failure = None
    wait_record = None
    stdout = stderr = None
    try:
        with metadata_lock(launch_lock, expected_uid) if launch_lock is not None else nullcontext():
            if prelaunch_check is not None:
                prelaunch_check()
            os.sched_setaffinity(0, selected_affinity)
            require(sorted(os.sched_getaffinity(0)) == selected_affinity, "controller CPU affinity did not take effect")
            stdout = (unit / "direct_controller_stdout.raw").open("xb", buffering=0)
            stderr = (unit / "direct_controller_stderr.raw").open("xb", buffering=0)
            started = now()
            child = subprocess.Popen(command, env=environment, cwd=unit, stdin=subprocess.DEVNULL,
                stdout=stdout, stderr=stderr, shell=False, close_fds=True)
            # Popen has returned a real PID; nothing below invents a future identity.
            child_identity = read_process(child.pid, expected_uid)
            require(child_identity["parent_pid"] == os.getpid() and child_identity["command"] == command
                    and child_identity["affinity"] == selected_affinity, "actual child parent, argv or inherited CPU affinity differs")
            spawn = {"schema_version": 1, "kind": "actual_direct_qualification_popen_return",
                "returned_at_utc": now(), "started_at_utc": started, "popen_pid": child.pid,
                "command": command, "observed_child": child_identity, "controller": read_process(os.getpid(), expected_uid),
                "cpu_affinity_is_exclusive": False, "slurm_cpu_allocation_claimed": False, **BOUNDARY}
            publish(unit / "direct_controller_spawn.json", spawn)
            if after_spawn is not None:
                after_spawn(child, child_identity)  # CPU tests only; never provided by the fixed CLI.
            current_child = read_process(child.pid, expected_uid)
            require(current_child["process_start_time"] == child_identity["process_start_time"]
                    and current_child["parent_pid"] == os.getpid() and current_child["command"] == command
                    and current_child["affinity"] == selected_affinity, "actual child exited or changed before durable registration")
            append_registration(unit, plan, child_identity, command, expected_uid)
    except BaseException as error:
        failure = {"type": type(error).__name__, "error": str(error), "at_utc": now()}
    finally:
        try:
            if child is not None:
                try:
                    returncode, timeouts = wait_exact(child, child_identity, expected_uid, seconds)
                except BaseException as error:
                    # An identity-read error must never bypass the actual wait or start a successor.
                    failure = failure or {"type": type(error).__name__, "error": str(error), "at_utc": now()}
                    returncode, timeouts = child.wait(), [{"type": type(error).__name__, "error": str(error)}]
                if timeouts:
                    failure = failure or {"type": "ControllerTimeoutOrWaitFailure",
                        "error": "the child did not complete within an uninterrupted exact controller wait",
                        "actual_returncode": returncode, "at_utc": now()}
                for stream in (stdout, stderr):
                    if stream is not None:
                        os.fsync(stream.fileno())
                if child_identity is not None:
                    wait_record = {"schema_version": 1, "kind": (
                            "owned_direct_process_wait_return_after_controller_intervention" if timeouts
                            else "owned_direct_process_wait_return"),
                        "command": command, "pid": child.pid, "hostname": registration_hostname,
                        "process_start_time": child_identity["process_start_time"], "uid": expected_uid,
                        "waited_pid": child.pid, "wait_semantics": "Popen.wait on this exact spawned child",
                        "returncode": returncode, "exited_normally": returncode >= 0,
                        "waiter_before": owner_record(parent), "waiter_after": owner_record(read_process(os.getpid(), expected_uid)),
                        "finished_at_utc": now(), "timeouts": timeouts,
                        **BOUNDARY}
                    publish(unit / "direct_process_wait_return.json", wait_record)
                    if timeouts:
                        publish(unit / "direct_controller_timeout_observation.json", {
                            "schema_version": 1, "kind": "direct_child_interrupted_controller_wait",
                            "wait": receipt(unit / "direct_process_wait_return.json"),
                            "actual_returncode": returncode, "actual_exited_normally": returncode >= 0,
                            "timeouts": timeouts, "ordinary_successful_terminal_proof_allowed": False,
                            **BOUNDARY})
                    publish(unit / "direct_controller_output_hashes.json", {
                        "schema_version": 1, "kind": "actual_direct_child_output_hashes_after_wait",
                        "wait": receipt(unit / "direct_process_wait_return.json"),
                        "stdout": log_receipt(unit / "direct_controller_stdout.raw"),
                        "stderr": log_receipt(unit / "direct_controller_stderr.raw"), **BOUNDARY})
                    if returncode == 0 and failure is None and not timeouts:
                        require((unit / "direct_current_launch.json").is_file(), "ordinary child success lacks durable launch proof")
                        publish(unit / "direct_terminal_process.json", {
                            "schema_version": 1, "kind": "direct_serval02_card1_terminal_process",
                            "launch": receipt(unit / "direct_current_launch.json"), "pid": child.pid,
                            "hostname": registration_hostname, "process_start_time": child_identity["process_start_time"],
                            "uid": expected_uid, "return_record": receipt(unit / "direct_process_wait_return.json"), **BOUNDARY})
                else:
                    publish(unit / "direct_controller_unidentified_wait.json", {
                        "kind": "actual_popen_wait_without_complete_identity", "pid": child.pid,
                        "command": command, "returncode": returncode, "at_utc": now(), **BOUNDARY})
        finally:
            for stream in (stdout, stderr):
                if stream is not None:
                    stream.close()
            os.sched_setaffinity(0, original_affinity)
            require(sorted(os.sched_getaffinity(0)) == original_affinity, "controller original CPU affinity was not restored")
    if failure is not None:
        publish(unit / "direct_controller_failure.json", {**failure, "schema_version": 1,
            "kind": "direct_qualification_controller_failure", "actual_child_pid": None if child is None else child.pid,
            "actual_wait_saved": wait_record is not None, **BOUNDARY})
    outcome = {"schema_version": 1, "kind": "direct_qualification_controller_outcome",
        "actual_child_started": child is not None, "actual_wait_saved": wait_record is not None,
        "passed": failure is None and wait_record is not None and wait_record["returncode"] == 0
            and not wait_record["timeouts"],
        "failure": failure, "qualification_independently_verified": False, **BOUNDARY}
    publish(unit / "direct_controller_outcome.json", outcome)
    return outcome


def actual_launch(commit):
    """No remote transport or automatic retry is implemented."""
    plan, command, _ = checked_inputs()
    check_publication(commit)
    require(socket.gethostname().split(".")[0] == HOST and pwd.getpwuid(os.getuid()).pw_name == "sl5nw"
            and os.getuid() == UID and os.geteuid() == UID and tuple(os.getresuid()) == (UID, UID, UID),
            "actual mode is restricted to the exact approved direct host and owner")
    require(not any(name.startswith("SLURM_") for name in os.environ),
            "actual direct mode cannot run in a Slurm allocation")
    own = read_process(os.getpid(), UID)
    require(own["hostname"] == HOST, "the actual controller host changed")
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == GPU_UUID,
            "actual controller must retain the exact reviewed original UUID selector")
    core = load_core(UNIT)

    def prelaunch_check():
        checked_inputs()
        core.validate_plan(plan, "selected_measurement")
        core.source_identity(plan)
        core.validate_history(plan)
        core.check_live_admission(core.current_admission_claim(plan))

    environment = child_environment(plan, os.environ)
    return supervise(plan, command, UNIT, environment, expected_uid=UID, cpu_count=8,
        registration_hostname=HOST, launch_lock=RUN / "queue/metadata.lock", prelaunch_check=prelaunch_check)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--launch", action="store_true")
    parser.add_argument("--publication-commit")
    options = parser.parse_args(argv)
    if options.launch:
        result = actual_launch(options.publication_commit)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result["passed"] else 1
    plan, command, _ = checked_inputs()
    check_unattempted(UNIT)
    if options.publication_commit is not None:
        check_publication(options.publication_commit)
    print(json.dumps({"kind": "direct_qualification_controller_default_check", "passed": True,
        "action_performed": False, "command": command, "plan": receipt(UNIT / "measurement_plan.json"),
        "actual_child_pid": None, "publication_required_before_launch": True, **BOUNDARY}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

