"""Stage immutable original history for one fresh direct common-correctness unit; start no producer."""
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import importlib.abc
import json
import os
from pathlib import Path
import re
import socket
import stat
import subprocess
import sys
import traceback

P = Path(__file__).absolute().parent
REQUEST_SHA = "938dabc08712584b39ae1807052fb43d03d9593880fa631e333f41b2e90af5f0"
HISTORY_SHA = "3da2c4491a0c9c123b7de529e0e25c916e7654bb0fd535fcb0169edde49860e0"
JOB = "6783003"
MONITOR_SOURCE = "79a01e851ff92ddba7d0e12eaf8164cab2028878"
NUMERICAL = {"jax", "jaxlib", "numpy", "torch", "flax", "optax", "cupy", "scipy", "jaxtari", "exploration_platform"}
BOUNDARY = {"direct_host_contact": False, "lock_probed": False, "gpu_work": False,
    "qualification_performed": False, "new_qualification_process_started": False,
    "helper_source_commit": None, "formal_qualification_plan": None, "producer_launch_command": None,
    "scientific_credit": 0, "case_credit": 0, "idea_credit": 0, "worker_admitted": False, "admission_granted": False}

def require(value, message):
    if not value:
        raise ValueError(message)

def encoded(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()

def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    require(path.parent.resolve(strict=True) == path.parent, "ordinary new output parent required")
    raw = value if isinstance(value, bytes) else encoded(value)
    with path.open("xb") as stream:
        stream.write(raw); stream.flush(); os.fsync(stream.fileno())
    descriptor = os.open(path.parent, os.O_DIRECTORY); os.fsync(descriptor); os.close(descriptor)
    return receipt(path)

@contextmanager
def pinned_reader(path):
    path = Path(path)
    require(path.is_absolute() and ".." not in path.parts, "exact ordinary source path required")
    parent = os.open("/", os.O_PATH | os.O_DIRECTORY)
    pinned = reader = None
    try:
        for part in path.parts[1:-1]:
            following = os.open(part, os.O_PATH | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent)
            os.close(parent); parent = following
        pinned = os.open(path.name, os.O_PATH | os.O_NOFOLLOW, dir_fd=parent)
        before = os.fstat(pinned)
        require(stat.S_ISREG(before.st_mode), "only a pinned ordinary regular file may be read")
        reader = os.open("/proc/self/fd/" + str(pinned), os.O_RDONLY)
        with os.fdopen(reader, "rb", closefd=False) as stream:
            yield stream, before
        after = os.fstat(reader)
        require(stable(before) == stable(after) == stable(os.stat(path, follow_symlinks=False)),
                "a captured source changed during its read")
    finally:
        if reader is not None: os.close(reader)
        if pinned is not None: os.close(pinned)
        os.close(parent)

def stable(value):
    return tuple(getattr(value, name) for name in ("st_dev", "st_ino", "st_mode", "st_uid", "st_gid",
        "st_size", "st_mtime_ns", "st_ctime_ns"))

def receipt(path):
    checksum = hashlib.sha256()
    with pinned_reader(path) as (stream, before):
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            checksum.update(block)
    return {"path": str(path), "sha256": checksum.hexdigest()}

def read(path):
    with pinned_reader(path) as (stream, before):
        require(before.st_size <= 16 * 1024 * 1024, "metadata document exceeds its bound")
        return json.loads(stream.read())

def run(command, allowed=(0,)):
    require(command[0] == "/usr/bin/git" or command == ["scontrol", "show", "job", "-o", JOB],
            "only local Git or the exact owned monitor query is allowed")
    if command[0] == "/usr/bin/git":
        require(not any(word in command for word in ("fetch", "push", "pull", "remote", "ls-remote")),
                "no network Git operation is allowed")
        if "clone" in command:
            require(Path(command[-2]).is_absolute() and Path(command[-1]).is_absolute(), "local absolute clone required")
    returned = subprocess.run(command, capture_output=True, text=True)
    with (P / "staging_commands.jsonl").open("a") as stream:
        stream.write(json.dumps({"command": command, "returncode": returned.returncode,
            "stdout": returned.stdout, "stderr": returned.stderr}, sort_keys=True) + "\n")
        stream.flush(); os.fsync(stream.fileno())
    require(returned.returncode in allowed, "metadata command failed; preserve this attempt")
    return returned

def field(raw, name):
    values = re.findall(r"(?:^|\s)" + re.escape(name) + r"=([^\s]*)", raw)
    require(len(values) == 1, "missing or repeated exact owned field: " + name)
    return values[0]

def fresh_running(request):
    campaign = Path(request["campaign_run_dir"])
    expected_script = campaign / "remote_setup" / (
        "2026-11-01-16-33_pixel-study-monitor-recovery_source-commit-" + MONITOR_SOURCE
        + "_processor-count-1_host-memory-gib-2_time-limit-days-7_poll-interval-seconds-1200_partition-nolim_account-sl5nw-base_gpu-count-0") / "monitor.sbatch"
    row = [JOB, "monitor", MONITOR_SOURCE, str(expected_script)]
    for name in ("submitted_jobids.txt", "submitted_jobids_uva.txt"):
        require((campaign / "slurm" / name).read_text().splitlines().count(JOB) == 1,
                "the current owned metadata allocation is missing or duplicated")
    actual_rows = [line.split("\t") for line in (campaign / "slurm/submissions.tsv").read_text().splitlines()
                   if line.split("\t", 1)[0] == JOB]
    require(actual_rows == [row], "the exact owned metadata source/row changed before any query")
    require(os.environ.get("SLURM_JOB_ID") == JOB and os.environ.get("SLURM_STEP_ID", "").isdigit()
            and os.getuid() == os.geteuid() == 2618919, "use only the current owned CPU monitor allocation")
    observed = run(["scontrol", "show", "job", "-o", JOB])
    raw = observed.stdout
    guard = {"schema_version": 1, "kind": "fresh_exact_owned_running_cpu_monitor_guard",
        "job_id": JOB, "step_id": os.environ["SLURM_STEP_ID"], "at_utc": datetime.now(timezone.utc).isoformat(),
        "command": ["scontrol", "show", "job", "-o", JOB], "returncode": observed.returncode,
        "stdout": raw, "stderr": observed.stderr, "actual_hostname": socket.gethostname(),
        "actual_pid": os.getpid(), "actual_cgroup": Path("/proc/self/cgroup").read_text(),
        "passed": False, **BOUNDARY}
    try:
        require(observed.stderr == "" and field(raw, "JobId") == JOB and field(raw, "JobState") == "RUNNING"
            and field(raw, "UserId") == "sl5nw(2618919)" and field(raw, "Account") == "sl5nw_base"
            and field(raw, "Partition") == "nolim" and field(raw, "QOS") == "normal"
            and field(raw, "NumCPUs") in {"1", "2"}
            and field(raw, "NodeList") == socket.gethostname().split(".")[0]
            and field(raw, "Command") == str(expected_script), "the exact owned CPU monitor is not running here")
        for name in ("ReqTRES", "AllocTRES"):
            resources = dict(value.split("=", 1) for value in field(raw, name).split(","))
            require(resources.get("mem") == "2G" and resources.get("node") == "1"
                and all(value == "0" for key, value in resources.items() if key.startswith("gres/")),
                "metadata work cannot inherit a GPU or different memory/node request")
        require("job_" + JOB in guard["actual_cgroup"], "the current process is not in the owned job cgroup")
        guard["passed"] = True
    finally:
        write(P / "fresh_owned_running_guard.json", guard)
    return guard

def clone(entry, destination):
    prefixes = ["src"] if entry["original_root"] == "/p/rlprojects/RND/reference_repos/jaxtari_pixel_study" else [
        "10_jax_exploration_platform/src", "10_jax_exploration_platform/scripts", "10_jax_exploration_platform/tests"]
    require(not destination.exists(), "each historical Git destination must be new")
    base = ["/usr/bin/git", "-c", "core.hooksPath=/dev/null"]
    run(base + ["clone", "--shared", "--no-hardlinks", "--no-checkout", entry["transport_repository"], str(destination)])
    run(base + ["-C", str(destination), "sparse-checkout", "init", "--cone"])
    run(base + ["-C", str(destination), "sparse-checkout", "set", *prefixes])
    run(base + ["-C", str(destination), "checkout", "--quiet", "--detach", entry["commit"]])
    require(run(base + ["-C", str(destination), "rev-parse", "HEAD"]).stdout.strip() == entry["commit"]
        and run(base + ["-C", str(destination), "status", "--porcelain", "--untracked-files=all"]).stdout == ""
        and run(base + ["-C", str(destination), "symbolic-ref", "--quiet", "HEAD"], allowed=(1,)).returncode == 1,
        "historical Git source must be exact, detached and clean")

def copy_payload(row, destination):
    checksum = hashlib.sha256(); count = 0
    with pinned_reader(row["captured_path"]) as (incoming, before):
        require(not before.st_mode & 0o222 and before.st_size == row["bytes"], "immutable captured bytes required")
        with destination.open("xb") as outgoing:
            for block in iter(lambda: incoming.read(1024 * 1024), b""):
                outgoing.write(block); checksum.update(block); count += len(block)
            outgoing.flush(); os.fsync(outgoing.fileno())
    require(count == row["bytes"] and checksum.hexdigest() == row["sha256"], "original captured payload hash changed")
    destination.chmod(0o444)
    require(receipt(destination)["sha256"] == row["sha256"], "new immutable payload copy differs")
    return count

def install_guards():
    require(not NUMERICAL.intersection(sys.modules), "numerical modules must not be loaded")
    class NoNumerical(importlib.abc.MetaPathFinder):
        def find_spec(self, name, path=None, target=None):
            if name.split(".")[0] in NUMERICAL:
                raise RuntimeError("numerical import forbidden during immutable-history staging")
    sys.meta_path.insert(0, NoNumerical())
    os.environ.update(GIT_ALLOW_PROTOCOL="file", GIT_CONFIG_NOSYSTEM="1", GIT_CONFIG_GLOBAL="/dev/null", GIT_LFS_SKIP_SMUDGE="1")
    def guard(event, args):
        if event.startswith("socket.") and event != "socket.gethostname" or event in {"ctypes.dlopen", "ctypes.dlsym"}:
            raise RuntimeError("network/library operation forbidden")
        if event == "open" and (str(args[0]).startswith(("/dev/nvidia", "/dev/dri"))
                                or str(args[0]) == "/tmp/sl5nw_serval02_card1.lock"):
            raise RuntimeError("GPU or standing-lock operation forbidden")
        if event == "subprocess.Popen":
            command = list(args[1])
            require(command[0] == "/usr/bin/git" or command == ["scontrol", "show", "job", "-o", JOB],
                    "only local Git and exact-owned monitor metadata query allowed")
    sys.addaudithook(guard)

def stage():
    install_guards()
    require(receipt(P / "preparation_request_unbound.json")["sha256"] == REQUEST_SHA
        and receipt(P / "history_preparation_inputs.json")["sha256"] == HISTORY_SHA, "fixed unbound metadata request changed")
    request = read(P / "preparation_request_unbound.json"); inputs = read(P / "history_preparation_inputs.json")
    require(request["packet"] == str(P) and request["data_unit"] == str(Path(request["campaign_run_dir"]) / "data" / request["unit_id"]), "the copied request must name its own exact packet and data unit")
    require(all(request[key] is None for key in ("helper_source_commit", "helper_source_root",
        "formal_qualification_plan", "producer_launch_command")), "this stage must not select a helper or launch")
    unit = Path(request["data_unit"]); history = unit / "direct_history"
    require(unit.is_dir() and {path.name for path in unit.iterdir()} == set(request["initial_standard_data_records"]), "preserve all earlier units and partial staging")
    for name, value in request["initial_standard_data_records"].items():
        require(receipt(unit / name)["sha256"] == value["sha256"] and receipt(value["path"]) == value, "initial standard data records changed before history staging")
    write(P / "history_staging_intent.json", {"schema_version": 1, "kind": "one_unbound_original_history_metadata_stage",
        "at_utc": datetime.now(timezone.utc).isoformat(), "job_id": JOB, "step_id": os.environ.get("SLURM_STEP_ID"),
        "request": receipt(P / "preparation_request_unbound.json"), "history_inputs": receipt(P / "history_preparation_inputs.json"),
        **BOUNDARY})
    guard = fresh_running(request)
    require(receipt(inputs["capture_manifest"]["path"]) == inputs["capture_manifest"], "original immutable capture manifest changed")
    capture = read(inputs["capture_manifest"]["path"]); rows = capture["logical_mappings"]
    require(capture["passed"] is True and len(rows) == 693 and capture["unique_payloads"] == 646
        and capture["physical_payload_bytes"] == 1309199145
        and len({row["logical_path"] for row in rows}) == 693, "complete original capture required")
    content = history / "content"; content.mkdir(parents=True)
    payloads = {}; byte_count = 0
    for row in rows:
        require(re.fullmatch(r"[0-9a-f]{64}", row["sha256"]), "invalid captured payload hash")
        if row["sha256"] not in payloads:
            target = content / (row["sha256"] + ".raw")
            byte_count += copy_payload(row, target); payloads[row["sha256"]] = target
            if len(payloads) % 128 == 0:
                print(json.dumps({"immutable_payloads_copied": len(payloads), "bytes": byte_count}), flush=True)
    require(len(payloads) == 646 and byte_count == 1309199145, "full immutable original bytes required")
    descriptor = os.open(content, os.O_DIRECTORY); os.fsync(descriptor); os.close(descriptor)
    cloned = []
    for entry in inputs["git_sources"]:
        require(re.fullmatch(r"[0-9a-f]{40}", entry["commit"]), "exact historical source commit required")
        destination = history / "git_sources" / ("original-source-commit-" + entry["commit"])
        destination.parent.mkdir(exist_ok=True)
        clone(entry, destination)
        cloned.append({"original_root": entry["original_root"], "relative_path": str(destination.relative_to(history)),
                       "commit": entry["commit"]})
    require(len(cloned) == 8, "all eight historical source identities are required")
    evidence = write(history / "evidence_manifest.json", {"schema_version": 1, "kind": "portable_raw_evidence",
        "files": [{"original_path": row["logical_path"], "relative_path": "content/" + row["sha256"] + ".raw",
                   "sha256": row["sha256"]} for row in rows], "git_roots": cloned})
    by_path = {row["logical_path"]: row for row in rows}
    history_inputs = {"schema_version": 1, "kind": "serval02_card1_frozen_history", "evidence_manifest": evidence}
    for name, expected in inputs["records"].items():
        row = by_path[expected["original_path"]]
        require(row["sha256"] == expected["sha256"], "original history identity changed")
        with pinned_reader(payloads[row["sha256"]]) as (stream, before):
            history_inputs[name] = write(history / (name + ".json"), stream.read())
    history_ref = write(history / "inputs.json", history_inputs)
    for name, value in request["initial_standard_data_records"].items():
        require(receipt(unit / name)["sha256"] == value["sha256"], "history staging altered an initial standard data record")
    require(not NUMERICAL.intersection(sys.modules), "staging loaded a numerical module")
    result = {"schema_version": 1, "kind": "unbound_helper_original_history_staging_completed", "passed": True,
        "at_utc": datetime.now(timezone.utc).isoformat(), "actual_job_id": JOB, "actual_step_id": os.environ["SLURM_STEP_ID"],
        "metadata_pid": os.getpid(), "metadata_hostname": socket.gethostname(),
        "fresh_running_guard": receipt(P / "fresh_owned_running_guard.json"),
        "history_inputs": history_ref, "logical_mappings": 693, "distinct_payloads": 646, "payload_bytes": byte_count,
        "historical_git_roots": cloned, "current_qualification_helper_selected": False,
        "formal_plan_created": False, "producer_command_created": False,
        "full_history_semantic_validation_by_future_reviewed_helper_pending": True,
        "numerical_modules_loaded": sorted(NUMERICAL.intersection(sys.modules)), **BOUNDARY}
    write(P / "history_staging_result.json", result)
    print(json.dumps({"passed": True, "payloads": 646, "logical_mappings": 693, "historical_git_roots": 8,
        "helper_source_commit": None, "formal_plan_created": False, "actual_qualification_process_started": False}), flush=True)

if __name__ == "__main__":
    try:
        stage()
    except BaseException as error:
        write(P / "history_staging_failure.json", {"kind": "unbound_history_staging_failed",
            "error_type": type(error).__name__, "error": str(error), "traceback": traceback.format_exc(), **BOUNDARY})
        raise

