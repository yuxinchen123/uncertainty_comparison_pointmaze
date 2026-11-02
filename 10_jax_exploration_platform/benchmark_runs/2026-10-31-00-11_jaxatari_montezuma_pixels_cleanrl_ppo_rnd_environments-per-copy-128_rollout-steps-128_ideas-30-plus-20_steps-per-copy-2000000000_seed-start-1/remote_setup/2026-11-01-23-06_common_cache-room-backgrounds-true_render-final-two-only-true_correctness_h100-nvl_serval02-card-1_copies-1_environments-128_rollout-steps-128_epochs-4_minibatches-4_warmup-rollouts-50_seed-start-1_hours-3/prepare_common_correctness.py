"""Prepare one common correctness unit from the accepted direct qualification and immutable original history."""

from datetime import datetime, timezone
import ctypes
import hashlib
import importlib.abc
import json
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
import sys
import traceback

P = Path(__file__).resolve(strict=True).parent
REQUEST_SHA = "f6843f5e60e62ab539cf4e4b0f5a605452e4b448f000b56530e3f5a8e511be0f"
HISTORY_INPUT_SHA = "3da2c4491a0c9c123b7de529e0e25c916e7654bb0fd535fcb0169edde49860e0"
NUMERICAL = {"jax", "jaxlib", "numpy", "torch", "flax", "optax", "cupy", "scipy", "jaxtari", "exploration_platform"}


def require(value, message):
    if not value:
        raise ValueError(message)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def ordinary(path):
    path = Path(path)
    require(path.is_absolute() and path.resolve(strict=True) == path and stat.S_ISREG(path.lstat().st_mode),
        "exact ordinary metadata/source file required")
    return path


def receipt(path):
    path = ordinary(path)
    checksum = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            checksum.update(block)
    return {"path": str(path), "sha256": checksum.hexdigest()}


def read(path):
    return json.loads(ordinary(path).read_bytes())


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    require(path.parent.resolve(strict=True) == path.parent, "ordinary output directory required")
    raw = value if isinstance(value, bytes) else (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    with path.open("xb") as out:
        out.write(raw); out.flush(); os.fsync(out.fileno())
    descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return receipt(path)


def git(*arguments, allowed_returncodes=(0,)):
    command = ["git", "-c", "core.hooksPath=/dev/null", *map(str, arguments)]
    result = subprocess.run(command, capture_output=True, text=True)
    with (P / "preparation_git_commands.jsonl").open("a") as out:
        out.write(json.dumps({"command": command, "returncode": result.returncode, "stdout": result.stdout,
            "stderr": result.stderr}, sort_keys=True) + "\n"); out.flush(); os.fsync(out.fileno())
    require(result.returncode in allowed_returncodes, "a local source preparation command failed; preserve this attempt")
    return result.stdout.strip()


def clone(source, destination, commit, prefixes):
    require(Path(source).is_absolute() and not destination.exists(), "use a new local detached source destination")
    git("clone", "--shared", "--no-hardlinks", "--no-checkout", source, destination)
    git("-C", destination, "sparse-checkout", "init", "--cone")
    git("-C", destination, "sparse-checkout", "set", *prefixes)
    git("-C", destination, "checkout", "--quiet", "--detach", commit)
    require(git("-C", destination, "rev-parse", "HEAD") == commit
        and git("-C", destination, "status", "--porcelain", "--untracked-files=all") == "",
        "the exact detached source is not clean")
    return {"root": str(destination), "commit": commit, "clean": True}


def copy_payload(row, destination):
    original = ordinary(row["captured_path"]); descriptor = os.open(original, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        before = os.fstat(descriptor)
        require(stat.S_ISREG(before.st_mode) and not before.st_mode & 0o222 and before.st_size == row["bytes"],
            "the original captured payload must remain read-only and complete")
        checksum = hashlib.sha256(); count = 0
        with os.fdopen(descriptor, "rb", closefd=False) as incoming, destination.open("xb") as outgoing:
            for block in iter(lambda: incoming.read(1024 * 1024), b""):
                outgoing.write(block); checksum.update(block); count += len(block)
            outgoing.flush(); os.fsync(outgoing.fileno())
        after = os.fstat(descriptor)
        require((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) ==
            (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
            and count == row["bytes"] and checksum.hexdigest() == row["sha256"], "captured original payload changed during copy")
        destination.chmod(0o444)
    finally:
        os.close(descriptor)


def install_guards():
    require(not NUMERICAL.intersection(sys.modules), "the metadata interpreter already imported numerical modules")
    os.environ["GIT_ALLOW_PROTOCOL"] = "file"
    os.environ["GIT_CONFIG_NOSYSTEM"] = "1"
    os.environ["GIT_CONFIG_GLOBAL"] = "/dev/null"
    os.environ["GIT_LFS_SKIP_SMUDGE"] = "1"
    class RefuseNumerical(importlib.abc.MetaPathFinder):
        def find_spec(self, name, path=None, target=None):
            if name.split(".")[0] in NUMERICAL:
                raise RuntimeError("numerical import forbidden during preparation: " + name)
    sys.meta_path.insert(0, RefuseNumerical())
    ctypes.CDLL = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("shared-library load forbidden"))
    real_popen = subprocess.Popen
    class LocalGitOnly(real_popen):
        def __init__(self, arguments, *args, **kwargs):
            command = list(arguments) if isinstance(arguments, (tuple, list)) else []
            require(command and command[0] == "git" and not any(word in command for word in
                ("push", "fetch", "pull", "remote", "ls-remote")), "only local Git metadata processes are allowed")
            if "clone" in command:
                require(Path(command[-2]).is_absolute() and Path(command[-1]).is_absolute(),
                    "a source clone cannot contact a network repository")
            super().__init__(arguments, *args, **kwargs)
    subprocess.Popen = LocalGitOnly
    def audit(event, arguments):
        if event in {"socket.connect", "socket.bind", "ctypes.dlopen"}:
            raise RuntimeError("network or shared-library operation forbidden")
        if event == "open" and isinstance(arguments[0], str):
            require(not arguments[0].startswith("/dev/") and arguments[0] != "/tmp/sl5nw_serval02_card1.lock",
                "no GPU device or standing lock access during metadata preparation")
    sys.addaudithook(audit)


def fresh_running_guard():
    require(not NUMERICAL.intersection(sys.modules), "metadata preparation must precede numerical imports")
    request = read(P / "preparation_request.json"); campaign = Path(request["campaign_run_dir"])
    rows = [line.split("\t") for line in (campaign / "slurm/submissions.tsv").read_text().splitlines() if line.split("\t", 1)[0] == "6783003"]
    for name in ("submitted_jobids.txt", "submitted_jobids_uva.txt"):
        require((campaign / "slurm" / name).read_text().splitlines().count("6783003") == 1, "owned compute ID missing or duplicated")
    require(len(rows) == 1 and rows[0][1:3] == ["monitor", "79a01e851ff92ddba7d0e12eaf8164cab2028878"], "owned compute source row changed")
    command = ["scontrol", "show", "job", "-o", "6783003"]
    returned = subprocess.run(command, capture_output=True, text=True)
    value = {"schema_version": 1, "kind": "fresh_owned_bound_preparation_running_guard", "job_id": "6783003", "step_id": os.environ["SLURM_STEP_ID"], "command": command, "returncode": returned.returncode, "stdout": returned.stdout, "stderr": returned.stderr, "passed": False}
    try:
        def field(name):
            found = re.findall(r"(?:^|\s)" + re.escape(name) + r"=([^\s]*)", returned.stdout)
            require(len(found) == 1, "missing/repeated owned field: " + name); return found[0]
        require(returned.returncode == 0 and returned.stderr == "" and field("JobId") == "6783003"
            and field("JobState") == "RUNNING" and field("UserId") == "sl5nw(2618919)"
            and field("Command") == rows[0][-1], "current owned CPU allocation is not running")
        value["passed"] = True
    finally:
        write(P / "bound_preparation_running_guard.json", value)


def configuration_contract(request, qualified):
    """Keep the exact original configuration and change only the two supplied environment flags."""
    require(request["packet"] == str(P) and request["data_unit"] == str(Path(request["campaign_run_dir"]) / "data" / request["unit_id"]),
        "the copied request must name its own exact packet and data unit")
    reference = json.loads(json.dumps(qualified["configuration"]))
    copies = request["configuration"]["learner"]["n_copies"]
    require(type(copies) is int and copies in (1, 2), "only the two approved independent copy counts are prepared")
    reference["learner"]["n_copies"] = copies
    candidate = json.loads(json.dumps(reference))
    candidate["environment"]["cache_room_backgrounds"] = True
    candidate["environment"]["render_final_two_only"] = True
    require(request["reference_configuration"] == reference and request["configuration"] == candidate
        and request["copy_seeds"] == list(range(1, copies + 1))
        and request["mode"] == "correctness" and request["rounds"] == 7
        and request["minimum_updates_per_round"] == 10
        and request["timing_rounds_executed_by_correctness_mode"] is False,
        "the original configuration, two flags, seeds or correctness-only scope changed")
    for key in ("source_root", "source_commit", "python", "runtime_environment", "asset_record"):
        request_key = {"source_root": "numerical_source_root", "source_commit": "numerical_source_commit"}.get(key, key)
        require(request[request_key] == qualified[key], "the qualified original target changed: " + key)
    require(request["helper_source_root"] == qualified["helpers"]["source_root"]
        and request["helper_source_commit"] == qualified["helpers"]["source_commit"]
        and request["helper_source_commit"] == "0a880984b4a03a2ff7335337cc18576628bc9fc1"
        and request["gpu_uuid"] == "GPU-fed69954-5a71-dbb2-ee91-544e0eb6b8a3"
        and request["expected_staged_helper_count"] == 26,
        "correctness must reuse the exact qualified backend26 source and physical selector")
    return copies


def prepare():
    """Prepare this one new unit on the owned CPU; never contact the direct host or GPU."""
    require(os.environ.get("SLURM_JOB_ID") == "6783003" and os.environ.get("SLURM_STEP_ID", "").isdecimal()
        and os.getuid() == os.geteuid() == 2618919, "only the exact owned CPU monitor may prepare")
    require(receipt(P / "preparation_request.json")["sha256"] == REQUEST_SHA
        and receipt(P / "history_preparation_inputs.json")["sha256"] == HISTORY_INPUT_SHA,
        "the reviewed request or immutable history declarations changed")
    request = read(P / "preparation_request.json")
    require(request["packet"] == str(P) and request["data_unit"] == str(Path(request["campaign_run_dir"]) / "data" / request["unit_id"]),
        "the reviewed request must name its own packet and data unit before any query")
    fresh_running_guard()
    install_guards()
    unit = Path(request["data_unit"]); history = unit / "direct_history"
    require(unit == Path(request["campaign_run_dir"]) / "data" / request["unit_id"]
        and {path.name for path in unit.iterdir()} == {"direct_history", *request["initial_standard_data_records"]},
        "the prepared unit must be new, with only reviewed initial records and staged history")
    for name, expected in request["initial_standard_data_records"].items():
        require(receipt(unit / name)["sha256"] == expected["sha256"] and receipt(expected["path"]) == expected,
            "an initial standard data record or immutable archive changed")
    write(P / "metadata_preparation_intent.json", {"schema_version": 1,
        "kind": "one_direct_common_correctness_metadata_preparation", "request": receipt(P / "preparation_request.json"),
        "at_utc": datetime.now(timezone.utc).isoformat(), "job_id": "6783003",
        "step_id": os.environ["SLURM_STEP_ID"], "server_contact": False, "actual_direct_process": None})
    try:
        # Before: accepted single-copy original qualification. After: its unchanged bytes are supplied as a prerequisite.
        for value in [request["qualification_plan"], request["qualification_consumer_verification"],
                request["qualification_terminal"], request["qualification_controller_outcome"],
                *request["direct_qualification_inputs"].values()]:
            require(receipt(value["path"]) == value, "an actual accepted qualification input changed")
        qualified = read(request["qualification_plan"]["path"])
        copies = configuration_contract(request, qualified)
        require(read(request["qualification_consumer_verification"]["path"])["passed"] is True
            and read(request["qualification_controller_outcome"]["path"])["passed"] is True,
            "the actual previous full consumer and normal controller wait must have passed")
        source = Path(request["helper_source_root"])
        require(git("-C", source, "rev-parse", "HEAD") == request["helper_source_commit"]
            and git("-C", source, "status", "--porcelain", "--untracked-files=all") == ""
            and git("-C", source, "symbolic-ref", "--quiet", "HEAD", allowed_returncodes=(1,)) == "",
            "the already qualified private source must remain clean and detached")
        staged = read(P / "history_staging_result.json")
        require(staged["passed"] is True and staged["distinct_payloads"] == 646
            and staged["logical_mappings"] == 693 and staged["payload_bytes"] == 1309199145
            and staged["helper_source_commit"] is None and receipt(history / "inputs.json") == staged["history_inputs"],
            "the complete fresh per-unit original history is required")
        scripts = source / "10_jax_exploration_platform/scripts/atari_pixel_campaign"
        sys.path.insert(0, str(scripts))
        import measure_selected as selected
        import direct_measurement as direct
        import train_optimized as training
        require(Path(selected.__file__).resolve() == scripts / "measure_selected.py"
            and Path(direct.__file__).resolve() == scripts / "direct_measurement.py",
            "only the actual qualified private helper source may prepare")
        for name, document in (("reference_configuration.json", request["reference_configuration"]),
                ("candidate_configuration.json", request["configuration"]), ("runtime_environment.json", request["runtime_environment"]),
                ("asset_record.json", request["asset_record"]), ("direct_binding_profile.json", direct.core().PROFILE)):
            write(P / name, document)
        backend = request["direct_backend_image_record"]
        require(receipt(backend["path"]) == backend, "the actual prepared package image changed")
        write(P / "initial_data_record_transition.json", {"kind": "preserve_initial_records_before_correctness_preparation",
            "preserved_immutable_records": request["initial_standard_data_records"], "data_unit": str(unit)})
        for name, expected in request["initial_standard_data_records"].items():
            require(receipt(unit / name)["sha256"] == expected["sha256"], "initial data changed before its transition")
            (unit / name).unlink()
        descriptor = os.open(unit, os.O_RDONLY | os.O_DIRECTORY); os.fsync(descriptor); os.close(descriptor)
        arguments = ["--prepare-only", "--mode", "correctness", "--source-root", request["numerical_source_root"],
            "--source-commit", request["numerical_source_commit"], "--configuration", str(P / "candidate_configuration.json"),
            "--reference-configuration", str(P / "reference_configuration.json"), "--runtime-environment", str(P / "runtime_environment.json"),
            "--asset-record", str(P / "asset_record.json"), "--run-dir", request["campaign_run_dir"], "--unit-id", request["unit_id"],
            "--gpu-uuid", request["gpu_uuid"], "--hardware-class", "h100", "--seed-start", "1", "--python", request["python"],
            "--jaxtari-source-root", qualified["jaxtari_source_root"], "--direct-binding-profile", str(P / "direct_binding_profile.json"),
            "--direct-history-inputs", str(history / "inputs.json"), "--direct-backend-image-record", backend["path"],
            "--direct-qualification-record", request["direct_qualification_inputs"]["result"]["path"],
            "--direct-qualification-review", request["direct_qualification_inputs"]["review"]["path"]]
        parsed = selected.parse_arguments(arguments)
        require(parsed.prepare_only is True and not parsed.direct_target_qualification,
            "correctness preparation cannot execute or repeat qualification")
        require(selected.prepare(parsed) == unit, "the actual prepared data unit changed")
        plan = read(unit / "measurement_plan.json")
        verified_history = direct.core().validate_history(plan)
        direct.verify_saved_sources(unit, plan)
        accepted = direct.verify_qualification_inputs(plan)
        package = direct.backend_support(plan).validate_prepared(plan)
        require(len(plan["helpers"]["files"]) == 26 and plan["mode"] == "correctness"
            and plan["copy_seeds"] == request["copy_seeds"]
            and plan["configuration"] == request["configuration"]
            and plan["reference_configuration"] == request["reference_configuration"]
            and plan["direct_qualification_inputs"] == request["direct_qualification_inputs"]
            and not any(name in plan for name in ("direct_qualification", "target_qualification", "common_correctness_inputs",
                "completed_study", "final_confirmation_protocol", "device_memory_limit_protocol")),
            "this unit must remain ordinary intermediate correctness under actual accepted qualification")
        require(not any((unit / name).exists() for name in ("direct_current_launch.json", "direct_device_binding.json", "result.json")),
            "metadata preparation cannot invent an actual numerical process, binding or correctness result")
        write(unit / "progress.json", {"state": "PLANNED", "completed": False, "formal_plan_created": True,
            "gpu_process_started": False, "mode": "correctness", "copies": copies, "scientific_credit": 0,
            "case_credit": 0, "idea_credit": 0, "worker_admitted": False, "admission_granted": False})
        # Preserve the initial packet records in their immutable archive, then publish only the exact new formal records.
        for name in ("manifest.yaml", "config_resolved.yaml", "experiment_background.md", "command.txt"):
            require(receipt(P / name)["sha256"] == request["initial_standard_data_records"][name]["sha256"],
                "the initial packet record changed before its formal transition")
            temporary = P / (name + ".formal-prepared")
            write(temporary, (unit / name).read_bytes())
            os.replace(temporary, P / name)
        raw = Path("/proc/self/stat").read_text()
        value = {"schema_version": 1, "kind": "direct_common_correctness_metadata_preparation_result", "passed": True,
            "at_utc": datetime.now(timezone.utc).isoformat(), "job_id": "6783003", "step_id": os.environ["SLURM_STEP_ID"],
            "pid": os.getpid(), "birth": raw.rsplit(")", 1)[1].split()[19], "cgroup": Path("/proc/self/cgroup").read_text(),
            "plan": receipt(unit / "measurement_plan.json"), "prepared": receipt(unit / "measurement_prepared.json"),
            "request": receipt(P / "preparation_request.json"), "helper_files": plan["helpers"]["files"],
            "qualification_inputs": request["direct_qualification_inputs"], "backend_expected": plan["direct_backend_image"]["expected"],
            "actual_package_pin": package, "history": verified_history, "copy_seeds": plan["copy_seeds"],
            "standard_data_records": {name: receipt(unit / name) for name in request["initial_standard_data_records"]},
            "initial_archives": request["initial_standard_data_records"], "actual_process": None, "server_contact": False,
            "device_opened": False, "lock_probed": False, "numerical_imports": False, "correctness_performed": False,
            "scientific_credit": 0, "case_credit": 0, "idea_credit": 0, "worker_admitted": False, "admission_granted": False}
        require(not NUMERICAL.intersection(sys.modules), "metadata preparation imported a numerical module")
        write(P / "metadata_preparation_result.json", value)
        print(json.dumps({"passed": True, "packet": str(P), "data_unit": str(unit),
            "helpers": len(plan["helpers"]["files"]), "copies": copies, "comparisons": verified_history["comparisons"]}), flush=True)
    except BaseException as error:
        write(P / "metadata_preparation_failure.json", {"at_utc": datetime.now(timezone.utc).isoformat(),
            "error_type": type(error).__name__, "error": str(error), "traceback": traceback.format_exc(),
            "retry_permitted": False, "server_contact": False, "device_opened": False, "scientific_credit": 0})
        raise


if __name__ == "__main__":
    prepare()
