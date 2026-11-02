"""Prepare only original direct qualification metadata on the handed-over owned CPU allocation."""

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
REQUEST_SHA = "6cfdcbaf5b4fd4435add05bbfce8a27d328196460811e238da2ab6762e371e3e"
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


def git(*arguments):
    command = ["git", "-c", "core.hooksPath=/dev/null", *map(str, arguments)]
    result = subprocess.run(command, capture_output=True, text=True)
    with (P / "preparation_git_commands.jsonl").open("a") as out:
        out.write(json.dumps({"command": command, "returncode": result.returncode, "stdout": result.stdout,
            "stderr": result.stderr}, sort_keys=True) + "\n"); out.flush(); os.fsync(out.fileno())
    require(result.returncode == 0, "a local source preparation command failed; preserve this attempt")
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


def prepare():
    require(os.environ.get("SLURM_JOB_ID") == "6782816" and os.environ.get("SLURM_STEP_ID", "").isdecimal()
        and os.getuid() == 2618919, "use only the newly handed-over owned CPU monitor allocation")
    install_guards()
    require(receipt(P / "preparation_request.json")["sha256"] == REQUEST_SHA
        and receipt(P / "history_preparation_inputs.json")["sha256"] == HISTORY_INPUT_SHA, "fixed preparation request changed")
    request = read(P / "preparation_request.json"); inputs = read(P / "history_preparation_inputs.json")
    source = Path(request["helper_source_root"]); unit = Path(request["data_unit"]); history = unit / "direct_history"
    require(not unit.exists() and not source.exists(), "preserve every older source and data unit")
    write(P / "metadata_preparation_intent.json", {"schema_version": 1, "kind": "one_shot_direct_qualification_metadata_preparation",
        "job_id": "6782816", "step_id": os.environ["SLURM_STEP_ID"], "at_utc": datetime.now(timezone.utc).isoformat(),
        "request": receipt(P / "preparation_request.json"), "history_inputs": receipt(P / "history_preparation_inputs.json"),
        "actual_direct_process": None, "host_contact": False})
    try:
        source_record = clone("/p/rlprojects/RND_worktrees/merge-atari-cleanrl", source,
            request["helper_source_commit"], ["10_jax_exploration_platform/src", "10_jax_exploration_platform/scripts",
                "10_jax_exploration_platform/tests"])
        require(receipt(inputs["capture_manifest"]["path"]) == inputs["capture_manifest"], "the original capture manifest changed")
        capture = read(inputs["capture_manifest"]["path"]); rows = capture["logical_mappings"]
        require(capture["passed"] is True and len(rows) == inputs["logical_mappings"]
            and capture["unique_payloads"] == 646 and capture["physical_payload_bytes"] == 1309199145,
            "the full original captured proof is incomplete")
        content = history / "content"; content.mkdir(parents=True)
        payloads = {}
        for row in rows:
            require(re.fullmatch(r"[0-9a-f]{64}", row["sha256"]) is not None, "invalid original payload hash")
            if row["sha256"] not in payloads:
                target = content / (row["sha256"] + ".raw")
                copy_payload(row, target); payloads[row["sha256"]] = target
        require(len(payloads) == 646, "all 646 distinct original payloads are required")
        cloned = []
        for entry in inputs["git_sources"]:
            commit = entry["commit"]; destination = history / "git_sources" / ("original-source-commit-" + commit)
            require(not any(row["relative_path"] == str(destination.relative_to(history)) for row in cloned),
                "a distinct original source identity requires a distinct frozen clone")
            destination.parent.mkdir(exist_ok=True)
            prefixes = ["src"] if entry["original_root"] == "/p/rlprojects/RND/reference_repos/jaxtari_pixel_study" else [
                "10_jax_exploration_platform/src", "10_jax_exploration_platform/scripts", "10_jax_exploration_platform/tests"]
            clone(entry["transport_repository"], destination, commit, prefixes)
            cloned.append({"original_root": entry["original_root"], "relative_path": str(destination.relative_to(history)), "commit": commit})
        manifest = {"schema_version": 1, "kind": "portable_raw_evidence", "files": [
            {"original_path": row["logical_path"], "relative_path": "content/" + row["sha256"] + ".raw",
                "sha256": row["sha256"]} for row in rows], "git_roots": cloned}
        evidence_ref = write(history / "evidence_manifest.json", manifest)
        by_path = {row["logical_path"]: row for row in rows}; history_inputs = {
            "schema_version": 1, "kind": "serval02_card1_frozen_history", "evidence_manifest": evidence_ref}
        for name, record in inputs["records"].items():
            row = by_path[record["original_path"]]; require(row["sha256"] == record["sha256"], "original history identity changed")
            history_inputs[name] = write(history / (name + ".json"), payloads[row["sha256"]].read_bytes())
        write(history / "inputs.json", history_inputs)
        scripts = source / "10_jax_exploration_platform/scripts/atari_pixel_campaign"; sys.path.insert(0, str(scripts))
        import measure_selected as selected
        import direct_measurement as direct
        require(Path(selected.__file__).resolve() == scripts / "measure_selected.py"
            and Path(direct.__file__).resolve() == scripts / "direct_measurement.py", "only exact detached helpers may prepare this packet")
        write(P / "original_configuration.json", request["configuration"])
        write(P / "runtime_environment.json", request["runtime_environment"])
        write(P / "asset_record.json", request["asset_record"])
        write(P / "direct_binding_profile.json", direct.core().PROFILE)
        arguments = ["--prepare-only", "--mode", "correctness", "--source-root", request["numerical_source_root"],
            "--source-commit", request["numerical_source_commit"], "--configuration", str(P / "original_configuration.json"),
            "--reference-configuration", str(P / "original_configuration.json"), "--runtime-environment", str(P / "runtime_environment.json"),
            "--asset-record", str(P / "asset_record.json"), "--run-dir", request["campaign_run_dir"], "--unit-id", request["unit_id"],
            "--gpu-uuid", request["gpu_uuid"], "--hardware-class", "h100", "--seed-start", "1", "--python", request["python"],
            "--jaxtari-source-root", request["qualification_software"]["runtime_provenance"].get("jaxtari_source_root",
                "/p/rlprojects/RND/reference_repos/jaxtari_pixel_study"),
            "--direct-binding-profile", str(P / "direct_binding_profile.json"), "--direct-history-inputs", str(history / "inputs.json"),
            "--direct-target-qualification", "--cleanrl-source-root", request["cleanrl_reference"]["source_root"],
            "--qualification-software-result", request["qualification_software"]["result"]["path"]]
        parsed = selected.parse_arguments(arguments); require(parsed.prepare_only is True, "metadata preparation cannot execute the unit")
        prepared = selected.prepare(parsed); require(prepared == unit, "unexpected prepared data unit")
        plan = read(unit / "measurement_plan.json")
        verified_history = direct.core().validate_history(plan)
        direct.verify_saved_sources(unit, plan); direct.verify_qualification_source_receipts(plan)
        test_contract = direct.shared().test_collection_contract(plan)
        require(sum(test_contract.values()) == 48 and plan["renderer_contract"]["rooms"] == 33
            and plan["renderer_contract"]["cases"] == 132 and plan["renderer_contract"]["synthetic_pixel_cases"] == 11,
            "the complete original test or renderer gate matrix changed")
        require(plan["helpers"]["source_commit"] == request["helper_source_commit"]
            and plan["direct_qualification"] == direct.QUALIFICATION and plan["copy_seeds"] == [1]
            and not any(key in plan for key in ("direct_qualification_inputs", "completed_study", "final_confirmation_protocol")),
            "standalone direct qualification must have no circular baseline/fifty/prior-qualification dependency")
        require(not (unit / "direct_current_launch.json").exists() and not (unit / "direct_device_binding.json").exists(),
            "metadata preparation cannot invent an actual process or physical binding")
        raw = Path("/proc/self/stat").read_text(); identity = {"job_id": "6782816", "step_id": os.environ["SLURM_STEP_ID"],
            "pid": os.getpid(), "birth": raw.rsplit(")", 1)[1].split()[19], "cgroup": Path("/proc/self/cgroup").read_text()}
        result = {"schema_version": 1, "kind": "full_original_direct_qualification_metadata_preparation_result",
            "passed": True, "at_utc": datetime.now(timezone.utc).isoformat(), "compute": identity,
            "helper_source": source_record, "helper_count": len(plan["helpers"]["files"]), "plan": receipt(unit / "measurement_plan.json"),
            "prepared": receipt(unit / "measurement_prepared.json"), "history": verified_history,
            "logical_file_mappings": len(rows), "original_payloads": len(payloads), "original_payload_bytes": 1309199145,
            "historical_git_roots": cloned, "original_test_cases": 48, "original_test_contract": test_contract,
            "renderer_contract": plan["renderer_contract"], "actual_direct_process": None, "lock_probed": False,
            "device_opened": False, "server_contact": False, "numerical_imports": False, "scientific_credit": 0,
            "case_credit": 0, "idea_credit": 0, "worker_admitted": False, "admission_granted": False}
        require(git("-C", source, "status", "--porcelain", "--untracked-files=all") == "", "helper source became dirty")
        write(P / "metadata_preparation_result.json", result)
        print(json.dumps({"passed": True, "packet": str(P), "data_unit": str(unit), "helper_count": result["helper_count"],
            "comparisons": verified_history["comparisons"], "jobs": verified_history["stop"]["jobs"]}), flush=True)
    except BaseException as error:
        write(P / "metadata_preparation_failure.json", {"at_utc": datetime.now(timezone.utc).isoformat(), "job_id": "6782816",
            "step_id": os.environ["SLURM_STEP_ID"], "error_type": type(error).__name__, "error": str(error),
            "traceback": traceback.format_exc(), "retry_permitted": False, "actual_direct_process": None,
            "server_contact": False, "device_opened": False, "scientific_credit": 0})
        raise


if __name__ == "__main__":
    prepare()

