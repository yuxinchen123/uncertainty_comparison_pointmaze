"""Review the new literal identities and unchanged read-only observer on owned CPU."""
import ast
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import time
import unittest

DIRECTORY = Path(__file__).resolve().parent
checks = []


def require(value, message):
    """Keep an explicit list of completed independent checks."""
    if not value:raise AssertionError(message)
    checks.append(message)


def reference(path):
    """Capture only bounded ordinary owned source and metadata bytes."""
    path = Path(path)
    require(path.resolve(strict=True) == path and not path.is_symlink() and path.is_file()
        and path.stat().st_uid == os.getuid() and path.stat().st_size <= 2500000, "bounded owned file: " + str(path))
    raw = path.read_bytes()
    return {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest(), "size_bytes": len(raw)}, raw


def audit(event, arguments):
    """Refuse actual target proc reads, numerical imports, libraries, networks and process actions."""
    if event == "import" and arguments[0].split(".")[0] in {"numpy", "jax", "jaxlib", "torch", "flax", "optax", "scipy", "cupy", "jaxtari"}:
        raise AssertionError("numerical import refused")
    if event in {"ctypes.dlopen", "subprocess.Popen", "os.system", "os.exec", "os.fork", "os.posix_spawn", "socket.connect", "socket.connect_ex", "os.kill", "os.killpg"}:
        raise AssertionError("process/library/network action refused: " + event)
    if event == "open" and isinstance(arguments[0], (str, bytes)):
        path = os.fsdecode(arguments[0])
        if path.startswith(("/proc/1042828", "/proc/1042870", "/dev/nvidia", "/dev/dri")):
            raise AssertionError("actual target/device read refused")


def main():
    """Bind genuine failed spawn/wait bytes and run only the four affected private-proc cases."""
    require(os.environ.get("SLURM_JOB_ID") == "6825983", "actual owned CPU allocation")
    started = time.monotonic();sys.addaudithook(audit)
    freeze_ref, raw = reference(DIRECTORY / "source_and_test_freeze.json")
    require(freeze_ref["sha256"] == "d1c364434f5ef9d7bf6470077a1d5e1684d6cf3115120cd691318312c8a98b52", "exact held source freeze")
    freeze = json.loads(raw)
    for name, expected in freeze["files"].items():
        actual, unused = reference(DIRECTORY / name)
        require(actual["sha256"] == expected["sha256"] and actual["size_bytes"] == expected["size_bytes"], "exact frozen bytes: " + name)
    derivation = json.loads((DIRECTORY / "identity_derivation.json").read_bytes())
    old_ref, old = reference(derivation["source_lineage"]["path"])
    require(old_ref == derivation["source_lineage"] and old_ref["sha256"] == "87bdf73927b03341ebe6b5b3d3600e8b77e62820b7f8c07f78c9ff5936ba9540", "accepted observer source lineage")
    source_ref, source = reference(DIRECTORY / "observe_exact_prior_h100_exits.py")
    old_text, new_text = old.decode(), source.decode()
    old_line = "IDENTITIES = {'controller': (386202, '84810007'), 'numerical_child': (386669, '84815419')}"
    new_line = "IDENTITIES = {'controller': (1042828, '87798085'), 'numerical_child': (1042870, '87798186')}"
    require(old_text.count(old_line) == 1 and old_text.replace(old_line, new_line) == new_text, "whole source changes only the two literal identities")
    functions = lambda text: {node.name: ast.get_source_segment(text, node) for node in ast.parse(text).body if isinstance(node, ast.FunctionDef)}
    require(functions(old_text) == functions(new_text) and len(functions(new_text)) == 9, "all nine approved function texts unchanged")
    records = {}
    for key in ("actual_spawn", "actual_failed_wait"):
        actual, raw = reference(derivation[key]["path"])
        require(actual == derivation[key], "actual immutable " + key)
        records[key] = json.loads(raw)
    spawn, waited = records["actual_spawn"], records["actual_failed_wait"]
    for role, expected in (("controller", (1042828, "87798085")), ("child", (1042870, "87798186"))):
        record = spawn[role]
        require(type(record["pid"]) is int and (record["pid"], record["process_start_time"]) == expected
            and record["hostname"] == "serval02" and record["uid"] == 2618919, "actual observed " + role + " identity")
    require(spawn["child"]["parent_pid"] == spawn["controller"]["pid"] and spawn["actual_popen_pid"] == 1042870,
        "actual child joins its actual controller and returned Popen identity")
    require(type(waited["returncode"]) is int and waited["returncode"] == 1 and waited["waited_pid"] == 1042870
        and waited["child"] == spawn["child"] and waited["interventions"] == [], "failed wait remains actual return1 without synthesized success")
    specification = importlib.util.spec_from_file_location("independent_exact_exit_fixture", DIRECTORY / "test_observe_exact_prior_h100_exits.py")
    fixture = importlib.util.module_from_spec(specification);specification.loader.exec_module(fixture)
    names = ["test_absent_roles_keep_real_current_observer_identity_and_utc", "test_present_same_birth_is_not_an_exit",
        "test_reused_own_birth_is_reported_without_claiming_absence", "test_unknown_role_and_runtime_arguments_refuse"]
    stream = io.StringIO();suite = unittest.TestSuite(fixture.PriorExitObserverTests(name) for name in names)
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    with (DIRECTORY / "independent_test_output.raw").open("x") as output:output.write(stream.getvalue())
    require(result.wasSuccessful() and result.testsRun == 4, "four affected private-proc fixture methods passed")
    for name, expected in freeze["files"].items():require(reference(DIRECTORY / name)[0]["sha256"] == expected["sha256"], "held file unchanged after checks: " + name)
    review = {"kind": "independent_read_only_failed_h100_exit_observer_source_review", "approved": True, "passed": True,
        "independent_of_author": True, "source_freeze": freeze_ref, "source": source_ref,
        "actual_job_id": os.environ["SLURM_JOB_ID"], "actual_step_id": os.environ.get("SLURM_STEP_ID"),
        "elapsed_seconds": time.monotonic() - started, "check_count": len(checks), "checks": checks,
        "affected_fixture_methods": 4, "target_invoked": False, "actual_process_exits_observed": False,
        "prior_unchanged_function_review_reused": True, "actual_failed_wait_remains_returncode_1": True,
        "gpu_or_native_or_signals_or_standing_lock_operations": 0, "unresolved_anomalies": []}
    with (DIRECTORY / "independent_source_review.json").open("x") as output:json.dump(review, output, indent=2, sort_keys=True);output.write("\n")
    print(json.dumps({"approved": True, "check_count": len(checks), "elapsed_seconds": review["elapsed_seconds"]}))


if __name__ == "__main__":
    main()
