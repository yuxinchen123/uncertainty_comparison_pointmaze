"""Require reviewed actual common correctness before intermediate A4500 confirmation."""
import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

NAME = "common_confirmation.py"
PREFIX = "10_jax_exploration_platform/scripts/atari_pixel_campaign/"
NUMERICAL = {"jax", "jaxlib", "numpy", "torch", "flax", "optax", "scipy", "cupy", "jaxtari", "exploration_platform"}
PROOF_NAME = "common_correctness_validation.json"
LIMIT = 1024 * 1024


def require(condition, message):
    """Reject an incomplete or changed prerequisite without a replacement path."""
    if not condition:
        raise ValueError(message)


def encoded(value):
    """Serialize strict metadata with stable key order."""
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def digest(raw):
    """Identify an exact saved byte sequence."""
    return hashlib.sha256(raw).hexdigest()


def read(path):
    """Read an ordinary exact JSON file with unique keys and finite numbers."""
    path = Path(path)
    require(path.is_absolute() and path.resolve(strict=True) == path and path.is_file(), "an exact ordinary prerequisite file is required")
    def unique(pairs):
        """Reject conflicting copies of one metadata key."""
        result = {}
        for key, value in pairs:
            require(key not in result, "duplicate common-confirmation key")
            result[key] = value
        return result
    raw = path.read_bytes()
    return json.loads(raw, object_pairs_hook=unique, parse_constant=lambda value: (_ for _ in ()).throw(ValueError("nonfinite prerequisite metadata"))), raw


def reference(path):
    """Keep a real ordinary artifact's exact path and bytes."""
    unused, raw = read(path)
    return {"path": str(path), "sha256": digest(raw)}


def checked_reference(value):
    """Refuse a changed referenced result or review."""
    require(isinstance(value, dict) and set(value) == {"path", "sha256"} and reference(value["path"]) == value, "common correctness prerequisite changed")
    return read(value["path"])[0]


def contract(plan):
    """Keep this opt-in path limited to seven alternating ten-update common comparisons."""
    require(plan.get("kind") == "selected_configuration_measurement" and plan.get("mode") == "confirmation"
        and "configured_isolation" in plan and plan.get("hardware_class") == "jaguar03_rtx_a4500"
        and plan.get("gpu_uuid", "absent") is None and not any(key in plan for key in (
            "direct_binding", "target_qualification", "final_confirmation_protocol", "final_protocol", "completed_study",
            "completed_study_manifest", "device_memory_limit_protocol", "fixed_total_scaling", "scaling_protocol")),
        "common correctness applies only to intermediate configured A4500 confirmation")
    require(all(type(plan.get(key)) is int and plan[key] == value for key, value in (
        ("rounds", 7), ("minimum_updates_per_round", 10), ("iterations", 10))), "common confirmation requires exactly seven pairs of ten complete updates")
    inputs = plan.get("common_correctness_inputs")
    require(isinstance(inputs, dict) and set(inputs) == {"result", "review"}, "actual common correctness and independent review are required")
    changes = [(scope, key, plan["reference_configuration"][scope][key], value)
        for scope in ("learner", "environment", "execution") for key, value in plan["configuration"][scope].items()
        if value != plan["reference_configuration"][scope][key]]
    require(sorted(changes) == [("environment", "cache_room_backgrounds", False, True),
        ("environment", "render_final_two_only", False, True)], "the common candidate must retain exactly its two independently checked environment changes")
    copies = plan["configuration"]["learner"]["n_copies"]
    require(type(copies) is int and copies in (1, 2) and plan["copy_seeds"] == list(range(1, copies + 1))
        and all(type(seed) is int for seed in plan["copy_seeds"]), "common confirmation changed its copy count or independent seeds")
    require(digest(Path(__file__).read_bytes()) == plan["helpers"]["files"].get(PREFIX + NAME), "the exact common-confirmation helper is not staged")
    return inputs


def validate_actual(plan, plan_sha256):
    """Rebuild the actual correctness evidence in this isolated metadata process."""
    inputs = contract(plan)
    import configured_measurement as support
    import post_comparison_validation as validation
    identity = support.core()
    result, review = checked_reference(inputs["result"]), checked_reference(inputs["review"])
    directory = Path(inputs["result"]["path"]).parent
    require(Path(inputs["result"]["path"]).name == "result.json", "common correctness requires the actual result filename")
    previous, binding = support.verify_saved_measurement(directory, result)
    require(previous.get("mode") == "correctness" and "common_correctness_inputs" not in previous
        and result.get("measurement_mode") == "correctness" and result.get("measurement_valid") is False
        and result.get("implementation_distinct") is True, "confirmation cannot replace independent correctness")
    validation.correctness(result)
    validation.configuration(plan["configuration"])
    validation.configuration(plan["reference_configuration"])
    validation.execution_identity(result)
    for key in ("source_root", "source_commit", "hardware_class", "runtime_environment", "asset_record", "copy_seeds",
                "configuration", "reference_configuration", "target_qualification_inputs"):
        require(identity.exact(previous.get(key), plan.get(key)), "the reviewed correctness differs from confirmation: " + key)
    require(identity.exact(previous["configured_isolation"]["profile"], plan["configured_isolation"]["profile"]), "common correctness changed its target trust profile")
    metadata = read(directory / "configured_isolation_metadata.json")[0]
    job = metadata["inputs"]["job_id"]
    numeric = job + "." + metadata["environment"]["SLURM_STEP_ID"]
    require(type(review.get("schema_version")) is int and review["schema_version"] == 1
        and review.get("kind") == "independent_actual_common_configuration_correctness_review"
        and review.get("approved") is True and review.get("passed") is True
        and review.get("result") == inputs["result"] and review.get("plan") == reference(directory / "measurement_plan.json")
        and review.get("job_id") == job and review.get("actual_numeric_step") == numeric
        and review.get("actual_terminal_state") == "COMPLETED" and review.get("actual_exit_code") == "0:0"
        and review.get("admission_granted") is False and review.get("worker_admitted") is False
        and all(type(review.get(key)) is int and review[key] == 0 for key in (
            "scientific_credit", "idea_credit", "case_credit", "throughput_credit", "final_training_credit")),
        "the actual common correctness needs its exact independent terminal review")
    raw = checked_reference(review["raw_audit"])
    require(raw.get("kind") == "independent_actual_common_a4500_correctness_raw_validation" and raw.get("passed") is True
        and raw.get("result") == inputs["result"] and raw.get("job_id") == job and raw.get("numeric_step") == numeric
        and raw.get("terminal_state") == "COMPLETED" and raw.get("exit_code") == "0:0"
        and raw.get("unchanged_staged_consumer_passed") is True, "the independent review lacks its actual raw validation")
    for key in (job, job + ".batch", numeric):
        require(raw["terminal_accounting_rows"][key][3:6] == ["COMPLETED", "0:0", "jaguar03"], "a recorded common correctness process did not finish normally")
    require(not NUMERICAL.intersection(sys.modules), "a numerical module entered the metadata child")
    return {"schema_version": 1, "kind": "isolated_common_correctness_validation", "passed": True,
        "plan_sha256": plan_sha256, "common_correctness_inputs": inputs,
        "correctness_plan": reference(directory / "measurement_plan.json"), "correctness_job_id": job,
        "correctness_numeric_step": numeric, "gpu_uuid_of_prior_correctness": binding["gpu_uuid"],
        "source_commit": plan["source_commit"], "helper_commit": plan["helpers"]["source_commit"],
        "parent_pid": os.getppid(), "validation_pid": os.getpid(), "finished_at_seconds": time.time(),
        "numerical_imports": [], "gpu_operations": 0, "scientific_credit": 0, "admission_granted": False}


def request(plan, raw=None, directory=None):
    """Run the full saved consumer in a fresh no-site process without changing parent imports."""
    contract(plan)
    require(not NUMERICAL.intersection(sys.modules), "common correctness must be checked before numerical initialization")
    raw = encoded(plan) if raw is None else raw
    executable = Path(directory) / "code" / NAME if directory is not None else Path(__file__).absolute()
    command = ["/usr/bin/python3", "-I", "-S", "-B", str(executable), "--plan-stdin"]
    try:
        result = subprocess.run(command, input=raw, capture_output=True, timeout=180, check=False)
    except BaseException as error:
        if directory is not None:
            publish_child_capture(directory, command, None, getattr(error, "stdout", None) or b"",
                getattr(error, "stderr", None) or b"", type(error).__name__)
        raise
    if directory is not None:
        publish_child_capture(directory, command, result.returncode, result.stdout, result.stderr)
    require(len(result.stdout) <= LIMIT and len(result.stderr) <= LIMIT, "common correctness child output exceeded its bound")
    require(result.returncode == 0 and not result.stderr, "isolated common correctness validation failed: " + result.stderr.decode(errors="replace"))
    proof = json.loads(result.stdout)
    require(proof.get("passed") is True and proof.get("plan_sha256") == digest(raw)
        and proof.get("common_correctness_inputs") == plan["common_correctness_inputs"]
        and type(proof.get("parent_pid")) is int and proof["parent_pid"] == os.getpid()
        and proof.get("numerical_imports") == [] and type(proof.get("gpu_operations")) is int and proof["gpu_operations"] == 0,
        "the isolated correctness child returned another plan or parent")
    return proof


def publish(path, raw):
    """Save one new durable validation artifact without replacing an earlier attempt."""
    require(path.parent.resolve(strict=True) == path.parent and not path.exists() and not path.is_symlink(),
        "preserve every prior common correctness validation artifact")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(raw); stream.flush(); os.fsync(stream.fileno())
    descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def publish_child_capture(directory, command, returncode, stdout, stderr, error=None):
    """Preserve actual child returns and raw bytes before any success parsing."""
    directory = Path(directory)
    publish(directory / "common_correctness_validation_stdout.raw", stdout)
    publish(directory / "common_correctness_validation_stderr.raw", stderr)
    publish(directory / "common_correctness_validation_return.json", encoded({"command": command,
        "returncode": returncode, "stdout_sha256": digest(stdout), "stderr_sha256": digest(stderr), "error": error}))


def before_compute(directory, plan):
    """Publish one actual successful child check before any numerical import or device work."""
    path = Path(directory) / PROOF_NAME
    require(not path.exists() and not path.is_symlink(), "preserve the prior common correctness validation attempt")
    raw = (Path(directory) / "measurement_plan.json").read_bytes()
    publish(Path(directory) / "common_correctness_validation_attempt.json", encoded({"plan_sha256": digest(raw),
        "common_correctness_inputs": plan["common_correctness_inputs"], "worker_pid": os.getpid(), "retry_permitted": False}))
    proof = request(plan, raw, directory)
    publish(path, encoded(proof))
    return proof


def result_binding(directory, plan):
    """Bind the saved result to the already completed pre-numerical validation."""
    contract(plan)
    path = Path(directory) / PROOF_NAME
    proof = read(path)[0]
    captured = read(Path(directory) / "common_correctness_validation_return.json")[0]
    stdout = (Path(directory) / "common_correctness_validation_stdout.raw").read_bytes()
    stderr = (Path(directory) / "common_correctness_validation_stderr.raw").read_bytes()
    command = ["/usr/bin/python3", "-I", "-S", "-B", str(Path(directory) / "code" / NAME), "--plan-stdin"]
    require(captured.get("command") == command and type(captured.get("returncode")) is int
        and captured["returncode"] == 0 and captured.get("error") is None
        and captured.get("stdout_sha256") == digest(stdout) and captured.get("stderr_sha256") == digest(stderr)
        and stderr == b"" and json.loads(stdout) == proof, "the actual metadata child result changed")
    require(proof.get("kind") == "isolated_common_correctness_validation" and proof.get("passed") is True
        and proof.get("plan_sha256") == digest((Path(directory) / "measurement_plan.json").read_bytes())
        and proof.get("common_correctness_inputs") == plan["common_correctness_inputs"], "common correctness validation no longer binds this plan")
    return {"common_correctness_inputs": plan["common_correctness_inputs"], "common_correctness_validation": reference(path)}


def verify_saved(directory, plan, result):
    """Recheck the saved prerequisite and its actual validation before accepting timing evidence."""
    expected = result_binding(directory, plan)
    require(all(result.get(key) == value for key, value in expected.items()), "the confirmation result omitted its reviewed correctness binding")
    proof = checked_reference(expected["common_correctness_validation"])
    require(type(proof.get("parent_pid")) is int and proof["parent_pid"] == result["execution_identity"]["pid"]
        and type(proof.get("finished_at_seconds")) in (int, float) and math.isfinite(proof["finished_at_seconds"])
        and proof["finished_at_seconds"] <= result["execution_identity"]["started_at_seconds"],
        "common correctness validation was not completed by this worker before numerical work")
    request(plan, (Path(directory) / "measurement_plan.json").read_bytes())


def main():
    """Expose only a bounded standard-library validation child with no launch operation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan-stdin", action="store_true", required=True)
    parser.parse_args()
    def guard(event, values):
        """Forbid numerical modules, library loads, devices and non-Git child commands."""
        if event == "import" and values[0].split(".")[0] in NUMERICAL:
            raise RuntimeError("numerical import forbidden in common correctness child")
        if event in {"socket.connect", "socket.bind", "os.system"} or event == "ctypes.dlopen" and values[0] is not None:
            raise RuntimeError("network or library operation forbidden in common correctness child")
        if event == "open" and isinstance(values[0], (str, bytes)) and os.fsdecode(values[0]).startswith("/dev/nvidia"):
            raise RuntimeError("device access forbidden in common correctness child")
        if event == "subprocess.Popen":
            command = values[1]
            require(isinstance(command, list) and len(command) > 3 and command[:2] == ["git", "-C"]
                and Path(command[2]).is_absolute() and command[3] in {
                    "show", "cat-file", "rev-parse", "status", "symbolic-ref", "ls-tree", "merge-base"},
                "only recorded local read-only Git commands are permitted")
    sys.addaudithook(guard)
    require(not NUMERICAL.intersection(sys.modules), "numerical module was already loaded")
    raw = sys.stdin.buffer.read(LIMIT + 1)
    require(len(raw) <= LIMIT, "common confirmation plan exceeds metadata bound")
    sys.path.insert(0, str(Path(__file__).absolute().parent))
    print(json.dumps(validate_actual(json.loads(raw), digest(raw)), allow_nan=False))


if __name__ == "__main__":
    main()
