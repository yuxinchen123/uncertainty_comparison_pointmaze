import ast
from pathlib import Path
p=Path(__file__).parent
s=(p/'accepted_20_09_launcher.py').read_text()
s=s.replace('KIND = "nsight_process_identity_diagnostic"', '''KIND = "configured_native_nsight_process_diagnostic"
PACKET_NAME = "2026-11-05-06-39_native-process-control_jaguar03_python-3.12_gpu-count-1_processor-count-1_host-memory-gib-2_total-seconds-300_work-seconds-240_cleanup-seconds-60_environment-transitions-0_training-updates-0"
PYTHON = "/p/rlprojects/RND/.venvs/atari_pixels_jax/bin/python"
PREFIX = "10_jax_exploration_platform/scripts/atari_pixel_campaign/"
HELPERS = ("diagnose_configured_native_process.py", "nsight_process_identity.py", "configured_native_process.py",
    "linux_pidfd_process_control.py", "native_process_evidence.py", "direct_tool_descendants.py", "direct_profile_process.py")
BUDGET = {"schema_version": 1, "work_seconds": 240, "cleanup_seconds": 60, "total_seconds": 300}
ZERO = {"scientific_credit": 0, "profile_credit": 0, "case_credit": 0, "idea_credit": 0,
    "environment_transitions": 0, "training_updates": 0, "admission_granted": False,
    "worker_admitted": False, "physical_gres_mapping_verified": False,
    "kernel_device_enforcement_verified": False, "standing_gpu_descriptor_held": False,
    "whole_tool_library_or_device_activity_quantified": False}
RESOURCES = {"partition": "gpu", "account": "sl5nw_base", "qos": "csresnolim", "reservation": "sl5nw_168",
    "node": "jaguar03", "nodes": 1, "tasks": 1, "requested_cpus": 1, "host_memory_gib": 2,
    "gres": "gpu:nvidia_rtx_a4500:1", "wall_seconds": 300, "requeue": False}''')
s=s.replace('import stat\n','import stat\nimport shlex\n')
def replacefn(name,body):
 global s
 n=next(n for n in ast.parse(s).body if isinstance(n,ast.FunctionDef) and n.name==name)
 lines=s.splitlines(keepends=True);s=''.join(lines[:n.lineno-1])+body.rstrip()+'\n'+''.join(lines[n.end_lineno:])
replacefn('checked_packet','''def same(left, right):
    """Require exact JSON types, including integer rather than Boolean counts."""
    if type(left) is not type(right): return False
    if type(right) is dict: return left.keys() == right.keys() and all(same(left[key], right[key]) for key in right)
    if type(right) is list: return len(left) == len(right) and all(same(a, b) for a, b in zip(left, right))
    return left == right


def expected_initial_paths():
    """Derive every canonical seven-helper/standard-record input without a future digest."""
    return {UNIT / "code" / name for name in HELPERS} | {UNIT / name for name in (
        "measurement_plan.json", "measurement_prepared.json", "manifest.yaml", "config_resolved.yaml", "command.txt", "experiment_background.md")} | {
        PACKET / name for name in ("job.sbatch", "manifest.yaml", "config_resolved.yaml", "command.txt",
            "experiment_background.md", "progress.json", "source_checksums.sha256", "held_submission_command.json")}


def prepared_records():
    """Bind actual canonical prepared bytes only; no target import, tool or scheduler is involved."""
    require(PACKET.name == PACKET_NAME and PACKET.parent.name == "remote_setup"
        and RUN.resolve(strict=True) == RUN and UNIT == RUN / "data" / PACKET_NAME, "the exact new diagnostic packet is required")
    original_path = PACKET / "immutable_artifacts.json"
    original = document(original_path)
    require(type(original) is dict and set(original) == {"files"} and type(original["files"]) is dict
        and set(original["files"]) == {str(path) for path in expected_initial_paths()},
        "the complete canonical prepared inventory is required")
    for name, expected in original["files"].items():
        require(type(expected) is dict and set(expected) == {"sha256", "bytes", "mode"}
            and type(expected["bytes"]) is int and expected["bytes"] >= 0,
            "typed immutable artifact size/mode/hash records are required")
        path = Path(name); raw = ordinary_bytes(path)
        require(hashlib.sha256(raw).hexdigest() == expected["sha256"] and len(raw) == expected["bytes"]
            and oct(path.stat().st_mode & 0o777) == expected["mode"], "a frozen diagnostic artifact changed")
    plan_path = UNIT / "measurement_plan.json";plan = document(plan_path)
    require(type(plan.get("schema_version")) is int and plan["schema_version"] == 1
        and plan.get("kind") == KIND and plan.get("packet") == str(PACKET) and plan.get("campaign_run_dir") == str(RUN)
        and plan.get("unit_id") == PACKET_NAME and plan.get("gpu_uuid") is None and plan.get("job_id") is None
        and plan.get("python") == PYTHON and plan.get("hardware_class") == "jaguar03_rtx_a4500"
        and "direct_binding" not in plan and "configured_isolation" not in plan
        and type(plan.get("maximum_active_seconds")) is int and plan["maximum_active_seconds"] == 240
        and same(plan.get("diagnostic_time_budget"), BUDGET) and same(plan.get("diagnostic_boundary"), ZERO),
        "the actual canonical selected-Python zero-credit diagnostic plan is required")
    helpers = plan.get("helpers")
    require(type(helpers) is dict and set(helpers) == {"source_commit", "files"}
        and type(helpers["source_commit"]) is str and re.fullmatch("[0-9a-f]{40}", helpers["source_commit"])
        and type(helpers["files"]) is dict and set(helpers["files"]) == {PREFIX + name for name in HELPERS},
        "the actual full source commit and exact seven-helper closure are required")
    for name in HELPERS:
        expected = helpers["files"][PREFIX + name]
        require(type(expected) is str and re.fullmatch("[0-9a-f]{64}", expected)
            and reference(UNIT / "code" / name)["sha256"] == expected, "a canonical staged helper differs")
        raw = subprocess.check_output(["git", "-C", str(INTEGRATION), "cat-file", "blob", helpers["source_commit"] + ":" + PREFIX + name])
        require(hashlib.sha256(raw).hexdigest() == expected, "a staged helper is not the declared source commit byte sequence")
    native = {"schema_version": 1, "backend": "linux-x86-64-libc-syscall-pidfd-version-1",
        "mode": "observe_configured_target_for_zero_credit_diagnostic", "expected_target_contract": None,
        "qualification": False, "python": PYTHON,
        "helper_hashes": {name: helpers["files"][PREFIX + name] for name in ("linux_pidfd_process_control.py", "native_process_evidence.py")}}
    configured = {"schema_version": 1, "protocol": "configured-allocation-native-process-control-version-1",
        "hardware_class": "jaguar03_rtx_a4500", "standing_gpu_descriptor_held": False,
        "physical_gres_mapping_verified": False, "kernel_device_enforcement_verified": False,
        "configured_isolation_trust": True,
        "helper_hashes": {name: helpers["files"][PREFIX + name] for name in HELPERS if name != "diagnose_configured_native_process.py"}}
    require(same(plan.get("native_process_control"), native) and same(plan.get("configured_native_process_control"), configured),
        "the exact explicit diagnostic-only native declarations are required")
    require(document(UNIT / "measurement_prepared.json") == {"plan_sha256": reference(plan_path)["sha256"]}
        and same(document(UNIT / "manifest.yaml"), plan)
        and same(document(UNIT / "config_resolved.yaml"), {**ZERO, "budget": BUDGET}), "the canonical preparation sidecars differ")
    tool = plan.get("profile_tool")
    require(type(tool) is dict and set(tool) == {"path", "version", "sha256"} and type(tool["path"]) is str
        and Path(tool["path"]).is_absolute() and type(tool["version"]) is str and tool["version"]
        and type(tool["sha256"]) is str and re.fullmatch("[0-9a-f]{64}", tool["sha256"]), "the actual tool metadata is incomplete")
    expected_command = [PYTHON, "-I", "-S", "-B", str(UNIT / "code/diagnose_configured_native_process.py"), "--prepared-unit", str(UNIT)]
    require(ordinary_bytes(UNIT / "command.txt") == (shlex.join(expected_command) + "\\n").encode(), "the canonical producer command changed")
    for name in ("manifest.yaml", "config_resolved.yaml", "command.txt", "experiment_background.md"):
        require(ordinary_bytes(PACKET / name) == ordinary_bytes(UNIT / name), "packet and canonical standard records differ")
    command = document(PACKET / "held_submission_command.json")
    require(command == ["sbatch", "--hold", "--parsable", "--no-requeue", "--partition=gpu", "--account=sl5nw_base",
        "--qos=csresnolim", "--reservation=sl5nw_168", "--nodelist=jaguar03", "--nodes=1", "--ntasks=1", "--cpus-per-task=1", "--mem=2G",
        "--gres=gpu:nvidia_rtx_a4500:1", "--time=00:05:00", str(PACKET / "job.sbatch")], "the exact held submission command changed")
    expected_checksums = {str(UNIT / "code" / name): helpers["files"][PREFIX + name] for name in HELPERS}
    expected_checksums.update({str(path): reference(path)["sha256"] for path in (plan_path, UNIT / "measurement_prepared.json")})
    lines = ordinary_bytes(PACKET / "source_checksums.sha256").decode().splitlines()
    require(len(lines) == len(expected_checksums) and all(re.fullmatch(r"[0-9a-f]{64}  /.+", line) for line in lines)
        and len({line[66:] for line in lines}) == len(lines)
        and {line[66:]: line[:64] for line in lines} == expected_checksums, "source checksums omit or substitute canonical bytes")
    binding = {"kind": "actual_configured_native_diagnostic_prepared_binding", "schema_version": 1,
        "packet": str(PACKET), "data_unit": str(UNIT), "diagnostic_kind": KIND,
        "plan": reference(plan_path), "source_commit": helpers["source_commit"], "resources": RESOURCES,
        "immutable_artifacts": reference(original_path),
        "canonical_inputs": {str(path): reference(path) for path in sorted(expected_initial_paths())},
        "submission_command": command, "actual_job_id": None, "actual_release_performed": False,
        "scientific_credit": 0, "profile_admission_granted": False}
    return plan, command, binding


def prepare_binding_once():
    """Publish only a real prepared metadata binding; it performs no scheduler operation."""
    actor()
    for path in (PACKET / "prepared_launch_binding.json", PACKET / "submission_intent.json", PACKET / "submission_receipt.json"):
        new_path(path)
    unused_plan, unused_command, binding = prepared_records()
    return publish(PACKET / "prepared_launch_binding.json", binding)


def checked_packet():
    """Require the exact later immutable binding; placeholders never authorize a launch."""
    plan, command, binding = prepared_records()
    require(same(document(PACKET / "prepared_launch_binding.json"), binding), "the actual canonical prepared binding changed")
    return plan, command''')
s=s.replace('review.get("passed") is True and review.get("source_freeze") == reference(freeze_path),', 'review.get("passed") is True and review.get("kind") == "independent_configured_native_diagnostic_prepared_launcher_review"\n            and review.get("source_freeze") == reference(freeze_path)\n            and review.get("prepared_binding") == reference(PACKET / "prepared_launch_binding.json"),')
s=s.replace('str(PACKET / "launch_diagnostic_once.py"), str(PACKET / "test_launch_diagnostic_once.py")}', 'str(PACKET / "launch_diagnostic_once.py"), str(PACKET / "test_launch_diagnostic_once.py"),\n                               str(PACKET / "prepared_launch_binding.json")}')
s=s.replace('"6783003"', '"6821437"')
s=s.replace('owned_nsight_process_diagnostic_', 'owned_configured_native_process_diagnostic_')
s=s.replace('one_nsight_diagnostic_held_submission_attempt', 'one_configured_native_diagnostic_held_submission_attempt')
s=s.replace('actual_diagnostic_three_ledgers_fsynced', 'actual_configured_native_diagnostic_three_ledgers_fsynced')
s=s.replace('actual_owned_nsight_diagnostic_fixed_identifier_release_binding', 'actual_owned_configured_native_diagnostic_fixed_identifier_release_binding')
s=s.replace('one_actual_nsight_diagnostic_release_attempt', 'one_actual_configured_native_diagnostic_release_attempt')
s=s.replace('nsight_diagnostic_launcher_default_check', 'configured_native_diagnostic_launcher_default_check')
s=s.replace('def query_held(job, stem):', 'def query_held(job, stem, plan):')
s=s.replace('    job = identifier(job)\n    result = scheduler', '    job = identifier(job)\n    require(owned_submission(plan)["job_id"] == job, "the query identifier is not this exact owned submission")\n    result = scheduler')
s=s.replace('query_held(actual["job_id"], "release_preparation_query")','query_held(actual["job_id"], "release_preparation_query", plan)')
s=s.replace('query_held(job, "release_final_query")','query_held(job, "release_final_query", plan)')
s=s.replace('    operations.add_argument("--submit-held-once", action="store_true")','    operations.add_argument("--prepare-binding-once", action="store_true")\n    operations.add_argument("--submit-held-once", action="store_true")')
s=s.replace('parser.add_argument("--publication-commit", required=True)', 'parser.add_argument("--publication-commit")')
s=s.replace('    if arguments.submit_held_once:', '    if arguments.prepare_binding_once:\n        require(arguments.publication_commit is None, "metadata binding makes no publication claim")\n        value = prepare_binding_once()\n    elif arguments.submit_held_once:')
# Re-read file identity after bounded descriptor read; never accept mutation during its read.
s=s.replace('require(len(raw) == status.st_size, "the launch artifact changed during its read")', 'after = os.fstat(descriptor)\n        require(len(raw) == status.st_size and (status.st_dev, status.st_ino, status.st_size, status.st_mtime_ns, status.st_ctime_ns)\n                == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns), "the launch artifact changed during its read")')
(p/'launch_diagnostic_once.py').write_text(s)
