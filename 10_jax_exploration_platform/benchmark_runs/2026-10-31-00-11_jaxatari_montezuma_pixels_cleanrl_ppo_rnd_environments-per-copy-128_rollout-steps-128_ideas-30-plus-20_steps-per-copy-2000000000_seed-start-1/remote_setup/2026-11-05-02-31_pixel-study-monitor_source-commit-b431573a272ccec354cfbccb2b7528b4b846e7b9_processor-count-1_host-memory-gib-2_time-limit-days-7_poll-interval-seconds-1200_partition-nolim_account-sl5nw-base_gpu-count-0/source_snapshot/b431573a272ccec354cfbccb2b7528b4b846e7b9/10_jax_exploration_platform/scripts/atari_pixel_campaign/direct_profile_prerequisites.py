"""Keep direct H100 profile prerequisites separate from configured GPU profiles.

Preparation checks reviewed metadata only. Historical correctness and timing
consumers run in a fresh owned-compute child; current history also requires that
owned context. No actual profile is admitted
until the separate real process, descriptor and current-context tests are reviewed.
"""
import argparse
import ast
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import re
import socket
import stat
import subprocess
import sys
import time

NAME = "direct_profile_prerequisites.py"
PREFIX = "10_jax_exploration_platform/scripts/atari_pixel_campaign/"
PROTOCOL = "direct-h100-reviewed-common-profile-version-1"
PROCESS_PROTOCOL = "direct-h100-owned-nsight-and-retained-descriptor-version-1"
CARD_ONE = "GPU-fed69954-5a71-dbb2-ee91-544e0eb6b8a3"
NUMERICAL = {"jax", "jaxlib", "numpy", "torch", "flax", "optax", "scipy", "cupy", "jaxtari", "exploration_platform"}
LIMIT = 1024 * 1024
PROOF_NAME = "direct_profile_baseline_validation.json"


def require(condition, message):
    """Refuse incomplete evidence without substituting another profile or consumer."""
    if not condition:
        raise ValueError(message)


def encoded(value):
    """Serialize finite metadata with stable keys."""
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def digest(raw):
    """Identify exact bytes without interpreting them as scientific evidence."""
    return hashlib.sha256(raw).hexdigest()


def reference_shape(value):
    """Check a reference's syntax without opening its scientific result."""
    require(type(value) is dict and set(value) == {"path", "sha256"}
        and type(value["path"]) is str and Path(value["path"]).is_absolute()
        and str(Path(value["path"])) == value["path"] and ".." not in Path(value["path"]).parts
        and type(value["sha256"]) is str and re.fullmatch(r"[0-9a-f]{64}", value["sha256"]),
        "an exact absolute path and SHA-256 reference are required")
    return value


def read(path, maximum=LIMIT):
    """Read bounded unique-key metadata through an ordinary pinned descriptor."""
    path = Path(path)
    require(path.is_absolute() and path.resolve(strict=True) == path and path.is_file() and not path.is_symlink(),
        "an ordinary exact metadata file is required")
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        before = os.fstat(descriptor)
        require(stat.S_ISREG(before.st_mode) and 0 < before.st_size <= maximum, "metadata exceeds its declared regular-file bound")
        raw = os.read(descriptor, maximum + 1)
        after, current = os.fstat(descriptor), path.stat()
        fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns", "st_mode", "st_uid")
        require(len(raw) == before.st_size and all(getattr(before, key) == getattr(after, key) == getattr(current, key)
            for key in fields), "metadata changed while its descriptor was being read")
    finally:
        os.close(descriptor)
    def unique(pairs):
        """Reject duplicate values for any metadata key."""
        value = {}
        for key, item in pairs:
            require(key not in value, "duplicate direct-profile metadata key")
            value[key] = item
        return value
    return json.loads(raw, object_pairs_hook=unique,
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError("nonfinite direct-profile metadata"))), raw


def reference(path):
    """Bind one bounded metadata artifact to its actual bytes."""
    unused, raw = read(path)
    return {"path": str(path), "sha256": digest(raw)}


def checked_reference(value):
    """Read only an explicitly requested exact metadata reference."""
    reference_shape(value)
    document, raw = read(value["path"])
    require(digest(raw) == value["sha256"], "direct-profile metadata changed")
    return document


def checked_actual_reference(plan, value):
    """Open an explicitly referenced scientific record only in actual owned compute."""
    compute_context(plan)
    reference_shape(value)
    document, raw = read(value["path"], 64 * 1024 * 1024)
    require(digest(raw) == value["sha256"], "actual direct scientific evidence changed")
    return document


def confirmation_unit(plan, reference):
    """Check the exact ordinary campaign result location without opening that result."""
    reference_shape(reference)
    unit = Path(reference["path"]).parent
    require(Path(reference["path"]).name == "result.json" and unit.parent == Path(plan["campaign_run_dir"]) / "data"
        and unit.resolve(strict=True) == unit and unit.name not in {"", ".", ".."},
        "the accepted confirmation must name its exact campaign data unit")
    return unit


def reserved_metadata(value, unit, name):
    """Check a known metadata filename and unit before any caller opens the supplied path."""
    reference_shape(value)
    require(value["path"] == str(Path(unit) / name), "the reviewed metadata path differs from its reserved unit filename")
    return value


def current(plan):
    """Recognize the explicit new direct protocol without accepting old unit registration."""
    return plan.get("direct_profile_protocol") == PROTOCOL


def require_admission():
    """Keep actual preparation, execution and acceptance closed pending real direct target review."""
    raise ValueError("direct H100 profiles remain disabled pending actual Nsight process/descriptor survival and current-CUDA-context qualification with independent review")


def helper_roster():
    """Derive the exact declared helper union from source literals without importing a consumer."""
    names = {"profile_selected.py", "direct_measurement.py", "configured_measurement.py",
        NAME, "direct_backend_image.py", "nsight_process_identity.py"}
    for filename in ("measure_selected.py", "serval02_card1_binding.py"):
        tree = ast.parse(Path(__file__).with_name(filename).read_text())
        declarations = [node.value for node in tree.body if isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == "HELPERS" for target in node.targets)]
        require(len(declarations) == 1, "one literal helper roster is required")
        values = ast.literal_eval(declarations[0])
        require(type(values) is tuple and all(type(value) is str and Path(value).name == value for value in values),
            "the source helper roster must contain literal adjacent filenames")
        names.update(values)
    return {PREFIX + name for name in names}


def contract(plan):
    """Limit metadata and backend support to the separately declared card-one capture."""
    require(current(plan) and plan.get("kind") == "selected_configuration_profile"
        and plan.get("hardware_class") == "h100" and plan.get("gpu_uuid") == CARD_ONE
        and plan.get("capture_process_protocol") == PROCESS_PROTOCOL
        and "direct_binding" in plan and not any(key in plan for key in (
            "configured_isolation", "configured_group", "target_qualification", "target_qualification_inputs",
            "direct_qualification", "final_confirmation_protocol", "final_protocol", "completed_study",
            "device_memory_limit_protocol", "fixed_total_scaling", "scaling_protocol", "common_correctness_inputs")),
        "direct profile prerequisites cannot admit configured, historical, capacity, scaling or final work")
    copies = plan["configuration"]["learner"]["n_copies"]
    require(type(copies) is int and copies in (1, 2) and plan.get("copy_seeds") == list(range(1, copies + 1))
        and all(type(seed) is int for seed in plan["copy_seeds"])
        and type(plan.get("profile_updates")) is int and plan["profile_updates"] > 0,
        "the common baseline profile must retain the actual one- or two-copy seed identities")
    expected = helper_roster()
    require(len(expected) == 29 and expected == set(plan["helpers"]["files"])
        and all(plan["helpers"]["files"][relative] == digest(Path(__file__).with_name(Path(relative).name).read_bytes())
            for relative in expected),
        "the new profile requires its own exact helper closure, not the old confirmation helper")
    backend = plan.get("direct_backend_image", {})
    require(type(backend.get("schema_version")) is int and backend["schema_version"] == 2
        and backend.get("capability") == "prepared_owning_package_image_and_same_process_mapping_version_2",
        "a direct profile requires the new prepared backend package record")
    reference_shape(backend["expected"])
    inputs = plan.get("direct_profile_prerequisites")
    require(type(inputs) is dict and set(inputs) == {"confirmation_plan", "confirmation", "confirmation_review",
        "correctness", "correctness_review", "qualification", "qualification_review"},
        "the actual reviewed common baseline and original qualification references are required")
    for value in inputs.values():
        reference_shape(value)
    return inputs


def review_metadata(plan, previous, review, inputs):
    """Check an existing independent timing review without reading its result or raw audit."""
    require(type(review.get("schema_version")) is int and review["schema_version"] == 1
        and review.get("kind") == "independent_actual_direct_common_configuration_confirmation_review"
        and review.get("approved") is True and review.get("passed") is True
        and review.get("independent_of_producer") is True and review.get("unresolved_anomalies") == []
        and type(review.get("reviewer_identity")) is str and bool(review["reviewer_identity"].strip())
        and review.get("decision") == "accepted" and review.get("raw_window_recomputation_passed") is True
        and type(review.get("paired_rounds")) is int and review["paired_rounds"] == 7
        and type(review.get("updates_per_arm_per_round")) is int and review["updates_per_arm_per_round"] == 10
        and type(review.get("reviewed_at_seconds")) in (int, float) and math.isfinite(review["reviewed_at_seconds"])
        and review.get("admission_granted") is False and review.get("worker_admitted") is False,
        "an actual complete independent accepted direct timing review is required")
    for key, expected in {"result": inputs["confirmation"], "plan": inputs["confirmation_plan"],
        "source_commit": plan["source_commit"], "helper_commit": previous["helpers"]["source_commit"],
        "configuration": plan["configuration"], "reference_configuration": previous["reference_configuration"],
        "copies": plan["configuration"]["learner"]["n_copies"], "copy_seeds": plan["copy_seeds"],
        "gpu_uuid": CARD_ONE}.items():
        require(encoded(review.get(key)) == encoded(expected), "independent direct timing review changed " + key)
    reference_shape(review["raw_audit"])


def validate_metadata(plan):
    """Recheck frozen plan/review inputs only, never actual correctness or timing results."""
    inputs = contract(plan)
    unit = confirmation_unit(plan, inputs["confirmation"])
    reserved_metadata(inputs["confirmation_plan"], unit, "measurement_plan.json")
    previous = checked_reference(inputs["confirmation_plan"])
    review = checked_reference(inputs["confirmation_review"])
    require(previous.get("kind") == "selected_configuration_measurement" and previous.get("mode") == "confirmation"
        and previous.get("hardware_class") == "h100" and previous.get("gpu_uuid") == CARD_ONE
        and "direct_binding" in previous and "configured_isolation" not in previous
        and type(previous.get("rounds")) is int and previous["rounds"] == 7
        and type(previous.get("iterations")) is int and previous["iterations"] == 10
        and type(previous.get("minimum_updates_per_round")) is int and previous["minimum_updates_per_round"] == 10
        and len(previous["helpers"]["files"]) == 27 and PREFIX + "direct_common_confirmation.py" in previous["helpers"]["files"],
        "the profile needs the actual separate direct common seven-by-ten confirmation")
    require(unit == Path(previous["campaign_run_dir"]) / "data" / previous["unit_id"]
        and inputs["confirmation_plan"]["path"] == str(unit / "measurement_plan.json")
        and inputs["confirmation"]["path"] == str(unit / "result.json")
        and previous["common_correctness_inputs"] == {"result": inputs["correctness"], "review": inputs["correctness_review"]}
        and previous["direct_qualification_inputs"] == {"result": inputs["qualification"], "review": inputs["qualification_review"]}
        and plan["direct_qualification_inputs"] == previous["direct_qualification_inputs"],
        "profile prerequisites changed the actual correctness, confirmation or original qualification unit")
    for key in ("source_root", "source_commit", "core", "paired_source", "python", "campaign_run_dir",
        "configuration", "runtime_environment", "asset_record", "copy_seeds"):
        require(encoded(previous[key]) == encoded(plan[key]), "direct profile changed accepted input: " + key)
    require(previous["direct_binding"]["profile"] == plan["direct_binding"]["profile"],
        "the direct profile differs from its actual direct common target")
    baseline = plan["accepted_baseline_specification"]
    require(baseline["correctness_record"] == inputs["correctness"]["path"]
        and baseline["confirmation_record"] == inputs["confirmation"]["path"]
        and plan["inputs"]["accepted_confirmation"] == inputs["confirmation"],
        "accepted baseline paths differ from the independently reviewed actual common units")
    # Package description excludes only preparation time and observer identity, not installed bytes or ownership.
    accepted_backend = checked_reference(reserved_metadata(previous["direct_backend_image"]["expected"],
        unit, "direct_backend_image_expected.json"))
    profile_unit = Path(plan["campaign_run_dir"]) / "data" / plan["unit_id"]
    profile_backend = checked_reference(reserved_metadata(plan["direct_backend_image"]["expected"],
        profile_unit, "direct_backend_image_expected.json"))
    require(all(encoded(accepted_backend[key]) == encoded(profile_backend[key])
        for key in ("selected_python", "site_packages", "image", "distribution")),
        "the profile changed the accepted prepared backend package image")
    review_metadata(plan, previous, review, inputs)
    return previous


def configure(plan, baseline, review_path, qualification_path, qualification_review_path):
    """Bind actual review metadata without opening any scientific result on a login host."""
    review = checked_reference(reference(review_path))
    reference_shape(review.get("result")); reference_shape(review.get("plan"))
    require(review["result"]["path"] == str(Path(baseline["confirmation_record"])),
        "the supplied real timing review belongs to another baseline")
    unit = confirmation_unit(plan, review["result"])
    reserved_metadata(review["plan"], unit, "measurement_plan.json")
    previous = checked_reference(review["plan"])
    plan["direct_profile_protocol"] = PROTOCOL
    plan["capture_process_protocol"] = PROCESS_PROTOCOL
    plan["direct_qualification_inputs"] = previous["direct_qualification_inputs"]
    require(plan["direct_qualification_inputs"]["result"]["path"] == str(qualification_path)
        and plan["direct_qualification_inputs"]["review"]["path"] == str(qualification_review_path),
        "explicit original qualification inputs differ from the actual accepted confirmation")
    plan["inputs"]["accepted_confirmation"] = review["result"]
    plan["inputs"]["direct_confirmation_review"] = reference(review_path)
    plan["direct_profile_prerequisites"] = {"confirmation_plan": review["plan"], "confirmation": review["result"],
        "confirmation_review": reference(review_path), "correctness": previous["common_correctness_inputs"]["result"],
        "correctness_review": previous["common_correctness_inputs"]["review"],
        "qualification": previous["direct_qualification_inputs"]["result"],
        "qualification_review": previous["direct_qualification_inputs"]["review"]}
    validate_metadata(plan)
    return plan


def configure_binding(plan, profile, history_inputs, backend_image_record):
    """Prepare profile binding metadata without invoking any scientific history validator."""
    support = importlib.import_module("direct_measurement")
    identity = support.core()
    identity.require_profile(profile)
    unit = Path(plan["campaign_run_dir"]) / "data" / plan["unit_id"]
    require(Path(history_inputs) == unit / "direct_history/inputs.json",
        "direct profile history must use its own reserved immutable input file")
    backend_path = Path(backend_image_record)
    require(backend_path.name == "direct_backend_image_expected.json"
        and backend_path.parent.parent == Path(plan["campaign_run_dir"]) / "data"
        and backend_path.resolve(strict=True) == backend_path,
        "direct profile backend input must be a reserved per-unit prepared metadata sidecar")
    plan["direct_profile_protocol"] = PROTOCOL
    plan["capture_process_protocol"] = PROCESS_PROTOCOL
    plan["direct_binding"] = {"schema_version": 1, "profile": profile, "capability": identity.CAPABILITY,
        "consumer": "component_profile", "physical_identity": identity.pending_identity(),
        "frozen_inputs": reference(history_inputs), "current_launch_inputs_path": str(unit / "direct_current_launch.json")}
    identity.validate_plan(plan, "component_profile")
    support.adjacent("direct_backend_image").configure_plan(plan, backend_path)
    return plan


def compute_context(plan):
    """Require this campaign's actual owned compute context before opening raw study evidence."""
    uid_rows = [line for line in Path("/proc/self/status").read_text().splitlines() if line.startswith("Uid:")]
    require(os.getuid() == os.geteuid() == 2618919 and os.getresuid() == (2618919, 2618919, 2618919)
        and len(uid_rows) == 1 and list(map(int, uid_rows[0].split()[1:])) == [2618919] * 4,
        "actual direct baseline validation requires the exact owner")
    hostname = socket.gethostname().split(".")[0]
    job, step = os.environ.get("SLURM_JOB_ID"), os.environ.get("SLURM_STEP_ID")
    if hostname == "serval02":
        require(job is None and step is None, "direct raw validation cannot use a scheduler fallback")
    else:
        require(type(job) is str and job.isdecimal() and int(job) > 0 and type(step) is str and step.isdecimal(),
            "actual baseline validation requires owned compute, never a login node")
        ledger = Path(plan["campaign_run_dir"]) / "slurm/submitted_jobids_uva.txt"
        require(ledger.resolve(strict=True) == ledger and ledger.read_text().splitlines().count(job) == 1,
            "baseline validator is outside the exact campaign-owned UVA ledger")
        expected = "/system.slice/slurmstepd.scope/job_" + job + "/step_" + step + "/user/task_0"
        require(Path("/proc/self/cgroup").read_text().splitlines() == ["0::" + expected],
            "baseline validator is outside its exact own numeric compute step")
    return {"hostname": hostname, "job_id": job, "step_id": step, "uid": 2618919}


def publish(path, raw):
    """Fsync new validation evidence and preserve every attempted file."""
    path = Path(path)
    require(path.parent.resolve(strict=True) == path.parent, "validation parent must be exact")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(raw); stream.flush(); os.fsync(stream.fileno())
    descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def before_compute(directory, plan):
    """Start isolated raw checks only after the caller's actual owned-compute gate passes."""
    context = compute_context(plan)
    validate_metadata(plan)
    validate_current_history(plan)
    require(not NUMERICAL.intersection(sys.modules), "baseline validation must precede numerical imports")
    raw = encoded(plan)
    command = ["/usr/bin/python3", "-I", "-S", "-B", str(Path(directory) / "code" / NAME), "--plan-stdin"]
    try:
        completed = subprocess.run(command, input=raw, capture_output=True, timeout=600, check=False)
    except BaseException as error:
        capture_return(directory, command, None, getattr(error, "stdout", None) or b"",
            getattr(error, "stderr", None) or b"", type(error).__name__)
        raise
    capture_return(directory, command, completed.returncode, completed.stdout, completed.stderr)
    require(len(completed.stdout) <= LIMIT and len(completed.stderr) <= LIMIT
        and completed.returncode == 0 and completed.stderr == b"", "actual isolated direct baseline validation failed")
    proof = json.loads(completed.stdout)
    require(proof.get("kind") == "actual_direct_profile_baseline_validation" and proof.get("passed") is True
        and proof.get("plan_sha256") == digest(raw) and proof.get("parent_pid") == os.getpid()
        and proof.get("prerequisites") == plan["direct_profile_prerequisites"]
        and proof.get("metadata_compute_context") == context and proof.get("numerical_imports") == []
        and type(proof.get("gpu_operations")) is int and proof["gpu_operations"] == 0,
        "the isolated direct baseline check returned another execution or plan")
    publish(Path(directory) / PROOF_NAME, encoded(proof))
    return proof


def validate_current_history(plan):
    """Run the unchanged current history validator only after actual owned-compute admission."""
    compute_context(plan)
    module = importlib.import_module("direct_measurement")
    unit = Path(plan["campaign_run_dir"]) / "data" / plan["unit_id"]
    require(Path(module.__file__).resolve(strict=True) == unit / "code/direct_measurement.py",
        "current history validation requires the actual staged direct helper")
    return module.core().validate_history(plan)


def capture_return(directory, command, returncode, stdout, stderr, error=None):
    """Preserve actual child output and status before interpreting success."""
    directory = Path(directory)
    publish(directory / "direct_profile_baseline_stdout.raw", stdout)
    publish(directory / "direct_profile_baseline_stderr.raw", stderr)
    publish(directory / "direct_profile_baseline_return.json", encoded({"command": command, "returncode": returncode,
        "stdout_sha256": digest(stdout), "stderr_sha256": digest(stderr), "error": error}))


def load_previous(previous, directory):
    """Load only the accepted unit's unchanged staged consumers in this fresh child."""
    code = Path(directory) / "code"
    require(code.resolve(strict=True) == code and len(previous["helpers"]["files"]) == 27,
        "the historical accepted confirmation must retain its exact 27-helper closure")
    for relative, expected in previous["helpers"]["files"].items():
        require(relative.startswith(PREFIX) and Path(relative).name == relative[len(PREFIX):],
            "historical helper path is outside its exact source directory")
        path = code / Path(relative).name
        require(path.resolve(strict=True) == path and path.is_file() and digest(path.read_bytes()) == expected,
            "the actual historical consumer closure changed")
    sys.path.insert(0, str(code))
    modules = {}
    for name in ("direct_measurement", "direct_common_confirmation", "post_comparison_validation"):
        require(name not in sys.modules, "a current helper would replace the exact historical consumer")
        module = importlib.import_module(name)
        require(Path(module.__file__).resolve(strict=True) == code / (name + ".py"), "historical consumer imported another source")
        modules[name] = module
    return modules


def validate_actual(plan, raw):
    """Recompute real saved correctness and paired timing under their exact original consumers."""
    context = compute_context(plan)
    previous = validate_metadata(plan)
    inputs = plan["direct_profile_prerequisites"]
    directory = Path(inputs["confirmation"]["path"]).parent
    modules = load_previous(previous, directory)
    common, validation = modules["direct_common_confirmation"], modules["post_comparison_validation"]
    bootstrap = common.stdlib_ctypes_bootstrap()
    roots = common.declared_git_roots(previous)
    allowed_child = ["/usr/bin/python3", "-I", "-S", "-B", str(directory / "code/direct_common_confirmation.py"), "--plan-stdin"]
    def guard(event, values):
        """Allow only exact historical prerequisite children and declared read-only source checks."""
        if event == "subprocess.Popen" and values[1] == allowed_child:
            return
        common.metadata_event_guard(event, values, roots)
    sys.addaudithook(guard)
    require(not NUMERICAL.intersection(sys.modules), "the direct baseline child initialized numerical code")
    # Actual result and review-audit reads begin only here, after both owned-compute checks.
    checked_actual_reference(plan, inputs["confirmation"]); checked_actual_reference(plan, inputs["correctness"])
    review = checked_reference(inputs["confirmation_review"])
    audit = checked_reference(review["raw_audit"])
    require(audit.get("passed") is True and audit.get("unchanged_staged_consumer_passed") is True
        and audit.get("result") == inputs["confirmation"] and audit.get("plan") == inputs["confirmation_plan"]
        and audit.get("raw_window_recomputation_passed") is True,
        "the genuine independent timing review lacks its complete actual raw audit")
    result = checked_actual_reference(plan, inputs["confirmation"])
    common.validate_pair_protocol(previous, result)
    evidence = validation.Evidence()
    registry_path = Path(plan["initial_registry_record"])
    unit = Path(plan["campaign_run_dir"]) / "data" / plan["unit_id"]
    require(registry_path == unit / "inputs/study_registry.json"
        and digest(registry_path.read_bytes()) == plan["inputs"]["study_registry"]["sha256"],
        "the staged initial registry changed before direct profile validation")
    registry = read(registry_path, 64 * 1024 * 1024)[0]
    initial = validate_initial_cases(registry, validation, evidence)
    baseline = validation.validate_combination(plan["accepted_baseline_specification"], evidence)
    require(baseline["configuration"] == plan["configuration"] and baseline["paired_rounds"] == 7
        and result["copy_seeds"] == plan["copy_seeds"] and result["execution_identity"]["finished_at_seconds"]
        <= review["reviewed_at_seconds"], "the actual accepted timing result changed configuration, seeds or review order")
    require(not NUMERICAL.intersection(sys.modules), "a numerical package entered the saved direct consumer")
    return {"schema_version": 1, "kind": "actual_direct_profile_baseline_validation", "passed": True,
        "plan_sha256": digest(raw), "prerequisites": inputs, "accepted_combination": baseline,
        "parent_pid": os.getppid(), "validation_pid": os.getpid(), "finished_at_seconds": time.time(),
        "metadata_compute_context": context, "historical_helper_commit": previous["helpers"]["source_commit"],
        "initial_comparisons_revalidated": initial,
        "historical_consumers_replayed_unchanged": True, "stdlib_ctypes_bootstrap": bootstrap,
        "numerical_imports": [], "gpu_operations": 0, "scientific_credit": 0, "admission_granted": False}


def validate_initial_cases(registry, validation, evidence):
    """Recheck all 120 original cases with the exact accepted historical validator on compute."""
    entries = registry.get("initial_ideas")
    require(type(entries) is list and len(entries) == 30 and len({row["name"] for row in entries}) == 30
        and registry.get("initial_matched_copy_counts") == [1, 2], "the initial thirty-idea registry is incomplete")
    count = 0
    for entry in entries:
        for context in validation.CONTEXTS:
            for copies in (1, 2):
                rows = [row for row in entry["results"][context] if type(row.get("copies")) is int and row["copies"] == copies]
                require(bool(rows), "an original hardware/copy comparison is absent")
                validation.validate_cohort_case(rows[-1]["result_path"], entry["name"], context, copies,
                    registry["reference_commit"], evidence)
                count += 1
    require(count == 120, "all 120 original comparisons must be revalidated")
    return count


def complete_capture(directory, plan):
    """Finish actual mapped backend capture after the unchanged core and full memory window."""
    compute_context(plan)
    contract(plan)
    directory = Path(directory)
    capture, raw = read(directory / "capture_result.json", 64 * 1024 * 1024)
    require(capture.get("capture_completed") is True, "post-core capture must contain a complete original window")
    module = importlib.import_module("direct_measurement")
    require(Path(module.__file__).resolve(strict=True) == directory / "code/direct_measurement.py",
        "backend completion must run through the exact staged direct helper")
    module.capture_backend_completion(directory, plan, capture)
    require((directory / "capture_result.json").read_bytes() == raw, "backend completion changed the immutable numerical capture")
    value = {"schema_version": 1, "kind": "direct_profile_post_core_backend_completion",
        "capture_result": {"path": str(directory / "capture_result.json"), "sha256": digest(raw)},
        "backend_capture": reference(directory / "direct_backend_capture.json"),
        "execution_identity": capture["execution_identity"], "original_capture_bytes_unchanged": True}
    publish(directory / "direct_profile_backend_completion.json", encoded(value))
    return value


def completed_capture_binding(directory, plan, captured):
    """Bind separate post-core and baseline proofs while retaining every original capture field."""
    compute_context(plan)
    validate_metadata(plan)
    directory = Path(directory)
    actual, raw = read(directory / "capture_result.json", 64 * 1024 * 1024)
    require(encoded(captured) == encoded(actual), "the outer profile publisher changed the original numerical capture")
    completion = checked_reference(reference(directory / "direct_profile_backend_completion.json"))
    require(type(completion.get("schema_version")) is int and completion["schema_version"] == 1
        and completion.get("kind") == "direct_profile_post_core_backend_completion"
        and completion.get("capture_result") == {"path": str(directory / "capture_result.json"), "sha256": digest(raw)}
        and completion.get("backend_capture") == reference(directory / "direct_backend_capture.json")
        and completion.get("execution_identity") == captured["execution_identity"]
        and completion.get("original_capture_bytes_unchanged") is True,
        "the outer profile lacks the exact same-process post-core backend completion")
    proof = checked_reference(reference(directory / PROOF_NAME))
    output = (directory / "direct_profile_baseline_stdout.raw").read_bytes()
    errors = (directory / "direct_profile_baseline_stderr.raw").read_bytes()
    returned = read(directory / "direct_profile_baseline_return.json")[0]
    supervisor = read(directory / "supervisor/direct_process_metadata.json")[0]
    require(proof.get("kind") == "actual_direct_profile_baseline_validation" and proof.get("passed") is True
        and proof.get("plan_sha256") == digest((directory / "measurement_plan.json").read_bytes())
        and proof.get("prerequisites") == plan["direct_profile_prerequisites"]
        and proof.get("parent_pid") == supervisor["after"]["identity"]["pid"]
        and type(proof.get("finished_at_seconds")) in (int, float) and math.isfinite(proof["finished_at_seconds"])
        and proof["finished_at_seconds"] <= captured["execution_identity"]["started_at_seconds"]
        and returned.get("command") == ["/usr/bin/python3", "-I", "-S", "-B", str(directory / "code" / NAME), "--plan-stdin"]
        and type(returned.get("returncode")) is int and returned["returncode"] == 0 and returned.get("error") is None
        and returned.get("stdout_sha256") == digest(output) and returned.get("stderr_sha256") == digest(errors)
        and errors == b"" and json.loads(output) == proof,
        "actual isolated accepted-baseline validation was not completed before this parent's capture")
    return {"direct_profile_backend_completion": reference(directory / "direct_profile_backend_completion.json"),
        "direct_profile_baseline_validation": reference(directory / PROOF_NAME)}


def main():
    """Expose only the private fresh metadata-validation child, with no numerical operation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan-stdin", action="store_true", required=True)
    parser.parse_args()
    raw = sys.stdin.buffer.read(LIMIT + 1)
    require(len(raw) <= LIMIT, "profile metadata exceeds its declared bound")
    plan = json.loads(raw)
    # This is deliberately before metadata validation, imports or any raw-result access.
    compute_context(plan)
    print(json.dumps(validate_actual(plan, raw), allow_nan=False))


if __name__ == "__main__":
    main()
