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
import importlib.util
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
SAVED_CONSUMERS = "direct_profile_saved_consumers.py"
PREFIX = "10_jax_exploration_platform/scripts/atari_pixel_campaign/"
PROTOCOL = "direct-h100-reviewed-common-profile-version-1"
PROCESS_PROTOCOL = "direct-h100-owned-nsight-and-retained-descriptor-version-1"
CARD_ONE = "GPU-fed69954-5a71-dbb2-ee91-544e0eb6b8a3"
NUMERICAL = {"jax", "jaxlib", "numpy", "torch", "flax", "optax", "scipy", "cupy", "jaxtari", "exploration_platform"}
INPUT_LIMIT = 1024 * 1024
LIMIT = 32 * 1024 * 1024
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


_SAVED_CONSUMER_MODULES = {}


def saved_consumers(plan=None):
    """Load only the adjacent explicitly declared read-only per-input metadata consumer."""
    path = Path(__file__).with_name(SAVED_CONSUMERS)
    require(path.resolve(strict=True) == path and path.is_file() and not path.is_symlink(),
        "the explicit saved profile-consumer helper is missing")
    raw = path.read_bytes()
    if plan is not None:
        require(plan["helpers"]["files"].get(PREFIX + SAVED_CONSUMERS) == digest(raw),
            "the exact saved profile-consumer helper changed")
    key = (str(path), digest(raw))
    if key in _SAVED_CONSUMER_MODULES:
        return _SAVED_CONSUMER_MODULES[key]
    specification = importlib.util.spec_from_file_location("explicit_profile_saved_consumers", path)
    module = importlib.util.module_from_spec(specification)
    exec(compile(raw, str(path), "exec"), module.__dict__)
    require(path.read_bytes() == raw, "the saved profile-consumer source changed during its actual import")
    _SAVED_CONSUMER_MODULES[key] = module
    return module


def helper_roster(plan=None):
    """Derive the exact declared helper union from source literals without importing a consumer."""
    require(plan is None or "native_process_control" not in plan,
        "formal profiles do not yet support the separate diagnostic-only native process contract")
    names = {"profile_selected.py", "direct_measurement.py", "configured_measurement.py",
        NAME, "direct_backend_image.py", "nsight_process_identity.py", "direct_profile_process.py",
        "direct_capture_release.py", "direct_tool_descendants.py", "profile_metadata_runtime.py", SAVED_CONSUMERS}
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
    expected = helper_roster(plan)
    require(expected == set(plan["helpers"]["files"])
        and all(plan["helpers"]["files"][relative] == digest(Path(__file__).with_name(Path(relative).name).read_bytes())
            for relative in expected),
        "the new profile requires its own exact helper closure, not the old confirmation helper")
    reader = saved_consumers(plan)
    reader.metadata_runtime(plan)
    require(encoded(plan.get("direct_profile_saved_consumer_protocol")) == encoded(reader.PROTOCOL),
        "new direct profile metadata requires the exact separate saved-consumer protocol")
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
        and PREFIX + "direct_common_confirmation.py" in previous["helpers"]["files"]
        and PREFIX + "direct_historical_consumer.py" in previous["helpers"]["files"],
        "the profile needs the actual separate direct common seven-by-ten confirmation")
    saved_consumers(plan).validate_common_metadata(previous, unit)
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
    plan["direct_profile_saved_consumer_protocol"] = dict(saved_consumers().PROTOCOL)
    runtime_path = Path(__file__).with_name("profile_metadata_runtime.py")
    spec = importlib.util.spec_from_file_location("_profile_configuration_metadata_runtime", runtime_path)
    module = importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    plan["direct_profile_metadata_protocol"] = dict(module.PROTOCOL)
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


def require_owned_stage(plan):
    """Operational metadata must execute its actual own staged closure, never an ambient checkout."""
    unit = Path(plan["campaign_run_dir"]) / "data" / plan["unit_id"]
    require(Path(__file__).resolve(strict=True) == unit / "code" / NAME,
        "operational profile metadata requires its exact own staged helper namespace")
    contract(plan)
    return unit


def before_compute(directory, plan):
    """Start one root origin and retain actual owned-child custody across the whole metadata phase."""
    started = time.monotonic();context = compute_context(plan)
    reader = saved_consumers(plan);runtime = reader.metadata_runtime(plan)
    runtime.capability()
    origin = runtime.origin(plan, reader.process(os.getpid()), started=started)
    budget = runtime.Budget(origin, plan);directory = Path(directory)
    publish(directory / "direct_profile_metadata_origin.json", encoded(origin))
    command = ["/usr/bin/python3", "-I", "-S", "-B", str(directory / "code" / NAME), "--plan-stdin"]
    parent = reader.process(os.getpid())
    unsigned = {"schema_version": 2, "plan": plan, "metadata_origin": origin, "parent": parent}
    envelope = {**unsigned, "challenge": digest(encoded(unsigned))}
    completed, scope = None, None
    try:
        with runtime.MetadataScope(plan, budget) as scope:
            with runtime.work_alarm(budget):
                require(Path(directory) == require_owned_stage(plan), "operational profile metadata directory differs from its exact plan unit")
                validate_metadata(plan)
                require(not NUMERICAL.intersection(sys.modules), "baseline metadata must precede numerical imports")
                completed = scope.run(command, input=encoded(envelope), env=dict(os.environ), maximum_bytes=LIMIT)
                require(completed.returncode == 0 and completed.stderr == b"" and len(completed.stdout) <= LIMIT,
                    "actual isolated direct baseline metadata failed")
                proof = reader.document(completed.stdout)
                require(proof.get("kind") == "actual_direct_profile_baseline_validation" and proof.get("passed") is True
                    and proof.get("plan_sha256") == digest(encoded(plan)) and proof.get("parent_pid") == os.getpid()
                    and proof.get("metadata_origin") == origin and proof.get("prerequisites") == plan["direct_profile_prerequisites"]
                    and proof.get("metadata_compute_context") == context and proof.get("numerical_imports") == [],
                    "isolated metadata returned another root origin, plan, context or role")
                child = completed.metadata_child
                worker_owner = proof["metadata_worker_scope"]["baseline"]["owner"]
                require(all(worker_owner[k] == child[k] for k in ("pid", "parent_pid", "process_start_time", "uids", "cgroup"))
                    and worker_owner["command"] == child["argv"], "metadata worker scope differs from the actual Popen child")
                runtime.validate_scope(proof["metadata_worker_scope"], budget, scope.observer, worker_owner)
                reader.validate_bundle(plan, proof["exact_saved_consumer_bundle"])
                require(proof.get("combination_metadata_dispatch_source_proof", {}).get("all_other_statement_syntax_unchanged") is True,
                    "isolated profile lost its unchanged combination metadata gates")
                budget.remaining()
        runtime.validate_scope(scope.proof, budget, scope.observer, scope.owner)
        with runtime.work_alarm(budget):
            validate_completed_metadata_phase(plan, proof, scope.proof, stdout=completed.stdout)
    except BaseException as error:
        capture_return(directory, command, None if completed is None else completed.returncode,
            b"" if completed is None else completed.stdout, b"" if completed is None else completed.stderr,
            {"type": type(error).__name__, "message": str(error), "metadata_origin": origin,
                "scope": None if scope is None else scope.proof})
        raise
    capture_return(directory, command, completed.returncode, completed.stdout, completed.stderr)
    publish(directory / "direct_profile_metadata_outer_scope.json", encoded(scope.proof))
    publish(directory / PROOF_NAME, encoded(proof))
    return proof


def validate_completed_metadata_phase(plan, proof, outer, *, stdout=None):
    """Require the new actual parent/worker/shared-origin closure; no historical proof bypass."""
    reader = saved_consumers(plan);runtime = reader.metadata_runtime(plan)
    require(type(proof.get("schema_version")) is int and proof["schema_version"] == 2
        and proof.get("kind") == "actual_direct_profile_baseline_validation" and proof.get("passed") is True
        and proof.get("plan_sha256") == digest(encoded(plan)) and proof.get("prerequisites") == plan["direct_profile_prerequisites"]
        and proof.get("numerical_imports") == [] and type(proof.get("gpu_operations")) is int and proof["gpu_operations"] == 0
        and type(proof.get("scientific_credit")) is int and proof["scientific_credit"] == 0
        and proof.get("admission_granted") is False, "new profile metadata cannot use an old or changed baseline proof")
    budget = runtime.Budget(proof["metadata_origin"], plan, verify_host=False)
    observer = runtime.process_observer(plan)
    owner, worker = outer["baseline"]["owner"], proof["metadata_worker_scope"]["baseline"]["owner"]
    for actual in (owner, worker):observer.validate_snapshot(actual, 2618919, owner["cgroup"])
    original = {key: owner[key] for key in ("pid", "parent_pid", "process_start_time", "uids", "cgroup")}
    original.update(argv=owner["command"], hostname=budget.origin["clock_domain"]["hostname"])
    command = ["/usr/bin/python3", "-I", "-S", "-B", str(Path(plan["campaign_run_dir"]) / "data" / plan["unit_id"] / "code" / NAME), "--plan-stdin"]
    require(budget.origin["owner"] == original and proof["parent_pid"] == worker["parent_pid"] == owner["pid"]
        and proof["validation_pid"] == worker["pid"] and worker["command"] == command
        and set(outer["spawned_roots"]) == {str(worker["pid"])}
        and outer["spawned_roots"][str(worker["pid"])]["command"] == command
        and outer["positive_nodes"][str(worker["pid"])]["initial"]["command"] == command
        and outer["root_waits"][str(worker["pid"])]["birth"] == worker["process_start_time"],
        "saved metadata baseline lost its actual original owner and exact waited worker")
    runtime.validate_scope(outer, budget, observer, owner)
    runtime.validate_scope(proof["metadata_worker_scope"], budget, observer, worker)
    require(type(stdout) is bytes and reader.document(stdout) == proof
        and outer["spawned_roots"][str(worker["pid"])]["captured_output"] == {
            "stdout": {"observed": True, "complete": True, "size_bytes": len(stdout), "sha256": digest(stdout)},
            "stderr": {"observed": True, "complete": True, "size_bytes": 0, "sha256": digest(b"")}},
        "saved outer metadata output differs from its actual complete child capture")
    worker_record = {key: worker[key] for key in ("pid", "parent_pid", "process_start_time", "uids", "cgroup")}
    worker_record.update(argv=worker["command"], hostname=budget.origin["clock_domain"]["hostname"])
    require(all(value["request"]["metadata_origin"] == budget.origin and value["request"]["parent"] == worker_record
            for value in proof["exact_saved_consumer_bundle"].values()),
        "saved baseline readers did not retain this exact actual parent origin")
    worker_roots = proof["metadata_worker_scope"]["spawned_roots"]
    require(set(worker_roots) == {str(value["proof"]["child"]["pid"]) for value in proof["exact_saved_consumer_bundle"].values()},
        "metadata worker root set differs from its three actual isolated readers")
    for value in proof["exact_saved_consumer_bundle"].values():
        waited = value["actual_child_wait"];root = worker_roots[str(value["proof"]["child"]["pid"])]
        require(root["command"] == waited["command"] and root["captured_output"]["stdout"]["sha256"] == waited["stdout_sha256"]
            and root["captured_output"]["stderr"]["sha256"] == waited["stderr_sha256"],
            "saved reader output differs from its actual parent kernel scope")
    require(type(proof.get("finished_at_seconds")) in (int,float) and math.isfinite(proof["finished_at_seconds"])
        and proof["metadata_worker_scope"]["terminal"]["observed_at_seconds"] <= proof["finished_at_seconds"]
        <= outer["root_waits"][str(worker["pid"])]["finished_at_seconds"], "saved metadata phase completion chronology changed")
    reader.validate_bundle(plan, proof["exact_saved_consumer_bundle"])
    common = validate_metadata(plan)
    new = "accepted_evidence_snapshot_protocol" in common
    require(type(proof.get("historically_verified_initial_cases")) is int and proof["historically_verified_initial_cases"] == 120
        and proof.get("original_cases_reexecuted") is (not new)
        and proof.get("historical_consumers_replayed_unchanged") is (not new)
        and proof.get("accepted_history_reused") is new
        and proof.get("current_confirmation_consumer_replayed_unchanged") is True,
        "saved baseline changed its explicit historical reuse scope")
    if new:
        expected = {"historically_verified_initial_cases": 120, "original_cases_reexecuted": False,
            "snapshot": common["accepted_evidence_snapshot_inputs"], "source_commit": plan["source_commit"],
            "configuration": plan["configuration"], "copy_seeds": plan["copy_seeds"],
            "scientific_credit": 0, "operational_admission": False}
        require(encoded(proof.get("initial_history_fact")) == encoded(expected)
            and type(proof.get("initial_comparisons_revalidated")) is int and proof["initial_comparisons_revalidated"] == 0,
            "new29 baseline changed its exact reviewed historical fact or invented case replay")
    else:
        require(type(proof.get("initial_comparisons_revalidated")) is int and proof["initial_comparisons_revalidated"] == 120,
            "legacy raw history did not retain its exact original120 case gate")
    return proof


def history_pointer(plan, common):
    """Describe only a genuine accepted common snapshot; no new qualification or future result is created."""
    require("accepted_evidence_snapshot_protocol" in common, "new profile history pointer requires the exact new29 reviewed fact")
    return {"schema_version": 1, "kind": "profile_reviewed_accepted_history",
        "campaign_run_dir": plan["campaign_run_dir"], "profile_unit_id": plan["unit_id"],
        "source_commit": plan["source_commit"], "configuration": plan["configuration"], "copy_seeds": plan["copy_seeds"],
        "common_plan": reference(Path(common["campaign_run_dir"]) / "data" / common["unit_id"] / "measurement_plan.json"),
        "accepted_evidence_snapshot_inputs": common["accepted_evidence_snapshot_inputs"],
        "historically_verified_initial_cases": 120, "original_cases_reexecuted": False,
        "scientific_credit": 0, "operational_admission": False}

def prepare_history_pointer(path, plan, common_plan):
    """Write one metadata-only profile pointer from actual existing common plan/reviewed fact bytes."""
    path = Path(path)
    require(path == Path(plan["campaign_run_dir"]) / "data" / plan["unit_id"] / "direct_history/inputs.json",
        "profile history pointer must use its own exact reserved path")
    saved_consumers().validate_common_metadata(common_plan, Path(common_plan["campaign_run_dir"]) / "data" / common_plan["unit_id"])
    value = history_pointer(plan, common_plan)
    publish(path, encoded(value))
    return reference(path)


def validate_current_history(plan):
    """Use the exact new29 historical fact; keep the explicit old28 raw path unchanged."""
    compute_context(plan);common = validate_metadata(plan)
    if "accepted_evidence_snapshot_protocol" in common:
        require(plan["direct_binding"]["frozen_inputs"]["path"] == str(Path(plan["campaign_run_dir"]) / "data" / plan["unit_id"] / "direct_history/inputs.json"),
            "new29 profile history pointer must use its own exact reserved unit path")
        actual = checked_reference(plan["direct_binding"]["frozen_inputs"])
        require(encoded(actual) == encoded(history_pointer(plan, common)), "profile history pointer changed its exact reviewed common fact")
        reader = saved_consumers(plan);unit = Path(common["campaign_run_dir"]) / "data" / common["unit_id"]
        snapshot = reader.reviewed_snapshot(common, unit)
        require(snapshot.qual["review"] == plan["direct_profile_prerequisites"]["qualification_review"]
            and snapshot.correctness["review"] == plan["direct_profile_prerequisites"]["correctness_review"]
            and snapshot.correctness["result"] == plan["direct_profile_prerequisites"]["correctness"],
            "profile history changed its exact original qualification/correctness facts")
        return {"historically_verified_initial_cases": 120, "original_cases_reexecuted": False,
            "snapshot": common["accepted_evidence_snapshot_inputs"], "source_commit": plan["source_commit"],
            "configuration": plan["configuration"], "copy_seeds": plan["copy_seeds"],
            "scientific_credit": 0, "operational_admission": False}
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


def validate_actual(plan, raw, metadata_origin):
    """Keep exact saved namespaces inside one actual shared metadata work/cleanup budget."""
    reader = saved_consumers(plan);runtime = reader.metadata_runtime(plan)
    budget = runtime.Budget(metadata_origin, plan)
    with runtime.MetadataScope(plan, budget) as scope:
        with runtime.work_alarm(budget):
            """Keep raw consumers separate while preserving every accepted combination check."""
            context = compute_context(plan)
            require_owned_stage(plan)
            previous = validate_metadata(plan)
            inputs = plan["direct_profile_prerequisites"]
            reader = saved_consumers(plan)
            history_fact = validate_current_history(plan)
            bootstrap = reader.install_profile_audit(plan, scope)
            require(not NUMERICAL.intersection(sys.modules), "direct baseline validation initialized numerical code")
            bundle = reader.read_all(plan, budget, scope)
            accepted, dispatch_source_proof = reader.combine(plan, bundle)
            review = checked_reference(inputs["confirmation_review"])
            audit = checked_reference(review["raw_audit"])
            require(audit.get("passed") is True and audit.get("unchanged_staged_consumer_passed") is True
                and audit.get("result") == inputs["confirmation"] and audit.get("plan") == inputs["confirmation_plan"]
                and audit.get("raw_window_recomputation_passed") is True,
                "the genuine independent timing review lacks its complete actual raw audit")
            result = checked_actual_reference(plan, inputs["confirmation"])
            path = Path(__file__).with_name("post_comparison_validation.py")
            expected = plan["helpers"]["files"][PREFIX + path.name]
            require(expected == previous["helpers"]["files"][PREFIX + path.name],
                "initial comparisons require the unchanged accepted metadata validator")
            validation = reader.source_module("profile_initial_comparison_metadata_validator", path, expected)
            if "accepted_evidence_snapshot_protocol" in previous:
                require(history_fact["historically_verified_initial_cases"] == 120 and history_fact["original_cases_reexecuted"] is False,
                    "new29 history must bind the genuine complete accepted fact")
                initial = 120
            else:
                evidence = validation.Evidence()
                registry_path = Path(plan["initial_registry_record"])
                unit = Path(plan["campaign_run_dir"]) / "data" / plan["unit_id"]
                require(registry_path == unit / "inputs/study_registry.json"
                    and digest(registry_path.read_bytes()) == plan["inputs"]["study_registry"]["sha256"],
                    "the staged initial registry changed before direct profile validation")
                registry = read(registry_path, 64 * 1024 * 1024)[0]
                initial = validate_initial_cases(registry, validation, evidence)
            require(accepted["configuration"] == plan["configuration"] and accepted["paired_rounds"] == 7
                and result["copy_seeds"] == plan["copy_seeds"] and result["execution_identity"]["finished_at_seconds"]
                <= review["reviewed_at_seconds"], "the actual accepted timing result changed configuration, seeds or review order")
            reader.validate_bundle(plan, bundle)
            require(not NUMERICAL.intersection(sys.modules), "a numerical package entered the saved direct consumer")
            proof = {"schema_version": 2, "kind": "actual_direct_profile_baseline_validation", "passed": True,
                "plan_sha256": digest(raw), "prerequisites": inputs, "accepted_combination": accepted,
                "parent_pid": os.getppid(), "validation_pid": os.getpid(), "finished_at_seconds": time.time(),
                "metadata_compute_context": context, "historical_helper_commit": previous["helpers"]["source_commit"],
                "initial_comparisons_revalidated": 0 if "accepted_evidence_snapshot_protocol" in previous else initial,
                "initial_history_fact": history_fact, "exact_saved_consumer_bundle": bundle,
                "combination_metadata_dispatch_source_proof": dispatch_source_proof,
                "historical_consumers_replayed_unchanged": "accepted_evidence_snapshot_protocol" not in previous,
                "accepted_history_reused": "accepted_evidence_snapshot_protocol" in previous,
                "current_confirmation_consumer_replayed_unchanged": True, "metadata_origin": metadata_origin,
                "historically_verified_initial_cases": initial,
                "original_cases_reexecuted": "accepted_evidence_snapshot_protocol" not in previous, "stdlib_ctypes_bootstrap": bootstrap,
                "numerical_imports": [], "gpu_operations": 0, "scientific_credit": 0, "admission_granted": False}

    runtime.validate_scope(scope.proof, budget, scope.observer, scope.owner)
    proof["metadata_worker_scope"] = scope.proof
    proof["finished_at_seconds"] = time.time()
    return proof


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
    outer = checked_reference(reference(directory / "direct_profile_metadata_outer_scope.json"))
    output = (directory / "direct_profile_baseline_stdout.raw").read_bytes()
    validate_completed_metadata_phase(plan, proof, outer, stdout=output)
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
    saved_consumers(plan).validate_bundle(plan, proof["exact_saved_consumer_bundle"])
    require(proof.get("combination_metadata_dispatch_source_proof", {}).get("all_other_statement_syntax_unchanged") is True,
        "the saved profile baseline omitted its exact metadata-dispatch source proof")
    return {"direct_profile_backend_completion": reference(directory / "direct_profile_backend_completion.json"),
        "direct_profile_baseline_validation": reference(directory / PROOF_NAME)}


def main():
    """Accept only the owned parent's same-origin bounded metadata envelope."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan-stdin", action="store_true", required=True);parser.parse_args()
    raw = sys.stdin.buffer.read(INPUT_LIMIT + 1)
    require(0 < len(raw) <= INPUT_LIMIT, "profile metadata envelope exceeds its exact input bound")
    reader = saved_consumers();envelope = reader.document(raw)
    require(type(envelope) is dict and set(envelope) == {"schema_version", "plan", "metadata_origin", "parent", "challenge"}
        and type(envelope["schema_version"]) is int and envelope["schema_version"] == 2,
        "new profile metadata requires its exact actual root envelope")
    plan = envelope["plan"];compute_context(plan)
    unsigned = {k: v for k, v in envelope.items() if k != "challenge"}
    require(envelope["challenge"] == digest(encoded(unsigned)) and envelope["parent"] == reader.process(os.getppid())
        and envelope["metadata_origin"]["owner"] == envelope["parent"], "metadata envelope parent or original budget changed")
    print(json.dumps(validate_actual(plan, encoded(plan), envelope["metadata_origin"]), allow_nan=False))


if __name__ == "__main__":
    main()
