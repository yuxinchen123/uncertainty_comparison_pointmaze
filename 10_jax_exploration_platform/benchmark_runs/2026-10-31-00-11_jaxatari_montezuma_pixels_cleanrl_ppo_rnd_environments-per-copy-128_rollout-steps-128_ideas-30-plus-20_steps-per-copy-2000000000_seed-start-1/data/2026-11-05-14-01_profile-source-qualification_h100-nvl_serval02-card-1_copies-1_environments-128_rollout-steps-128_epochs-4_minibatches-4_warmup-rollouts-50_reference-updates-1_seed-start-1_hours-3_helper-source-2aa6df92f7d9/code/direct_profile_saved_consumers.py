"""Read each direct profile prerequisite with its exact saved metadata consumer.

This helper never admits profiling or executes numerical code. Each saved
record is read in a different standard-library process. The current profile
backend helper cannot replace the qualification or confirmation backend helper.
"""
import argparse
import ast
import base64
import hashlib
import importlib.abc
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import select
import signal
import socket
import stat
import statistics
import subprocess
import sys
import time

NAME = "direct_profile_saved_consumers.py"
PREFIX = "10_jax_exploration_platform/scripts/atari_pixel_campaign/"
ORIGINAL_COMMIT = "0a880984b4a03a2ff7335337cc18576628bc9fc1"
CARD_ONE = "GPU-fed69954-5a71-dbb2-ee91-544e0eb6b8a3"
PROTOCOL = {"schema_version": 2, "capability": "separate_exact_saved_direct_readers_before_profile",
    "operations": ["qualification", "correctness", "confirmation"], "operational_admission": False}
NUMERICAL = {"jax", "jaxlib", "jax_cuda12_plugin", "jax_cuda13_plugin", "numpy", "torch", "flax", "optax",
    "scipy", "cupy", "nvidia", "tensorflow", "numba", "ml_dtypes", "jaxtari", "exploration_platform"}
INPUT_LIMIT = 1024 * 1024
LIMIT = 32 * 1024 * 1024
DOCUMENT_LIMIT = 64 * 1024 * 1024
STATISTIC_PRIMITIVES = {"execution_identity", "runtime_environment", "direct_record", "configured_record",
    "identity", "_identity_fields", "correctness", "configuration", "canonical_digest", "measured_round"}
_REVIEWED_SNAPSHOT_MODULES = {}


def require(value, message):
    """Refuse a changed source, input, process or metadata-only role."""
    if not value:
        raise ValueError(message)


def encoded(value):
    """Use finite stable JSON when joining independently produced metadata."""
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def digest(raw):
    """Identify supplied bytes without loading a library or numerical module."""
    return hashlib.sha256(raw).hexdigest()


def ordinary(path, *, directory=False):
    """Require a canonical ordinary file or directory without accepting a link."""
    path = Path(path)
    require(path.is_absolute() and path.resolve(strict=True) == path and not path.is_symlink(),
        "saved evidence requires an exact ordinary path")
    value = path.stat()
    require(value.st_uid == 2618919 and (stat.S_ISDIR(value.st_mode) if directory else stat.S_ISREG(value.st_mode)),
        "saved evidence belongs to another owner or file kind")
    return path


def document(raw):
    """Refuse duplicate keys and nonfinite values in saved JSON metadata."""
    def unique(pairs):
        """Keep one value for every explicit key."""
        value = {}
        for key, item in pairs:
            require(key not in value, "duplicate saved-consumer JSON key")
            value[key] = item
        return value
    return json.loads(raw, object_pairs_hook=unique,
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError("nonfinite saved-consumer JSON")))


def read(path, maximum=DOCUMENT_LIMIT):
    """Read a bounded pinned descriptor and reject concurrent file replacement."""
    path = ordinary(path)
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        before = os.fstat(descriptor)
        require(0 < before.st_size <= maximum, "saved evidence exceeds its exact size bound")
        blocks, remaining = [], before.st_size
        while remaining:
            block = os.read(descriptor, min(remaining, 1024 * 1024))
            require(bool(block), "saved evidence ended before its pinned size")
            blocks.append(block); remaining -= len(block)
        raw = b"".join(blocks)
        after, current = os.fstat(descriptor), path.stat()
        fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns", "st_mode", "st_uid")
        require(all(getattr(before, key) == getattr(after, key) == getattr(current, key) for key in fields),
            "saved evidence changed during its descriptor read")
    finally:
        os.close(descriptor)
    return document(raw), raw


def reference_shape(value):
    """Require one exact absolute path and a complete lowercase SHA-256 hash."""
    require(type(value) is dict and set(value) == {"path", "sha256"}
        and type(value["path"]) is str and Path(value["path"]).is_absolute()
        and str(Path(value["path"])) == value["path"] and ".." not in Path(value["path"]).parts
        and type(value["sha256"]) is str and re.fullmatch(r"[0-9a-f]{64}", value["sha256"]),
        "an exact saved-consumer reference is required")
    return value


def checked(value):
    """Read only the exact bytes named by a saved metadata reference."""
    reference_shape(value)
    result, raw = read(value["path"])
    require(digest(raw) == value["sha256"], "saved-consumer reference bytes changed")
    return result


def reference(path):
    """Identify an existing bounded JSON record, without writing into its old unit."""
    unused, raw = read(path)
    return {"path": str(path), "sha256": digest(raw)}


def process(pid):
    """Read only this process or a positively named newly spawned own process."""
    require(type(pid) is int and pid > 0, "an exact positive metadata process is required")
    directory = Path("/proc") / str(pid)
    before = directory.stat()
    require(before.st_uid == 2618919, "refuse process files belonging to another owner")
    raw = (directory / "stat").read_text(); fields = raw.rsplit(")", 1)[1].split()
    status = (directory / "status").read_text()
    uids = re.findall(r"^Uid:\s*([0-9]+)\s+([0-9]+)\s+([0-9]+)\s+([0-9]+)\s*$", status, re.M)
    require(len(uids) == 1 and [int(value) for value in uids[0]] == [2618919] * 4,
        "all four metadata-process UIDs must belong to this owner")
    argv = (directory / "cmdline").read_bytes().rstrip(b"\0").split(b"\0")
    cgroup = (directory / "cgroup").read_text()
    after = (directory / "stat").read_text().rsplit(")", 1)[1].split()
    require(after[19] == fields[19] and directory.stat().st_ino == before.st_ino,
        "metadata process birth changed during its bounded read")
    return {"pid": pid, "parent_pid": int(fields[1]), "process_start_time": fields[19],
        "uids": [int(value) for value in uids[0]], "argv": [value.decode() for value in argv],
        "hostname": socket.gethostname().split(".")[0], "cgroup": cgroup}


def validate_process_record(value):
    """Require exact typed own-process fields in both actual and saved child custody."""
    require(type(value) is dict and set(value) == {"pid", "parent_pid", "process_start_time", "uids", "argv", "hostname", "cgroup"}
        and type(value["pid"]) is int and value["pid"] > 0
        and type(value["parent_pid"]) is int and value["parent_pid"] > 0 and value["parent_pid"] != value["pid"]
        and type(value["process_start_time"]) is str and re.fullmatch(r"[1-9][0-9]*", value["process_start_time"])
        and type(value["uids"]) is list and len(value["uids"]) == 4 and all(type(uid) is int and uid == 2618919 for uid in value["uids"])
        and type(value["argv"]) is list and value["argv"] and all(type(word) is str and word for word in value["argv"])
        and type(value["hostname"]) is str and bool(value["hostname"]) and type(value["cgroup"]) is str and bool(value["cgroup"]),
        "saved metadata process lost its typed PID, parent, birth, all UIDs, command, host or cgroup")
    return value


def compute_context(plan):
    """Require the actual owner and kernel compute context before any raw result read."""
    current = process(os.getpid())
    require(os.getuid() == os.geteuid() == 2618919, "saved readers require the exact campaign owner")
    job, step = os.environ.get("SLURM_JOB_ID"), os.environ.get("SLURM_STEP_ID")
    if current["hostname"] == "serval02":
        require(job is None and step is None, "direct saved reads cannot borrow a scheduler context")
    else:
        require(type(job) is str and re.fullmatch(r"[1-9][0-9]*", job)
            and type(step) is str and re.fullmatch(r"[0-9]+", step), "saved readers require a numeric owned compute step")
        expected = "0::/system.slice/slurmstepd.scope/job_" + job + "/step_" + step + "/user/task_0"
        require(current["cgroup"].splitlines() == [expected], "saved reader differs from its actual kernel compute step")
        ledger = ordinary(Path(plan["campaign_run_dir"]) / "slurm/submitted_jobids_uva.txt")
        require(ledger.read_text().splitlines().count(job) == 1, "saved reader is outside this campaign's owned ledger")
    return {"hostname": current["hostname"], "job_id": job, "step_id": step, "uids": current["uids"]}


def contract(plan):
    """Accept only the explicit profile metadata capability, without admitting actual profiling."""
    require(plan.get("kind") == "selected_configuration_profile" and plan.get("hardware_class") == "h100"
        and plan.get("gpu_uuid") == CARD_ONE and "direct_binding" in plan
        and encoded(plan.get("direct_profile_saved_consumer_protocol")) == encoded(PROTOCOL)
        and not any(key in plan for key in ("configured_isolation", "configured_group", "final_protocol",
            "final_confirmation_protocol", "device_memory_limit_protocol", "completed_study", "scaling_protocol", "native_process_control")),
        "saved consumers support only the separately declared direct profile metadata protocol")
    require(plan["helpers"]["files"].get(PREFIX + NAME) == digest(Path(__file__).read_bytes()),
        "the current profile omitted or changed its exact saved-consumer helper")
    metadata_runtime(plan)
    inputs = plan.get("direct_profile_prerequisites")
    require(type(inputs) is dict and set(inputs) == {"confirmation_plan", "confirmation", "confirmation_review",
        "correctness", "correctness_review", "qualification", "qualification_review"}, "complete exact saved prerequisite references are required")
    for value in inputs.values():
        reference_shape(value)
    return inputs


def metadata_inputs(plan):
    """Validate reserved plan paths without opening scientific result files."""
    inputs = contract(plan)
    campaign = ordinary(plan["campaign_run_dir"], directory=True)
    common = checked(inputs["confirmation_plan"])
    common_unit = Path(inputs["confirmation_plan"]["path"]).parent
    require(inputs["confirmation_plan"]["path"] == str(common_unit / "measurement_plan.json")
        and common_unit == campaign / "data" / common["unit_id"]
        and inputs["confirmation"]["path"] == str(common_unit / "result.json")
        and common.get("kind") == "selected_configuration_measurement" and common.get("mode") == "confirmation"
        and common.get("common_correctness_inputs") == {"result": inputs["correctness"], "review": inputs["correctness_review"]}
        and common.get("direct_qualification_inputs") == {"result": inputs["qualification"], "review": inputs["qualification_review"]},
        "the profile changed its actual common, correctness or qualification plan")
    require(all(type(common.get(key)) is int and common[key] == value for key, value in (
        ("rounds", 7), ("iterations", 10), ("minimum_updates_per_round", 10))),
        "saved common confirmation requires seven pairs of ten complete updates")
    for key in ("source_root", "source_commit", "core", "paired_source", "python", "campaign_run_dir",
            "configuration", "runtime_environment", "asset_record", "copy_seeds"):
        require(encoded(common[key]) == encoded(plan[key]), "saved common confirmation changed " + key)
    require(common["direct_binding"]["profile"] == plan["direct_binding"]["profile"], "saved direct target profile changed")
    return common, common_unit


def validate_common_metadata(previous, directory):
    """Check the complete declared staged common source set without reading a result."""
    require(previous.get("kind") == "selected_configuration_measurement" and previous.get("mode") == "confirmation"
        and "common_correctness_inputs" in previous
        and ("direct_historical_consumer_protocol" in previous) != ("accepted_evidence_snapshot_protocol" in previous),
        "the saved prerequisite is not its explicit current-common metadata plan")
    sources = {}
    for relative, expected in previous["helpers"]["files"].items():
        require(type(relative) is str and relative.startswith(PREFIX) and relative[len(PREFIX):] == Path(relative).name
            and relative.endswith(".py") and type(expected) is str and re.fullmatch(r"[0-9a-f]{64}", expected),
            "a saved common metadata source path or hash changed")
        path = ordinary(Path(directory) / "code" / Path(relative).name)
        raw = path.read_bytes()
        require(digest(raw) == expected, "a staged common metadata helper changed")
        sources[Path(relative).stem] = {"path": str(path), "sha256": expected, "raw": raw}
    snapshot = "accepted_evidence_snapshot_protocol" in previous
    source_roster(sources, True, accepted_snapshot=snapshot)
    # Before: a declared protocol; after: equality to the exact hash-bound saved source's literal declaration.
    source = sources["accepted_evidence_snapshot" if snapshot else "direct_common_confirmation"]["raw"]
    name = "PROTOCOL" if snapshot else "HISTORICAL_PROTOCOL"
    values = [node.value for node in ast.parse(source).body if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == name for target in node.targets)]
    require(len(values) == 1, "one exact saved metadata protocol declaration is required")
    field = "accepted_evidence_snapshot_protocol" if snapshot else "direct_historical_consumer_protocol"
    require(encoded(previous[field]) == encoded(ast.literal_eval(values[0])),
        "the saved common protocol differs from its own exact staged source")
    return {name: {key: value[key] for key in ("path", "sha256")} for name, value in sources.items()}


def source_module(name, path, expected):
    """Load one exact metadata helper under a private name, never replacing a historical module."""
    path = ordinary(path); raw = path.read_bytes()
    require(digest(raw) == expected, "the explicitly selected metadata helper bytes changed")
    specification = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(specification)
    exec(compile(raw, str(path), "exec"), module.__dict__)
    require(digest(ordinary(path).read_bytes()) == expected, "the pinned metadata helper changed during its actual import")
    return module


def literal_helpers(raw):
    """Read one source's explicit helper declaration without executing it."""
    values = [node.value for node in ast.parse(raw).body if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "HELPERS" for target in node.targets)]
    require(len(values) == 1, "one literal historical helper declaration is required")
    result = ast.literal_eval(values[0])
    require(type(result) is tuple and all(type(name) is str and Path(name).name == name and name.endswith(".py") for name in result),
        "historical helper declarations need exact adjacent Python filenames")
    return result


def source_roster(sources, common, *, accepted_snapshot=False):
    """Derive the exact old or common source set instead of accepting a helper count."""
    require(type(common) is bool and type(accepted_snapshot) is bool and (not accepted_snapshot or common)
        and "measure_selected" in sources and "serval02_card1_binding" in sources,
        "the historical source declaration is incomplete")
    names = set(literal_helpers(sources["measure_selected"]["raw"]))
    names.update(literal_helpers(sources["serval02_card1_binding"]["raw"]))
    names.update(("direct_measurement.py", "configured_measurement.py", "direct_backend_image.py"))
    if common:
        names.update(("direct_common_confirmation.py", "direct_historical_consumer.py"))
    if accepted_snapshot:
        names.add("accepted_evidence_snapshot.py")
    require(names == {name + ".py" for name in sources}, "the complete exact saved helper source set differs")
    return sorted(names)


def metadata_runtime(plan):
    """Require the exact declared current metadata runtime, never a historical helper alias."""
    path = Path(__file__).with_name("profile_metadata_runtime.py")
    require(plan["helpers"]["files"].get(PREFIX + path.name) == digest(path.read_bytes()),
        "the current profile must stage its exact shared metadata runtime")
    module = source_module("_profile_saved_metadata_runtime", path, plan["helpers"]["files"][PREFIX + path.name])
    require(encoded(plan.get("direct_profile_metadata_protocol")) == encoded(module.PROTOCOL),
        "new saved readers require the exact shared-at-most300second metadata protocol")
    return module


def reviewed_snapshot(common, directory):
    """Reuse one exact source-bound metadata consumer; retain its permanent failure semantics."""
    require("accepted_evidence_snapshot_protocol" in common, "reviewed snapshot reuse is new29 only")
    path = ordinary(Path(directory) / "code/accepted_evidence_snapshot.py")
    expected = common["helpers"]["files"][PREFIX + path.name]
    require(digest(path.read_bytes()) == expected, "the exact reviewed snapshot consumer source changed")
    key = (str(path), expected)
    if key not in _REVIEWED_SNAPSHOT_MODULES:
        _REVIEWED_SNAPSHOT_MODULES[key] = source_module("_profile_exact_reviewed_snapshot_fact", path, expected)
    return _REVIEWED_SNAPSHOT_MODULES[key].consume(common)


def policy_roots(plan, common, directory, guard):
    """Derive new29 roots from its reviewed historical fact, preserving old28's own policy."""
    if "accepted_evidence_snapshot_protocol" not in common:
        # This old28 branch does not import a canonical accepted-snapshot helper.
        guard.contract(common)
        return guard.declared_git_roots(common), {"kind": "exact_old28_common_policy"}
    snapshot = reviewed_snapshot(common, directory)
    roots = {"/p/rlprojects/RND/08_cleanrl_ppo_rnd/cleanrl",
        "/p/rlprojects/RND/reference_repos/jaxtari_pixel_study"}
    plans = [plan, common]
    manifests = []
    for accepted in (snapshot.qual, *snapshot.proof["accepted"]["correctness"].values()):
        actual = snapshot.files.json(accepted["plan"]);plans.append(actual)
        manifest_ref = accepted["historical_manifest"]
        manifest = snapshot.files.json(manifest_ref);staging = Path(manifest_ref["path"]).parent
        require(manifest.get("kind") == "portable_raw_evidence" and len(manifest["git_roots"]) == 8,
            "new29 policy requires all exact reviewed original source-root mappings")
        targets = set()
        for row in manifest["git_roots"]:
            relative = Path(row["relative_path"])
            require(not relative.is_absolute() and ".." not in relative.parts and str(relative) not in {"", "."}
                and re.fullmatch(r"[0-9a-f]{40}", row["commit"]), "reviewed source root mapping changed")
            target = ordinary(staging / relative, directory=True)
            require(target.is_relative_to(staging) and str(target) not in targets, "reviewed metadata source roots repeat or escape history")
            targets.add(str(target))
        require(len(targets) == 8, "new29 metadata policy omitted an original source root")
        roots.update(targets);manifests.append(manifest_ref)
    for value in plans:
        require(value["campaign_run_dir"] == plan["campaign_run_dir"], "metadata source roots cross campaigns")
        for key in ("helpers", "core", "paired_source"):
            root = ordinary(value[key]["source_root"], directory=True)
            require(re.fullmatch(r"[0-9a-f]{40}", value[key]["source_commit"]), "source root lacks exact immutable commit")
            roots.add(str(root))
    return roots, {"kind": "new29_reviewed_snapshot_metadata_policy",
        "snapshot": common["accepted_evidence_snapshot_inputs"], "original_manifests": manifests,
        "historical_payloads_rehashed": False}

def common_policy(plan, common, directory, scope, *, allow_saved_child=False):
    """Use an explicitly source-bound policy interface without ambient common contract imports."""
    validate_common_metadata(common, directory)
    path = directory / "code/direct_common_confirmation.py"
    expected = common["helpers"]["files"][PREFIX + path.name]
    guard = source_module("_profile_stateless_common_metadata_policy", path, expected)
    environment = guard.metadata_environment(os.environ)
    os.environ.clear();os.environ.update(environment)
    bootstrap = guard.stdlib_ctypes_bootstrap();guard.validate_bootstrap_record(bootstrap)
    roots, history = policy_roots(plan, common, directory, guard)
    # Bind the unchanged source-level policy primitives explicitly. In particular
    # do not call new29 contract() until its complete canonical namespace exists.
    names = ("metadata_environment", "require_metadata_environment", "source_inspection_git",
        "metadata_event_guard", "verify_repository_configuration", "finish_repository_configurations")
    tree = ast.parse(path.read_bytes())
    primitives = {node.name: digest(ast.dump(node, include_attributes=False).encode()) for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in names}
    require(set(primitives) == set(names), "saved common policy source lacks its exact metadata primitive interface")
    scope.git_guard = lambda event, values: guard.metadata_event_guard(event, values, roots)
    def policy(event, values):
        allowed = ["/usr/bin/python3", "-I", "-S", "-B", str(ordinary(__file__)), "--read-only-stdin"]
        if allow_saved_child and event == "subprocess.Popen" and values[1] == allowed:
            require(values[0] == allowed[0] and values[2] is None
                and digest(Path(__file__).read_bytes()) == plan["helpers"]["files"][PREFIX + NAME],
                "only the exact current saved-reader source may start")
            guard.require_metadata_environment(values[3]);return
        readonly_audit(guard, roots, directory, common, event, values)
    sys.addaudithook(policy)
    return guard, environment, bootstrap, {"guard": {"path": str(path), "sha256": expected},
        "stateless_policy_function_asts": primitives, "common_helper_closure": common["helpers"],
        "declared_git_roots": sorted(roots), "history_policy": history,
        "new29_contract_executed_in_outer_namespace": False}


def install_profile_audit(plan, scope):
    """Protect the outer coordinator without common contract in an ambient namespace."""
    contract(plan);compute_context(plan)
    common, directory = metadata_inputs(plan)
    guard, environment, bootstrap, description = common_policy(plan, common, directory, scope, allow_saved_child=True)
    scope.common_policy_evidence = description
    return bootstrap


def source_closure(previous, directory, guard, environment, *, common):
    """Join every actual staged helper to its saved source bytes and immutable Git commit."""
    value = previous["helpers"]; root = ordinary(value["source_root"], directory=True)
    commit = value["source_commit"]
    require(type(commit) is str and re.fullmatch(r"[0-9a-f]{40}", commit), "a complete saved helper commit is required")
    require(common or commit == ORIGINAL_COMMIT, "original qualification and correctness need their exact original helper commit")
    head = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], env=environment, text=True).strip()
    require(head == commit, "the saved helper checkout changed its exact commit")
    sources = {}
    for relative, expected in value["files"].items():
        require(type(relative) is str and relative.startswith(PREFIX) and relative[len(PREFIX):] == Path(relative).name
            and relative.endswith(".py") and re.fullmatch(r"[0-9a-f]{64}", expected), "a saved helper path or hash is invalid")
        staged = ordinary(directory / "code" / Path(relative).name)
        source = ordinary(root / relative)
        raw = staged.read_bytes()
        committed = subprocess.check_output(["git", "-C", str(root), "show", commit + ":" + relative], env=environment)
        require(digest(raw) == expected == digest(source.read_bytes()) == digest(committed), "a saved staged/source/Git helper changed")
        sources[Path(relative).stem] = {"path": str(staged), "sha256": expected, "raw": raw}
    source_roster(sources, common, accepted_snapshot="accepted_evidence_snapshot_protocol" in previous)
    return sources


class ExactSavedLoader(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """Load only the complete already verified source set under its own original module names."""

    def __init__(self, sources):
        """Retain exact staged names and the modules actually executed by the saved reader."""
        self.sources, self.loaded = sources, {}

    def find_spec(self, fullname, path=None, target=None):
        """Deny numerical packages and resolve only explicitly bound historical helper names."""
        require(fullname.split(".", 1)[0] not in NUMERICAL, "numerical imports are forbidden in saved readers")
        return importlib.util.spec_from_loader(fullname, self, origin=self.sources[fullname]["path"]) if fullname in self.sources else None

    def create_module(self, specification):
        """Use ordinary Python metadata modules with no runtime substitutions."""
        return None

    def exec_module(self, module):
        """Execute exactly the pinned source bytes under their original staged filename."""
        source = self.sources[module.__name__]
        require(digest(source["raw"]) == source["sha256"] == digest(ordinary(source["path"]).read_bytes()),
            "saved helper source changed before import")
        module.__file__ = source["path"]
        self.loaded[module.__name__] = {key: source[key] for key in ("path", "sha256")}
        exec(compile(source["raw"], source["path"], "exec"), module.__dict__)


def install_sources(sources):
    """Reserve the saved namespace only in a fresh child with no ambient helper modules."""
    require(not set(sources).intersection(sys.modules), "another source already occupies the saved helper namespace")
    loader = ExactSavedLoader(sources); sys.meta_path.insert(0, loader)
    for name, source in sources.items():
        lazy = importlib.util.LazyLoader(loader)
        specification = importlib.util.spec_from_loader(name, lazy, origin=source["path"])
        module = importlib.util.module_from_spec(specification); sys.modules[name] = module
        lazy.exec_module(module)
    return loader


def readonly_audit(guard, roots, common_unit, common_plan, event, values):
    """Refuse writes and allow only pinned read-only prerequisite and Git children."""
    if event == "open":
        mode, flags = values[1:3]
        require(not (type(mode) is str and any(character in mode for character in "wax+"))
            and not (type(flags) is int and flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND)),
            "saved metadata readers cannot write files")
    require(event not in {"os.remove", "os.rename", "os.rmdir", "os.mkdir", "os.link", "os.symlink", "os.chmod", "os.chown", "os.truncate"},
        "saved metadata readers cannot mutate filesystem entries")
    if event == "subprocess.Popen":
        command = values[1]
        allowed = ["/usr/bin/python3", "-I", "-S", "-B", str(common_unit / "code/direct_common_confirmation.py"), "--plan-stdin"]
        if command == allowed:
            require(values[0] == allowed[0] and values[2] is None
                and digest(ordinary(allowed[4]).read_bytes()) == common_plan["helpers"]["files"][PREFIX + "direct_common_confirmation.py"],
                "another common prerequisite child cannot replace the saved source")
            guard.require_metadata_environment(values[3])
            return
    guard.metadata_event_guard(event, values, roots)


def execute_saved(payload):
    """Read one prerequisite with its exact namespace under the SAME root metadata origin."""
    require(type(payload) is dict and set(payload) == {"schema_version", "operation", "plan", "plan_sha256",
        "parent", "created_at_ns", "challenge", "metadata_origin"}
        and type(payload["schema_version"]) is int and payload["schema_version"] == 2
        and payload["operation"] in PROTOCOL["operations"] and type(payload["created_at_ns"]) is int
        and payload["created_at_ns"] > 0, "unsupported saved-reader child payload")
    plan = payload["plan"];inputs = contract(plan);runtime = metadata_runtime(plan)
    unsigned = {key: value for key, value in payload.items() if key != "challenge"}
    require(payload["plan_sha256"] == digest(encoded(plan)) and payload["challenge"] == digest(encoded(unsigned))
        and payload["parent"] == process(os.getppid()), "saved child has another actual parent, challenge or plan")
    budget = runtime.Budget(payload["metadata_origin"], plan);context = compute_context(plan)
    with runtime.MetadataScope(plan, budget) as scope:
        with runtime.work_alarm(budget):
            unit = Path(plan["campaign_run_dir"]) / "data" / plan["unit_id"]
            require(Path(__file__).resolve(strict=True) == unit / "code" / NAME,
                "the actual saved-reader child must use its exact own staged namespace")
            profile_path = unit / "code/direct_profile_prerequisites.py"
            profile = source_module("_exact_saved_profile_stage_contract", profile_path,
                plan["helpers"]["files"][PREFIX + profile_path.name])
            profile.require_owned_stage(plan)
            common, common_unit = metadata_inputs(plan)
            guard, environment, bootstrap, policy = common_policy(plan, common, common_unit, scope)
            common_sources = source_closure(common, common_unit, guard, environment, common=True)
            operation = payload["operation"];result_ref = inputs[operation];result = checked(result_ref)
            directory = ordinary(Path(result_ref["path"]).parent, directory=True)
            previous = read(directory / "measurement_plan.json")[0]
            require(directory == Path(plan["campaign_run_dir"]) / "data" / previous["unit_id"]
                and previous["campaign_run_dir"] == plan["campaign_run_dir"], "saved result crossed campaign/unit")
            is_common = operation == "confirmation"
            require((is_common and previous == common) or (not is_common and "common_correctness_inputs" not in previous),
                "saved consumer role differs from actual plan")
            historical_sources = source_closure(previous, directory, guard, environment, common=is_common)
            fact_mode = not is_common and "accepted_evidence_snapshot_protocol" in common
            sources = common_sources if fact_mode else historical_sources
            loader = install_sources(sources)
            if is_common or fact_mode:
                import direct_common_confirmation as common_reader
                common_reader.contract(common)
            import direct_measurement as support
            fact = None
            if fact_mode:
                returned, fact = accepted_history_fact(common, operation, result_ref, previous,
                    common_reader.accepted_snapshot_support(common).consume(common))
            elif operation == "qualification":
                require(Path(result_ref["path"]).name == "direct_qualification_result.json", "wrong qualification result kind")
                returned = support.verify_qualification_inputs(common)
                require(encoded(returned) == encoded(result), "original qualification consumer returned another record")
            else:
                require(Path(result_ref["path"]).name == "result.json", "wrong saved comparison result kind")
                returned = support.verify_saved_measurement(directory, result)
                import post_comparison_validation as validation
                validation.correctness(result);validation.execution_identity(result)
                if operation == "correctness":
                    require(previous.get("mode") == "correctness" and result.get("measurement_valid") is False,
                        "correctness role cannot use timing evidence")
                else:
                    common_reader.validate_pair_protocol(previous, result)
            source_closure(previous, directory, guard, environment, common=is_common)
            source_closure(common, common_unit, guard, environment, common=True)
            require(checked(result_ref) == result and not NUMERICAL.intersection(sys.modules),
                "saved result changed or numerical code entered")
            configuration = guard.finish_repository_configurations()
            active = budget.snapshot();budget.verify(active)
    runtime.validate_scope(scope.proof, budget, scope.observer, scope.owner)
    return {"schema_version": 2, "kind": "exact_saved_direct_profile_prerequisite_read_only_proof", "passed": True,
        "operation": operation, "plan_sha256": payload["plan_sha256"], "challenge": payload["challenge"],
        "parent": payload["parent"], "child": process(os.getpid()), "metadata_compute_context": context,
        "result": result_ref, "saved_plan": reference(directory / "measurement_plan.json"), "saved_helper_closure": previous["helpers"],
        "staged_sources": {name: {key: source[key] for key in ("path", "sha256")} for name, source in sources.items()},
        "declared_historical_sources": {name: {key: source[key] for key in ("path", "sha256")} for name, source in historical_sources.items()},
        "actually_loaded_saved_modules": loader.loaded,
        "execution_source_mode": "reviewed_snapshot_history_fact" if fact_mode else "unchanged_saved_consumer",
        "accepted_history_fact": fact,
        "unchanged_consumer_return_sha256": None if fact_mode else digest(encoded(returned)),
        "accepted_snapshot_return_sha256": digest(encoded(returned)) if fact_mode else None,
        "repository_configuration_validation": configuration, "stdlib_ctypes_bootstrap": bootstrap,
        "common_policy_namespace": policy, "metadata_origin": payload["metadata_origin"],
        "metadata_kernel_scope": scope.proof, "metadata_work_completed": active,
        "numerical_imports": [], "gpu_operations": 0, "scientific_credit": 0, "operational_admission": False}


def accepted_history_fact(common, operation, result_ref, previous, snapshot):
    """Reuse only the genuine independently reviewed new29 Q/C facts; never replay old raw gates."""
    require(operation in ("qualification", "correctness") and "accepted_evidence_snapshot_protocol" in common,
        "reviewed historical reuse cannot replace new confirmation or another protocol")
    snapshot.check(common)
    accepted = snapshot.qual if operation == "qualification" else snapshot.correctness
    result = snapshot.qualification_result if operation == "qualification" else snapshot.correctness_result
    require(accepted["result"] == result_ref and encoded(snapshot.files.json(accepted["plan"])) == encoded(previous)
        and accepted["helper_commit"] == previous["helpers"]["source_commit"] == ORIGINAL_COMMIT,
        "reviewed historical fact changed its exact role, original plan or helper source")
    binding = result["resolved_device_binding"]
    binding_ref = {"path": str(Path(result_ref["path"]).parent / "direct_device_binding.json"), "sha256": digest(encoded(binding))}
    require(encoded(snapshot.files.json(binding_ref)) == encoded(binding)
        and binding["gpu_uuid"] == CARD_ONE, "reviewed historical fact changed its exact accepted physical binding")
    returned = result if operation == "qualification" else [previous, binding]
    fact = {"schema_version": 1, "kind": "exact_reviewed_accepted_profile_history_fact",
        "operation": operation, "snapshot": common["accepted_evidence_snapshot_inputs"], "accepted": accepted,
        "binding": binding_ref, "original_helper_source": previous["helpers"],
        "historical_payloads_rehashed": False, "original_cases_reexecuted": False,
        "old_saved_consumer_reexecuted": False, "new_NUM_or_target_qualification": False,
        "returned_contract_sha256": digest(encoded(returned)), "scientific_credit": 0, "operational_admission": False}
    return returned, fact


def request(plan, operation, budget, scope):
    """Wait on one pinned own child within the same already-started coordinator budget."""
    require(operation in PROTOCOL["operations"], "unsupported saved-reader operation")
    contract(plan);context = compute_context(plan);metadata_inputs(plan)
    require(not NUMERICAL.intersection(sys.modules), "saved readers must precede numerical imports")
    runtime = metadata_runtime(plan);runtime.capability();budget.remaining()
    parent = process(os.getpid())
    expected = ["/usr/bin/python3", "-I", "-S", "-B",
        str(Path(plan["campaign_run_dir"]) / "data" / plan["unit_id"] / "code/direct_profile_prerequisites.py"), "--plan-stdin"]
    require(parent["argv"] == expected, "saved readers require their actual staged profile metadata parent")
    unsigned = {"schema_version": 2, "operation": operation, "plan": plan,
        "plan_sha256": digest(encoded(plan)), "parent": parent, "created_at_ns": time.time_ns(),
        "metadata_origin": budget.origin}
    payload = {**unsigned, "challenge": digest(encoded(unsigned))}
    raw = encoded(payload);require(len(raw) <= INPUT_LIMIT, "saved metadata request exceeds its input bound")
    command = ["/usr/bin/python3", "-I", "-S", "-B", str(ordinary(__file__)), "--read-only-stdin"]
    started = time.time();completed = scope.run(command, input=raw, env=dict(os.environ), maximum_bytes=LIMIT)
    child = completed.metadata_child
    wait = {"command": command, "actual_child": child, "started_at_seconds": started, "finished_at_seconds": time.time(),
        "returncode": completed.returncode, "exact_popen_child_reaped": completed.returncode is not None,
        "pid_handle_reported_exit": scope.observer.handle_exited(scope.handles[child["pid"]]),
        "parent_before": parent, "parent_after": process(os.getpid()),
        "stdout_base64": base64.b64encode(completed.stdout).decode(), "stderr_base64": base64.b64encode(completed.stderr).decode(),
        "stdout_sha256": digest(completed.stdout), "stderr_sha256": digest(completed.stderr), "failure": None,
        "metadata_origin": budget.origin, "metadata_wait": completed.metadata_wait, "metadata_budget": budget.snapshot()}
    require(completed.returncode == 0 and completed.stderr == b"" and wait["pid_handle_reported_exit"]
        and wait["parent_after"] == parent and len(completed.stdout) <= LIMIT and len(completed.stderr) <= LIMIT,
        "exact saved reader failed: " + json.dumps(wait, sort_keys=True))
    proof = document(completed.stdout);validate_proof(plan, operation, proof, wait, payload, context)
    return {"proof": proof, "actual_child_wait": wait, "request": {key: value for key, value in payload.items() if key != "plan"}}


def validate_proof(plan, operation, proof, wait, payload, context):
    """Reject a substituted source, operation, actual child, challenge or nonzero-credit role."""
    inputs = contract(plan)
    validate_process_record(payload["parent"])
    validate_process_record(wait["actual_child"])
    validate_process_record(proof.get("parent"))
    validate_process_record(proof.get("child"))
    require(type(proof.get("schema_version")) is int and proof["schema_version"] == 2
        and proof.get("kind") == "exact_saved_direct_profile_prerequisite_read_only_proof" and proof.get("passed") is True
        and proof.get("operation") == operation and proof.get("plan_sha256") == digest(encoded(plan))
        and proof.get("challenge") == payload["challenge"] and proof.get("parent") == payload["parent"]
        and proof.get("child") == wait["actual_child"] and proof.get("metadata_compute_context") == context
        and proof.get("result") == inputs[operation] and proof.get("numerical_imports") == []
        and type(proof.get("gpu_operations")) is int and proof["gpu_operations"] == 0
        and type(proof.get("scientific_credit")) is int and proof["scientific_credit"] == 0
        and proof.get("operational_admission") is False, "saved reader proof changed its actual source, process, input or role")
    runtime = metadata_runtime(plan)
    budget = runtime.Budget(payload["metadata_origin"], plan, verify_host=False)
    require(proof.get("metadata_origin") == wait.get("metadata_origin") == payload["metadata_origin"],
        "saved metadata reader restarted or changed its single shared origin")
    budget.verify(proof["metadata_work_completed"]);budget.verify(wait["metadata_budget"])
    owner = proof["metadata_kernel_scope"]["baseline"]["owner"]
    observer = runtime.process_observer(plan)
    observer.validate_snapshot(owner, 2618919, proof["child"]["cgroup"])
    require(all(owner[key] == proof["child"][key] for key in ("pid", "parent_pid", "process_start_time", "uids", "cgroup"))
        and owner["command"] == proof["child"]["argv"], "saved metadata scope owner differs from its actual reader child")
    runtime.validate_scope(proof["metadata_kernel_scope"], budget, observer, owner)
    common, common_unit = metadata_inputs(plan)
    policy = proof.get("common_policy_namespace", {})
    guard_path = common_unit / "code/direct_common_confirmation.py"
    guard_sha = common["helpers"]["files"][PREFIX + guard_path.name]
    raw_guard = ordinary(guard_path).read_bytes()
    primitive_names = {"metadata_environment", "require_metadata_environment", "source_inspection_git",
        "metadata_event_guard", "verify_repository_configuration", "finish_repository_configurations"}
    primitive_hashes = {node.name: digest(ast.dump(node, include_attributes=False).encode())
        for node in ast.parse(raw_guard).body if isinstance(node, ast.FunctionDef) and node.name in primitive_names}
    require(digest(raw_guard) == guard_sha and policy.get("guard") == {"path": str(guard_path), "sha256": guard_sha}
        and policy.get("common_helper_closure") == common["helpers"]
        and policy.get("stateless_policy_function_asts") == primitive_hashes
        and set(primitive_hashes) == primitive_names
        and policy.get("new29_contract_executed_in_outer_namespace") is False,
        "saved reader policy did not use the exact common stateless-source interface")
    if "accepted_evidence_snapshot_protocol" in common:
        require(policy.get("history_policy", {}).get("kind") == "new29_reviewed_snapshot_metadata_policy"
            and policy["history_policy"]["snapshot"] == common["accepted_evidence_snapshot_inputs"]
            and policy["history_policy"]["historical_payloads_rehashed"] is False,
            "new29 policy lost its exact reviewed historical fact")
    root_wait = wait.get("metadata_wait", {})
    require(root_wait.get("kind") == "actual_metadata_popen_wait" and root_wait.get("pid") == proof["child"]["pid"]
        and root_wait.get("birth") == proof["child"]["process_start_time"]
        and root_wait.get("command") == wait["command"] and type(root_wait.get("returncode")) is int
        and root_wait["returncode"] == 0 and wait["started_at_seconds"] <= root_wait["finished_at_seconds"]
        <= wait["finished_at_seconds"], "saved exact metadata Popen wait changed")
    previous = checked(proof["saved_plan"])
    require(proof["saved_plan"]["path"] == str(Path(inputs[operation]["path"]).parent / "measurement_plan.json")
        and proof.get("saved_helper_closure") == previous["helpers"], "saved reader proof changed its exact historical plan")
    historical_expected = {Path(relative).stem: {"path": str(Path(inputs[operation]["path"]).parent / "code" / Path(relative).name),
        "sha256": sha} for relative, sha in previous["helpers"]["files"].items()}
    fact_mode = operation != "confirmation" and "accepted_evidence_snapshot_protocol" in common
    source_plan = common if fact_mode else previous
    source_unit = common_unit if fact_mode else Path(inputs[operation]["path"]).parent
    expected = {Path(relative).stem: {"path": str(source_unit / "code" / Path(relative).name), "sha256": sha}
        for relative, sha in source_plan["helpers"]["files"].items()}
    loaded = proof.get("actually_loaded_saved_modules")
    require(proof.get("declared_historical_sources") == historical_expected and proof.get("staged_sources") == expected
        and type(loaded) is dict and "direct_measurement" in loaded and all(expected.get(name) == value for name, value in loaded.items()),
        "saved reader lost its exact actually loaded namespace or declared original helper set")
    if fact_mode:
        snapshot = reviewed_snapshot(common, common_unit)
        returned, fact = accepted_history_fact(common, operation, inputs[operation], previous, snapshot)
        require(proof.get("execution_source_mode") == "reviewed_snapshot_history_fact"
            and encoded(proof.get("accepted_history_fact")) == encoded(fact)
            and proof.get("accepted_snapshot_return_sha256") == digest(encoded(returned))
            and proof.get("unchanged_consumer_return_sha256") is None,
            "new29 historical metadata reuse changed its exact fact or pretended to execute old raw gates")
    else:
        require(proof.get("execution_source_mode") == "unchanged_saved_consumer" and proof.get("accepted_history_fact") is None
            and proof.get("accepted_snapshot_return_sha256") is None
            and type(proof.get("unchanged_consumer_return_sha256")) is str
            and re.fullmatch(r"[0-9a-f]{64}", proof["unchanged_consumer_return_sha256"]),
            "actual unchanged saved consumer proof changed its explicit mode or normal return")
    repository = proof.get("repository_configuration_validation", {})
    require(type(repository) is dict and repository.get("kind") == "same_user_pinned_repository_configuration_and_mutation_detection"
        and type(repository.get("repositories")) is dict and repository["repositories"]
        and repository.get("filters_and_includes_refused") is True
        and repository.get("scope") == "Pinned same-user source/configuration trust and mutation detection; no kernel-enforced file immutability.",
        "saved reader omitted its pinned repository configuration and mutation proof")
    bootstrap = proof.get("stdlib_ctypes_bootstrap", {})
    require(type(bootstrap) is dict and bootstrap.get("python_api_target") is None
        and bootstrap.get("own_process_python_api_initialized_before_validation_audit") is True
        and bootstrap.get("bootstrap_scope") == "stdlib ctypes own-process Python API handle only"
        and bootstrap.get("later_library_loads_and_symbol_calls_refused") is True,
        "saved reader changed its restricted standard-library bootstrap")
    return proof


def validate_bundle(plan, bundle):
    """Recheck every saved actual return without querying a historical process identifier."""
    inputs = contract(plan)
    require(type(bundle) is dict and set(bundle) == set(PROTOCOL["operations"]), "all three separate actual saved readers are required")
    children, origins = set(), set()
    for operation, item in bundle.items():
        require(type(item) is dict and set(item) == {"proof", "actual_child_wait", "request"}, "a saved reader needs its exact request, proof and wait")
        proof, wait, request_record = item["proof"], item["actual_child_wait"], item["request"]
        require(type(request_record) is dict and set(request_record) == {"schema_version", "operation", "plan_sha256", "parent", "created_at_ns", "challenge", "metadata_origin"}
            and request_record["operation"] == operation, "saved reader request metadata changed")
        require(type(request_record["schema_version"]) is int and request_record["schema_version"] == 2,
            "old saved reader schemas cannot bypass the new shared budget")
        payload = {**request_record, "plan": plan}
        origins.add(digest(encoded(payload["metadata_origin"])))
        unsigned = {key: value for key, value in payload.items() if key != "challenge"}
        require(payload["challenge"] == digest(encoded(unsigned)), "saved reader request challenge changed")
        stdout = base64.b64decode(wait["stdout_base64"], validate=True)
        stderr = base64.b64decode(wait["stderr_base64"], validate=True)
        command = ["/usr/bin/python3", "-I", "-S", "-B", str(ordinary(__file__)), "--read-only-stdin"]
        require(wait.get("command") == command and type(wait.get("returncode")) is int and wait["returncode"] == 0
            and wait.get("exact_popen_child_reaped") is True and wait.get("pid_handle_reported_exit") is True
            and wait.get("failure") is None and stderr == b"" and len(stdout) <= LIMIT
            and digest(stdout) == wait.get("stdout_sha256") and digest(stderr) == wait.get("stderr_sha256")
            and document(stdout) == proof and wait["parent_before"] == wait["parent_after"] == request_record["parent"],
            "a saved reader lacks its original normal actual child wait and unchanged captured output")
        parent, child = request_record["parent"], wait["actual_child"]
        require(child["parent_pid"] == parent["pid"] and child["argv"] == command and child["cgroup"] == parent["cgroup"]
            and child["hostname"] == parent["hostname"] and child["uids"] == parent["uids"] == [2618919] * 4,
            "saved metadata child custody differs from its actual parent")
        context = proof["metadata_compute_context"]
        validate_proof(plan, operation, proof, wait, payload, context)
        require(context["hostname"] == parent["hostname"] and context["uids"] == parent["uids"], "saved compute context differs from its actual parent")
        if context["hostname"] == "serval02":
            require(context["job_id"] is None and context["step_id"] is None, "saved direct context cannot claim a scheduler allocation")
        else:
            job, step = context["job_id"], context["step_id"]
            require(type(job) is str and re.fullmatch(r"[1-9][0-9]*", job) and type(step) is str and re.fullmatch(r"[0-9]+", step)
                and parent["cgroup"].splitlines() == ["0::/system.slice/slurmstepd.scope/job_" + job + "/step_" + step + "/user/task_0"],
                "saved actual child context differs from its recorded kernel compute step")
        require(all(type(wait[key]) in {int, float} and math.isfinite(wait[key]) and wait[key] > 0
            for key in ("started_at_seconds", "finished_at_seconds")) and wait["started_at_seconds"] < wait["finished_at_seconds"],
            "saved metadata child interval is incomplete")
        identity = (child["hostname"], child["pid"], child["process_start_time"])
        require(identity not in children, "one metadata child was reused for different saved-reader operations")
        children.add(identity)
        checked(inputs[operation])
    require(len(origins) == 1, "the three saved readers did not share one actual root origin")
    return bundle


def read_all(plan, budget, scope):
    """Run all three exact saved namespaces under the same root origin, never restarting a timer."""
    contract(plan);compute_context(plan);metadata_inputs(plan);budget.remaining()
    bundle = {operation: request(plan, operation, budget, scope) for operation in PROTOCOL["operations"]}
    budget.remaining()
    return validate_bundle(plan, bundle)


class ExactSavedEvidence:
    """Expose only the two already validated records to unchanged combination checks."""

    def __init__(self, plan, bundle):
        """Keep the actual three-reader proof separate from the current profile namespace."""
        self.inputs, self.plan = contract(plan), plan
        self.bundle = validate_bundle(plan, bundle)
        self.resolver = None
        self.roles = {self.inputs[role]["path"]: role for role in ("correctness", "confirmation")}

    def path_identity(self, path):
        """Return only one of the two exact actual saved result paths."""
        require(type(path) in {str, Path} or isinstance(path, Path), "saved result path must be explicit")
        path = str(path)
        require(path in self.roles and str(ordinary(path)) == path, "another result cannot enter the profile combination check")
        return path

    def read(self, path):
        """Read an exact result only after the corresponding unchanged saved reader passed."""
        path = self.path_identity(path)
        return checked(self.inputs[self.roles[path]])

    def require_direct_profile(self, profile):
        """Retain the original exact-profile gate using both completed source-bound readers."""
        require(encoded(profile) == encoded(self.plan["direct_binding"]["profile"]), "the combination profile differs from the actual profile target")
        for role in ("correctness", "confirmation"):
            previous = checked(self.bundle[role]["proof"]["saved_plan"])
            require(encoded(previous["direct_binding"]["profile"]) == encoded(profile), "the saved reader validated another exact physical target profile")

    def verified_direct_plan(self, path, record):
        """Replace only same-namespace dispatch with the exact completed per-record consumer."""
        path = self.path_identity(path); role = self.roles[path]
        require(encoded(record) == encoded(self.read(path)), "the combination supplied another saved record")
        proof = self.bundle[role]["proof"]
        previous = checked(proof["saved_plan"])
        binding = read(Path(path).parent / "direct_device_binding.json")[0]
        returned_sha = (proof["accepted_snapshot_return_sha256"] if proof["execution_source_mode"] == "reviewed_snapshot_history_fact"
            else proof["unchanged_consumer_return_sha256"])
        require(returned_sha == digest(encoded([previous, binding])),
            "the unchanged saved consumer returned another plan or physical binding")
        return previous, binding


def statistic_source_proof(validation_path):
    """Prove that only saved-record dispatch changes in the copied metadata decision function."""
    original_tree = ast.parse(ordinary(validation_path).read_text())
    original = next(node for node in original_tree.body if isinstance(node, ast.FunctionDef) and node.name == "validate_combination")
    current = next(node for node in ast.parse(Path(__file__).read_text()).body
        if isinstance(node, ast.FunctionDef) and node.name == "combine_records_with_verified_readers")
    expected = ast.parse(ast.unparse(original)).body[0]
    expected.name = current.name
    expected.args.args.append(ast.arg(arg="validation"))
    branch = next(node for node in expected.body if isinstance(node, ast.If) and isinstance(node.test, ast.Name) and node.test.id == "direct")
    require(isinstance(branch.body[1], ast.Import) and ast.unparse(branch.body[1]) == "import direct_measurement as support",
        "the original direct saved-reader dispatch layout changed")
    removed = ast.unparse(branch.body.pop(1))
    profile_gate = branch.body[2]
    require(ast.unparse(profile_gate) == "support.core().require_profile(profile)", "the exact original direct-profile gate changed")
    branch.body[2] = ast.parse("evidence.require_direct_profile(profile)").body[0]
    loop = branch.body[3]
    expected_call = ast.parse("plan, unused = support.verify_saved_measurement(Path(path).parent, record)").body[0]
    require(isinstance(loop, ast.For) and ast.dump(loop.body[0], include_attributes=False) == ast.dump(expected_call, include_attributes=False),
        "the exact original saved-record dispatch changed")
    loop.body[0] = ast.parse("plan, unused = evidence.verified_direct_plan(path, record)").body[0]
    class QualifyMetadata(ast.NodeTransformer):
        """Keep original predicates explicit through one unchanged metadata-validator argument."""
        def visit_Name(self, node):
            """Qualify only the enumerated existing metadata predicate references."""
            return ast.copy_location(ast.Attribute(value=ast.Name(id="validation", ctx=ast.Load()), attr=node.id, ctx=node.ctx), node) \
                if isinstance(node.ctx, ast.Load) and node.id in STATISTIC_PRIMITIVES else node
    expected = ast.fix_missing_locations(QualifyMetadata().visit(expected))
    require(ast.dump(expected, include_attributes=False) == ast.dump(current, include_attributes=False),
        "the profile changed an original source, field, statistic or decision check outside its explicit saved-reader dispatch")
    return {"kind": "unchanged_combination_checks_with_exact_saved_reader_dispatch", "original_source": {
        "path": str(validation_path), "sha256": digest(Path(validation_path).read_bytes())},
        "original_function_ast_sha256": digest(ast.dump(original, include_attributes=False).encode()),
        "profile_function_ast_sha256": digest(ast.dump(current, include_attributes=False).encode()),
        "removed_metadata_import": removed, "replaced_metadata_dispatch": [
            {"before": "support.core().require_profile(profile)", "after": "evidence.require_direct_profile(profile)"},
            {"before": "plan, unused = support.verify_saved_measurement(Path(path).parent, record)",
                "after": "plan, unused = evidence.verified_direct_plan(path, record)"}],
        "all_other_statement_syntax_unchanged": True, "tolerances_or_statistics_changed": False}


def combine(plan, bundle):
    """Keep every original combination check after all exact saved-record readers pass."""
    contract(plan); compute_context(plan)
    common, unused = metadata_inputs(plan)
    path = ordinary(Path(__file__).with_name("post_comparison_validation.py"))
    expected = plan["helpers"]["files"][PREFIX + path.name]
    require(expected == common["helpers"]["files"][PREFIX + path.name] == digest(path.read_bytes()),
        "the profile changed its exact accepted combination metadata predicates")
    validation = source_module("saved_profile_combination_metadata_predicates", path, expected)
    source_proof = statistic_source_proof(path)
    evidence = ExactSavedEvidence(plan, bundle)
    value = combine_records_with_verified_readers(plan["accepted_baseline_specification"], evidence, validation)
    require(value["configuration"] == plan["configuration"] and value["paired_rounds"] == 7,
        "the separately validated saved combination differs from the current profile")
    return value, source_proof


def combine_records_with_verified_readers(specification, evidence, validation, *, final=False):
    """Require independent correctness and seven longer paired confirmation rounds for the selected combination."""
    (first_path, second_path) = (specification['correctness_record'], specification['confirmation_record'])
    if evidence.path_identity(first_path) == evidence.path_identity(second_path):
        raise ValueError('combination correctness and confirmation require separate recorded executions')
    (first, second) = (evidence.read(first_path), evidence.read(second_path))
    if validation.execution_identity(first) == validation.execution_identity(second):
        raise ValueError('combination correctness and confirmation reused the same process execution')
    (context, source) = (specification['hardware_class'], specification['source_commit'])
    runtime = validation.runtime_environment(specification['runtime_environment'])
    direct = validation.direct_record(first) or validation.direct_record(second)
    configured = not direct and (validation.configured_record(first) or validation.configured_record(second))
    if direct:
        if final or evidence.resolver is not None or (not all((validation.direct_record(record) for record in (first, second)))):
            raise ValueError('direct final,portable or mixed evidence requires separate explicit support')
        profile = specification.get('direct_binding_profile')
        evidence.require_direct_profile(profile)
        for (path, record) in ((first_path, first), (second_path, second)):
            (plan, unused) = evidence.verified_direct_plan(path, record)
            if plan['direct_binding']['profile'] != profile:
                raise ValueError('the direct confirmation profile differs from its actual qualified process')
    elif configured:
        if final or evidence.resolver is not None:
            raise ValueError('configured-isolation final/portable evidence needs separate explicit support')
        import configured_measurement as support
        profile = specification.get('configured_isolation_profile')
        support.core().require_profile(profile)
        for (path, record) in ((first_path, first), (second_path, second)):
            (plan, unused) = support.verify_saved_measurement(Path(path).parent, record)
            if plan['configured_isolation']['profile'] != profile:
                raise ValueError('the explicit selected profile differs from the actual qualified target')
    for record in (first, second):
        if configured or direct:
            validation._identity_fields(record, context, source, runtime)
        else:
            validation.identity(record, context, source, runtime)
        validation.correctness(record)
    selected = validation.configuration(second['candidate'])
    reference = validation.configuration(second['reference'])
    if validation.configuration(first['reference']) != reference or validation.canonical_digest(reference) == validation.canonical_digest(selected) or reference['learner']['n_copies'] != selected['learner']['n_copies'] or any((record.get('implementation_distinct') is not True for record in (first, second))):
        raise ValueError('combination correctness and confirmation need the same distinct full reference')
    if validation.configuration(first['candidate']) != selected or first.get('copies') != second.get('copies'):
        raise ValueError('the independently checked and confirmed combinations differ')
    copies = selected['learner']['n_copies']
    if second.get('copies') != copies or second.get('measurement_valid') is not True or second.get('contaminated') is not False:
        raise ValueError('combination confirmation is not a valid matched measurement')
    rounds = second.get('rounds', [])
    if final and len(rounds) != 11:
        raise ValueError('final confirmation requires exactly eleven paired rounds')
    if len(rounds) < 7:
        raise ValueError('combination confirmation requires seven paired rounds')
    ratios = []
    for (index, pair) in enumerate(rounds):
        expected_order = ['reference', 'candidate'] if index % 2 == 0 else ['candidate', 'reference']
        if pair.get('order') != expected_order or pair['reference'].get('iterations') != pair['candidate'].get('iterations'):
            raise ValueError('confirmation order or work is not matched')
        for name in ('reference', 'candidate'):
            validation.measured_round(pair[name], copies, minimum_updates=20 if final else 10)
        ratios.append(pair['reference']['seconds'] / pair['candidate']['seconds'])
    wins = sum((ratio > 1 for ratio in ratios))
    probability = sum((math.comb(len(ratios), count) for count in range(wins, len(ratios) + 1))) / 2 ** len(ratios)
    if statistics.median(ratios) <= 1 or probability > 0.05 or second.get('decision') != 'accepted':
        raise ValueError('the selected combination has no repeatable longer-window improvement')
    return {'hardware_class': context, 'source_commit': source, 'configuration': selected, 'runtime_environment': runtime, 'copies': copies, 'paired_rounds': len(rounds), 'median_speedup': statistics.median(ratios), 'one_sided_sign_probability': probability}


def main():
    """Expose only the bounded read-only child entry point, never a producer or launcher."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--read-only-stdin", action="store_true", required=True)
    parser.parse_args()
    raw = sys.stdin.buffer.read(INPUT_LIMIT + 1)
    require(0 < len(raw) <= INPUT_LIMIT, "saved-reader child input exceeds its exact bound")
    result = execute_saved(document(raw))
    output = encoded(result); require(len(output) <= LIMIT, "saved-reader proof exceeds its bound")
    sys.stdout.buffer.write(output); sys.stdout.buffer.flush()


if __name__ == "__main__":
    main()
