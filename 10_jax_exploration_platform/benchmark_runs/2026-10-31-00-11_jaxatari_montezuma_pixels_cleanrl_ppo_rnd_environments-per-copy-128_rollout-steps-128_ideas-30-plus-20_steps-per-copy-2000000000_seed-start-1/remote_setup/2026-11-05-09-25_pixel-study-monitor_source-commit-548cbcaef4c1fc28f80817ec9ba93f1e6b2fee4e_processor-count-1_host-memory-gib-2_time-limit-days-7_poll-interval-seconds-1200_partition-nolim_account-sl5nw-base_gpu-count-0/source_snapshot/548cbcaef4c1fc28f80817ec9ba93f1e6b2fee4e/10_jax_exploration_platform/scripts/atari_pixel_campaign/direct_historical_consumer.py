"""Replay exact old direct consumers in a fresh, read-only metadata process."""
import argparse
import base64
import hashlib
import importlib.abc
import importlib.util
import json
import os
from pathlib import Path
import re
import select
import subprocess
import sys
import time

NAME = "direct_historical_consumer.py"
GUARD = "direct_common_confirmation.py"
PREFIX = "10_jax_exploration_platform/scripts/atari_pixel_campaign/"
OLD_COMMIT = "0a880984b4a03a2ff7335337cc18576628bc9fc1"
NUMERICAL = {"jax", "jaxlib", "jax_cuda12_plugin", "jax_cuda13_plugin", "numpy", "torch", "flax", "optax",
    "scipy", "cupy", "nvidia", "tensorflow", "numba", "ml_dtypes", "jaxtari", "exploration_platform"}
PROTOCOL = {"schema_version": 1, "capability": "exact_old_staged_direct_consumer_read_only_process",
    "old_helper_commit": OLD_COMMIT, "old_helper_count": 26, "current_common_helper_count": 28,
    "digest_cache": "disabled", "operational_admission": False}
LIMIT = 1024 * 1024
DOCUMENT_LIMIT = 64 * 1024 * 1024
WAIT_SECONDS = 900


def require(condition, message):
    """Reject changed bytes, identities or a broader consumer role."""
    if not condition:
        raise ValueError(message)


def encoded(value):
    """Serialize only finite bounded metadata with stable key order."""
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def digest(raw):
    """Identify exact source, input and returned bytes."""
    return hashlib.sha256(raw).hexdigest()


def document(raw):
    """Reject duplicate keys and nonfinite numbers in every process message."""
    def unique(pairs):
        """Keep one exact meaning for each metadata key."""
        result = {}
        for key, value in pairs:
            require(key not in result, "duplicate historical-consumer metadata key")
            result[key] = value
        return result
    return json.loads(raw, object_pairs_hook=unique,
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError("nonfinite historical metadata")))


def ordinary(path):
    """Require a canonical, ordinary file without a symlink replacement path."""
    path = Path(path)
    require(path.is_absolute() and path.resolve(strict=True) == path and path.is_file() and not path.is_symlink()
        and path.stat().st_uid == 2618919,
        "historical metadata needs an exact ordinary file")
    return path


def read(path, maximum=DOCUMENT_LIMIT):
    """Read bounded strict JSON while preserving every duplicate-key and finite-value check."""
    path = ordinary(path)
    require(path.stat().st_size <= maximum, "historical metadata exceeds its declared size bound")
    raw = path.read_bytes()
    value = document(raw)
    return value, raw


def reference(path):
    """Bind a real metadata file's canonical path and exact bytes."""
    unused, raw = read(path)
    return {"path": str(path), "sha256": digest(raw)}


def checked(value):
    """Refuse a substituted plan, actual result or genuine review."""
    require(type(value) is dict and set(value) == {"path", "sha256"} and reference(value["path"]) == value,
        "historical-consumer reference changed")
    return read(value["path"])[0]


def process(pid):
    """Observe only this process or its already established own parent/child."""
    root = Path("/proc") / str(pid)
    raw = (root / "stat").read_text()
    fields = raw.rsplit(")", 1)[1].split()
    uid = next(line for line in (root / "status").read_text().splitlines() if line.startswith("Uid:"))
    values = list(map(int, uid.split()[1:]))
    require(values == [2618919] * 4, "historical metadata process has another owner")
    return {"pid": pid, "parent_pid": int(fields[1]), "birth_counter": fields[19], "all_uids": values,
        "command": (root / "cmdline").read_bytes().rstrip(b"\0").decode().split("\0"),
        "cgroup": (root / "cgroup").read_text()}


def same_process(first, second):
    """Bind stable process identity without comparing changing CPU counters."""
    return first == second


def contract(plan):
    """Allow only the explicit new common-comparison metadata capability."""
    require(plan.get("kind") == "selected_configuration_measurement" and plan.get("mode") == "confirmation"
        and plan.get("hardware_class") == "h100" and plan.get("gpu_uuid") == "GPU-fed69954-5a71-dbb2-ee91-544e0eb6b8a3"
        and plan.get("source_commit") == "d28e6735e79788f618acb0ebfde8609db1f776af"
        and encoded(plan.get("direct_historical_consumer_protocol")) == encoded(PROTOCOL)
        and "common_correctness_inputs" in plan and "direct_qualification_inputs" in plan
        and "direct_binding" in plan and not any(name in plan for name in (
            "direct_qualification", "target_qualification", "configured_isolation", "final_protocol",
            "final_confirmation_protocol", "completed_study", "device_memory_limit_protocol", "scaling_protocol")),
        "historical replay has no old operational, final or broader admission path")
    require(len(plan["helpers"]["files"]) == 28
        and plan["helpers"]["files"].get(PREFIX + NAME) == digest(Path(__file__).read_bytes()),
        "the explicit twenty-eight-helper bridge source must be exact")
    return plan


def current_guard(plan):
    """Load only the current hash-bound metadata guard without its main entry."""
    path = ordinary(Path(__file__).with_name(GUARD))
    raw = path.read_bytes()
    require(plan["helpers"]["files"].get(PREFIX + GUARD) == digest(raw), "current guard source changed")
    specification = importlib.util.spec_from_file_location("exact_current_direct_guard_for_historical_metadata", path)
    module = importlib.util.module_from_spec(specification)
    exec(compile(raw, str(path), "exec"), module.__dict__)
    return module


def current_closure(plan):
    """Check every current helper against staged bytes, source files and its real Git commit."""
    root = Path(plan["helpers"]["source_root"])
    commit = plan["helpers"]["source_commit"]
    environment = current_guard(plan).metadata_environment(os.environ)
    require(root.resolve(strict=True) == root and root.is_dir()
        and type(commit) is str and re.fullmatch(r"[0-9a-f]{40}", commit), "current helper source identity changed")
    for relative, expected in plan["helpers"]["files"].items():
        require(relative.startswith(PREFIX) and Path(relative).as_posix() == relative
            and ".." not in Path(relative).parts, "current helper source path is invalid")
        staged = ordinary(Path(__file__).with_name(Path(relative).name))
        source = ordinary(root / relative)
        committed = subprocess.check_output(["git", "-C", str(root), "show", commit + ":" + relative], env=environment)
        require(digest(staged.read_bytes()) == expected == digest(source.read_bytes()) == digest(committed),
            "the complete current helper closure changed")
    return {"source_root": str(root), "source_commit": commit, "files": dict(plan["helpers"]["files"])}


def selected_records(plan, operation):
    """Bind only the actual qualification or matching prior correctness inputs."""
    require(operation in {"qualification", "correctness"}, "unsupported historical read-only operation")
    inputs = plan["direct_qualification_inputs"] if operation == "qualification" else plan["common_correctness_inputs"]
    require(type(inputs) is dict and set(inputs) == {"result", "review"}, "two exact historical input references are required")
    expected_name = "direct_qualification_result.json" if operation == "qualification" else "result.json"
    campaign = Path(plan["campaign_run_dir"])
    result_path = Path(inputs["result"]["path"])
    review_path = Path(inputs["review"]["path"])
    require(result_path.name == expected_name and result_path.is_relative_to(campaign / "data")
        and review_path.suffix == ".json" and review_path.is_relative_to(campaign / "remote_setup"),
        "another location or result kind cannot replace the historical consumer")
    result = checked(inputs["result"])
    checked(inputs["review"])
    directory = ordinary(result_path).parent
    previous, unused = read(directory / "measurement_plan.json")
    require(directory == Path(plan["campaign_run_dir"]) / "data" / previous["unit_id"]
        and previous["campaign_run_dir"] == plan["campaign_run_dir"]
        and previous["helpers"]["source_commit"] == OLD_COMMIT
        and len(previous["helpers"]["files"]) == 26 and "common_correctness_inputs" not in previous,
        "historical replay requires the exact original twenty-six-helper unit")
    return directory, previous, result, inputs


def old_closure(plan, directory, previous):
    """Pin all old staged and source bytes against their original committed closure."""
    expected = previous["helpers"]["files"]
    environment = current_guard(plan).metadata_environment(os.environ)
    require(set(expected) == set(plan["helpers"]["files"]) - {PREFIX + NAME, PREFIX + GUARD},
        "the complete original helper closure is required")
    root = Path(previous["helpers"]["source_root"])
    require(root.resolve(strict=True) == root and root.is_dir(), "old source root changed")
    sources = {}
    for relative, wanted in expected.items():
        require(relative.startswith(PREFIX) and Path(relative).name not in sources, "old helper basename is ambiguous")
        path = ordinary(directory / "code" / Path(relative).name)
        source = ordinary(root / relative)
        raw = path.read_bytes()
        committed = subprocess.check_output(["git", "-C", str(root), "show", OLD_COMMIT + ":" + relative], env=environment)
        require(digest(raw) == wanted == digest(source.read_bytes()) == digest(committed),
            "the original staged/source/Git helper bytes changed")
        sources[Path(relative).stem] = {"path": str(path), "sha256": wanted, "raw": raw}
    require(len(sources) == 26 and all(name in sources for name in (
        "direct_measurement", "measure_selected", "post_comparison_validation", "train_optimized", "portable_evidence")),
        "the old consumer is missing a real dependency")
    return sources


class ExactOldLoader(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """Execute only the already pinned old source bytes for their original flat module names."""

    def __init__(self, sources):
        """Keep a closed source map and record actual old-module loading."""
        self.sources = sources
        self.loaded = {}

    def find_spec(self, fullname, path=None, target=None):
        """Refuse numerical packages and provide only exact declared old metadata sources."""
        require(fullname.split(".", 1)[0] not in NUMERICAL, "numerical import forbidden in historical metadata")
        if fullname in self.sources:
            return importlib.util.spec_from_loader(fullname, self, origin=self.sources[fullname]["path"])
        return None

    def create_module(self, specification):
        """Use ordinary Python module objects without a runtime substitution."""
        return None

    def exec_module(self, module):
        """Compile the exact old bytes and retain their real staged source filename."""
        source = self.sources[module.__name__]
        require(digest(source["raw"]) == source["sha256"], "closed old source buffer changed")
        require(digest(ordinary(source["path"]).read_bytes()) == source["sha256"],
            "old staged source changed before its actual import")
        module.__file__ = source["path"]
        self.loaded[module.__name__] = {key: source[key] for key in ("path", "sha256")}
        exec(compile(source["raw"], source["path"], "exec"), module.__dict__)


def install_old_modules(sources):
    """Reserve every old name with a closed lazy loader before old adjacent imports can run."""
    require(not set(sources).intersection(sys.modules), "old metadata modules were already replaced or loaded")
    loader = ExactOldLoader(sources)
    sys.meta_path.insert(0, loader)
    # Before: no old helper names. After: exact old bytes, loaded only if actually used.
    for name, source in sources.items():
        lazy = importlib.util.LazyLoader(loader)
        specification = importlib.util.spec_from_loader(name, lazy, origin=source["path"])
        module = importlib.util.module_from_spec(specification)
        sys.modules[name] = module
        lazy.exec_module(module)
    return loader


def readonly_audit(guard, roots, event, values):
    """Keep historical replay read-only and allow only the existing exact local Git commands."""
    if event == "import":
        require(values[0].split(".", 1)[0] not in NUMERICAL, "numerical or GPU-package import forbidden")
    if event == "open":
        mode = values[1] if len(values) > 1 else None
        flags = values[2] if len(values) > 2 else 0
        require(not (isinstance(mode, str) and any(letter in mode for letter in "wa+"))
            and not (isinstance(flags, int) and flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND)),
            "historical consumer cannot write files")
    require(event not in {"os.remove", "os.rename", "os.mkdir", "os.rmdir", "os.chmod", "os.chown", "os.utime",
        "os.link", "os.symlink", "os.truncate"},
        "historical consumer cannot change filesystem entries")
    require(event not in {"os.exec", "os.posix_spawn", "os.spawn", "os.fork", "pty.spawn"},
        "historical consumer cannot execute an unlisted process")
    if event == "subprocess.Popen":
        command = list(values[1]) if isinstance(values[1], (list, tuple)) else []
        require(command and Path(command[0]).name == "git", "historical consumer cannot start Python or a launcher")
    guard.metadata_event_guard(event, values, roots)


def child(payload):
    """Replay one exact old reader in an isolated namespace on real owned compute."""
    require(type(payload) is dict and set(payload) == {"schema_version", "operation", "plan", "plan_sha256", "parent", "created_at_ns", "challenge"}
        and type(payload["schema_version"]) is int and payload["schema_version"] == 1
        and type(payload["created_at_ns"]) is int and payload["created_at_ns"] > 0,
        "unsupported historical child request")
    plan = contract(payload["plan"])
    require(payload["plan_sha256"] == digest(encoded(plan)) and payload["parent"]["pid"] == os.getppid(),
        "historical child input or actual parent changed")
    unsigned = {key: value for key, value in payload.items() if key != "challenge"}
    require(payload["challenge"] == digest(encoded(unsigned)) and same_process(process(os.getppid()), payload["parent"]),
        "historical child challenge or real parent identity changed")
    guard = current_guard(plan)
    environment = guard.metadata_environment(os.environ)
    os.environ.clear()
    os.environ.update(environment)
    context = guard.metadata_compute_context(plan)
    guard.contract(plan)
    roots = guard.declared_git_roots(plan)
    bootstrap = guard.stdlib_ctypes_bootstrap()
    guard.validate_bootstrap_record(bootstrap)
    require(not NUMERICAL.intersection(sys.modules), "historical child already imported numerical code")
    sys.addaudithook(lambda event, values: readonly_audit(guard, roots, event, values))
    current = current_closure(plan)
    directory, previous, result, inputs = selected_records(plan, payload["operation"])
    sources = old_closure(plan, directory, previous)
    loader = install_old_modules(sources)
    import direct_measurement as support
    if payload["operation"] == "qualification":
        returned = support.verify_qualification_inputs(plan)
    else:
        import post_comparison_validation as validation
        returned = support.verify_saved_measurement(directory, result)
        validation.correctness(result)
        validation.execution_identity(result)
    require(not NUMERICAL.intersection(sys.modules), "an old consumer imported numerical code")
    old_closure(plan, directory, previous)
    current_closure(plan)
    require(checked(inputs["result"]) == result and checked(inputs["review"]) is not None,
        "old result or review changed during replay")
    repository_configuration = guard.finish_repository_configurations()
    return {"schema_version": 1, "kind": "exact_old_direct_consumer_read_only_proof", "passed": True,
        "operation": payload["operation"], "plan_sha256": payload["plan_sha256"], "challenge": payload["challenge"],
        "parent": payload["parent"], "child": process(os.getpid()), "metadata_compute_context": context,
        "old_inputs": inputs, "old_plan": reference(directory / "measurement_plan.json"),
        "current_helper_closure": current, "old_helper_commit": OLD_COMMIT,
        "old_staged_files": {name: {key: source[key] for key in ("path", "sha256")}
            for name, source in sources.items()}, "actually_loaded_old_modules": loader.loaded,
        "old_consumer_returned_normally": True, "old_consumer_return_sha256": digest(encoded(returned)),
        "repository_configuration_validation": repository_configuration,
        "stdlib_ctypes_bootstrap": bootstrap, "later_ctypes_load_or_symbol_events": 0,
        "native_standard_library_extension_activity": "not counted as GPU or ctypes operations", "numerical_imports": [],
        "gpu_operations": 0, "scientific_credit": 0, "admission_granted": False, "operational_admission": False}


def request(plan, operation):
    """Bind one real own child, original bytes and a normal wait before accepting its old-reader proof."""
    plan = document(encoded(plan))
    contract(plan)
    guard = current_guard(plan)
    guard.metadata_compute_context(plan)
    guard.contract(plan)
    current_closure(plan)
    require(not NUMERICAL.intersection(sys.modules), "old readers must run before numerical initialization")
    directory, previous, unused_result, old_inputs = selected_records(plan, operation)
    old_plan = reference(directory / "measurement_plan.json")
    old_sources = {Path(relative).stem: {"path": str(directory / "code" / Path(relative).name), "sha256": expected}
        for relative, expected in previous["helpers"]["files"].items()}
    parent = process(os.getpid())
    unsigned = {"schema_version": 1, "operation": operation, "plan": plan,
        "plan_sha256": digest(encoded(plan)), "parent": parent, "created_at_ns": time.time_ns()}
    payload = {**unsigned, "challenge": digest(encoded(unsigned))}
    raw = encoded(payload)
    require(len(raw) <= LIMIT, "historical metadata request exceeds its bound")
    command = ["/usr/bin/python3", "-I", "-S", "-B", str(ordinary(__file__)), "--read-only-stdin"]
    environment = guard.metadata_environment(os.environ)
    started = time.time()
    spawned = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=environment)
    handle = None
    observed = None
    error = None
    timeout = False
    try:
        handle = os.pidfd_open(spawned.pid)
        observed = process(spawned.pid)
        require(observed["parent_pid"] == os.getpid() and observed["command"] == command,
            "historical consumer actual child identity changed")
        stdout, stderr = spawned.communicate(input=raw, timeout=WAIT_SECONDS)
    except BaseException as failure:
        error = failure
        timeout = isinstance(failure, subprocess.TimeoutExpired)
        if spawned.poll() is None:
            spawned.kill()
        stdout, stderr = spawned.communicate()
    finally:
        ended = time.time()
        ended_handle = handle is not None and bool(select.select([handle], [], [], 0)[0])
        if handle is not None:
            os.close(handle)
    wait = {"command": command, "actual_child": observed, "actual_pid": spawned.pid,
        "parent_before": parent, "parent_after": process(os.getpid()), "started_at_seconds": started,
        "finished_at_seconds": ended, "returncode": spawned.returncode, "timeout": timeout,
        "exact_popen_child_reaped": spawned.returncode is not None, "pid_handle_reported_exit": ended_handle,
        "stdout_sha256": digest(stdout), "stderr_sha256": digest(stderr),
        "stdout_base64": base64.b64encode(stdout).decode(), "stderr_base64": base64.b64encode(stderr).decode(),
        "failure": None if error is None else {"type": type(error).__name__, "message": str(error)}}
    require(len(stdout) <= LIMIT and len(stderr) <= LIMIT, "historical consumer output exceeded its bound")
    require(error is None and spawned.returncode == 0 and not stderr and ended_handle
        and same_process(parent, wait["parent_after"]), "historical consumer failed: " + json.dumps(wait, sort_keys=True))
    proof = document(stdout)
    validate_returned_proof(proof, payload, observed, old_inputs, old_plan, old_sources, plan)
    guard.validate_bootstrap_record(proof.get("stdlib_ctypes_bootstrap"))
    current_closure(plan)
    selected_records(plan, operation)
    return {"proof": proof, "actual_child_wait": wait}


def validate_returned_proof(proof, payload, observed, old_inputs, old_plan, old_sources, plan):
    """Reject any substituted child, challenge, input closure or zero-credit role in a returned proof."""
    repository = proof.get("repository_configuration_validation", {})
    require(type(proof.get("schema_version")) is int and proof["schema_version"] == 1
        and proof.get("kind") == "exact_old_direct_consumer_read_only_proof"
        and proof.get("passed") is True and proof.get("operation") == payload["operation"]
        and proof.get("plan_sha256") == payload["plan_sha256"] and proof.get("challenge") == payload["challenge"]
        and proof.get("parent") == payload["parent"] and proof.get("child") == observed
        and proof.get("old_helper_commit") == OLD_COMMIT and proof.get("old_consumer_returned_normally") is True
        and proof.get("old_inputs") == old_inputs and proof.get("old_plan") == old_plan
        and proof.get("old_staged_files") == old_sources and proof.get("current_helper_closure") == plan["helpers"]
        and type(proof.get("old_consumer_return_sha256")) is str
        and re.fullmatch(r"[0-9a-f]{64}", proof["old_consumer_return_sha256"])
        and type(proof.get("actually_loaded_old_modules")) is dict
        and "direct_measurement" in proof["actually_loaded_old_modules"]
        and all(old_sources.get(name) == value for name, value in proof["actually_loaded_old_modules"].items())
        and proof.get("numerical_imports") == [] and type(proof.get("gpu_operations")) is int and proof["gpu_operations"] == 0
        and type(proof.get("scientific_credit")) is int and proof["scientific_credit"] == 0
        and type(proof.get("later_ctypes_load_or_symbol_events")) is int and proof["later_ctypes_load_or_symbol_events"] == 0
        and proof.get("admission_granted") is False
        and proof.get("operational_admission") is False,
        "historical consumer proof substituted a child, input, source or role")
    require(type(repository) is dict
        and repository.get("kind") == "same_user_pinned_repository_configuration_and_mutation_detection"
        and type(repository.get("repositories")) is dict and repository["repositories"]
        and repository.get("filters_and_includes_refused") is True
        and repository.get("scope") == "Pinned same-user source/configuration trust and mutation detection; no kernel-enforced file immutability.",
        "historical consumer omitted its exact pinned repository configuration proof")
    return proof


def main():
    """Expose only a bounded historical read, never a producer or launcher entry."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--read-only-stdin", required=True, action="store_true")
    parser.parse_args()
    raw = sys.stdin.buffer.read(LIMIT + 1)
    require(len(raw) <= LIMIT, "historical child request exceeds its bound")
    payload = document(raw)
    result = encoded(child(payload))
    require(len(result) <= LIMIT, "historical child proof exceeds its bound")
    sys.stdout.buffer.write(result)


if __name__ == "__main__":
    main()
