"""Train the validated final pixel learner and resume its complete state exactly.

Admission and checkpoint metadata use only the standard library. Numerical
imports happen only after evidence, source, runtime, and output checks pass.
No training is authorized by the existence of this script alone.

Both the numerical source and these helpers require actual frozen Git
checkouts; an archive without Git metadata is insufficient. Every raw evidence
path must be accessible on the training server through explicitly staged and
revalidated receipts. asset_record names an exhaustive asset manifest outside
its asset root, with the manifest's SHA256 fixed in the final specification.
Wheel-only remote installs also supply jaxtari_source_root: a staged checkout
of the pinned public JAXtari commit used to verify installed Python files.
"""

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib
import importlib.util
from importlib import metadata
import json
import math
import os
from pathlib import Path
import pickle
import shlex
import signal
import socket
import stat
import subprocess
import sys
import time
from urllib.parse import unquote, urlparse


SCRIPT_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_ROOT.parents[2]
REQUIRED_RUNTIME = ("JAX_ENABLE_X64", "JAX_DEFAULT_MATMUL_PRECISION", "XLA_FLAGS")
STATE_FIELDS = {"environment", "observations", "done", "parameters", "target", "optimizer",
                "observation_statistics", "reward_statistics", "reward_filter", "iteration", "copy_seeds"}
PACKAGE_NAMES = ("jax", "jaxlib", "jax-cuda12-plugin", "jax-cuda12-pjrt",
                 "nvidia-cudnn-cu12", "nvidia-cublas-cu12", "nvidia-cuda-runtime-cu12",
                 "jaxtari", "flax", "optax", "numpy")
JAXTARI_COMMIT = "04ab9aec318c72162e2e53503a594a489d7e162b"
SERVAL02_CARD1 = "GPU-fed69954-5a71-dbb2-ee91-544e0eb6b8a3"
SERVAL02_STANDING_LOCK = Path("/tmp/sl5nw_serval02_card1.lock")


def utc_now():
    """Write timestamps in a host-independent, timezone-aware form."""
    return datetime.now(timezone.utc).isoformat()


def canonical_digest(value):
    """Bind identity to values rather than JSON whitespace or key order."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def component_source_commit(root, environment):
    """Identify a component-test archive or exact Git root without searching an ancestor repo."""
    root = Path(root).resolve(strict=True)
    supplied = environment.get("RND_COMPONENT_SOURCE_COMMIT")
    if supplied is not None:
        if (not isinstance(supplied, str) or len(supplied) != 40
                or any(character not in "0123456789abcdef" for character in supplied)
                or root.name != supplied):
            raise ValueError("the component archive requires its full commit and matching snapshot directory")
        return supplied
    top = subprocess.check_output(["git", "-C", str(root), "rev-parse", "--show-toplevel"], text=True).strip()
    if Path(top).resolve() != root:
        raise ValueError("the component fixture cannot use an ancestor repository's Git identity")
    commit = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    if len(commit) != 40 or any(character not in "0123456789abcdef" for character in commit):
        raise ValueError("the component Git checkout must record a full commit")
    return commit


def training_placement(environment, hostname):
    """Require a Slurm compute allocation or the explicitly authorized serval02 card."""
    host = hostname.split(".")[0]
    if host == "serval05" or any(word in host for word in ("portal", "login")):
        raise RuntimeError("final training cannot run on serval05 or a login host")
    if host == "serval02":
        if environment.get("CUDA_VISIBLE_DEVICES") != SERVAL02_CARD1:
            raise ValueError("direct training requires the exact authorized serval02 card 1 UUID")
        return {"kind": "serval02", "gpu_uuid": SERVAL02_CARD1, "standing_lock": str(SERVAL02_STANDING_LOCK)}
    job = environment.get("SLURM_JOB_ID", "")
    if not job.isdecimal() or int(job) < 1:
        raise RuntimeError("jaguar03 and other training hosts require a Slurm compute allocation")
    return {"kind": "slurm", "slurm_job_id": job}


@contextmanager
def placement_guard(placement):
    """Hold the direct host's standing lock and verify GPU modes before numerical imports."""
    if placement["kind"] == "slurm":
        yield
        return
    from queue_worker import gpu_identity, gpu_readiness, standing_case_lock
    with standing_case_lock(Path(placement["standing_lock"])) as (acquired, unused):
        if not acquired:
            raise RuntimeError("serval02 card 1's standing lock is already held")
        gpu_identity(placement["gpu_uuid"], "h100")
        readiness = gpu_readiness(placement["gpu_uuid"], "h100")
        if readiness.get("ready") is not True:
            raise RuntimeError("serval02 card 1 is not ready: " + json.dumps(readiness, sort_keys=True))
        yield


def atomic_json(path, value):
    """Publish a complete, flushed metadata record through one rename."""
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    with temporary.open("w") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def append_record(path, value):
    """Flush one complete record before the next execution group begins."""
    with path.open("a") as stream:
        stream.write(json.dumps(value, allow_nan=False) + "\n")
        stream.flush()


def validate_specification(specification):
    """Reject incomplete identities and altered final budgets before numerical imports."""
    # Full configuration/evidence checks belong to validate_training_admission.
    if specification.get("schema_version") != 1:
        raise ValueError("training specification schema_version must be one")
    required = {"source_root", "source_commit", "hardware_class", "configuration",
                "copy_seeds", "runtime_environment", "final_verification", "validation", "asset_record"}
    if not required <= specification.keys():
        raise ValueError("training specification is missing its validated identity")
    configuration = specification["configuration"]
    if set(configuration) != {"learner", "environment", "execution"}:
        raise ValueError("training needs full learner, environment, and execution settings")
    learner = configuration["learner"]
    expected = {"total_timesteps": 2_000_000_000, "n_envs": 128, "num_steps": 128,
                "update_epochs": 4, "num_minibatches": 4}
    if any(type(learner.get(name)) is not int or learner[name] != value
           for name, value in expected.items()):
        raise ValueError("final training must retain the 2B/128/128/4/4 learner contract")
    seeds = specification["copy_seeds"]
    if (type(learner.get("n_copies")) is not int or learner["n_copies"] < 1
            or not isinstance(seeds, list) or len(seeds) != learner["n_copies"]
            or any(type(seed) is not int or not 0 <= seed < 2 ** 32 for seed in seeds)
            or len(set(seeds)) != len(seeds)):
        raise ValueError("copy_seeds must contain one distinct unsigned 32-bit seed per policy")
    if any(learner.get(name) for name in ("per_copy_learning_rates", "per_copy_intrinsic_coefficients",
                                         "per_copy_scalar_settings")):
        raise ValueError("the final infrastructure run does not search per-copy hyperparameters")
    if configuration["environment"].get("num_envs") != 128:
        raise ValueError("the full environment must retain 128 environments per policy")
    execution = configuration["execution"]
    if set(execution) != {"compilation_mode", "iterations_per_call", "report_readout", "report_every_iterations"}:
        raise ValueError("execution must contain exactly the four Executor settings")
    if any(type(execution[name]) is not int or execution[name] < 1
           for name in ("iterations_per_call", "report_every_iterations")):
        raise ValueError("execution groups and report intervals must be positive integers")
    if execution["compilation_mode"] not in {"whole_iteration", "separate_phases"}:
        raise ValueError("unsupported compilation mode")
    if execution["report_readout"] not in {"none", "per_update", "scheduled"}:
        raise ValueError("unsupported report readout")
    if execution["iterations_per_call"] != 1 and (
            execution["compilation_mode"] == "separate_phases" or execution["report_readout"] != "none"):
        raise ValueError("grouping cannot change separate compilation or reporting boundaries")
    return specification


def enforce_runtime(required, environment):
    """Apply the validated precision/flags and refuse conflicting explicit settings."""
    if set(required) != set(REQUIRED_RUNTIME):
        raise ValueError("runtime admission must include exactly the three required JAX/XLA settings")
    if required["JAX_ENABLE_X64"] != "true" or required["JAX_DEFAULT_MATMUL_PRECISION"] != "highest":
        raise ValueError("final training requires float64 statistics and highest float32 precision")
    flags = shlex.split(required["XLA_FLAGS"])
    deterministic = [flag for flag in flags if flag.startswith("--xla_gpu_exclude_nondeterministic_ops=")]
    if deterministic != ["--xla_gpu_exclude_nondeterministic_ops=true"]:
        raise ValueError("the validated runtime must enable deterministic XLA operations exactly once")
    if any("cache" in flag or "autotune_results" in flag for flag in flags):
        raise ValueError("final training has no admission for imported compiler cache artifacts")
    for name, value in environment.items():
        if name.startswith(("JAX_COMPILATION_CACHE", "JAX_PERSISTENT_CACHE")) and value:
            raise ValueError("final training cannot inherit an unvalidated persistent cache setting: " + name)
    if environment.get("JAX_ENABLE_COMPILATION_CACHE", "false") != "false":
        raise ValueError("final training requires persistent compilation caches to be disabled")
    environment["JAX_ENABLE_COMPILATION_CACHE"] = "false"
    environment["JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES"] = ""
    normalized = {**required, "XLA_FLAGS": shlex.join(flags)}
    for name, value in normalized.items():
        actual = environment.get(name)
        if name == "XLA_FLAGS" and actual is not None:
            actual = shlex.join(shlex.split(actual))
        if actual is not None and actual != value:
            raise ValueError(f"runtime setting {name} differs from its validated value")
        environment[name] = value
    return {name: value for name, value in sorted(environment.items())
            if name.startswith(("JAX_", "JAXTARI_", "XLA_", "CUDA_", "CUDNN_", "NVIDIA_"))
            and name != "CUDA_VISIBLE_DEVICES"}


def source_receipt(root, commit, relative_paths=None):
    """Require an exact Git snapshot and bind every explicitly used helper file."""
    root = Path(root).resolve(strict=True)
    actual = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    if actual != commit or len(commit) != 40:
        raise ValueError("training source is not at the exact admitted commit")
    paths = list(relative_paths or [])
    changes = subprocess.check_output(
        ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=all", "--", *paths], text=True)
    if changes.strip():
        raise ValueError("training source has uncommitted runtime changes: " + changes.strip())
    files = {}
    for relative in paths:
        original = subprocess.check_output(["git", "-C", str(root), "show", f"{commit}:{relative}"])
        current = (root / relative).read_bytes()
        if current != original:
            raise ValueError("a runtime helper differs from its Git snapshot")
        files[relative] = hashlib.sha256(current).hexdigest()
    return {"source_root": str(root), "source_commit": commit, "files": files}


def file_sha256(path):
    """Hash large checkpoints and asset files without retaining their bytes in memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def jaxtari_python_receipt(source_root=None):
    """Check installed JAXtari Python bytes against the pinned Git tree without importing it."""
    distribution = metadata.distribution("jaxtari")
    direct = json.loads(distribution.read_text("direct_url.json") or "{}")
    if source_root is None:
        url = urlparse(direct.get("url", ""))
        candidate = Path(unquote(url.path)) if url.scheme == "file" else None
        if candidate is not None and candidate.is_dir():
            source_root = candidate
    if source_root is None or not Path(source_root).is_absolute():
        raise ValueError("stage the pinned JAXtari Git checkout and supply jaxtari_source_root")
    root = Path(source_root).resolve(strict=True)
    top = subprocess.check_output(["git", "-C", str(root), "rev-parse", "--show-toplevel"], text=True).strip()
    commit = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    if Path(top).resolve() != root or commit != JAXTARI_COMMIT:
        raise ValueError("the JAXtari source is not the exact pinned Git checkout")
    entries = subprocess.check_output(["git", "-C", str(root), "ls-tree", "-rz", commit, "--", "src/jaxtari"])
    expected = {}
    for entry in entries.split(b"\0"):
        if entry:
            header, name = entry.split(b"\t", 1)
            path = name.decode()
            if path.endswith(".py"):
                expected[path.removeprefix("src/")] = header.decode().split()[2]
    installed = {str(path): distribution.locate_file(path) for path in distribution.files or []
                 if str(path).startswith("jaxtari/") and str(path).endswith(".py")}
    if not expected or set(expected) != set(installed):
        raise ValueError("installed JAXtari Python files differ from the pinned source file set")
    files = {}
    for name, path in installed.items():
        contents = Path(path).read_bytes()
        blob = hashlib.sha1(b"blob " + str(len(contents)).encode() + b"\0" + contents).hexdigest()
        if blob != expected[name]:
            raise ValueError("installed JAXtari Python differs from the pinned source: " + name)
        files[name] = {"git_blob": blob, "sha256": hashlib.sha256(contents).hexdigest()}
    return {"source_root": str(root), "source_commit": commit, "version": distribution.version,
            "direct_url": direct, "python_files": files, "tree_digest": canonical_digest(files)}


def installed_gpu_library_receipts():
    """Hash the installed cuDNN and CUDA runtime images before any training update."""
    files = {}
    for name in ("nvidia-cudnn-cu12", "nvidia-cuda-runtime-cu12"):
        distribution = metadata.distribution(name)
        for relative in distribution.files or []:
            basename = Path(relative).name
            if basename.startswith(("libcudnn", "libcudart")) and ".so" in basename:
                path = Path(distribution.locate_file(relative)).resolve(strict=True)
                files[str(path)] = {"package": name, "version": distribution.version, "sha256": file_sha256(path)}
    if not any(Path(path).name.startswith("libcudnn.so") for path in files) or not any(
            Path(path).name.startswith("libcudart.so") for path in files):
        raise ValueError("the installed environment lacks its cuDNN/CUDA runtime images")
    return files


def actual_gpu_library_receipt(backend, expected):
    """Query already loaded libraries and verify their actual bytes against checkpoint identity."""
    # Import this identity-bound helper explicitly, regardless of the numerical source's search path.
    specification = importlib.util.spec_from_file_location("optimized_training_gpu_libraries",
                                                          SCRIPT_ROOT / "diagnose_gpu_execution.py")
    helper = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(helper)
    receipt = helper.loaded_gpu_libraries(backend)
    loaded = {str(Path(path).resolve(strict=True)) for path in receipt["loaded_shared_libraries"]
              if Path(path).name.startswith(("libcudnn", "libcudart"))}
    verified = {}
    for path in loaded:
        if path not in expected or file_sha256(Path(path)) != expected[path]["sha256"]:
            raise ValueError("the actually loaded GPU library differs from the installed immutable image: " + path)
        verified[path] = expected[path]
    return {**receipt, "verified_package_libraries": verified, "pid": os.getpid(), "time": utc_now()}


def validate_gpu_library_receipt(receipt, backend):
    """Validate checkpoint library receipts without importing array modules or loading a library."""
    expected = backend.get("installed_gpu_libraries")
    if expected is None:
        return
    if (not isinstance(receipt, dict) or type(receipt.get("cudnnGetVersion")) is not int
            or type(receipt.get("cudaRuntimeGetVersion")) is not int
            or min(receipt["cudnnGetVersion"], receipt["cudaRuntimeGetVersion"]) <= 0):
        raise ValueError("the trained checkpoint lacks verified actual CUDA/cuDNN libraries")
    verified = receipt.get("verified_package_libraries", {})
    if (any(path not in expected or value != expected[path] for path, value in verified.items())
            or any(str(Path(receipt[name]).resolve()) not in verified
                   for name in ("cudnn_library", "cuda_runtime_library"))):
        raise ValueError("checkpoint loaded-library receipts differ from their immutable images")


def verify_assets(record, environment):
    """Check every asset against a fixed manifest before any game initialization."""
    if set(record) != {"root", "manifest", "sha256"}:
        raise ValueError("asset_record requires root, manifest, and its SHA256")
    root, manifest_path = Path(record["root"]), Path(record["manifest"])
    if not root.is_absolute() or not manifest_path.is_absolute():
        raise ValueError("asset paths must be absolute")
    root, manifest_path = root.resolve(strict=True), manifest_path.resolve(strict=True)
    if not root.is_dir() or manifest_path.is_relative_to(root) or file_sha256(manifest_path) != record["sha256"]:
        raise ValueError("asset manifest must match its hash and remain outside the asset tree")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema_version") != 1 or manifest.get("source_commit") != JAXTARI_COMMIT:
        raise ValueError("asset manifest must name the pinned JAXtari source")
    files = manifest["files"]
    actual = {str(path.relative_to(root)): path for path in root.rglob("*") if path.is_file()}
    if not isinstance(files, dict) or not files or set(files) != set(actual):
        raise ValueError("asset manifest does not cover exactly the available asset files")
    for name, path in actual.items():
        if not path.resolve().is_relative_to(root) or file_sha256(path) != files[name]:
            raise ValueError("asset file differs from its immutable manifest: " + name)
    configured = environment.get("XDG_DATA_HOME")
    if configured is not None and Path(configured).resolve() != root:
        raise ValueError("XDG_DATA_HOME differs from the immutable asset root")
    environment["XDG_DATA_HOME"] = str(root)
    return {"root": str(root), "manifest_sha256": record["sha256"], "source_commit": JAXTARI_COMMIT,
            "file_count": len(files), "tree_digest": canonical_digest(files)}


def checkpoint_identity(specification, helpers, runtime_environment, backend, assets):
    """Keep resume identity independent of changing report or registry receipts."""
    return {"schema_version": 1, "source_root": specification["source_root"],
            "source_commit": specification["source_commit"], "hardware_class": specification["hardware_class"],
            "configuration": specification["configuration"], "copy_seeds": specification["copy_seeds"],
            "runtime_environment": runtime_environment, "backend": backend, "helpers": helpers, "assets": assets}


def validate_host_state(state, identity, iteration):
    """Check true schedule position and seed ownership without selecting state fields."""
    if not STATE_FIELDS <= set(getattr(state, "_fields", ())):
        raise ValueError("checkpoint must contain the complete PixelTrainState")
    if int(state.iteration) != iteration:
        raise ValueError("checkpoint iteration does not match the actual learner state")
    if [int(seed) for seed in state.copy_seeds] != identity["copy_seeds"]:
        raise ValueError("checkpoint state seeds do not match the training identity")
    learner = identity["configuration"]["learner"]
    target = learner["total_timesteps"] // (learner["n_envs"] * learner["num_steps"])
    if type(iteration) is not int or not 0 <= iteration <= target:
        raise ValueError("checkpoint iteration is outside the original training schedule")


def validate_report_state(report, execution, iteration):
    """Keep unfinished reporting windows attached to their original update sequence."""
    if set(report) != {"pending_sums", "pending_count", "last_report_iteration"}:
        raise ValueError("checkpoint report state is incomplete")
    count, last = report["pending_count"], report["last_report_iteration"]
    if type(count) is not int or type(last) is not int or not 0 <= last <= iteration:
        raise ValueError("checkpoint reporting counters are invalid")
    if execution["report_readout"] == "none":
        if count != 0 or report["pending_sums"] is not None or last != 0:
            raise ValueError("unreported execution cannot contain a reporting window")
    elif (not 0 <= count < execution["report_every_iterations"] or last + count != iteration
          or (report["pending_sums"] is None) != (count == 0)):
        raise ValueError("checkpoint reporting window disagrees with the true iteration")


def checkpoint_paths(path, create=False):
    """Bound every transaction file to one real directory and the reserved checkpoint names."""
    path = Path(path).absolute()
    if path.name != "latest.pkl" or path != path.resolve():
        raise ValueError("checkpoint requires an exact latest.pkl path without symlink ancestors")
    if create:
        missing, parent = [], path.parent
        while not parent.exists():
            missing.append(parent)
            parent = parent.parent
        path.parent.mkdir(parents=True, exist_ok=True)
        # Persist new directory entries as well as the checkpoint's own later file entries.
        for created in reversed(missing):
            descriptor = os.open(created.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
    directory = path.parent
    if (directory.resolve(strict=True) != directory or not directory.is_dir()
            or directory.stat().st_uid != os.getuid()):
        raise ValueError("checkpoint directory must be an existing real local directory")
    names = {"payload": "latest.pkl", "metadata": "latest.metadata.json", "lock": "checkpoint.lock",
             "marker": "latest.transaction.json", "marker_next": "latest.transaction.next.json",
             "previous_payload": "latest.previous.pkl", "previous_metadata": "latest.previous.metadata.json",
             "next_payload": "latest.next.pkl", "next_metadata": "latest.next.metadata.json"}
    paths = {name: directory / basename for name, basename in names.items()}
    for reserved in paths.values():
        if reserved.is_symlink():
            raise ValueError("checkpoint transaction paths cannot be symlinks")
        if reserved.exists():
            observed = reserved.stat()
            if not stat.S_ISREG(observed.st_mode) or observed.st_uid != os.getuid():
                raise ValueError("checkpoint transaction paths must be regular files owned by this user")
    return paths


@contextmanager
def checkpoint_lock(path, create=False):
    """Serialize save, restore, and metadata recovery without waiting for an active writer."""
    paths = checkpoint_paths(path, create=create)
    descriptor = os.open(paths["lock"], os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    with os.fdopen(descriptor, "a+") as stream:
        observed = os.fstat(stream.fileno())
        if not stat.S_ISREG(observed.st_mode) or observed.st_uid != os.getuid():
            raise ValueError("checkpoint lock must be a regular file owned by this user")
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        # Recheck paths after acquiring the same lock used by every checkpoint operation.
        yield checkpoint_paths(path)


def checkpoint_sync_directory(paths):
    """Persist renames and removals before advancing or discarding a transaction record."""
    descriptor = os.open(paths["payload"].parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def checkpoint_write(path, writer):
    """Create one reserved staged file without replacing any existing file."""
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        writer(stream)
        stream.flush()
        os.fsync(stream.fileno())


def checkpoint_json_bytes(value):
    """Keep the existing readable checkpoint sidecar encoding."""
    return (json.dumps(value, indent=2, allow_nan=False) + "\n").encode()


def checkpoint_publish_marker(paths, marker):
    """Replace a complete durable marker before any canonical checkpoint file changes."""
    checkpoint_write(paths["marker_next"], lambda stream: stream.write(checkpoint_json_bytes(marker)))
    paths["marker_next"].replace(paths["marker"])
    checkpoint_sync_directory(paths)


def validate_checkpoint_metadata(metadata_record, identity, path):
    """Verify a raw sidecar's exact schedule, precision identity, seeds, and payload binding."""
    required = {"schema_version", "kind", "checkpoint_path", "checkpoint_sha256", "identity_digest",
                "source_commit", "configuration_digest", "iteration", "copy_seeds", "target_updates",
                "steps_per_copy", "pending_report_updates", "elapsed_seconds", "written_utc",
                "loaded_gpu_libraries"}
    if not isinstance(metadata_record, dict) or set(metadata_record) != required:
        raise ValueError("checkpoint metadata is incomplete or has unknown fields")
    if metadata_record["identity_digest"] != canonical_digest(identity):
        raise ValueError("checkpoint identity differs from the admitted training run")
    learner = identity["configuration"]["learner"]
    iteration = metadata_record["iteration"]
    target = learner["total_timesteps"] // (learner["n_envs"] * learner["num_steps"])
    expected = {"kind": "optimized_training_checkpoint", "checkpoint_path": str(path),
                "identity_digest": canonical_digest(identity), "source_commit": identity["source_commit"],
                "configuration_digest": canonical_digest(identity["configuration"]),
                "copy_seeds": identity["copy_seeds"], "target_updates": target}
    if (type(metadata_record["schema_version"]) is not int or metadata_record["schema_version"] != 1
            or any(metadata_record[name] != value for name, value in expected.items())
            or type(iteration) is not int or not 0 <= iteration <= target
            or type(metadata_record["target_updates"]) is not int
            or not isinstance(metadata_record["copy_seeds"], list)
            or any(type(seed) is not int for seed in metadata_record["copy_seeds"])
            or type(metadata_record["steps_per_copy"]) is not int
            or metadata_record["steps_per_copy"] != iteration * learner["n_envs"] * learner["num_steps"]):
        raise ValueError("checkpoint metadata differs from the admitted identity or original schedule")
    checksum = metadata_record["checkpoint_sha256"]
    if not isinstance(checksum, str) or len(checksum) != 64 or any(c not in "0123456789abcdef" for c in checksum):
        raise ValueError("checkpoint metadata requires a literal SHA256 payload hash")
    elapsed, count = metadata_record["elapsed_seconds"], metadata_record["pending_report_updates"]
    execution = identity["configuration"]["execution"]
    if (type(elapsed) not in (int, float) or not math.isfinite(elapsed) or elapsed < 0
            or type(count) is not int or count < 0 or count > iteration
            or (execution["report_readout"] == "none" and count != 0)
            or (execution["report_readout"] != "none" and count >= execution["report_every_iterations"])):
        raise ValueError("checkpoint metadata reporting counters or elapsed runtime are invalid")
    written = datetime.fromisoformat(metadata_record["written_utc"])
    if written.tzinfo is None or written.utcoffset().total_seconds() != 0:
        raise ValueError("checkpoint metadata requires a timezone-aware UTC write time")
    if iteration:
        validate_gpu_library_receipt(metadata_record["loaded_gpu_libraries"], identity["backend"])


def checkpoint_pair_record(paths, identity, payload_name="payload", metadata_name="metadata"):
    """Read a complete pair without importing or unpickling numerical state."""
    payload, sidecar = paths[payload_name], paths[metadata_name]
    if not payload.exists() and not sidecar.exists():
        return None
    if not payload.exists() or not sidecar.exists():
        raise ValueError("checkpoint payload and metadata must both exist")
    metadata_record = json.loads(sidecar.read_text())
    validate_checkpoint_metadata(metadata_record, identity, paths["payload"])
    checksum = file_sha256(payload)
    if checksum != metadata_record["checkpoint_sha256"]:
        raise ValueError("checkpoint payload hash differs from its complete metadata pair")
    return {"checkpoint_sha256": checksum, "metadata_sha256": file_sha256(sidecar), "metadata": metadata_record}


def validate_checkpoint_pair_record(record, identity, path):
    """Reject forged marker records before they can authorize any file replacement."""
    if not isinstance(record, dict) or set(record) != {"checkpoint_sha256", "metadata_sha256", "metadata"}:
        raise ValueError("checkpoint transaction pair record is incomplete")
    validate_checkpoint_metadata(record["metadata"], identity, path)
    for name in ("checkpoint_sha256", "metadata_sha256"):
        checksum = record[name]
        if not isinstance(checksum, str) or len(checksum) != 64 or any(c not in "0123456789abcdef" for c in checksum):
            raise ValueError("checkpoint transaction requires literal SHA256 hashes")
    if record["checkpoint_sha256"] != record["metadata"]["checkpoint_sha256"]:
        raise ValueError("checkpoint transaction payload hashes disagree")


def validate_checkpoint_marker(marker, paths, identity):
    """Accept only this writer's two transaction phases and exact full admitted identity."""
    fields = {"schema_version", "kind", "phase", "checkpoint_path", "identity", "identity_digest", "previous", "next"}
    if (not isinstance(marker, dict) or set(marker) != fields or type(marker["schema_version"]) is not int
            or marker["schema_version"] != 1 or marker["kind"] != "optimized_checkpoint_transaction"
            or marker["phase"] not in ("preparing", "prepared")
            or marker["checkpoint_path"] != str(paths["payload"])
            or canonical_digest(marker["identity"]) != canonical_digest(identity)
            or marker["identity_digest"] != canonical_digest(identity)):
        raise ValueError("checkpoint transaction differs from the admitted identity or reserved path")
    if marker["previous"] is not None:
        validate_checkpoint_pair_record(marker["previous"], identity, paths["payload"])
    if marker["phase"] == "prepared":
        validate_checkpoint_pair_record(marker["next"], identity, paths["payload"])
    elif marker["next"] is not None:
        raise ValueError("a preparing checkpoint transaction cannot claim a complete next pair")


def checkpoint_matches(path, record, role):
    """Match an existing reserved file to the exact bytes captured in its pair record."""
    return path.exists() and record is not None and file_sha256(path) == record[role + "_sha256"]


def checkpoint_cleanup(paths):
    """Remove temporary generations before removing their durable transaction marker last."""
    for name in ("previous_payload", "previous_metadata", "next_payload", "next_metadata", "marker_next"):
        if paths[name].exists():
            paths[name].unlink()
            checkpoint_sync_directory(paths)
    paths["marker"].unlink()
    checkpoint_sync_directory(paths)


def recover_checkpoint_locked(paths, identity):
    """Finish only a fully recorded generation, or discard a still uncommitted preparation."""
    transient = ("previous_payload", "previous_metadata", "next_payload", "next_metadata", "marker_next")
    if not paths["marker"].exists():
        # Before the first marker rename, only its reserved temporary file can exist.
        # A complete canonical pair proves the previous generation is still intact;
        # no bytes from that uncommitted temporary file authorize any state change.
        if paths["marker_next"].exists() and not any(paths[name].exists() for name in transient[:-1]):
            if checkpoint_pair_record(paths, identity) is None:
                raise ValueError("uncommitted marker has no previous complete checkpoint")
            paths["marker_next"].unlink()
            checkpoint_sync_directory(paths)
            return {"recovered": True, "action": "discarded_uncommitted_marker"}
        if any(paths[name].exists() for name in transient):
            raise ValueError("checkpoint transaction files exist without a durable marker")
        checkpoint_pair_record(paths, identity)
        return {"recovered": False}
    marker = json.loads(paths["marker"].read_text())
    validate_checkpoint_marker(marker, paths, identity)
    previous, following = marker["previous"], marker["next"]
    if marker["phase"] == "preparing":
        if checkpoint_pair_record(paths, identity) != previous:
            raise ValueError("uncommitted checkpoint preparation changed its previous complete pair")
        for name, role in (("previous_payload", "checkpoint"), ("previous_metadata", "metadata")):
            if paths[name].exists() and not checkpoint_matches(paths[name], previous, role):
                raise ValueError("previous checkpoint hard link differs from the transaction")
        checkpoint_cleanup(paths)
        return {"recovered": True, "action": "discarded_uncommitted_preparation"}
    # Example: canonical(old payload, old metadata) -> canonical(new payload, old metadata)
    # -> canonical(new payload, new metadata). Every present byte must have a known role.
    for name, role in (("payload", "checkpoint"), ("metadata", "metadata")):
        if paths[name].exists() and not any(checkpoint_matches(paths[name], record, role) for record in (previous, following)):
            raise ValueError("canonical checkpoint file differs from both recorded generations")
    for prefix, record in (("previous", previous), ("next", following)):
        for suffix, role in (("payload", "checkpoint"), ("metadata", "metadata")):
            if paths[prefix + "_" + suffix].exists() and not checkpoint_matches(paths[prefix + "_" + suffix], record, role):
                raise ValueError("staged checkpoint file differs from its recorded generation")
    for name in ("metadata", "previous_metadata", "next_metadata"):
        if paths[name].exists():
            records = [record for record in (previous, following) if checkpoint_matches(paths[name], record, "metadata")]
            if not records or not any(json.loads(paths[name].read_text()) == record["metadata"] for record in records):
                raise ValueError("checkpoint sidecar bytes disagree with the transaction's recorded metadata")
    if paths["marker_next"].exists():
        raise ValueError("prepared checkpoint has an unexpected second transaction marker")
    # Prove both next files before changing either canonical file. A metadata-only
    # observer verifies hashes and JSON here; numerical restore remains in load_checkpoint.
    for name, role in (("payload", "checkpoint"), ("metadata", "metadata")):
        candidates = [paths[name], paths["next_" + name]]
        matching = [candidate for candidate in candidates if checkpoint_matches(candidate, following, role)]
        if not matching or (name == "metadata" and json.loads(matching[0].read_text()) != following["metadata"]):
            raise ValueError("checkpoint transaction has no complete recorded next generation")
    complete = all(checkpoint_matches(paths[name], following, role)
                   for name, role in (("payload", "checkpoint"), ("metadata", "metadata")))
    if not complete and previous is not None and not all(
            checkpoint_matches(paths[name], previous, role)
            for name, role in (("previous_payload", "checkpoint"), ("previous_metadata", "metadata"))):
        raise ValueError("checkpoint publication lost its previous complete generation")
    for name, role in (("payload", "checkpoint"), ("metadata", "metadata")):
        if not checkpoint_matches(paths[name], following, role):
            staged = paths["next_" + name]
            if not checkpoint_matches(staged, following, role):
                raise ValueError("checkpoint transaction has no complete recorded next generation")
            staged.replace(paths[name])
            checkpoint_sync_directory(paths)
    if checkpoint_pair_record(paths, identity) != following:
        raise ValueError("published checkpoint pair differs from the durable transaction")
    checkpoint_cleanup(paths)
    return {"recovered": True, "action": "finished_prepared_generation", "checkpoint_sha256": following["checkpoint_sha256"]}


def recover_checkpoint(path, identity):
    """Recover metadata only under the shared lock; callers must prove the previous process stopped."""
    with checkpoint_lock(path) as paths:
        return recover_checkpoint_locked(paths, identity)


def save_checkpoint(path, state, report, identity, iteration, elapsed_seconds, loaded_libraries=None):
    """Keep the previous complete generation until the next payload and sidecar are durable."""
    validate_host_state(state, identity, iteration)
    validate_report_state(report, identity["configuration"]["execution"], iteration)
    if type(elapsed_seconds) not in (int, float) or not math.isfinite(elapsed_seconds) or elapsed_seconds < 0:
        raise ValueError("checkpoint elapsed runtime is invalid")
    if iteration:
        validate_gpu_library_receipt(loaded_libraries, identity["backend"])
    payload = {"schema_version": 1, "identity": identity, "identity_digest": canonical_digest(identity),
               "iteration": iteration, "elapsed_seconds": elapsed_seconds, "written_utc": utc_now(),
               "state": state, "report_state": report, "loaded_gpu_libraries": loaded_libraries}
    with checkpoint_lock(path, create=True) as paths:
        recover_checkpoint_locked(paths, identity)
        previous = checkpoint_pair_record(paths, identity)
        marker = {"schema_version": 1, "kind": "optimized_checkpoint_transaction", "phase": "preparing",
                  "checkpoint_path": str(paths["payload"]), "identity": identity,
                  "identity_digest": canonical_digest(identity), "previous": previous, "next": None}
        checkpoint_publish_marker(paths, marker)
        if previous is not None:
            os.link(paths["payload"], paths["previous_payload"], follow_symlinks=False)
            os.link(paths["metadata"], paths["previous_metadata"], follow_symlinks=False)
            checkpoint_sync_directory(paths)
        checkpoint_write(paths["next_payload"], lambda stream: pickle.dump(payload, stream, protocol=pickle.HIGHEST_PROTOCOL))
        learner = identity["configuration"]["learner"]
        metadata_record = {"schema_version": 1, "kind": "optimized_training_checkpoint",
            "checkpoint_path": str(paths["payload"]), "checkpoint_sha256": file_sha256(paths["next_payload"]),
            "identity_digest": payload["identity_digest"], "source_commit": identity["source_commit"],
            "configuration_digest": canonical_digest(identity["configuration"]), "iteration": iteration,
            "copy_seeds": identity["copy_seeds"], "target_updates": learner["total_timesteps"] // (learner["n_envs"] * learner["num_steps"]),
            "steps_per_copy": iteration * learner["n_envs"] * learner["num_steps"], "pending_report_updates": report["pending_count"],
            "elapsed_seconds": elapsed_seconds, "written_utc": payload["written_utc"], "loaded_gpu_libraries": loaded_libraries}
        checkpoint_write(paths["next_metadata"], lambda stream: stream.write(checkpoint_json_bytes(metadata_record)))
        checkpoint_sync_directory(paths)
        marker.update(phase="prepared", next=checkpoint_pair_record(paths, identity, "next_payload", "next_metadata"))
        checkpoint_publish_marker(paths, marker)
        recover_checkpoint_locked(paths, identity)


def load_checkpoint(path, identity):
    """Refuse changed source, configuration, seeds, precision, or mismatched checkpoint bytes."""
    with checkpoint_lock(path) as paths:
        recover_checkpoint_locked(paths, identity)
        pair = checkpoint_pair_record(paths, identity)
        if pair is None:
            raise ValueError("checkpoint payload and metadata are missing")
        with paths["payload"].open("rb") as stream:
            payload = pickle.load(stream)
        if (type(payload.get("schema_version")) is not int or payload["schema_version"] != 1
                or canonical_digest(payload.get("identity")) != canonical_digest(identity)
                or payload.get("identity_digest") != canonical_digest(identity)):
            raise ValueError("checkpoint identity differs from the admitted training run")
        validate_host_state(payload["state"], identity, payload["iteration"])
        validate_report_state(payload["report_state"], identity["configuration"]["execution"], payload["iteration"])
        if payload["iteration"]:
            validate_gpu_library_receipt(payload.get("loaded_gpu_libraries"), identity["backend"])
        metadata_record = pair["metadata"]
        if (any(payload[name] != metadata_record[name] for name in ("iteration", "elapsed_seconds", "written_utc", "loaded_gpu_libraries"))
                or payload["report_state"]["pending_count"] != metadata_record["pending_report_updates"]):
            raise ValueError("checkpoint payload counters differ from the verified sidecar")
        return payload


class CheckpointDeadline:
    """Schedule complete calls around measured checkpoint deadlines without claiming real-time bounds."""

    def __init__(self, interval_seconds, save_started, save_completed):
        """Start the deadline at a completed durable save and retain its actual write duration."""
        if type(interval_seconds) not in (int, float) or not math.isfinite(interval_seconds) or not 0 < interval_seconds <= 1200:
            raise ValueError("checkpoint interval must be finite and at most 1200 seconds")
        self.interval_seconds = interval_seconds
        self.cold_call_seconds = {}
        self.maximum_warm_call_seconds = {}
        self.calls_since_checkpoint = 0
        self.maximum_save_seconds = self.duration(save_started, save_completed)
        self.last_completed = save_completed

    @staticmethod
    def duration(started, completed):
        """Reject invalid or reversed monotonic observations before using them as timing evidence."""
        if (any(type(value) not in (int, float) or not math.isfinite(value) for value in (started, completed))
                or started < 0 or completed < started):
            raise ValueError("checkpoint scheduling requires valid monotonic timing observations")
        return completed - started

    @property
    def save_allowance_seconds(self):
        """Reserve twice the slowest measured save and at least thirty seconds for variation."""
        return max(30.0, 2 * self.maximum_save_seconds)

    def plan(self, now, preferred_updates):
        """Keep a fitting group, use a fitting single update near a deadline, or checkpoint first."""
        self.duration(self.last_completed, now)
        if type(preferred_updates) is not int or preferred_updates < 1:
            raise ValueError("checkpoint scheduling requires a positive complete update count")
        save = self.save_allowance_seconds
        single = self.maximum_warm_call_seconds.get(1)
        single = None if single is None else 2 * single
        if save >= self.interval_seconds or (single is not None and single + save >= self.interval_seconds):
            raise ValueError("measured single-update and checkpoint save allowances cannot fit the checkpoint interval")
        count = preferred_updates
        call = self.maximum_warm_call_seconds.get(count)
        call = None if call is None else 2 * call
        deadline = self.last_completed + self.interval_seconds
        # Example: 200s grouped allowance cannot use 80s remaining, but a 20s
        # original update plus 30s save can, without padding or changing the learner.
        if count > 1 and call is not None:
            if call + save >= self.interval_seconds:
                count, call = 1, single
            elif now + call + save >= deadline and single is not None and now + single + save < deadline:
                count, call = 1, single
        if call is None:
            # An unmeasured/cold variant gets a fresh checkpoint window and an
            # immediate save after measurement; its first duration is not invented.
            before = self.calls_since_checkpoint > 0 or now - self.last_completed >= save
        else:
            before = now + call + save >= deadline
        return {"updates": count, "checkpoint_before_call": before, "unmeasured_call": call is None,
                "call_allowance_seconds": call, "save_allowance_seconds": save}

    def observe_call(self, updates, started, completed):
        """Keep cold compilation separate from warmed calls through readout and progress writes."""
        seconds = self.duration(started, completed)
        if type(updates) is not int or updates < 1 or started < self.last_completed:
            raise ValueError("checkpoint call observation does not follow its completed checkpoint")
        if updates not in self.cold_call_seconds:
            self.cold_call_seconds[updates] = seconds
            calibration = "cold"
        else:
            calibration = "warm_calibration" if updates not in self.maximum_warm_call_seconds else None
            self.maximum_warm_call_seconds[updates] = max(seconds, self.maximum_warm_call_seconds.get(updates, 0.0))
        self.calls_since_checkpoint += 1
        return calibration

    def prepare_call(self, preferred_updates, checkpoint_now, should_stop, clock=time.monotonic):
        """Recheck signals and the walltime boundary after any pre-call checkpoint finishes."""
        if should_stop():
            return None
        decision = self.plan(clock(), preferred_updates)
        if decision["checkpoint_before_call"]:
            checkpoint_now("before_training_call")
            if should_stop():
                return None
            decision = self.plan(clock(), preferred_updates)
            if decision["checkpoint_before_call"]:
                raise RuntimeError("checkpoint timing overhead leaves no safe window for the next complete call")
        return decision["updates"]

    def checkpoint_due(self, now):
        """Begin writing while the measured serialization allowance still fits the current deadline."""
        self.duration(self.last_completed, now)
        return now + self.save_allowance_seconds >= self.last_completed + self.interval_seconds

    def completed_checkpoint(self, started, completed):
        """Record actual completion spacing, including write time, before resetting the next deadline."""
        seconds = self.duration(started, completed)
        if started < self.last_completed:
            raise ValueError("checkpoint write observation predates the previous completed checkpoint")
        spacing = completed - self.last_completed
        self.maximum_save_seconds = max(seconds, self.maximum_save_seconds)
        self.last_completed = completed
        self.calls_since_checkpoint = 0
        return self.record(spacing, seconds)

    def record(self, spacing=None, save_seconds=None):
        """Expose actual timing and an explicit violation rather than inferred deadline compliance."""
        return {"checkpoint_interval_seconds": self.interval_seconds,
                "completed_checkpoint_spacing_seconds": spacing,
                "checkpoint_save_seconds": self.maximum_save_seconds if save_seconds is None else save_seconds,
                "maximum_observed_save_seconds": self.maximum_save_seconds,
                "cold_call_seconds": {str(count): seconds for count, seconds in self.cold_call_seconds.items()},
                "maximum_observed_warm_call_seconds": {str(count): seconds for count, seconds in self.maximum_warm_call_seconds.items()},
                "checkpoint_save_allowance_seconds": self.save_allowance_seconds,
                "checkpoint_deadline_exceeded": spacing is not None and spacing > self.interval_seconds}


def checkpoint_with_timing(write_checkpoint, write_timing, interval_seconds, deadline=None, clock=time.monotonic):
    """Finish the full durable state, publish timing evidence, then reject an observed deadline violation."""
    started = clock()
    write_checkpoint()
    completed = clock()
    if deadline is None:
        deadline = CheckpointDeadline(interval_seconds, started, completed)
        timing = deadline.record()
    else:
        timing = deadline.completed_checkpoint(started, completed)
    write_timing(timing)
    if timing["checkpoint_deadline_exceeded"]:
        raise RuntimeError("completed checkpoint spacing exceeded the configured limit; full state was saved and requires diagnosis")
    return deadline


def next_update_count(iteration, target, group_size):
    """Use a complete group when it fits, then single original updates without padding."""
    if any(type(value) is not int for value in (iteration, target, group_size)):
        raise ValueError("update counters must be integers")
    if not 0 <= iteration <= target or group_size < 1:
        raise ValueError("update counters or execution group size are invalid")
    remaining = target - iteration
    return 0 if remaining == 0 else group_size if remaining >= group_size else 1


@contextmanager
def unit_lock(directory):
    """Prevent two processes from updating or checkpointing the same training unit."""
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "training.lock").open("a+") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def new_attempt(directory):
    """Keep each process failure distinct from later checkpoint-resume attempts."""
    attempts = directory / "attempts"
    attempts.mkdir(exist_ok=True)
    numbers = [int(path.name.removeprefix("attempt-")) for path in attempts.glob("attempt-*")]
    number = max(numbers, default=0) + 1
    path = attempts / f"attempt-{number:04d}"
    path.mkdir()
    return number, path


def prepare_unit(directory, specification):
    """Refuse output reuse before replacing any existing experiment records."""
    marker = directory / "training_unit.json"
    unit_identity = {"kind": "validated_optimized_training", "specification_digest": canonical_digest(
        {name: value for name, value in specification.items() if name != "validation"})}
    if marker.exists():
        if json.loads(marker.read_text()) != unit_identity:
            raise ValueError("the existing training unit belongs to a different specification")
    elif any(path.name != "training.lock" for path in directory.iterdir()):
        raise ValueError("the selected unit already contains non-training experiment records")
    else:
        atomic_json(marker, unit_identity)


class TrainingExecution:
    """Use the validated Executor while retaining report windows across saves."""

    def __init__(self, executor, paired, jax, numpy, jnp, report=None):
        """Keep physical state and pending metrics without reinitializing policy schedules."""
        self.executor, self.paired, self.jax = executor, paired, jax
        self.numpy, self.jnp = numpy, jnp
        self.report = report or {"pending_sums": None, "pending_count": 0, "last_report_iteration": 0}

    def advance(self, state, count, iteration):
        """Complete original updates and publish only complete reporting intervals."""
        options = self.executor.options
        if options["report_readout"] == "none":
            # Before: 3 updates remain for a group of 5; after: three unpadded one-update calls.
            if count == self.executor.group_size:
                state, metrics = self.executor.run(state, count, initial_iteration=iteration)
            elif count == 1 and count < self.executor.group_size:
                state, metrics = self.executor.one(state)
            else:
                raise ValueError("training call does not match its validated execution group or tail")
            return state, metrics, []
        if count != 1:
            raise ValueError("reported training must advance one complete update at a time")

        # This is Executor's report operation with its accumulator retained between calls.
        state, metrics = self.executor.one(state)
        per_update = options["report_readout"] == "per_update"
        values = self.jax.device_get(metrics) if per_update else metrics
        arrays = self.numpy if per_update else self.jnp
        self.report["pending_sums"] = self.paired.accumulate_report(self.report["pending_sums"], values, arrays)
        self.report["pending_count"] += 1
        reports = []
        interval = options["report_every_iterations"]
        if self.report["pending_count"] == interval:
            window = self.report["pending_sums"] if per_update else self.jax.device_get(self.report["pending_sums"])
            reports.append({"iteration": iteration + 1, "window_updates": interval,
                            "metrics": self.paired.finish_report(window, interval)})
            self.report = {"pending_sums": None, "pending_count": 0, "last_report_iteration": iteration + 1}
        return state, metrics, reports


def run(arguments):
    """Train only after admission, restoring schedules and full state together."""
    placement = training_placement(os.environ, socket.gethostname())
    from post_comparison_validation import validate_training_admission

    # Admission rereads raw evidence; a flag or an old receipt alone never starts training.
    specification = validate_specification(validate_training_admission(arguments.final_spec))
    source_root = Path(specification["source_root"]).resolve(strict=True)
    source_receipt(source_root, specification["source_commit"])
    helper_commit = subprocess.check_output(["git", "-C", str(PROJECT_ROOT), "rev-parse", "HEAD"], text=True).strip()
    helpers = source_receipt(PROJECT_ROOT, helper_commit, [str((SCRIPT_ROOT / name).relative_to(PROJECT_ROOT))
        for name in ("train_optimized.py", "post_comparison_validation.py", "short_learning.py", "portable_evidence.py", "infrastructure_retry.py", "pregather_async_retry.py",
                     "h100_nvl_slurm_queue.py", "h100_nvl_slurm_case.py", "h100_nvl_slurm_evidence.py", "h100_nvl_slurm_site_validation.py",
                     "h100_nvl_slurm_qualification.py", "h100_nvl_allocation_observation.py",
                     "controller.py", "ideas.py", "queue_worker.py",
                     "diagnose_gpu_execution.py", 'device_memory_limit.py', 'validate_training_deployment.py')])
    runtime = enforce_runtime(specification["runtime_environment"], os.environ)
    for name in ("experiment_background.md", "manifest.yaml", "config_resolved.yaml", "command.txt"):
        if not (arguments.run_dir / name).is_file():
            raise ValueError("the campaign is missing its launch-time record: " + name)

    directory = arguments.run_dir / "data" / arguments.unit_id
    with unit_lock(directory):
        assets = verify_assets(specification["asset_record"], os.environ)
        prepare_unit(directory, specification)
        attempt, attempt_directory = new_attempt(directory)
        waiting = {"state": "waiting_for_standing_lock" if placement["kind"] == "serval02" else "validating_allocation",
                   "attempt": attempt, "pid": os.getpid(), "time": utc_now(), "placement": placement}
        atomic_json(attempt_directory / "progress.json", waiting)
        atomic_json(directory / "progress.json", waiting)
        try:
            with placement_guard(placement):
                _train(arguments, specification, source_root, helpers, runtime, assets, directory, attempt, attempt_directory)
        except Exception as error:
            if not (attempt_directory / "failure.json").exists():
                failure = {"error": repr(error), "failure_stage": "placement", "attempt": attempt,
                           "pid": os.getpid(), "time": utc_now()}
                atomic_json(attempt_directory / "failure.json", failure)
                atomic_json(directory / "failure.json", failure)
                atomic_json(directory / "progress.json", {**failure, "state": "failed"})
            raise


def _train(arguments, specification, source_root, helpers, runtime, assets, directory, attempt, attempt_directory,
           *, operational_paths=None, metadata_hooks=None):
    """Keep numerical imports and device work inside the admitted, locked unit."""
    stop = {"reason": None}
    started = time.monotonic()

    def request_stop(signum, frame):
        """Save after the active complete update/group instead of interrupting device state."""
        stop["reason"] = signal.Signals(signum).name

    def progress(value):
        """Publish the current process attempt without inheriting an earlier failure."""
        record = {**value, "attempt": attempt, "pid": os.getpid(), "time": utc_now()}
        atomic_json(attempt_directory / "progress.json", record)
        atomic_json(directory / "progress.json", record)

    for signum in (signal.SIGUSR1, signal.SIGTERM, signal.SIGINT):
        signal.signal(signum, request_stop)
    progress({"state": "validating_runtime"})
    try:
        public_source = specification.get("jaxtari_source_root") if operational_paths is None else operational_paths["jaxtari_source_root"]
        jaxtari_source = jaxtari_python_receipt(public_source)
        installed_libraries = installed_gpu_library_receipts()
        packages = {name: metadata.version(name) for name in PACKAGE_NAMES}
        if metadata_hooks is not None:
            metadata_hooks.before_numerical_imports({"jaxtari_source": jaxtari_source,
                "gpu_libraries": installed_libraries, "packages": packages})
        # Load the Executor and numerical implementation from the admitted source only.
        sys.path.insert(0, str(source_root / "10_jax_exploration_platform" / "src"))
        sys.path.insert(0, str(source_root / "10_jax_exploration_platform" / "scripts" / "atari_pixel_campaign"))
        if "jax" in sys.modules:
            raise RuntimeError("JAX was imported before the validated runtime settings were applied")
        import jax
        import jax.numpy as jnp
        import numpy as np
        paired = importlib.import_module("paired_benchmark")
        from exploration_platform.atari_pixels.config import PixelPPOConfig
        from exploration_platform.atari_pixels.environment import AtariPixelEnvironment
        from exploration_platform.atari_pixels.training import PixelPPORND
        expected_paired = source_root / "10_jax_exploration_platform/scripts/atari_pixel_campaign/paired_benchmark.py"
        if Path(paired.__file__).resolve() != expected_paired:
            raise ValueError("the Executor was not imported from its admitted source")
        jax.config.update("jax_enable_x64", True)
        jax.config.update("jax_default_matmul_precision", "highest")
        devices = jax.devices()
        if len(devices) != 1 or devices[0].platform != "gpu":
            raise RuntimeError("final training requires exactly one visible GPU")
        device_kind = devices[0].device_kind
        hardware_class = ("h100" if "H100" in device_kind else
                          "jaguar03_rtx_a4500" if "A4500" in device_kind and socket.gethostname().split(".")[0] == "jaguar03"
                          else "unsupported")
        if hardware_class != specification["hardware_class"]:
            raise RuntimeError("the actual GPU differs from the admitted hardware class")
        query = subprocess.check_output(["nvidia-smi", "--query-gpu=index,name,memory.total,driver_version,uuid",
                                         "--format=csv,noheader"], text=True).strip()
        backend = {"device_kind": device_kind, "driver_version": query.splitlines()[0].split(",")[3].strip(),
                   "packages": packages,
                   "jaxtari_source": jaxtari_source, "installed_gpu_libraries": installed_libraries}
        identity = checkpoint_identity(specification, helpers, runtime, backend, assets)
        if metadata_hooks is not None:
            # Metadata hooks receive no learner state and cannot replace source/configuration/seeds in this constructor.
            bound = metadata_hooks.identity_metadata(runtime, backend, assets)
            if set(bound) != {"runtime_environment", "backend", "assets", "deployment"}:
                raise ValueError("training metadata hooks must return only the declared identity metadata")
            identity = checkpoint_identity(specification, helpers, bound["runtime_environment"], bound["backend"], bound["assets"])
            identity["deployment"] = bound["deployment"]
        identity_path = directory / "training_identity.json"
        if identity_path.exists() and json.loads(identity_path.read_text()) != identity:
            raise ValueError("the existing training unit has a different identity")
        atomic_json(identity_path, identity)
        atomic_json(directory / "final_specification.json", specification)
        atomic_json(directory / "validation_receipt.json", specification["validation"])
        atomic_json(directory / "arguments.json", {key: str(value) if isinstance(value, Path) else value
                                                  for key, value in vars(arguments).items()})
        atomic_json(directory / "hardware.json", {"hostname": socket.gethostname(), "device_kind": device_kind,
                    "gpu_query": query,
                    "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""), "pid": os.getpid(),
                    "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
                    "runtime_environment": runtime, "backend": backend})
        configuration = specification["configuration"]
        config = PixelPPOConfig(**configuration["learner"])
        environment = AtariPixelEnvironment(**configuration["environment"])
        jaxtari = sys.modules.get("jaxtari")
        expected_jaxtari = Path(metadata.distribution("jaxtari").locate_file("jaxtari/__init__.py")).resolve(strict=True)
        if jaxtari is None or Path(jaxtari.__file__).resolve() != expected_jaxtari:
            raise ValueError("the imported JAXtari package differs from the verified installed Python files")
        trainer = PixelPPORND(config, environment)
        executor = paired.Executor(trainer, configuration["execution"])
        checkpoint = directory / "checkpoints/latest.pkl"
        prior_seconds = 0.0
        loaded_libraries = None
        libraries_verified_in_this_process = False
        if checkpoint.exists():
            payload = load_checkpoint(checkpoint, identity)
            state = paired.device_copy(payload["state"])
            report = payload["report_state"]
            if configuration["execution"]["report_readout"] == "scheduled" and report["pending_sums"] is not None:
                report["pending_sums"] = paired.device_copy(report["pending_sums"])
            prior_seconds = payload["elapsed_seconds"]
            loaded_libraries = payload.get("loaded_gpu_libraries")
        else:
            progress({"state": "initializing"})
            state = trainer.initialize(tuple(specification["copy_seeds"]))
            jax.block_until_ready(state)
            progress({"state": "normalization_warmup"})
            state = trainer.warmup(state)
            jax.block_until_ready(state)
            report = None
        execution = TrainingExecution(executor, paired, jax, np, jnp, report)
        update = int(jax.device_get(state.iteration))
        target = config.total_iterations
        validate_report_state(execution.report, configuration["execution"], update)
        atomic_json(directory / "config_resolved.json", configuration)
        append_record(directory / "metrics.jsonl", {"record": "resumed" if checkpoint.exists() else "initialized",
                      "update": update, "process_seconds": time.monotonic() - started, "time": utc_now()})

        checkpoint_deadline = None

        def checkpoint_now(reason):
            """Save every state leaf and preserve timing evidence before any deadline failure."""
            nonlocal checkpoint_deadline

            def write_checkpoint():
                """Include device synchronization, transfer, serialization and transaction cleanup in save time."""
                jax.block_until_ready(state)
                save_checkpoint(checkpoint, jax.device_get(state), jax.device_get(execution.report), identity,
                                update, prior_seconds + time.monotonic() - started, loaded_libraries)

            def write_timing(timing):
                """Keep both current-attempt evidence and the metrics history for an observed violation."""
                record = {"schema_version": 1, "kind": "optimized_training_checkpoint_timing", **timing,
                          "reason": reason, "update": update, "attempt": attempt, "pid": os.getpid(), "time": utc_now()}
                atomic_json(attempt_directory / "checkpoint_timing.json", record)
                atomic_json(directory / "checkpoint_timing.json", record)
                append_record(directory / "metrics.jsonl", {"record": "checkpoint_timing", **record})

            checkpoint_deadline = checkpoint_with_timing(write_checkpoint, write_timing,
                                                         arguments.checkpoint_seconds, checkpoint_deadline)

        # The first save also measures serialization on this actual target before training calls.
        checkpoint_now("initial")
        def should_stop():
            """Honor a signal or process time limit again after a proactive checkpoint consumes time."""
            if stop["reason"] is not None:
                return True
            if arguments.maximum_seconds and time.monotonic() - started >= arguments.maximum_seconds:
                stop["reason"] = "maximum_seconds"
                return True
            return False

        while update < target and not should_stop():
            preferred = next_update_count(update, target, executor.group_size)
            count = checkpoint_deadline.prepare_call(preferred, checkpoint_now, should_stop)
            if count is None:
                break
            before = update
            call_started = time.monotonic()
            state, metrics, reports = execution.advance(state, count, before)
            jax.block_until_ready(state)
            update = int(jax.device_get(state.iteration))
            if update != before + count:
                raise RuntimeError("the learner counter did not advance by the exact complete update count")
            if not libraries_verified_in_this_process:
                loaded_libraries = actual_gpu_library_receipt(devices[0].client, installed_libraries)
                validate_gpu_library_receipt(loaded_libraries, backend)
                atomic_json(attempt_directory / "loaded_gpu_libraries.json", loaded_libraries)
                atomic_json(directory / "loaded_gpu_libraries.json", loaded_libraries)
                if metadata_hooks is not None:
                    metadata_hooks.after_first_update(loaded_libraries)
                libraries_verified_in_this_process = True
            # Grouped metrics describe its last update; scheduled rows retain the original full windows.
            if configuration["execution"]["report_readout"] == "none":
                reports = [{"iteration": update, "window_updates": 1,
                            "metrics_scope": "last_update_in_execution_group", "metrics": jax.device_get(metrics)}]
            for row in reports:
                values = {name: np.asarray(value).tolist() for name, value in row["metrics"].items()}
                append_record(directory / "metrics.jsonl", {"record": "training_metrics", **row, "metrics": values})
            status = {"state": "running", "update": update, "target_updates": target,
                        "updates_in_call": count, "seconds": time.monotonic() - call_started,
                        "process_elapsed_seconds": time.monotonic() - started,
                        "recorded_elapsed_seconds_total": prior_seconds + time.monotonic() - started,
                        "steps_per_copy": update * config.batch_size,
                        "copy_progress": [{"seed": seed, "updates": update,
                                           "transitions": update * config.batch_size,
                                           "transitions_per_environment": update * config.num_steps}
                                          for seed in specification["copy_seeds"]],
                        "pending_report_updates": execution.report["pending_count"],
                        "time": utc_now(), "pid": os.getpid()}
            progress(status)
            calibration = checkpoint_deadline.observe_call(count, call_started, time.monotonic())
            if calibration is not None or checkpoint_deadline.checkpoint_due(time.monotonic()):
                checkpoint_now(calibration + "_call_measured" if calibration else "deadline_approaching")
        checkpoint_now("final")
        result = {"completed": update == target, "update": update, "target_updates": target,
                  "requested_steps_per_copy": config.total_timesteps, "steps_per_copy": update * config.batch_size,
                  "copy_seeds": specification["copy_seeds"], "copies": config.n_copies,
                  "pending_report_updates": execution.report["pending_count"],
                  "last_report_iteration": execution.report["last_report_iteration"],
                  "stop_reason": "completed" if update == target else stop["reason"],
                  "process_elapsed_seconds": time.monotonic() - started,
                  "recorded_elapsed_seconds_total": prior_seconds + time.monotonic() - started,
                  "identity_digest": canonical_digest(identity), "attempt": attempt, "time": utc_now()}
        atomic_json(directory / "result.json", result)
        atomic_json(attempt_directory / "result.json", result)
        progress({**result, "state": "completed" if result["completed"] else "checkpointed"})
        print(json.dumps(result, allow_nan=False), flush=True)
    except Exception as error:
        # A failed run remains failed; recovery starts from its last complete checkpoint.
        failure = {"error": repr(error), "time": utc_now(), "pid": os.getpid(), "attempt": attempt}
        atomic_json(attempt_directory / "failure.json", failure)
        atomic_json(directory / "failure.json", failure)
        progress({"state": "failed", "error": repr(error)})
        raise


def parse_arguments(argv=None):
    """Allow placement and stop boundaries without exposing learner overrides."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--final-spec", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--unit-id", required=True)
    parser.add_argument("--checkpoint-seconds", type=int, default=1200)
    parser.add_argument("--maximum-seconds", type=float, default=0)
    arguments = parser.parse_args(argv)
    if (Path(arguments.unit_id).name != arguments.unit_id or arguments.unit_id in {"", ".", ".."}
            or not 0 < arguments.checkpoint_seconds <= 1200
            or not math.isfinite(arguments.maximum_seconds) or arguments.maximum_seconds < 0):
        parser.error("use a plain unit name, checkpoint interval in (0,1200], and nonnegative finite maximum seconds")
    arguments.run_dir = arguments.run_dir.resolve(strict=True)
    arguments.final_spec = arguments.final_spec.resolve(strict=True)
    return arguments


if __name__ == "__main__":
    run(parse_arguments())
