"""Prepare and verify one H100 NVL site validation without submitting or admitting workers.

Only serval06 through serval09 are eligible. Original source admission remains
unchanged. A passed result is site evidence for a later reviewed placement
decision, never permission to claim a campaign case or count a scientific idea.
"""

import argparse
from contextlib import contextmanager
import csv
from datetime import datetime, timezone
import hashlib
import importlib.util
from importlib import metadata
import json
import os
from pathlib import Path
import re
import shlex
import signal
import socket
import stat
import subprocess
import sys
import time
import traceback
import xml.etree.ElementTree as ET


PREFIX = "10_jax_exploration_platform/scripts/atari_pixel_campaign/"
SOURCE_COMMIT = "d28e6735e79788f618acb0ebfde8609db1f776af"
CLEANRL_COMMIT = "fe8d8a03c41a7ef5b523e2e354bd01c363e786bb"
HOSTS = ("serval06", "serval07", "serval08", "serval09")
MODEL = "NVIDIA H100 NVL"
MEMORY_MIB = 95830
DRIVER = "590.48.01"
REFERENCE_GPU = "GPU-fed69954-5a71-dbb2-ee91-544e0eb6b8a3"
RUNTIME = {"JAX_ENABLE_X64": "true", "JAX_DEFAULT_MATMUL_PRECISION": "highest",
           "XLA_FLAGS": "--xla_gpu_exclude_nondeterministic_ops=true",
           "JAX_ENABLE_COMPILATION_CACHE": "false", "JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES": "",
           "JAXTARI_SKIP_SPRITE_UPDATE": "1", "XLA_PYTHON_CLIENT_MEM_FRACTION": "0.85",
           "XLA_PYTHON_CLIENT_PREALLOCATE": "false", "OMP_NUM_THREADS": "8",
           "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "PYTHONNOUSERSITE": "1", "PYTHONUNBUFFERED": "1"}
ALLOCATION_KEYS = ("SLURM_JOB_ID", "SLURM_STEP_ID", "SLURM_JOB_GPUS", "SLURM_STEP_GPUS", "SLURM_GPUS_ON_NODE", "SLURM_CPUS_PER_TASK",
                   "SLURM_CPU_BIND_TYPE", "SLURM_CPU_BIND_LIST", "SLURM_CPU_BIND", "CUDA_VISIBLE_DEVICES")
GATE_ENVIRONMENT_POLICY = {"reject_nonempty_prefixes": ["PYTEST_"], "reject_nonempty_names": ["PYTHONOPTIMIZE"],
                           "cleanrl_reference": "exact original Git blob only", "error_values_recorded": False}
HELPERS = ("h100_nvl_slurm_site_validation.py", "controller.py", "ideas.py", "admit_measurements.py",
           "train_optimized.py", "diagnose_gpu_execution.py", "post_comparison_validation.py",
           "infrastructure_retry.py", "pregather_async_retry.py")
QUALIFICATION_HELPERS = ("h100_nvl_slurm_qualification.py", "h100_nvl_allocation_observation.py")
PACKAGE_NAMES = ("jax", "jaxlib", "jax-cuda12-plugin", "jax-cuda12-pjrt", "nvidia-cudnn-cu12",
                 "nvidia-cublas-cu12", "nvidia-cuda-runtime-cu12", "jaxtari", "flax", "optax", "numpy")
EXPECTED_DTYPES = {name: ["float64"] if "statistics" in name else ["float32"] for name in
    ("parameters", "target", "optimizer_first", "optimizer_second", "observation_statistics", "reward_statistics")}
JOB_SHAPE = {"account": "sl5nw_base", "partition": "gpu", "qos": "normal", "nodes": 1,
             "ntasks": 1, "gres": "gpu:nvidia_h100_nvl:1", "cpus_per_task": 8,
             "memory": "64G", "time": "03:00:00", "eligible_nodes": list(HOSTS),
             "partition_caps": {"cpus": 400, "memory": "4T", "gpus": 40, "maximum_time": "4-00:00:00"}}
SUBMISSION_CONTROL = {"submit_held": True, "helper_submits_or_releases": False, "release_exact_saved_id_only": True,
    "parent_procedure": ["Submit the recorded command with --hold.",
        "Write and fsync the exact returned UVA job ID in slurm/submitted_jobids_uva.txt and write and fsync its submission receipt.",
        "Only after both records are durable, release that exact saved ID with scontrol release."],
    "release_command_template": ["scontrol", "release", "<exact_saved_uva_job_id>"]}


def load(name):
    """Load only the exact adjacent metadata helper; do not replace frozen numerical modules."""
    path = Path(__file__).resolve().with_name(name + ".py")
    specification = importlib.util.spec_from_file_location("h100_site_" + name, path)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def qualification_profile(plan):
    """Accept only the explicit optional target profile; keep the original reference driver unchanged."""
    profile = plan.get("qualification_profile")
    if profile is not None:
        if plan.get("schema_version") != 3:
            raise ValueError("the optional Slurm qualification profile is absent or unsupported")
        load("h100_nvl_slurm_qualification").require_profile(profile)
    return profile


def expected_driver(plan):
    """Separate the opt-in target driver from the original direct-server reference identity."""
    profile = qualification_profile(plan)
    return profile["expected_driver"] if profile is not None else plan["reference"]["driver"]


def capture_driver_image(identity, directory):
    """Stream an independent immutable shared copy while the actual mapped driver remains on this compute node."""
    directory = Path(directory).resolve(strict=True)
    original = Path(identity["driver_library"])
    if (not original.is_absolute() or not original.name.startswith("libcuda.so")
            or identity.get("pid") != os.getpid() or not re.fullmatch(r"[0-9a-f]{64}", identity["driver_library_sha256"])):
        raise ValueError("capture only this process's exact mapped CUDA driver identity")
    # A new ordinary file owns copied bytes; no link to a mutable node-local driver is created.
    folder = directory / "captured_driver"
    if folder.is_symlink():
        raise ValueError("the shared driver capture directory cannot be a symlink")
    folder.mkdir(exist_ok=True)
    target = folder / original.name
    digest, count = hashlib.sha256(), 0
    with original.open("rb") as source, target.open("xb") as output:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            output.write(block)
            digest.update(block)
            count += len(block)
        if digest.hexdigest() != identity["driver_library_sha256"] or count < 1:
            raise ValueError("the actual mapped CUDA driver bytes changed before shared capture")
        output.flush()
        os.fsync(output.fileno())
    for path in (folder, directory):
        descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    return {"schema_version": 1, "kind": "captured_mapped_cuda_driver_image", "pid": os.getpid(),
        "original": {"path": str(original), "sha256": identity["driver_library_sha256"]},
        "captured": {"path": str(target), "sha256": identity["driver_library_sha256"]}, "bytes": count}


def validate_driver_capture(capture, identity, directory, pid):
    """Validate captured shared bytes without opening the original node-local driver path."""
    expected = Path(directory) / "captured_driver" / Path(identity["driver_library"]).name
    if (capture.get("schema_version") != 1 or capture.get("kind") != "captured_mapped_cuda_driver_image"
            or capture.get("pid") != pid or capture.get("original") != {"path": identity["driver_library"], "sha256": identity["driver_library_sha256"]}
            or capture.get("captured") != {"path": str(expected), "sha256": identity["driver_library_sha256"]}
            or type(capture.get("bytes")) is not int or capture["bytes"] < 1
            or expected.is_symlink() or expected.parent.is_symlink()
            or expected.stat().st_size != capture["bytes"] or sha256(expected) != identity["driver_library_sha256"]):
        raise ValueError("the immutable shared CUDA driver capture changed its bytes, process or original identity")
    return capture


def required_helpers(profile):
    """Pin the two extra metadata files only for the explicit target-driver qualification."""
    return HELPERS + QUALIFICATION_HELPERS if profile is not None else HELPERS


def profile_background(profile):
    """Describe the separately qualified driver and the exact limits of configured-isolation evidence."""
    if profile is None:
        return ""
    return ("The original direct-server reference driver 590.48.01 is retained. The explicit Slurm profile separately qualifies "
        "target driver 580.159.04. It trusts configured scheduler task/cgroup device isolation; live BPF device rules are not "
        "independently inspected, and a numeric Slurm-to-NVML/minor/PCI mapping is not reconstructed. Every parent and child "
        "must provide fresh own allocation and task evidence, and the reference probe must bind its actual JAX CUDA ordinal "
        "to the selected physical UUID and PCI address. Before the allocation ends, the probe copies its actual mapped CUDA "
        "driver bytes into immutable shared artifacts so CPU and portable readers do not require the node-local driver.\n\n")


def read(path):
    """Decode one explicitly named document."""
    return json.loads(Path(path).read_bytes())


def sha256(path):
    """Hash installed images in bounded chunks without importing numerical libraries."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def receipt(path):
    """Retain the exact absolute path and bytes used by this operational check."""
    path = Path(path).resolve(strict=True)
    return {"path": str(path), "sha256": sha256(path)}


def bound(record):
    """Refuse a changed input instead of finding an alternative path."""
    if receipt(record["path"]) != record:
        raise ValueError("a site validation input changed: " + record["path"])
    return read(record["path"])


def write_once(path, value):
    """Publish complete exclusive metadata and preserve existing bytes."""
    contents = value.encode() if isinstance(value, str) else (json.dumps(value, indent=2, allow_nan=False) + "\n").encode()
    load("infrastructure_retry").write_once(Path(path), contents)


def process_start_time(pid):
    """Read Linux's original process birth counter without sending a signal."""
    try:
        return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[19]
    except FileNotFoundError:
        return None


def package_versions(python=None):
    """Read the chosen interpreter's distribution metadata without importing array packages."""
    if python is None or str(Path(sys.executable).absolute()) == str(Path(python).absolute()):
        return {name: metadata.version(name) for name in PACKAGE_NAMES}
    command = [str(python), "-I", "-c", "import importlib.metadata,json; print(json.dumps({name:importlib.metadata.version(name) for name in "
               + repr(PACKAGE_NAMES) + "}))"]
    return json.loads(query(command))


def require_runtime(environment):
    """Reject additional JAX/XLA execution controls and conflicting allocator aliases before JAX."""
    extra = sorted(name for name, value in environment.items() if value
        and name.startswith(("JAX_", "JAXTARI_", "XLA_", "CUDA_", "CUDNN_", "NVIDIA_"))
        and name not in {*RUNTIME, "CUDA_VISIBLE_DEVICES"})
    if extra:
        raise ValueError("unapproved site validation execution controls: " + ", ".join(extra))
    if any(environment.get(name) not in (None, "") for name in ("TF_GPU_ALLOCATOR", "XLA_CLIENT_MEM_FRACTION")):
        raise ValueError("the original allocator aliases must remain unset")
    for name, expected in RUNTIME.items():
        if environment.get(name) != expected:
            raise ValueError("site validation runtime differs for " + name)
    return {name: environment[name] for name in RUNTIME}


def source_identity(root, commit, paths):
    """Require a clean detached real Git root and committed bytes for every used source file."""
    return load("infrastructure_retry").git_receipt(root, commit, paths)


def verify_source_identity(record):
    """Rebuild the exact recorded source identity rather than accepting a supplied hash list."""
    if source_identity(record["source_root"], record["source_commit"], sorted(record["files"])) != record:
        raise ValueError("the site validation source identity changed")


def numerical_paths(root):
    """List the complete committed learner, campaign and gate closure from the original tree."""
    paths = query(["git", "-C", str(root), "ls-tree", "-r", "--name-only", SOURCE_COMMIT, "--",
        "10_jax_exploration_platform/src/exploration_platform/__init__.py",
        "10_jax_exploration_platform/src/exploration_platform/atari_pixels", PREFIX,
        "10_jax_exploration_platform/tests/atari_pixels", "10_jax_exploration_platform/tests/test_atari_pixel_environment.py"]).splitlines()
    return sorted(path for path in paths if path.endswith((".py", "worker_environment.sh")))


def cleanrl_identity(root):
    """Bind the exact original file whose three model definitions the frozen equations tests execute."""
    root = Path(root).resolve(strict=True)
    path = root / "cleanrl/ppo_rnd_envpool.py"
    top = query(["git", "-C", str(root), "rev-parse", "--show-toplevel"]).strip()
    commit = query(["git", "-C", str(root), "rev-parse", "HEAD"]).strip()
    original = subprocess.check_output(["git", "-C", str(root), "show", CLEANRL_COMMIT + ":cleanrl/ppo_rnd_envpool.py"])
    if (Path(top).resolve() != root or commit != CLEANRL_COMMIT or path.read_bytes() != original
            or query(["git", "-C", str(root), "status", "--porcelain", "--", "cleanrl/ppo_rnd_envpool.py"]).strip()):
        raise ValueError("the equations gate reference must be the unchanged original CleanRL file")
    return {"source_root": str(root), "source_commit": CLEANRL_COMMIT, "file": receipt(path),
        "git_blob": hashlib.sha1(b"blob " + str(len(original)).encode() + b"\0" + original).hexdigest()}


def require_gate_controls(environment, reference):
    """Do not let inherited pytest selectors or a different external model reference change the full gates."""
    unexpected = sorted(name for name, value in environment.items() if value and
        (name.startswith("PYTEST_") or name == "PYTHONOPTIMIZE"))
    if unexpected:
        raise ValueError("the original full learner gate cannot inherit: " + ", ".join(unexpected))
    if environment.get("CLEANRL_REFERENCE_FILE") not in (None, "", reference["file"]["path"]):
        raise ValueError("CLEANRL_REFERENCE_FILE differs from the exact original CleanRL reference")


def original_configuration(source, admission):
    """Require every learner and environment field to match both frozen defaults and admitted reference."""
    defaults = load("diagnose_gpu_execution").default_configuration(Path(source))
    defaults["learner"]["n_copies"] = 1
    expected = bound(admission["reference_configuration"])
    if json.loads(json.dumps(defaults["learner"])) != {**expected, "n_copies": 1}:
        raise ValueError("the site probe differs from the admitted original learner configuration")
    return json.loads(json.dumps({name: defaults[name] for name in ("learner", "environment")}))


def snapshot_admission(run):
    """Recheck original raw gates under the existing source-admission lock without holding it during GPU work."""
    controller = load("controller")
    admission = load("admit_measurements")
    with controller.queue_lock(run):
        ready = controller.ready_specification(run)
        if ready is None or ready["source_commit"] != SOURCE_COMMIT:
            raise ValueError("the original frozen numerical admission is not ready")
        gate = read(ready["gate_records"]["h100"])
        checks = {kind: admission.verified_check(gate["checks"][kind]["path"], SOURCE_COMMIT, "h100", kind,
                    run, ready["source_root"]) for kind in ("learner", "renderer")}
        paths = [Path(run) / controller.SENTINEL_NAME, Path(ready["reference_configuration_path"]),
                 *map(Path, ready["gate_records"].values())]
        return {"source_root": ready["source_root"], "source_commit": ready["source_commit"],
                "files": [receipt(path) for path in paths], "original_h100_checks": checks,
                "reference_configuration": receipt(ready["reference_configuration_path"])}


def check_admission(run, expected):
    """Reject stale site evidence while leaving the running campaign's admission and queue unchanged."""
    if snapshot_admission(run) != expected:
        raise ValueError("the source admission changed during the site validation; no workers are admitted")


def admission_copy_path(directory, index, original):
    """Give each captured source-admission file an exact location inside its new prepared directory."""
    return Path(directory) / "admission_files" / ("record-" + str(index + 1) + "_" + Path(original["path"]).name)


def archive_admission(directory, admission):
    """Copy unchanged original admission bytes before publishing the new prepared plan."""
    files = admission["files"]
    if len(files) != 4 or len({item["path"] for item in files}) != 4:
        raise ValueError("capture exactly the four distinct original source-admission files")
    captured = []
    for index, original in enumerate(files):
        # Before: original logical path plus SHA. After: the same bytes at one explicit captured path.
        contents = Path(original["path"]).read_bytes()
        if hashlib.sha256(contents).hexdigest() != original["sha256"] or receipt(original["path"]) != original:
            raise ValueError("an original source-admission file changed before it was captured")
        target = admission_copy_path(directory, index, original)
        load("infrastructure_retry").write_once(target, contents)
        captured.append({"original": original, "captured": receipt(target)})
    result = {"schema_version": 1, "kind": "original_source_admission_bytes", "files": captured}
    validate_admission_archive(directory, admission, result)
    return result


def validate_admission_archive(directory, admission, archive):
    """Verify complete immutable copies without reading a historical mutable admission path."""
    files = admission["files"]
    if (not isinstance(archive, dict) or archive.get("schema_version") != 1
            or archive.get("kind") != "original_source_admission_bytes" or len(files) != 4
            or len({item["path"] for item in files}) != 4 or len(archive.get("files", [])) != 4):
        raise ValueError("the prepared plan needs an exact four-file immutable admission archive")
    for index, (original, item) in enumerate(zip(files, archive["files"])):
        expected_path = str(admission_copy_path(directory, index, original))
        if (set(item) != {"original", "captured"} or item["original"] != original
                or item["captured"].get("path") != expected_path
                or item["captured"].get("sha256") != original["sha256"]
                or receipt(expected_path) != item["captured"]):
            raise ValueError("an immutable admission copy changed its original identity, bytes or exact captured path")
    return archive


def software_reference(result_path, installed_path, assets_path):
    """Rebuild a completed formal UVA packet before using its actual full shared-software map."""
    result_path, installed_path, assets_path = (Path(path).resolve(strict=True) for path in (result_path, installed_path, assets_path))
    case = read(result_path.with_name("case.json"))
    result = read(result_path)
    if (result.get("attempt") != 4 or result.get("case_completed") is not True
            or result.get("correctness_passed") is not True or result.get("measurement_valid") is not True
            or installed_path != result_path.with_name("installed_backend.json") or assets_path != result_path.with_name("asset_receipt.json")):
        raise ValueError("shared software must come from a complete actual fourth-attempt packet")
    validation = load("post_comparison_validation")
    proof = validation.validate_cohort_case(result_path, case["hypothesis"], case["hardware_class"],
        case["configuration"]["n_copies"], SOURCE_COMMIT, validation.Evidence())
    installed = read(installed_path)
    if set(installed.get("packages", {})) != set(PACKAGE_NAMES):
        raise ValueError("the actual shared-software packet lacks the complete package map")
    return {"result": receipt(result_path), "installed_backend": receipt(installed_path),
        "asset_receipt": receipt(assets_path), "packages": installed["packages"], "runtime_provenance": proof,
        "scope": "actual completed UVA A4500 shared-software observations; no H100 model or driver is inferred from this packet"}


def reference_identity(directory, installed_path, assets_path, software_result):
    """Keep old observed H100 versions separate from the separately captured shared-environment image hashes."""
    directory = Path(directory).resolve(strict=True)
    hardware = read(directory / "hardware.json")
    query = [value.strip() for value in next(csv.reader([hardware["gpu_query"]]))]
    libraries = read(directory / "loaded_gpu_libraries.json")
    diagnostic = read(directory / "diagnostic_plan.json")
    if (len(query) != 4 or query != [REFERENCE_GPU, MODEL, str(MEMORY_MIB) + " MiB", DRIVER]
            or hardware.get("hostname") != "serval02" or diagnostic["core"]["source_commit"] != SOURCE_COMMIT):
        raise ValueError("the supplied reference is not the original deterministic serval02 H100 NVL")
    original_runtime = hardware["runtime_environment"]
    if (any(original_runtime.get(name) != RUNTIME[name] for name in ("JAX_ENABLE_X64", "JAX_DEFAULT_MATMUL_PRECISION",
            "XLA_PYTHON_CLIENT_PREALLOCATE", "XLA_PYTHON_CLIENT_MEM_FRACTION", "OMP_NUM_THREADS"))
            or shlex.split(original_runtime.get("XLA_FLAGS", "")) != ["--xla_gpu_exclude_nondeterministic_ops=true"]):
        raise ValueError("the reference changed its original full-precision deterministic allocator controls")
    if (libraries.get("platform") != "gpu" or not libraries.get("platform_version")
            or any(type(libraries.get(name)) is not int for name in ("cudnnGetVersion", "cudaRuntimeGetVersion"))):
        raise ValueError("the H100 reference lacks actual backend/CUDA/cuDNN observations")
    if read(directory / "dtypes.json") != EXPECTED_DTYPES:
        raise ValueError("the H100 reference did not record the original float32 networks and float64 statistics")
    return {"hardware": receipt(directory / "hardware.json"), "loaded_libraries": receipt(directory / "loaded_gpu_libraries.json"),
        "diagnostic_plan": receipt(directory / "diagnostic_plan.json"), "dtype_record": receipt(directory / "dtypes.json"),
        "model": MODEL, "memory_mib": MEMORY_MIB, "driver": query[3],
        "backend": {name: libraries[name] for name in ("platform", "platform_version", "cudnnGetVersion", "cudaRuntimeGetVersion")},
        "shared_installed_backend": receipt(installed_path), "shared_asset_receipt": receipt(assets_path),
        "software": software_reference(software_result, installed_path, assets_path),
        "scope": "old H100 API/version observations plus separately recorded current shared-environment image hashes; no historical driver-image hash is invented"}


def resolve_allocation(environment, hostname, gpu_csv, gpu_xml):
    """Resolve one exact NVL from step-local visible indices and Slurm physical device minors."""
    host = hostname.split(".")[0]
    job = environment.get("SLURM_JOB_ID", "")
    if host not in HOSTS or not re.fullmatch(r"[1-9][0-9]*", job):
        raise ValueError("the site validation requires an owned serval06-through-serval09 Slurm step")
    rows = list(csv.reader(gpu_csv.splitlines()))
    devices = ET.fromstring(gpu_xml).findall("gpu")
    if len(rows) != 1 or len(devices) != 1 or len(rows[0]) != 5:
        raise ValueError("the Slurm step must expose exactly one whole physical H100 NVL")
    index, gpu, model, memory, driver = [value.strip() for value in rows[0]]
    device = devices[0]
    minor = device.findtext("minor_number", "")
    if (not index.isdecimal() or not minor.isdecimal() or not gpu.startswith("GPU-") or model != MODEL
            or memory != str(MEMORY_MIB) or device.findtext("uuid") != gpu
            or device.findtext("product_name") != MODEL):
        raise ValueError("visible GPU model, memory, UUID and physical-minor records disagree")
    allocations = {name: environment[name] for name in ("SLURM_JOB_GPUS", "SLURM_STEP_GPUS") if environment.get(name)}
    if not allocations or any(value not in {minor, gpu} for value in allocations.values()):
        raise ValueError("every recorded Slurm job/step allocation must name this exact physical GPU")
    visible = environment.get("CUDA_VISIBLE_DEVICES", "")
    if visible not in {"0", index, minor, gpu}:
        raise ValueError("step CUDA visibility differs from its physical Slurm allocation")
    return {"hostname": host, "job_id": job, "gpu_uuid": gpu, "model": model, "memory_mib": int(memory), "driver": driver,
        "physical_minor": minor, "visible_index": index, "cuda_visible_devices_original": visible,
        "slurm_allocations": allocations, "gpu_csv": gpu_csv, "gpu_xml": gpu_xml}


def query(arguments):
    """Run a bounded metadata query and keep its actual output."""
    return subprocess.check_output(arguments, text=True, timeout=20)


def capture_allocation_details(owned):
    """Record only the owned scheduler row, device-file numbers and GPU-only GRES mapping for diagnosis."""
    job = owned["job_id"]
    if not re.fullmatch(r"[1-9][0-9]*", job) or job not in owned["ledger_prefix"]:
        raise ValueError("diagnostic scheduler capture requires the already verified exact owned job")
    command = ["scontrol", "show", "job", "-d", job]
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=False, timeout=20)
        scheduler = {"command": command, "returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr}
    except subprocess.TimeoutExpired:
        scheduler = {"command": command, "status": "timeout"}
    except FileNotFoundError:
        scheduler = {"command": command, "status": "executable unavailable"}
    devices = []
    for path in sorted(Path("/dev").glob("nvidia*")):
        information = path.lstat()
        devices.append({"path": str(path), "character_device": stat.S_ISCHR(information.st_mode),
            "major": os.major(information.st_rdev) if stat.S_ISCHR(information.st_mode) else None,
            "minor": os.minor(information.st_rdev) if stat.S_ISCHR(information.st_mode) else None})
    config = Path("/etc/slurm/gres.conf")
    mapping = {"path": str(config), "status": "absent", "gpu_mapping_lines": []}
    if config.exists():
        try:
            lines = config.read_text().splitlines()
            mapping.update(status="readable", gpu_mapping_lines=[line for line in lines if line.strip()
                and not line.lstrip().startswith("#") and ("AutoDetect=" in line or re.search(r"\bName=gpu(?:\s|$)", line))])
        except PermissionError:
            mapping["status"] = "permission denied"
    return {"schema_version": 1, "scheduler": scheduler, "device_files": devices, "gpu_gres_mapping": mapping,
        "purpose": "diagnosis only; no alternative GPU mapping is accepted"}


def require_owned_job(run, environment, cgroup):
    """Require this campaign's UVA ledger before starting any numerical child."""
    job = environment.get("SLURM_JOB_ID", "")
    ledger = Path(run) / "slurm/submitted_jobids_uva.txt"
    lines = ledger.read_text().splitlines() if ledger.exists() else []
    if (not re.fullmatch(r"[1-9][0-9]*", job) or job not in lines
            or not re.search(r"(?:^|/)job_" + re.escape(job) + r"(?:/|$)", cgroup, re.MULTILINE)):
        raise ValueError("the H100 site validation requires its exact campaign-owned UVA Slurm job/cgroup")
    return {"ledger_path": str(ledger), "ledger_prefix": lines, "job_id": job, "cgroup": cgroup}


def require_selected_card(gpu):
    """Inspect only the allocated physical GPU; never query process owners or other direct-server cards."""
    modes = list(csv.reader(query(["nvidia-smi", "-i", gpu, "--query-gpu=mig.mode.current,memory.used", "--format=csv,noheader,nounits"]).splitlines()))
    processes = list(csv.reader(query(["nvidia-smi", "-i", gpu, "--query-compute-apps=pid,gpu_uuid", "--format=csv,noheader"]).splitlines()))
    confidential = query(["nvidia-smi", "conf-compute", "-i", gpu, "-f"])
    if (len(modes) != 1 or len(modes[0]) != 2 or modes[0][0].strip() != "Disabled" or int(modes[0][1]) >= 1024
            or processes or [line.split(":", 1)[1].strip() for line in confidential.splitlines() if "CC status:" in line] != ["OFF"]):
        raise ValueError("the allocated H100 must be whole, idle, below one GiB used and outside confidential-computing mode")
    return {"gpu_uuid": gpu, "mig_mode": "Disabled", "memory_used_mib": int(modes[0][1]), "compute_processes": [], "confidential_compute": "OFF"}


def submission_command(run, directory):
    """Build a held submission only; the parent durably records its exact ID before releasing it."""
    return ["sbatch", "--parsable", "--hold", "--account=sl5nw_base", "--partition=gpu", "--qos=normal", "--nodes=1", "--ntasks=1",
        "--constraint=h100_94gb&amd_epyc_9354", "--gres=gpu:nvidia_h100_nvl:1", "--cpus-per-task=8", "--mem=64G", "--time=03:00:00",
        "--job-name=atari_h100_nvl_site_validation", "--output=" + str(Path(run) / "slurm/jobs/h100_nvl_site_validation_%j.log"), str(Path(directory) / "job.sbatch")]


def prepare(arguments):
    """Create an immutable unsubmitted plan; do not reserve GPUs, claim cases or modify admission."""
    run, helper = arguments.run_dir.resolve(strict=True), arguments.helper_root.resolve(strict=True)
    directory = arguments.output_dir.resolve()
    if directory.exists() or directory.parent != run / "remote_setup":
        raise ValueError("prepare a fresh timestamped directory directly under this campaign's remote_setup")
    if not arguments.python.is_absolute() or not os.access(arguments.python, os.X_OK):
        raise ValueError("the site validation Python must name the existing shared approved interpreter")
    commit = query(["git", "-C", str(helper), "rev-parse", "HEAD"]).strip()
    profile_name = getattr(arguments, "qualification_profile", None)
    profile = None if profile_name is None else load("h100_nvl_slurm_qualification").PROFILE
    if profile is not None and profile_name != profile["name"]:
        raise ValueError("the requested Slurm qualification profile is unsupported")
    helpers = source_identity(helper, commit, [PREFIX + name for name in required_helpers(profile)])
    if Path(__file__).resolve() != helper / PREFIX / "h100_nvl_slurm_site_validation.py":
        raise ValueError("prepare through the exact detached committed helper checkout")
    admission = snapshot_admission(run)
    source = Path(admission["source_root"])
    numerical = source_identity(source, SOURCE_COMMIT, numerical_paths(source))
    cleanrl = cleanrl_identity(arguments.cleanrl_root)
    require_gate_controls(os.environ, cleanrl)
    reference = reference_identity(arguments.reference_directory, arguments.installed_backend, arguments.asset_receipt, arguments.software_result)
    configuration = original_configuration(source, admission)
    fresh_packages = package_versions(arguments.python)
    if fresh_packages != reference["software"]["packages"]:
        raise ValueError("the chosen interpreter differs from the actual completed shared-software packet")
    plan = {"schema_version": 2, "kind": "h100_nvl_slurm_site_validation", "campaign_run_dir": str(run),
        "source_commit": SOURCE_COMMIT, "helpers": helpers, "numerical_source": numerical, "admission": admission,
        "cleanrl_reference": cleanrl,
        "gate_environment_policy": GATE_ENVIRONMENT_POLICY,
        "reference": reference, "configuration": configuration,
        "runtime_environment": RUNTIME, "packages": reference["software"]["packages"], "fresh_packages_at_prepare": fresh_packages,
        "asset_record": bound(receipt(arguments.asset_record)), "jaxtari_source_root": str(arguments.jaxtari_source_root.resolve(strict=True)),
        "python": str(arguments.python.absolute()), "job_shape": JOB_SHAPE, "reference_updates": 1,
        "submission_control": SUBMISSION_CONTROL,
        "workers_permitted": False, "campaign_claims_permitted": False, "scientific_hypothesis_credit": 0,
        "created_at_utc": datetime.now(timezone.utc).isoformat()}
    if profile is not None:
        # Before: original driver 590 reference identity. After: same reference plus a separately named driver 580 target profile.
        plan.update(schema_version=3, qualification_profile=profile)
    directory.mkdir()
    plan["admission_archive"] = archive_admission(directory, admission)
    write_once(directory / "site_validation_plan.json", plan)
    executable = helper / PREFIX / "h100_nvl_slurm_site_validation.py"
    # Slurm may rewrite CUDA visibility when entering a step; resolve only inside that step.
    batch = ["#!/bin/bash", "set -euo pipefail", "source " + shlex.quote(str(source / PREFIX / "worker_environment.sh"))]
    batch += ["export " + name + "=" + shlex.quote(value) for name, value in RUNTIME.items()]
    batch.append("export CLEANRL_REFERENCE_FILE=" + shlex.quote(cleanrl["file"]["path"]))
    command = [str(arguments.python), str(executable), "run", "--directory", str(directory)]
    batch.append("exec srun --ntasks=1 --cpus-per-task=8 --cpu-bind=threads " + shlex.join(command))
    write_once(directory / "job.sbatch", "\n".join(batch) + "\n")
    submission = submission_command(run, directory)
    write_once(directory / "command.txt", shlex.join(submission) + "\n")
    write_once(directory / "config_resolved.yaml", {"configuration": plan["configuration"], "runtime": RUNTIME, "job_shape": JOB_SHAPE,
        **({"qualification_profile": profile} if profile is not None else {})})
    write_once(directory / "experiment_background.md", "## Purpose\n\nVerify one matching Slurm H100 NVL site before any separately reviewed worker placement. This site validation admits no workers and claims no campaign cases.\n\n## Key hyperparameters\n\nThe original one-policy reference uses 128 environments, 128 rollout steps, four epochs, four minibatches, fifty warmup rollouts and one complete training update. The original full learner and renderer gates run first in fresh sequential children.\n\n## Code and config changes\n\nKeep the frozen numerical source, highest float32 neural arithmetic, float64 statistics, deterministic operations and original allocator settings. Require the reference model, memory, backend and library identities, and the selected qualification driver. Resolve the allocated physical UUID inside srun and bind eight CPUs. Submit held. The parent must write and fsync the exact UVA job ID and submission receipt, then run scontrol release for only that saved ID. This helper never submits or releases a job.\n\n" + profile_background(profile) + "## Git state\n\nNumerical source: " + SOURCE_COMMIT + ". Metadata helpers: " + commit + ". Exact input identities are in site_validation_plan.json. No submission or current admission change is performed by preparation.\n")
    write_once(directory / "manifest.yaml", {"plan": receipt(directory / "site_validation_plan.json"), "job_script": receipt(directory / "job.sbatch"),
        "helpers": helpers, "state": "prepared_not_submitted", "submission_command": submission,
        "submission_control": SUBMISSION_CONTROL})
    return directory


def prepared(directory):
    """Rebuild every immutable plan input before a site validation, probe, or verification uses it."""
    directory = Path(directory).resolve(strict=True)
    manifest = read(directory / "manifest.yaml")
    plan = bound(manifest["plan"])
    profile = qualification_profile(plan)
    if (manifest["plan"]["path"] != str(directory / "site_validation_plan.json")
            or receipt(directory / "job.sbatch") != manifest["job_script"]
            or plan.get("schema_version") != (3 if profile is not None else 2) or plan.get("kind") != "h100_nvl_slurm_site_validation"
            or plan.get("source_commit") != SOURCE_COMMIT or plan.get("runtime_environment") != RUNTIME
            or plan.get("job_shape") != JOB_SHAPE or plan.get("reference_updates") != 1
            or plan.get("submission_control") != SUBMISSION_CONTROL
            or plan.get("gate_environment_policy") != GATE_ENVIRONMENT_POLICY
            or plan.get("workers_permitted") is not False or plan.get("campaign_claims_permitted") is not False
            or plan.get("scientific_hypothesis_credit") != 0):
        raise ValueError("the prepared site validation changed its source, runtime or no-worker scope")
    if Path(__file__).resolve() != Path(plan["helpers"]["source_root"]) / PREFIX / "h100_nvl_slurm_site_validation.py":
        raise ValueError("execute only the prepared committed site validation helper")
    verify_source_identity(plan["helpers"])
    verify_source_identity(plan["numerical_source"])
    if cleanrl_identity(plan["cleanrl_reference"]["source_root"]) != plan["cleanrl_reference"]:
        raise ValueError("the original CleanRL reference changed after preparation")
    if (set(plan["helpers"]["files"]) != {PREFIX + name for name in required_helpers(profile)}
            or plan["numerical_source"]["source_commit"] != SOURCE_COMMIT
            or plan["numerical_source"]["source_root"] != plan["admission"]["source_root"]
            or sorted(plan["numerical_source"]["files"]) != numerical_paths(plan["numerical_source"]["source_root"])):
        raise ValueError("the site plan omitted part of its exact helper or original numerical closure")
    for item in plan["admission"]["files"]:
        bound(item)
    validate_admission_archive(directory, plan["admission"], plan.get("admission_archive"))
    for name in ("hardware", "loaded_libraries", "diagnostic_plan", "dtype_record", "shared_installed_backend", "shared_asset_receipt"):
        bound(plan["reference"][name])
    if (plan["reference"]["driver"] != DRIVER or plan["reference"]["model"] != MODEL
            or plan["reference"]["memory_mib"] != MEMORY_MIB
            or plan["configuration"] != original_configuration(plan["numerical_source"]["source_root"], plan["admission"])):
        raise ValueError("the reference probe must retain the complete original fixed configuration")
    reference = plan["reference"]
    software = reference["software"]
    rebuilt = reference_identity(Path(reference["hardware"]["path"]).parent, reference["shared_installed_backend"]["path"],
        reference["shared_asset_receipt"]["path"], software["result"]["path"])
    if (rebuilt != reference or plan["packages"] != rebuilt["software"]["packages"]
            or plan["fresh_packages_at_prepare"] != rebuilt["software"]["packages"]):
        raise ValueError("the site's canonical software proof changed")
    return directory, plan


def placement(directory, plan):
    """Check a real owned Slurm allocation and record its original step fields before narrowing to a UUID."""
    require_runtime(os.environ)
    require_gate_controls(os.environ, plan["cleanrl_reference"])
    host = socket.gethostname().split(".")[0]
    owned = require_owned_job(plan["campaign_run_dir"], os.environ, Path("/proc/self/cgroup").read_text())
    csv_text = query(["nvidia-smi", "--query-gpu=index,uuid,name,memory.total,driver_version", "--format=csv,noheader,nounits"])
    xml_text = query(["nvidia-smi", "-q", "-x"])
    # Keep actual source observations even when the mapping rejects; never record broad environment prefixes.
    write_once(directory / "allocation_observation.json", {"schema_version": 1, "hostname": host,
        "allocation_environment": {name: os.environ.get(name) for name in ALLOCATION_KEYS},
        "absent_allocation_fields": [name for name in ALLOCATION_KEYS if name not in os.environ],
        "gpu_csv": csv_text, "gpu_xml": xml_text, "owned_job": owned,
        "cpu_affinity": sorted(os.sched_getaffinity(0)), "pid": os.getpid(),
        "process_start_time": process_start_time(os.getpid())})
    write_once(directory / "allocation_details.json", capture_allocation_details(owned))
    profile = qualification_profile(plan)
    if profile is None:
        proof = resolve_allocation(os.environ, host, csv_text, xml_text)
    else:
        qualifier = load("h100_nvl_slurm_qualification")
        observation = qualifier.capture(load("h100_nvl_allocation_observation"), os.environ, host, csv_text, xml_text,
            query, process_start_time(os.getpid()))
        write_once(directory / "qualification_observation.json", observation)
        proof = qualifier.validate_allocation(observation)
    if proof["driver"] != expected_driver(plan):
        raise ValueError("the allocated driver differs from the selected H100 qualification profile; stop before JAX")
    gpu = proof["gpu_uuid"]
    cpu = load("infrastructure_retry").cpu_allocation(os.environ, [gpu], os.sched_getaffinity(0))
    cpus = cpu["gpu_cpu_affinities"][gpu]
    os.sched_setaffinity(0, cpus)
    if sorted(os.sched_getaffinity(0)) != cpus or len(cpus) != 8:
        raise ValueError("the site validation did not obtain its exact eight-CPU affinity")
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    modes = require_selected_card(gpu)
    record = {"schema_version": 1, "allocation": proof, "owned_job": owned, "cpu_allocation": cpu,
        "cpu_affinity": cpus, "runtime_environment": require_runtime(os.environ), "gpu_modes": modes,
        "parent_pid": os.getpid(), "parent_start_time": process_start_time(os.getpid()),
        "step_environment": {name: os.environ[name] for name in ALLOCATION_KEYS if name in os.environ}}
    if profile is not None:
        record["qualification_observation"] = receipt(directory / "qualification_observation.json")
    write_once(directory / "site_assignment.json", record)
    return record


def verify_assignment(record, plan):
    """Check the complete saved site identity without relying on current global CUDA indices."""
    proof = record["allocation"]
    environment = {**proof["slurm_allocations"], "SLURM_JOB_ID": proof["job_id"],
                   "CUDA_VISIBLE_DEVICES": proof["cuda_visible_devices_original"]}
    profile = qualification_profile(plan)
    rebuilt = (resolve_allocation(environment, proof["hostname"], proof["gpu_csv"], proof["gpu_xml"]) if profile is None
        else load("h100_nvl_slurm_qualification").validate_allocation(bound(record["qualification_observation"])))
    if rebuilt != proof:
        raise ValueError("the site allocation record changed")
    if profile is not None:
        observation = bound(record["qualification_observation"])
        if observation["pid"] != record["parent_pid"] or observation["process_start_time"] != record["parent_start_time"]:
            raise ValueError("the qualification assignment substituted another parent process")
    if proof["driver"] != expected_driver(plan) or record["runtime_environment"] != RUNTIME:
        raise ValueError("the site driver or runtime differs from the selected qualification profile")
    gpu = proof["gpu_uuid"]
    cpu = record["cpu_allocation"]
    recomputed = load("infrastructure_retry").cpu_allocation({"SLURM_CPUS_PER_TASK": cpu["slurm_cpus_per_task"],
        "SLURM_CPU_BIND_TYPE": cpu["slurm_cpu_bind_type"], "SLURM_CPU_BIND_LIST": cpu["slurm_cpu_bind_list"],
        "SLURM_CPU_BIND": cpu["slurm_cpu_bind"]}, [gpu], cpu["parent_available_cpus"])
    if cpu != recomputed or record["cpu_affinity"] != cpu["gpu_cpu_affinities"][gpu] or len(record["cpu_affinity"]) != 8:
        raise ValueError("the site CPU allocation changed")
    owned = record["owned_job"]
    ledger = Path(plan["campaign_run_dir"]) / "slurm/submitted_jobids_uva.txt"
    current = ledger.read_text().splitlines()
    if (owned["ledger_path"] != str(ledger) or current[:len(owned["ledger_prefix"])] != owned["ledger_prefix"]
            or owned["job_id"] != proof["job_id"] or owned["job_id"] not in owned["ledger_prefix"]
            or not re.search(r"(?:^|/)job_" + re.escape(owned["job_id"]) + r"(?:/|$)", owned["cgroup"], re.MULTILINE)):
        raise ValueError("the site lacks its original append-only owned UVA allocation proof")
    expected_modes = {"gpu_uuid": gpu, "mig_mode": "Disabled", "compute_processes": [], "confidential_compute": "OFF"}
    if any(record["gpu_modes"].get(key) != value for key, value in expected_modes.items()) or not 0 <= record["gpu_modes"]["memory_used_mib"] < 1024:
        raise ValueError("the site did not prove an idle whole physical GPU")
    return proof


@contextmanager
def interrupted():
    """Route termination through exact-owned-child cleanup and restore the previous handlers."""
    previous = {name: signal.getsignal(name) for name in (signal.SIGTERM, signal.SIGINT)}
    def stop(signum, frame):
        raise InterruptedError("the H100 site validation received signal " + str(signum))
    try:
        for name in previous:
            signal.signal(name, stop)
        yield
    finally:
        for name, handler in previous.items():
            signal.signal(name, handler)


def stop_child(process, birth):
    """Terminate only the still-living process group started and birth-checked by this site validation."""
    # Before the first birth read succeeds, Popen's unreaped direct-child relationship
    # proves ownership; the child has not been released to import numerical code.
    if process.poll() is None and (birth is None or process_start_time(process.pid) == birth):
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            if process.poll() is None and (birth is None or process_start_time(process.pid) == birth):
                os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=15)


def child_command(plan, directory, kind):
    """Use a fresh recorded child; only original full gate paths or the fixed reference probe are supported."""
    helper = Path(plan["helpers"]["source_root"]) / PREFIX / "h100_nvl_slurm_site_validation.py"
    return [plan["python"], str(helper), "child", "--directory", str(directory), "--kind", kind]


def original_command(plan, kind):
    """Keep the original complete numerical tests unchanged and selector-free."""
    source = Path(plan["numerical_source"]["source_root"]) / "10_jax_exploration_platform"
    if kind == "learner":
        return [plan["python"], "-m", "pytest", "-q", str(source / "tests/atari_pixels"), str(source / "tests/test_atari_pixel_environment.py")]
    if kind == "renderer":
        return [plan["python"], str(source / "scripts/atari_pixel_campaign/check_rendering.py"), "--candidate", "all"]
    raise ValueError("the site validation only runs the original learner and renderer gates")


def run_child(plan, directory, kind, assignment):
    """Record a child PID and birth counter before releasing it to import numerical code."""
    process, birth = None, None
    with (directory / (kind + ".log")).open("x") as log:
        try:
            require_gate_controls(os.environ, plan["cleanrl_reference"])
            environment = {**os.environ, **RUNTIME, "CUDA_VISIBLE_DEVICES": assignment["allocation"]["gpu_uuid"],
                "XDG_DATA_HOME": plan["asset_record"]["root"],
                "CLEANRL_REFERENCE_FILE": plan["cleanrl_reference"]["file"]["path"],
                "PYTHONPATH": str(Path(plan["numerical_source"]["source_root"]) / "10_jax_exploration_platform/src")}
            process = subprocess.Popen(child_command(plan, directory, kind), cwd=plan["numerical_source"]["source_root"],
                env=environment, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            birth = process_start_time(process.pid)
            if birth is None:
                raise RuntimeError("the site child exited before its birth counter was recorded")
            owner = {"pid": process.pid, "process_start_time": birth, "parent_pid": os.getpid(),
                "parent_start_time": process_start_time(os.getpid()), "kind": kind, "job_id": assignment["allocation"]["job_id"]}
            write_once(directory / (kind + "_owner.json"), owner)
            write_once(directory / (kind + ".released"), "released\n")
            code = process.wait()
            if code != 0:
                raise RuntimeError("the original site " + kind + " child failed with exit code " + str(code))
            write_once(directory / (kind + "_terminal.json"), {**owner, "exit_code": code, "log": receipt(directory / (kind + ".log"))})
        except BaseException:
            if process is not None:
                stop_child(process, birth)
            raise


def child(directory, kind):
    """Recheck source, allocation and recorded release before the original gate or reference probe imports JAX."""
    directory, plan = prepared(directory)
    assignment = read(directory / "site_assignment.json")
    proof = verify_assignment(assignment, plan)
    require_owned_job(plan["campaign_run_dir"], os.environ, Path("/proc/self/cgroup").read_text())
    runtime = require_runtime(os.environ)
    require_gate_controls(os.environ, plan["cleanrl_reference"])
    if (str(Path(sys.executable).absolute()) != plan["python"] or os.environ.get("CUDA_VISIBLE_DEVICES") != proof["gpu_uuid"]
            or sorted(os.sched_getaffinity(0)) != assignment["cpu_affinity"]
            or any(os.environ.get(name) != value for name, value in proof["slurm_allocations"].items())):
        raise ValueError("the site child did not inherit its exact interpreter, physical allocation and eight CPUs")
    started = time.monotonic()
    while not (directory / (kind + ".released")).exists():
        if time.monotonic() - started > 60 or process_start_time(assignment["parent_pid"]) != assignment["parent_start_time"]:
            raise RuntimeError("the site parent did not record and release its exact child")
        time.sleep(.1)
    owner = read(directory / (kind + "_owner.json"))
    if (owner["pid"] != os.getpid() or owner["process_start_time"] != process_start_time(os.getpid())
            or owner["parent_pid"] != assignment["parent_pid"] or owner["parent_start_time"] != assignment["parent_start_time"]
            or owner["kind"] != kind or owner["job_id"] != proof["job_id"]):
        raise ValueError("the released site child has another process, parent or allocation")
    require_selected_card(proof["gpu_uuid"])
    if package_versions() != plan["packages"]:
        raise ValueError("the site validation interpreter's installed package versions changed")
    training = load("train_optimized")
    assets = training.verify_assets(plan["asset_record"], os.environ)
    installed = {"jaxtari_source": training.jaxtari_python_receipt(plan["jaxtari_source_root"]),
                 "gpu_libraries": training.installed_gpu_library_receipts()}
    installed["packages"] = package_versions()
    if installed != bound(plan["reference"]["shared_installed_backend"]) or assets != bound(plan["reference"]["shared_asset_receipt"]):
        raise ValueError("the site installed JAXtari/GPU images or assets differ from the captured shared environment")
    process = {**owner, "source_commit": SOURCE_COMMIT, "configuration": plan["configuration"],
        "runtime_environment": runtime, "gpu_uuid": proof["gpu_uuid"], "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "allocation_environment": {name: os.environ[name] for name in ALLOCATION_KEYS if name in os.environ},
        "cleanrl_reference": plan["cleanrl_reference"],
        "gate_environment_policy": GATE_ENVIRONMENT_POLICY,
        "asset_receipt": assets, "installed_backend": installed,
        "original_command": original_command(plan, kind) if kind != "probe" else None}
    if qualification_profile(plan) is not None:
        qualifier = load("h100_nvl_slurm_qualification")
        child_csv = query(["nvidia-smi", "-i", proof["gpu_uuid"],
            "--query-gpu=index,uuid,name,memory.total,driver_version", "--format=csv,noheader,nounits"])
        child_xml = query(["nvidia-smi", "-i", proof["gpu_uuid"], "-q", "-x"])
        child_observation = qualifier.capture(load("h100_nvl_allocation_observation"), os.environ, proof["hostname"],
            child_csv, child_xml, query, process_start_time(os.getpid()))
        child_allocation = qualifier.validate_allocation(child_observation)
        if any(child_allocation[name] != proof[name] for name in ("job_id", "step_id", "hostname", "gpu_uuid", "pci_address")):
            raise ValueError("the qualification child changed its actual job, step or physical GPU")
        process["qualification_observation"] = child_observation
    write_once(directory / (kind + "_process.json"), process)
    if kind != "probe":
        command = original_command(plan, kind)
        os.execv(command[0], command)
    return reference_probe(directory, plan, process, training)


def reference_probe(directory, plan, process, training):
    """Run one fixed original reference update solely to verify real dtypes and mapped libraries."""
    forbidden = ("controller", "ideas", "paired_benchmark", "exploration_platform", "jaxtari", "jax", "numpy")
    if any(name == prefix or name.startswith(prefix + ".") for name in sys.modules for prefix in forbidden):
        raise ValueError("the reference probe requires a fresh process without current or numerical modules")
    source = Path(plan["numerical_source"]["source_root"])
    sys.path.insert(0, str(source / "10_jax_exploration_platform/src"))
    import jax
    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_default_matmul_precision", "highest")
    devices = jax.devices()
    if len(devices) != 1 or devices[0].platform != "gpu" or devices[0].device_kind != MODEL:
        raise ValueError("the reference probe must see exactly the assigned whole H100 NVL")
    cuda_identity = None
    if qualification_profile(plan) is not None:
        qualifier = load("h100_nvl_slurm_qualification")
        allocation = qualifier.validate_allocation(process["qualification_observation"])
        cuda_identity = qualifier.validate_cuda_identity(qualifier.cuda_device_identity(devices[0]), allocation, process["pid"])
    from exploration_platform.atari_pixels.config import PixelPPOConfig
    from exploration_platform.atari_pixels.environment import AtariPixelEnvironment
    from exploration_platform.atari_pixels.training import PixelPPORND
    values = dict(plan["configuration"]["learner"])
    for name in ("per_copy_learning_rates", "per_copy_intrinsic_coefficients", "per_copy_scalar_settings"):
        values[name] = tuple(values[name])
    environment = AtariPixelEnvironment(**plan["configuration"]["environment"])
    trainer = PixelPPORND(PixelPPOConfig(**values), environment)
    state = trainer.initialize((1,))
    state = trainer.warmup(state)
    state, _ = trainer.train_iteration(state)
    jax.block_until_ready(state)
    floating = [leaf for leaf in jax.tree.leaves(state) if getattr(leaf, "dtype", None) is not None
                and leaf.dtype.kind in "fc"]
    import jax.numpy as jnp
    finite = {"passed": bool(jax.device_get(jnp.all(jnp.stack([jnp.all(jnp.isfinite(leaf)) for leaf in floating])))) if floating else False,
              "floating_leaves_checked": len(floating), "scope": "complete updated PixelTrainState"}
    if not finite["passed"] or finite["floating_leaves_checked"] < 1:
        raise ValueError("the updated full-size reference state contains non-finite floating values")
    groups = {"parameters": state.parameters, "target": state.target, "optimizer_first": state.optimizer.first,
        "optimizer_second": state.optimizer.second, "observation_statistics": state.observation_statistics,
        "reward_statistics": state.reward_statistics}
    dtypes = {name: sorted({str(leaf.dtype) for leaf in jax.tree.leaves(tree)}) for name, tree in groups.items()}
    if dtypes != EXPECTED_DTYPES or int(state.iteration) != 1:
        raise ValueError("the site reference did not preserve original numerical precision and one complete update")
    numerical_paths = {}
    for name, module in list(sys.modules.items()):
        if name == "exploration_platform" or name.startswith("exploration_platform.atari_pixels"):
            path = Path(module.__file__).resolve()
            if source / "10_jax_exploration_platform/src/exploration_platform" not in path.parents:
                raise ValueError("the site imported a different numerical implementation")
            numerical_paths[name] = str(path)
    installed = process["installed_backend"]
    distribution = metadata.distribution("jaxtari")
    if Path(sys.modules["jaxtari"].__file__).resolve() != Path(distribution.locate_file("jaxtari/__init__.py")).resolve():
        raise ValueError("the site imported a shadowed JAXtari package")
    loaded = training.actual_gpu_library_receipt(devices[0].client, installed["gpu_libraries"])
    validate_loaded(loaded, plan, installed)
    result = {"completed": True, "kind": "original_reference_site_probe", "scientific_hypothesis_credit": 0,
        "formal_measurement": False, "capacity_evidence": False, "source_commit": SOURCE_COMMIT,
        "pid": os.getpid(), "process_start_time": process_start_time(os.getpid()), "gpu_uuid": process["gpu_uuid"],
        "iteration": int(state.iteration), "dtypes": dtypes, "finite_state": finite,
        "loaded_libraries": loaded, "numerical_module_paths": numerical_paths}
    if cuda_identity is not None:
        result["qualification_cuda_identity"] = cuda_identity
        result["qualification_driver_image"] = capture_driver_image(cuda_identity, directory)
    write_once(directory / "probe_result.json", result)
    return result


def validate_loaded(loaded, plan, installed):
    """Require reference API versions and verified actual mapped image bytes without inventing old hashes."""
    if any(loaded.get(name) != value for name, value in plan["reference"]["backend"].items()):
        raise ValueError("the actual loaded backend/CUDA/cuDNN differs from the H100 reference")
    verified = loaded.get("verified_package_libraries", {})
    mapped = {path for path in loaded.get("loaded_shared_libraries", []) if Path(path).name.startswith(("libcudnn", "libcudart"))}
    if (not any(Path(path).name.startswith("libcudnn.so") for path in mapped)
            or not any(Path(path).name.startswith("libcudart.so") for path in mapped)
            or set(verified) != mapped or any(verified[path] != installed["gpu_libraries"].get(path) for path in mapped)
            or any(loaded.get(name) not in verified for name in ("cudnn_library", "cuda_runtime_library"))):
        raise ValueError("the site lacks actual mapped-library byte verification")
    for path in mapped:
        if sha256(path) != verified[path]["sha256"]:
            raise ValueError("a mapped site GPU image changed")
    drivers = [path for path in loaded.get("loaded_shared_libraries", []) if Path(path).name.startswith("libcuda.so")]
    if not drivers or any(not path.endswith("." + expected_driver(plan)) for path in drivers):
        raise ValueError("the actual loaded CUDA driver does not match the recorded H100 driver version")


def validate_numerical_paths(paths, source):
    """Require the real original model/state modules and reject every substituted package path."""
    required = {"exploration_platform.atari_pixels." + name for name in ("config", "environment", "training", "networks", "state")}
    if not required <= set(paths):
        raise ValueError("the site probe omitted its actual original numerical module identities")
    for name, path in paths.items():
        relative = Path(*name.split("."))
        suffix = "__init__.py" if name in ("exploration_platform", "exploration_platform.atari_pixels") else None
        expected = Path(source) / "10_jax_exploration_platform/src" / relative
        expected = expected / suffix if suffix else expected.with_suffix(".py")
        if path != str(expected):
            raise ValueError("the site probe imported a different numerical module path")


def run_site_validation(directory):
    """Run original gates and a separate reference probe on an owned allocation; admit and claim nothing."""
    directory, plan = prepared(directory)
    if (directory / "result.json").exists() or (directory / "failure.json").exists() or (directory / "site_assignment.json").exists():
        raise ValueError("a previous site validation is immutable; prepare a new explicitly reviewed launch")
    try:
        check_admission(plan["campaign_run_dir"], plan["admission"])
        assignment = placement(directory, plan)
        import fcntl
        with (Path("/tmp") / ("sl5nw_atari_pixel_benchmark_" + assignment["allocation"]["gpu_uuid"] + ".lock")).open("a+") as lock, interrupted():
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            require_selected_card(assignment["allocation"]["gpu_uuid"])
            for kind in ("learner", "renderer", "probe"):
                check_admission(plan["campaign_run_dir"], plan["admission"])
                run_child(plan, directory, kind, assignment)
            check_admission(plan["campaign_run_dir"], plan["admission"])
            result = {"schema_version": 1, "kind": "h100_nvl_slurm_site_evidence", "passed": True,
                "plan": receipt(directory / "site_validation_plan.json"), "assignment": receipt(directory / "site_assignment.json"),
                "children": {kind: {name: receipt(directory / (kind + suffix)) for name, suffix in
                    (("owner", "_owner.json"), ("process", "_process.json"), ("terminal", "_terminal.json"))}
                    for kind in ("learner", "renderer", "probe")}, "probe": receipt(directory / "probe_result.json"),
                "workers_permitted": False, "campaign_claims_permitted": False, "scientific_hypothesis_credit": 0,
                "completed_at_utc": datetime.now(timezone.utc).isoformat()}
            if qualification_profile(plan) is not None:
                result["qualification_profile"] = plan["qualification_profile"]
            # The public verifier rebuilds this exact proof; no source-ready or queue file is published.
            verify_result(directory, plan, result)
            write_once(directory / "result.json", result)
            return result
    except BaseException as error:
        write_once(directory / "failure.json", {"passed": False, "kind": "h100_nvl_slurm_site_failure",
            "error": str(error), "exception_type": type(error).__name__, "traceback": traceback.format_exc(),
            "campaign_claims_permitted": False, "scientific_hypothesis_credit": 0,
            "failed_at_utc": datetime.now(timezone.utc).isoformat()})
        raise


def verify_result(directory, plan, result):
    """Validate actual original child commands, completed work and fresh process/byte identities."""
    if (result.get("schema_version") != 1 or result.get("kind") != "h100_nvl_slurm_site_evidence" or result.get("passed") is not True
            or result.get("workers_permitted") is not False or result.get("campaign_claims_permitted") is not False
            or result.get("scientific_hypothesis_credit") != 0 or result.get("plan") != receipt(directory / "site_validation_plan.json")):
        raise ValueError("a site result cannot grant workers, source admission or scientific credit")
    if result.get("qualification_profile") != qualification_profile(plan):
        raise ValueError("the site result omitted or changed its explicit target qualification profile")
    if result["assignment"]["path"] != str(directory / "site_assignment.json") or result["probe"]["path"] != str(directory / "probe_result.json"):
        raise ValueError("a site result substituted another assignment or probe record")
    assignment = bound(result["assignment"])
    proof = verify_assignment(assignment, plan)
    processes = set()
    for kind in ("learner", "renderer", "probe"):
        records = result["children"][kind]
        owner, process, terminal = (bound(records[name]) for name in ("owner", "process", "terminal"))
        expected_paths = {"owner": directory / (kind + "_owner.json"), "process": directory / (kind + "_process.json"), "terminal": directory / (kind + "_terminal.json")}
        if any(records[name]["path"] != str(path) for name, path in expected_paths.items()):
            raise ValueError("a site result substituted another child record")
        identity = (owner.get("pid"), owner.get("process_start_time"))
        if (type(identity[0]) is not int or identity[0] < 1 or not str(identity[1]).isdigit() or identity in processes
                or terminal.get("exit_code") != 0 or terminal.get("log") != receipt(directory / (kind + ".log"))
                or any(process.get(key) != value or terminal.get(key) != value for key, value in owner.items())
                or owner.get("parent_pid") != assignment["parent_pid"] or owner.get("parent_start_time") != assignment["parent_start_time"]
                or owner.get("kind") != kind or owner.get("job_id") != proof["job_id"]
                or process.get("gpu_uuid") != proof["gpu_uuid"] or process.get("runtime_environment") != RUNTIME
                or process.get("cpu_affinity") != assignment["cpu_affinity"] or process.get("source_commit") != SOURCE_COMMIT
                or process.get("configuration") != plan["configuration"]):
            raise ValueError("site gates need distinct recorded successful children with the exact source/runtime/allocation")
        if qualification_profile(plan) is not None:
            child_proof = load("h100_nvl_slurm_qualification").validate_allocation(process.get("qualification_observation", {}))
            if (process["qualification_observation"]["pid"] != process["pid"]
                    or process["qualification_observation"]["process_start_time"] != process["process_start_time"]
                    or any(child_proof[name] != proof[name] for name in ("job_id", "step_id", "hostname", "gpu_uuid", "pci_address"))):
                raise ValueError("site child lacks its actual qualified process, cgroup and physical GPU identity")
        processes.add(identity)
        if process.get("original_command") != (original_command(plan, kind) if kind != "probe" else None):
            raise ValueError("the site gate did not execute the original full check command")
        if (process.get("cleanrl_reference") != plan["cleanrl_reference"]
                or process.get("gate_environment_policy") != GATE_ENVIRONMENT_POLICY):
            raise ValueError("the site gate changed the original CleanRL model reference")
        if (process["installed_backend"] != bound(plan["reference"]["shared_installed_backend"])
                or process["asset_receipt"] != bound(plan["reference"]["shared_asset_receipt"])):
            raise ValueError("the site child changed the pinned shared installation or game assets")
    probe = bound(result["probe"])
    process = bound(result["children"]["probe"]["process"])
    if (probe.get("kind") != "original_reference_site_probe" or probe.get("completed") is not True
            or probe.get("iteration") != 1 or probe.get("dtypes") != EXPECTED_DTYPES
            or any(probe.get(name) != process.get(name) for name in ("pid", "process_start_time", "gpu_uuid", "source_commit"))
            or probe.get("formal_measurement") is not False or probe.get("capacity_evidence") is not False
            or probe.get("scientific_hypothesis_credit") != 0 or probe["loaded_libraries"].get("pid") != process["pid"]):
        raise ValueError("the site probe lacks its actual one-update float32/float64 and same-process library proof")
    finite = probe.get("finite_state", {})
    if (finite.get("passed") is not True or type(finite.get("floating_leaves_checked")) is not int
            or finite["floating_leaves_checked"] < 1 or finite.get("scope") != "complete updated PixelTrainState"):
        raise ValueError("the site probe lacks a successful finite check of the complete updated state")
    validate_loaded(probe["loaded_libraries"], plan, process["installed_backend"])
    if qualification_profile(plan) is not None:
        identity = load("h100_nvl_slurm_qualification").validate_cuda_identity(probe.get("qualification_cuda_identity", {}), proof, process["pid"])
        if identity["driver_library"] not in probe["loaded_libraries"]["loaded_shared_libraries"]:
            raise ValueError("the actual JAX CUDA identity changed its mapped qualified driver image bytes")
        validate_driver_capture(probe.get("qualification_driver_image", {}), identity, directory, process["pid"])
    validate_numerical_paths(probe.get("numerical_module_paths", {}), plan["numerical_source"]["source_root"])
    return {"passed": True, "scope": "one exact allocated H100 NVL site only", "hostname": proof["hostname"],
        "gpu_uuid": proof["gpu_uuid"], "job_id": proof["job_id"], "workers_permitted": False,
        "campaign_claims_permitted": False, "scientific_hypothesis_credit": 0,
        **({"qualification_profile": plan["qualification_profile"]} if qualification_profile(plan) is not None else {})}


def verify(directory):
    """Rebuild completed site evidence under the current admission lock without publishing placement permission."""
    directory, plan = prepared(directory)
    if (directory / "failure.json").exists():
        raise ValueError("the site validation retained a failure")
    check_admission(plan["campaign_run_dir"], plan["admission"])
    result = verify_result(directory, plan, read(directory / "result.json"))
    check_admission(plan["campaign_run_dir"], plan["admission"])
    return result


def main():
    """Prepare, run a future allocated site validation, or verify evidence; submit and claim nothing."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    preparation = commands.add_parser("prepare")
    preparation.add_argument("--qualification-profile", choices=("slurm-configured-device-isolation-driver-580.159.04",))
    for name in ("run-dir", "helper-root", "reference-directory", "software-result", "installed-backend", "asset-receipt", "asset-record", "jaxtari-source-root", "cleanrl-root", "output-dir", "python"):
        preparation.add_argument("--" + name, type=Path, required=True)
    for name in ("run", "verify", "child"):
        command = commands.add_parser(name)
        command.add_argument("--directory", type=Path, required=True)
        if name == "child":
            command.add_argument("--kind", choices=("learner", "renderer", "probe"), required=True)
    arguments = parser.parse_args()
    if arguments.command == "prepare":
        result = {"prepared_directory": str(prepare(arguments)), "submitted": False}
    elif arguments.command == "run":
        result = run_site_validation(arguments.directory)
    elif arguments.command == "verify":
        result = verify(arguments.directory)
    else:
        result = child(arguments.directory, arguments.kind)
    print(json.dumps(result, indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
