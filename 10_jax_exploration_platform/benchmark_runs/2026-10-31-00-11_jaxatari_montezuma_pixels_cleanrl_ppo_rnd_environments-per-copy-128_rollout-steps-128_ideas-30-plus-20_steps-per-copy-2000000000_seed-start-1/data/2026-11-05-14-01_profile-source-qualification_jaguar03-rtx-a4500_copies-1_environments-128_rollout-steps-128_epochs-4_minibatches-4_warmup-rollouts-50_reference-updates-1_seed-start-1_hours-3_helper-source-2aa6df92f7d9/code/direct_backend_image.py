"""Pin and capture the actual direct worker's installed XLA backend package image.

All operations are standard-library file/metadata operations. No numerical
package is imported, no CUDA entry point or device is used, and no shared library is loaded.
The existing CUDA/cuDNN/runtime receipts keep their original separate meaning.
"""
import argparse
import base64
import csv
from email.parser import BytesParser
import hashlib
from importlib import metadata
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import socket
import stat
import sys
import time

NAME = "direct_backend_image.py"
PREFIX = "10_jax_exploration_platform/scripts/atari_pixel_campaign/"
CAPABILITY = "prepared_owning_package_image_and_same_process_mapping_version_2"
CARD_ONE = "GPU-fed69954-5a71-dbb2-ee91-544e0eb6b8a3"
NUMERICAL = {"jax", "jaxlib", "numpy", "torch", "flax", "optax", "cupy", "scipy", "jaxtari", "exploration_platform"}
SMALL_FILE_LIMIT = 1024 * 1024


def require(value, message):
    """Reject missing or contradictory package-image evidence."""
    if not value:
        raise ValueError(message)


def encoded(value):
    """Encode strict stable metadata without nonfinite values."""
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def digest(raw):
    """Hash one exact byte sequence."""
    return hashlib.sha256(raw).hexdigest()


def ordinary(path, *, directory=False):
    """Refuse path aliases and symlinks before a file or directory is used."""
    path = Path(path)
    require(path.is_absolute() and path.resolve(strict=True) == path, "one exact ordinary absolute path is required")
    information = path.lstat()
    require(stat.S_ISDIR(information.st_mode) if directory else stat.S_ISREG(information.st_mode),
        "an ordinary directory" if directory else "an ordinary regular file" + " is required")
    return path


def file_identity(information):
    """Record filesystem identity, never a GPU minor or scheduler selector."""
    return {"device_major": os.major(information.st_dev), "device_minor": os.minor(information.st_dev),
        "inode": information.st_ino}


def stream_image(path, target=None, maximum_bytes=None):
    """Hash or copy one pinned ordinary descriptor and reject changes during its read."""
    path = ordinary(path)
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    output = None
    value, count = hashlib.sha256(), 0
    try:
        before = os.fstat(descriptor)
        require(stat.S_ISREG(before.st_mode) and before.st_size > 0
            and (maximum_bytes is None or before.st_size <= maximum_bytes), "nonempty bounded ordinary image required")
        if target is not None:
            target = Path(target)
            ordinary(target.parent, directory=True)
            output = os.fdopen(os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600), "wb")
        # Keep large images out of memory and retain the exact source descriptor.
        while block := os.read(descriptor, 1024 * 1024):
            value.update(block)
            count += len(block)
            require(maximum_bytes is None or count <= maximum_bytes, "source exceeded its planned copy size")
            if output is not None:
                output.write(block)
        ordinary(path)
        after, current = os.fstat(descriptor), path.lstat()
        fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns", "st_mode", "st_uid")
        require(all(getattr(before, key) == getattr(after, key) == getattr(current, key) for key in fields)
            and count == before.st_size, "file changed while its descriptor was being read")
        if output is not None:
            output.flush()
            os.fsync(output.fileno())
    finally:
        os.close(descriptor)
        if output is not None:
            output.close()
    if target is not None:
        synchronize(target.parent)
    fields = lambda info: {"file_identity": file_identity(info), "size_bytes": info.st_size,
        "modification_time_ns": info.st_mtime_ns, "change_time_ns": info.st_ctime_ns,
        "uid": info.st_uid, "mode": stat.S_IMODE(info.st_mode), "regular_file": stat.S_ISREG(info.st_mode)}
    return {"path": str(path), "sha256": value.hexdigest(), "size_bytes": count,
        "file_identity": file_identity(before), "descriptor_before": fields(before), "descriptor_after": fields(after)}


def small_bytes(path):
    """Read bounded metadata through a pinned ordinary descriptor."""
    path = ordinary(path)
    require(path.stat().st_size <= SMALL_FILE_LIMIT, "metadata file exceeds its explicit bound")
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        before = os.fstat(descriptor)
        require(stat.S_ISREG(before.st_mode) and before.st_size <= SMALL_FILE_LIMIT,
            "one bounded ordinary metadata descriptor is required before reading")
        raw = os.read(descriptor, SMALL_FILE_LIMIT + 1)
        after, current = os.fstat(descriptor), path.lstat()
        fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns", "st_mode", "st_uid")
        require(len(raw) <= SMALL_FILE_LIMIT and len(raw) == before.st_size
            and all(getattr(before, key) == getattr(after, key) == getattr(current, key) for key in fields),
            "metadata file changed or its read was incomplete")
    finally:
        os.close(descriptor)
    return raw


def document(path):
    """Read finite unique-key JSON metadata without importing numerical code."""
    def unique(pairs):
        """Reject repeated metadata keys."""
        value = {}
        for key, item in pairs:
            require(key not in value, "duplicate backend-image metadata key")
            value[key] = item
        return value
    return json.loads(small_bytes(path), object_pairs_hook=unique,
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError("nonfinite backend-image metadata")))


def reference(path):
    """Bind a small exact metadata file to its bytes."""
    return {"path": str(ordinary(path)), "sha256": digest(small_bytes(path))}


def checked_reference(value):
    """Refuse a changed or substituted small metadata reference."""
    require(type(value) is dict and set(value) == {"path", "sha256"} and reference(value["path"]) == value,
        "backend-image metadata path or bytes changed")
    return document(value["path"])


def synchronize(directory):
    """Durably record completed file publication in its ordinary parent directory."""
    descriptor = os.open(ordinary(directory, directory=True), os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def publish_bytes(path, raw):
    """Publish exact new metadata bytes without replacing another attempt."""
    path = Path(path)
    ordinary(path.parent, directory=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    synchronize(path.parent)
    return reference(path)


def publish(path, value):
    """Publish one strict JSON record without replacing another attempt."""
    return publish_bytes(path, encoded(value))


def record_entry(raw, relative, image):
    """Rebuild the exact wheel RECORD hash and byte count for this package image."""
    rows = list(csv.reader(raw.decode("utf-8", errors="strict").splitlines(), strict=True))
    require(all(len(row) == 3 for row in rows), "malformed owning distribution RECORD")
    found = [row for row in rows if row[0] == relative]
    require(len(found) == 1, "the owning RECORD needs one exact backend image entry")
    path, value, size = found[0]
    require(value.startswith("sha256=") and re.fullmatch(r"[A-Za-z0-9_-]{43}", value[7:])
        and re.fullmatch(r"[1-9][0-9]*", size), "the backend RECORD entry needs SHA256 and an exact byte count")
    binary_hash = base64.urlsafe_b64decode(value[7:] + "=").hex()
    require(binary_hash == image["sha256"] and int(size) == image["size_bytes"],
        "installed image differs from its owning distribution RECORD")
    return {"relative_path": path, "hash": value, "size_bytes": int(size)}


def package_headers(raw):
    """Require one unambiguous actual package name and version from METADATA."""
    value = BytesParser().parsebytes(raw)
    names, versions = value.get_all("Name", []), value.get_all("Version", [])
    require(len(names) == len(versions) == 1 and names[0].strip() and versions[0].strip(),
        "owning package METADATA needs one name and one version")
    return {"name": names[0].strip(), "version": versions[0].strip()}


def describe_image(site_packages, image_path, *, retain_descriptor=False):
    """Discover the unique real distribution owner and hash its actual image and records."""
    site_packages, image_path = ordinary(site_packages, directory=True), ordinary(image_path)
    require(site_packages.name == "site-packages" and image_path.is_relative_to(site_packages)
        and image_path.name == "xla_cuda_plugin.so", "select the exact installed XLA backend image")
    image = stream_image(image_path)
    owners = []
    # Ownership comes from actual installed file rosters, never from an assumed package name.
    for distribution in metadata.distributions(path=[str(site_packages)]):
        files = distribution.files
        if files is None:
            continue
        matches = [entry for entry in files if Path(str(entry)).name == image_path.name
            and Path(distribution.locate_file(entry)).resolve(strict=True) == image_path]
        for entry in matches:
            records = [entry for entry in files if Path(str(entry)).name == "RECORD"
                and Path(str(entry)).parent.name.endswith(".dist-info")]
            descriptions = [entry for entry in files if Path(str(entry)).name == "METADATA"
                and Path(str(entry)).parent.name.endswith(".dist-info")]
            require(len(records) == len(descriptions) == 1, "one owning distribution RECORD/METADATA pair is required")
            record_path = ordinary(Path(distribution.locate_file(records[0])))
            metadata_path = ordinary(Path(distribution.locate_file(descriptions[0])))
            require(record_path.is_relative_to(site_packages) and metadata_path.parent == record_path.parent,
                "owning distribution records must stay in the selected site-packages directory")
            relative = str(entry)
            require(not PurePosixPath(relative).is_absolute() and ".." not in PurePosixPath(relative).parts
                and str(PurePosixPath(relative)) == relative, "backend RECORD image identity must be canonical and relative")
            headers = package_headers(small_bytes(metadata_path))
            require(distribution.version == headers["version"], "distribution version changed during discovery")
            owners.append({**headers, "metadata": reference(metadata_path), "record": reference(record_path),
                "record_entry": record_entry(small_bytes(record_path), relative, image)})
    require(len(owners) == 1, "the actual installed backend image must have exactly one distribution owner")
    described = {"site_packages": str(site_packages),
        "image": {key: image[key] for key in ("path", "sha256", "size_bytes")}, "distribution": owners[0],
        "observed_file_identity": image["file_identity"]}
    if retain_descriptor:
        described.update(observed_descriptor_before=image["descriptor_before"],
            observed_descriptor_after=image["descriptor_after"])
    return described


def own_process():
    """Record this actual own process and all four kernel UIDs without foreign process reads."""
    status = [line for line in Path("/proc/self/status").read_text().splitlines() if line.startswith("Uid:")]
    raw = Path("/proc/self/stat").read_text()
    require(len(status) == 1 and list(map(int, status[0].split()[1:])) == [2618919] * 4
        and os.getuid() == os.geteuid() == 2618919, "the exact direct-study owner is required")
    return {"hostname": socket.gethostname().split(".")[0], "pid": os.getpid(),
        "process_start_time": raw.rsplit(")", 1)[1].split()[19], "uid": 2618919,
        "uid_bytes": status[0] + "\n", "uid_sha256": digest((status[0] + "\n").encode())}


def prepare_image(python, site_packages, image_path):
    """Create actual pre-GPU package-image metadata using only standard-library file reads."""
    require(not NUMERICAL.intersection(sys.modules), "package-image preparation cannot import numerical packages")
    python = Path(python).absolute()
    site_packages = ordinary(site_packages, directory=True)
    require(python.is_file() and python.parent.name == "bin"
        and site_packages.is_relative_to(python.parent.parent / "lib"), "select the exact target Python environment")
    described = describe_image(site_packages, image_path)
    return {"schema_version": 2, "kind": "prepared_direct_backend_package_image", "capability": CAPABILITY,
        "selected_python": str(python), **described, "created_at_seconds": time.time(),
        "producer_source_sha256": digest(Path(__file__).read_bytes()), "observer": own_process(),
        "observing_python": str(Path(sys.executable).absolute()), "numerical_imports": [], "gpu_operations": 0}


def validate_expected(value, python=None):
    """Require the complete immutable pre-GPU package-image protocol, never a legacy record."""
    require(type(value) is dict and set(value) == {"schema_version", "kind", "capability", "selected_python",
        "site_packages", "image", "distribution", "observed_file_identity", "created_at_seconds",
        "producer_source_sha256", "observer", "observing_python", "numerical_imports", "gpu_operations"}
        and type(value["schema_version"]) is int and value["schema_version"] == 2
        and value["kind"] == "prepared_direct_backend_package_image" and value["capability"] == CAPABILITY
        and value["numerical_imports"] == [] and type(value["gpu_operations"]) is int and value["gpu_operations"] == 0,
        "new direct execution requires its prepared backend package-image schema two")
    require(python is None or value["selected_python"] == python, "prepared backend belongs to another Python environment")
    require(type(value["created_at_seconds"]) in {int, float} and math.isfinite(value["created_at_seconds"])
        and value["created_at_seconds"] > 0 and value["producer_source_sha256"] == digest(Path(__file__).read_bytes()),
        "package-image preparation time or exact producer source changed")
    image, owner = value["image"], value["distribution"]
    require(set(image) == {"path", "sha256", "size_bytes"} and type(image["path"]) is str
        and Path(image["path"]).is_absolute() and Path(image["path"]).name == "xla_cuda_plugin.so"
        and re.fullmatch(r"[0-9a-f]{64}", image["sha256"]) and type(image["size_bytes"]) is int and image["size_bytes"] > 0,
        "prepared package image needs an exact path, SHA256 and byte count")
    require(set(owner) == {"name", "version", "metadata", "record", "record_entry"}
        and type(owner["name"]) is str and owner["name"] and type(owner["version"]) is str and owner["version"],
        "prepared package distribution identity is incomplete")
    identity = value["observed_file_identity"]
    require(type(identity) is dict and set(identity) == {"device_major", "device_minor", "inode"}
        and all(type(item) is int and item >= 0 for item in identity.values()) and identity["inode"] > 0,
        "preparation filesystem identity is incomplete")
    observer = value["observer"]
    require(type(observer["pid"]) is int and observer["pid"] > 0 and observer["uid"] == 2618919
        and re.fullmatch(r"[1-9][0-9]*", observer["process_start_time"])
        and observer["uid_sha256"] == digest(observer["uid_bytes"].encode())
        and list(map(int, observer["uid_bytes"].split()[1:])) == [2618919] * 4,
        "preparation lacks its actual own UID/PID/birth")
    return value


def plan_path(plan, directory):
    """Keep selected measurements and the separately declared profile window distinct."""
    selected = plan.get("kind") == "selected_configuration_measurement" and plan.get("mode") in {"correctness", "confirmation"}
    require(selected or profile_plan(plan), "this backend protocol requires a selected measurement or its explicit direct profile")
    return Path(directory) / "measurement_plan.json"


def profile_plan(plan):
    """Recognize only the new direct profile, never a historical or configured profile."""
    return (plan.get("kind") == "selected_configuration_profile" and plan.get("mode") in {None, "profile"}
        and plan.get("direct_profile_protocol") == "direct-h100-reviewed-common-profile-version-1"
        and "direct_binding" in plan and "configured_isolation" not in plan)


def configure_plan(plan, record_path):
    """Copy one actual prepared record into the immutable unit and bind its exact real bytes."""
    expected = checked_reference(reference(record_path))
    validate_expected(expected, plan["python"])
    directory = Path(plan["campaign_run_dir"]) / "data" / plan["unit_id"]
    target = directory / "direct_backend_image_expected.json"
    raw = small_bytes(record_path)
    if target.exists():
        require(small_bytes(target) == raw, "preserve the prior different prepared backend image")
    else:
        publish_bytes(target, raw)
        require(small_bytes(target) == raw, "prepared record copy changed its bytes")
    plan["direct_backend_image"] = {"schema_version": 2, "capability": CAPABILITY, "expected": reference(target)}
    plan["inputs"]["direct_backend_image_record"] = reference(record_path)
    return plan


def expected_for_plan(plan):
    """Require the exact staged new helper and current plan's immutable expected-image sidecar."""
    require("direct_binding" in plan and plan.get("hardware_class") == "h100" and plan.get("gpu_uuid") == CARD_ONE
        and "configured_isolation" not in plan, "backend package-image proof is only for direct serval02 card one")
    value = plan.get("direct_backend_image")
    directory = Path(plan["campaign_run_dir"]) / "data" / plan["unit_id"]
    require(type(value) is dict and set(value) == {"schema_version", "capability", "expected"}
        and type(value["schema_version"]) is int and value["schema_version"] == 2 and value["capability"] == CAPABILITY
        and value["expected"]["path"] == str(directory / "direct_backend_image_expected.json"),
        "new direct operational readers refuse missing or legacy backend package-image evidence")
    require(plan["helpers"]["files"].get(PREFIX + NAME) == digest(Path(__file__).read_bytes()),
        "the exact new backend helper is not in the staged source closure")
    return validate_expected(checked_reference(value["expected"]), plan["python"])


def validate_prepared(plan, *, retain_actual=False):
    """Recheck actual installed image and owning metadata before numerical initialization."""
    expected = expected_for_plan(plan)
    actual = describe_image(expected["site_packages"], expected["image"]["path"], retain_descriptor=retain_actual)
    require(all(actual[key] == expected[key] for key in ("site_packages", "image", "distribution")),
        "installed backend package image or distribution metadata changed after its pre-GPU pin")
    if "qualification_software" in plan:
        versions = plan["qualification_software"]["packages"]
    else:
        require("direct_qualification_inputs" in plan, "new direct backend must retain actual full-original software qualification")
        result_path = Path(plan["direct_qualification_inputs"]["result"]["path"])
        qualified = document(result_path.parent / "measurement_plan.json")
        versions = qualified["qualification_software"]["packages"]
    normalized = {re.sub(r"[-_.]+", "-", name).lower(): value for name, value in versions.items()}
    require(len(normalized) == len(versions) and normalized.get(re.sub(r"[-_.]+", "-", expected["distribution"]["name"]).lower()) ==
        expected["distribution"]["version"], "backend owner/version differs from the unchanged original software gates")
    return (expected, actual) if retain_actual else expected


def mapped_snapshot(image_path):
    """Record this process's actual mapped filesystem image, without touching a GPU."""
    with Path("/proc/self/maps").open("rb") as stream:
        raw = stream.read(SMALL_FILE_LIMIT + 1)
    require(len(raw) <= SMALL_FILE_LIMIT, "own mapping metadata exceeds its read limit")
    raw = raw.decode("utf-8", errors="strict")
    selected = "".join(line + "\n" for line in raw.splitlines()
        if len(line.split(None, 5)) == 6
        and Path(line.split(None, 5)[5].removesuffix(" (deleted)")).name == "xla_cuda_plugin.so")
    rows = mapping_rows(selected, image_path)
    return {"path": "/proc/self/maps", "raw": selected, "sha256": digest(selected.encode()),
        "file_identity": rows[0], "observed_at_seconds": time.time()}


def mapping_rows(raw, image_path):
    """Parse mapped filesystem inode/device fields without equating them to GPU identifiers."""
    rows = []
    for line in raw.splitlines():
        parts = line.split(None, 5)
        require(len(parts) == 6 and parts[5] == image_path
            and re.fullmatch(r"[0-9a-f]+-[0-9a-f]+", parts[0])
            and re.fullmatch(r"[r-][w-][x-][ps]", parts[1])
            and re.fullmatch(r"[0-9a-f]+", parts[2]) and re.fullmatch(r"[0-9a-f]+:[0-9a-f]+", parts[3])
            and re.fullmatch(r"[1-9][0-9]*", parts[4]), "malformed actual backend mapping row")
        major, minor = parts[3].split(":")
        rows.append({"device_major": int(major, 16), "device_minor": int(minor, 16), "inode": int(parts[4])})
    require(rows and all(row == rows[0] for row in rows), "one mapped backend filesystem inode is required")
    return rows


def binding_process(plan, binding):
    """Join the actual current own process to the already resolved native CUDA binding."""
    current = own_process()
    process = binding["process"]
    require(current["hostname"] == "serval02" and binding["gpu_uuid"] == CARD_ONE
        and all(current[key] == process[key] for key in ("hostname", "pid", "process_start_time", "uid"))
        and binding["native"]["pid"] == current["pid"], "backend capture is not this actual bound direct process")
    return current


def captured_reference(target, source, expected, maximum_bytes):
    """Copy one pinned source image/record and verify its exact planned bytes."""
    value = stream_image(source, target, maximum_bytes)
    require(value["sha256"] == expected["sha256"], "captured backend/package metadata differs from its pre-GPU pin")
    return {"original": expected, "captured": {"path": str(target), "sha256": value["sha256"]},
        "size_bytes": value["size_bytes"], "source_file_identity": value["file_identity"],
        "descriptor_before": value["descriptor_before"], "descriptor_after": value["descriptor_after"]}


def capture_begin(directory, plan, binding):
    """Preserve the current mapped package image and metadata before controlled learner work."""
    directory = Path(directory)
    require(not any((directory / name).exists() for name in ("direct_backend_capture_begin.json",
        "direct_backend_capture_attempt.json", "direct_backend_capture.json")), "preserve this backend capture attempt; do not retry")
    expected = validate_prepared(plan)
    process = binding_process(plan, binding)
    publish(directory / "direct_backend_capture_attempt.json", {"schema_version": 2, "capability": CAPABILITY,
        "process": process, "expected": plan["direct_backend_image"]["expected"], "retry_permitted": False})
    try:
        before = mapped_snapshot(expected["image"]["path"])
        target = directory / "direct_backend"
        target.mkdir()
        image = captured_reference(target / "xla_cuda_plugin.so", expected["image"]["path"],
            {key: expected["image"][key] for key in ("path", "sha256")}, expected["image"]["size_bytes"])
        metadata_copy = captured_reference(target / "METADATA", expected["distribution"]["metadata"]["path"],
            expected["distribution"]["metadata"], SMALL_FILE_LIMIT)
        record_copy = captured_reference(target / "RECORD", expected["distribution"]["record"]["path"],
            expected["distribution"]["record"], SMALL_FILE_LIMIT)
        after = mapped_snapshot(expected["image"]["path"])
        require(before["file_identity"] == after["file_identity"] == image["source_file_identity"]
            and image["size_bytes"] == expected["image"]["size_bytes"], "actual mapped inode changed during shared capture")
        value = {"schema_version": 2, "kind": "direct_current_process_backend_package_capture_begin", "capability": CAPABILITY,
            "plan": reference(plan_path(plan, directory)), "expected": plan["direct_backend_image"]["expected"],
            "binding": reference(directory / "direct_device_binding.json"), "process": process,
            "maps_before_copy": before, "maps_after_copy": after, "image": image,
            "metadata": metadata_copy, "record": record_copy}
        publish(directory / "direct_backend_capture_begin.json", value)
        return value
    except BaseException as error:
        publish(directory / "direct_backend_capture_failure.json", {"schema_version": 2,
            "error_type": type(error).__name__, "error": str(error), "measurement_valid": False})
        raise


def capture_complete(directory, plan, result):
    """Record the same mapped package image after actual numerical work, outside measured pairs."""
    directory = Path(directory)
    package_started = time.time()
    expected, package = validate_prepared(plan, retain_actual=True)
    package_finished = time.time()
    begin = checked_reference(reference(directory / "direct_backend_capture_begin.json"))
    binding = checked_reference(reference(directory / "direct_device_binding.json"))
    process = binding_process(plan, binding)
    after = mapped_snapshot(expected["image"]["path"])
    require(process == begin["process"] and after["file_identity"] == begin["image"]["source_file_identity"]
        and result["loaded_gpu_libraries"]["pid"] == process["pid"], "post-work backend process or mapped image changed")
    completed_work(plan, result)
    value = {"schema_version": 2, "kind": "direct_current_process_captured_backend_package_image", "capability": CAPABILITY,
        "plan": reference(plan_path(plan, directory)), "expected": plan["direct_backend_image"]["expected"],
        "binding": reference(directory / "direct_device_binding.json"), "begin": reference(directory / "direct_backend_capture_begin.json"),
        "process": process, "maps_after_numerical_work": after,
        "post_installed_package": {"started_at_seconds": package_started,
            "finished_at_seconds": package_finished, "description": package},
        "loaded_gpu_libraries_sha256": digest(encoded(result["loaded_gpu_libraries"])),
        "numerical_execution": result["execution_identity"],
        "completed_work_kind": completed_work(plan, result)}
    publish(directory / "direct_backend_capture.json", value)
    validate_capture(directory, plan, binding, result["loaded_gpu_libraries"], result)
    return value


def completed_work(plan, result):
    """Require actual completed work before publishing or reading a backend completion."""
    if profile_plan(plan):
        updates = plan.get("profile_updates")
        start, finish = result.get("profile_start_iteration"), result.get("profile_end_iteration")
        require(type(updates) is int and updates > 0 and type(start) is int and start >= 0
            and type(finish) is int and finish - start == updates and result.get("profile_updates") == updates
            and type(result.get("profile_updates")) is int and result.get("capture_completed") is True
            and type(result.get("normalization_warmup_iterations")) is int
            and result["normalization_warmup_iterations"] == 50
            and type(result.get("compiled_signatures_before")) is list
            and result["compiled_signatures_before"] == result.get("compiled_signatures_after")
            and result.get("configuration") == plan["configuration"]
            and result.get("copy_seeds") == plan["copy_seeds"]
            and result.get("actual_runtime_environment") == plan["runtime_environment"],
            "backend profile completion needs the entire unchanged compiled full-update window")
        return "complete_selected_profile_window"
    if "direct_qualification" in plan:
        require(result.get("all_original_gates_passed") is True and type(result.get("reference_iteration")) is int
            and result["reference_iteration"] == 1 and type(result.get("reference_optimizer_step")) is int
            and result["reference_optimizer_step"] == 16,
            "backend completion cannot precede the complete original reference update")
    elif plan["mode"] == "correctness":
        require(result.get("correctness_passed") is True and result.get("correctness_attempted") is True,
            "backend completion cannot precede actual correctness work")
    else:
        require(len(result.get("rounds", [])) == 7, "backend completion cannot precede all actual intermediate pairs")
    return "full_original_qualification" if "direct_qualification" in plan else plan["mode"]


def validate_capture(directory, plan, binding, loaded, result):
    """Require new planned, copied and post-work mapping evidence; refuse all legacy receipts."""
    directory = Path(directory)
    expected = expected_for_plan(plan)
    begin = checked_reference(reference(directory / "direct_backend_capture_begin.json"))
    final = checked_reference(reference(directory / "direct_backend_capture.json"))
    require(type(begin.get("schema_version")) is type(final.get("schema_version")) is int
        and begin["schema_version"] == final["schema_version"] == 2
        and begin.get("kind") == "direct_current_process_backend_package_capture_begin"
        and final.get("kind") == "direct_current_process_captured_backend_package_image"
        and begin.get("capability") == final.get("capability") == CAPABILITY,
        "new direct backend reader refuses missing/legacy begin or completion evidence")
    process = binding["process"]
    for value in (begin, final):
        require(value["plan"] == reference(plan_path(plan, directory))
            and value["expected"] == plan["direct_backend_image"]["expected"]
            and value["binding"] == reference(directory / "direct_device_binding.json")
            and all(value["process"][key] == process[key] for key in ("hostname", "pid", "process_start_time", "uid")),
            "actual backend proof changed its plan, expected image, native binding or own process")
    require(final["begin"] == reference(directory / "direct_backend_capture_begin.json")
        and final["process"] == begin["process"] and loaded["pid"] == process["pid"]
        and final["completed_work_kind"] == completed_work(plan, result)
        and final["loaded_gpu_libraries_sha256"] == digest(encoded(loaded)), "post-work proof changed its separate CUDA receipt or initial capture")
    image = begin["image"]
    wanted = directory / "direct_backend/xla_cuda_plugin.so"
    require(image["original"] == {key: expected["image"][key] for key in ("path", "sha256")}
        and image["captured"] == {"path": str(wanted), "sha256": expected["image"]["sha256"]}
        and type(image["size_bytes"]) is int and image["size_bytes"] == expected["image"]["size_bytes"],
        "copied backend is not the exact prepared package image")
    actual = stream_image(wanted)
    require(actual["sha256"] == expected["image"]["sha256"] and actual["size_bytes"] == image["size_bytes"],
        "shared actual backend bytes changed")
    for name, filename in (("metadata", "METADATA"), ("record", "RECORD")):
        value = begin[name]
        require(value["original"] == expected["distribution"][name]
            and value["captured"] == {"path": str(directory / "direct_backend" / filename),
                "sha256": expected["distribution"][name]["sha256"]}
            and digest(small_bytes(value["captured"]["path"])) == value["captured"]["sha256"],
            "shared owning package metadata bytes changed")
    require(package_headers(small_bytes(begin["metadata"]["captured"]["path"])) ==
        {key: expected["distribution"][key] for key in ("name", "version")}
        and record_entry(small_bytes(begin["record"]["captured"]["path"]),
            expected["distribution"]["record_entry"]["relative_path"], expected["image"]) ==
                expected["distribution"]["record_entry"], "captured package ownership/RECORD proof changed")
    maps = [begin["maps_before_copy"], begin["maps_after_copy"], final["maps_after_numerical_work"]]
    post = final["post_installed_package"]
    require(type(post) is dict and set(post) == {"started_at_seconds", "finished_at_seconds", "description"},
        "post-work installed package evidence is incomplete")
    package = post["description"]
    require(type(package) is dict and set(package) == {"site_packages", "image", "distribution", "observed_file_identity",
        "observed_descriptor_before", "observed_descriptor_after"}
        and all(package[key] == expected[key] for key in ("site_packages", "image", "distribution"))
        and package["observed_file_identity"] == image["source_file_identity"]
        and package["observed_descriptor_before"] == package["observed_descriptor_after"] == image["descriptor_before"],
        "post-work actual image, owning records or mapped descriptor changed")
    execution = result["execution_identity"]
    require(final["numerical_execution"] == execution
        and all(execution[key] == process[key] for key in ("hostname", "pid", "process_start_time"))
        and all(type(execution[key]) in (int, float) and math.isfinite(execution[key])
            for key in ("started_at_seconds", "finished_at_seconds"))
        and all(type(post[key]) in (int, float) and math.isfinite(post[key])
            for key in ("started_at_seconds", "finished_at_seconds")), "backend execution and post-work timestamps are incomplete")
    if profile_plan(plan):
        # Before: mapped capture precedes the original core's execution timestamp.
        # After: preserve that timestamp and join it to the complete surrounding memory window.
        memory = result["profile_memory"]["assigned_gpu_process_observation"]
        started, finished = memory.get("window_started_at_seconds"), memory.get("window_finished_at_seconds")
        require(type(started) in (int, float) and math.isfinite(started)
            and type(finished) in (int, float) and math.isfinite(finished)
            and maps[0]["observed_at_seconds"] <= maps[1]["observed_at_seconds"] <= started
            <= execution["started_at_seconds"] <= execution["finished_at_seconds"] <= finished
            <= post["started_at_seconds"] <= post["finished_at_seconds"] <= maps[2]["observed_at_seconds"],
            "profile package capture does not surround the complete original memory and execution windows")
    else:
        require(execution["started_at_seconds"] <= maps[0]["observed_at_seconds"]
            <= maps[1]["observed_at_seconds"] <= execution["finished_at_seconds"]
            <= post["started_at_seconds"] <= post["finished_at_seconds"] <= maps[2]["observed_at_seconds"],
            "backend snapshots do not surround this actual full numerical execution")
    require(image["descriptor_before"] == image["descriptor_after"]
        and image["descriptor_before"]["file_identity"] == image["source_file_identity"]
        and image["descriptor_before"]["size_bytes"] == expected["image"]["size_bytes"]
        and image["descriptor_before"]["regular_file"] is True,
        "runtime mapped image descriptor changed during the pinned copy")
    previous = expected["created_at_seconds"]
    for value in maps:
        require(value["path"] == "/proc/self/maps" and digest(value["raw"].encode()) == value["sha256"]
            and mapping_rows(value["raw"], expected["image"]["path"])[0] == value["file_identity"] == image["source_file_identity"]
            and type(value["observed_at_seconds"]) in (int, float) and math.isfinite(value["observed_at_seconds"])
            and value["observed_at_seconds"] >= previous, "actual before/after package-image mapping changed")
        previous = value["observed_at_seconds"]
    return final


def main():
    """Expose only standard-library pre-GPU metadata preparation, with no launch operation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--site-packages", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    value = prepare_image(arguments.python, arguments.site_packages, arguments.image)
    print(json.dumps({"prepared": publish(arguments.output, value), "gpu_operations": 0, "numerical_imports": []}))


if __name__ == "__main__":
    main()
