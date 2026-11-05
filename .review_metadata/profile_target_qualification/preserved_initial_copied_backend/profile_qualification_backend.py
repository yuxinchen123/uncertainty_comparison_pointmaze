"""Separate immutable backend evidence for a full-original profile qualification.

The AA and H100 callers retain their actual, distinct device-binding policies.
This module reads standard-library metadata and files only. It neither imports a
numerical package nor loads a library or grants scientific/profile admission.
Existing direct and group backend protocols are unchanged.
"""
import argparse
import importlib.util
import math
import os
from pathlib import Path
import re
import sys
import time

NAME = "profile_qualification_backend.py"
PREFIX = "10_jax_exploration_platform/scripts/atari_pixel_campaign/"
KIND = "full_original_profile_source_target_qualification"
PROTOCOL = "full-original-profile-source-target-qualification-version-1"
CAPABILITY = "profile_qualification_owning_package_and_same_process_mapped_image_version_2"
PREPARED_KIND = "prepared_profile_qualification_backend_package_image"
CARD_ONE = "GPU-fed69954-5a71-dbb2-ee91-544e0eb6b8a3"
BEGIN = "profile_qualification_backend_begin.json"
FINAL = "profile_qualification_backend_complete.json"
ATTEMPT = "profile_qualification_backend_attempt.json"
FAILURE = "profile_qualification_backend_failure.json"
SIDECAR = "profile_qualification_backend_expected.json"
CAPTURE = "profile_qualification_backend"


def primitives():
    """Load exact adjacent file/package primitives, never an ambient backend module."""
    path = Path(__file__).with_name("direct_backend_image.py")
    specification = importlib.util.spec_from_file_location("profile_qualification_file_primitives", path)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


files = primitives()
require = files.require


def helpers():
    """Declare the complete separate adapter and immutable file primitive closure."""
    return (NAME, "direct_backend_image.py")


def source_hashes():
    """Bind both exact helper byte sequences before any controlled work."""
    return {name: files.digest(Path(__file__).with_name(name).read_bytes()) for name in helpers()}


def number(value):
    """Exclude booleans and nonfinite or nonpositive timestamps."""
    return type(value) in {int, float} and math.isfinite(value) and value > 0


def path_text(value):
    """Check a canonical absolute spelling without opening a target-host path."""
    return (type(value) is str and Path(value).is_absolute() and str(Path(value)) == value
        and ".." not in Path(value).parts)


def image_description(value, *, name=None):
    """Require an exact image path, SHA256 and positive typed byte count."""
    require(type(value) is dict and set(value) == {"path", "sha256", "size_bytes"}
        and path_text(value["path"]) and (name is None or Path(value["path"]).name == name)
        and type(value["sha256"]) is str and re.fullmatch(r"[0-9a-f]{64}", value["sha256"])
        and type(value["size_bytes"]) is int and value["size_bytes"] > 0,
        "one exact image path, SHA256 and byte count is required")
    return value


def reference_shape(value):
    """Validate a saved small reference without opening its original target-host path."""
    require(type(value) is dict and set(value) == {"path", "sha256"} and path_text(value["path"])
        and type(value["sha256"]) is str and re.fullmatch(r"[0-9a-f]{64}", value["sha256"]),
        "one exact metadata path and SHA256 is required")
    return value


def prepare_image(python, site_packages, image_path):
    """Pin the owning package under the actual selected Python, without numerical imports."""
    require(not files.NUMERICAL.intersection(sys.modules), "backend prepin must precede numerical imports")
    python = str(Path(python).absolute())
    require(path_text(python) and Path(python).parent.name == "bin"
        and str(Path(sys.executable).absolute()) == python,
        "prepin must run under the actual selected environment Python")
    site_packages = files.ordinary(site_packages, directory=True)
    require(site_packages.is_relative_to(Path(python).parent.parent / "lib"),
        "backend package must belong to the selected Python environment")
    executable = Path("/proc/self/exe").resolve(strict=True)
    require(Path(python).resolve(strict=True) == executable, "selected Python and actual executable differ")
    python_image = files.stream_image(executable)
    described = files.describe_image(site_packages, image_path)
    return {"schema_version": 2, "kind": PREPARED_KIND, "capability": CAPABILITY,
        "selected_python": python, "python_image": {key: python_image[key] for key in ("path", "sha256", "size_bytes")},
        **described, "created_at_seconds": time.time(), "producer_sources": source_hashes(),
        "observer": files.own_process(), "observing_python": str(Path(sys.executable).absolute()),
        "numerical_imports": [], "gpu_operations": 0, "scientific_credit": 0, "admission_granted": False}


def validate_expected(value, python):
    """Validate the separate strict prepin schema without reading original installed files."""
    require(type(value) is dict and set(value) == {"schema_version", "kind", "capability", "selected_python",
        "python_image", "site_packages", "image", "distribution", "observed_file_identity", "created_at_seconds",
        "producer_sources", "observer", "observing_python", "numerical_imports", "gpu_operations",
        "scientific_credit", "admission_granted"}
        and type(value["schema_version"]) is int and value["schema_version"] == 2
        and value["kind"] == PREPARED_KIND and value["capability"] == CAPABILITY
        and value["selected_python"] == python and path_text(python) and Path(python).parent.name == "bin"
        and value["observing_python"] == python and value["producer_sources"] == source_hashes()
        and value["numerical_imports"] == [] and type(value["gpu_operations"]) is int and value["gpu_operations"] == 0
        and type(value["scientific_credit"]) is int and value["scientific_credit"] == 0
        and value["admission_granted"] is False and number(value["created_at_seconds"]),
        "a separate exact schema-two profile-qualification backend prepin is required")
    image_description(value["python_image"])
    image = image_description(value["image"], name="xla_cuda_plugin.so")
    owner = value["distribution"]
    require(type(owner) is dict and set(owner) == {"name", "version", "metadata", "record", "record_entry"}
        and type(owner["name"]) is str and owner["name"] and type(owner["version"]) is str and owner["version"],
        "the actual unique owning distribution is incomplete")
    site = value["site_packages"]
    require(path_text(site) and Path(site).name == "site-packages"
        and Path(site).is_relative_to(Path(python).parent.parent / "lib")
        and Path(image["path"]).is_relative_to(site), "the package image belongs to another selected Python environment")
    for key, filename in (("metadata", "METADATA"), ("record", "RECORD")):
        ref = reference_shape(owner[key])
        require(Path(ref["path"]).is_relative_to(site) and Path(ref["path"]).name == filename,
            "the owning package records must be exact files inside the selected site root")
    require(Path(owner["metadata"]["path"]).parent == Path(owner["record"]["path"]).parent
        and Path(owner["metadata"]["path"]).parent.name.endswith(".dist-info"),
        "one owning distribution METADATA/RECORD directory is required")
    entry = owner["record_entry"]
    require(type(entry) is dict and set(entry) == {"relative_path", "hash", "size_bytes"}
        and entry["relative_path"] == Path(image["path"]).relative_to(site).as_posix()
        and type(entry["hash"]) is str and re.fullmatch(r"sha256=[A-Za-z0-9_-]{43}", entry["hash"])
        and type(entry["size_bytes"]) is int and entry["size_bytes"] == image["size_bytes"],
        "one exact owning RECORD entry is required")
    identity, observer = value["observed_file_identity"], value["observer"]
    require(type(identity) is dict and set(identity) == {"device_major", "device_minor", "inode"}
        and all(type(item) is int and item >= 0 for item in identity.values()) and identity["inode"] > 0,
        "the prepin filesystem identity is incomplete")
    require(type(observer) is dict and type(observer.get("pid")) is int and observer["pid"] > 0
        and type(observer.get("hostname")) is str and observer["hostname"]
        and type(observer.get("uid")) is int and observer["uid"] == 2618919
        and type(observer.get("process_start_time")) is str
        and re.fullmatch(r"[1-9][0-9]*", observer["process_start_time"])
        and type(observer.get("uid_bytes")) is str and observer["uid_sha256"] == files.digest(observer["uid_bytes"].encode())
        and re.fullmatch(r"Uid:\s*2618919\s+2618919\s+2618919\s+2618919\s*", observer["uid_bytes"]),
        "prepin must retain its actual own process and all four UIDs")
    return value


def unit_directory(plan):
    """Recognize only the separate qualification and its real AA or direct-H100 policy."""
    declaration = plan.get("profile_target_qualification")
    require(type(plan.get("schema_version")) is int and plan["schema_version"] == 1
        and plan.get("kind") == KIND and type(declaration) is dict
        and set(declaration) == {"schema_version", "protocol", "parent"}
        and type(declaration["schema_version"]) is int and declaration["schema_version"] == 1
        and declaration["protocol"] == PROTOCOL, "one explicit profile-source target qualification is required")
    reference_shape(declaration["parent"])
    parent = files.checked_reference(declaration["parent"])
    require(type(parent.get("schema_version")) is int and parent["schema_version"] == 1
        and parent.get("kind") == "profile_target_qualification_controller",
        "the qualification requires its actual separate controller plan bytes")
    require(type(plan.get("unit_id")) is str and Path(plan["unit_id"]).name == plan["unit_id"]
        and plan["unit_id"] not in {"", ".", ".."} and path_text(plan.get("campaign_run_dir"))
        and not any(key in plan for key in ("group_backend_image", "direct_backend_image", "final_confirmation_protocol",
            "device_memory_limit_protocol", "completed_study")), "qualification backend cannot inherit another protocol")
    aa = plan.get("hardware_class") == "a4500" and "configured_isolation" in plan and "direct_binding" not in plan
    h100 = (plan.get("hardware_class") == "h100" and "direct_binding" in plan
        and "configured_isolation" not in plan and plan.get("gpu_uuid") == CARD_ONE)
    require(aa or h100, "retain the real AA configured or card-one direct H100 policy")
    directory = files.ordinary(Path(plan["campaign_run_dir"]) / "data" / plan["unit_id"], directory=True)
    return directory


def configure_plan(plan, prepared_image_record):
    """Copy an actual separate prepin before the caller freezes its real qualification plan."""
    directory = unit_directory(plan)
    expected = files.checked_reference(files.reference(prepared_image_record))
    validate_expected(expected, plan["python"])
    target = directory / SIDECAR
    require(Path(prepared_image_record) != target, "retain the actual external prepin separately from its unit copy")
    raw = files.small_bytes(prepared_image_record)
    if target.exists() or target.is_symlink():
        require(not target.is_symlink() and files.small_bytes(target) == raw, "preserve the prior different backend sidecar")
    else:
        files.publish_bytes(target, raw)
    plan["profile_qualification_backend"] = {"schema_version": 2, "capability": CAPABILITY, "expected": files.reference(target)}
    plan["inputs"]["profile_qualification_backend_record"] = files.reference(prepared_image_record)
    return plan


def expected_for_plan(plan):
    """Require the exact two-source closure and both immutable prepin references."""
    directory = unit_directory(plan)
    require(all(plan["helpers"]["files"].get(PREFIX + name) == sha for name, sha in source_hashes().items()),
        "both exact qualification backend helpers must be staged")
    declaration = plan.get("profile_qualification_backend")
    require(type(declaration) is dict and set(declaration) == {"schema_version", "capability", "expected"}
        and type(declaration["schema_version"]) is int and declaration["schema_version"] == 2
        and declaration["capability"] == CAPABILITY and declaration["expected"]["path"] == str(directory / SIDECAR),
        "the separate prepared qualification backend sidecar is required")
    expected = files.checked_reference(declaration["expected"])
    external = plan["inputs"]["profile_qualification_backend_record"]
    require(external["path"] != declaration["expected"]["path"] and files.checked_reference(external) == expected,
        "unit sidecar differs from its actual original prepin")
    return validate_expected(expected, plan["python"])


def validate_prepared(plan, *, retain_actual=False):
    """Rehash the installed package and actual selected executable on the numerical host."""
    expected = expected_for_plan(plan)
    require(str(Path(sys.executable).absolute()) == plan["python"]
        and str(Path("/proc/self/exe").resolve(strict=True)) == expected["python_image"]["path"],
        "backend work is running under another actual Python executable")
    executable = files.stream_image(expected["python_image"]["path"])
    require(all(executable[key] == expected["python_image"][key] for key in ("path", "sha256", "size_bytes")),
        "actual selected Python bytes changed after preparation")
    actual = files.describe_image(expected["site_packages"], expected["image"]["path"], retain_descriptor=retain_actual)
    require(all(actual[key] == expected[key] for key in ("site_packages", "image", "distribution")),
        "actual installed backend or owning package changed after preparation")
    versions = plan["qualification_software"]["packages"]
    normalized = {re.sub(r"[-_.]+", "-", name).lower(): version for name, version in versions.items()}
    require(len(normalized) == len(versions)
        and normalized.get(re.sub(r"[-_.]+", "-", expected["distribution"]["name"]).lower()) == expected["distribution"]["version"],
        "owning backend package differs from the unchanged full-original software gates")
    return (expected, actual) if retain_actual else expected


def binding_file(plan):
    """Keep real configured and direct binding records separate."""
    return "direct_device_binding.json" if plan["hardware_class"] == "h100" else "resolved_device_binding.json"


def process_join(directory, plan, binding, *, current):
    """Join the separately validated binding to its exact plan and same-process current context."""
    directory = files.ordinary(directory, directory=True)
    require(directory == unit_directory(plan)
        and files.encoded(files.document(directory / "measurement_plan.json")) == files.encoded(plan)
        and files.encoded(files.document(directory / binding_file(plan))) == files.encoded(binding),
        "backend evidence belongs to another unit or binding")
    process, native = binding["process"], binding["native"]
    h100 = plan["hardware_class"] == "h100"
    require(type(process.get("uid")) is int and process["uid"] == 2618919
        and type(process.get("pid")) is int and process["pid"] > 0
        and process.get("hostname") == ("serval02" if h100 else "jaguar03")
        and type(process.get("process_start_time")) is str and re.fullmatch(r"[1-9][0-9]*", process["process_start_time"])
        and binding.get("kind") == ("serval02_card1_current_process_resolved_device_binding" if h100
            else "uva_current_process_resolved_device_binding"), "backend requires the real target's actual owning process")
    plan_ref = files.reference(directory / "measurement_plan.json")
    require((binding.get("plan") == plan_ref if h100 else binding.get("plan_sha256") == plan_ref["sha256"])
        and type(native.get("pid")) is int and native["pid"] == process["pid"]
        and type(native.get("current_context")) is int and native["current_context"] > 0
        and type(native.get("cuda_device_handle")) is int and native["cuda_device_handle"] >= 0
        and type(native.get("current_context_device")) is int and native["current_context_device"] == native["cuda_device_handle"]
        and type(native.get("cuda_device_count")) is int and native["cuda_device_count"] == 1
        and type(native.get("jax_local_hardware_id")) is int and native["jax_local_hardware_id"] == 0
        and native.get("api_return_codes") == {key: 0 for key in ("cuDeviceGetCount", "cuDeviceGet", "cuCtxGetCurrent",
            "cuCtxGetDevice", "cuDeviceGetUuid_v2", "cuDeviceGetPCIBusId")}
        and all(type(value) is int for value in native["api_return_codes"].values())
        and native.get("jax_identity_basis") == "sole local JAX GPU, sole CUDA-visible ordinal, and that device's current CUDA context"
        and native.get("jax_device_kind") == ("NVIDIA H100 NVL" if h100 else "NVIDIA RTX A4500")
        and type(binding.get("gpu_uuid")) is str
        and re.fullmatch(r"GPU-[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}", binding["gpu_uuid"])
        and native.get("gpu_uuid") == binding["gpu_uuid"] and (not h100 or binding["gpu_uuid"] == CARD_ONE)
        and type(binding.get("pci_address")) is str and re.fullmatch(r"[0-9a-f]{4}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]", binding["pci_address"])
        and native.get("pci_address") == binding["pci_address"]
        and type(binding.get("physical_minor")) is int and binding["physical_minor"] >= 0
        and files.encoded(binding.get("device")) == files.encoded({"path": "/dev/nvidia" + str(binding["physical_minor"]), "character_device": True,
            "symlink": False, "major": 195, "minor": binding["physical_minor"]})
        and binding.get("physical_gres_mapping_verified") is False
        and binding.get("kernel_device_enforcement_verified") is False,
        "actual same-process context, UUID, PCI/minor or explicit trust boundary changed")
    if current:
        actual = files.own_process()
        require(all(actual[key] == process[key] for key in ("hostname", "pid", "process_start_time", "uid")),
            "backend capture can read only this actual own bound process")
        return actual
    return process


def begin(directory, plan, binding):
    """Capture actual mapped bytes before both profile training and original-reference work."""
    directory = files.ordinary(directory, directory=True)
    require(not any((directory / name).exists() or (directory / name).is_symlink()
        for name in (ATTEMPT, BEGIN, FINAL, FAILURE, CAPTURE)), "preserve the prior backend attempt; no retry is permitted")
    expected = validate_prepared(plan)
    process = process_join(directory, plan, binding, current=True)
    files.publish(directory / ATTEMPT, {"schema_version": 2, "capability": CAPABILITY,
        "plan": files.reference(directory / "measurement_plan.json"), "process": process, "retry_permitted": False})
    try:
        before = files.mapped_snapshot(expected["image"]["path"])
        target = directory / CAPTURE
        target.mkdir()
        captures = {}
        for key, filename, original, limit in (("image", "xla_cuda_plugin.so", expected["image"], expected["image"]["size_bytes"]),
                ("metadata", "METADATA", expected["distribution"]["metadata"], files.SMALL_FILE_LIMIT),
                ("record", "RECORD", expected["distribution"]["record"], files.SMALL_FILE_LIMIT)):
            captures[key] = files.captured_reference(target / filename, original["path"],
                {name: original[name] for name in ("path", "sha256")}, limit)
        after = files.mapped_snapshot(expected["image"]["path"])
        require(before["file_identity"] == after["file_identity"] == captures["image"]["source_file_identity"],
            "the mapped backend image changed during its immutable shared copy")
        value = {"schema_version": 2, "kind": "profile_qualification_backend_begin", "capability": CAPABILITY,
            "plan": files.reference(directory / "measurement_plan.json"), "expected": plan["profile_qualification_backend"]["expected"],
            "binding": files.reference(directory / binding_file(plan)), "process": process,
            "gpu_uuid": binding["gpu_uuid"], "pci_address": binding["pci_address"], "physical_minor": binding["physical_minor"],
            "maps_before_copy": before, "maps_after_copy": after, **captures,
            "scientific_credit": 0, "admission_granted": False}
        files.publish(directory / BEGIN, value)
        return value
    except BaseException as error:
        files.publish(directory / FAILURE, {"schema_version": 2, "error_type": type(error).__name__,
            "error": str(error), "measurement_valid": False, "admission_granted": False})
        raise


def completed_work(plan, work):
    """Require both actual phases and retain unknown synthetic-test transition costs."""
    copies = plan["configuration"]["learner"]["n_copies"]
    require(type(copies) is int and copies in {1, 2} and len(plan["copy_seeds"]) == copies
        and type(plan["profile_updates"]) is int and plan["profile_updates"] > 0
        and type(work) is dict and work.get("kind") == "full_original_profile_source_target_qualification_work"
        and work.get("all_original_gates_passed") is True
        and files.encoded(work.get("configuration")) == files.encoded(plan["configuration"])
        and files.encoded(work.get("copy_seeds")) == files.encoded(plan["copy_seeds"])
        and files.encoded(work.get("actual_runtime_environment")) == files.encoded(plan["runtime_environment"])
        and work.get("test_fixture_work") == "unknown" and type(work.get("scientific_credit")) is int
        and work["scientific_credit"] == 0 and work.get("admission_granted") is False,
        "backend completion requires actual separate full-original and profile work")
    reference = work.get("original_reference")
    wanted = {"copies": 1, "normalization_warmup_iterations": 50, "reference_iteration": 1,
        "reference_optimizer_step": 16, "executed_work": {"normalization_warmup_transitions": 819200,
            "training_transitions": 16384, "complete_updates": 1, "minibatch_updates": 16}}
    require(files.encoded(reference) == files.encoded(wanted),
        "the original reference remains one copy, fifty warmups and one complete unchanged update")
    capture = work.get("profile_capture")
    require(type(capture) is dict and type(capture.get("copies")) is int and capture["copies"] == copies
        and capture.get("capture_completed") is True and type(capture.get("profile_updates")) is int
        and capture["profile_updates"] == plan["profile_updates"]
        and type(capture.get("normalization_warmup_iterations")) is int and capture["normalization_warmup_iterations"] == 50
        and type(capture.get("compile_warmup_updates")) is int and capture["compile_warmup_updates"] > 0
        and type(capture.get("profile_start_iteration")) is int
        and capture["profile_start_iteration"] == capture["compile_warmup_updates"]
        and type(capture.get("profile_end_iteration")) is int
        and capture["profile_end_iteration"] - capture["profile_start_iteration"] == plan["profile_updates"]
        and type(capture.get("compiled_signatures_before")) is list and capture["compiled_signatures_before"]
        and files.encoded(capture["compiled_signatures_before"]) == files.encoded(capture.get("compiled_signatures_after")),
        "the exact full profile and actual compilation-update window must be complete")
    return "full_original_reference_and_profile_capture"


def complete(directory, plan, loaded_gpu_libraries, work_record):
    """Finish after both actual phases, retaining the actual post-work installed image hash."""
    directory = files.ordinary(directory, directory=True)
    require(not any((directory / name).exists() or (directory / name).is_symlink() for name in (FINAL, FAILURE)),
        "preserve the prior backend completion or failure")
    completed_work(plan, work_record)
    started = time.time()
    expected, package = validate_prepared(plan, retain_actual=True)
    finished = time.time()
    first = files.document(directory / BEGIN)
    binding = files.document(directory / binding_file(plan))
    process = process_join(directory, plan, binding, current=True)
    after = files.mapped_snapshot(expected["image"]["path"])
    require(first["process"] == process and after["file_identity"] == first["image"]["source_file_identity"],
        "the actual post-work process or mapped backend changed")
    value = {"schema_version": 2, "kind": "profile_qualification_backend_complete", "capability": CAPABILITY,
        "plan": files.reference(directory / "measurement_plan.json"), "expected": plan["profile_qualification_backend"]["expected"],
        "binding": files.reference(directory / binding_file(plan)), "begin": files.reference(directory / BEGIN),
        "process": process, "gpu_uuid": binding["gpu_uuid"], "pci_address": binding["pci_address"],
        "physical_minor": binding["physical_minor"], "maps_after_numerical_work": after,
        "post_installed_package": {"started_at_seconds": started, "finished_at_seconds": finished, "description": package},
        "loaded_gpu_libraries_sha256": files.digest(files.encoded(loaded_gpu_libraries)),
        "work_record_sha256": files.digest(files.encoded(work_record)), "completed_work_kind": completed_work(plan, work_record),
        "scientific_credit": 0, "admission_granted": False}
    files.publish(directory / FINAL, value)
    validate_saved(directory, plan, binding, loaded_gpu_libraries, work_record)
    return value


def validate_saved(directory, plan, binding, loaded, work):
    """Verify only shared captured bytes; never open an installed target path on a CPU verifier."""
    directory = files.ordinary(directory, directory=True)
    require(not (directory / FAILURE).exists() and not (directory / FAILURE).is_symlink(),
        "a failed backend attempt cannot earn a completed package proof")
    expected = expected_for_plan(plan)
    process = process_join(directory, plan, binding, current=False)
    first, final = files.document(directory / BEGIN), files.document(directory / FINAL)
    for value, kind in ((first, "profile_qualification_backend_begin"), (final, "profile_qualification_backend_complete")):
        require(type(value.get("schema_version")) is int and value["schema_version"] == 2
            and value.get("kind") == kind and value.get("capability") == CAPABILITY
            and value.get("plan") == files.reference(directory / "measurement_plan.json")
            and value.get("expected") == plan["profile_qualification_backend"]["expected"]
            and value.get("binding") == files.reference(directory / binding_file(plan))
            and all(type(value["process"][key]) is type(process[key]) and value["process"][key] == process[key]
                for key in ("hostname", "pid", "process_start_time", "uid"))
            and value.get("gpu_uuid") == binding["gpu_uuid"] and value.get("pci_address") == binding["pci_address"]
            and type(value.get("physical_minor")) is int and value["physical_minor"] == binding["physical_minor"]
            and type(value.get("scientific_credit")) is int and value["scientific_credit"] == 0
            and value.get("admission_granted") is False, "backend proof changed its actual unit, target or process")
    require(final["begin"] == files.reference(directory / BEGIN) and first["process"] == final["process"]
        and type(loaded.get("pid")) is int and loaded["pid"] == process["pid"]
        and final["loaded_gpu_libraries_sha256"] == files.digest(files.encoded(loaded))
        and final["work_record_sha256"] == files.digest(files.encoded(work))
        and final["completed_work_kind"] == completed_work(plan, work),
        "backend completion changed its original work or separate CUDA/cuDNN receipt")
    image = first["image"]
    for key, filename, original in (("image", "xla_cuda_plugin.so", expected["image"]),
            ("metadata", "METADATA", expected["distribution"]["metadata"]),
            ("record", "RECORD", expected["distribution"]["record"])):
        captured = first[key]
        require(captured["original"] == {name: original[name] for name in ("path", "sha256")}
            and captured["captured"] == {"path": str(directory / CAPTURE / filename), "sha256": original["sha256"]},
            "backend capture must name its exact immutable shared byte copy")
        actual = files.stream_image(captured["captured"]["path"], maximum_bytes=(
            expected["image"]["size_bytes"] if key == "image" else files.SMALL_FILE_LIMIT))
        require(actual["sha256"] == original["sha256"] and type(captured["size_bytes"]) is int
            and actual["size_bytes"] == captured["size_bytes"]
            and (key != "image" or actual["size_bytes"] == expected["image"]["size_bytes"]),
            "shared captured package bytes changed or were replaced")
    require(files.package_headers(files.small_bytes(first["metadata"]["captured"]["path"])) ==
        {key: expected["distribution"][key] for key in ("name", "version")}
        and files.record_entry(files.small_bytes(first["record"]["captured"]["path"]),
            expected["distribution"]["record_entry"]["relative_path"], expected["image"]) == expected["distribution"]["record_entry"],
        "captured METADATA/RECORD no longer uniquely own the exact captured image")
    maps = [first["maps_before_copy"], first["maps_after_copy"], final["maps_after_numerical_work"]]
    post = final["post_installed_package"]
    require(type(post) is dict and set(post) == {"started_at_seconds", "finished_at_seconds", "description"},
        "actual post-work package hash pass is missing")
    package = post["description"]
    require(type(package) is dict and set(package) == {"site_packages", "image", "distribution", "observed_file_identity",
            "observed_descriptor_before", "observed_descriptor_after"}
        and all(package[key] == expected[key] for key in ("site_packages", "image", "distribution"))
        and package["observed_file_identity"] == image["source_file_identity"]
        and package["observed_descriptor_before"] == package["observed_descriptor_after"] == image["descriptor_before"],
        "post-work package bytes or the original mapped file descriptor changed")
    require(image["descriptor_before"] == image["descriptor_after"]
        and image["descriptor_before"]["file_identity"] == image["source_file_identity"]
        and type(image["descriptor_before"]["size_bytes"]) is int
        and image["descriptor_before"]["size_bytes"] == expected["image"]["size_bytes"]
        and image["descriptor_before"]["regular_file"] is True,
        "the original actual mapped descriptor did not remain unchanged during capture")
    previous = expected["created_at_seconds"]
    for snapshot in maps:
        require(snapshot["path"] == "/proc/self/maps" and type(snapshot["raw"]) is str
            and files.digest(snapshot["raw"].encode()) == snapshot["sha256"]
            and files.mapping_rows(snapshot["raw"], expected["image"]["path"])[0] ==
                snapshot["file_identity"] == image["source_file_identity"]
            and number(snapshot["observed_at_seconds"]) and snapshot["observed_at_seconds"] >= previous,
            "backend same-process mapped image bytes, inode or chronology changed")
        previous = snapshot["observed_at_seconds"]
    execution, gates = work["execution_identity"], work["original_gate_execution"]
    for phase in (execution, gates):
        require(type(phase) is dict and set(phase) == {"hostname", "pid", "process_start_time", "started_at_seconds", "finished_at_seconds"}
            and all(type(phase[key]) is type(process[key]) and phase[key] == process[key]
                for key in ("hostname", "pid", "process_start_time"))
            and number(phase["started_at_seconds"]) and number(phase["finished_at_seconds"]),
            "both actual completed phases must name the same owning Python process and birth")
    require(number(post["started_at_seconds"]) and number(post["finished_at_seconds"])
        and maps[1]["observed_at_seconds"] <= execution["started_at_seconds"] <= execution["finished_at_seconds"]
        <= gates["started_at_seconds"] <= gates["finished_at_seconds"] <= post["started_at_seconds"]
        <= post["finished_at_seconds"] <= maps[2]["observed_at_seconds"],
        "backend begin/completion do not surround both actual complete numerical phases")
    return final


def main():
    """Expose only a zero-GPU immutable prepin operation under the selected Python."""
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("python", "site-packages", "image", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    arguments = parser.parse_args()
    value = prepare_image(arguments.python, arguments.site_packages, arguments.image)
    print(files.encoded({"prepared": files.publish(arguments.output, value), "gpu_operations": 0,
        "numerical_imports": [], "admission_granted": False}).decode(), end="")


if __name__ == "__main__":
    main()
