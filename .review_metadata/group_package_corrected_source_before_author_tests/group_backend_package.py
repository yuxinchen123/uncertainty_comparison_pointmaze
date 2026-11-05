"""Pin and capture each A4500 group worker's actual installed XLA package image.

Only standard-library file and process metadata is read here. The caller proves
the group CUDA binding separately. CUDA, cuDNN and runtime receipts remain separate.
"""
import importlib.util
import math
import os
from pathlib import Path
import re
import sys
import time

NAME = "group_backend_package.py"
PREFIX = "10_jax_exploration_platform/scripts/atari_pixel_campaign/"
CAPABILITY = "prepared_package_image_and_group_worker_mapping_version_2"
PREPARED_KIND = "prepared_group_backend_package_image"
MODES = {"qualification": "configured_group_original_member_qualification",
    "correctness": "configured_group_selected_member_correctness",
    "measurement": "configured_group_common_wall_member_measurement",
    "exact_eight_eligibility": "configured_group_exact_eight_member_memory_observation"}


def primitives():
    """Load only the adjacent reviewed standard-library package-image primitives."""
    path = Path(__file__).with_name("direct_backend_image.py")
    specification = importlib.util.spec_from_file_location("group_package_file_primitives", path)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


files = primitives()
require = files.require


def helper_names():
    """Declare both exact source files needed by the separate group protocol."""
    return (NAME, "direct_backend_image.py")


def source_hashes():
    """Identify this producer and the exact shared file/package primitive source."""
    return {name: files.digest(Path(__file__).with_name(name).read_bytes()) for name in helper_names()}


def group_directory(plan):
    """Require the explicit group schema and return its one ordinary unit directory."""
    require(type(plan.get("schema_version")) is int and plan["schema_version"] == 2
        and plan.get("kind") == "configured_a4500_scaling_group"
        and plan.get("hardware_class") == "jaguar03_rtx_a4500"
        and type(plan.get("gpu_count")) is int and plan["gpu_count"] in {1, 8}
        and plan.get("mode") in MODES, "an explicit schema-two A4500 group mode is required")
    require(type(plan.get("unit_id")) is str and Path(plan["unit_id"]).name == plan["unit_id"]
        and plan["unit_id"] not in {"", ".", ".."}
        and not any(key in plan for key in ("direct_binding", "configured_isolation",
            "device_memory_limit_protocol", "final_confirmation_protocol", "portable_evidence")),
        "backend proof cannot change group, final, capacity or portable protocols")
    return files.ordinary(Path(plan["campaign_run_dir"]) / "data" / plan["unit_id"], directory=True)


def worker_index(directory, plan):
    """Accept only the separately reserved worker directory named in this exact group."""
    directory = files.ordinary(directory, directory=True)
    parent = files.ordinary(group_directory(plan) / "workers", directory=True)
    match = re.fullmatch(r"worker_([1-8])", directory.name)
    require(directory.parent == parent and match is not None
        and int(match.group(1)) <= plan["gpu_count"], "backend capture belongs to another group worker")
    return int(match.group(1)) - 1


def prepare_image(python, site_packages, image_path):
    """Discover and pin the actual owning package before any numerical import or GPU work."""
    require(not files.NUMERICAL.intersection(sys.modules), "group prepin must precede numerical imports")
    python = Path(python).absolute()
    site_packages = files.ordinary(site_packages, directory=True)
    require(python.is_file() and python.parent.name == "bin"
        and site_packages.is_relative_to(python.parent.parent / "lib"), "select the exact target Python environment")
    described = files.describe_image(site_packages, image_path)
    return {"schema_version": 2, "kind": PREPARED_KIND, "capability": CAPABILITY,
        "selected_python": str(python), **described, "created_at_seconds": time.time(),
        "producer_sources": source_hashes(), "observer": files.own_process(),
        "observing_python": str(Path(sys.executable).absolute()), "numerical_imports": [],
        "gpu_operations": 0, "scientific_credit": 0, "admission_granted": False}


def validate_expected(value, python):
    """Refuse legacy, changed-source or incomplete pre-GPU group package records."""
    require(type(value) is dict and set(value) == {"schema_version", "kind", "capability", "selected_python",
        "site_packages", "image", "distribution", "observed_file_identity", "created_at_seconds",
        "producer_sources", "observer", "observing_python", "numerical_imports", "gpu_operations",
        "scientific_credit", "admission_granted"}
        and type(value["schema_version"]) is int and value["schema_version"] == 2
        and value["kind"] == PREPARED_KIND and value["capability"] == CAPABILITY
        and value["selected_python"] == python and value["producer_sources"] == source_hashes()
        and value["numerical_imports"] == [] and type(value["gpu_operations"]) is int
        and value["gpu_operations"] == 0 and type(value["scientific_credit"]) is int
        and value["scientific_credit"] == 0 and value["admission_granted"] is False,
        "a new exact group package-image prepin is required")
    # Validate typed identities independently of later actual installed-file reads.
    require(type(value["created_at_seconds"]) in {int, float} and math.isfinite(value["created_at_seconds"])
        and value["created_at_seconds"] > 0 and type(value["observing_python"]) is str
        and Path(value["observing_python"]).is_absolute(), "preparation time or observer Python is invalid")
    selected_python = value["selected_python"]
    require(type(selected_python) is str and Path(selected_python).is_absolute()
        and str(Path(selected_python)) == selected_python and ".." not in Path(selected_python).parts
        and Path(selected_python).parent.name == "bin", "the selected Python must name its exact environment bin path")
    image, owner = value["image"], value["distribution"]
    require(type(image) is dict and set(image) == {"path", "sha256", "size_bytes"}
        and type(image["path"]) is str and Path(image["path"]).is_absolute()
        and Path(image["path"]).name == "xla_cuda_plugin.so"
        and type(image["sha256"]) is str and re.fullmatch(r"[0-9a-f]{64}", image["sha256"])
        and type(image["size_bytes"]) is int and image["size_bytes"] > 0,
        "the exact package image path, byte count and hash are required")
    require(type(owner) is dict and set(owner) == {"name", "version", "metadata", "record", "record_entry"}
        and type(owner["name"]) is str and owner["name"] and type(owner["version"]) is str and owner["version"],
        "the actual discovered owning distribution is incomplete")
    require(type(value["site_packages"]) is str and Path(value["site_packages"]).is_absolute()
        and str(Path(value["site_packages"])) == value["site_packages"]
        and ".." not in Path(value["site_packages"]).parts
        and Path(value["site_packages"]).name == "site-packages"
        and Path(value["site_packages"]).is_relative_to(Path(selected_python).parent.parent / "lib")
        and Path(image["path"]).is_relative_to(value["site_packages"])
        and all(type(owner[name]) is dict and set(owner[name]) == {"path", "sha256"}
            and type(owner[name]["path"]) is str and Path(owner[name]["path"]).is_absolute()
            and Path(owner[name]["path"]).is_relative_to(value["site_packages"])
            and type(owner[name]["sha256"]) is str
            and re.fullmatch(r"[0-9a-f]{64}", owner[name]["sha256"]) for name in ("metadata", "record"))
        and Path(owner["metadata"]["path"]).name == "METADATA"
        and Path(owner["record"]["path"]).name == "RECORD"
        and Path(owner["metadata"]["path"]).parent == Path(owner["record"]["path"]).parent,
        "prepared owner records must name exact files inside the selected package environment")
    identity, observer = value["observed_file_identity"], value["observer"]
    require(type(identity) is dict and set(identity) == {"device_major", "device_minor", "inode"}
        and all(type(item) is int and item >= 0 for item in identity.values()) and identity["inode"] > 0,
        "preparation filesystem identity is incomplete")
    require(type(observer) is dict and type(observer["pid"]) is int and observer["pid"] > 0
        and type(observer["uid"]) is int and observer["uid"] == 2618919
        and type(observer["process_start_time"]) is str and re.fullmatch(r"[1-9][0-9]*", observer["process_start_time"])
        and observer["uid_sha256"] == files.digest(observer["uid_bytes"].encode())
        and list(map(int, observer["uid_bytes"].split()[1:])) == [2618919] * 4,
        "preparation must record its actual own PID, birth and four UIDs")
    return value


def configure_plan(plan, record_path):
    """Copy the actual external prepin into the group and bind both exact file references."""
    directory = group_directory(plan)
    expected = files.checked_reference(files.reference(record_path))
    validate_expected(expected, plan["python"])
    target = directory / "backend_image_expected.json"
    require(Path(record_path) != target, "the group requires a separately prepared external image record")
    raw = files.small_bytes(record_path)
    if target.exists() or target.is_symlink():
        require(not target.is_symlink() and files.small_bytes(target) == raw,
            "preserve the prior different group package-image record")
    else:
        files.publish_bytes(target, raw)
    plan["group_backend_image"] = {"schema_version": 2, "capability": CAPABILITY, "expected": files.reference(target)}
    plan["inputs"]["backend_image_record"] = files.reference(record_path)
    return plan


def expected_for_plan(plan):
    """Join the staged helper sources, immutable sidecar and original external prepin."""
    directory = group_directory(plan)
    for name, wanted in source_hashes().items():
        require(plan["helpers"]["files"].get(PREFIX + name) == wanted,
            "the exact group backend producer and shared primitives must both be staged")
    value = plan.get("group_backend_image")
    require(type(value) is dict and set(value) == {"schema_version", "capability", "expected"}
        and type(value["schema_version"]) is int and value["schema_version"] == 2
        and value["capability"] == CAPABILITY
        and value["expected"]["path"] == str(directory / "backend_image_expected.json"),
        "group backend readers refuse missing or legacy image evidence")
    expected = files.checked_reference(value["expected"])
    require(files.checked_reference(plan["inputs"]["backend_image_record"]) == expected
        and plan["inputs"]["backend_image_record"]["path"] != value["expected"]["path"],
        "the sidecar differs from its actual separate pre-GPU source record")
    return validate_expected(expected, plan["python"])


def validate_prepared(plan, *, retain_actual=False):
    """Rehash the actual installed image and discovered owning records before controlled work."""
    expected = expected_for_plan(plan)
    actual = files.describe_image(expected["site_packages"], expected["image"]["path"],
        retain_descriptor=retain_actual)
    require(all(actual[key] == expected[key] for key in ("site_packages", "image", "distribution")),
        "installed image or owning package changed after the pre-GPU group pin")
    return (expected, actual) if retain_actual else expected


def process_join(directory, plan, metadata, binding, *, current):
    """Join only this worker's already validated group metadata and native physical binding."""
    index = worker_index(directory, plan)
    require(files.encoded(files.document(group_directory(plan) / "scaling_plan.json")) == files.encoded(plan),
        "the supplied backend plan differs from the actual frozen group plan")
    require(metadata.get("role") == "worker" and metadata.get("directory") == str(directory)
        and files.checked_reference(files.reference(Path(directory) / "group_step_metadata.json")) == metadata
        and files.checked_reference(files.reference(Path(directory) / "resolved_device_binding.json")) == binding,
        "backend capture requires this worker's actual saved group metadata and binding")
    process = binding["process"]
    require(type(process["uid"]) is int and process["uid"] == 2618919 and process["hostname"] == "jaguar03"
        and type(process["pid"]) is int and process["pid"] > 0
        and type(process["process_start_time"]) is str and re.fullmatch(r"[1-9][0-9]*", process["process_start_time"])
        and all(process[key] == metadata["process"][key] for key in ("hostname", "pid", "process_start_time", "uid"))
        and type(binding["native"]["pid"]) is int and binding["native"]["pid"] == process["pid"]
        and type(binding["gpu_uuid"]) is str
        and re.fullmatch(r"GPU-[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}", binding["gpu_uuid"])
        and binding["native"]["gpu_uuid"] == binding["gpu_uuid"]
        and binding["plan_sha256"] == files.reference(group_directory(plan) / "scaling_plan.json")["sha256"]
        and binding["metadata_sha256"] == files.digest(files.encoded(metadata)),
        "backend package capture changed its actual native worker identity")
    if current:
        actual = files.own_process()
        require(all(actual[key] == process[key] for key in ("hostname", "pid", "process_start_time", "uid")),
            "backend capture can only read this current own worker")
        return index, actual
    return index, process


def capture_begin(directory, plan, metadata, binding):
    """Copy the actual mapped package and owning records before controlled learner work."""
    directory = files.ordinary(directory, directory=True)
    names = ("group_backend_capture_attempt.json", "group_backend_capture_begin.json",
        "group_backend_capture.json", "group_backend_capture_failure.json", "group_backend")
    require(not any((directory / name).exists() or (directory / name).is_symlink() for name in names),
        "preserve this worker's backend capture attempt; no retry is permitted")
    expected = validate_prepared(plan)
    index, process = process_join(directory, plan, metadata, binding, current=True)
    files.publish(directory / names[0], {"schema_version": 2, "capability": CAPABILITY, "worker_index": index,
        "process": process, "expected": plan["group_backend_image"]["expected"], "retry_permitted": False})
    try:
        # Before: installed image plus two owner files; after: three immutable shared copies.
        before = files.mapped_snapshot(expected["image"]["path"])
        target = directory / "group_backend"
        target.mkdir()
        image = files.captured_reference(target / "xla_cuda_plugin.so", expected["image"]["path"],
            {key: expected["image"][key] for key in ("path", "sha256")}, expected["image"]["size_bytes"])
        metadata_copy = files.captured_reference(target / "METADATA", expected["distribution"]["metadata"]["path"],
            expected["distribution"]["metadata"], files.SMALL_FILE_LIMIT)
        record_copy = files.captured_reference(target / "RECORD", expected["distribution"]["record"]["path"],
            expected["distribution"]["record"], files.SMALL_FILE_LIMIT)
        after = files.mapped_snapshot(expected["image"]["path"])
        require(before["file_identity"] == after["file_identity"] == image["source_file_identity"]
            and image["size_bytes"] == expected["image"]["size_bytes"], "mapped image changed during shared capture")
        value = {"schema_version": 2, "kind": "group_worker_backend_package_capture_begin",
            "capability": CAPABILITY, "worker_index": index,
            "plan": files.reference(group_directory(plan) / "scaling_plan.json"),
            "expected": plan["group_backend_image"]["expected"],
            "metadata_input": files.reference(directory / "group_step_metadata.json"),
            "binding": files.reference(directory / "resolved_device_binding.json"), "gpu_uuid": binding["gpu_uuid"],
            "process": process, "maps_before_copy": before, "maps_after_copy": after,
            "image": image, "metadata": metadata_copy, "record": record_copy,
            "scientific_credit": 0, "admission_granted": False}
        files.publish(directory / names[1], value)
        return value
    except BaseException as error:
        files.publish(directory / "group_backend_capture_failure.json", {"schema_version": 2,
            "error_type": type(error).__name__, "error": str(error), "measurement_valid": False})
        raise


def completed_work(directory, plan, result):
    """Require the explicitly selected completed worker operation, never partial or relabeled work."""
    index = worker_index(directory, plan)
    require(type(result.get("schema_version")) is int and result["schema_version"] == 2
        and result.get("kind") == MODES[plan["mode"]] and result.get("completed") is True
        and files.encoded(result.get("copy_seeds")) == files.encoded(plan["copy_seeds"][index])
        and type(result.get("scientific_credit")) is int and result["scientific_credit"] == 0
        and result.get("admission_granted") is False, "backend completion requires the exact completed zero-credit worker kind")
    if plan["mode"] == "qualification":
        copies = plan["configuration"]["learner"]["n_copies"]
        require(type(copies) is int and copies > 0 and len(plan["copy_seeds"][index]) == copies,
            "qualification copy count must match its independent policy seed count")
        # Before: one copy has 819200 warmup and 16384 training transitions; two copies double both totals.
        # Iteration and optimizer counters remain the per-copy full-update counters.
        require(result.get("all_original_gates_passed") is True
            and type(result.get("reference_iteration")) is int and result["reference_iteration"] == 1
            and type(result.get("reference_optimizer_step")) is int and result["reference_optimizer_step"] == 16
            and files.encoded(result.get("executed_work")) == files.encoded({"normalization_warmup_transitions": 819200 * copies,
                "training_transitions": 16384 * copies, "complete_updates": 1, "minibatch_updates": 16}),
            "the full original controlled reference update is incomplete")
    elif plan["mode"] == "correctness":
        require(result.get("correctness_attempted") is True and result.get("correctness_passed") is True
            and result.get("implementation_distinct") is True and result.get("measurement_valid") is False,
            "backend completion cannot precede actual distinct-implementation correctness")
    else:
        copies, iterations = plan["configuration"]["learner"]["n_copies"], plan["iterations"]
        require(type(copies) is int and copies > 0 and type(iterations) is int and iterations >= 10
            and type(plan["rounds"]) is int and plan["rounds"] == 7
            and type(result.get("rounds")) is list and len(result["rounds"]) == 7,
            "backend completion needs all seven complete original update windows")
        if plan["mode"] == "exact_eight_eligibility":
            require(plan["gpu_count"] == 1 and copies == 8,
                "exact-eight evidence must measure eight independent copies on one actual worker")
        previous = None
        for number, row in enumerate(result["rounds"], 1):
            origin = row["report_origin_iteration"]
            require(type(origin) is int and origin >= 0 and (previous is None or origin == previous)
                and type(row["round"]) is int and row["round"] == number
                and type(row["iterations"]) is int and row["iterations"] == iterations
                and type(row["copies"]) is int and row["copies"] == copies
                and type(row["environments_per_copy"]) is int and row["environments_per_copy"] == 128
                and type(row["rollout_steps"]) is int and row["rollout_steps"] == 128
                and type(row["training_transitions"]) is int
                and row["training_transitions"] == iterations * 128 * 128 * copies
                and type(row["actual_end_iteration"]) is int and row["actual_end_iteration"] == origin + iterations
                and type(row["actual_optimizer_step"]) is int and row["actual_optimizer_step"] == (origin + iterations) * 16,
                "a worker window is partial or changed original work counters")
            previous = row["actual_end_iteration"]
    return plan["mode"]


def capture_complete(directory, plan, result):
    """Retain the same own package image and actual post-work installed-file receipts."""
    directory = files.ordinary(directory, directory=True)
    require(not (directory / "group_backend_capture.json").exists()
        and not (directory / "group_backend_capture.json").is_symlink(), "preserve the prior backend completion")
    started = time.time()
    expected, actual = validate_prepared(plan, retain_actual=True)
    finished = time.time()
    begin = files.checked_reference(files.reference(directory / "group_backend_capture_begin.json"))
    metadata = files.checked_reference(files.reference(directory / "group_step_metadata.json"))
    binding = files.checked_reference(files.reference(directory / "resolved_device_binding.json"))
    index, process = process_join(directory, plan, metadata, binding, current=True)
    after = files.mapped_snapshot(expected["image"]["path"])
    require(begin["process"] == process and after["file_identity"] == begin["image"]["source_file_identity"],
        "post-work mapped image or actual process changed")
    mode = completed_work(directory, plan, result)
    value = {"schema_version": 2, "kind": "group_worker_captured_backend_package_image", "capability": CAPABILITY,
        "worker_index": index, "plan": files.reference(group_directory(plan) / "scaling_plan.json"),
        "expected": plan["group_backend_image"]["expected"],
        "metadata_input": files.reference(directory / "group_step_metadata.json"),
        "binding": files.reference(directory / "resolved_device_binding.json"),
        "begin": files.reference(directory / "group_backend_capture_begin.json"), "gpu_uuid": binding["gpu_uuid"],
        "process": process, "maps_after_numerical_work": after,
        "post_installed_package": {"started_at_seconds": started, "finished_at_seconds": finished, "description": actual},
        "loaded_gpu_libraries_sha256": files.digest(files.encoded(result["loaded_gpu_libraries"])),
        "numerical_execution": result["execution_identity"], "completed_work_kind": mode,
        "scientific_credit": 0, "admission_granted": False}
    files.publish(directory / "group_backend_capture.json", value)
    validate_capture(directory, plan, metadata, binding, result["loaded_gpu_libraries"], result)
    return value


def validate_capture(directory, plan, metadata, binding, loaded, result):
    """Verify exact shared image bytes and same-process pre/post evidence without GPU calls."""
    directory = files.ordinary(directory, directory=True)
    expected = expected_for_plan(plan)
    index, process = process_join(directory, plan, metadata, binding, current=False)
    begin = files.checked_reference(files.reference(directory / "group_backend_capture_begin.json"))
    final = files.checked_reference(files.reference(directory / "group_backend_capture.json"))
    for value, kind in ((begin, "group_worker_backend_package_capture_begin"),
            (final, "group_worker_captured_backend_package_image")):
        require(type(value.get("schema_version")) is int and value["schema_version"] == 2
            and value.get("kind") == kind and value.get("capability") == CAPABILITY
            and type(value.get("worker_index")) is int and value["worker_index"] == index
            and value["plan"] == files.reference(group_directory(plan) / "scaling_plan.json")
            and value["expected"] == plan["group_backend_image"]["expected"]
            and value["metadata_input"] == files.reference(directory / "group_step_metadata.json")
            and value["binding"] == files.reference(directory / "resolved_device_binding.json")
            and value["gpu_uuid"] == binding["gpu_uuid"]
            and all(value["process"][key] == process[key] for key in ("hostname", "pid", "process_start_time", "uid"))
            and type(value.get("scientific_credit")) is int and value["scientific_credit"] == 0
            and value.get("admission_granted") is False,
            "backend proof changed its group, worker, source pin or actual bound process")
    require(final["begin"] == files.reference(directory / "group_backend_capture_begin.json")
        and final["process"] == begin["process"] and type(loaded["pid"]) is int and loaded["pid"] == process["pid"]
        and files.encoded(result["loaded_gpu_libraries"]) == files.encoded(loaded)
        and final["loaded_gpu_libraries_sha256"] == files.digest(files.encoded(loaded))
        and final["completed_work_kind"] == completed_work(directory, plan, result),
        "backend completion changed the separate actual CUDA receipt or completed work")
    image = begin["image"]
    require(image["original"] == {key: expected["image"][key] for key in ("path", "sha256")}
        and image["captured"] == {"path": str(directory / "group_backend/xla_cuda_plugin.so"), "sha256": expected["image"]["sha256"]}
        and type(image["size_bytes"]) is int and image["size_bytes"] == expected["image"]["size_bytes"],
        "captured backend does not match the exact prepared group package image")
    actual = files.stream_image(image["captured"]["path"])
    require(actual["sha256"] == expected["image"]["sha256"] and actual["size_bytes"] == image["size_bytes"],
        "shared captured backend bytes changed")
    for name, filename in (("metadata", "METADATA"), ("record", "RECORD")):
        value = begin[name]
        require(value["original"] == expected["distribution"][name]
            and value["captured"] == {"path": str(directory / "group_backend" / filename),
                "sha256": expected["distribution"][name]["sha256"]}
            and files.digest(files.small_bytes(value["captured"]["path"])) == value["captured"]["sha256"],
            "captured owning package bytes changed")
    require(files.package_headers(files.small_bytes(begin["metadata"]["captured"]["path"])) ==
        {key: expected["distribution"][key] for key in ("name", "version")}
        and files.record_entry(files.small_bytes(begin["record"]["captured"]["path"]),
            expected["distribution"]["record_entry"]["relative_path"], expected["image"]) ==
            expected["distribution"]["record_entry"], "shared package RECORD no longer owns the captured image")
    maps = [begin["maps_before_copy"], begin["maps_after_copy"], final["maps_after_numerical_work"]]
    post = final["post_installed_package"]
    require(type(post) is dict and set(post) == {"started_at_seconds", "finished_at_seconds", "description"},
        "actual post-work package hash pass is missing")
    package = post["description"]
    require(type(package) is dict and set(package) == {"site_packages", "image", "distribution",
        "observed_file_identity", "observed_descriptor_before", "observed_descriptor_after"}
        and all(package[key] == expected[key] for key in ("site_packages", "image", "distribution"))
        and package["observed_file_identity"] == image["source_file_identity"]
        and package["observed_descriptor_before"] == package["observed_descriptor_after"] == image["descriptor_before"],
        "actual post-work package bytes or pinned descriptor changed")
    execution = result["execution_identity"]
    require(type(execution) is dict and type(execution.get("pid")) is int and execution["pid"] > 0
        and type(execution.get("hostname")) is str and type(execution.get("process_start_time")) is str
        and files.encoded(final["numerical_execution"]) == files.encoded(execution)
        and all(type(execution[key]) is type(process[key]) and execution[key] == process[key]
            for key in ("hostname", "pid", "process_start_time"))
        and all(type(item[key]) in {int, float} and math.isfinite(item[key])
            for item in (execution, post) for key in ("started_at_seconds", "finished_at_seconds"))
        and expected["created_at_seconds"] <= execution["started_at_seconds"] <= maps[0]["observed_at_seconds"]
        <= maps[1]["observed_at_seconds"] <= execution["finished_at_seconds"]
        <= post["started_at_seconds"] <= post["finished_at_seconds"] <= maps[2]["observed_at_seconds"],
        "backend observations do not surround this actual completed worker execution")
    require(image["descriptor_before"] == image["descriptor_after"]
        and image["descriptor_before"]["file_identity"] == image["source_file_identity"]
        and image["descriptor_before"]["size_bytes"] == expected["image"]["size_bytes"]
        and image["descriptor_before"]["regular_file"] is True,
        "runtime copied image descriptor changed")
    previous = expected["created_at_seconds"]
    for value in maps:
        require(value["path"] == "/proc/self/maps" and files.digest(value["raw"].encode()) == value["sha256"]
            and files.mapping_rows(value["raw"], expected["image"]["path"])[0] ==
                value["file_identity"] == image["source_file_identity"]
            and type(value["observed_at_seconds"]) in {int, float}
            and math.isfinite(value["observed_at_seconds"]) and value["observed_at_seconds"] >= previous,
            "actual own-process mapping bytes, inode or time changed")
        previous = value["observed_at_seconds"]
    return final

