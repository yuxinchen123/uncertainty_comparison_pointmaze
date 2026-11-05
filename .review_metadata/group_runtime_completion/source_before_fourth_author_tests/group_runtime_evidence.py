"""Read complete group workers and common-wall results on explicitly owned compute.

All physical identities come from saved same-process bindings. These readers do
not query GPUs, infer scheduler-to-physical ordering, or admit ordinary/final work.
"""
import math
import os
from pathlib import Path
import re

import uva_group_binding as group

package = group.adjacent("group_backend_package")
configured = group.adjacent("configured_measurement")
selected = group.adjacent("measure_selected")
OWNER = 2618919
REVIEW_KIND = "independent_configured_group_member_review"


def require_owned_compute(plan):
    """Refuse raw scientific recomputation outside a numeric step of this campaign's owned job."""
    job, step = os.environ.get("SLURM_JOB_ID", ""), os.environ.get("SLURM_STEP_ID", "")
    require = group.require
    require(re.fullmatch(r"[1-9][0-9]*", job) and re.fullmatch(r"[0-9]+", step)
        and os.getuid() == os.geteuid() == OWNER, "group evidence requires an actual owned numeric compute step")
    ids = (Path(plan["campaign_run_dir"]) / "slurm/submitted_jobids_uva.txt").read_text().splitlines()
    require(ids.count(job) == 1, "the current evidence-reading allocation is absent from this campaign's owned ledger")


def validate_compilation_warmup(rows, iterations, origin):
    """Rebuild the actual nonzero training origin from complete successful compilation calls."""
    group.require(type(iterations) is int and iterations >= 10 and type(origin) is int and origin > 0
        and isinstance(rows, list) and 2 <= len(rows) <= 6, "complete compilation warmup calls and their actual origin are required")
    total, previous = 0, None
    for number, row in enumerate(rows, 1):
        group.require(type(row) is dict and type(row.get("call")) is int and row["call"] == number
            and type(row.get("updates")) is int and row["updates"] == iterations
            and type(row.get("seconds")) in {int, float} and math.isfinite(row["seconds"]) and row["seconds"] > 0,
            "compilation warmup changed a full update count, duration or order")
        sizes = row.get("compiled_signatures")
        group.require(isinstance(sizes, list) and sizes and all(type(x) is int and x > 0 for x in sizes),
            "actual compiled-signature counts are required")
        total += row["updates"]
        if number == len(rows):
            group.require(sizes == previous, "the final two compilation calls did not have stable signatures")
        previous = sizes
    group.require(total == origin, "compilation warmup training transitions cannot be omitted or relabelled as iteration zero")
    return total


def validate_round_counters(plan, result):
    """Join all seven complete windows to the actual compilation-warmup counters."""
    origin = validate_compilation_warmup(result.get("warmup"), plan["iterations"], result.get("training_start_iteration"))
    rows = result.get("rounds")
    group.require(isinstance(rows, list) and len(rows) == 7, "all seven actual worker windows are required")
    for number, row in enumerate(rows, 1):
        start = origin + (number - 1) * plan["iterations"]
        group.require(type(row.get("training_start_iteration")) is int and row["training_start_iteration"] == origin
            and type(row.get("report_origin_iteration")) is int and row["report_origin_iteration"] == start
            and type(row.get("actual_end_iteration")) is int and row["actual_end_iteration"] == start + plan["iterations"]
            and type(row.get("actual_optimizer_step")) is int and row["actual_optimizer_step"] == (start + plan["iterations"]) * 16,
            "a worker window changed its actual training or optimizer origin")
    return origin


def validate_original_gates(directory, plan, result):
    """Rebuild complete original test/renderer outcomes and count-specific controlled work."""
    original = group.adjacent("group_original_qualification")
    original.validate_configuration(plan)
    configured.verify_qualification_source_receipts(plan)
    tests = group.read_receipt(group.receipt(directory / "original_tests.json"))
    configured.validate_test_collection(plan, tests, (directory / "original_tests.stdout").read_text())
    group.require(type(tests.get("returncode")) is int and tests["returncode"] == 0
        and tests["reports"] == result["original_test_reports"]
        and tests["collected_nodeids"] == result["original_collected_test_nodeids"],
        "the completed original test matrix differs from its actual raw reports")
    renderer = group.read_receipt(group.receipt(directory / "original_renderer.json"))
    group.require(renderer == result["renderer"] and renderer.get("checks") == {
        "cache": "passed", "palette": "passed", "fused": "passed"}
        and group.exact(result.get("renderer_contract"), plan["renderer_contract"])
        and all(type(renderer.get(key)) is int and renderer[key] == plan["renderer_contract"][key]
            for key in ("rooms", "cases", "synthetic_pixel_cases")), "the original full renderer matrix changed")
    group.require(set(result.get("gate_artifacts", {})) == {
        "original_tests.json", "original_tests.stdout", "original_tests.stderr", "original_renderer.json"}
        and all(group.receipt(directory / name) == item for name, item in result["gate_artifacts"].items())
        and group.exact(result.get("dtypes"), configured.EXPECTED_DTYPES)
        and type(result.get("finite_state_leaves")) is int and result["finite_state_leaves"] > 0
        and result.get("frozen_target_unchanged") is True and result.get("policy_parameters_changed") is True,
        "full original gates require their exact raw artifacts, finite precision groups and actual policy update")
    memory = result["qualification_memory"]
    proof = memory["assigned_gpu_process_observation"]
    rebuilt = configured.validate_window(proof)
    interval, binding = result["qualification_interval"], result["resolved_device_binding"]
    group.require(all(group.exact(memory.get(key), value) for key, value in rebuilt.items())
        and proof["gpu_uuid"] == binding["gpu_uuid"] and proof["pid"] == binding["process"]["pid"]
        and all(type(interval[key]) in {int, float} and math.isfinite(interval[key])
            for key in ("started_at_seconds", "finished_at_seconds"))
        and proof["window_started_at_seconds"] <= interval["started_at_seconds"]
        < interval["finished_at_seconds"] <= proof["window_finished_at_seconds"],
        "the complete own-only memory window must cover the actual controlled original gates and update")


def verify_library_evidence(directory, plan, metadata, binding, result):
    """Keep CUDA/cuDNN receipts separate from the captured actual owning backend package."""
    installed = group.read_receipt(group.receipt(directory / "installed_backend.json"))
    loaded = group.read_receipt(group.receipt(directory / "loaded_gpu_libraries.json"))
    group.require(result["installed_backend"] == installed and result["loaded_gpu_libraries"] == loaded
        and type(loaded.get("pid")) is int and loaded["pid"] == metadata["process"]["pid"],
        "loaded library evidence belongs to another actual numerical process")
    selected.training.validate_gpu_library_receipt(loaded, {"installed_gpu_libraries": installed["gpu_libraries"]})
    for path, receipt in loaded["verified_package_libraries"].items():
        group.require(group.receipt(path)["sha256"] == receipt["sha256"], "an actual installed CUDA/cuDNN image changed")
    return package.validate_capture(directory, plan, metadata, binding, loaded, result)


def validate_controlled_window(result, name):
    """Require complete own-only memory coverage around the actual named controlled operation."""
    memory, interval = result[name + "_memory"], result[name + "_interval"]
    proof, binding = memory["assigned_gpu_process_observation"], result["resolved_device_binding"]
    rebuilt = configured.validate_window(proof)
    group.require(all(group.exact(memory.get(key), value) for key, value in rebuilt.items())
        and proof["gpu_uuid"] == binding["gpu_uuid"] and proof["pid"] == binding["process"]["pid"]
        and all(type(interval[key]) in {int, float} and math.isfinite(interval[key])
            for key in ("started_at_seconds", "finished_at_seconds"))
        and proof["window_started_at_seconds"] <= interval["started_at_seconds"]
        < interval["finished_at_seconds"] <= proof["window_finished_at_seconds"],
        "the complete own-only observation does not cover the actual controlled " + name + " operation")


def verify_saved_member(directory, *, terminal=True):
    """Recompute one actual worker's source, binding, package, counts and complete operation."""
    directory = Path(directory)
    parent = directory.parent.parent
    plan = group.read_receipt(group.receipt(parent / "scaling_plan.json"))
    require_owned_compute(plan)
    runtime = group.adjacent("group_runtime")
    runtime.validate_plan(plan)
    group.adjacent("configured_scaling_evidence").verify_sources(parent, plan)
    configured.verify_qualification_source_receipts(plan)
    index = package.worker_index(directory, plan)
    group.source_identity(plan)
    metadata = group.read_receipt(group.receipt(directory / "group_step_metadata.json"))
    binding = group.read_receipt(group.receipt(directory / "resolved_device_binding.json"))
    group.validate_binding(plan, metadata, binding, directory)
    identity = group.read_receipt(group.receipt(directory / "group_process_identity.json"))
    runtime.validate_process(identity, metadata, runtime.numeric_command(plan, "worker", index))
    result = group.read_receipt(group.receipt(directory / "result.json"))
    group.require(result.get("schema_version") == 2 and type(result.get("schema_version")) is int
        and result.get("kind") == package.MODES[plan["mode"]] and result.get("completed") is True
        and group.exact(result.get("configuration"), plan["configuration"])
        and group.exact(result.get("reference_configuration"), plan["reference_configuration"])
        and group.exact(result.get("copy_seeds"), plan["copy_seeds"][index])
        and result.get("source_commit") == plan["source_commit"] and result.get("helpers") == plan["helpers"]
        and result.get("runtime_environment") == plan["runtime_environment"]
        and result.get("asset_record") == plan["asset_record"]
        and result.get("group_metadata") == group.receipt(directory / "group_step_metadata.json")
        and result.get("resolved_device_binding") == binding
        and result.get("worker_index") == index and type(result.get("worker_index")) is int
        and type(result.get("scientific_credit")) is int and result["scientific_credit"] == 0
        and result.get("admission_granted") is False, "worker result changed its source, configuration, seeds or actual member")
    package.completed_work(directory, plan, result)
    verify_library_evidence(directory, plan, metadata, binding, result)
    if plan["mode"] == "qualification":
        validate_original_gates(directory, plan, result)
    elif plan["mode"] == "correctness":
        group.require(result.get("correctness_attempted") is True and result.get("correctness_passed") is True
            and result.get("implementation_distinct") is True and result.get("measurement_valid") is False
            and set(result["correctness"]) == {"initialization", "passed", "supplied_inputs", "complete_iterations", "complete_iteration"}
            and result["correctness"]["passed"] is True and type(result["correctness"]["complete_iterations"]) is int
            and result["correctness"]["complete_iterations"] > 0,
            "count-specific correctness must retain all unchanged comparison outputs")
        selected.validate_correctness_protocol([plan["reference_configuration"], plan["configuration"]])
        validate_controlled_window(result, "correctness")
    else:
        validate_round_counters(plan, result)
    if terminal:
        resolved = group.read_receipt(group.receipt(parent / "resolved_group_binding.json"))
        coordinator = group.read_receipt(group.receipt(parent / "group_step_metadata.json"))
        coordinator_identity = group.read_receipt(group.receipt(parent / "group_process_identity.json"))
        runtime.validate_process(coordinator_identity, coordinator, runtime.numeric_command(plan, "coordinator"))
        rebuilt = group.resolve_group(plan, coordinator,
            [{key: row[key] for key in ("metadata", "binding")} for row in resolved["members"]])
        group.require(group.exact(resolved, rebuilt) and resolved["members"][index]["binding"] ==
            group.receipt(directory / "resolved_device_binding.json"), "the actual worker is absent from its complete resolved group")
        group.adjacent("configured_scaling_evidence").verify_terminal(parent, plan, resolved, coordinator)
        runtime.verify_wrapper_waits(plan)
    return result, plan, binding


def reviewed_member(item, mode, plan, member):
    """Require a genuine independent review of the exact matching physical worker and count."""
    group.require(type(item) is dict and set(item) == {"result", "review"}, "each member prerequisite needs exact result and independent review")
    path = Path(item["result"]["path"])
    group.require(path.name == "result.json" and group.receipt(path) == item["result"], "member prerequisite bytes changed")
    result, previous, binding = verify_saved_member(path.parent)
    review = group.read_receipt(item["review"])
    group.require(previous["mode"] == mode and binding["gpu_uuid"] == member["gpu_uuid"]
        and previous["source_commit"] == plan["source_commit"] and previous["helpers"] == plan["helpers"]
        and previous["runtime_environment"] == plan["runtime_environment"]
        and previous["asset_record"] == plan["asset_record"]
        and previous["configuration"]["learner"]["n_copies"] == plan["configuration"]["learner"]["n_copies"],
        "the prerequisite differs from the actual physical target, source, packages or independent copy count")
    before, current = package.expected_for_plan(previous), package.expected_for_plan(plan)
    group.require(all(before[key] == current[key] for key in ("selected_python", "site_packages", "image", "distribution"))
        and previous["qualification_software"] == plan["qualification_software"],
        "member prerequisite used another actual installed package, Python or original software input")
    if mode == "correctness":
        group.require(previous["configuration"] == plan["configuration"]
            and previous["reference_configuration"] == plan["reference_configuration"],
            "member correctness used another actual selected implementation")
    job = group.owned_inputs(previous)["job_id"]
    group.require(type(review.get("schema_version")) is int and review["schema_version"] == 2
        and review.get("kind") == REVIEW_KIND and review.get("approved") is True
        and review.get("result") == item["result"] and review.get("mode") == mode
        and review.get("job_id") == job and review.get("actual_terminal_state") == "COMPLETED"
        and review.get("actual_exit_code") == "0:0" and review.get("admission_granted") is False
        and type(review.get("scientific_credit")) is int and review["scientific_credit"] == 0,
        "an actual terminal independent member review is required")
    return result


def verify_prerequisites(plan, resolved):
    """Match actual physical devices, rather than assuming repeated scheduler worker order."""
    if plan["mode"] == "qualification":
        return {}
    required = ["qualification"] if plan["mode"] == "correctness" else ["qualification", "correctness"]
    result = {}
    for mode in required:
        rows = plan.get("member_" + mode)
        group.require(isinstance(rows, list) and len(rows) >= plan["gpu_count"], "complete actual count-specific member prerequisites are missing")
        by_uuid = {}
        for item in rows:
            raw = group.read_receipt(item["result"])
            uuid = raw["resolved_device_binding"]["gpu_uuid"]
            group.require(uuid not in by_uuid, "a physical member prerequisite was duplicated")
            by_uuid[uuid] = item
        result[mode] = []
        for member in resolved["members"]:
            group.require(member["gpu_uuid"] in by_uuid, "the actual allocated physical card lacks reviewed count-specific evidence")
            result[mode].append(reviewed_member(by_uuid[member["gpu_uuid"]], mode, plan, member))
    return result


def verify_saved_group(directory, *, allow_eligibility=False):
    """Rebuild every member, full common-wall interval, lifecycle and reviewed study prerequisite."""
    directory = Path(directory)
    plan = group.read_receipt(group.receipt(directory / "scaling_plan.json"))
    require_owned_compute(plan)
    runtime = group.adjacent("group_runtime")
    runtime.validate_plan(plan)
    aggregation = group.adjacent("configured_scaling_evidence")
    group.require(plan["mode"] in {"measurement", "exact_eight_eligibility"}
        and (allow_eligibility or plan["mode"] != "exact_eight_eligibility"), "qualification and memory-only observations are not ordinary scaling results")
    aggregation.validate_protocol(plan)
    aggregation.verify_sources(directory, plan)
    coordinator = group.read_receipt(group.receipt(directory / "group_step_metadata.json"))
    runtime.validate_process(group.read_receipt(group.receipt(directory / "group_process_identity.json")),
        coordinator, runtime.numeric_command(plan, "coordinator"))
    resolved = group.read_receipt(group.receipt(directory / "resolved_group_binding.json"))
    rebuilt = group.resolve_group(plan, coordinator, [{key: row[key] for key in ("metadata", "binding")} for row in resolved["members"]])
    group.require(group.exact(resolved, rebuilt), "saved group differs from actual distinct member bindings")
    verify_prerequisites(plan, resolved)
    aggregation.verify_terminal(directory, plan, resolved, coordinator)
    runtime.verify_wrapper_waits(plan)
    result = group.read_receipt(group.receipt(directory / "result.json"))
    aggregation.verify_completed_phase(plan, result["execution_identity"]["started_at_seconds"], resolved)
    wanted = aggregation.ELIGIBILITY_KIND if plan["mode"] == "exact_eight_eligibility" else aggregation.RESULT_KIND
    group.require(type(result.get("schema_version")) is int and result["schema_version"] == 2
        and result.get("kind") == wanted and result.get("completed") is True
        and result.get("configuration") == plan["configuration"] and result.get("helpers") == plan["helpers"]
        and result.get("source_commit") == plan["source_commit"] and result.get("copy_seeds") == plan["copy_seeds"]
        and result.get("runtime_environment") == plan["runtime_environment"]
        and result.get("resolved_group_binding") == group.receipt(directory / "resolved_group_binding.json")
        and isinstance(result.get("rounds"), list) and len(result["rounds"]) == 7
        and len(result.get("worker_results", [])) == plan["gpu_count"], "complete exact group result is missing")
    previous, origins = 0., set()
    for number, row in enumerate(result["rounds"], 1):
        group.require(row["start_time_seconds"] >= previous
            and group.exact(row, aggregation.aggregate_round(plan, resolved, row["workers"], row["start_time_seconds"], row["end_time_seconds"], number)),
            "group intervals overlap or changed their actual common-wall work")
        previous = row["end_time_seconds"]
    for index, ref in enumerate(result["worker_results"]):
        worker = group.worker_directory(plan, index)
        group.require(ref == group.receipt(worker / "result.json"), "a terminal worker was substituted")
        member, member_plan, unused = verify_saved_member(worker)
        group.require(member_plan == plan and member["rounds"] == [row["workers"][index] for row in result["rounds"]],
            "common intervals differ from actual numerical-worker records")
        origins.add(member["training_start_iteration"])
    group.require(len(origins) == 1 and type(result.get("training_start_iteration")) is int
        and result["training_start_iteration"] == next(iter(origins)), "common worker compilation origins differ")
    return result, plan, resolved
