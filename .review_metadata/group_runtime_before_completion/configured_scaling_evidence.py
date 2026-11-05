"""Read explicit configured A4500 group measurements using common wall time.

These readers are separate from the historical/default final readers. Native
files only are supported here. Capacity failures and portable/final admission
remain unsupported. An exact-eight completed measurement can supply a memory
headroom exclusion; a smaller failed count cannot.
"""
import copy
import hashlib
import math
from pathlib import Path
import statistics

import uva_group_binding as group

validation = group.adjacent("post_comparison_validation")
configured = group.adjacent("configured_measurement")
SCOPE = "one_shared_host_monotonic_group_interval"
RESULT_KIND = "configured_a4500_common_wall_time_result"
ELIGIBILITY_KIND = "configured_a4500_exact_eight_completed_memory_observation"


def require(value, message):
    """Reject missing or contradictory group measurements."""
    group.require(value, message)


def validate_protocol(plan):
    """Keep full original work and global independent seed identities explicit."""
    group.validate_plan(plan)
    configuration = validation.configuration(plan["configuration"])
    validation.runtime_environment(plan["runtime_environment"])
    copies, count = configuration["learner"]["n_copies"], plan["gpu_count"]
    protocol = plan["scaling_protocol"]
    require(set(protocol) == {"kind", "global_copy_seeds", "gradient_averaging", "measurement_scope"}
        and protocol["kind"] in {"same_copies_per_card", "fixed_total_eight", "exact_eight_eligibility"}
        and protocol["gradient_averaging"] is False and protocol["measurement_scope"] == SCOPE,
        "explicit independent-copy scaling protocol is required")
    seeds = plan["copy_seeds"]
    flat = [seed for block in seeds for seed in block]
    require(len(seeds) == count and all(len(block) == copies for block in seeds)
        and len(flat) == len(set(flat)) == copies * count
        and all(type(seed) is int and 0 <= seed < 2 ** 32 for seed in flat)
        and flat == protocol["global_copy_seeds"]
        and flat == list(range(flat[0], flat[0] + len(flat))), "global copy seed identities changed")
    require(plan["rounds"] == 7 and type(plan["rounds"]) is int
        and plan["minimum_updates_per_round"] == 10 and type(plan["minimum_updates_per_round"]) is int
        and plan["iterations"] == validation.complete_window([configuration], 10),
        "scaling needs seven complete windows of at least ten original updates")
    if protocol["kind"] == "fixed_total_eight":
        require(copies * count == 8, "fixed-total comparison is exactly eight independent copies")
    if protocol["kind"] == "exact_eight_eligibility":
        require(count == 1 and copies == 8, "memory eligibility needs an actual exact-eight one-card measurement")
    require(plan.get("compilation_cache_policy") == "disabled"
        and not any(key in plan for key in ("final_protocol", "device_memory_limit_protocol", "portable_evidence")),
        "unsupported final, instrumented or portable group protocol")
    return configuration


def worker_round(plan, binding, row, start, end, round_index, *, require_headroom=True):
    """Rebuild complete raw samples and exact work within the common group interval."""
    copies = plan["configuration"]["learner"]["n_copies"]
    index = binding["index"]
    physical = group.read_receipt(binding["binding"])
    process = physical["process"]
    require(type(row.get("round")) is int and row["round"] == round_index
        and row.get("gpu_uuid") == binding["gpu_uuid"]
        and row.get("copy_seeds") == plan["copy_seeds"][index]
        and row.get("pid") == process["pid"] and row.get("process_start_time") == process["process_start_time"]
        and row.get("step_id") == binding["step"] and row.get("iterations") == plan["iterations"],
        "raw worker changed its round, actual physical process, step, seeds or full work")
    validation.measured_round(row, copies)
    first, last, duration = row["start_time_seconds"], row["end_time_seconds"], row["seconds"]
    require(all(type(value) in {int, float} and math.isfinite(value) for value in (first, last, duration))
        and start <= first < last <= end
        and abs(last - first - duration) <= max(1e-6, duration * 1e-6),
        "raw worker interval is not contained in its group interval")
    proof = row["memory"]["assigned_gpu_process_observation"]
    rebuilt = configured.validate_window(proof)
    require(proof["gpu_uuid"] == binding["gpu_uuid"] and proof["pid"] == process["pid"]
        and proof["window_started_at_seconds"] <= first < last <= proof["window_finished_at_seconds"]
        and all(group.exact(row["memory"].get(key), value) for key, value in rebuilt.items()),
        "the complete assigned-only sample window does not cover this actual worker")
    samples = rebuilt["device_memory_samples"]
    fractions = [sample["free_bytes"] / sample["total_bytes"] for sample in samples]
    require(fractions and (not require_headroom or min(fractions) >= .15),
        "fifteen percent actual free device memory is required")
    origin = (round_index - 1) * plan["iterations"]
    require(row.get("numerical_state_finite") is True and row.get("frozen_target_unchanged") is True
        and row.get("report_origin_iteration") == origin
        and row.get("actual_end_iteration") == origin + plan["iterations"]
        and row.get("actual_optimizer_step") == (origin + plan["iterations"]) * 16
        and group.exact(row.get("dtypes"), configured.EXPECTED_DTYPES),
        "finite full state, frozen target, exact counters and original precision evidence are required")
    return min(fractions)


def aggregate_round(plan, resolved, rows, start, end, round_index):
    """Divide actual total transitions by one shared wall interval, never sum rates."""
    validate_protocol(plan)
    require(type(round_index) is int and 1 <= round_index <= 7
        and type(start) in {int, float} and type(end) in {int, float}
        and math.isfinite(start) and math.isfinite(end) and 0 < start < end
        and len(rows) == plan["gpu_count"] == len(resolved["members"]),
        "incomplete or nonfinite common group interval")
    eligibility = plan["scaling_protocol"]["kind"] == "exact_eight_eligibility"
    headroom = [worker_round(plan, member, row, start, end, round_index, require_headroom=not eligibility)
        for member, row in zip(resolved["members"], rows)]
    require(max(row["start_time_seconds"] for row in rows) < min(row["end_time_seconds"] for row in rows),
        "physical workers did not train concurrently")
    work = sum(row["training_transitions"] for row in rows)
    require(work == plan["gpu_count"] * plan["configuration"]["learner"]["n_copies"] * 128 * 128 * plan["iterations"],
        "group work includes missing, padded or duplicated transitions")
    return {"round": round_index, "measurement_scope": SCOPE, "start_time_seconds": start,
        "end_time_seconds": end, "seconds": end - start, "training_transitions": work,
        "transitions_per_second_total": work / (end - start), "minimum_free_memory_fraction": min(headroom),
        "workers": rows}


def verify_sources(directory, plan):
    """Verify committed original numerical files and the entire actual staged closure."""
    selected = group.adjacent("measure_selected")
    root = Path(plan["source_root"])
    require(selected.diagnostic.core_identity(root, plan["source_commit"]) == plan["core"],
        "group original numerical source changed")
    for name in ("paired_source", "helpers"):
        value = plan[name]
        require(selected.training.source_receipt(Path(value["source_root"]), value["source_commit"],
            list(value["files"])) == value, "group committed source receipt changed")
    require(plan["core"]["source_root"] == plan["paired_source"]["source_root"] == plan["source_root"]
        and plan["core"]["source_commit"] == plan["paired_source"]["source_commit"] == plan["source_commit"],
        "group source roots or numerical commits differ")
    group.source_identity(plan)
    selected.validate_complete_fields(root, plan["configuration"])
    require(Path(directory) == group.unit_directory(plan), "saved group is outside its exact campaign unit")


def verify_target_qualifications(plan, resolved):
    """Require a genuine full-original qualification for every actual physical member."""
    inputs = plan.get("target_qualifications")
    require(isinstance(inputs, list) and len(inputs) == plan["gpu_count"],
        "a previous single-device qualification cannot qualify an eight-device group")
    seen = set()
    for member, item in zip(resolved["members"], inputs):
        require(set(item) == {"result", "review"}, "each member needs its actual qualification and independent review")
        directory = Path(item["result"]["path"]).parent
        original = group.read_receipt(group.receipt(directory / "measurement_plan.json"))
        outer = copy.deepcopy(original)
        outer["target_qualification_inputs"] = item
        result = configured.verify_qualification_inputs(outer)
        require(result["resolved_device_binding"]["gpu_uuid"] == member["gpu_uuid"]
            and result["source_commit"] == plan["source_commit"]
            and result["runtime_environment"] == plan["runtime_environment"]
            and result["asset_record"] == plan["asset_record"]
            and result["helpers"]["source_commit"] == plan["helpers"]["source_commit"],
            "member qualification differs from this physical target or selected source")
        require(member["gpu_uuid"] not in seen, "a physical qualification was reused for another card")
        seen.add(member["gpu_uuid"])
    return inputs



def verify_completed_phase(plan, before_seconds, resolved):
    """Require actual reviewed fifty-idea and selected-packing evidence before scaling."""
    native = validation.Evidence()
    registry = Path(plan["campaign_run_dir"]) / "study_registry.json"
    proof = validation.verify_completed_study(plan.get("completed_study"), plan["source_commit"],
        plan["runtime_environment"], native, before_seconds=before_seconds,
        registry_sha256=group.receipt(registry)["sha256"], hardware_class="jaguar03_rtx_a4500",
        selected_configuration=plan["configuration"])
    packing = plan.get("selected_packing")
    require(isinstance(packing, dict), "actual selected packing evidence is required")
    checked = validation.validate_packing(packing, native)
    require(checked.get("complete") is True and checked["source_commit"] == plan["source_commit"]
        and checked["hardware_class"] == "jaguar03_rtx_a4500"
        and validation.configuration_without_copies(checked["configuration"]) ==
            validation.configuration_without_copies(plan["configuration"])
        and checked["points"][0]["gpu_uuid"] in {member["gpu_uuid"] for member in resolved["members"]},
        "selected packing is incomplete or changed source, target or implementation")
    if plan["scaling_protocol"]["kind"] == "same_copies_per_card":
        require(plan["configuration"]["learner"]["n_copies"] == checked["selected_copies"],
            "same-copies scaling must retain the actual selected packing count")
    return proof


def verify_terminal(directory, plan, resolved, coordinator):
    """Require exact-owned job and numeric-step completed accounting without querying anything."""
    value = group.read_receipt(group.receipt(Path(directory) / "terminal_accounting.json"))
    job = group.owned_inputs(plan)["job_id"]
    command = ["sacct", "-n", "-P", "-j", job, "--format=JobIDRaw,State,ExitCode"]
    group.single.validate_query(value, command)
    rows = [line.split("|") for line in value["stdout"].splitlines() if line.strip()]
    parsed = {}
    for row in rows:
        require(len(row) in {3, 4} and (len(row) == 3 or row[3] == "")
            and (row[0] == job or row[0].startswith(job + ".")) and row[0] not in parsed,
            "terminal accounting contains a foreign, duplicate or malformed row")
        parsed[row[0]] = row[1:3]
    wanted = {job, job + "." + coordinator["environment"]["SLURM_STEP_ID"]}
    wanted.update(job + "." + member["step"] for member in resolved["members"])
    require(wanted <= set(parsed) and all(parsed[name] == ["COMPLETED", "0:0"] for name in wanted),
        "actual whole job or a numeric coordinator/worker step did not complete normally")
    return value

def verify_saved_group(directory, *, allow_eligibility=False):
    """Read native complete group evidence; grant no default/final/portable admission."""
    directory = Path(directory)
    plan = group.read_receipt(group.receipt(directory / "scaling_plan.json"))
    validate_protocol(plan)
    require(group.read_receipt(group.receipt(directory / "scaling_prepared.json")) ==
        {"plan_sha256": group.receipt(directory / "scaling_plan.json")["sha256"]},
        "frozen group plan changed")
    verify_sources(directory, plan)
    result = group.read_receipt(group.receipt(directory / "result.json"))
    resolved = group.read_receipt(group.receipt(directory / "resolved_group_binding.json"))
    coordinator = group.read_receipt(group.receipt(directory / "group_step_metadata.json"))
    rebuilt = group.resolve_group(plan, coordinator,
        [{key: row[key] for key in ("metadata", "binding")} for row in resolved["members"]])
    require(group.exact(resolved, rebuilt), "saved actual group binding changed")
    verify_target_qualifications(plan, resolved)
    verify_terminal(directory, plan, resolved, coordinator)
    verify_completed_phase(plan, result["execution_identity"]["started_at_seconds"], resolved)
    eligibility = plan["scaling_protocol"]["kind"] == "exact_eight_eligibility"
    require(not eligibility or allow_eligibility, "eligibility observations are not scaling measurements")
    require(result.get("kind") == (ELIGIBILITY_KIND if eligibility else RESULT_KIND)
        and result.get("completed") is True and result.get("configuration") == plan["configuration"]
        and result.get("source_commit") == plan["source_commit"] and result.get("helpers") == plan["helpers"]
        and result.get("copy_seeds") == plan["copy_seeds"] and result.get("runtime_environment") == plan["runtime_environment"]
        and result.get("resolved_group_binding") == group.receipt(directory / "resolved_group_binding.json")
        and result.get("normalization_warmup_iterations") == 50 and result.get("training_start_iteration") == 0
        and len(result.get("rounds", [])) == 7, "incomplete actual group result or changed source/configuration")
    previous = 0.
    for number, row in enumerate(result["rounds"], 1):
        require(row["start_time_seconds"] >= previous, "group rounds overlap or move backward")
        require(group.exact(row, aggregate_round(plan, resolved, row["workers"],
            row["start_time_seconds"], row["end_time_seconds"], number)), "saved common group work/rate changed")
        previous = row["end_time_seconds"]
    require(len(result.get("worker_results", [])) == plan["gpu_count"], "missing terminal worker results")
    for index, recorded in enumerate(result["worker_results"]):
        worker = group.worker_directory(plan, index)
        require(recorded == group.receipt(worker / "result.json"), "worker result bytes changed")
        value = group.read_receipt(recorded)
        require(value["completed"] is True and value["configuration"] == plan["configuration"]
            and value["copy_seeds"] == plan["copy_seeds"][index]
            and value["rounds"] == [row["workers"][index] for row in result["rounds"]],
            "group rounds do not match the actual saved worker")
        loaded = group.read_receipt(group.receipt(worker / "loaded_gpu_libraries.json"))
        backend = group.read_receipt(group.receipt(worker / "installed_backend.json"))
        binding = group.read_receipt(resolved["members"][index]["binding"])
        require(loaded["pid"] == binding["process"]["pid"] and value["loaded_gpu_libraries"] == loaded,
            "loaded backend does not belong to the actual physical worker")
        selected = group.adjacent("measure_selected")
        selected.training.validate_gpu_library_receipt(loaded, {"installed_gpu_libraries": backend["gpu_libraries"]})
        for path, item in loaded["verified_package_libraries"].items():
            require(group.receipt(path)["sha256"] == item["sha256"], "actual installed CUDA/cuDNN bytes changed")
        capture = group.read_receipt(group.receipt(worker / "captured_backend.json"))
        require(capture["captured"]["sha256"] == capture["original"]["sha256"]
            and group.receipt(capture["captured"]["path"]) == capture["captured"]
            and capture["original"]["path"] in loaded["loaded_shared_libraries"],
            "actual mapped backend bytes were not captured immutably")
    return result, plan, resolved


def compare_groups(one_directory, eight_directory, mode):
    """Compare matched global seeds and complete common-wall windows."""
    require(mode in {"same_copies_per_card", "fixed_total_eight"}, "unsupported scaling comparison")
    groups = [verify_saved_group(path) for path in (one_directory, eight_directory)]
    first, second = groups
    require(first[1]["gpu_count"] == 1 and second[1]["gpu_count"] == 8
        and all(item[1]["scaling_protocol"]["kind"] == mode for item in groups)
        and validation.configuration_without_copies(first[1]["configuration"]) ==
            validation.configuration_without_copies(second[1]["configuration"])
        and first[1]["source_commit"] == second[1]["source_commit"]
        and first[1]["runtime_environment"] == second[1]["runtime_environment"]
        and first[1]["asset_record"] == second[1]["asset_record"], "groups changed selected implementation or target identity")
    identities = [{(row["pid"], row["process_start_time"]) for row in item[2]["members"]} for item in groups]
    require(not (identities[0] & identities[1]) and first[2]["members"][0]["gpu_uuid"] in
        {row["gpu_uuid"] for row in second[2]["members"]},
        "one/eight placements require fresh processes and a common selected physical GPU")
    seeds = [item[1]["scaling_protocol"]["global_copy_seeds"] for item in groups]
    if mode == "fixed_total_eight":
        require(seeds[0] == seeds[1] and len(seeds[0]) == 8, "fixed-total global seed identities changed")
    else:
        require(first[1]["configuration"] == second[1]["configuration"]
            and first[1]["copy_seeds"][0] == second[1]["copy_seeds"][0], "per-card work or first global seeds changed")
    rates = [[row["transitions_per_second_total"] for row in item[0]["rounds"]] for item in groups]
    return {"complete": True, "comparison": mode, "rounds": 7,
        "median_aggregate_speedup": statistics.median(b / a for a, b in zip(*rates)),
        "global_copy_seeds": seeds, "scientific_admission_granted": False,
        "final_admission_granted": False, "portable_admission_granted": False}


def exact_eight_ineligibility(directory):
    """Accept only completed exact-eight memory evidence, never a smaller failure."""
    result, plan, resolved = verify_saved_group(directory, allow_eligibility=True)
    require(plan["scaling_protocol"]["kind"] == "exact_eight_eligibility"
        and plan["gpu_count"] == 1 and plan["configuration"]["learner"]["n_copies"] == 8
        and min(row["minimum_free_memory_fraction"] for row in result["rounds"]) < .15,
        "exact-eight exclusion needs complete actual memory below fifteen-percent headroom")
    return {"status": "not_applicable", "exact_total_copies": 8,
        "eligibility_kind": "completed_exact_eight_below_memory_headroom",
        "record": group.receipt(Path(directory) / "result.json"), "final_admission_granted": False,
        "portable_admission_granted": False}
