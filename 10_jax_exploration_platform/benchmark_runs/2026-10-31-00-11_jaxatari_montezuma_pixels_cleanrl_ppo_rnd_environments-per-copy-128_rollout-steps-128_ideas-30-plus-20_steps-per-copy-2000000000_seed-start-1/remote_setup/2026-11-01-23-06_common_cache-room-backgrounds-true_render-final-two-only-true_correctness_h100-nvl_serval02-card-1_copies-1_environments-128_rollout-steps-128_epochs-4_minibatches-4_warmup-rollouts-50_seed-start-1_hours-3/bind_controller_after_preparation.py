"""Bind a reviewed controller to actual completed metadata and actual prior exits only."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path

P = Path(__file__).resolve(strict=True).parent
spec = importlib.util.spec_from_file_location("fresh_correctness_controller_binding", P / "direct_qualification_controller.py")
controller = importlib.util.module_from_spec(spec)
spec.loader.exec_module(controller)


def check_new_binding_paths(packet):
    """Reject prior ordinary and dangling temporary attempt paths before creating any new intent."""
    for name in ("controller_binding_intent.json", "serial_predecessor_exit_inputs.json", "packet_metadata_freeze.json",
            "controller_bound_metadata.json", "controller_source_freeze.json", "controller_binding_result.json"):
        path = packet / name
        temporary = path.with_name(path.name + ".prepared")
        controller.require(not path.exists() and not path.is_symlink()
            and not temporary.exists() and not temporary.is_symlink(),
            "preserve the earlier metadata binding attempt: " + name)


def main(argv=None):
    """Create one immutable bound packet after CPU preparation and real predecessor exits."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exit-observation", required=True, type=Path)
    args = parser.parse_args(argv)
    request = controller.document(P / "preparation_request.json")
    unit = Path(request["data_unit"])
    controller.require(request["packet"] == str(P) and unit == controller.UNIT,
        "the reviewed request must name this exact actual packet and unit")
    check_new_binding_paths(P)
    controller.check_unattempted(unit)
    prepared = controller.document(P / "metadata_preparation_result.json")
    software = controller.document(P / "shared_software_assets_staged_verification.json")
    plan = controller.document(unit / "measurement_plan.json")
    controller.require(prepared.get("passed") is True and software.get("passed") is True
        and prepared["plan"] == software["plan"] == controller.receipt(unit / "measurement_plan.json"),
        "actual CPU preparation and current software proof must both pass for this exact real plan")
    previous = Path(request["serial_predecessor_unit"])
    previous_packet = controller.RUN / "remote_setup" / previous.name
    observation = args.exit_observation
    controller.require(observation.is_absolute() and observation.parent == previous_packet
        and observation.resolve(strict=True) == observation,
        "the actual exit observation must belong to this exact prior unit packet")
    result_path = previous / ("direct_qualification_result.json" if request["serial_order"] == 1 else "result.json")
    previous_files = {"plan": previous / "measurement_plan.json", "result": result_path,
        "terminal": previous / "direct_terminal_process.json", "wait": previous / "direct_process_wait_return.json",
        "outcome": previous / "direct_controller_outcome.json", "metadata": previous / "direct_process_metadata.json",
        "controller_spawn": previous_packet / "root_actual_controller_spawn.json", "exit_observation": observation}
    references = {name: controller.receipt(path) for name, path in previous_files.items()}
    controller.publish(P / "controller_binding_intent.json", {"schema_version": 1,
        "kind": "one_actual_correctness_controller_metadata_binding", "request": controller.receipt(P / "preparation_request.json"),
        "preparation": controller.receipt(P / "metadata_preparation_result.json"),
        "software": controller.receipt(P / "shared_software_assets_staged_verification.json"),
        "actual_predecessor_unit": str(previous), "actual_predecessor_records": references,
        "at_utc": controller.now(), "host_contact": False, "gpu_operations": 0})
    serial = {"schema_version": 1, "kind": "actual_owned_direct_serial_predecessor_exit_inputs",
        "predecessor_unit": str(previous), **references}
    serial_ref = controller.publish(P / "serial_predecessor_exit_inputs.json", serial)
    controller.check_serial_predecessor_inputs(request, serial_ref)
    # Freeze the same mandatory metadata and complete helper closure that the reader independently requires.
    files = controller.packet_metadata_paths(plan)
    packet_freeze = {"schema_version": 1, "kind": "actual_direct_common_correctness_packet_metadata_freeze",
        "files": {str(path): controller.receipt(path)["sha256"] for path in sorted(set(files))},
        "gpu_process_started": False, "scientific_credit": 0, "case_credit": 0, "idea_credit": 0}
    freeze_ref = controller.publish(P / "packet_metadata_freeze.json", packet_freeze)
    binding = {"schema_version": 1, "kind": "actual_prepared_direct_common_correctness_controller_inputs",
        "plan": controller.receipt(unit / "measurement_plan.json"),
        "preparation": controller.receipt(P / "metadata_preparation_result.json"),
        "software": controller.receipt(P / "shared_software_assets_staged_verification.json"),
        "packet_freeze": freeze_ref, "request": controller.receipt(P / "preparation_request.json"),
        "serial_predecessor_exit": serial_ref}
    bound_ref = controller.publish(P / "controller_bound_metadata.json", binding)
    controller.checked_inputs()
    source_files = [P / name for name in ("direct_qualification_controller.py", "test_direct_qualification_controller.py",
        "run_direct_controller_metadata_tests.py", "bind_controller_after_preparation.py", "controller_bound_metadata.json")]
    source = {"schema_version": 1, "kind": "direct_qualification_controller_source_freeze",
        "files": {str(path): controller.receipt(path)["sha256"] for path in source_files},
        "controller_owns_gpu_standing_descriptor": False, "producer_owns_gpu_standing_descriptor": True,
        "actual_controller_pid": None, "actual_numerical_child_pid": None, "scientific_credit": 0}
    source_ref = controller.publish(P / "controller_source_freeze.json", source)
    result = {"schema_version": 1, "kind": "actual_direct_common_correctness_controller_bound_metadata",
        "passed": True, "at_utc": controller.now(), "plan": binding["plan"],
        "bound_inputs": bound_ref, "packet_freeze": freeze_ref, "source_freeze": source_ref,
        "serial_predecessor_exit": serial_ref, "files_frozen": len(packet_freeze["files"]),
        "launch_performed": False, "host_contact": False, "gpu_operations": 0, "scientific_credit": 0}
    controller.publish(P / "controller_binding_result.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
