"""Prepare the display-only index from existing independently reviewed campaign result records."""
from datetime import datetime, timezone
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
from unittest import mock

PACKET = Path(__file__).resolve().parent
RUN = PACKET.parent.parent


def reference(path):
    """Use exact ordinary metadata paths and current bytes hashes, without changing existing records."""
    path = Path(path)
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def write(path, value):
    """Fsync one new proposed display index or check receipt, preserving every earlier attempt."""
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def actual_record(binding):
    """Check a standing reference against the exact current metadata bytes before using it."""
    path = Path(binding["path"])
    if reference(path) != binding:
        raise ValueError("a standing actual result or review reference changed")
    return binding


def main():
    """Join four accepted correctness and two accepted confirmation records for display only."""
    if os.environ.get("SLURM_JOB_ID") != "6783003" or not os.environ.get("SLURM_STEP_ID", "").isdigit():
        raise ValueError("an actual numeric step of owned CPU monitor 6783003 is required")
    actions_path = RUN / "next_actions.json"
    actions_before = reference(actions_path)
    actions = json.loads(actions_path.read_text())
    source_record = json.loads((PACKET / "source_snapshot.json").read_text())
    source = Path(source_record["source_root"])
    sys.path.insert(0, str(source / "10_jax_exploration_platform/scripts/atari_pixel_campaign"))
    controller = importlib.import_module("controller")
    correctness, confirmation = [], []
    aa = actions["new_common_correctness_submissions"]
    for count in ("1", "2"):
        unit = Path(aa["jobs_actual"][count]["unit"])
        correctness.append({"plan": reference(unit / "measurement_plan.json"),
                            "result": reference(unit / "result.json"),
                            "review": actual_record(aa["independent_raw_review"])})
        direct = actions["h100_common_correctness_preparation"]["copies_" + count]
        correctness.append({"plan": actual_record(direct["plan"]),
                            "result": actual_record(direct["actual_result"]),
                            "review": actual_record(direct["independent_actual_review"])})
        review = actual_record(actions["common_a4500_confirmation_preparation"]["independent_reviews"][count])
        reviewed = json.loads(Path(review["path"]).read_text())
        packet = Path(actions["common_a4500_confirmation_preparation"]["jobs_actual"][count]["held_binding"]["path"]).parent
        unit = RUN / "data" / packet.name
        confirmation.append({"plan": reference(unit / "measurement_plan.json"),
                             "result": reference(unit / "result.json"), "review": review})
    configuration_path = Path(actions["unconfirmed_common_configuration_preparation"]["directory"]) /         "candidate_cache-room-backgrounds-true_render-final-two-only-true_policy-copies-1_seed-start-1.json"
    index = {"schema_version": 1, "kind": "post_initial_reporting_evidence",
             "source_commit": "d28e6735e79788f618acb0ebfde8609db1f776af",
             "configuration_record": reference(configuration_path),
             "common_correctness": correctness, "common_confirmation": confirmation,
             "kernel_profiles": [], "packing": []}
    proposal_path = PACKET / "proposed_post_initial_reporting_evidence.json"
    write(proposal_path, index)
    configuration = controller.reporting_configuration(json.loads(configuration_path.read_text()))
    checks, failures = [], []
    for requirement, entries in (("common_correctness", correctness), ("common_confirmation", confirmation)):
        for entry in entries:
            try:
                context, copies = controller.reporting_reviewed_unit(RUN, entry, requirement, index["source_commit"], configuration, 1)
                checks.append({"requirement": requirement, "hardware_class": context, "copies": copies, "passed": True})
            except Exception as error:
                failures.append({"requirement": requirement, "plan": entry["plan"], "error": repr(error)})
    actual_read = controller.read_json
    fixture = dict(actions)
    fixture["post_initial_reporting_evidence"] = index
    def read_display_actions(path):
        """Supply only the proposed display index in memory while retaining every actual evidence path."""
        return fixture if Path(path) == actions_path else actual_read(path)
    with mock.patch.object(controller, "read_json", side_effect=read_display_actions):
        requirements = controller.post_initial_reporting_requirements(RUN, index["source_commit"], [1, 2], 1)
    expected_missing = {
        "common_correctness": [],
        "common_confirmation": [{"hardware_class": "h100", "copies": 1}, {"hardware_class": "h100", "copies": 2}],
        "kernel_profiles": [{"hardware_class": "h100", "copies": None}, {"hardware_class": "jaguar03_rtx_a4500", "copies": None}],
        "packing": [{"hardware_class": "h100", "copies": None}, {"hardware_class": "jaguar03_rtx_a4500", "copies": None}]}
    requirements_match = all(row["missing"] == expected_missing[row["name"]] and not row["errors"] for row in requirements)
    unchanged = actions_before == reference(actions_path)
    value = {"schema_version": 1, "kind": "display_only_post_initial_reporting_index_join_check",
        "passed": len(checks) == 6 and not failures and requirements_match,
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "job_id": os.environ["SLURM_JOB_ID"],
        "step_id": os.environ["SLURM_STEP_ID"], "source_commit": source_record["source_commit"],
        "proposal": reference(proposal_path), "configuration_record": index["configuration_record"],
        "checks": checks, "failures": failures, "requirements": requirements,
        "requirements_match_expected_remaining_work": requirements_match,
        "next_actions_before": actions_before, "next_actions_after": reference(actions_path),
        "next_actions_unchanged_during_check": unchanged,
        "actual_shared_next_actions_written": False, "temporary_in_memory_display_actions_only": True,
        "scientific_evidence_recomputed": False, "new_admission_authority": False, "scientific_credit": 0}
    write(PACKET / "proposed_reporting_index_join_check.json", value)
    print(json.dumps({"passed": value["passed"], "checks": checks, "failures": failures,
        "requirements": requirements, "receipt": reference(PACKET / "proposed_reporting_index_join_check.json")}, sort_keys=True))
    return 0 if value["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

