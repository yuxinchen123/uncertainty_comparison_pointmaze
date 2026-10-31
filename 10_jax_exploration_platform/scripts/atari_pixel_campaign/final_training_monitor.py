"""Submit or continue explicitly prepared UVA final units from an owned compute monitor.

Preparing the policy only hashes existing metadata. A tick rebuilds real measured
admission on the campaign's CPU or GPU allocation, before one bounded submission.
GPU children retain the launcher's separate jaguar03 allocation and raw gates.
"""

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import json
from pathlib import Path
import subprocess

import controller
import launch_final_training as launcher

POLICY_NAME = "monitor_policy.json"
RESOURCES = {"gpus", "walltime_seconds", "host_memory_gib_per_gpu", "cpus_per_gpu",
             "checkpoint_reserve_seconds", "allow_partial_group"}


def resources(value):
    """Require an explicit bounded jaguar03 job shape and checkpoint reserve."""
    if not isinstance(value, dict) or set(value) != RESOURCES:
        raise ValueError("final monitor policy needs every explicit resource field")
    if any(type(value[name]) is not int for name in RESOURCES - {"allow_partial_group"}):
        raise ValueError("final monitor resource counts must be integers, not booleans")
    if (type(value["allow_partial_group"]) is not bool or not 1 <= value["gpus"] <= 8
            or value["cpus_per_gpu"] != 8 or value["host_memory_gib_per_gpu"] < 64
            or value["gpus"] * value["host_memory_gib_per_gpu"] > 1000
            or value["checkpoint_reserve_seconds"] < 1500
            or not 2 * value["checkpoint_reserve_seconds"] < value["walltime_seconds"] <= 345600):
        raise ValueError("use at most eight A4500s, eight CPUs and at least 64 GiB per card, with bounded checkpointed walltime")
    return dict(value)


def unit_records(run):
    """Read existing A4500 descriptors without registering units or inventing seeds."""
    directory = Path(run) / "final_training_queue/units"
    records = {}
    for path in sorted(directory.glob("*.json")):
        unit = launcher.read(path)
        if unit.get("hardware_class") != "jaguar03_rtx_a4500":
            continue
        if path.name != unit["unit_id"] + ".json" or Path(unit["campaign_run_dir"]).resolve() != run:
            raise ValueError("a registered final descriptor names a different unit or campaign")
        records[unit["unit_id"]] = {"path": str(path), "sha256": launcher.file_digest(path)}
    return records


def prepare_policy(run, input_path):
    """Freeze caller-selected registered units and resources using metadata checks only."""
    run = Path(run).resolve(strict=True)
    proposal = launcher.read(input_path)
    if (not isinstance(proposal, dict) or set(proposal) != {"schema_version", "unit_ids", "resources"}
            or type(proposal["schema_version"]) is not int or proposal["schema_version"] != 1
            or not isinstance(proposal["unit_ids"], list) or not proposal["unit_ids"]
            or any(not isinstance(name, str) or not name for name in proposal["unit_ids"])
            or len(proposal["unit_ids"]) != len(set(proposal["unit_ids"]))):
        raise ValueError("supply explicit distinct registered final unit names and resources")
    shape = resources(proposal["resources"])
    with controller.queue_lock(run), launcher.queue_lock(run) as queue:
        records = unit_records(run)
        if set(records) != set(proposal["unit_ids"]):
            raise ValueError("monitor policy must explicitly name every registered A4500 final unit")
        for record in records.values():
            launcher.verify_unit(launcher.read(record["path"]))
        ready = controller.ready_specification(run)
        final_path = run / "final_verification.json"
        final = launcher.read(final_path)
        if final.get("kind") != "final_verification" or final.get("source_commit") != ready["source_commit"]:
            raise ValueError("prepare policy requires the current final raw-evidence receipt")
        policy = {"schema_version": 1, "kind": "uva_final_training_monitor_policy", "campaign_run_dir": str(run),
            "units": records, "resources": shape, "numerical_source": {"root": ready["source_root"], "commit": ready["source_commit"]},
            "source_admission": {"path": str(run / "SOURCE_AND_GATES_READY.json"),
                                 "sha256": launcher.file_digest(run / "SOURCE_AND_GATES_READY.json")},
            "final_verification": {"path": str(final_path), "sha256": launcher.file_digest(final_path)}}
        verify_policy(run, policy)
        launcher.atomic(queue / POLICY_NAME, policy, immutable=True)
    return policy


def verify_policy(run, policy):
    """Reject changed source, proof, registered units or any unapproved policy fields."""
    required = {"schema_version", "kind", "campaign_run_dir", "units", "resources", "numerical_source",
                "source_admission", "final_verification"}
    if (not isinstance(policy, dict) or set(policy) != required or type(policy["schema_version"]) is not int
            or policy["schema_version"] != 1 or policy["kind"] != "uva_final_training_monitor_policy"
            or policy["campaign_run_dir"] != str(run) or not policy["units"]):
        raise ValueError("final monitor requires its exact explicit prepared policy")
    resources(policy["resources"])
    for key, name in (("source_admission", "SOURCE_AND_GATES_READY.json"), ("final_verification", "final_verification.json")):
        if policy[key]["path"] != str(run / name):
            raise ValueError("final monitor policy names a different campaign proof")
        launcher.verify_receipt(policy[key])
    if policy["units"] != unit_records(run):
        raise ValueError("registered final units differ from the explicit monitor policy")
    for receipt in policy["units"].values():
        launcher.verify_receipt(receipt)
        unit = launcher.read(receipt["path"])
        launcher.verify_unit(unit)
        if unit["numerical_source"] != policy["numerical_source"]:
            raise ValueError("a final unit differs from the policy's fixed numerical source")
        specification = launcher.read(unit["final_specification"]["path"])
        if specification["final_verification"] != policy["final_verification"]["path"]:
            raise ValueError("a final unit names another final-verification proof")


@contextmanager
def monitor_lock(queue):
    """Keep concurrent monitor ticks from making separate submission decisions."""
    with (queue / "monitor.lock").open("a+") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def decide(run, policy, controller_state, controller_error):
    """Rebuild all final gates on owned compute and make at most one generic-job submission."""
    verify_policy(run, policy)
    if controller_error:
        raise ValueError("comparison controller failed: " + str(controller_error))
    if not controller_state or controller_state.get("phase") != "ready_for_final":
        held = [launcher.read(path) for name in policy["units"]
                if (path := run / "final_training_queue/status" / (name + ".json")).exists()
                and launcher.read(path).get("state") == "held"]
        reason = "campaign phase: " + (controller_state or {}).get("phase", "controller not run")
        return {"state": "held" if held else "waiting", "reason": reason + ("; final unit requires diagnosis" if held else ""), "units": held}
    # The monitor may be a CPU allocation; jaguar03 admission remains inside the launched GPU worker.
    controller.final_validation_allocation(run)
    with controller.queue_lock(run):
        # Source-admission revisions use this same lock; recheck exact bytes after acquiring it.
        verify_policy(run, policy)
        ready = controller.ready_specification(run)
        if {"root": ready["source_root"], "commit": ready["source_commit"]} != policy["numerical_source"]:
            raise ValueError("current source admission differs from the prepared final monitor policy")
        if any(any((run / "queue" / state).glob("*.json")) for state in ("pending", "running", "failed")):
            raise ValueError("comparison work remains pending, running or failed")
        if not controller.final_requirements_met(launcher.read(run / "study_registry.json"), launcher.read(run / "final_verification.json"),
                                                  run_dir=run, source_commit=ready["source_commit"]):
            raise ValueError("all fifty mechanisms and final raw gates are not complete")
        snapshot = launcher.status(run)
        if {unit["unit_id"] for unit in snapshot["units"]} != set(policy["units"]):
            raise ValueError("final queue status differs from the explicitly registered units")
        if snapshot["counts"]["held"]:
            return {"state": "held", "reason": "a final unit requires diagnosis; no automatic application retry", **snapshot}
        if any(state in launcher.ACTIVE for state in snapshot["jobs"].values()):
            return {"state": "waiting", "reason": "an exact owned final job is active or pending", **snapshot}
        if snapshot["counts"]["completed"] == len(policy["units"]):
            return {"state": "finished", "reason": "all explicit UVA units have validated completion", **snapshot}
        count = snapshot["counts"]["ready"]
        shape = dict(policy["resources"])
        partial = shape.pop("allow_partial_group")
        if not count or count < shape["gpus"] and not partial:
            return {"state": "waiting", "reason": "not enough explicit ready units for the prepared job shape", **snapshot}
        shape["gpus"] = min(shape["gpus"], count)
        job = launcher.submit(run, **shape)
        return {"state": "submitted", "reason": "one generic UVA final job durably recorded", "submission": job, **snapshot}


def tick(run, *, controller_state=None, controller_error=None):
    """Remain disabled without a policy; publish explicit holds instead of retrying invalid work."""
    run = Path(run).resolve(strict=True)
    queue = run / "final_training_queue"
    path = queue / POLICY_NAME
    if not path.exists():
        return {"state": "disabled", "reason": "no explicit prepared final submission policy", "units": []}
    try:
        with monitor_lock(queue):
            try:
                policy = launcher.read(path)
                result = decide(run, policy, controller_state, controller_error)
            except (ArithmeticError, AttributeError, IndexError, KeyError, TypeError, ValueError, OSError, subprocess.SubprocessError) as error:
                result = {"state": "held", "reason": f"{type(error).__name__}: {error}", "units": []}
            result.update(time=datetime.now(timezone.utc).isoformat(), policy_sha256=launcher.file_digest(path))
            launcher.atomic(queue / "monitor_latest.json", result)
            return result
    except BlockingIOError:
        return {"state": "waiting", "reason": "another final monitor tick holds the submission decision lock", "units": []}


def main():
    """Prepare explicit metadata or perform one allocated-compute monitor decision."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    preparation = commands.add_parser("prepare-policy")
    preparation.add_argument("--run-dir", type=Path, required=True)
    preparation.add_argument("--input", type=Path, required=True)
    execution = commands.add_parser("tick")
    execution.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare-policy":
        result = prepare_policy(args.run_dir, args.input)
    else:
        result = tick(args.run_dir, controller_state=launcher.read(args.run_dir / "campaign_state.json"))
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
