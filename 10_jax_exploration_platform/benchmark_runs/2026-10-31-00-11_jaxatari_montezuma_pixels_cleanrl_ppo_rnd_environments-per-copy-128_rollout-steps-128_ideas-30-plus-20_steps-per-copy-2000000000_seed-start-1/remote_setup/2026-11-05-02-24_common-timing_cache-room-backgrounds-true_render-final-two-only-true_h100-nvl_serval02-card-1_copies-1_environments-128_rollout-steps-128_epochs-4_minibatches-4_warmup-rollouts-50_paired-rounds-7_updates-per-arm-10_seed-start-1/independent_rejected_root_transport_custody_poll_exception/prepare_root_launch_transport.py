"""Prepare the exact once-only root transport after the real controller source is published.

This program writes local metadata only. It never contacts serval02, starts a
controller, probes a process, opens a device or touches the standing GPU lock.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

P = Path(__file__).resolve(strict=True).parent
spec = importlib.util.spec_from_file_location("correctness_launch_metadata_controller", P / "direct_qualification_controller.py")
controller = importlib.util.module_from_spec(spec)
spec.loader.exec_module(controller)
TEMPLATE_SHA256 = "72e3140611f691dd0038b57f3b0d85704eec43a3a134c5e5ac85d672451dd08b"
OLD_PUBLISHED_CONTROLLER = "5ce06b66777137e9a3a5dc24e5396da61525cfea"


def main(argv=None):
    """Bind the existing reviewed root transport to one actual published correctness controller."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--publication-commit", required=True)
    args = parser.parse_args(argv)
    plan, command, unused = controller.checked_inputs()
    review = controller.check_publication(args.publication_commit)
    controller.check_unattempted(controller.UNIT)
    template = P / "root_controller_transport_barrier_template.py.txt"
    raw = controller.ordinary_bytes(template)
    controller.require(hashlib.sha256(raw).hexdigest() == TEMPLATE_SHA256,
        "the exact prior reviewed root transport template changed")
    old = raw.decode()
    controller.require(old.count(OLD_PUBLISHED_CONTROLLER) == 1
        and "str(P/'direct_qualification_controller.py')" in old,
        "the exact proven root transport publication pin or fixed controller command changed")
    source = old.replace(OLD_PUBLISHED_CONTROLLER, args.publication_commit)
    directory = controller.RUN / "slurm/direct_jobs" / P.name
    directory.mkdir(parents=True, exist_ok=True)
    controller.require(directory.resolve(strict=True) == directory, "ordinary exact transport-log directory required")
    argv = ["/usr/bin/env", "-i", "PATH=/usr/local/bin:/usr/bin:/bin", "LANG=C.UTF-8", "TZ=UTC",
        "CUDA_VISIBLE_DEVICES=" + controller.GPU_UUID, "PYTHONDONTWRITEBYTECODE=1", "PYTHONNOUSERSITE=1",
        "/usr/bin/python3", "-I", "-S", "-B", str(P / "direct_qualification_controller.py"),
        "--launch", "--publication-commit", args.publication_commit]
    record = {"schema_version": 1, "kind": "actual_published_common_timing_controller_once_transport_preparation",
        "publication": args.publication_commit, "controller_review": review,
        "controller_source_freeze": controller.receipt(P / "controller_source_freeze.json"),
        "packet_freeze": controller.receipt(P / "packet_metadata_freeze.json"),
        "plan": controller.receipt(controller.UNIT / "measurement_plan.json"),
        "host": "serval02", "command": argv, "producer_command": command,
        "stdout": str(directory / "controller.stdout"), "stderr": str(directory / "controller.stderr"),
        "automatic_retry_permitted": False, "transport_executed": False,
        "actual_controller_pid": None, "actual_numerical_child_pid": None,
        "controller_owns_gpu_standing_descriptor": False, "producer_owns_gpu_standing_descriptor": True,
        "scientific_credit": 0, "case_credit": 0, "idea_credit": 0}
    for name in ("actual_launch_command_preparation.json", "root_start_controller_once.py"):
        path = P / name
        controller.require(not path.exists() and not path.is_symlink()
            and not path.with_name(path.name + ".prepared").exists()
            and not path.with_name(path.name + ".prepared").is_symlink(),
            "preserve the previous launch metadata attempt")
    command_ref = controller.publish(P / "actual_launch_command_preparation.json", record)
    source_ref = controller.publish(P / "root_start_controller_once.py", source.encode())
    controller.publish(P / "root_transport_metadata_result.json", {"schema_version": 1,
        "kind": "actual_published_common_timing_controller_transport_metadata_result", "passed": True,
        "command_preparation": command_ref, "root_transport_source": source_ref,
        "controller_publication": args.publication_commit, "actual_remote_transport_performed": False,
        "host_contact": False, "gpu_operations": 0, "scientific_credit": 0})
    print(json.dumps({"passed": True, "command": command_ref, "source": source_ref,
        "host_contact": False, "launch_performed": False}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
