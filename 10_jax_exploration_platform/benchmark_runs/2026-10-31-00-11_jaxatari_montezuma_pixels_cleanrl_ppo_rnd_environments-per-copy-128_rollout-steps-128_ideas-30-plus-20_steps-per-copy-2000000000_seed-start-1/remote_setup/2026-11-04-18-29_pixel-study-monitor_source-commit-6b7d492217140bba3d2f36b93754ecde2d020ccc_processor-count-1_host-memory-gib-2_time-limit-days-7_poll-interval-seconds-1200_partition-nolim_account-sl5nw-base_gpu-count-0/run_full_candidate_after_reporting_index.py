"""Run one actual full candidate monitor report on the existing owned CPU allocation."""
from contextlib import redirect_stdout, redirect_stderr
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import traceback

PACKET = Path(__file__).resolve().parent
RUN = PACKET.parent.parent
COMMIT = "6b7d492217140bba3d2f36b93754ecde2d020ccc"


def receipt(path):
    """Bind each captured candidate artifact to its actual bytes."""
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def publish(path, value):
    """Fsync one new candidate artifact and preserve any previous attempt."""
    raw = value if isinstance(value, bytes) else (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    with path.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())


def main():
    """Verify the complete source, serialize large evidence reads and run one unmodified full monitor tick."""
    if os.environ.get("SLURM_JOB_ID") != "6783003" or not os.environ.get("SLURM_STEP_ID", "").isdigit():
        raise ValueError("the full candidate requires an actual numeric step of owned CPU allocation 6783003")
    if os.getuid() != os.geteuid() or os.getuid() != 2618919:
        raise ValueError("the actual campaign owner is required")
    for name in ("candidate_after_reporting_index_full_report_receipt.json", "candidate_after_reporting_index_full_stdout.raw", "candidate_after_reporting_index_full_stderr.raw",
                 "candidate_after_reporting_index_full_report.md", "candidate_after_reporting_index_full_status.json"):
        if os.path.lexists(PACKET / name):
            raise ValueError("preserve the earlier full candidate attempt")
    proposed_index = json.loads((PACKET / "proposed_post_initial_reporting_evidence.json").read_text())
    if json.loads((RUN / "next_actions.json").read_text()).get("post_initial_reporting_evidence") != proposed_index:
        raise ValueError("root must adopt the exact checked display index before this second actual full candidate")
    actions_before = receipt(RUN / "next_actions.json")
    provenance_before = receipt(RUN / "campaign_provenance.json")
    source_record = json.loads((PACKET / "source_snapshot.json").read_text())
    if source_record["source_commit"] != COMMIT:
        raise ValueError("the candidate source commit changed")
    source = Path(source_record["source_root"])
    script = source / "10_jax_exploration_platform/scripts/atari_pixel_campaign/monitor.py"
    verified = subprocess.run(["/usr/bin/python3", "-I", "-S", "-B", str(PACKET / "verify_source.py")],
                              capture_output=True, text=True, check=True)
    subprocess.run(["bash", "-n", str(PACKET / "monitor.sbatch")], check=True)
    commands = []
    numerical = {"jax", "jaxlib", "numpy", "torch", "cupy", "pynvml", "flax", "optax", "jaxtari"}

    def owned_identifiers():
        """Read only the current campaign UVA ownership ledger before any scheduler query."""
        return set((RUN / "slurm/submitted_jobids_uva.txt").read_text().split())

    def audit(event, values):
        """Refuse numerical, GPU, network and job-changing operations during the actual candidate tick."""
        if event == "import" and str(values[0]).split(".")[0] in numerical:
            raise RuntimeError("numerical imports are forbidden during the CPU monitor candidate")
        if event == "ctypes.dlopen" and values[0] is not None:
            raise RuntimeError("loading a GPU or other native library is forbidden during the candidate")
        if event in {"socket.__new__", "os.system", "os.posix_spawn"}:
            raise RuntimeError("network or shell operations are forbidden during the candidate")
        if event == "open" and isinstance(values[0], (str, bytes)) and os.fsdecode(values[0]).startswith(("/dev/nvidia", "/dev/dri")):
            raise RuntimeError("GPU device access is forbidden during the candidate")
        if event == "subprocess.Popen":
            command = values[1]
            if not isinstance(command, (list, tuple)) or not command:
                raise RuntimeError("the monitor candidate requires an explicit bounded read-only command")
            command = [str(value) for value in command]
            executable = Path(command[0]).name
            if executable == "squeue" and "-j" in command:
                queried = set(command[command.index("-j") + 1].split(","))
                if not queried or not queried <= owned_identifiers():
                    raise RuntimeError("a monitor queue query escaped the campaign UVA ledger")
            elif executable == "sacct":
                selector = next((value.split("=", 1)[1] for value in command if value.startswith("--jobs=")), None)
                if selector is None and "--jobs" in command:
                    selector = command[command.index("--jobs") + 1]
                queried = set((selector or "").split(","))
                if not queried or not queried <= owned_identifiers():
                    raise RuntimeError("a monitor accounting query escaped the campaign UVA ledger")
            elif executable == "git":
                if any(value in {"add", "commit", "push", "checkout", "switch", "reset", "clean", "merge", "rebase"} for value in command[1:]):
                    raise RuntimeError("a monitor candidate Git command would change source or publication")
            else:
                raise RuntimeError("the candidate attempted an unsupported native command: " + executable)
            commands.append(command)

    started = datetime.now(timezone.utc).isoformat()
    output, errors = io.StringIO(), io.StringIO()
    failure = None
    completed = None
    sys.path.insert(0, str(script.parent))
    sys.addaudithook(audit)
    with (RUN / "queue/cpu_monitor_large_artifact_read.lock").open("a+") as guard:
        fcntl.flock(guard.fileno(), fcntl.LOCK_EX)
        try:
            with redirect_stdout(output), redirect_stderr(errors):
                specification = importlib.util.spec_from_file_location("exact_candidate_monitor", script)
                module = importlib.util.module_from_spec(specification)
                specification.loader.exec_module(module)
                completed = module.tick(RUN)
        except Exception:
            failure = traceback.format_exc()
            errors.write(failure)
        report_path = RUN / "20_mins_monitoring/latest_report.md"
        status_path = RUN / "20_mins_monitoring/latest_status.json"
        publish(PACKET / "candidate_after_reporting_index_full_stdout.raw", output.getvalue().encode())
        publish(PACKET / "candidate_after_reporting_index_full_stderr.raw", errors.getvalue().encode())
        if failure is None:
            publish(PACKET / "candidate_after_reporting_index_full_report.md", report_path.read_bytes())
            publish(PACKET / "candidate_after_reporting_index_full_status.json", status_path.read_bytes())
    status = json.loads((PACKET / "candidate_after_reporting_index_full_status.json").read_text()) if failure is None else None
    text = (PACKET / "candidate_after_reporting_index_full_report.md").read_text() if failure is None else ""
    sections = [name for name in ("TRAINING", "WAITING", "HELD", "FINISHED", "PLANNED") if name in text.splitlines()]
    state = status["campaign"] if status is not None else {}
    requirements = state.get("post_initial_reporting_requirements", state.get("reporting_requirements", []))
    imported = sorted(name for name in sys.modules if name.split(".")[0] in numerical)
    passed = failure is None and len(sections) == 5 and not imported
    value = {"schema_version": 1, "kind": "actual_full_cpu_monitor_candidate", "passed": passed,
        "source_commit": COMMIT, "source_manifest": receipt(PACKET / "source_snapshot.json"),
        "plan": receipt(PACKET / "monitor_replacement_plan.json"), "script": receipt(script),
        "job_id": os.environ["SLURM_JOB_ID"], "step_id": os.environ["SLURM_STEP_ID"],
        "host": socket.gethostname(), "uid": os.getuid(), "pid": os.getpid(),
        "cgroup": Path("/proc/self/cgroup").read_text(), "started_at_utc": started,
        "finished_at_utc": datetime.now(timezone.utc).isoformat(), "full_tick_completed": failure is None,
        "study_complete_return": completed, "source_verifier_stdout": verified.stdout,
        "report_sections": sections, "queue_counts": state.get("queue_counts"),
        "phase": state.get("phase"), "next_action": state.get("next_action"),
        "remaining_reporting_requirements": requirements,
        "stdout": receipt(PACKET / "candidate_after_reporting_index_full_stdout.raw"), "stderr": receipt(PACKET / "candidate_after_reporting_index_full_stderr.raw"),
        "report": receipt(PACKET / "candidate_after_reporting_index_full_report.md") if failure is None else None,
        "status": receipt(PACKET / "candidate_after_reporting_index_full_status.json") if failure is None else None,
        "read_only_native_commands": commands, "numerical_modules_imported": imported,
        "actual_new_monitor_runtime_verified": False, "actual_new_first_report_verified": False,
        "old_monitor_kept_running": True, "new_jobs_submitted": 0, "scientific_credit": 0,
        "next_actions_before": actions_before, "next_actions_after": receipt(RUN / "next_actions.json"),
        "provenance_before": provenance_before, "provenance_after": receipt(RUN / "campaign_provenance.json"),
        "candidate_does_not_write_next_actions_or_provenance": True, "failure": failure}
    publish(PACKET / "candidate_after_reporting_index_full_report_receipt.json", value)
    print(json.dumps({"passed": passed, "step_id": value["step_id"], "receipt": receipt(PACKET / "candidate_after_reporting_index_full_report_receipt.json"),
        "queue_counts": value["queue_counts"], "phase": value["phase"], "report_sections": sections}, sort_keys=True))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())

