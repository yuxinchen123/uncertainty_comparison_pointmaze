"""Record independent metadata-only checks without replaying scientific evidence."""
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "10_jax_exploration_platform/scripts/atari_pixel_campaign"
sys.path.insert(0, str(SCRIPTS))
import direct_common_confirmation as guard
import direct_historical_consumer as historical


def main():
    """Exercise audit decisions only; never execute a supplied subprocess command."""
    if os.environ.get("SLURM_JOB_ID") != "6783003" or not os.environ.get("SLURM_STEP_ID", "").isdigit():
        raise RuntimeError("independent checks require the exact owned CPU allocation")
    checks = []

    def decision(name, function, arguments):
        """Record the real decision and exception for one synthetic audit event."""
        try:
            function(*arguments)
        except (ValueError, RuntimeError) as error:
            checks.append({"name": name, "allowed": False, "exception": str(error)})
        else:
            checks.append({"name": name, "allowed": True})

    guard._ACTIVE_HISTORICAL_SOURCE_SHA256 = hashlib.sha256((SCRIPTS / historical.NAME).read_bytes()).hexdigest()
    expected = ["/usr/bin/python3", "-I", "-S", "-B", str(SCRIPTS / historical.NAME), "--read-only-stdin"]
    decision("exact declared historical command", guard.metadata_event_guard,
        ("subprocess.Popen", ("/usr/bin/python3", expected, None, None), set()))
    decision("historical argv with substituted executable", guard.metadata_event_guard,
        ("subprocess.Popen", ("/synthetic/unapproved-executable", expected, None, None), set()))
    decision("unlisted Python command", guard.metadata_event_guard,
        ("subprocess.Popen", ("/usr/bin/python3", ["/usr/bin/python3", "-c", "pass"], None, None), set()))
    decision("Git argv with substituted executable", historical.readonly_audit,
        (guard, {"/synthetic/source"}, "subprocess.Popen", ("/synthetic/unapproved-executable",
            ["git", "-C", "/synthetic/source", "show", "a" * 40 + ":source.py"], None, None)))
    decision("Git diff with external executable option", historical.readonly_audit,
        (guard, {"/synthetic/source"}, "subprocess.Popen", ("git",
            ["git", "-C", "/synthetic/source", "diff", "--ext-diff"], None, None)))
    decision("Git foreign root", historical.readonly_audit,
        (guard, {"/synthetic/source"}, "subprocess.Popen", ("git",
            ["git", "-C", "/synthetic/foreign", "show", "HEAD"], None, None)))
    for event, values in (("socket.connect", (None, None)), ("ctypes.dlopen", ("libcuda.so",)),
            ("open", ("/dev/nvidia0", "rb", 0)), ("open", ("/tmp/sl5nw_serval02_card1.lock", "rb", 0)),
            ("open", ("/synthetic/output", "wb", os.O_WRONLY)), ("os.fork", ())):
        decision("refused event " + event + " " + str(values), historical.readonly_audit,
            (guard, set(), event, values))
    print(json.dumps({"kind": "independent_synthetic_audit_decisions", "actual_step": os.environ["SLURM_STEP_ID"],
        "process": historical.process(os.getpid()), "finished_at_seconds": time.time(), "checks": checks,
        "actual_subprocess_commands_executed": 0, "saved_scientific_evidence_recomputed": False,
        "numerical_imports": [], "scientific_credit": 0, "admission_granted": False}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
