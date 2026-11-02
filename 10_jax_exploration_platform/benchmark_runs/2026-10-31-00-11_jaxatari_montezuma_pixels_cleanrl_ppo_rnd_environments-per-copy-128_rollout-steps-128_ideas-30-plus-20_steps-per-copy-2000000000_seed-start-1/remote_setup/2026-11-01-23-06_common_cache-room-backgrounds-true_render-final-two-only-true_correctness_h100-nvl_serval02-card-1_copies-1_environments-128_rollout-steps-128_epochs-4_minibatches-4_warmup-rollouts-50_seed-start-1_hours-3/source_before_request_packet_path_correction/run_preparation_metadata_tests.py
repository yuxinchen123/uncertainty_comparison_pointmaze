"""Run only the frozen source controls inside the exact owned CPU allocation."""
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import unittest

P = Path(__file__).resolve().parent


def require(value, message):
    """Reject an unsupported source-test context before any scheduler query."""
    if not value:
        raise ValueError(message)


def main():
    """Verify exact ownership and source, run metadata controls, then save actual results."""
    require(os.environ.get("SLURM_JOB_ID") == "6783003" and os.environ.get("SLURM_STEP_ID", "").isdecimal()
        and os.getuid() == os.geteuid() == 2618919, "only the exact owned CPU allocation may run")
    request = json.loads((P / "preparation_request.json").read_bytes())
    run = Path(request["campaign_run_dir"])
    for name in ("submitted_jobids.txt", "submitted_jobids_uva.txt"):
        require((run / "slurm" / name).read_text().splitlines().count("6783003") == 1,
            "the exact owned CPU identifier is missing or duplicated")
    rows = [line.split("\t") for line in (run / "slurm/submissions.tsv").read_text().splitlines()
        if line.split("\t", 1)[0] == "6783003"]
    require(len(rows) == 1 and rows[0][1:3] == ["monitor", "79a01e851ff92ddba7d0e12eaf8164cab2028878"],
        "the actual owned source/row differs before query")
    command = ["scontrol", "show", "job", "-o", "6783003"]
    returned = subprocess.run(command, capture_output=True, text=True)
    require(returned.returncode == 0 and returned.stderr == ""
        and "JobId=6783003 " in returned.stdout and "JobState=RUNNING " in returned.stdout
        and "UserId=sl5nw(2618919) " in returned.stdout and "Command=" + rows[0][-1] in returned.stdout,
        "the exact owned CPU allocation is not currently running")
    freeze_path = P / "preparation_source_freeze.json"
    freeze = json.loads(freeze_path.read_bytes())
    def current():
        return {path: hashlib.sha256(Path(path).read_bytes()).hexdigest() for path in freeze["files"]}
    require(current() == freeze["files"], "frozen source bytes changed before tests")
    spec = importlib.util.spec_from_file_location("common_correctness_metadata_tests", P / "test_prepare_common_correctness.py")
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    tests = unittest.defaultTestLoader.loadTestsFromModule(module)
    result = unittest.TextTestRunner(verbosity=2).run(tests)
    require(current() == freeze["files"], "frozen source bytes changed during tests")
    record = {"schema_version": 1, "kind": "direct_common_correctness_metadata_author_tests",
        "passed": result.wasSuccessful(), "tests": result.testsRun, "failures": len(result.failures),
        "errors": len(result.errors), "job_id": "6783003", "step_id": os.environ["SLURM_STEP_ID"],
        "source_freeze": {"path": str(freeze_path), "sha256": hashlib.sha256(freeze_path.read_bytes()).hexdigest()},
        "at_utc": datetime.now(timezone.utc).isoformat(), "source_files": freeze["files"], "owned_query": {
            "command": command, "returncode": returned.returncode, "stdout": returned.stdout, "stderr": returned.stderr},
        "history_staged": False, "actual_qualification_result_opened": False, "numerical_work": False, "gpu_work": False,
        "host_contact": False, "scientific_credit": 0, "case_credit": 0, "idea_credit": 0}
    path = P / ("author_metadata_tests_" + os.environ["SLURM_STEP_ID"].replace(".", "_") + ".json")
    with path.open("x") as stream:
        json.dump(record, stream, indent=2, sort_keys=True); stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
    print(json.dumps({"record": str(path), "tests": result.testsRun, "passed": result.wasSuccessful()}))
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
