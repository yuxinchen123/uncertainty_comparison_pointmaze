"""Check the frozen raw-review source using only private metadata on owned CPU."""
import ast
import ctypes
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest

HERE = Path(__file__).resolve().parent
PACKET = HERE.parent
FROZEN_SHA = "f90e911f27fd42b37619e86f0c4df3351a0a160a207ca1126d101cb3e202eb8a"
NUMERICAL = {"jax", "jaxlib", "numpy", "torch", "flax", "optax", "cupy", "scipy", "jaxtari", "exploration_platform"}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def encoded(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def write(path, value):
    with path.open("xb") as stream:
        stream.write(encoded(value)); stream.flush(); os.fsync(stream.fileno())


def main():
    if os.environ.get("SLURM_JOB_ID") != "6783003" or not os.environ.get("SLURM_STEP_ID", "").isdecimal():
        raise ValueError("independent controls require the exact owned numeric CPU step")
    expected = "/system.slice/slurmstepd.scope/job_6783003/step_" + os.environ["SLURM_STEP_ID"] + "/user/task_0"
    if Path("/proc/self/cgroup").read_text().splitlines() != ["0::" + expected] or os.getuid() != 2618919 or os.geteuid() != 2618919:
        raise ValueError("independent controls require the exact own CPU identity")
    freeze_path = PACKET / "source_freeze.json"
    if digest(freeze_path) != FROZEN_SHA:
        raise ValueError("the independently reviewed freeze changed")
    freeze = json.loads(freeze_path.read_bytes())
    def current():
        return {path: digest(path) for path in freeze["files"]}
    before = current()
    if before != freeze["files"]:
        raise ValueError("a frozen source file changed before independent controls")
    old = PACKET / "source_before_closed_freeze_and_strict_exit_types"
    if digest(old / "source_freeze.json") != "c04896a73c62316805e615cc386472a1aa921d78d20e9c1fed09413ba87c5cd7":
        raise ValueError("the original failed source freeze was not preserved")
    old_freeze = json.loads((old / "source_freeze.json").read_bytes())
    old_expected = {str(old / Path(path).name): checksum for path, checksum in old_freeze["files"].items()}
    old_expected[str(old / "source_freeze.json")] = digest(old / "source_freeze.json")
    failed = PACKET / "independent_source_review"
    old_expected.update({str(path): digest(path) for path in failed.iterdir() if path.is_file()})
    def old_current():
        return {path: digest(path) for path in old_expected}
    if old_current() != old_expected:
        raise ValueError("the old failed source or review bytes changed")
    inputs = json.loads((PACKET / "prepared_inputs.json").read_bytes())
    run = Path(inputs["campaign_run_dir"])
    for name in ("submitted_jobids.txt", "submitted_jobids_uva.txt"):
        if (run / "slurm" / name).read_text().splitlines().count("6783003") != 1:
            raise ValueError("the exact owned CPU ledger is missing or duplicated")
    rows = [line.split("\t") for line in (run / "slurm/submissions.tsv").read_text().splitlines()
        if line.split("\t", 1)[0] == "6783003"]
    if len(rows) != 1 or rows[0][1:3] != ["monitor", "79a01e851ff92ddba7d0e12eaf8164cab2028878"]:
        raise ValueError("the exact owned CPU source row changed")
    cold_command = ["/usr/bin/python3", "-I", "-S", "-B", str(PACKET / "review_actual_common_correctness.py"), "--copies", "1"]
    def audit(event, values):
        if event == "import" and values[0].split(".")[0] in NUMERICAL:
            raise RuntimeError("numerical imports are forbidden in independent source controls")
        if event.startswith("socket.") and event != "socket.gethostname" or event in {"ctypes.dlopen", "ctypes.dlsym"}:
            raise RuntimeError("network and library actions are forbidden in independent source controls")
        if event == "open" and str(values[0]).startswith(("/dev/nvidia", "/dev/dri", "/tmp/sl5nw_serval02_card1.lock")):
            raise RuntimeError("device and standing-lock actions are forbidden")
        if event == "open" and str(values[0]).endswith("/result.json") and "private" not in str(values[0]):
            raise RuntimeError("no actual correctness result may be opened")
        if event == "subprocess.Popen" and (values[1] != cold_command or values[2] is not None or values[3] is not None):
            raise RuntimeError("only the exact frozen source-only cold child is allowed")
    sys.addaudithook(audit)
    specification = importlib.util.spec_from_file_location("independent_raw_review_author_controls", PACKET / "test_review_actual_common_correctness.py")
    author = importlib.util.module_from_spec(specification); specification.loader.exec_module(author)
    review = author.review

    class ExtraRefusals(unittest.TestCase):
        def fixture(self):
            case = author.TerminalTests("test_exact_two_exits_and_reviewed_source_pass_without_scientific_work")
            case.setUp(); self.addCleanup(case.doCleanups)
            return case

        def test_float_expected_child_pid_must_refuse(self):
            case = self.fixture()
            case.observation["observations"][1]["expected_pid"] = 101.0; case.save()
            with self.assertRaises(ValueError): case.check()

        def test_float_expected_controller_pid_must_refuse(self):
            case = self.fixture()
            case.observation["observations"][0]["expected_pid"] = 303.0; case.save()
            with self.assertRaises(ValueError): case.check()

        def test_approved_freeze_with_changed_actual_source_must_refuse(self):
            case = self.fixture()
            source = case.packet / "review_actual_common_correctness.py"
            source.write_text("private metadata fixture changed unreviewed source\n")
            with self.assertRaises(ValueError): case.check()


        def test_float_actual_process_uid_must_refuse(self):
            case = self.fixture()
            case.metadata["after"]["identity"]["uid"] = 2618919.0; case.save()
            with self.assertRaises(ValueError): case.check()

        def test_float_actual_controller_pid_must_refuse(self):
            case = self.fixture()
            case.spawn["controller"]["pid"] = 303.0; case.save()
            with self.assertRaises(ValueError): case.check()

        def test_noncanonical_actual_controller_birth_must_refuse(self):
            case = self.fixture()
            case.spawn["controller"]["process_start_time"] = "0404"; case.save()
            with self.assertRaises(ValueError): case.check()

        def test_unknown_source_freeze_kind_must_refuse(self):
            case = self.fixture()
            value = review.read(case.packet / "source_freeze.json"); value["kind"] = "different_source"
            case.put(case.packet / "source_freeze.json", value)
            case.approval["source_freeze"] = review.reference(case.packet / "source_freeze.json"); case.save()
            with self.assertRaises(ValueError): case.check()

        def test_extra_source_freeze_path_must_refuse(self):
            case = self.fixture()
            value = review.read(case.packet / "source_freeze.json")
            extra = case.packet / "unreviewed.py"; extra.write_text("private never-executed fixture")
            value["files"][str(extra)] = digest(extra); case.put(case.packet / "source_freeze.json", value)
            case.approval["source_freeze"] = review.reference(case.packet / "source_freeze.json"); case.save()
            with self.assertRaises(ValueError): case.check()

        def test_symlinked_source_freeze_member_must_refuse(self):
            case = self.fixture()
            member = case.packet / "prepared_inputs.json"
            moved = case.packet / "private_other_source"; member.rename(moved); member.symlink_to(moved)
            with self.assertRaises(ValueError): case.check()

        def test_no_completed_flag_is_required_for_actual_correctness(self):
            case = author.RawFieldContractTests("test_exact_raw_field_contract_follows_complete_consumer")
            case.setUp(); self.addCleanup(case.doCleanups)
            self.assertNotIn("completed", case.result)
            unused, unused_identity, raw = review.actual_raw(case.directory, case.plan, {}, case.guard, case.support)
            self.assertTrue(raw["passed"])

        def test_raw_validation_must_finish_before_review_publication(self):
            tree = ast.parse((PACKET / "review_actual_common_correctness.py").read_text())
            main_function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main")
            calls = sorted([(node.lineno, ast.unparse(node)) for node in ast.walk(main_function) if isinstance(node, ast.Call)])
            raw_line = next(line for line, expression in calls if expression.startswith("actual_raw("))
            publication = next(line for line, expression in calls if "'independent_actual_review.json'" in expression and expression.startswith("publish("))
            raw_publication = next(line for line, expression in calls if "'raw_validation.json'" in expression and expression.startswith("publish("))
            self.assertLess(raw_line, raw_publication); self.assertLess(raw_publication, publication)

        def test_guard_refuses_dri_load_symbol_and_non_git_subprocess(self):
            unused, plan, qualified, guard = review.prepared(1)
            events = [("open", ("/dev/dri/renderD128", "r", 0)), ("ctypes.dlopen", (None,)),
                ("ctypes.dlsym", (None, "private")), ("subprocess.Popen", ("/usr/bin/python3", ["/usr/bin/python3"], None, None))]
            for event, arguments in events:
                with self.subTest(event=event):
                    with self.assertRaises((ValueError, RuntimeError)):
                        guard.metadata_event_guard(event, arguments, set())

    suite = unittest.TestSuite([unittest.defaultTestLoader.loadTestsFromModule(author),
        unittest.defaultTestLoader.loadTestsFromTestCase(ExtraRefusals)])
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    cold = subprocess.run(cold_command, capture_output=True, check=False, timeout=180)
    cold_path = HERE / ("cold_source_only_" + os.environ["SLURM_STEP_ID"] + ".json")
    write(cold_path, {"command": cold_command, "returncode": cold.returncode,
        "stdout": cold.stdout.decode(), "stderr": cold.stderr.decode(),
        "actual_correctness_result_opened": False, "actual_terminal_audit": False})
    cold_passed = cold.returncode == 0 and cold.stderr == b"" and json.loads(cold.stdout).get("actual_correctness_result_opened") is False
    after = current()
    if after != before or old_current() != old_expected:
        raise ValueError("frozen source or earlier failure evidence changed during independent review")
    record = {"schema_version": 1, "kind": "independent_h100_common_correctness_raw_review_source_review",
        "approved": result.wasSuccessful() and cold_passed, "passed": result.wasSuccessful() and cold_passed,
        "independent_of_author": True, "source_freeze": {"path": str(freeze_path), "sha256": FROZEN_SHA},
        "tests": result.testsRun, "failures": [{"test": str(test), "traceback": trace} for test, trace in result.failures],
        "errors": [{"test": str(test), "traceback": trace} for test, trace in result.errors],
        "cold_default_source_only_passed": cold_passed, "cold_record": {"path": str(cold_path), "sha256": digest(cold_path)},
        "at_utc": datetime.now(timezone.utc).isoformat(), "job_id": "6783003", "step_id": os.environ["SLURM_STEP_ID"],
        "source_files_before": before, "source_files_after": after, "actual_correctness_result_opened": False,
        "preserved_old_source_and_failed_review": old_expected,
        "actual_terminal_audit_or_review": False, "host_or_gpu_actions": False, "scientific_credit": 0,
        "source_read_completed": True}
    path = HERE / ("review_" + os.environ["SLURM_STEP_ID"] + ".json")
    write(path, record)
    print(json.dumps({"record": str(path), "approved": record["approved"], "tests": result.testsRun,
        "failures": len(result.failures), "errors": len(result.errors), "cold_passed": cold_passed}))
    return 0 if record["approved"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

