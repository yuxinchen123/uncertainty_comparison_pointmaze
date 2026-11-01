"""Exercise fixed metadata transport boundaries without SSH or numerical libraries."""

import copy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("tested_native_staging", ROOT / "stage_aces_full_site_inputs.py")
driver = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(driver)
PLAN = ROOT / "aces_full_site_native_staging_plan.json"


class StagingTests(unittest.TestCase):
    """Keep network, scheduler and numerical execution absent from every fixture."""

    def setUp(self):
        """Read the reviewed plan and create a private directory for synthetic byte streams."""
        self.plan = json.loads(PLAN.read_bytes())
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name).resolve()

    def expected(self, contents):
        """Bind one small synthetic transfer's exact SHA and size."""
        return {"sha256": hashlib.sha256(contents).hexdigest(), "bytes": len(contents)}

    def test_reviewed_plan_layout_and_only_prepare_command(self):
        """The original reviewed plan pins all thirteen targets and actual metadata preparation."""
        checked = driver.checked_plan(PLAN.read_bytes())
        self.assertEqual(checked, self.plan)
        self.assertEqual(len(driver.checkout_commands(checked)), 10)
        self.assertTrue(all(command[0] == "git" for command in driver.checkout_commands(checked)))
        self.assertEqual(checked["native_prepare_command"][3], "prepare")
        self.assertNotIn("run", checked["native_prepare_command"])

    def test_foreign_plan_and_changed_local_bytes_fail_before_any_network(self):
        """Plan/hash validation completes before a network process can be created."""
        altered = self.directory / "changed_plan.json"
        altered.write_bytes(PLAN.read_bytes() + b" ")
        with mock.patch.object(driver.subprocess, "Popen") as network:
            with self.assertRaisesRegex(ValueError, "reviewed SHA256"):
                driver.validate_local(altered)
            network.assert_not_called()
        small = self.directory / "input.json"
        small.write_bytes(b"changed")
        plan = copy.deepcopy(self.plan)
        plan["exact_copy_allowlist"][0] = {**plan["exact_copy_allowlist"][0], "local_path": str(small),
            "bytes": 7, "sha256": "0" * 64}
        contents = json.dumps(plan).encode()
        altered.write_bytes(contents)
        with mock.patch.object(driver, "PLAN_SHA256", driver.digest(contents)), mock.patch.object(driver.subprocess, "Popen") as network:
            with self.assertRaisesRegex(ValueError, "local input changed"):
                driver.validate_local(altered)
            network.assert_not_called()

    def test_existing_master_options_do_not_check_or_authenticate(self):
        """A missing existing master has no proxy or authentication fallback and no separate check."""
        command = driver.ssh_command("# synthetic reviewed source\n")
        for option in ("ControlMaster=no", "ProxyCommand=false", "BatchMode=yes", "NumberOfPasswordPrompts=0", "ConnectionAttempts=1"):
            self.assertIn(option, command)
        self.assertNotIn("-O", command)
        self.assertEqual(command[-2], "aces")
        self.assertIn("--remote-receive", command[-1])
        self.assertIn(driver.PYTHON, command[-1])

    def test_stream_publication_preserves_identical_or_conflicting_existing_bytes(self):
        """Complete identical input is retained while a different existing file remains unchanged."""
        path = self.directory / "input.json"
        contents = b"synthetic exact input\n"
        first = driver.receive_file(io.BytesIO(contents), path, self.expected(contents))
        self.assertEqual(first["sha256"], self.expected(contents)["sha256"])
        driver.receive_file(io.BytesIO(contents), path, self.expected(contents))
        with self.assertRaisesRegex(ValueError, "different bytes"):
            driver.receive_file(io.BytesIO(b"other"), path, self.expected(b"other"))
        self.assertEqual(path.read_bytes(), contents)

    def test_bad_or_truncated_stream_never_publishes_a_final_file(self):
        """A wrong SHA or missing tail leaves no artifact at the reserved final path."""
        for stream, expected in ((b"different", self.expected(b"expected!")), (b"short", self.expected(b"longer input"))):
            target = self.directory / "new_input.json"
            with self.assertRaises(ValueError):
                driver.receive_file(io.BytesIO(stream), target, expected)
            self.assertFalse(target.exists())
            self.assertEqual(list(self.directory.glob(".received-*")), [])

    def test_sender_is_bounded_by_exact_length_and_incremental_hash(self):
        """A sender transmits the full expected input and rejects a changed size or byte value."""
        path = self.directory / "input.bin"
        path.write_bytes(b"synthetic exact bytes")
        output = io.BytesIO()
        driver.send_file(output, path, self.expected(path.read_bytes()))
        self.assertEqual(output.getvalue(), path.read_bytes())
        for expected in (self.expected(b"short"), {"bytes": path.stat().st_size, "sha256": "0" * 64}, self.expected(b"a longer synthetic byte stream")):
            with self.subTest(expected=expected), self.assertRaisesRegex(ValueError, "streamed file"):
                driver.send_file(io.BytesIO(), path, expected)

    def test_race_during_new_file_publication_preserves_conflicting_bytes(self):
        """An existing path created after validation cannot be overwritten by publication."""
        path = self.directory / "raced.json"
        original = driver.os.link
        def conflicting_link(source, target):
            """Insert competing bytes at the exact publication boundary."""
            Path(target).write_bytes(b"conflicting new bytes")
            return original(source, target)
        with mock.patch.object(driver.os, "link", side_effect=conflicting_link):
            with self.assertRaisesRegex(ValueError, "conflicting bytes appeared"):
                driver.receive_file(io.BytesIO(b"reviewed bytes"), path, self.expected(b"reviewed bytes"))
        self.assertEqual(path.read_bytes(), b"conflicting new bytes")

    def test_linked_file_or_parent_is_rejected_without_external_writes(self):
        """Neither an artifact nor its parent may redirect copied bytes through a symlink."""
        external = self.directory / "external"
        external.mkdir()
        link = self.directory / "linked"
        link.symlink_to(external, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "ordinary absolute"):
            driver.receive_file(io.BytesIO(b"payload"), link / "new.json", self.expected(b"payload"))
        self.assertFalse((external / "new.json").exists())
        target = external / "old.json"
        target.write_bytes(b"payload")
        file_link = self.directory / "linked.json"
        file_link.symlink_to(target)
        with self.assertRaisesRegex(ValueError, "ordinary absolute"):
            driver.receive_file(io.BytesIO(b"payload"), file_link, self.expected(b"payload"))

    def test_fresh_checkouts_and_no_other_program_is_allowed(self):
        """Only the fixed Git construction and metadata prepare action can reach subprocess.run."""
        with mock.patch.object(driver.subprocess, "run") as process:
            for command in (["sbatch", "x"], [driver.PYTHON, "-I", "/wrong/helper.py", "prepare"],
                    [driver.PYTHON, "-I", str(driver.BASE / "git/checkouts/study_helpers_8e43cb1dc963" / driver.PREFIX / "aces_slurm_site_validation.py"), "run"]):
                with self.subTest(command=command), self.assertRaisesRegex(ValueError, "different program"):
                    driver.execute(command, [])
            process.assert_not_called()
        commands = driver.checkout_commands(self.plan)
        self.assertIn(["git", "clone", "--bare", self.plan["native_cleanrl_checkout_required"]["bundle"],
            self.plan["remote_inputs_directory"] + "/cleanrl_repository.git"], commands)
        self.assertFalse(any("fetch" in command for command in commands))

    def response(self, change=None):
        """Build explicitly synthetic exact returned metadata and its corresponding raw files."""
        prepared = driver.prepared_files(self.plan)
        stage = Path(self.plan["remote_inputs_directory"])
        prepared += [("staging/" + name, stage / name) for name in ("old_roots_before.json", "old_roots_after.json")]
        records, values = [], []
        for relative, path in prepared:
            contents = ("synthetic saved metadata for " + relative + "\n").encode()
            records.append({"relative_path": relative, "path": str(path), **self.expected(contents)})
            values.append(contents)
        receipt = {"schema_version": 1, "kind": "aces_actual_native_metadata_preparation", "passed": True,
            "plan_sha256": driver.PLAN_SHA256, "driver_sha256": "a" * 64, "uid": 56626, "python": driver.PYTHON,
            "old_roots_before": {"synthetic": "unchanged"}, "old_roots_after": {"synthetic": "unchanged"},
            "numerical_imports": [], "jobs_submitted": 0, "workers_permitted": False, "case_credit": 0, "idea_credit": 0,
            "admission_credit": 0, "prepared_files": list(records)}
        contents = (json.dumps(receipt) + "\n").encode()
        records.append({"relative_path": "staging/native_preparation_receipt.json", "path": str(stage / "native_preparation_receipt.json"), **self.expected(contents)})
        values.append(contents)
        header = {"receipt": receipt, "files": records}
        if change:
            change(header)
        path = self.directory / "response.raw"
        path.write_bytes((json.dumps(header) + "\n").encode() + b"".join(values))
        return path, receipt

    def test_complete_metadata_response_binds_all_raw_files_and_zero_credit(self):
        """All exact returned records are saved without a native qualification or job claim."""
        path, receipt = self.response()
        self.assertEqual(driver.accept_response(path, self.directory, self.plan, "a" * 64), receipt)
        self.assertEqual(len(list((self.directory / "received").rglob("*.*"))), 18)
        self.assertNotIn("jax", sys.modules)
        self.assertNotIn("numpy", sys.modules)

    def test_response_rejects_path_escape_foreign_plan_and_numeric_credit_substitute(self):
        """A successful-looking header cannot redirect files or substitute foreign identity or credit."""
        for name, change in (("path", lambda h: h["files"][0].update(path="/outside")),
                ("plan", lambda h: h["receipt"].update(plan_sha256="f" * 64)),
                ("credit", lambda h: h["receipt"].update(idea_credit=False))):
            with self.subTest(name=name):
                path, _ = self.response(change)
                with self.assertRaises(ValueError):
                    driver.accept_response(path, self.directory, self.plan, "a" * 64)

    def test_complete_receiver_saves_real_bytes_before_returning_its_receipt(self):
        """The production receiver returns its complete immutable prepared-file stream without self-reference."""
        base = self.directory / "remote"
        (base / "git/repository.git").mkdir(parents=True)
        root = base / "remote_setup/inputs"
        plan = copy.deepcopy(self.plan)
        plan.update(remote_inputs_directory=str(root), remote_packet_directory=str(base / "packet"), remote_data_unit=str(base / "data"))
        plan["native_numerical_checkout_required"]["path"] = str(root / "numerical_source")
        plan["native_cleanrl_checkout_required"].update(path=str(root / "cleanrl_source"), bundle=str(root / "cleanrl.bundle"))
        rows, chunks = [], []
        for index in range(13):
            contents = ("synthetic received input " + str(index)).encode()
            rows.append({"local_path": "/synthetic/local/" + str(index), "remote_path": str(root / (str(index) + ".json")), **self.expected(contents)})
            chunks.append(contents)
        plan["exact_copy_allowlist"] = rows
        source = b"# synthetic receiver source"
        payload = (json.dumps({"plan": "{}", "driver_sha256": driver.digest(source)}) + "\n").encode() + b"".join(chunks)
        input_stream, output_stream = io.BytesIO(payload), io.BytesIO()
        helper = mock.Mock()
        helper.prepared.return_value = (Path(plan["remote_packet_directory"]), Path(plan["remote_data_unit"]), {"native": "synthetic metadata"})
        def execute(command, commands, **keywords):
            """Create only the synthetic files that native metadata preparation would publish."""
            commands.append({"argv": command, "returncode": 0})
            if command == plan["native_prepare_command"]:
                for relative, path in driver.prepared_files(plan):
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(("synthetic prepared bytes " + relative).encode())
            return driver.CLEANRL_COMMIT + " HEAD" if command[1:3] == ["bundle", "list-heads"] else ""
        original_read = Path.read_bytes
        def read_bytes(path):
            """Provide only this own synthetic process's command line while reading other actual fixture bytes."""
            return b"python\0-c\0" + source + b"\0" if str(path) == "/proc/self/cmdline" else original_read(path)
        with mock.patch.object(driver, "BASE", base), mock.patch.object(driver, "checked_plan", return_value=plan), \
                mock.patch.object(driver.os, "getuid", return_value=56626), \
                mock.patch.object(driver.pwd, "getpwuid", return_value=types.SimpleNamespace(pw_name="u.sl359075")), \
                mock.patch.object(driver.socket, "gethostname", return_value="alogin1.cluster"), \
                mock.patch.object(driver.sys, "executable", driver.PYTHON), \
                mock.patch.object(driver.sys, "stdin", types.SimpleNamespace(buffer=input_stream)), \
                mock.patch.object(driver.sys, "stdout", types.SimpleNamespace(buffer=output_stream)), \
                mock.patch.object(Path, "read_bytes", read_bytes), \
                mock.patch.object(driver, "old_roots", return_value={"synthetic old root": "unchanged"}), \
                mock.patch.object(driver, "helper_module", return_value=helper), \
                mock.patch.object(driver, "execute", side_effect=execute), \
                mock.patch.object(driver, "git_identity", return_value={"synthetic": True}), \
                mock.patch.object(driver, "mapping_check", return_value={"synthetic": "same original admission"}), \
                mock.patch.dict(driver.os.environ, {}, clear=True), mock.patch.object(driver, "REMOTE_COMMANDS", []):
            driver.remote_receive()
        response = self.directory / "actual_receiver_response.raw"
        response.write_bytes(output_stream.getvalue())
        output = self.directory / "local_receipt"
        output.mkdir()
        receipt = driver.accept_response(response, output, plan, driver.digest(source))
        self.assertEqual(len(receipt["copied"]), 13)
        self.assertEqual(len(receipt["prepared_files"]), 17)
        self.assertTrue(receipt["passed"])
        helper.checked_software.assert_called_once_with({"native": "synthetic metadata"})

    def test_remote_failure_preserves_commands_and_never_retries(self):
        """An actual receiver exception retains command evidence and propagates the failure."""
        output = io.BytesIO()
        with mock.patch.object(driver.sys, "argv", ["-c", "--remote-receive"]), \
                mock.patch.object(driver.sys, "stdout", types.SimpleNamespace(buffer=output)), \
                mock.patch.object(driver.signal, "signal"), mock.patch.object(driver.signal, "alarm") as alarm, \
                mock.patch.object(driver, "REMOTE_COMMANDS", [{"argv": ["git", "rev-parse", "HEAD"], "returncode": 1}]), \
                mock.patch.object(driver, "remote_receive", side_effect=ValueError("synthetic stopped metadata command")) as receive:
            with self.assertRaisesRegex(ValueError, "synthetic stopped"):
                driver.main()
        failure = json.loads(output.getvalue())
        self.assertEqual(receive.call_count, 1)
        self.assertFalse(failure["automatic_retry_permitted"])
        self.assertEqual(failure["commands"][0]["returncode"], 1)
        self.assertEqual(alarm.call_args_list, [mock.call(540), mock.call(0)])

    def test_selector_environment_is_explicit_and_unrelated_gate_controls_reject(self):
        """Native preparation uses the known assets/reference and rejects inherited pytest selectors."""
        with mock.patch.dict(driver.os.environ, {}, clear=True):
            environment = driver.native_environment(self.plan)
            self.assertEqual(environment["XDG_DATA_HOME"], str(driver.BASE / "assets"))
            self.assertTrue(environment["CLEANRL_REFERENCE_FILE"].endswith("cleanrl_source/cleanrl/ppo_rnd_envpool.py"))
        with mock.patch.dict(driver.os.environ, {"PYTEST_ADDOPTS": "-k partial"}, clear=True):
            with self.assertRaisesRegex(ValueError, "pytest selectors"):
                driver.native_environment(self.plan)


if __name__ == "__main__":
    unittest.main()
