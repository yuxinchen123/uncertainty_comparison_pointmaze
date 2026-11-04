"""Check synthetic owned-process records without launching Nsight, processes, signals or array libraries."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
SCRIPTS = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("synthetic_nsight_identity", SCRIPTS / "nsight_process_identity.py")
M = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(M)


def stat_text(pid, parent, group, session, birth):
    """Write a labelled synthetic Linux stat record with independently supplied identity fields."""
    fields = ["S", str(parent), str(group), str(session)] + ["0"] * 16
    fields[19] = str(birth)
    return str(pid) + " (synthetic python) " + " ".join(fields) + "\n"


class ProcessObservationTests(unittest.TestCase):
    """Use private proc-file fixtures and mocked pidfds only."""
    def setUp(self):
        """Create a supervisor, Nsight root, new-session waiter and new-session capture."""
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.cgroup = "0::/system.slice/slurmstepd.scope/job_123456789/step_0/user/task_0\n"
        self.capture_command = ["/synthetic/python", "/unit/code/profile_selected.py", "--capture-unit", "/unit"]
        for pid, parent, group, session, birth, command, children in (
            (100, 99, 100, 100, 10, ["supervisor"], []),
            (101, 100, 101, 101, 20, ["nsys", "profile"], [102]),
            (102, 101, 102, 102, 30, ["metadata-waiter"], [103]),
            (103, 102, 103, 103, 40, self.capture_command, [])):
            self.process(pid, parent, group, session, birth, command, children)
        self.reader = M.ProcReader(self.root)
        self.parent = self.reader.snapshot(100, 888)
        self.closed, self.signalled, self.dead = [], [], set()
        self.patches = [
            patch.object(M, "open_process_handle", side_effect=lambda pid: pid + 1000),
            patch.object(M, "close_process_handle", side_effect=self.closed.append),
            patch.object(M, "handle_exited", side_effect=lambda fd, milliseconds=0: fd in self.dead),
            patch.object(M, "signal_handle", side_effect=self.signal)]
        for item in self.patches: item.start(); self.addCleanup(item.stop)

    def process(self, pid, parent, group, session, birth, command, children):
        """Write one complete synthetic process, including its own kernel child list."""
        p = self.root / str(pid); (p / "task" / str(pid)).mkdir(parents=True)
        (p / "stat").write_text(stat_text(pid, parent, group, session, birth))
        (p / "status").write_text("Name:\tsynthetic\nUid:\t888\t888\t888\t888\n")
        (p / "cgroup").write_text(self.cgroup)
        (p / "cmdline").write_bytes(b"\0".join(value.encode() for value in command) + b"\0")
        (p / "exe").symlink_to("/synthetic/executable")
        (p / "task" / str(pid) / "children").write_text(" ".join(str(child) for child in children))

    def signal(self, descriptor, number):
        """Record synthetic signals and make only that pinned synthetic handle exit."""
        self.signalled.append((descriptor, int(number))); self.dead.add(descriptor)

    def observe(self):
        """Exercise actual collection and serialized reconstruction on the synthetic tree."""
        observer = M.OwnedProcesses(101, "20", self.parent, self.reader)
        self.addCleanup(observer.close)
        value = observer.observe(103, "40", self.capture_command)
        return observer, json.loads(json.dumps(value))

    def test_different_process_groups_and_sessions_pass_with_real_edges(self):
        """A new session is valid only with an independently proved positive ancestry path."""
        observer, value = self.observe()
        rebuilt = M.validate_observation(value, {"pid": 103, "process_start_time": "40"}, self.capture_command, self.parent)
        self.assertEqual(rebuilt["path"], [102, 103])
        self.assertNotEqual(rebuilt["capture"]["process_group"], rebuilt["root"]["pid"])
        self.assertFalse(value["whole_tree_completeness_established"])

    def test_foreign_user_refuses_before_command_inspection(self):
        """Do not retain a foreign descendant command even if a child list names its PID."""
        (self.root / "102/status").write_text("Uid:\t999\t999\t999\t999\n")
        (self.root / "102/cmdline").write_bytes(b"\xff")
        with self.assertRaisesRegex(ValueError, "not owned"):
            self.observe()

    def test_other_allocation_refuses_before_command_inspection(self):
        """An actual PID from another cgroup cannot authorize a capture."""
        (self.root / "102/cgroup").write_text(self.cgroup.replace("123456789", "987654321"))
        (self.root / "102/cmdline").write_bytes(b"\xff")
        with self.assertRaisesRegex(ValueError, "allocation"):
            self.observe()

    def test_mismatched_parent_edge_refuses(self):
        """A child-list candidate must independently report the same actual parent."""
        (self.root / "103/stat").write_text(stat_text(103, 777, 103, 103, 40))
        with self.assertRaisesRegex(ValueError, "edge"):
            self.observe()

    def test_wrong_capture_birth_or_source_command_refuses(self):
        """A reused PID or another Python entry cannot receive release."""
        for birth, command in (("41", self.capture_command), ("40", ["different-python"])):
            observer = M.OwnedProcesses(101, "20", self.parent, self.reader)
            try:
                with self.assertRaisesRegex(ValueError, "birth or exact source"):
                    observer.observe(103, birth, command)
            finally:
                observer.close()

    def test_missing_claimed_pid_never_causes_arbitrary_proc_inspection(self):
        """Search only observed owned descendants, rather than walking up a claimed foreign PID."""
        observer = M.OwnedProcesses(101, "20", self.parent, self.reader)
        self.addCleanup(observer.close)
        with self.assertRaisesRegex(ValueError, "positively observed"):
            observer.observe(999999999, "1", self.capture_command)
        self.assertEqual(set(observer.nodes), {101, 102, 103})

    def test_exit_or_reuse_during_recheck_refuses(self):
        """Even an initially valid edge cannot outlive its pinned identity."""
        observer = M.OwnedProcesses(101, "20", self.parent, self.reader)
        self.addCleanup(observer.close)
        self.dead.add(1101)
        with self.assertRaisesRegex(ValueError, "changed before release"):
            observer.observe(103, "40", self.capture_command)

    def test_launcher_image_transition_preserves_the_pinned_identity(self):
        """Allow an observed executable transition without inventing the installed Nsight tree."""
        observer = M.OwnedProcesses(101, "20", self.parent, self.reader)
        self.addCleanup(observer.close)
        (self.root / "101/cmdline").write_bytes(b"another-owned-launcher\0")
        (self.root / "101/exe").unlink()
        (self.root / "101/exe").symlink_to("/synthetic/actual-launcher")
        value = json.loads(json.dumps(observer.observe(103, "40", self.capture_command)))
        M.validate_observation(value, {"pid": 103, "process_start_time": "40"}, self.capture_command, self.parent)
        self.assertEqual(value["root_initial"]["command"], ["nsys", "profile"])
        self.assertEqual(value["nodes"]["101"]["command"], ["another-owned-launcher"])

    def test_launcher_transition_cannot_change_its_birth_or_user(self):
        """A permitted executable change never permits another process or user."""
        observer = M.OwnedProcesses(101, "20", self.parent, self.reader)
        self.addCleanup(observer.close)
        (self.root / "101/stat").write_text(stat_text(101, 100, 101, 101, 21))
        with self.assertRaisesRegex(ValueError, "root identity changed"):
            observer.observe(103, "40", self.capture_command)

    def test_both_raw_user_snapshots_and_exact_types_are_required(self):
        """A later user change or Boolean user identifier cannot authorize release."""
        for field, value in (("status_after", "Uid:\t999\t999\t999\t999\n"), ("uids", [True] * 4)):
            unused, observed = self.observe()
            observed["nodes"]["103"][field] = value
            with self.assertRaisesRegex(ValueError, "ownership"):
                M.validate_observation(observed, {"pid": 103, "process_start_time": "40"}, self.capture_command, self.parent)

    def test_saved_edge_requires_original_raw_child_list(self):
        """Coherent summary fields cannot replace actual positive kernel child-list evidence."""
        unused_observer, value = self.observe()
        value["child_lists"][1]["thread_children"][0].update(raw="", sha256=hashlib.sha256(b"").hexdigest())
        with self.assertRaisesRegex(ValueError, "edge"):
            M.validate_observation(value, {"pid": 103, "process_start_time": "40"}, self.capture_command, self.parent)

    def test_saved_command_and_user_raw_bytes_are_checked(self):
        """Reject changed summaries under unchanged raw process observations."""
        for key, changed in (("command", ["another"]), ("uids", [999] * 4)):
            unused_observer, value = self.observe()
            value["nodes"]["103"][key] = changed
            with self.assertRaises(ValueError):
                M.validate_observation(value, {"pid": 103, "process_start_time": "40"}, self.capture_command, self.parent)

    def test_saved_pid_alias_or_boolean_identity_refuses(self):
        """Keep canonical PID strings and exact integer identity types."""
        unused_observer, value = self.observe()
        value["nodes"]["0103"] = value["nodes"].pop("103")
        with self.assertRaisesRegex(ValueError, "canonical"):
            M.validate_observation(value, {"pid": 103, "process_start_time": "40"}, self.capture_command, self.parent)
        unused_observer, value = self.observe()
        value["nodes"]["103"]["pid"] = True
        with self.assertRaises(ValueError):
            M.validate_observation(value, {"pid": 103, "process_start_time": "40"}, self.capture_command, self.parent)

    def test_cleanup_uses_only_pinned_handles_and_actual_popen_wait(self):
        """Never select a signal target from a process group or an unproved PID."""
        observer, unused_value = self.observe()
        child = unittest.mock.Mock(pid=101); child.wait.return_value = -15
        result = observer.stop(child, timeout=0)
        child.wait.assert_called_once_with()
        self.assertEqual(result["popen_returncode"], -15)
        self.assertEqual({descriptor for descriptor, number in self.signalled}, {1101, 1102, 1103})
        self.assertTrue(result["unobserved_descendant_exit_not_claimed"])


if __name__ == "__main__":
    unittest.main()
