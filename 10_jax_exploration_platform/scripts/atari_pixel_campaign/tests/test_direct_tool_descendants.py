"""Run actual CPU-only orphan/startup failures and require lock retention through kernel closure."""
import json
import os
from pathlib import Path
import subprocess
import shutil
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import direct_tool_descendants as scope_source
from test_nsight_process_identity import M, stat_text

SCRIPTS = Path(__file__).resolve().parents[1]


class PinnedParentChildListTests(unittest.TestCase):
    """Exercise private proc directories and synthetic process handles, never actual other processes."""
    def setUp(self):
        """Prepare a known non-direct parent with pinned birth30 and a private child list."""
        self.temporary = tempfile.TemporaryDirectory(); self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.cgroup = "0::/synthetic-owned-allocation\n"
        self.dead, self.reads = set(), []
        self.process(100, 99, 10, [101])
        self.process(102, 101, 30, [303])
        self.scope = scope_source.OwnedToolScope.__new__(scope_source.OwnedToolScope)
        self.scope.pid, self.scope.uid, self.scope.birth, self.scope.cgroup = 100, 888, "10", self.cgroup
        self.scope.producer_handle = 1100
        self.scope.handles = {102: 1102}
        self.scope.nodes = {102: {"initial": {"process_start_time": "30"}}}
        self.scope.observer = SimpleNamespace(parsed_uids=M.parsed_uids,
            handle_exited=lambda handle: handle in self.dead)
        self.original_bounded = scope_source.bounded
        self.proc_patch = patch.object(scope_source, "PROC_ROOT", self.root)
        self.proc_patch.start(); self.addCleanup(self.proc_patch.stop)

    def process(self, pid, parent, birth, children):
        """Write one labelled private process directory, including its thread's stat bytes."""
        path = self.root / str(pid); thread = path / "task" / str(pid)
        thread.mkdir(parents=True)
        raw = stat_text(pid, parent, pid, pid, birth)
        (path / "stat").write_text(raw); (thread / "stat").write_text(raw)
        (path / "status").write_text("Uid:\t888\t888\t888\t888\n")
        (path / "cgroup").write_text(self.cgroup)
        (thread / "children").write_text(" ".join(str(child) for child in children))

    def test_pinned_live_parent_keeps_raw_birth_and_thread_identity_before_and_after(self):
        """The saved positive read includes both actual pinned-parent and pinned-thread identities."""
        children, rows = self.scope.child_lists(102)
        self.assertEqual(children, {303})
        self.assertEqual(rows[0]["parent_read_identity"]["expected_birth"], "30")
        self.assertTrue(rows[0]["parent_read_identity"]["pidfd_live_after"])
        self.assertEqual(scope_source.parsed_identity(rows[0]["thread_stat_after"])["pid"], 102)

    def test_reused_non_direct_parent_is_refused_before_reading_any_child_list(self):
        """A same-UID PID with another birth cannot introduce its child303 into the owned scope."""
        (self.root / "102/stat").write_text(stat_text(102, 101, 102, 102, 99))
        def read(directory, name):
            self.reads.append(name)
            return self.original_bounded(directory, name)
        with patch.object(scope_source, "bounded", side_effect=read), self.assertRaisesRegex(ValueError, "pinned birth"):
            self.scope.child_lists(102)
        self.assertNotIn("children", self.reads)
        self.assertNotIn(303, self.scope.handles)

    def test_exited_parent_handle_refuses_before_opening_the_numeric_proc_directory(self):
        """An already exited process handle cannot authorize inspection of a reused numeric PID."""
        self.dead.add(1102)
        with patch.object(scope_source.os, "open", side_effect=AssertionError("numeric proc access")), \
            self.assertRaisesRegex(ValueError, "exited before child-list"):
            self.scope.child_lists(102)

    def test_parent_identity_change_during_child_read_cannot_return_child_edges(self):
        """A changed birth, owner or cgroup after child access leaves scope closure unverified."""
        for field in ("birth", "uid", "cgroup"):
            with self.subTest(field=field):
                changed = []
                def read(directory, name):
                    raw = self.original_bounded(directory, name)
                    if name == "children" and not changed:
                        changed.append(True)
                        if field == "birth":(self.root / "102/stat").write_text(stat_text(102, 101, 102, 102, 99))
                        elif field == "uid":(self.root / "102/status").write_text("Uid:\t999\t999\t999\t999\n")
                        else:(self.root / "102/cgroup").write_text("0::/another-allocation\n")
                    return raw
                with patch.object(scope_source, "bounded", side_effect=read), self.assertRaises(ValueError):
                    self.scope.child_lists(102)
                self.assertNotIn(303, self.scope.handles)
                (self.root / "102/stat").write_text(stat_text(102, 101, 102, 102, 30))
                (self.root / "102/status").write_text("Uid:\t888\t888\t888\t888\n")
                (self.root / "102/cgroup").write_text(self.cgroup)

    def test_parent_handle_exit_after_child_read_cannot_return_child_edges(self):
        """The final retained-handle check remains required after all child bytes were read."""
        def read(directory, name):
            raw = self.original_bounded(directory, name)
            if name == "children":self.dead.add(1102)
            return raw
        with patch.object(scope_source, "bounded", side_effect=read), self.assertRaisesRegex(ValueError, "handle exited"):
            self.scope.child_lists(102)
        self.assertNotIn(303, self.scope.handles)

    def test_numeric_directory_replacement_cannot_redirect_the_pinned_child_read(self):
        """All child reads stay relative to the original parent directory rather than reopening its PID."""
        changed = []
        def read(directory, name):
            raw = self.original_bounded(directory, name)
            if name == "children" and not changed:
                changed.append(True)
                (self.root / "102").rename(self.root / "old-parent-pinned-directory")
                self.process(102, 777, 99, [909])
            return raw
        with patch.object(scope_source, "bounded", side_effect=read):
            children, rows = self.scope.child_lists(102)
        self.assertEqual(children, {303})
        self.assertNotIn(909, children)
        self.assertEqual(rows[0]["parent_read_identity"]["after"]["process_start_time"], "30")

    def test_thread_identity_change_during_read_is_refused(self):
        """Even an unchanged parent cannot make a reused thread's child list acceptable."""
        def read(directory, name):
            raw = self.original_bounded(directory, name)
            if name == "children":(self.root / "102/task/102/stat").write_text(stat_text(102, 101, 102, 102, 99))
            return raw
        with patch.object(scope_source, "bounded", side_effect=read), self.assertRaisesRegex(ValueError, "own-thread identity"):
            self.scope.child_lists(102)

    def test_unknown_parent_has_no_proc_directory_access(self):
        """Only a previously pinned positive parent may supply another process edge."""
        with patch.object(scope_source.os, "open", side_effect=AssertionError("unknown proc access")), \
            self.assertRaisesRegex(ValueError, "already pinned owned parent"):
            self.scope.child_lists(303)


class RealCpuDescendantClosureTests(unittest.TestCase):
    """Use isolated native-process-control fixtures; no Nsight, GPU or standing lock is opened."""
    def regression(self, mode):
        """Run the real source cleanup and retain exact temporary process/lock evidence."""
        with tempfile.TemporaryDirectory() as temporary:
            result = subprocess.run(["/usr/bin/python3", "-I", "-S", "-B",
                str(SCRIPTS / "tests/direct_tool_descendants_cpu_worker.py"), temporary, mode],
                capture_output=True, timeout=25, check=False, close_fds=True, pass_fds=())
            self.assertEqual(result.returncode, 0, result.stderr.decode() + result.stdout.decode())
            record = json.loads(result.stdout)
            preserved = os.environ.get("DIRECT_CPU_SCOPE_REVIEW_OUTPUT")
            if preserved:
                target = Path(preserved) / mode
                target.mkdir(parents=True, exist_ok=False)
                mappings = []
                source_root = Path(temporary)
                for source in sorted(source_root.rglob("*.json")):
                    if "code" in source.parts:
                        continue
                    destination = target / source.relative_to(source_root)
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(source, destination)
                    mappings.append({"original_path": str(source), "preserved_path": str(destination)})
                (target / "worker_stdout.raw").write_bytes(result.stdout)
                (target / "worker_stderr.raw").write_bytes(result.stderr)
                (target / "temporary_evidence_paths.json").write_text(json.dumps(mappings, indent=2) + "\n")
            self.assertTrue(record["passed"])
            self.assertTrue(record["payload_exit_observed_by_independent_pidfd"])
            self.assertTrue(record["temporary_lock_retained_until_all_descendants_exited"])
            self.assertTrue(record["temporary_lock_reacquired_only_after_verified_descendant_exit"])
            self.assertTrue(record["kernel_terminal"]["actual_echild"])
            self.assertFalse(record["standing_gpu_lock_opened"])
            self.assertFalse(record["nsight_executed"])
            self.assertEqual(record["gpu_operations"], 0)
            return record

    def test_startup_timeout_cleans_actual_owned_payload_before_temporary_lock_release(self):
        """Reproduce the rejected startup path with a real different-session CPU payload."""
        record = self.regression("startup_timeout")
        self.assertTrue(record["actual_root_waits"])

    def test_fast_failed_root_and_orphaned_payload_are_reaped_before_lock_release(self):
        """A failed root and double-forked orphan cannot escape the producer's subreaper scope."""
        record = self.regression("orphan_root_failure")
        self.assertIn("middle_pid", record["actual_processes"])


if __name__ == "__main__":
    unittest.main()
