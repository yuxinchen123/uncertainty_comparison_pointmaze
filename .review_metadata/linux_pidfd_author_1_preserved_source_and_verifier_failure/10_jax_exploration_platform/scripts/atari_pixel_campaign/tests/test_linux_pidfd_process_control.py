"""Test explicit native process handles on owned CPU only, never on a GPU or standing lock."""
import copy
import ctypes
import errno
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import unittest
from unittest.mock import patch

SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS))
import linux_pidfd_process_control as native
import nsight_process_identity as metadata


def retain(name, value):
    """Save only labelled CPU-native evidence under the explicit test output path."""
    root = os.environ.get("NATIVE_PIDFD_REVIEW_OUTPUT")
    if root:
        path = Path(root); path.mkdir(parents=True, exist_ok=True)
        output = path / (name + ".json")
        with output.open("x") as stream:json.dump(value, stream, indent=2); stream.write("\n")


class NativePidfdContractTests(unittest.TestCase):
    """Refuse incorrect contracts before a native handle can manage any process."""
    @classmethod
    def setUpClass(cls):
        """Capture actual local primary metadata only within this campaign's bounded CPU allocation."""
        assert os.environ.get("SLURM_JOB_ID") == "6783003" and os.environ["SLURM_STEP_ID"].isdecimal()
        cls.contract, cls.bootstrap = native.current_contract()

    def test_contract_uses_actual_public_headers_symbols_and_selected_python(self):
        """The captured contract has exact syscall declarations and actual mapped libc offsets."""
        self.assertEqual(self.contract["syscall_numbers"], {"pidfd_open": 434, "pidfd_send_signal": 424})
        self.assertEqual(set(self.contract["headers"]), set(native.HEADERS))
        self.assertEqual(self.contract["symbols"]["syscall"]["image"], self.contract["symbols"]["close"]["image"])
        self.assertEqual(self.contract["selected_python"]["path"], str(Path(sys.executable).resolve()))
        self.assertFalse(self.bootstrap["gpu_or_cuda_qualification"])

    def test_native_backend_requires_an_explicit_kind_and_exact_contract(self):
        """A missing declaration, changed image, header, ABI or interpreter never enables this provider."""
        for field in ("kind", "abi", "syscall_numbers", "headers", "symbols", "selected_python", "extra"):
            contract = copy.deepcopy(self.contract)
            if field == "kind":contract[field] = "python-default"
            elif field == "abi":contract[field]["long_bytes"] = 4
            elif field == "syscall_numbers":contract[field]["pidfd_open"] = 435
            elif field == "headers":contract[field][native.HEADERS[0]]["sha256"] = "0" * 64
            elif field == "symbols":contract[field]["syscall"]["symbol_file_offset"] += 1
            elif field == "selected_python":contract[field]["sha256"] = "0" * 64
            else:contract["unknown_future_field"] = True
            with self.subTest(field=field), self.assertRaises(ValueError):native.NativePidfdProvider(contract)

    def test_actual_self_preflight_opens_signals_zero_and_closes_before_other_use(self):
        """Three successful native calls establish process control, with no claim about CUDA."""
        provider = native.NativePidfdProvider(self.contract)
        evidence = provider.evidence()
        self.assertEqual([row["name"] for row in evidence["calls"]], ["pidfd_open", "pidfd_send_signal", "close"])
        self.assertEqual(evidence["calls"][0]["arguments"]["pid"], os.getpid())
        self.assertEqual(evidence["calls"][1]["arguments"]["signal"], 0)
        self.assertTrue(all(type(row["result"]) is int and type(row["errno"]) is int and row["errno"] == 0 for row in evidence["calls"]))
        self.assertEqual(evidence["open_handle_count"], 0)
        self.assertFalse(evidence["python_exports_changed"])
        self.assertFalse(evidence["gpu_or_cuda_qualification"])
        retain("actual_self_preflight", evidence)

    def test_unsupported_kernel_response_is_recorded_and_refused_without_fallback(self):
        """A labelled synthetic ENOSYS response proves refusal, not actual kernel incompatibility."""
        def unsupported(provider, pid, phase):
            def returned():ctypes.set_errno(errno.ENOSYS); return -1
            return provider.invoke("pidfd_open", {"syscall": 434, "pid": pid, "flags": 0}, returned, phase=phase)
        with patch.object(native.NativePidfdProvider, "_open", unsupported):
            with self.assertRaises(OSError) as refused:native.NativePidfdProvider(self.contract)
        self.assertEqual(refused.exception.errno, errno.ENOSYS)
        evidence = refused.exception.native_process_control_evidence
        self.assertFalse(evidence["preflight_passed"])
        self.assertEqual(evidence["calls"][0]["result"], -1)
        self.assertEqual(evidence["calls"][0]["errno"], errno.ENOSYS)
        retain("synthetic_unsupported_kernel_response", {"synthetic_kernel_response": True, "evidence": evidence})

    def test_unknown_and_boolean_handles_signals_and_pids_are_refused(self):
        """No malformed argument reaches a target process or becomes a numeric-PID signal."""
        provider = native.NativePidfdProvider(self.contract)
        before = len(provider.calls)
        for value in (True, -1, 0, 2147483648, "100"):
            with self.subTest(pid=value), self.assertRaises(ValueError):provider.open_process_handle(value)
        for value in (True, -1, 123456):
            with self.subTest(fd=value), self.assertRaises(ValueError):provider.signal_handle(value, 0)
            with self.subTest(close=value), self.assertRaises(ValueError):provider.close_process_handle(value)
        self.assertEqual(len(provider.calls), before)

    def test_native_provider_does_not_fabricate_missing_python_exports(self):
        """The selected interpreter retains its actual build capabilities and default refusal."""
        before = (getattr(os, "pidfd_open", None), getattr(signal, "pidfd_send_signal", None))
        provider = native.NativePidfdProvider(self.contract)
        observer = native.NativeProcessObserver(metadata, provider)
        self.assertEqual(before, (getattr(os, "pidfd_open", None), getattr(signal, "pidfd_send_signal", None)))
        if not all(callable(value) for value in before):
            with self.assertRaisesRegex(ValueError, "lacks required"):metadata.require_process_control_apis()
        with self.assertRaisesRegex(ValueError, "cannot fabricate"):observer.require_process_control_apis()

    def test_inherited_or_not_yet_ready_state_cannot_manage_a_process(self):
        """Explicit provider state belongs to its one actual producer and completed preflight."""
        provider = native.NativePidfdProvider(self.contract)
        provider.owner_pid += 1
        with self.assertRaisesRegex(ValueError, "another process"):provider.open_process_handle(os.getpid())
        provider.owner_pid = os.getpid(); provider.preflight_passed = False
        with self.assertRaisesRegex(ValueError, "preflight"):provider.open_process_handle(os.getpid())

    def test_adapter_keeps_default_identity_functions_and_explicit_control_separate(self):
        """No generic module function is replaced when an explicit adapter is created."""
        original = metadata.open_process_handle
        provider = native.NativePidfdProvider(self.contract)
        observer = native.NativeProcessObserver(metadata, provider)
        self.assertIs(metadata.open_process_handle, original)
        self.assertIs(observer.ProcReader, metadata.ProcReader)
        self.assertIs(observer.open_process_handle.__self__, provider)

    def test_actual_owned_child_uses_same_uid_cgroup_birth_checks_and_pidfd_cleanup(self):
        """Manage one fresh CPU child through the unchanged ancestry reader with the explicit provider."""
        provider = native.NativePidfdProvider(self.contract)
        observer = native.NativeProcessObserver(metadata, provider)
        before = (getattr(os, "pidfd_open", None), getattr(signal, "pidfd_send_signal", None))
        parent = metadata.ProcReader().snapshot(os.getpid(), os.getuid())
        child = subprocess.Popen([sys.executable, "-I", "-S", "-B", "-c", "import sys;print('ready',flush=True);sys.stdin.readline()"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, close_fds=True, pass_fds=())
        owned = None
        try:
            self.assertEqual(child.stdout.readline(), b"ready\n")
            actual = metadata.ProcReader().snapshot(child.pid, os.getuid(), parent["cgroup"])
            owned = observer.OwnedProcesses(child.pid, actual["process_start_time"], parent)
            self.assertFalse(provider.handle_exited(owned.handles[child.pid]))
            provider.signal_handle(owned.handles[child.pid], 0)
            stopped = owned.stop(child, timeout=5)
            self.assertEqual(stopped["popen_returncode"], -int(signal.SIGTERM))
            owned.close(); owned = None
            evidence = provider.evidence()
            self.assertEqual(evidence["open_handle_count"], 0)
            self.assertEqual(before, (getattr(os, "pidfd_open", None), getattr(signal, "pidfd_send_signal", None)))
            self.assertTrue(all(type(row["result"]) is int and row["errno"] == 0 for row in evidence["calls"]))
            retain("actual_temporary_owned_child", {"child_pid": child.pid, "child_birth": actual["process_start_time"],
                "source_identity": actual, "parent": parent, "stop": stopped, "native_process_control": evidence,
                "environment_transitions": 0, "gpu_operations": 0, "standing_gpu_lock_opened": False})
        finally:
            if child.poll() is None:
                if owned is not None:owned.stop(child, timeout=5)
                else:child.terminate(); child.wait(timeout=5)
            if owned is not None:owned.close()
            child.stdin.close(); child.stdout.close(); child.stderr.close()


if __name__ == "__main__":
    unittest.main()
