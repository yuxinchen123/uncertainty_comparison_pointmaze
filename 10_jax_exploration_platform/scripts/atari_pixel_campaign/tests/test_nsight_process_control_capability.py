"""Check interpreter API refusal and pre-tool ordering without actual process, tool or device operations."""
from contextlib import contextmanager
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

SCRIPTS = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("capability_checked_nsight_diagnostic", SCRIPTS / "diagnose_nsight_process_tree.py")
D = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(D)
M = D.identity


@contextmanager
def apis(open_function=None, send_function=None):
    """Replace only the API namespaces with clearly synthetic callable or missing entries."""
    with mock.patch.object(M.os, "pidfd_open", open_function, create=True), \
            mock.patch.object(M.signal, "pidfd_send_signal", send_function, create=True):
        yield


class InterpreterCapabilityTests(unittest.TestCase):
    """Refuse unsupported Python exports before any tool command or process creation."""
    def test_both_callable_apis_pass_without_opening_or_signalling(self):
        """An API-presence check does not claim or execute a kernel syscall."""
        opening, sending = mock.Mock(), mock.Mock()
        with apis(opening, sending):
            self.assertEqual(M.require_process_control_apis(), {"os.pidfd_open": True, "signal.pidfd_send_signal": True})
        opening.assert_not_called()
        sending.assert_not_called()

    def test_missing_open_names_only_the_absent_export(self):
        """The failure describes this interpreter's missing export without blaming the kernel."""
        with apis(None, mock.Mock()), self.assertRaisesRegex(ValueError, "APIs: os.pidfd_open$"):
            M.require_process_control_apis()

    def test_missing_send_names_only_the_absent_export(self):
        """An available open function cannot hide a missing descriptor-signal API."""
        with apis(mock.Mock(), None), self.assertRaisesRegex(ValueError, "APIs: signal.pidfd_send_signal$"):
            M.require_process_control_apis()

    def test_both_missing_exports_are_reported(self):
        """The actually observed selected-interpreter shape reports both missing names."""
        with apis(), self.assertRaisesRegex(ValueError, "APIs: os.pidfd_open, signal.pidfd_send_signal$"):
            M.require_process_control_apis()

    def test_noncallable_exports_are_not_accepted_as_capabilities(self):
        """An attribute with a non-function value cannot enable a later tool launch."""
        for opening, sending in (("not callable", mock.Mock()), (mock.Mock(), False)):
            with self.subTest(opening=repr(opening), sending=repr(sending)), apis(opening, sending):
                with self.assertRaises(ValueError):
                    M.require_process_control_apis()

    def test_process_open_checks_required_apis_then_uses_only_descriptor_open(self):
        """The unchanged process-control path still opens the exact requested PID with no signal fallback."""
        opening, sending = mock.Mock(return_value=71), mock.Mock()
        with apis(opening, sending):
            self.assertEqual(M.open_process_handle(123), 71)
        opening.assert_called_once_with(123, 0)
        sending.assert_not_called()

    def test_missing_signal_export_refuses_before_process_open(self):
        """No descriptor is opened if this interpreter cannot support its required cleanup operation."""
        opening = mock.Mock()
        with apis(opening, None), self.assertRaises(ValueError):
            M.open_process_handle(123)
        opening.assert_not_called()

    def test_actual_diagnostic_run_refuses_before_plan_tool_or_popen(self):
        """The missing-export failure occurs before any plan read, version command or process creation."""
        with apis(), mock.patch.object(D, "read_plan") as plan, mock.patch.object(D.subprocess, "run") as version, \
                mock.patch.object(D.subprocess, "Popen") as process:
            with self.assertRaisesRegex(ValueError, "this Python interpreter lacks"):
                D.run(Path("/synthetic/unopened/unit"))
        plan.assert_not_called()
        version.assert_not_called()
        process.assert_not_called()

    def test_supported_api_record_is_written_before_version_without_kernel_claim(self):
        """A passing API check records its actual interpreter before the simulated tool version call."""
        with tempfile.TemporaryDirectory() as temporary:
            unit = Path(temporary)
            tool = unit / "synthetic_unexecuted_tool"
            tool.write_bytes(b"synthetic tool bytes; never execute")
            plan = {"profile_tool": {"path": str(tool), "sha256": M.receipt(tool)["sha256"], "version": "synthetic version"}}
            def version(*arguments, **keywords):
                """Inspect the actual metadata file before returning a synthetic version failure."""
                value = json.loads((unit / "process_control_capabilities.json").read_text())
                self.assertEqual(value["required_apis_callable"], {"os.pidfd_open": True, "signal.pidfd_send_signal": True})
                self.assertFalse(value["kernel_support_tested_by_this_check"])
                self.assertEqual(value["scientific_credit"], 0)
                return SimpleNamespace(returncode=1, stdout=b"", stderr=b"synthetic failure")
            with apis(mock.Mock(), mock.Mock()), mock.patch.object(D, "read_plan", return_value=(unit, plan)), \
                    mock.patch.object(D, "wait_for_owned_launch"), mock.patch.object(D.subprocess, "run", side_effect=version), \
                    mock.patch.object(D.subprocess, "Popen") as process:
                with self.assertRaisesRegex(ValueError, "installed Nsight tool changed"):
                    D.run(unit)
                process.assert_not_called()

    def test_kernel_open_error_propagates_without_pid_signal_fallback(self):
        """A synthetic kernel error is retained directly after the interpreter API check passes."""
        opening, sending = mock.Mock(side_effect=OSError(38, "synthetic unsupported syscall")), mock.Mock()
        with apis(opening, sending), self.assertRaisesRegex(OSError, "synthetic unsupported syscall"):
            M.open_process_handle(123)
        opening.assert_called_once_with(123, 0)
        sending.assert_not_called()

