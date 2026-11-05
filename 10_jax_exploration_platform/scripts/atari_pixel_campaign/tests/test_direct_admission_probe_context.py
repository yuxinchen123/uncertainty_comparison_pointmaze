"""Check only complete job and step components in saved CPU cgroup text."""
import importlib.util
from pathlib import Path
import unittest

PATH = Path(__file__).resolve().parents[1] / "probe_direct_current_admission.py"
SPEC = importlib.util.spec_from_file_location("direct_claim_context_fixture", PATH)
probe = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(probe)


class DirectAdmissionProbeContextTests(unittest.TestCase):
    def test_exact_owned_components(self):
        raw = "0::/system.slice/slurmstepd.scope/job_6821437/step_33/user/task_0\n"
        self.assertTrue(probe.owned_step_path(raw, "6821437", "33"))
        self.assertTrue(probe.owned_step_path("0::/job_6821437/step_33", "6821437", "33"))

    def test_job_and_step_prefixes_refused(self):
        raw = "0::/system.slice/slurmstepd.scope/job_6821437/step_33/user/task_0\n"
        self.assertFalse(probe.owned_step_path(raw, "6821437", "3"))
        self.assertFalse(probe.owned_step_path(raw, "682143", "33"))

    def test_separate_or_longer_components_refused(self):
        self.assertFalse(probe.owned_step_path("0::/job_6821437/other/step_33/", "6821437", "33"))
        self.assertFalse(probe.owned_step_path("0::/job_6821437/step_33_extra/", "6821437", "33"))


if __name__ == "__main__":
    unittest.main()
