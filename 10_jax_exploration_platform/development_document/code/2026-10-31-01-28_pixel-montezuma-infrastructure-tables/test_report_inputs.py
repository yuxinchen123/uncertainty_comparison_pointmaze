"""Metadata checks for excluding probes and retaining exact result provenance."""

import json
from pathlib import Path
import tempfile
import unittest

import make_all


class ReportInputTests(unittest.TestCase):
    def test_timing_protocol_matches_resolved_configuration(self):
        """Accept the reviewed timing windows and reject an outdated ordinary window."""
        with tempfile.TemporaryDirectory() as directory:
            campaign = Path(directory)
            # Before: reviewed scalar settings; after: the same values in flat YAML.
            settings = dict(make_all.source.RESOLVED_SETTINGS)
            configuration = campaign / "config_resolved.yaml"
            configuration.write_text("".join(f"{key}: {json.dumps(value)}\n"
                                             for key, value in settings.items()))
            (campaign / "manifest.yaml").write_text(
                f"cleanrl_commit: {make_all.source.CLEANRL_COMMIT}\n"
                f"jaxtari_commit: {make_all.source.JAXTARI_COMMIT}\n")
            make_all.check_fixed_settings(campaign)

            # Before: one ordinary update; after: obsolete ten-update window, rejected.
            settings["paired_timing_minimum_complete_updates_per_round"] = 10
            configuration.write_text("".join(f"{key}: {json.dumps(value)}\n"
                                             for key, value in settings.items()))
            with self.assertRaisesRegex(ValueError, "paired_timing_minimum_complete_updates_per_round"):
                make_all.check_fixed_settings(campaign)

    def test_unregistered_probe_is_never_used(self):
        with tempfile.TemporaryDirectory() as directory:
            campaign = Path(directory)
            (campaign / "resource_probe_result.json").write_text("malformed probe output")
            registry = {"initial_ideas": [], "adaptive_batches": []}
            self.assertEqual(make_all.registered_cases(campaign, registry), [])

    def test_duplicate_registered_result_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            campaign = Path(directory)
            result = {"idea": "frame storage", "hardware_class": "h100", "copies": 1,
                      "decision": "rejected", "correctness_passed": False}
            (campaign / "result.json").write_text(json.dumps(result))
            summary = {**result, "result_path": "result.json"}
            registry = {"initial_ideas": [{"name": "frame storage", "results": {
                "h100": [summary, summary]}}], "adaptive_batches": []}
            with self.assertRaisesRegex(ValueError, "duplicate registered result"):
                make_all.registered_cases(campaign, registry)

    def test_every_authorized_copy_count_is_required(self):
        complete_one_copy = {"copies": 1, "implementation_distinct": True,
                             "measurement_valid": True, "correctness_passed": True}
        complete_two_copies = {**complete_one_copy, "copies": 2}
        registry = {"initial_ideas": [{"name": "frame storage", "results": {
            "h100": [complete_one_copy, complete_two_copies],
            "jaguar03_rtx_a4500": [complete_one_copy]}}],
            "adaptive_batches": [], "initial_matched_copy_counts": [1, 2]}
        table = make_all.status_table(registry, {})
        row = next(line for line in table.splitlines() if "covering every required copy count" in line)
        self.assertIn("0 of 1", row)
        registry["initial_ideas"][0]["results"]["jaguar03_rtx_a4500"].append(complete_two_copies)
        table = make_all.status_table(registry, {})
        row = next(line for line in table.splitlines() if "covering every required copy count" in line)
        self.assertIn("1 of 1", row)

    def test_table_generation_preserves_caption_and_label(self):
        with tempfile.TemporaryDirectory() as directory:
            document = Path(directory) / "document.tex"
            before = ("\\begin{table}\n% >>> AUTO-GENERATED TABLE START: example\n"
                      "old tabular\n% <<< AUTO-GENERATED TABLE END: example\n"
                      "\\caption{Human-reviewed wording}\n\\label{tab:example}\n\\end{table}\n")
            document.write_text(before)
            make_all.splice(document, {"example": "new tabular"})
            self.assertEqual(document.read_text(), before.replace("old tabular", "new tabular"))
            with self.assertRaisesRegex(ValueError, "exactly one marker pair"):
                make_all.splice(document, {"missing": "new tabular"})


if __name__ == "__main__":
    unittest.main()
