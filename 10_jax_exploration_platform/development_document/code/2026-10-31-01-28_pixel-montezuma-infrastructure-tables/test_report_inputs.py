"""Metadata checks for excluding probes and retaining exact result provenance."""

import json
from pathlib import Path
import sys
import tempfile
import unittest

import make_all


class ReportInputTests(unittest.TestCase):
    def measured_row(self, name, *, hardware="jaguar03_rtx_a4500", copies=1, commit="a" * 40,
                     host="jaguar03", scope="training_iteration_execution", reference=None):
        """Construct one clearly synthetic registered timing without analyzing numerical data."""
        result = {"idea": name, "hardware_class": hardware, "hostname": host, "copies": copies,
                  "source_commit": commit, "measurement_scope": scope,
                  "reference": reference or {"learner": {"n_envs": 128}, "execution": {}},
                  "decision": "accepted", "measurement_valid": True, "correctness_passed": True,
                  "paired_speedups": [1.1] * 7, "median_speedup": 1.1}
        return {"result": result, "registry_summary": {"implementation_distinct": True},
                "path": f"/synthetic/{hardware}/{copies}/{commit}/{host}/{scope}/{name}.json",
                "sha256": make_all.digest(result)}

    def manifest_files(self, directory, records):
        """Create placeholder files for metadata validation without plotting or a PDF library."""
        panels = make_all.plot_specifications(records, "synthetic-fingerprint", directory)
        for panel in panels:
            Path(panel["path"]).write_bytes(b"%PDF-1.4\nsynthetic metadata fixture\n")
        manifest = directory / "plot_manifest.json"
        manifest.write_text(json.dumps(panels))
        return panels, manifest

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

    def test_panels_have_at_most_ten_rows_and_include_every_mechanism(self):
        """Twenty-one registered mechanisms become three complete ordered panels."""
        rows = [self.measured_row(f"mechanism_number_{index:02d}") for index in range(21)]
        panels = make_all.plot_specifications(rows, "synthetic-fingerprint")
        self.assertEqual([len(panel["mechanism_names"]) for panel in panels], [10, 10, 1])
        self.assertEqual([panel["panel_index"] for panel in panels], [1, 2, 3])
        self.assertEqual([panel["panel_count"] for panel in panels], [3, 3, 3])
        self.assertEqual([item for panel in panels for item in panel["results"]],
                         [{"path": row["path"], "sha256": row["sha256"]} for row in rows])
        self.assertNotIn("matplotlib", sys.modules)
        self.assertNotIn("numpy", sys.modules)

    def test_long_labels_wrap_in_full_and_bound_total_panel_height(self):
        """A long mechanism label stays complete rather than being shortened or squeezed."""
        rows = [self.measured_row("reconstruct_reset_aware_observation_stacks_from_individual_pixel_frames_"
                                 f"using_variant_{index}") for index in range(21)]
        panels = make_all.plot_specifications(rows, "synthetic-fingerprint")
        for panel in panels:
            self.assertLessEqual(len(panel["mechanism_names"]), 10)
            self.assertLessEqual(sum(len(label.splitlines()) for label in panel["wrapped_labels"]), 20)
            for name, label in zip(panel["mechanism_names"], panel["wrapped_labels"]):
                self.assertIn("\n", label)
                self.assertEqual(" ".join(label.split()), name.replace("_", " "))

    def test_cohorts_never_mix_copy_counts_source_reference_host_or_scope(self):
        """Every distinct scientific and hardware context receives its own figure cohort."""
        rows = [self.measured_row("base"), self.measured_row("two_copies", copies=2),
                self.measured_row("other_hardware", hardware="h100", host="serval02"),
                self.measured_row("other_source", commit="b" * 40),
                self.measured_row("other_host", host="other_host"),
                self.measured_row("report_readout", scope="training_with_report_readout"),
                self.measured_row("changed_reference", reference={"execution": {"iterations_per_call": 5}})]
        invalid = self.measured_row("invalid")
        invalid["result"]["measurement_valid"] = False
        panels = make_all.plot_specifications(rows + [invalid], "synthetic-fingerprint")
        self.assertEqual(len(panels), 7)
        by_path = {row["path"]: row["result"] for row in rows}
        for panel in panels:
            self.assertEqual(len(panel["results"]), 1)
            result = by_path[panel["results"][0]["path"]]
            for field, result_field in (("hardware", "hardware_class"), ("host", "hostname"),
                                        ("copies", "copies"), ("source_commit", "source_commit"),
                                        ("scope", "measurement_scope")):
                self.assertEqual(panel[field], result[result_field])
            self.assertEqual(panel["reference_fingerprint"], make_all.digest(result["reference"]))

    def test_every_manifest_panel_gets_its_own_exact_scope_figure(self):
        """Include later panels and two-copy results using current manifest provenance."""
        rows = [self.measured_row(f"mechanism_number_{index:02d}") for index in range(21)]
        rows += [self.measured_row("reporting", hardware="h100", copies=2, host="serval02",
                                   scope="training_with_report_readout")]
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            panels, manifest = self.manifest_files(directory, rows)
            text = make_all.figure_source(manifest, panels, directory / "document.tex")
            self.assertEqual(text.count(r"\begin{figure}"), 4)
            self.assertEqual(text.count(r"\includegraphics"), 4)
            self.assertEqual(text.count(r"\label{fig:pixel-montezuma-background-cache-paired}"), 1)
            for panel, block in zip(panels, text.split(r"\begin{figure}")[1:]):
                self.assertIn(Path(panel["path"]).name, block)
                self.assertIn(make_all.HARDWARE_NAMES[panel["hardware"]], block)
                self.assertIn(f"with {panel['copies']} independent policy", block)
                self.assertIn(panel["source_commit"], block)
                self.assertIn(panel["reference_fingerprint"], block)
                self.assertIn(panel["scope"].replace("_", " "), block)
                self.assertIn(f"Panel {panel['panel_index']} of {panel['panel_count']}", block)
                self.assertIn("do not establish learning returns", block)

    def test_stale_fingerprint_incomplete_manifest_and_missing_pdf_fail(self):
        """A metadata-only refresh cannot silently insert old or incomplete plot output."""
        rows = [self.measured_row("one_copy"), self.measured_row("two_copies", copies=2)]
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            panels, manifest = self.manifest_files(directory, rows)
            for changed in ([{**panels[0], "input_fingerprint": "stale"}, panels[1]], panels[:1], panels * 2):
                manifest.write_text(json.dumps(changed))
                with self.assertRaisesRegex(ValueError, "current manifest"):
                    make_all.figure_source(manifest, panels, directory / "document.tex")
            manifest.write_text(json.dumps(panels))
            Path(panels[1]["path"]).unlink()
            with self.assertRaisesRegex(ValueError, "missing PDF"):
                make_all.figure_source(manifest, panels, directory / "document.tex")

    def test_historical_cache_citations_name_both_gpu_figures(self):
        """Cache evidence has separate stable citations even when other panels appear first."""
        rows = [self.measured_row("cache_room_backgrounds", hardware="h100", host="serval02"),
                self.measured_row("cache_room_backgrounds"), self.measured_row("other_setting", copies=2)]
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            panels, manifest = self.manifest_files(directory, rows)
            text = make_all.figure_source(manifest, panels, directory / "document.tex")
        self.assertEqual(text.count(r"\label{fig:pixel-montezuma-cache-h100-one-copy}"), 1)
        self.assertEqual(text.count(r"\label{fig:pixel-montezuma-cache-a4500-one-copy}"), 1)

    def test_figure_splice_preserves_table_blocks_and_reviewed_prose(self):
        """Only the separately marked figure region changes, with unique markers required."""
        marker = make_all.FIGURE_MARKER
        table = ("% >>> AUTO-GENERATED TABLE START: unchanged\noriginal table\n"
                 "% <<< AUTO-GENERATED TABLE END: unchanged\n")
        before = ("Reviewed opening prose.\n" + table
                  + f"% >>> AUTO-GENERATED FIGURES START: {marker}\nold figures\n"
                  + f"% <<< AUTO-GENERATED FIGURES END: {marker}\nReviewed closing prose.\n")
        with tempfile.TemporaryDirectory() as temporary:
            document = Path(temporary) / "document.tex"
            document.write_text(before)
            make_all.splice(document, {}, figures="new figure panels")
            self.assertEqual(document.read_text(), before.replace("old figures", "new figure panels"))
            for text in (table, before + before):
                document.write_text(text)
                with self.assertRaisesRegex(ValueError, "exactly one marker pair"):
                    make_all.splice(document, {}, figures="new figure panels")
                self.assertEqual(document.read_text(), text)


if __name__ == "__main__":
    unittest.main()
