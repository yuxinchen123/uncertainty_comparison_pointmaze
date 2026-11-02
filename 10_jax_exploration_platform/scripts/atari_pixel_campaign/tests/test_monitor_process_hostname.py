"""Verify both actual process host field formats without changing live records."""

import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import monitor

FIXTURE = Path(__file__).with_name("fixtures") / "actual_processes_host_and_hostname_2026_11_02.json"


class ProcessHostnameTests(unittest.TestCase):
    """Keep actual direct identity and historical rows readable, with explicit invalid-input errors."""

    def test_actual_registry_and_hostname_only_row(self):
        """The actual sixteen-row registry includes the new direct row without a host alias."""
        records = json.loads(FIXTURE.read_text())
        hosts = monitor.process_hosts(records)
        self.assertEqual(len(records), 16)
        self.assertEqual(len(hosts), 16)
        current = [record for record in records if "hostname" in record and "host" not in record]
        self.assertEqual(len(current), 1)
        self.assertEqual(current[0]["kind"], "selected_configuration_measurement")
        self.assertEqual(hosts[current[0]["unit"]], "serval02")
        self.assertEqual(hosts, {record["unit"]: "serval02" for record in records})

    def test_host_only_hostname_only_and_exact_agreed_aliases(self):
        """Historical, current and equal dual fields give the same explicit host value."""
        for fields in ({"host": "serval02"}, {"hostname": "serval02"},
                {"host": "serval02", "hostname": "serval02"}):
            with self.subTest(fields=fields):
                self.assertEqual(monitor.process_hosts([{"unit": "qualification", **fields}]),
                    {"qualification": "serval02"})
        self.assertEqual(monitor.process_hosts([{"unit": "qualified_unit", "hostname": "node.example.edu"}]),
            {"qualified_unit": "node.example.edu"})
        self.assertEqual(monitor.process_hosts(None), {})
        self.assertEqual(monitor.process_hosts([]), {})

    def test_conflicting_or_missing_aliases_refuse(self):
        """A missing or contradictory host cannot acquire a guessed identity."""
        for fields in ({}, {"host": "serval02", "hostname": "serval01"},
                {"host": None, "hostname": "serval02"}, {"host": "", "hostname": "serval02"}):
            with self.subTest(fields=fields), self.assertRaises(ValueError):
                monitor.process_hosts([{"unit": "qualification", **fields}])

    def test_invalid_host_and_unit_values_refuse(self):
        """Malformed names, types and directory components remain errors."""
        for value in (None, True, 12, [], "", " serval02", "serval02 ", "serval02\n", "a/b", "a..b",
                "-serval02", "serval02-", "a" * 64, "a." * 127 + "b"):
            with self.subTest(host=value), self.assertRaises(ValueError):
                monitor.process_hosts([{"unit": "qualification", "hostname": value}])
        for value in (None, True, 12, "", ".", "..", "a/b", "a\x00b", "a" * 256):
            with self.subTest(unit=value), self.assertRaises(ValueError):
                monitor.process_hosts([{"unit": value, "host": "serval02"}])
        for value in ({}, "rows", [None], [12]):
            with self.subTest(records=value), self.assertRaises(ValueError):
                monitor.process_hosts(value)

    def test_duplicate_units_refuse_even_when_hosts_agree(self):
        """Two process rows cannot silently replace one another's identity."""
        for host in ("serval02", "serval01"):
            with self.subTest(host=host), self.assertRaises(ValueError):
                monitor.process_hosts([{"unit": "qualification", "host": "serval02"},
                    {"unit": "qualification", "hostname": host}])

    def test_actual_hostname_only_unit_renders_all_sections(self):
        """The unchanged actual row reaches the full report without a KeyError or scientific credit."""
        records = json.loads(FIXTURE.read_text())
        current = next(record for record in records if "hostname" in record and "host" not in record)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "processes.json").write_bytes(FIXTURE.read_bytes())
            unit = root / "data" / current["unit"]
            unit.mkdir(parents=True)
            (unit / "progress.json").write_text(json.dumps({"state": "initializing"}))
            rows = monitor.unit_rows(root)
            self.assertEqual(rows[0]["pool"], "serval02")
            with patch.object(monitor, "own_jobs", return_value=[]), contextlib.redirect_stdout(io.StringIO()):
                monitor.tick(root)
            report = (root / "20_mins_monitoring/latest_report.md").read_text()
            for name in ("TRAINING", "WAITING", "HELD", "FINISHED", "PLANNED"):
                self.assertIn(name + "\n", report)
            self.assertIn("serval02", report)
            self.assertEqual(json.loads((root / "20_mins_monitoring/latest_status.json").read_text())["campaign"].get("queue_counts", {}).get("completed", 0), 0)


if __name__ == "__main__":
    unittest.main()
