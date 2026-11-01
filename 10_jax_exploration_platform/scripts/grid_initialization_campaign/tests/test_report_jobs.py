"""Check asynchronous report jobs without starting numerical work or contacting Slurm."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import monitor
import run_report
import submit


def write_record(path, record):
    """Create a small test record using only the standard library."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record))


class ReportJobTests(unittest.TestCase):
    """Keep slow rendering outside the monitor and preserve honest completion counts."""

    def test_render_records_start_count_only_after_success(self):
        """Results arriving during rendering must remain due for a subsequent report."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_record(root/'data/research/first/summary.json', {'completed': True})
            def build(*args, **kwargs):
                """Simulate a result arriving while the separate renderer runs."""
                self.assertFalse((root/'last_rendered_result_count.json').exists())
                write_record(root/'data/research/second/summary.json', {'completed': True})
            with patch.dict(run_report.os.environ, {'SLURM_JOB_ID': '91'}), \
                    patch.object(run_report.socket, 'gethostname', return_value='compute'), \
                    patch.object(run_report.subprocess, 'run', side_effect=build):
                self.assertEqual(run_report.render_report(root, final=True), 1)
            marker = json.loads((root/'last_rendered_result_count.json').read_text())
            self.assertEqual(marker['count'], 1)
            self.assertTrue(marker['final'])
            self.assertEqual(run_report.completed_result_count(root), 2)

    def test_failed_renderer_does_not_advance_marker(self):
        """A failed plotting process must be eligible for a later retry."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_record(root/'last_rendered_result_count.json', {'count': 0})
            with patch.dict(run_report.os.environ, {'SLURM_JOB_ID': '91'}), \
                    patch.object(run_report.socket, 'gethostname', return_value='compute'), \
                    patch.object(run_report.subprocess, 'run', side_effect=subprocess.CalledProcessError(1, ['build'])):
                with self.assertRaises(subprocess.CalledProcessError):
                    run_report.render_report(root)
            self.assertEqual(json.loads((root/'last_rendered_result_count.json').read_text()), {'count': 0})

    def test_monitor_coalesces_one_active_report(self):
        """One live exact-id report blocks another even when more results have arrived."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root/'queue').mkdir()
            write_record(root/'slurm/submissions/91.json', {'stage': 'report', 'job_id': '91'})
            with patch.object(monitor, 'own_live_jobs', return_value={'91': []}), \
                    patch.object(monitor, 'submit_report_job') as send:
                result = monitor.ensure_report_job(root, final=True)
            self.assertEqual(result['job_id'], '91')
            self.assertFalse(result['completed'])
            send.assert_not_called()

    def test_final_report_is_submitted_even_if_interim_count_is_current(self):
        """Terminal monitoring waits for an explicitly successful final report."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root/'queue').mkdir()
            write_record(root/'last_rendered_result_count.json', {'count': 0, 'final': False})
            with patch.object(monitor, 'own_live_jobs', return_value={}), \
                    patch.object(monitor, 'submit_report_job', return_value='92') as send:
                result = monitor.ensure_report_job(root, final=True)
            self.assertEqual(result['job_id'], '92')
            send.assert_called_once_with(root, final=True)
            write_record(root/'last_rendered_result_count.json', {'count': 0, 'final': True})
            with patch.object(monitor, 'own_live_jobs', return_value={}), \
                    patch.object(monitor, 'submit_report_job') as send:
                self.assertTrue(monitor.ensure_report_job(root, final=True)['completed'])
                send.assert_not_called()

    def test_report_submission_uses_owner_cpu_pool(self):
        """A plotting job cannot take a GPU partition or the collaborator's capacity."""
        capacity = {'uva-sl5nw': {'reachable': True, 'pools': [
            {'partition': 'cpu', 'cpu_free': 20, 'memory_gib_free': 32}]}}
        with patch.object(submit.subprocess, 'check_output', return_value=json.dumps(capacity)), \
                patch.object(submit, 'submit_job', return_value='93') as send:
            self.assertEqual(submit.submit_report_job(Path('/example'), final=True), '93')
        send.assert_called_once_with(Path('/example'), 'uva-sl5nw', 'report',
                                     partition='cpu', report=True, final_report=True)

    def test_bounded_tail_ignores_incomplete_last_line(self):
        """A concurrent append must not make a complete prior progress record unreadable."""
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/'progress.jsonl'
            path.write_bytes(b'{"iteration": 4}\n{"iteration":')
            self.assertEqual(monitor.last_progress_record(path), {'iteration': 4})

    def test_status_preserves_readable_card_name(self):
        """Render the complete catalog card name instead of a scheduler prefix."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_record(root/'slurm/submissions/91.json', {'job_id': '91', 'stage': 'research',
                'gpu_type': 'nvidia_geforce_gtx_1080_ti', 'cards': 1})
            catalog = root/'catalog.json'
            write_record(catalog, {'classes': [{'gres_type': 'nvidia_geforce_gtx_1080_ti',
                                               'display_name': 'GTX 1080 Ti'}]})
            with patch.object(monitor, 'RESOURCE_FILE', catalog):
                lines = monitor.status_lines(root, {'phase': 'calibration'},
                    {'91': ['91', 'sl5nw', 'gpu', 'RUNNING', '', '', '']},
                    {'done': 1, 'failed': 0, 'running': 1, 'pending': 0})
            self.assertIn('1 GTX 1080 Ti', '\n'.join(lines))
            self.assertTrue(all(len(line) <= 100 for line in lines))

    def test_monitor_sleep_subtracts_tick_duration(self):
        """A slow control tick must not extend the twenty-minute interval further."""
        with patch.object(sys, 'argv', ['monitor.py', '--run-dir', '/example', '--loop']), \
                patch.object(monitor, 'tick', return_value=({'phase': 'calibration'}, {})), \
                patch.object(monitor.time, 'monotonic', side_effect=[100, 350]), \
                patch.object(monitor.time, 'sleep', side_effect=InterruptedError) as sleep:
            with self.assertRaises(InterruptedError):
                monitor.main()
        sleep.assert_called_once_with(950)


if __name__ == '__main__':
    unittest.main()
