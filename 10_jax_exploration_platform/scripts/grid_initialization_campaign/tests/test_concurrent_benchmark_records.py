"""Exercise independent processes publishing the same compatibility JSON record."""

import json
import multiprocessing
from pathlib import Path
import stat
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmark import atomic_json


def concurrent_writer(path, identity, start):
    """Publish complete distinguishable records after all processes can overlap."""
    start.wait(timeout=30)
    for iteration in range(32):
        atomic_json(path, {'writer': identity, 'iteration': iteration,
                           'payload': [identity]*2048})


class ConcurrentBenchmarkRecordTests(unittest.TestCase):
    """Protect against the shared fixed-temporary-path failure seen in real jobs."""

    def test_concurrent_processes_publish_one_complete_record_without_collisions(self):
        """All writers must finish while the final file remains whole and group writable."""
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)/'compatibility.json'
            context = multiprocessing.get_context('spawn')
            start = context.Event()
            processes = [context.Process(target=concurrent_writer, args=(output, identity, start))
                         for identity in range(4)]
            try:
                for process in processes:
                    process.start()
                start.set()
                for process in processes:
                    process.join(timeout=120)
                self.assertEqual([process.exitcode for process in processes], [0]*4)
            finally:
                # Clean up only child processes created by this individual test.
                for process in processes:
                    if process.is_alive():
                        process.terminate()
                        process.join(timeout=10)
            record = json.loads(output.read_text())
            self.assertIn(record['writer'], range(4))
            self.assertEqual(record['iteration'], 31)
            self.assertEqual(record['payload'], [record['writer']]*2048)
            self.assertEqual(stat.S_IMODE(output.stat().st_mode), 0o664)
            self.assertEqual(list(Path(temporary).iterdir()), [output])

    def test_failed_serialization_preserves_prior_record_and_removes_own_temporary(self):
        """Non-finite values must fail without replacing good JSON or leaving partial files."""
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)/'compatibility.json'
            atomic_json(output, {'completed': True})
            with self.assertRaises(ValueError):
                atomic_json(output, {'invalid': float('inf')})
            self.assertEqual(json.loads(output.read_text()), {'completed': True})
            self.assertEqual(list(Path(temporary).iterdir()), [output])


if __name__ == '__main__':
    unittest.main()
