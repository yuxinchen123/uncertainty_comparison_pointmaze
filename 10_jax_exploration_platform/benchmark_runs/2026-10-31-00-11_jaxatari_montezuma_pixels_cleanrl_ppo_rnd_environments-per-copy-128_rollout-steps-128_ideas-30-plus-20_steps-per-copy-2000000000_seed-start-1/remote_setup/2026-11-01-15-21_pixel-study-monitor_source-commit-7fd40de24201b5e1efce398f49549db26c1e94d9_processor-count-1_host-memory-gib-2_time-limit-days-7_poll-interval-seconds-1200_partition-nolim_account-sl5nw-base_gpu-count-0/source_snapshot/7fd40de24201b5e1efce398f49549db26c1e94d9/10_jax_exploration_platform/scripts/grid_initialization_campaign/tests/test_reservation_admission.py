"""Keep reservation admission independent while preserving live ordinary-pool accounting."""

import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import submit


def capacity(cards=0, cpus=16, memory=0):
    """Represent the owner's real ordinary-pool free counters without reservation addback."""
    return {'uva-sl5nw': {'reachable': True, 'pools': [{'partition': 'gpu',
        'gpu_free': cards, 'cpu_free': cpus, 'memory_gib_free': memory}]}}


class ReservationAdmissionTests(unittest.TestCase):
    """Exercise full quotas, physical headroom, historical records, and duplicate prevention."""

    def plan_reserved(self, root, *, cpu_count=64, base_capacity=None, active=None):
        """Run the real planner against one reserved node with external calls mocked."""
        for state in ('pending', 'running'):
            (root/'queue'/state).mkdir(parents=True, exist_ok=True)
        (root/'slurm/submissions').mkdir(parents=True, exist_ok=True)
        for index in range(8):
            (root/'queue/pending'/f'{index}.json').write_text(json.dumps({'stage': 'research'}))
        catalog = root/'catalog.json'
        catalog.write_text(json.dumps({'classes': [{'nodes': ['jaguar03'], 'partition': 'gpu',
            'gres_type': 'a4500', 'gpu_mem_mb': 20000, 'gpus_per_node': 8, 'capability_rank': 6}]}))
        node = (f'NodeName=jaguar03 CPUTot={cpu_count} CPUEfctv={cpu_count} CPUAlloc=0 '
                'RealMemory=262144 AllocMem=0 MemSpecLimit=4000 State=IDLE+RESERVED '
                'Gres=gpu:a4500:8 AllocTRES= ReservationName=owner_reservation')
        supplied = base_capacity if base_capacity is not None else capacity()
        def output(command, **kwargs):
            """Supply only account counters and physical node facts."""
            return json.dumps(supplied) if '--json' in command else node
        with patch.object(submit, 'RESOURCE_FILE', catalog), \
                patch.object(submit, 'own_live_jobs', return_value=active or {}), \
                patch.object(submit, 'owner_reservation', return_value='owner_reservation'), \
                patch.object(submit, 'compatibility_classes', return_value=({'a4500'}, set())), \
                patch.object(submit.subprocess, 'check_output', side_effect=output), \
                patch.object(submit, 'submit_job', return_value='101') as send:
            result = submit.top_up(root, 'research')
        return result, send

    def test_full_ordinary_pool_still_admits_all_eight_reserved_cards(self):
        """A full base pool must not reduce an eight-card physical reservation to zero."""
        with tempfile.TemporaryDirectory() as temporary:
            result, send = self.plan_reserved(Path(temporary))
        self.assertEqual(result, ['101'])
        self.assertEqual(send.call_args.kwargs['cards'], 8)
        self.assertEqual(send.call_args.kwargs['reservation'], 'owner_reservation')
        self.assertEqual(send.call_args.args[1], 'uva-sl5nw')

    def test_reserved_admission_preserves_sixteen_free_cpus(self):
        """Physical CPU headroom still limits admission and produces one node job."""
        with tempfile.TemporaryDirectory() as temporary:
            result, send = self.plan_reserved(Path(temporary), cpu_count=44)
        self.assertEqual(result, ['101'])
        self.assertEqual(send.call_count, 1)
        self.assertEqual(send.call_args.kwargs['cards'], 7)

    def test_submitting_reserved_work_does_not_mutate_the_current_ordinary_budget(self):
        """Keep current budgets separate; the next scheduler reading includes new usage."""
        actual = submit.capacity_budgets
        observed = {}
        def capture(*arguments):
            """Retain the real ordinary-budget object for inspection after admission."""
            observed['budget'] = actual(*arguments)
            return observed['budget']
        with tempfile.TemporaryDirectory() as temporary:
            with patch.object(submit, 'capacity_budgets', side_effect=capture):
                result, send = self.plan_reserved(Path(temporary),
                    base_capacity=capacity(4, 80, 100))
        self.assertEqual(send.call_args.kwargs['cards'], 8)
        self.assertEqual(observed['budget'][('uva-sl5nw', 'gpu')],
                         {'cards': 4, 'cpus': 64, 'memory_gib': 100})

    def test_existing_partial_reserved_job_prevents_a_duplicate_node_job(self):
        """The controller must wait for explicit replacement of an existing one-card job."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root/'slurm/submissions').mkdir(parents=True)
            (root/'slurm/submissions/91.json').write_text(json.dumps({'job_id': '91',
                'account': 'sl5nw', 'partition': 'gpu', 'stage': 'research', 'node': 'jaguar03',
                'gpu_type': 'a4500', 'cards': 1, 'cpus': 4, 'memory_gib': 8,
                'reservation': 'owner_reservation'}))
            result, send = self.plan_reserved(root,
                active={'91': ['91', 'sl5nw', 'gpu', 'RUNNING', '', 'jaguar03', '']})
        self.assertEqual(result, [])
        send.assert_not_called()

    def test_running_reservation_is_not_added_back_to_ordinary_free_counters(self):
        """The next live capacity reading continues to count actual reserved usage."""
        reserved = {'job_id': '91', 'account': 'sl5nw', 'partition': 'gpu', 'cards': 8,
                    'cpus': 32, 'memory_gib': 64, 'reservation': 'owner_reservation'}
        budgets = submit.capacity_budgets(capacity(), [reserved], {'91': ['', '', '', 'RUNNING']})
        self.assertEqual(budgets[('uva-sl5nw', 'gpu')],
                         {'cards': 0, 'cpus': 0, 'memory_gib': 0})

    def test_pending_reservation_is_not_debited_from_ordinary_budget(self):
        """Pending reserved requests use their physical reservation budget exactly once."""
        record = {'job_id': '91', 'account': 'sl5nw', 'partition': 'gpu', 'cards': 8,
                  'cpus': 32, 'memory_gib': 64, 'node': 'jaguar03', 'reservation': 'owner_reservation'}
        live = {'91': ['', '', '', 'PENDING']}
        ordinary = submit.capacity_budgets(capacity(4, 80, 100), [record], live)
        self.assertEqual(ordinary[('uva-sl5nw', 'gpu')],
                         {'cards': 4, 'cpus': 64, 'memory_gib': 100})
        reserved = submit.reservation_budget('jaguar03',
            {'cards': 8, 'cpus': 64, 'memory_gib': 128}, [record], live, 'owner_reservation')
        self.assertEqual(reserved, {'cards': 0, 'cpus': 16, 'memory_gib': 64})

    def test_legacy_live_reservation_is_recovered_without_rewriting_record(self):
        """Use actual scheduler reservation/QoS fields for old active job records."""
        original = {'job_id': '91', 'account': 'sl5nw'}
        with patch.object(submit.subprocess, 'check_output',
                          return_value='JobId=91 QOS=csresnolim Reservation=owner_reservation'):
            enriched = submit.classify_submission_reservations([original], {'91': []})
        self.assertEqual(enriched[0]['reservation'], 'owner_reservation')
        self.assertEqual(enriched[0]['qos'], 'csresnolim')
        self.assertNotIn('reservation', original)


if __name__ == '__main__':
    unittest.main()
