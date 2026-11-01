"""Check account resource admission and exact study cancellation with scheduler calls mocked."""

import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import monitor
import submit


def capacity_record(gpu_cards=2, gpu_cpus=24, gpu_memory=64, gnolim_cards=0):
    """Build a small live account reading without calling the scheduler."""
    return {'reachable': True, 'pools': [
        {'partition': 'gpu', 'gpu_free': gpu_cards, 'cpu_free': gpu_cpus,
         'memory_gib_free': gpu_memory},
        {'partition': 'gnolim', 'gpu_free': gnolim_cards, 'cpu_free': 4*gnolim_cards,
         'memory_gib_free': 16*gnolim_cards}]}


class SubmissionLimitTests(unittest.TestCase):
    """Verify every requested resource remains within its independent bound."""

    def test_owner_headroom_and_pending_requests_are_reserved(self):
        """Count existing pending jobs once in addition to account-wide running usage."""
        capacity = {'uva-sl5nw': capacity_record(), 'uva-yuxinchen': capacity_record()}
        pending = {'job_id': '91', 'account': 'sl5nw', 'partition': 'gpu', 'cards': 1,
                   'cpus': 4, 'memory_gib': 16}
        budgets = submit.capacity_budgets(capacity, [pending], {'91': ['91', '', '', 'PENDING']})
        self.assertEqual(budgets[('uva-sl5nw', 'gpu')], {'cards': 1, 'cpus': 4, 'memory_gib': 48})
        self.assertEqual(budgets[('uva-yuxinchen', 'gpu')]['cpus'], 24)
        capacity['uva-yuxinchen']['reachable'] = False
        self.assertNotIn(('uva-yuxinchen', 'gpu'), submit.capacity_budgets(capacity, [], {}))

    def test_node_resources_count_cards_and_exclude_down_nodes(self):
        """Read configured minus allocated resources and refuse unhealthy nodes."""
        row = ('NodeName=example CPUTot=16 CPUEfctv=14 CPUAlloc=6 RealMemory=65536 '
               'AllocMem=16384 State=MIXED Gres=gpu:nvidia_a40:4(S:0) AllocTRES=cpu=6,gres/gpu=2')
        nodes = submit.node_resources(row+'\n'+row.replace('example', 'broken').replace('MIXED', 'DOWN'))
        self.assertEqual(nodes['example']['cards'], 2)
        self.assertEqual(nodes['example']['cpus'], 8)
        self.assertEqual(nodes['example']['memory_gib'], 48)
        self.assertNotIn('broken', nodes)

    def test_top_up_uses_both_partitions_and_enforces_cpu_memory_card_caps(self):
        """Exercise the real planner while replacing all external scheduler operations."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for state in ('pending', 'running'):
                (root/'queue'/state).mkdir(parents=True)
            (root/'slurm/submissions').mkdir(parents=True)
            for index in range(10):
                (root/'queue/pending'/f'{index}.json').write_text(json.dumps({'stage': 'research'}))
            classes = [{'nodes': ['node1', 'node2'], 'partition': 'gpu', 'gres_type': 'type1',
                        'gpu_mem_mb': 20000, 'gpus_per_node': 4, 'capability_rank': 1},
                       {'nodes': ['node3'], 'partition': 'gnolim', 'gres_type': 'type2',
                        'gpu_mem_mb': 12000, 'gpus_per_node': 4, 'capability_rank': 2}]
            catalog = root/'catalog.json'
            catalog.write_text(json.dumps({'classes': classes}))
            capacity = {'uva-sl5nw': capacity_record(1, 20, 16),
                        'uva-yuxinchen': capacity_record(2, 8, 32, 1)}
            node_rows = '\n'.join(f'NodeName=node{i} CPUTot=32 CPUAlloc=0 RealMemory=131072 '
                'AllocMem=0 State=IDLE Gres=gpu:type:4 AllocTRES=' for i in (1, 2, 3))
            submitted = []
            def fake_check_output(command, **kwargs):
                """Return only the two scheduler readings expected by this test."""
                return json.dumps(capacity) if '--json' in command else node_rows
            def fake_submit(run_dir, identity, stage, **kwargs):
                """Capture the concrete requested resource shape without submitting it."""
                submitted.append((identity, kwargs))
                return str(100+len(submitted))
            with patch.object(submit, 'RESOURCE_FILE', catalog), patch.object(submit, 'own_live_jobs', return_value={}), \
                    patch.object(submit, 'owner_reservation', return_value=None), \
                    patch.object(submit, 'compatibility_classes', return_value=({'type1', 'type2'}, set())), \
                    patch.object(submit.subprocess, 'check_output', side_effect=fake_check_output), \
                    patch.object(submit, 'submit_job', side_effect=fake_submit):
                identifiers = submit.top_up(root, 'research')
            self.assertEqual(len(identifiers), 3)
            self.assertEqual(sum(options['cards'] for _, options in submitted), 4)
            self.assertEqual({options['partition'] for _, options in submitted}, {'gpu', 'gnolim'})
            self.assertEqual(len({options['node'] for _, options in submitted}), len(submitted))
            self.assertTrue(all(not options['canary'] for _, options in submitted))

    def test_deadline_cancels_only_recorded_training_ids(self):
        """Exclude the monitor and unrelated live identifiers from exact cancellation."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root/'slurm/submissions').mkdir(parents=True)
            (root/'slurm/submitted_jobids_initial_bonus_sl5nw.txt').write_text('91\n92\n')
            for identifier, stage in [('91', 'research'), ('92', 'monitor')]:
                (root/'slurm/submissions'/f'{identifier}.json').write_text(json.dumps(
                    {'job_id': identifier, 'account': 'sl5nw', 'stage': stage}))
            with patch.object(monitor.subprocess, 'run') as run:
                cancelled = monitor.stop_workers(root, {'91': [], '92': [], '999': []})
            self.assertEqual(cancelled, ['91'])
            self.assertEqual(run.call_args.args[0][-1], '91')
            self.assertNotIn('--all', run.call_args.args[0])
            self.assertTrue((root/'STOP').exists())


if __name__ == '__main__':
    unittest.main()
