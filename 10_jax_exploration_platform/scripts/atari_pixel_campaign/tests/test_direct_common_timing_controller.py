"""Test common timing metadata gates using CPU fixtures and accepted saved records only."""
import ast
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS))
import direct_common_timing_controller as control
import prepare_direct_common_timing as preparation

RUN = Path('/p/rlprojects/RND_worktrees/merge-atari-cleanrl/10_jax_exploration_platform/benchmark_runs/2026-10-31-00-11_jaxatari_montezuma_pixels_cleanrl_ppo_rnd_environments-per-copy-128_rollout-steps-128_ideas-30-plus-20_steps-per-copy-2000000000_seed-start-1')
NAMES = ('require', 'now', 'encoded', 'digest', 'fsync_directory', 'ordinary_bytes', 'document', 'stable_stat',
    'receipt', 'log_receipt', 'publish', 'metadata_lock', 'no_numerical_modules', 'check_unattempted',
    'check_publication', 'load_core', 'read_process', 'owner_record', 'chosen_affinity', 'child_environment',
    'append_registration', 'wait_exact', 'supervise', 'actual_launch', 'main')


def saved_correctness_packet():
    """Locate the genuine completed second-copy correctness without any live process query."""
    return Path(json.loads((RUN / 'next_actions.json').read_bytes())['h100_common_correctness_preparation']['copies_2']['packet'])


class TimingControllerMetadataTests(unittest.TestCase):
    """Exercise changed admission gates without numerical imports, native calls, tools or GPUs."""
    def test_twenty_five_process_publication_functions_unchanged(self):
        """The proven process and publication mechanisms retain their exact abstract syntax."""
        original = ast.parse((saved_correctness_packet() / 'direct_qualification_controller.py').read_text())
        current = ast.parse(Path(control.__file__).read_text())
        functions = lambda tree: {x.name: ast.dump(x, include_attributes=False) for x in tree.body if isinstance(x, ast.FunctionDef)}
        old, new = functions(original), functions(current)
        self.assertEqual(len(NAMES), 25)
        for name in NAMES:
            with self.subTest(name=name):self.assertEqual(old[name], new[name])

    def fixture(self, directory):
        """Create explicitly synthetic helper byte identities, never an accepted snapshot or launch."""
        root, unit = Path(directory) / 'helpers', Path(directory) / 'unit'
        root.mkdir();(unit / 'code').mkdir(parents=True)
        names = ['synthetic_helper_' + str(index) + '.py' for index in range(29)]
        files = {}
        for name in names:
            relative = '10_jax_exploration_platform/scripts/atari_pixel_campaign/' + name
            path = root / relative;path.parent.mkdir(parents=True, exist_ok=True)
            raw = ('# explicitly synthetic metadata fixture ' + name + '\n').encode()
            path.write_bytes(raw);(unit / 'code' / name).write_bytes(raw);files[relative] = hashlib.sha256(raw).hexdigest()
        refs = {key: {'path': '/synthetic/' + key, 'sha256': str(index + 1) * 64}
            for index, key in enumerate(('plan', 'result', 'review'))}
        selected = SimpleNamespace(qual=refs, correctness=refs,
            proof={'accepted': {'correctness': {'1': refs, '2': refs}}})
        support = SimpleNamespace(required_helper_names=lambda: names, consume=lambda plan: selected)
        inputs = {'proof': refs['plan'], 'review': refs['review']}
        plan = {'mode': 'confirmation', 'rounds': 7, 'iterations': 10, 'minimum_updates_per_round': 10,
            'helpers': {'source_commit': 'a' * 40, 'files': files}, 'configuration': {'learner': {'n_copies': 1}}, 'copy_seeds': [1],
            'accepted_evidence_snapshot_inputs': inputs, 'common_correctness_inputs': {x: refs[x] for x in ('result', 'review')}}
        request = {'helper_source_root': str(root), 'helper_source_commit': 'a' * 40, 'configuration': {'learner': {'n_copies': 1}}, 'copy_seeds': [1], 'serial_order': 1,
            'accepted_evidence_snapshot_inputs': inputs, 'common_correctness_inputs': plan['common_correctness_inputs'],
            'accepted_correctness': {'1': copy.deepcopy(refs), '2': copy.deepcopy(refs)}, 'qualification_plan': refs['plan'],
            'direct_qualification_inputs': {x: refs[x] for x in ('result', 'review')}}
        return root, unit, plan, request, support

    def test_exact_new_namespace_and_record_references(self):
        """The changed gate joins every staged helper to the actual declared Git and checkout bytes."""
        with tempfile.TemporaryDirectory() as temporary:
            root, unit, plan, request, support = self.fixture(temporary)
            def git(command, **kwargs):
                """Return only the exact synthetic blob requested; never execute a command."""
                self.assertEqual(command[:4], ['git', '-C', str(root), 'show'])
                self.assertTrue(command[4].startswith('a' * 40 + ':'))
                return (root / command[4].split(':', 1)[1]).read_bytes()
            with patch.object(control, 'UNIT', unit), patch.object(control.subprocess, 'check_output', side_effect=git), patch.object(control, 'timing_support', return_value=support):
                self.assertIs(control.timing_snapshot(plan, request), support.consume(plan))
                changed = copy.deepcopy(request);changed['accepted_correctness']['2']['result']['sha256'] = '0' * 64
                with self.assertRaisesRegex(ValueError, 'genuinely accepted'):control.timing_snapshot(plan, changed)
                altered = next((unit / 'code').iterdir());altered.write_text('# altered staged source\n')
                with self.assertRaisesRegex(ValueError, 'Git, checkout and staged'):control.timing_snapshot(plan, request)

    def test_old_correctness_helper_count_protocol_and_missing_proofs_refused(self):
        """Old operational modes or an incomplete twenty-nine-source/proof declaration refuse before any Git call."""
        with tempfile.TemporaryDirectory() as temporary:
            root, unit, plan, request, support = self.fixture(temporary)
            variants = []
            for key, value in [('mode', 'correctness'), ('rounds', 11), ('iterations', 20),
                    ('minimum_updates_per_round', True), ('native_process_control', {}), ('common_correctness_inputs', {})]:
                altered = copy.deepcopy(plan);altered[key] = value;variants.append(altered)
            altered = copy.deepcopy(plan);altered['helpers']['files'].pop(next(iter(altered['helpers']['files'])));variants.append(altered)
            altered = copy.deepcopy(plan);altered['accepted_evidence_snapshot_inputs'] = {};variants.append(altered)
            with patch.object(control.subprocess, 'check_output', side_effect=AssertionError('no command before declared scope is valid')):
                for altered in variants:
                    with self.subTest(mode=altered.get('mode'), count=len(altered['helpers']['files'])), self.assertRaises(ValueError):
                        control.timing_snapshot(altered, request)

    def predecessor(self, packet):
        """Build a labelled reader input from genuine accepted second-copy process records only."""
        prior_packet = saved_correctness_packet();prior = RUN / 'data' / prior_packet.name
        refs = {name: control.receipt(path) for name, path in {
            'plan': prior / 'measurement_plan.json', 'result': prior / 'result.json',
            'terminal': prior / 'direct_terminal_process.json', 'wait': prior / 'direct_process_wait_return.json',
            'outcome': prior / 'direct_controller_outcome.json', 'metadata': prior / 'direct_process_metadata.json',
            'controller_spawn': prior_packet / 'root_actual_controller_spawn.json',
            'exit_observation': prior_packet / 'root_exact_exit_observation_2026-11-05-02-24.json'}.items()}
        data = {'schema_version': 1, 'kind': 'actual_owned_direct_serial_predecessor_exit_inputs', 'predecessor_unit': str(prior), **refs}
        path = packet / 'serial_predecessor_exit_inputs.json';path.write_bytes(control.encoded(data))
        plan = control.document(refs['plan']['path'])
        request = {'serial_predecessor_unit': str(prior), 'serial_order': 1,
            'numerical_source_root': plan['source_root'], 'helper_source_root': plan['helpers']['source_root'],
            'helper_source_commit': plan['helpers']['source_commit'], 'python': plan['python'],
            'runtime_environment': plan['runtime_environment'], 'asset_record': plan['asset_record'],
            'accepted_evidence_snapshot_inputs': {'proof': {'path': '/synthetic/reader-only-snapshot-not-admitted'}},
            'configuration': {'learner': {'n_copies': 1}}, 'copy_seeds': [1]}
        return path, request, refs

    def test_actual_saved_correctness_wait_and_both_exits_reader(self):
        """Timing order one accepts only the existing normal second-copy wait and exact saved exits."""
        with tempfile.TemporaryDirectory() as temporary:
            packet = Path(temporary);path, request, refs = self.predecessor(packet);real = control.document
            def read(name):
                """Supply only a labelled container around actual accepted result/plan refs, not an admission proof."""
                if str(name) == '/synthetic/reader-only-snapshot-not-admitted':
                    return {'accepted': {'correctness': {'2': {key: refs[key] for key in ('plan', 'result')}}}}
                return real(name)
            with patch.object(control, 'PACKET', packet), patch.object(control, 'RUN', RUN), patch.object(control, 'document', side_effect=read):
                self.assertEqual(control.check_serial_predecessor_inputs(request, control.receipt(path))['wait'], refs['wait'])
                with self.assertRaises(ValueError):control.check_serial_predecessor_inputs(request, {'path': str(path), 'sha256': '0' * 64})
                bad = copy.deepcopy(real(refs['wait']['path']));bad['returncode'] = 1
                def failed(name):return bad if str(name) == refs['wait']['path'] else read(name)
                with patch.object(control, 'document', side_effect=failed), self.assertRaisesRegex(ValueError, 'uninterrupted successful'):
                    control.check_serial_predecessor_inputs(request, control.receipt(path))
                second = copy.deepcopy(request);second['serial_order'] = 2
                with self.assertRaisesRegex(ValueError, 'serial position'):
                    control.check_serial_predecessor_inputs(second, control.receipt(path))

    def test_binding_requires_actual_cold_proof_and_no_future_record(self):
        """Absent actual proof refuses; no unit, source review or successful metadata is fabricated."""
        with self.assertRaises((ValueError, FileNotFoundError)):preparation.checked_reference({'path': '/not-an-actual-proof', 'sha256': '0' * 64})

    def test_only_five_admission_functions_changed_and_staged_mapping_is_explicit(self):
        """The canonical timing source keeps legacy staging names only for unchanged publication checks."""
        source = Path(preparation.__file__).read_text()
        self.assertIn("SCRIPTS / 'direct_common_timing_controller.py', packet / 'direct_qualification_controller.py'", source)
        self.assertIn("'legacy_names_admit_no_correctness_operations': True", source)
        self.assertIn("direct_common_timing_preparation_request", Path(control.__file__).read_text())
        self.assertNotIn('7d07da121c41c531d51cbd25b73579735212ead8', source)


class TimingCopyOrderCorrectionTests(unittest.TestCase):
    """Cover only the independent serial-order finding and canonical preparer staging correction."""
    def test_exact_one_and_two_copy_orders(self):
        """Each current copy count has its own strict matching order and complete independent seeds."""
        for count in (1, 2):
            request = {'configuration': {'learner': {'n_copies': count}}, 'serial_order': count,
                'copy_seeds': list(range(1, count + 1))}
            plan = {'configuration': copy.deepcopy(request['configuration']), 'copy_seeds': request['copy_seeds']}
            self.assertEqual(control.timing_copy_order(request, plan), count)

    def test_crossed_boolean_and_missing_seed_orders_refused_before_reads(self):
        """Two/order-one, one/order-two and Boolean or partial identities cannot reach prior records."""
        for count, order, seeds in [(2, 1, [1, 2]), (1, 2, [1]), (True, 1, [1]), (1, True, [1]),
                (2, 2, [1]), (1, 1, [True])]:
            request = {'configuration': {'learner': {'n_copies': count}}, 'serial_order': order, 'copy_seeds': seeds}
            with self.subTest(count=count, order=order, seeds=seeds), patch.object(control, 'receipt', side_effect=AssertionError('no prior read')):
                with self.assertRaisesRegex(ValueError, 'serial position'):
                    control.check_serial_predecessor_inputs(request, {'path': '/not-read', 'sha256': '0' * 64})

    def test_actual_plan_copy_count_cannot_differ_from_request(self):
        """A one-copy declaration cannot admit a current two-copy timing plan."""
        request = {'configuration': {'learner': {'n_copies': 1}}, 'serial_order': 1, 'copy_seeds': [1]}
        for count in (2, True):
            plan = {'configuration': {'learner': {'n_copies': count}}, 'copy_seeds': [1]}
            with self.subTest(count=count), self.assertRaisesRegex(ValueError, 'current timing plan'):
                control.timing_copy_order(request, plan)

    def test_proper_one_copy_request_uses_actual_saved_second_correctness(self):
        """The corrected genuine-predecessor control explicitly declares current count one."""
        TimingControllerMetadataTests('test_actual_saved_correctness_wait_and_both_exits_reader').test_actual_saved_correctness_wait_and_both_exits_reader()

    def test_canonical_preparer_import_and_archival_only_mapping(self):
        """The canonical source imports on CPU; every advertised phase uses it, never the archival packet copy."""
        canonical = Path(preparation.__file__).resolve(strict=True)
        result = subprocess.run([sys.executable, '-I', '-S', '-B', str(canonical), '--help'],
            capture_output=True, timeout=15, check=False)
        self.assertEqual(result.returncode, 0, result.stderr.decode())
        self.assertIn(b'prepare,verify,bind', result.stdout)
        tree = ast.parse(canonical.read_text())
        function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'prepare')
        body = ast.get_source_segment(canonical.read_text(), function)
        self.assertIn('canonical = Path(__file__).resolve(strict=True)', body)
        self.assertIn("'copied_preparer_is_archival_only': True", body)
        self.assertIn("'all_metadata_phase_commands': phases", body)
        self.assertIn('str(canonical)', body)


if __name__ == '__main__':
    unittest.main()
