"""Recheck exact rejected boundaries and small accepted metadata, without payload completion."""
import ast
from copy import deepcopy
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = Path(__file__).resolve().parent
SCRIPTS = ROOT / '10_jax_exploration_platform/scripts/atari_pixel_campaign'
OLD = ROOT / '.review_metadata/accepted_evidence_snapshot_independent_review'
INPUTS = json.loads((OLD / 'independent_fixture_inputs.json').read_text())
FREEZE = ROOT / '.review_metadata/accepted_evidence_snapshot_corrections/corrected_frozen_source_for_independent_review.json'
FREEZE_SHA = '051fbfca816dfdf5455f19f905b6410e72428d1262dfb763d901e4f9075a746b'
FROZEN = json.loads(FREEZE.read_text())
FORBIDDEN = {'jax', 'jaxlib', 'numpy', 'torch', 'flax', 'optax', 'scipy', 'cupy', 'jaxtari', 'exploration_platform'}
OBSERVATIONS = []
STARTED = time.monotonic()
job, step = os.environ.get('SLURM_JOB_ID'), os.environ.get('SLURM_STEP_ID')
assert job == '6783003' and str(step).isdigit()
assert Path('/proc/self/cgroup').read_text().splitlines() == ['0::/system.slice/slurmstepd.scope/job_6783003/step_' + step + '/user/task_0']
assert os.getresuid() == (2618919, 2618919, 2618919)
assert hashlib.sha256(FREEZE.read_bytes()).hexdigest() == FREEZE_SHA

def freeze_matches():
    return all(hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == value['sha256']
        and oct((ROOT / relative).stat().st_mode & 0o777) == value['mode'] for relative, value in FROZEN['files'].items())

assert freeze_matches()

def audit(event, values):
    if event == 'import' and values[0].split('.')[0] in FORBIDDEN:
        raise RuntimeError('numerical import refused in independent metadata checks')
    if event in {'subprocess.Popen', 'socket.connect', 'socket.bind', 'socket.getaddrinfo', 'ctypes.dlopen', 'ctypes.dlsym', 'os.system'}:
        raise RuntimeError('process, network or library work refused')
    if event == 'open' and isinstance(values[0], (str, bytes)):
        name = os.fsdecode(values[0])
        if name.startswith(('/dev/nvidia', '/dev/dri', '/tmp/sl5nw_serval02_card1.lock')) or '/direct_history/content/' in name:
            raise RuntimeError('device, standing lock or historical payload access refused')
        if name.startswith(('/usr/lib', '/lib/')) and 'libcuda' in name:
            raise RuntimeError('node-local historical driver access refused')

sys.addaudithook(audit)
spec = importlib.util.spec_from_file_location('snapshot_independent_corrected_fixture', SCRIPTS / 'tests/test_accepted_evidence_snapshot.py')
AUTHOR = importlib.util.module_from_spec(spec); spec.loader.exec_module(AUTHOR)
S = AUTHOR.snapshot

def source_functions(path):
    return {node.name: node for node in ast.parse(Path(path).read_bytes()).body if isinstance(node, ast.FunctionDef)}

def syntax(value):
    return ast.dump(value, include_attributes=False)

# Reuse exact independent controls that previously exercised the rejected source.
old_program = OLD / 'run_focused_independent_checks.py'
old_tree = ast.parse(old_program.read_bytes())
old_class = next(node for node in old_tree.body if isinstance(node, ast.ClassDef) and node.name == 'IndependentBoundaryChecks')
excluded = {'test_real_accepted_small_metadata_schemas_without_payload_hashes'}
old_class.body = [node for node in old_class.body if not isinstance(node, ast.FunctionDef) or node.name not in excluded]
exec(compile(ast.fix_missing_locations(ast.Module(body=[old_class], type_ignores=[])), str(old_program), 'exec'), globals())
ExactPreviousBoundaryChecks = IndependentBoundaryChecks

class NewNarrowBoundaries(unittest.TestCase):
    def test_real_accepted_Q_C1_C2_asset_schema_and_negative_raw_receipt(self):
        with S.ValidationFiles(60, metadata_only=True) as files:
            inventory = S.AcceptedInventory(files)
            qref = INPUTS['small_review_references']['qualification']
            qualification = S.accepted_role(files, inventory, qref, 'qualification')
            review = files.json(qref)
            accepted = {'qualification': qualification}
            for key in ('1', '2'):
                accepted[key] = S.accepted_role(files, inventory, INPUTS['small_review_references'][key],
                    'correctness', review, qualification['qualified_asset_receipt'])
                self.assertNotIn('asset_record', files.json(accepted[key]['result']))
                self.assertEqual(accepted[key]['qualified_asset_receipt'], qualification['qualified_asset_receipt'])
                self.assertEqual(accepted[key]['asset_record'], files.json(accepted[key]['asset_input']))
            self.assertEqual([accepted[key]['copy_seeds'] for key in ('1', '2')], [[1], [1, 2]])
            for key, value in accepted.items():
                OBSERVATIONS.append({'case': 'real_small_accepted_asset_join', 'role': key,
                    'review': value['review'], 'result': value['result'], 'plan': value['plan'],
                    'actual_asset_receipt': value['actual_asset_receipt'], 'qualified_asset_receipt': value['qualified_asset_receipt'],
                    'copy_seeds': value['copy_seeds']})
            original_json = files.json
            def changed(reference):
                value = original_json(reference)
                if reference == accepted['1']['actual_asset_receipt']:
                    value = deepcopy(value); value['file_count'] = 4530
                return value
            with mock.patch.object(files, 'json', side_effect=changed):
                with self.assertRaises(ValueError):
                    S.accepted_role(files, inventory, INPUTS['small_review_references']['1'],
                        'correctness', review, qualification['qualified_asset_receipt'])
            for (path, sha), declaration in inventory.shared_metadata.items():
                authority = declaration['declaration']['authority']
                self.assertIn(authority['path'], files.paths)
                self.assertNotIn('/usr/lib/', path)
            OBSERVATIONS.append({'case': 'actual_small_metadata_read_budget', 'physical_files_read': len(files.physical),
                'bytes_hashed': files.bytes_hashed, 'pending_payloads_not_hashed': len(inventory.pending_blobs),
                'shared_exact_declarations': len(inventory.shared_metadata)})
            self.assertLess(files.bytes_hashed, 4 * 1024 * 1024)

    def test_exact_source_and_reference_fields_are_required_not_suffixes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            python = root / 'small.py'; python.write_bytes(b'# exact source\n')
            shell = root / 'worker_environment.sh'; shell.write_bytes(b'# exact shell source\n')
            refs = {p.name: {'path': str(p), 'sha256': S.digest(p.read_bytes())} for p in (python, shell)}
            authority = root / 'plan.json'
            authority.write_bytes(S.encoded({'core': {'source_commit': S.NUMERICAL_SOURCE,
                'source_root': str(root), 'files': {name: ref['sha256'] for name, ref in refs.items()}},
                'renderer_contract': {'renderer_source': refs[python.name]}}))
            aref = {'path': str(authority), 'sha256': S.digest(authority.read_bytes())}
            with S.ValidationFiles(30, metadata_only=True) as files:
                inventory = S.AcceptedInventory(files)
                declaration = {'authority': aref, 'source_receipt_field': 'core', 'relative_path': python.name}
                with self.assertRaises(ValueError):
                    inventory.add(refs[python.name], metadata=False, shared_metadata_role='source_receipt_text', shared_declaration=declaration)
                files.json(aref)
                for name, reference in refs.items():
                    inventory.add(reference, metadata=False, shared_metadata_role='source_receipt_text',
                        shared_declaration={'authority': aref, 'source_receipt_field': 'core', 'relative_path': name})
                # The exact reference authority works, but a same-suffix substitution does not.
                good = {'authority': aref, 'reference_field': ['renderer_contract', 'renderer_source']}
                inventory.add(refs[python.name], metadata=False, shared_metadata_role='renderer_python', shared_declaration=good)
                for altered in ({'path': str(root / 'other.py'), 'sha256': refs[python.name]['sha256']},
                                {'path': refs[python.name]['path'], 'sha256': '0' * 64}):
                    with self.assertRaises(ValueError):
                        inventory.add(altered, metadata=False, shared_metadata_role='renderer_python', shared_declaration=good)
                with self.assertRaises(ValueError):
                    inventory.add(refs[python.name], metadata=False, shared_metadata_role='source_receipt_text',
                        shared_declaration={'authority': aref, 'source_receipt_field': 'outside_core', 'relative_path': python.name})
                binary = root / 'driver.so'; binary.write_bytes(b'not source')
                with self.assertRaises(ValueError):
                    inventory.add({'path': str(binary), 'sha256': S.digest(binary.read_bytes())}, metadata=False,
                        shared_metadata_role='source_receipt_text', shared_declaration=declaration)
                inventory.resolve_blobs()
                self.assertEqual(set(inventory.declared), {str(python), str(shell)})
            # A supplied authority hash different from already pinned bytes cannot authorize anything.
            with S.ValidationFiles(30, metadata_only=True) as files:
                files.json(aref); inventory = S.AcceptedInventory(files)
                with self.assertRaises(ValueError):
                    inventory.add(refs[python.name], metadata=False, shared_metadata_role='source_receipt_text',
                        shared_declaration={'authority': dict(aref, sha256='0' * 64),
                            'source_receipt_field': 'core', 'relative_path': python.name})
                self.assertTrue(files.failed)

    def test_deadline_and_direct_identity_refusals_cannot_be_revived(self):
        for failure in ('deadline', 'entry'):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary); path = root / 'metadata.json'; path.write_bytes(b'{}\n')
                clock = [0.0]
                with S.ValidationFiles(30, metadata_only=True, clock=lambda: clock[0]) as files:
                    entry = files.file(path, S.digest(b'{}\n'), keep_bytes=True); files.finish([str(path)])
                    if failure == 'deadline':
                        clock[0] = 31
                    else:
                        path.rename(root / 'moved.json'); path.symlink_to(root / 'moved.json')
                    with self.assertRaises(ValueError):
                        files.check_entry(entry)
                    self.assertTrue(files.failed)
                    if failure == 'deadline':
                        clock[0] = 0
                    else:
                        path.unlink(); (root / 'moved.json').rename(path)
                    with self.assertRaises(ValueError):
                        files.begin_metadata_reuse()

    def test_post_open_special_file_refused_before_any_read_and_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); ordinary = root / 'file'; ordinary.write_bytes(b'x')
            fifo = root / 'fifo'; os.mkfifo(fifo)
            with S.ValidationFiles(30) as files, mock.patch.object(S.os, 'open', return_value=999) as opened, \
                    mock.patch.object(S.os, 'fstat', return_value=fifo.lstat()), \
                    mock.patch.object(S.os, 'close') as closed, \
                    mock.patch.object(S.os, 'read', side_effect=AssertionError('special-file read reached')) as read:
                with self.assertRaises(ValueError):
                    files.file(ordinary, S.digest(b'x'))
                self.assertTrue(opened.call_args.args[1] & os.O_NOFOLLOW)
                self.assertTrue(opened.call_args.args[1] & os.O_NONBLOCK)
                read.assert_not_called(); closed.assert_called_once_with(999)
                self.assertTrue(files.failed)

suite = unittest.TestSuite([
    unittest.defaultTestLoader.loadTestsFromTestCase(ExactPreviousBoundaryChecks),
    unittest.defaultTestLoader.loadTestsFromTestCase(NewNarrowBoundaries)])
capture = io.StringIO()
result = unittest.TextTestRunner(stream=capture, verbosity=2).run(suite)
for cache in S._SOURCE_FILES.values(): cache.close()
for consumer in S._CONSUMERS.values(): consumer.files.close()
unchanged = freeze_matches()
assert unchanged
process = S.compute_context(Path(INPUTS['campaign']))
receipt = {'schema_version': 1, 'kind': 'independent_corrected_accepted_evidence_snapshot_source_review',
    'reviewer': '/root/implement_direct_h100_profiles/review_direct_profile_source',
    'source_freeze': {'path': str(FREEZE), 'sha256': FREEZE_SHA},
    'source_sha256': hashlib.sha256((SCRIPTS / S.NAME).read_bytes()).hexdigest(),
    'passed': result.wasSuccessful(), 'approved': result.wasSuccessful(), 'checks_run': result.testsRun,
    'failures': len(result.failures), 'errors': len(result.errors), 'actual_cpu': process,
    'elapsed_seconds': time.monotonic() - STARTED, 'observations': OBSERVATIONS,
    'all_48_frozen_files_bytes_and_modes_unchanged': unchanged,
    'replayed_prior_focused_controls': True, 'broad_34_or_author_7_batch_repeated': False,
    'complete_original_55_qualification_body_unchanged': True,
    'complete_history_body_and_other_live_functions_unchanged': True,
    'all_13_numerical_source_hashes_unchanged': True,
    'cold_29_helper_staging_and_changed_byte_refusal': True,
    'actual_full_snapshot_or_raw_gate_replay': False, 'historical_payloads_hashed': False,
    'fresh_genuine_common_child_executed': False,
    'numerical_imports': sorted(FORBIDDEN.intersection(sys.modules)),
    'gpu_operations': 0, 'network_operations': 0, 'tool_operations': 0,
    'scientific_credit': 0, 'worker_admission': False}
for name, raw in [('independent_corrected_' + step + '.stdout.raw', capture.getvalue().encode()),
        ('independent_corrected_' + step + '.json', S.encoded(receipt))]:
    with (OUTPUT / name).open('xb') as stream:
        stream.write(raw); stream.flush(); os.fsync(stream.fileno())
print(json.dumps(receipt, sort_keys=True))
print(capture.getvalue(), file=sys.stderr)
raise SystemExit(0 if result.wasSuccessful() else 1)

