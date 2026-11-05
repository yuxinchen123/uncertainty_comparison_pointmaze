"""Independently inspect only small real metadata and focused temporary snapshot boundaries."""
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
INPUTS = json.loads((OUTPUT / 'independent_fixture_inputs.json').read_text())
FROZEN = json.loads(Path(INPUTS['frozen_source_path']).read_text())
FORBIDDEN = {'jax', 'jaxlib', 'numpy', 'torch', 'flax', 'optax', 'scipy', 'cupy', 'jaxtari', 'exploration_platform'}
OBSERVATIONS = []
STARTED = time.monotonic()


def audit(event, values):
    """Refuse numerical/device/library/network work and all actual child processes in this check."""
    if event == 'import' and values[0].split('.')[0] in FORBIDDEN:
        raise RuntimeError('numerical import refused in independent metadata fixtures')
    if event in {'subprocess.Popen', 'socket.connect', 'socket.bind', 'socket.getaddrinfo', 'ctypes.dlopen', 'ctypes.dlsym', 'os.system'}:
        raise RuntimeError('process, network or library work refused in independent metadata fixtures')
    if event == 'open' and isinstance(values[0], (str, bytes)):
        path = os.fsdecode(values[0])
        if path.startswith(('/dev/nvidia', '/dev/dri', '/tmp/sl5nw_serval02_card1.lock')) or '/direct_history/content/' in path:
            raise RuntimeError('device, standing lock or historical payload read refused in independent metadata fixtures')
        if path.startswith(('/usr/lib', '/lib/')) and 'libcuda' in path:
            raise RuntimeError('historical node-local driver read refused in independent metadata fixtures')


def source_functions(path):
    """Compare source syntax without importing any numerical or device code."""
    return {n.name: n for n in ast.parse(Path(path).read_bytes()).body if isinstance(n, ast.FunctionDef)}


def syntax(value):
    """Ignore source line numbers while retaining every operation and literal type."""
    return ast.dump(value, include_attributes=False)


def source_freeze_matches():
    """Verify every one of the fourteen source/evidence files without changing the freeze."""
    return all(hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == value['sha256']
        and oct((ROOT / relative).stat().st_mode & 0o777) == value['mode'] for relative, value in FROZEN['files'].items())


job, step = os.environ.get('SLURM_JOB_ID'), os.environ.get('SLURM_STEP_ID')
assert job == '6783003' and str(step).isdigit()
assert Path('/proc/self/cgroup').read_text().splitlines() == ['0::/system.slice/slurmstepd.scope/job_6783003/step_' + step + '/user/task_0']
assert os.getuid() == os.geteuid() == 2618919
assert hashlib.sha256(Path(INPUTS['frozen_source_path']).read_bytes()).hexdigest() == INPUTS['frozen_source_sha256']
assert source_freeze_matches()
sys.addaudithook(audit)
specification = importlib.util.spec_from_file_location('accepted_snapshot_independent_fixture_helpers', SCRIPTS / 'tests/test_accepted_evidence_snapshot.py')
AUTHOR = importlib.util.module_from_spec(specification)
specification.loader.exec_module(AUTHOR)
S = AUTHOR.snapshot


class IndependentBoundaryChecks(unittest.TestCase):
    """Use only tiny files, explicit synthetic review fixtures and real small accepted records."""

    def fixture(self):
        """Borrow the author's complete synthetic positive record without changing any target predicate."""
        value = AUTHOR.ReviewedMetadataTests(methodName='test_reviewed_fact_consumes_without_opening_any_historical_payload')
        value.setUp()
        self.addCleanup(value.tearDown)
        return value

    def test_invalid_metadata_cache_stays_failed_after_parent_path_restoration(self):
        """A failed alias check must permanently invalidate the same retained metadata cache."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); current = root / 'current'; current.mkdir()
            path = current / 'input.json'; path.write_bytes(b'{}\n')
            with S.ValidationFiles(30, metadata_only=True) as files:
                files.file(path, S.digest(b'{}\n'), keep_bytes=True); files.finish([str(path)])
                current.rename(root / 'moved')
                current.symlink_to(root / 'moved', target_is_directory=True)
                with self.assertRaises(ValueError):
                    files.begin_metadata_reuse()
                failed_after_refusal = files.failed
                current.unlink(); (root / 'moved').rename(current)
                accepted_after_refusal = False
                try:
                    files.begin_metadata_reuse(); files.finish([str(path)])
                    accepted_after_refusal = True
                except ValueError:
                    pass
                OBSERVATIONS.append({'case': 'permanent_metadata_failure', 'failed_flag_after_refusal': failed_after_refusal,
                    'same_cache_accepted_after_restoration': accepted_after_refusal})
                self.assertFalse(accepted_after_refusal, 'the same failed metadata cache was accepted again')

    def test_nonregular_file_is_refused_before_open(self):
        """The ordinary-file boundary must reject a FIFO before opening a special file."""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'owned_fifo'; os.mkfifo(path)
            with S.ValidationFiles(30) as files:
                with mock.patch.object(S.os, 'open', side_effect=AssertionError('nonregular path reached os.open')):
                    with self.assertRaises(ValueError):
                        files.file(path, '0' * 64)

    def test_unmapped_historical_node_driver_has_no_live_path_fallback(self):
        """A missing captured driver reference must not become a read of today's node-local path."""
        with S.ValidationFiles(30) as files:
            inventory = S.AcceptedInventory(files)
            inventory.add({'path': '/usr/lib/x86_64-linux-gnu/libcuda.so.590.48.01', 'sha256': '0' * 64}, metadata=False)
            with self.assertRaises(ValueError):
                inventory.resolve_blobs()

    def test_real_owned_compute_context_rejects_forged_job_environment(self):
        """The current kernel cgroup and all UIDs must match the declared job and step."""
        value = S.compute_context(Path(INPUTS['campaign']))
        self.assertEqual(value['job_id'], '6783003')
        self.assertEqual(value['step_id'], step)
        self.assertEqual(value['uids'], [2618919] * 4)
        with mock.patch.dict(os.environ, {'SLURM_JOB_ID': '987654', 'SLURM_STEP_ID': '0'}):
            with self.assertRaises(ValueError):
                S.compute_context(Path(INPUTS['campaign']))

    def test_boolean_schema_and_seed_substitutions_are_refused(self):
        """Equal-looking Boolean counters cannot replace exact schemas or independent seeds."""
        for mutation in ['proof_schema', 'review_schema', 'seeds', 'copies']:
            with self.subTest(mutation=mutation):
                f = AUTHOR.ReviewedMetadataTests(methodName='test_reviewed_fact_consumes_without_opening_any_historical_payload')
                f.setUp()
                try:
                    if mutation == 'proof_schema':
                        f.proof['schema_version'] = True
                        f.plan['accepted_evidence_snapshot_inputs']['proof'] = f.write('proof.json', f.proof)
                        f.review['snapshot'] = f.plan['accepted_evidence_snapshot_inputs']['proof']; f.replace_review()
                    elif mutation == 'review_schema':
                        f.review['schema_version'] = True; f.replace_review()
                    elif mutation == 'seeds':
                        f.plan['copy_seeds'] = [True]
                    else:
                        f.plan['configuration']['learner']['n_copies'] = True
                    with self.assertRaises(ValueError):
                        S.consume(f.plan)
                finally:
                    f.tearDown()

    def test_cpu_descriptor_identities_are_not_asserted_on_other_hosts(self):
        """A complete explicitly synthetic CPU fact is not compared to a worker host's device numbers."""
        f = self.fixture()
        value = S.consume(f.plan)
        self.assertEqual(value.proof['producer']['hostname'], 'different_cpu_host')
        self.assertFalse(S.history(f.plan)['cpu_descriptor_identity_asserted_on_gpu'])

    def test_cold_exact_twenty_nine_helper_staging_and_changed_bytes_refusal(self):
        """A fresh staged namespace contains exactly the intended source set before any raw snapshot exists."""
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory)
            names = S.required_helper_names(); self.assertEqual(len(names), 29)
            for name in names:
                (target / name).write_bytes((SCRIPTS / name).read_bytes())
            specification = importlib.util.spec_from_file_location('cold_independent_snapshot_source', target / S.NAME)
            cold = importlib.util.module_from_spec(specification); specification.loader.exec_module(cold)
            plan = AUTHOR.CurrentPlanTests().plan()
            try:
                cold.require_plan(plan)
                (target / 'ideas.py').write_bytes((target / 'ideas.py').read_bytes() + b'\n# temporary changed source\n')
                with self.assertRaises(ValueError):
                    cold.require_plan(plan)
            finally:
                for cache in cold._SOURCE_FILES.values(): cache.close()
            common = source_functions(target / 'direct_common_confirmation.py')
            self.assertIn('accepted_snapshot_support(plan).common_validation(plan, plan_sha256)', ast.unparse(common['validate_actual']))
            self.assertIn('reviewed_accepted_direct_common_correctness_validation', ast.unparse(common['request']))

    def test_original_reader_bodies_and_all_live_functions_are_unchanged(self):
        """Only explicit new dispatches change; actual compute, binding, backend and numerical source remain intact."""
        old = Path(INPUTS['old_common_root']) / '10_jax_exploration_platform/scripts/atari_pixel_campaign'
        qualified = Path(INPUTS['old_qualification_unit']) / 'code/direct_measurement.py'
        original = source_functions(qualified)['verify_qualification_inputs']
        current = source_functions(SCRIPTS / 'direct_measurement.py')['verify_qualification_inputs']
        self.assertEqual(len(original.body), 55)
        self.assertEqual([syntax(n) for n in original.body], [syntax(n) for n in current.body[:1] + current.body[3:]])
        changed = {'direct_common_confirmation.py': {'accepted_snapshot_support', 'contract', 'validate_actual', 'request', 'result_binding', 'main'},
            'direct_measurement.py': {'verify_qualification_inputs'}, 'measure_selected.py': {'required_helpers', 'prepare', 'parse_arguments'},
            'serval02_card1_binding.py': {'validate_history'}}
        for filename, allowed in changed.items():
            before, after = source_functions(old / filename), source_functions(SCRIPTS / filename)
            self.assertEqual(set(after) - set(before), {'accepted_snapshot_support'} if filename == 'direct_common_confirmation.py' else set())
            self.assertEqual(set(before) - set(after), set())
            for name in set(before) - allowed:
                self.assertEqual(syntax(before[name]), syntax(after[name]), filename + ':' + name)
            if filename == 'serval02_card1_binding.py':
                self.assertEqual([syntax(n) for n in before['validate_history'].body],
                    [syntax(n) for n in after['validate_history'].body[:1] + after['validate_history'].body[2:]])
        plan = json.loads(Path(INPUTS['old_plan']).read_text()); protected = {}
        for label in ('core', 'paired_source'):
            for relative, expected in plan[label]['files'].items():
                expected = expected['sha256'] if isinstance(expected, dict) else expected
                self.assertEqual(hashlib.sha256((Path(plan[label]['source_root']) / relative).read_bytes()).hexdigest(), expected)
                protected[relative] = expected
        self.assertEqual(len(protected), 13)

    def test_real_accepted_small_metadata_schemas_without_payload_hashes(self):
        """Inspect genuine Q/C1/C2 schema joins only; do not run inventory completion or old raw consumers."""
        qualification_review = json.loads(Path(INPUTS['small_review_references']['qualification']['path']).read_text())
        for key in ['qualification', '1', '2']:
            with self.subTest(role=key), S.ValidationFiles(60, metadata_only=True) as files:
                inventory = S.AcceptedInventory(files)
                value = S.accepted_role(files, inventory, INPUTS['small_review_references'][key],
                    'qualification' if key == 'qualification' else 'correctness', qualification_review)
                OBSERVATIONS.append({'case': 'real_small_metadata_schema', 'role': key, 'passed': True,
                    'review': value['review'], 'copy_seeds': value['copy_seeds'], 'small_physical_files_read': len(files.physical),
                    'small_metadata_bytes_read': files.bytes_hashed, 'declared_large_payloads_not_read': len(inventory.pending_blobs)})


capture = io.StringIO()
result = unittest.TextTestRunner(stream=capture, verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(IndependentBoundaryChecks))
for cache in S._SOURCE_FILES.values(): cache.close()
for consumer in S._CONSUMERS.values(): consumer.files.close()
assert source_freeze_matches()
process = S.compute_context(Path(INPUTS['campaign']))
receipt = {'schema_version': 1, 'kind': 'independent_focused_accepted_evidence_snapshot_source_review',
    'reviewer': '/root/prepare_fresh_historical_probe', 'source_freeze_sha256': INPUTS['frozen_source_sha256'],
    'passed': result.wasSuccessful(), 'approved': result.wasSuccessful(), 'checks_run': result.testsRun,
    'failures': len(result.failures), 'errors': len(result.errors), 'actual_cpu': process,
    'elapsed_seconds': time.monotonic() - STARTED, 'observations': OBSERVATIONS,
    'source_freeze_unchanged': True, 'author_34_case_batch_repeated': False,
    'actual_full_snapshot_or_raw_gate_replay': False, 'fresh_genuine_common_child_executed': False,
    'cold_twenty_nine_helper_namespace_checked': True, 'numerical_imports': sorted(FORBIDDEN.intersection(sys.modules)),
    'gpu_operations': 0, 'network_operations': 0, 'scientific_credit': 0, 'worker_admission': False}
for name, raw in [('focused_independent_checks.stdout.raw', capture.getvalue().encode()),
        ('focused_independent_review.json', S.encoded(receipt))]:
    with (OUTPUT / name).open('xb') as stream:
        stream.write(raw);stream.flush();os.fsync(stream.fileno())
print(json.dumps(receipt, sort_keys=True))
print(capture.getvalue(), file=sys.stderr)
raise SystemExit(0 if result.wasSuccessful() else 1)
