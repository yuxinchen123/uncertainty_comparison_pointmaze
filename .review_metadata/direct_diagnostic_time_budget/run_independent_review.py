"""Check the one-origin diagnostic budget with finite owned CPU fixtures."""
import ast
import copy
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

W = Path(__file__).resolve().parents[2]
M = W / '.review_metadata/direct_diagnostic_time_budget'
S = W / '10_jax_exploration_platform/scripts/atari_pixel_campaign'
STEP = os.environ.get('SLURM_STEP_ID', '')
OUT = M / ('independent_' + STEP + '.json')
ARTIFACTS = M / ('independent_CPU_' + STEP)
assert os.environ.get('SLURM_JOB_ID') == '6783003' and STEP.isdecimal()
assert os.environ.get('SLURM_CPUS_PER_TASK') == '1' and not os.environ.get('SLURM_STEP_GPUS')
assert not os.environ.get('CUDA_VISIBLE_DEVICES') and not OUT.exists() and not ARTIFACTS.exists()
ARTIFACTS.mkdir()
hashes, findings = {}, []


def sha(path):
    path = Path(path)
    value = hashlib.sha256(path.read_bytes()).hexdigest();hashes[str(path)] = value
    return value


def save(name, value):
    path = ARTIFACTS / (name + '.json')
    raw = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n').encode()
    with path.open('xb') as stream:stream.write(raw);stream.flush();os.fsync(stream.fileno())
    return {'path': str(path), 'sha256': hashlib.sha256(raw).hexdigest()}


freeze_path = M / 'author_source_freeze_6.json'
assert sha(freeze_path) == '040d326f412ab70bd608074cc90dc5c3758bf7cd8b29496b04a6acbed3e5f462'
assert sha(M / 'author_handoff.json') == '16319da44a50d1d899f498926042e07fe191bbe81b85d1795d133dc0f1b743be'
freeze = json.loads(freeze_path.read_bytes())
assert len(freeze['protected_files']) == 34
for path, expected in freeze['files'].items():assert sha(W / path) == expected
for path, expected in freeze['protected_files'].items():assert sha(path) == expected
scope_text = (S / 'direct_tool_descendants.py').read_text()
for name, expected in freeze['unchanged_tool_scope_function_hashes'].items():
    candidates = [node for node in ast.walk(ast.parse(scope_text)) if isinstance(node, ast.FunctionDef) and node.name == name]
    assert len(candidates) == 1 and hashlib.sha256(ast.get_source_segment(scope_text, candidates[0]).encode()).hexdigest() == expected
handoff = json.loads((M / 'author_handoff.json').read_bytes())
for value in handoff['actual_CPU_proofs'].values():assert sha(value['path']) == value['sha256']
for value in [handoff['latest_focused_execution'], *handoff['earlier_focused_executions']]:assert sha(value['path']) == value['sha256']
before = dict(hashes)
sys.path.insert(0, str(S));sys.path.insert(0, str(S / 'tests'))
import test_direct_diagnostic_time_budget as tested
import test_direct_diagnostic_budget_boundaries as boundaries
direct, scope_source = tested.direct, tested.scope_source
os.environ['DIAGNOSTIC_BUDGET_CPU_REVIEW_OUTPUT'] = str(ARTIFACTS / 'actual_own_child')


class IndependentSlowClosureBoundary(unittest.TestCase):
    def test_closure_returning_after_cleanup_expiry_is_never_labelled_within_budget(self):
        """Assume a verified but late synthetic closure; check only the producer's budget classification."""
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            tested.M.publish(directory / 'tool_descendant_scope_begin.json', {'explicit_synthetic_begin': True})
            scope = scope_source.OwnedToolScope.__new__(scope_source.OwnedToolScope)
            scope.directory, scope.deadline = directory, tested.deadline()
            scope.pid, scope.birth, scope.failed, scope.closed = 100, '10', False, False
            scope.read_errors, scope.budget_inconclusive_published = [], False
            scope.grace_seconds, scope.active = 120, True
            scope.previous = {'value': 0}
            scope.request_counts_before = dict(scope_source._REQUEST_COUNTS)
            scope.waits, scope.nodes, scope.events, scope.interventions = {}, {}, [], []
            scope.handles, scope.producer_handle = {}, 777
            scope.check_descriptor = Mock(return_value={'explicit_synthetic_custody': True})
            scope.scan, scope.progress = Mock(), Mock()
            clock, subreaper = [350.], [1]
            def close_late():
                clock[0] = 401.
                return {'explicit_synthetic_verified_closure_after_expiry': True}
            scope.closure = close_late
            scope.signal_known = Mock()
            scope.observer = SimpleNamespace(publish=tested.M.publish, receipt=tested.M.receipt,
                close_process_handle=Mock())
            def fake_subreaper(value=None):
                if value is not None:subreaper[0] = value
                return {'value': subreaper[0], 'returncode': 0, 'errno': 0}
            with patch.object(direct.time, 'monotonic', side_effect=lambda: clock[0]), \
                    patch.object(scope_source, 'child_subreaper', side_effect=fake_subreaper):
                scope.finish(False)
            terminal = tested.M.read_json(directory / 'tool_descendant_scope_terminal.json')
            saved_rejected = False
            try:scope.deadline.verify_snapshot(terminal['diagnostic_deadline'], phase='total', cleanup_required=True)
            except ValueError:saved_rejected = True
            value = {'synthetic_clock_and_closure_only': True, 'actual_process_or_descriptor_operations': 0,
                'cleanup_begin': 350., 'total_and_cleanup_deadline': 400., 'closure_return': 401.,
                'published_within_budget': terminal['within_diagnostic_budget'],
                'budget_inconclusive_published': scope.budget_inconclusive_published,
                'pure_saved_deadline_validator_rejected_late_record': saved_rejected,
                'terminal': terminal, 'source_sha256': sha(S / 'direct_tool_descendants.py')}
            findings.append({'name': 'slow_closure_crosses_deadline_before_terminal_classification',
                'incorrect_within_budget_claim': terminal['within_diagnostic_budget'],
                'receipt': save('slow_closure_boundary', value)})
            self.assertFalse(terminal['within_diagnostic_budget'],
                'closure completed after the shared deadline but the producer labelled its terminal within budget')


started = time.time()
suite = unittest.TestSuite([
    unittest.defaultTestLoader.loadTestsFromTestCase(tested.DeadlineArithmeticTests),
    tested.OriginAndControllerTests('test_origin_is_durable_before_popen_and_has_no_future_producer_identity'),
    tested.OriginAndControllerTests('test_live_role_refuses_a_different_host_or_boot_domain'),
    tested.OriginAndControllerTests('test_controller_timeout_has_no_extra_wait_or_holder_signal'),
    unittest.defaultTestLoader.loadTestsFromTestCase(tested.ActualControllerTimerTests),
    unittest.defaultTestLoader.loadTestsFromTestCase(boundaries.ReleaseBoundaryTests),
    unittest.defaultTestLoader.loadTestsFromTestCase(boundaries.NativeBudgetRoleTests),
    unittest.defaultTestLoader.loadTestsFromTestCase(IndependentSlowClosureBoundary)])
result = unittest.TextTestRunner(verbosity=2).run(suite)
assert all(sha(path) == expected for path, expected in before.items())
assert time.time() - started < 240
assert not any(name.split('.')[0] in {'jax', 'jaxlib', 'numpy', 'torch', 'flax', 'optax', 'cupy', 'jaxtari', 'exploration_platform'} for name in sys.modules)
raw_artifacts = {str(path): sha(path) for path in ARTIFACTS.rglob('*.json')}
payload = {'kind': 'independent_single_origin_300_second_diagnostic_budget_review',
    'passed': result.wasSuccessful(), 'source_approved': result.wasSuccessful(),
    'job_id': os.environ['SLURM_JOB_ID'], 'step_id': STEP, 'pid': os.getpid(),
    'sys_executable': sys.executable, 'requested_memory_mib': 512, 'hard_slurm_time_limit_seconds': 300,
    'started_at_seconds': started, 'finished_at_seconds': time.time(), 'tests_run': result.testsRun,
    'failures': [{'test': str(test), 'traceback': text} for test, text in result.failures],
    'errors': [{'test': str(test), 'traceback': text} for test, text in result.errors],
    'findings': findings, 'files_sha256': hashes, 'actual_CPU_raw_sha256': raw_artifacts,
    'source_freeze': {'path': str(freeze_path), 'sha256': sha(freeze_path)},
    'review_program': {'path': str(Path(__file__).resolve()), 'sha256': sha(Path(__file__).resolve())},
    'frozen_files_and_34_protected_unchanged': True, 'scope_functions_unchanged': 18,
    'actual_target_Nsight_GPU_standing_lock_actions': False, 'scientific_credit': 0, 'admission_granted': False}
raw = (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + '\n').encode()
with OUT.open('xb') as stream:stream.write(raw);stream.flush();os.fsync(stream.fileno())
print(json.dumps({'passed': payload['passed'], 'tests_run': result.testsRun, 'failures': len(result.failures),
                  'errors': len(result.errors), 'receipt': str(OUT), 'sha256': hashlib.sha256(raw).hexdigest()}))
raise SystemExit(0 if result.wasSuccessful() else 1)
