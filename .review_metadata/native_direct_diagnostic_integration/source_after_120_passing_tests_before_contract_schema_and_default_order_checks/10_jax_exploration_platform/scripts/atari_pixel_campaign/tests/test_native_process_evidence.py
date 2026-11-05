"""Test explicit diagnostic policy and saved native records without any native process operation."""
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS))
import native_process_evidence as evidence
import direct_profile_process as direct
import diagnose_direct_nsight_process as diagnostic


def declared_plan():
    """Build explicitly labelled metadata with no actual target contract or operation."""
    plan = {'kind': evidence.DIAGNOSTIC, 'python': '/synthetic/selected/python',
        'helpers': {'files': {evidence.PREFIX + name: str(index + 1) * 64
            for index, name in enumerate(evidence.HELPERS)}}}
    plan['native_process_control'] = evidence.declaration(plan)
    return plan


def synthetic_calls():
    """Construct a labelled self-preflight shape; no descriptor is opened by this fixture."""
    # Before: three declared operations; after: exact call/event pairs for one synthetic generation.
    owner = {'pid': 101, 'command': ['/synthetic/selected/python'], 'executable': '/synthetic/selected/python'}
    # These declarations are synthetic reader inputs; no image exists or is mapped by this fixture.
    image = {'path': '/synthetic/public/libc.so.6', 'inode': 42, 'device': os.makedev(8, 1),
        'sha256': '0' * 64, 'size_bytes': 8192, 'mode': 0o755, 'uid': 0}
    contract = {'kind': evidence.KIND,
        'selected_python': {'actual_running_image_path': owner['executable']},
        'symbols': {name: {'image': copy.deepcopy(image), 'symbol_file_offset': offset}
            for name, offset in [('syscall', 0x100), ('close', 0x200)]}}
    symbols = {name: {'image': copy.deepcopy(image), 'symbol_file_offset': offset,
        'actual_pointer': 0x1000 + offset,
        'actual_mapping': '1000-2000 r-xp 00000000 08:01 42 /synthetic/public/libc.so.6'}
        for name, offset in [('syscall', 0x100), ('close', 0x200)]}
    raw = 'pos:\t0\nflags:\t02000002\nmnt_id:\t5\nino:\t77\nPid:\t101\nNSpid:\t101\n'
    identity = {'device': 6, 'inode': 77, 'pid': 101, 'raw_fdinfo': raw,
        'fdinfo_sha256': hashlib.sha256(raw.encode()).hexdigest()}
    specifications = [('pidfd_open', {'syscall': 434, 'pid': 101, 'flags': 0}, 7, 'ordinary'),
        ('pidfd_send_signal', {'syscall': 424, 'fd': 7, 'signal': 0, 'siginfo': None, 'flags': 0}, 0, 'ordinary'),
        ('close', {'fd': 7}, 0, 'reserved_descriptor_cleanup')]
    calls, events = [], []
    for index, (name, arguments, returned, budget) in enumerate(specifications):
        start = 10.0 + index
        calls.append({'sequence': index, 'name': name, 'phase': 'self_preflight', 'arguments': arguments,
            'result': returned, 'errno': 0, 'started_at_seconds': start, 'finished_at_seconds': start + .1,
            'producer_pid': 101, 'call_budget': budget})
        events.append({'event_sequence': index, 'operation': name, 'descriptor': 7, 'call_sequence': index,
            'open_generation': 0, 'opened_pid': 101, 'identity': copy.deepcopy(identity),
            'observed_at_seconds': start + .15 if index == 0 else start - .1})
    value = {'kind': evidence.KIND, 'contract': contract, 'preflight_passed': True, 'preflight_call_count': 3,
        'open_handle_count': 0, 'native_call_failure': False, 'numeric_pid_signal_fallback': False,
        'python_exports_changed': False, 'gpu_or_cuda_qualification': False, 'scientific_credit': 0,
        'ordinary_call_limit': evidence.LIMIT, 'maximum_cleanup_records_per_open': 3,
        'ordinary_call_count': 2, 'reserved_cleanup_call_count': 1, 'calls': calls, 'descriptor_events': events,
        'bootstrap': {'pid': 101, 'sys_executable': owner['command'][0], 'symbols': symbols,
            'sys_version': 'explicit synthetic Python version; no interpreter or target qualification',
            'gpu_or_cuda_qualification': False}}
    return value, owner, {101: owner}, contract


class NativeDiagnosticPolicyTests(unittest.TestCase):
    """Preserve default helper selection while refusing implicit or formal native target observation."""
    def test_default_five_and_explicit_native_seven_helpers(self):
        """Only the explicit native selection adds the two declared helpers."""
        self.assertEqual(diagnostic.required_helpers(), diagnostic.HELPERS)
        self.assertEqual(len(diagnostic.required_helpers()), 5)
        self.assertEqual(diagnostic.required_helpers(True), (*diagnostic.HELPERS, *evidence.HELPERS))
        self.assertEqual(len(set(diagnostic.required_helpers(True))), 7)
        for value in (1, 'native', None, {}, []):
            with self.subTest(value=value), self.assertRaises(ValueError):diagnostic.required_helpers(value)

    def test_explicit_policy_has_no_invented_target_contract(self):
        """Preparation records diagnostic observation with no qualification or future target hash."""
        plan = declared_plan()
        self.assertEqual(evidence.policy(plan), plan['native_process_control'])
        self.assertIsNone(evidence.policy(plan)['expected_target_contract'])
        self.assertIs(evidence.policy(plan)['qualification'], False)

    def test_formal_or_missing_selection_never_chooses_native_automatically(self):
        """Missing Python exports cannot enable this diagnostic-only policy in another producer."""
        plan = declared_plan()
        for kind in ('selected_configuration_profile', 'nsight_process_identity_diagnostic', 'final_training', None):
            changed = copy.deepcopy(plan);changed['kind'] = kind
            with self.subTest(kind=kind), self.assertRaises(ValueError):evidence.policy(changed)
        changed = copy.deepcopy(plan);del changed['native_process_control']
        with self.assertRaises(ValueError):evidence.policy(changed)

    def test_policy_requires_exact_types_fields_backend_and_hashes(self):
        """Boolean versions, foreign helper bytes and undeclared policy fields refuse before startup."""
        plan = declared_plan()
        changes = [('schema_version', True), ('qualification', 0), ('python', '/different/python'),
            ('backend', 'automatic'), ('mode', 'reviewed_target_contract'),
            ('expected_target_contract', {'sha256': '0' * 64}), ('extra', True)]
        for field, value in changes:
            changed = copy.deepcopy(plan);changed['native_process_control'][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):evidence.policy(changed)
        changed = copy.deepcopy(plan)
        name = evidence.HELPERS[0]
        changed['helpers']['files'][evidence.PREFIX + name] = 'z' * 64
        changed['native_process_control'] = evidence.declaration(changed)
        with self.assertRaises(ValueError):evidence.policy(changed)
        changed = copy.deepcopy(plan);del changed['native_process_control']['helper_hashes'][name]
        with self.assertRaises(ValueError):evidence.policy(changed)

    def test_default_observer_stays_on_the_original_metadata_reader(self):
        """A default plan neither imports a provider nor accepts a supplied native runtime."""
        plan = {'kind': evidence.DIAGNOSTIC}
        generic = object()
        with patch.object(direct, 'generic', return_value=generic) as read, patch.object(direct, 'adjacent_helper') as adjacent:
            self.assertIs(direct.runtime_observer(plan), generic)
            read.assert_called_once_with(plan);adjacent.assert_not_called()
            with self.assertRaises(ValueError):direct.runtime_observer(plan, object())

    def test_native_observer_requires_one_prestarted_owned_runtime(self):
        """The runtime is checked explicitly; reader calls do not start another provider."""
        plan = declared_plan()
        runtime = SimpleNamespace(require_owner=Mock())
        observer = SimpleNamespace(native_process_runtime=runtime)
        with patch.object(direct, 'adjacent_helper', return_value=evidence), patch.object(direct, 'generic') as generic:
            with self.assertRaises(ValueError):direct.runtime_observer(plan)
            with self.assertRaises(ValueError):direct.runtime_observer(plan, object())
            self.assertIs(direct.runtime_observer(plan, observer), observer)
            runtime.require_owner.assert_called_once_with(plan);generic.assert_not_called()
            runtime.require_owner.side_effect = ValueError('explicit synthetic inherited-runtime refusal')
            with self.assertRaisesRegex(ValueError, 'inherited-runtime'):direct.runtime_observer(plan, observer)

    def test_unknown_role_cannot_select_another_role_artifact(self):
        """Only the three fixed role names define saved native artifact paths."""
        for role in evidence.ROLES:
            paths = evidence.role_files(Path('/synthetic/unit'), role)
            self.assertEqual(set(paths), {'begin', 'terminal', 'failure'})
            self.assertEqual(paths['begin'].name, 'native_process_' + role + '_begin.json')
        for role in ('controller', '../producer', None, 1):
            with self.subTest(role=role), self.assertRaises(ValueError):evidence.role_files(Path('/synthetic/unit'), role)

    def test_saved_helper_import_is_pure_in_a_fresh_standard_library_process(self):
        """Importing the saved reader never imports ctypes, a provider or a numerical package."""
        self.assertEqual(os.environ.get('SLURM_JOB_ID'), '6783003')
        program = """import importlib.util,json,sys
path=sys.argv[1]
spec=importlib.util.spec_from_file_location('pure_native_saved_reader_fixture',path)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
forbidden={'ctypes','linux_pidfd_process_control','jax','jaxlib','numpy','torch','flax','optax','scipy','cupy','jaxtari','exploration_platform'}
present=sorted(name for name in sys.modules if name.split('.')[0] in forbidden)
assert not present,present
print(json.dumps({'forbidden_imports':present,'saved_reader_callable':callable(module.validate_saved)}))
"""
        result = subprocess.run([sys.executable, '-I', '-S', '-B', '-c', program,
            str(SCRIPTS / 'native_process_evidence.py')], capture_output=True, timeout=10, check=True)
        self.assertEqual(result.stderr, b'')
        self.assertEqual(json.loads(result.stdout), {'forbidden_imports': [], 'saved_reader_callable': True})


class NativeSavedCallTests(unittest.TestCase):
    """Use explicitly synthetic saved operations; these tests never issue a native syscall."""
    def test_complete_synthetic_self_preflight_is_read_without_native_loading(self):
        """The saved call reader accepts exact metadata without loading the native provider."""
        value, owner, allowed, contract = synthetic_calls()
        with patch.object(evidence, 'load_provider', side_effect=AssertionError('saved reader loaded native operations')):
            self.assertEqual(evidence.validate_calls(value, owner, allowed, contract),
                {'calls': 3, 'ordinary_calls': 2, 'cleanup_calls': 1})

    def test_reused_descriptor_number_requires_a_new_open_generation_after_close(self):
        """A later open may reuse a descriptor number only after the earlier generation closes."""
        value, owner, allowed, contract = synthetic_calls()
        later_calls, later_events = copy.deepcopy(value['calls']), copy.deepcopy(value['descriptor_events'])
        for call, event in zip(later_calls, later_events):
            call['sequence'] += 3;call['phase'] = 'owned_process'
            call['started_at_seconds'] += 3;call['finished_at_seconds'] += 3
            event['event_sequence'] += 3;event['call_sequence'] += 3;event['open_generation'] = 3
            event['observed_at_seconds'] += 3
        value['calls'].extend(later_calls);value['descriptor_events'].extend(later_events)
        value['ordinary_call_count'] = 4;value['reserved_cleanup_call_count'] = 2
        self.assertEqual(evidence.validate_calls(value, owner, allowed, contract)['calls'], 6)
        changed = copy.deepcopy(value);changed['descriptor_events'][4]['open_generation'] = 0
        with self.assertRaises(ValueError):evidence.validate_calls(changed, owner, allowed, contract)

    def test_missing_identity_duplicate_generation_and_changed_fdinfo_refuse(self):
        """Missing or substituted raw handle observations cannot support a completed saved record."""
        value, owner, allowed, contract = synthetic_calls()
        variations = []
        changed = copy.deepcopy(value);changed['descriptor_events'].pop();variations.append(changed)
        changed = copy.deepcopy(value);changed['descriptor_events'][1]['call_sequence'] = 0;variations.append(changed)
        changed = copy.deepcopy(value);changed['descriptor_events'][1]['open_generation'] = True;variations.append(changed)
        changed = copy.deepcopy(value);changed['descriptor_events'][1]['identity']['inode'] += 1;variations.append(changed)
        changed = copy.deepcopy(value);changed['descriptor_events'][1]['identity']['raw_fdinfo'] += 'Pid:\t999\n';variations.append(changed)
        changed = copy.deepcopy(value);changed['descriptor_events'][1]['identity']['fdinfo_sha256'] = '0' * 64;variations.append(changed)
        changed = copy.deepcopy(value);changed['calls'][2]['arguments']['fd'] = 8;variations.append(changed)
        for index, changed in enumerate(variations):
            with self.subTest(index=index), self.assertRaises(ValueError):evidence.validate_calls(changed, owner, allowed, contract)

    def test_nonfinite_times_wrong_returns_and_boolean_counts_refuse(self):
        """Successful metadata needs ordered finite times and exact integer identities and returns."""
        value, owner, allowed, contract = synthetic_calls()
        changes = [('ordinary_call_limit', True), ('ordinary_call_count', True), ('open_handle_count', False),
            ('preflight_call_count', True), ('reserved_cleanup_call_count', False)]
        for field, replacement in changes:
            changed = copy.deepcopy(value);changed[field] = replacement
            with self.subTest(field=field), self.assertRaises(ValueError):evidence.validate_calls(changed, owner, allowed, contract)
        for field, replacement in [('result', False), ('errno', 1), ('sequence', True),
                ('producer_pid', 999), ('started_at_seconds', float('nan')), ('finished_at_seconds', 0)]:
            changed = copy.deepcopy(value);changed['calls'][1][field] = replacement
            with self.subTest(field=field), self.assertRaises(ValueError):evidence.validate_calls(changed, owner, allowed, contract)

    def test_unknown_process_signal_syscall_and_cleanup_budget_refuse(self):
        """No unproved process or changed native operation can be accepted as successful saved work."""
        value, owner, allowed, contract = synthetic_calls()
        variations = []
        changed = copy.deepcopy(value);changed['calls'][0]['arguments']['pid'] = 999;variations.append(changed)
        changed = copy.deepcopy(value);changed['calls'][0]['arguments']['syscall'] = 435;variations.append(changed)
        changed = copy.deepcopy(value);changed['calls'][1]['arguments']['signal'] = 2;variations.append(changed)
        changed = copy.deepcopy(value);changed['calls'][1]['call_budget'] = 'reserved_descriptor_cleanup';variations.append(changed)
        changed = copy.deepcopy(value);changed['calls'][2]['call_budget'] = 'ordinary';variations.append(changed)
        changed = copy.deepcopy(value);changed['calls'][2]['name'] = 'arbitrary';variations.append(changed)
        for index, changed in enumerate(variations):
            with self.subTest(index=index), self.assertRaises(ValueError):evidence.validate_calls(changed, owner, allowed, contract)

    def test_partial_preflight_or_unknown_open_handles_never_complete(self):
        """Self preflight and terminal handle counts are required even if every remaining record looks valid."""
        value, owner, allowed, contract = synthetic_calls()
        for field, replacement in [('preflight_passed', False), ('native_call_failure', True), ('open_handle_count', 1),
                ('numeric_pid_signal_fallback', True), ('python_exports_changed', True), ('gpu_or_cuda_qualification', True)]:
            changed = copy.deepcopy(value);changed[field] = replacement
            with self.subTest(field=field), self.assertRaises(ValueError):evidence.validate_calls(changed, owner, allowed, contract)
        changed = copy.deepcopy(value);changed['calls'][1]['phase'] = 'owned_process'
        with self.assertRaises(ValueError):evidence.validate_calls(changed, owner, allowed, contract)


if __name__ == '__main__':
    unittest.main()
