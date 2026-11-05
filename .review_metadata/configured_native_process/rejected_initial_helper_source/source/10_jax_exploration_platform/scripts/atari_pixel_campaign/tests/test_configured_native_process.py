"""Check configured native declarations and exact saved-role dispatch with no target or tool operation."""
import ast
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS))
import configured_native_process as native
import diagnose_configured_native_process as diagnostic
import native_process_evidence as evidence
import nsight_process_identity as identity


def declared_plan(kind=native.DIAGNOSTIC):
    """Create explicitly synthetic metadata with real helper hashes and no actual target contract."""
    plan = {'kind': kind, 'python': '/synthetic/selected/python', 'hardware_class': 'jaguar03_rtx_a4500',
        'gpu_uuid': None, 'maximum_active_seconds': 240, 'diagnostic_time_budget': dict(native.BUDGET),
        'helpers': {'files': {native.PREFIX + name: hashlib.sha256((SCRIPTS / name).read_bytes()).hexdigest()
            for name in diagnostic.HELPERS}}}
    plan.update(native.declaration(plan))
    return plan


class ConfiguredNativePolicyTests(unittest.TestCase):
    """Refuse implicit selection, old direct evidence and substituted allocation policy fields."""
    def test_distinct_observation_has_no_target_or_admission(self):
        """Diagnostic declaration uses seven exact helpers and no future target bytes."""
        plan = declared_plan()
        self.assertEqual(len(diagnostic.HELPERS), 7)
        self.assertEqual(native.policy(plan), evidence.policy(plan))
        self.assertIsNone(plan['native_process_control']['expected_target_contract'])
        self.assertIs(plan['configured_native_process_control']['standing_gpu_descriptor_held'], False)
        self.assertIs(plan['configured_native_process_control']['physical_gres_mapping_verified'], False)

    def test_missing_selection_never_chooses_native(self):
        """An old configured/default plan remains on its declared existing process mechanism."""
        self.assertIsNone(native.optional_policy({'kind': 'selected_configuration_profile'}))
        changed = declared_plan();del changed['configured_native_process_control']
        with self.assertRaises(ValueError):native.optional_policy(changed)

    def test_exact_budget_types_and_no_direct_binding(self):
        """A Boolean version, changed limits or direct-H100 field refuses before native startup."""
        for field, value in [('schema_version', True), ('work_seconds', 239), ('cleanup_seconds', 61), ('total_seconds', 600)]:
            changed = declared_plan();changed['diagnostic_time_budget'][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):native.policy(changed)
        for field, value in [('direct_binding', {}), ('gpu_uuid', 'GPU-not-an-allocation-selector'), ('hardware_class', 'h100'), ('maximum_active_seconds', True)]:
            changed = declared_plan();changed[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):native.policy(changed)

    def test_wrong_declaration_helpers_modes_and_added_fields_refuse(self):
        """Explicit declarations cannot silently change helper bytes, trust or target-observation mode."""
        for key, value in [('schema_version', True), ('standing_gpu_descriptor_held', True), ('physical_gres_mapping_verified', True), ('unknown', False)]:
            changed = declared_plan();changed['configured_native_process_control'][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):native.policy(changed)
        changed = declared_plan();changed['native_process_control']['mode'] = evidence.MODE
        with self.assertRaises(ValueError):native.policy(changed)
        changed = declared_plan();changed['configured_native_process_control']['helper_hashes']['configured_native_process.py'] = 'a' * 64
        with self.assertRaises(ValueError):native.policy(changed)

    def test_formal_selection_requires_actual_separate_target_references(self):
        """Diagnostic observation and H100 native component records alone cannot enable A4500 profiles."""
        changed = declared_plan(native.PROFILE);changed['configured_isolation'] = {'synthetic': True}
        with self.assertRaises(ValueError):native.policy(changed)
        for kind in ('final_training', 'direct_nsight_process_descriptor_diagnostic', 'nsight_process_identity_diagnostic'):
            changed = declared_plan();changed['kind'] = kind
            with self.subTest(kind=kind), self.assertRaises(ValueError):native.policy(changed)

    def test_exact_source_commands_keep_default_diagnostic_unchanged(self):
        """Only the new kind selects the new standard-library entry and its fixed roles."""
        directory = Path('/synthetic/unit');plan = declared_plan()
        producer = native.producer_command(directory, plan)
        self.assertEqual(producer, [plan['python'], '-I', '-S', '-B', str(directory / 'code/diagnose_configured_native_process.py'), '--prepared-unit', str(directory)])
        waiter, capture = identity.commands(directory, plan)
        self.assertEqual(waiter[-2:], ['--waiter-unit', str(directory)])
        self.assertEqual(capture[-2:], ['--probe-unit', str(directory)])
        self.assertIn('diagnose_configured_native_process.py', waiter[-3])
        old = {'kind': 'nsight_process_identity_diagnostic', 'python': plan['python']}
        self.assertIn('diagnose_nsight_process_tree.py', identity.commands(directory, old)[0][-3])

    def test_runtime_requires_actual_noninherited_provider(self):
        """Reader helpers cannot create a provider or accept an unrelated runtime object."""
        plan = declared_plan();runtime = SimpleNamespace(require_owner=Mock())
        observer = SimpleNamespace(native_process_runtime=runtime);runtime.observer = observer
        self.assertIs(native.runtime_observer(plan, observer), observer)
        runtime.require_owner.assert_called_once_with(plan)
        with self.assertRaises(ValueError):native.runtime_observer(plan, SimpleNamespace())
        runtime.observer = object()
        with self.assertRaises(ValueError):native.runtime_observer(plan, observer)

    def test_scope_custody_is_allocation_and_before_tools(self):
        """The explicit scope receives the provider and deadline without claiming a GPU descriptor."""
        plan = declared_plan();runtime = SimpleNamespace(require_owner=Mock())
        observer = SimpleNamespace(native_process_runtime=runtime);runtime.observer = observer
        scope = SimpleNamespace(OwnedToolScope=Mock(return_value=object()))
        with patch.object(native, 'adjacent', return_value=scope):
            native.tool_scope('/synthetic/unit', plan, observer, {'synthetic_owner': True}, object())
        self.assertEqual(scope.OwnedToolScope.call_args.kwargs['custody_kind'], 'configured_allocation_process')
        self.assertEqual(scope.OwnedToolScope.call_args.kwargs['grace_seconds'], 60)

    def test_saved_policy_import_has_no_native_library_import(self):
        """A fresh pure reader can import both policy helpers while ctypes loading is forbidden."""
        code = 'import sys;sys.path.insert(0,' + repr(str(SCRIPTS)) + ');import ctypes;ctypes.CDLL=lambda *a,**k:(_ for _ in ()).throw(RuntimeError("no native reader"));import configured_native_process,native_process_evidence;print("pure")'
        actual = subprocess.run([sys.executable, '-I', '-S', '-B', '-c', code], capture_output=True, text=True, timeout=10)
        self.assertEqual((actual.returncode, actual.stdout.strip(), actual.stderr), (0, 'pure', ''))

    def test_saved_references_are_mode_specific_and_do_not_admit(self):
        """Successful role files must include the real payload result, acknowledgment and timing boundary."""
        plan = declared_plan()
        self.assertEqual(native.completion_references(plan, 'capture'), ('capture_release_observed.json', 'native_capture_payload_timing.json', 'probe_result.json'))
        plan = declared_plan(native.PROFILE)
        self.assertEqual(native.completion_references(plan, 'capture')[-1], 'capture_result.json')
        self.assertEqual(native.completion_references(plan, 'producer'), ('tool_descendant_scope_terminal.json', 'tool_descendant_scope_restored.json', 'profiler_actual_wait.json'))
        with self.assertRaises(ValueError):native.completion_references(plan, 'another_process')

    def test_producer_bootstrap_precedes_any_tool_scope(self):
        """Execution main starts its own role before producer can enter the scoped tool command."""
        source = ast.parse((SCRIPTS / 'diagnose_configured_native_process.py').read_text())
        main = next(node for node in source.body if isinstance(node, ast.FunctionDef) and node.name == 'main')
        bootstrap = [node.lineno for node in ast.walk(main) if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == 'bootstrap']
        producer = [node.lineno for node in ast.walk(main) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == 'producer']
        self.assertEqual(len(bootstrap), 1);self.assertEqual(len(producer), 1);self.assertLess(bootstrap[0], producer[0])

    def test_actual_cpu_native_scope_closes_exact_child_without_gpu_descriptor(self):
        """Use one fresh owned CPU child to verify native descriptor closure and actual ECHILD."""
        code = '''import json,os,pathlib,sys,time
sys.path.insert(0,SOURCE)
import nsight_process_identity as identity
import linux_pidfd_process_control as native
import direct_tool_descendants as tools
root=pathlib.Path(OUTPUT)
contract,unused=native.current_contract()
provider=native.NativePidfdProvider(contract)
observer=native.NativeProcessObserver(identity,provider)
owner=identity.ProcReader().snapshot(os.getpid(),os.getuid())
plan={'profile_tool':{'path':sys.executable},'configured_native_process_control':{'fixture_only':True}}
scope=tools.OwnedToolScope(root,plan,observer,owner,lambda:{'standing_gpu_descriptor_held':False,'explicit_cpu_fixture':True},grace_seconds=1,custody_kind='configured_allocation_process')
with scope:
    child=scope.spawn([sys.executable,'-I','-S','-B','-c','import time;time.sleep(.05)'])
    child.wait(timeout=5)
    scope.record_wait(child)
terminal=json.loads((root/'tool_descendant_scope_terminal.json').read_text())
assert terminal['terminal']['actual_echild'] is True
assert terminal['all_tool_descendants_exited_before_allocation_custody_release'] is True
assert 'all_tool_descendants_exited_before_descriptor_release' not in terminal
assert provider.evidence()['open_handle_count']==0
assert provider.evidence()['native_call_failure'] is False
print(json.dumps({'actual_echild':True,'retained_handles':0,'nsight_executed':False,'standing_gpu_descriptor_held':False,'scientific_credit':0}))
'''.replace('SOURCE', repr(str(SCRIPTS)))
        with tempfile.TemporaryDirectory() as output:
            actual = subprocess.run([sys.executable, '-I', '-S', '-B', '-c', code.replace('OUTPUT', repr(output))],
                capture_output=True, text=True, timeout=20)
            self.assertEqual(actual.returncode, 0, actual.stderr)
            self.assertEqual(json.loads(actual.stdout), {'actual_echild': True, 'retained_handles': 0,
                'nsight_executed': False, 'standing_gpu_descriptor_held': False, 'scientific_credit': 0})


if __name__ == '__main__':unittest.main()
