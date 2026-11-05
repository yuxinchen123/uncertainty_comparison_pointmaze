"""Metadata-only plumbing fixtures. All target/source/tool successes below are synthetic premises."""
import ast
from contextlib import contextmanager, nullcontext
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

SCRIPTS=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(SCRIPTS))
import profile_selected as profile


class NativeProfilePlumbingTests(unittest.TestCase):
    def test_core_direct_route_and_admission_stay_closed(self):
        """The exact published capture core and H100 branch retain their existing syntax."""
        previous=SCRIPTS.parents[2]/'.review_metadata/profile_selected_native_slice/profile_selected_published_e206.py'
        baseline=ast.parse(previous.read_bytes());current=ast.parse((SCRIPTS/'profile_selected.py').read_bytes())
        def node(tree,name):return next(value for value in tree.body if isinstance(value,ast.FunctionDef) and value.name==name)
        self.assertEqual(ast.dump(node(baseline,'_capture_core'),include_attributes=False),ast.dump(node(current,'_capture_core'),include_attributes=False))
        self.assertEqual(ast.dump(node(baseline,'require_reviewed_profile_process_admission'),include_attributes=False),
            ast.dump(node(current,'require_reviewed_profile_process_admission'),include_attributes=False))
        for tree in (baseline,current):
            branch=next(value for value in node(tree,'required_helpers').body if isinstance(value,ast.If)
                and isinstance(value.test,ast.Name) and value.test.id=='direct_profile')
            if tree is baseline:expected=ast.dump(branch,include_attributes=False)
            else:self.assertEqual(ast.dump(branch,include_attributes=False),expected)
        with self.assertRaises(ValueError):profile.require_reviewed_profile_process_admission()
        args=SimpleNamespace(configured_isolation_profile=Path('/synthetic/configured.json'),direct_binding_profile=None)
        with patch.object(profile,'configured_native_reader',side_effect=AssertionError('late helper touched')):
            with self.assertRaises(ValueError):profile.prepare(args)

    def test_default_and_explicit_native_helper_selection(self):
        """Only an explicit configured owned-process declaration can stage native dependencies."""
        with patch.object(profile,'configured_native_reader',side_effect=AssertionError('implicit native import')):
            self.assertNotIn('configured_native_process.py',profile.required_helpers())
            for args in ((False,False,False,False),(False,True,False,False),(True,False,False,False)):
                with self.subTest(args=args),self.assertRaises(ValueError):profile.required_helpers(*args,configured_native=True)
        native=SimpleNamespace(HELPERS=('configured_native_process.py','linux_pidfd_process_control.py','native_process_evidence.py'))
        with patch.object(profile,'configured_native_reader',return_value=native):
            ordinary=profile.required_helpers(True,False,False,True)
            added=profile.required_helpers(True,False,False,True,configured_native=True)
        self.assertEqual(set(added),set(ordinary)|set(native.HELPERS))
        self.assertIsNone(profile.configured_native_reader({'helpers':{'files':{}}}))
        with self.assertRaises(ValueError):profile.configured_native_reader({'helpers':{'files':{
            '10_jax_exploration_platform/scripts/atari_pixel_campaign/configured_native_process.py':'0'*64}}})

    def producer_fixture(self, fail_version=False):
        """Exercise real producer control flow with unavailable qualification/tool work explicitly stubbed."""
        temporary=tempfile.TemporaryDirectory();self.addCleanup(temporary.cleanup);directory=Path(temporary.name)
        events=[];plan={'asset_record':{'synthetic':True},'profile_tool':{'path':'/synthetic/nsys'},'profile_updates':1}
        deadline=SimpleNamespace(remaining=lambda phase='work':240,snapshot=lambda:{'fixture_clock_only':True})
        owner={'command':['synthetic-producer-only']}
        observer=SimpleNamespace(read_json=lambda path:{'synthetic_upstream_record':str(path)},
            validate_saved=lambda *args:events.append('base-saved-reader'))
        active=SimpleNamespace(observer=observer,owner=owner,role='producer')
        scope=SimpleNamespace(active=False,observer=observer,plan=plan,deadline=deadline)
        @contextmanager
        def tool_scope(*args,**kwargs):
            self.assertIn('bootstrap',events);scope.active=True;events.append('scope-enter')
            try:yield scope
            finally:scope.active=False;events.append('scope-closed')
        @contextmanager
        def card(*args):
            events.append('allocation-enter')
            try:yield None
            finally:events.append('allocation-exit')
        def complete(refs):
            self.assertFalse(scope.active);self.assertIn('scope-closed',events);events.append('terminal')
        def failure(error):
            self.assertFalse(scope.active);events.append('failure-after-closure')
        active.complete=complete;active.failure=failure
        def bootstrap(*args,**kwargs):events.append('bootstrap');return active
        def origin(*args,**kwargs):events.append('origin');return deadline
        native=SimpleNamespace(context=lambda *args:owner,producer_command=lambda *args:owner['command'],deadline=origin,
            bootstrap=bootstrap,tool_scope=tool_scope,completion_references=lambda *args:('synthetic-refs',),
            validate_saved=lambda *args:{'synthetic_closed_native_proof':True},saved_deadline=lambda *args:deadline)
        def version(directory,actual,actual_scope,actual_deadline,phase):
            self.assertTrue(actual_scope is scope and actual_scope.active and actual_deadline is deadline)
            events.append('version-'+phase)
            if fail_version:raise RuntimeError('synthetic tool failure')
        def capture(directory,actual,provider,actual_scope,actual_deadline):
            self.assertIs(provider,active);self.assertTrue(actual_scope is scope and actual_scope.active)
            self.assertIs(actual_deadline,deadline);events.append('capture-and-export')
            return {'profile_updates':1,'synthetic_payload_only':True}
        support=SimpleNamespace(core=lambda:SimpleNamespace(collect_metadata=lambda *args:{'synthetic':True}),
            verify_qualification_inputs=lambda *args:events.append('original-qualification'))
        patches=[patch.object(profile,'configured_native_reader',return_value=native),patch.object(profile,'process_reader',return_value=observer),
            patch.object(profile,'interruption_cleanup',return_value=nullcontext()),patch.object(profile.measurement,'card_admission',side_effect=card),
            patch.object(profile.measurement,'producer_support',return_value=support),patch.object(profile,'validate_measured_baseline',side_effect=lambda *a:events.append('original-baseline')),
            patch.object(profile.training,'verify_assets',return_value={'synthetic':True}),patch.object(profile.training,'jaxtari_python_receipt',return_value={'synthetic':True}),
            patch.object(profile.training,'installed_gpu_library_receipts',return_value={'synthetic':True}),patch.object(profile,'configured_native_tool_record',side_effect=version),
            patch.object(profile,'capture_configured_native_processes',side_effect=capture),patch.object(profile.diagnostic,'record_progress')]
        from contextlib import ExitStack
        with ExitStack() as stack:
            for item in patches:stack.enter_context(item)
            if fail_version:
                with self.assertRaisesRegex(RuntimeError,'synthetic tool failure'):profile.run_configured_native_prepared(directory,plan,{})
                self.assertFalse((directory/'result.json').exists())
            else:
                result=profile.run_configured_native_prepared(directory,plan,{})
                self.assertEqual(result['configured_native_process_proof'],{'synthetic_closed_native_proof':True})
        return events

    def test_one_producer_bootstrap_precedes_scope_and_publication_follows_close(self):
        events=self.producer_fixture()
        self.assertEqual(events.count('bootstrap'),1)
        for first,second in (('origin','bootstrap'),('bootstrap','scope-enter'),('scope-enter','version-before'),
                ('version-before','capture-and-export'),('capture-and-export','scope-closed'),('scope-closed','terminal'),('terminal','base-saved-reader')):
            self.assertLess(events.index(first),events.index(second))

    def test_failed_first_version_closes_scope_before_allocation_and_failure(self):
        events=self.producer_fixture(fail_version=True)
        self.assertNotIn('capture-and-export',events);self.assertNotIn('terminal',events)
        self.assertLess(events.index('scope-closed'),events.index('allocation-exit'))
        self.assertLess(events.index('allocation-exit'),events.index('failure-after-closure'))

    def test_all_tools_use_scope_and_numerical_timing_is_an_outer_record(self):
        """Source-bound interception prevents an unscoped export or Popen fallback in the new route."""
        tree=ast.parse((SCRIPTS/'profile_selected.py').read_bytes())
        nodes={value.name:value for value in tree.body if isinstance(value,ast.FunctionDef)}
        version=ast.unparse(nodes['configured_native_tool_record']);capture=ast.unparse(nodes['capture_configured_native_processes'])
        self.assertIn('scope.run(',version);self.assertIn('scope.spawn(',capture);self.assertIn('scope.wait(',capture)
        self.assertIn('run_command=scope.run',capture)
        for name in ('run_configured_native_prepared','capture_configured_native_processes','run_configured_native_waiter','capture_configured_native_child'):
            text=ast.unparse(nodes[name])
            for forbidden in ('os.kill(', 'os.killpg(', '.terminate(', '.kill(', 'os.pidfd_open(', 'signal.pidfd_send_signal('):
                self.assertNotIn(forbidden,text,name)
        waiter=ast.unparse(nodes['run_configured_native_waiter'])
        self.assertIn('observer.open_process_handle(child.pid)',waiter);self.assertIn('observer.close_process_handle(child_handle)',waiter)
        self.assertIn('observer.close_process_handle(parent)',waiter)
        child=ast.unparse(nodes['capture_configured_native_child'])
        self.assertIn("'capture_release_observed.json'",child)
        self.assertIn("'native_capture_payload_timing.json'",child)
        self.assertEqual(sum(isinstance(value,ast.Call) and isinstance(value.func,ast.Name) and value.func.id=='_capture_core'
            for value in ast.walk(nodes['capture_configured_native_child'])),1)


if __name__=='__main__':unittest.main()
