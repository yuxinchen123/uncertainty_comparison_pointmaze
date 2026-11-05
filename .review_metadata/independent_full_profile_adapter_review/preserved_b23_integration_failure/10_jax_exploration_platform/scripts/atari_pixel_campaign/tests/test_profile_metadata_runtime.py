"""Bounded own-CPU metadata fixtures; no learner, tool, standing descriptor or GPU work."""
import ast
from copy import deepcopy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import sys
import tempfile
import time
import unittest
from unittest.mock import patch
SCRIPTS=Path(__file__).resolve().parents[1]
def load(name):
    path=SCRIPTS/(name+'.py');spec=importlib.util.spec_from_file_location('fixture_'+name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module
runtime=load('profile_metadata_runtime')

def plan():
    return {'helpers':{'files':{runtime.PREFIX+'nsight_process_identity.py':hashlib.sha256((SCRIPTS/'nsight_process_identity.py').read_bytes()).hexdigest()}}}

def owner():
    return {'pid':os.getpid(),'parent_pid':os.getppid(),'process_start_time':Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19],
        'uids':[os.getuid()]*4,'argv':['clearly_synthetic_CPU_fixture'],'hostname':runtime.socket.gethostname().split('.')[0],
        'cgroup':Path('/proc/self/cgroup').read_text()}

class SharedBudgetChecks(unittest.TestCase):
    def test_one_origin_does_not_restart_between_roles(self):
        p=plan();start=time.monotonic();origin=runtime.origin(p,owner(),started=start);clock=[start+239]
        b=runtime.Budget(origin,p,clock=lambda:clock[0]);self.assertAlmostEqual(b.remaining(),1)
        clone=runtime.Budget(deepcopy(origin),p,clock=lambda:clock[0]);self.assertAlmostEqual(clone.remaining(),1)
        clock[0]=start+240
        with self.assertRaises(ValueError):clone.remaining()
        clone.begin_cleanup();self.assertAlmostEqual(clone.remaining('cleanup'),60)
        clock[0]=start+300
        with self.assertRaises(ValueError):clone.remaining('cleanup')
        self.assertEqual(clone.remaining('cleanup',allow_zero=True),0)
    def test_changed_schema_plan_host_and_phase_cannot_reset_budget(self):
        p=plan();origin=runtime.origin(p,owner())
        for key,value in [('schema_version',True),('total_end_monotonic',origin['total_end_monotonic']+1),('plan_sha256','0'*64)]:
            value_origin=deepcopy(origin);value_origin[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):runtime.Budget(value_origin,p)
        value_origin=deepcopy(origin);value_origin['clock_domain']['hostname']='other-host'
        with self.assertRaises(ValueError):runtime.Budget(value_origin,p)
        budget=runtime.Budget(origin,p)
        record=budget.snapshot();record['observed_monotonic']=origin['total_end_monotonic']+0.001
        with self.assertRaises(ValueError):budget.verify(record,cleanup=True)
    def test_missing_pidfd_exports_refuse_before_native_or_spawn(self):
        with patch.object(runtime.os,'pidfd_open',None),patch.object(runtime,'prepare_native_control',side_effect=AssertionError('nativebootstrap')),\
                patch.object(runtime.subprocess,'Popen',side_effect=AssertionError('Popen')):
            with self.assertRaises(ValueError):runtime.MetadataScope(plan(),runtime.Budget(runtime.origin(plan(),owner()),plan()))
    def test_empty_baseline_and_exact_own_CPU_child_close(self):
        p=plan();budget=runtime.Budget(runtime.origin(p,owner()),p)
        with runtime.MetadataScope(p,budget) as scope:
            result=scope.run(['/usr/bin/python3','-I','-S','-B','-c','print("metadata CPU fixture")'],timeout=5)
            self.assertEqual(result.returncode,0);self.assertEqual(result.stdout,b'metadata CPU fixture\n')
        self.assertTrue(scope.proof['all_metadata_descendants_exited'])
        runtime.validate_scope(scope.proof,budget,scope.observer,scope.owner)
        bad=deepcopy(scope.proof);bad['verified_restore']['value']=1-bad['baseline']['previous_subreaper']['value']
        with self.assertRaises(ValueError):runtime.validate_scope(bad,budget,scope.observer,scope.owner)
    def test_failed_own_CPU_child_is_waited_without_false_success(self):
        p=plan();budget=runtime.Budget(runtime.origin(p,owner()),p)
        with runtime.MetadataScope(p,budget) as scope:
            result=scope.run(['/usr/bin/python3','-I','-S','-B','-c','raise SystemExit(7)'],timeout=5)
            self.assertEqual(result.returncode,7)
        with self.assertRaises(ValueError):runtime.validate_scope(scope.proof,budget,scope.observer,scope.owner)
    def test_saved_scope_rejects_missing_wait_hash_parent_and_types(self):
        p=plan();budget=runtime.Budget(runtime.origin(p,owner()),p)
        with runtime.MetadataScope(p,budget) as scope:
            scope.run(['/usr/bin/python3','-I','-S','-B','-c','import sys;sys.stdin.buffer.read();print("own CPU")'],input=b'release',timeout=5)
        source=scope.proof;key=next(iter(source['positive_nodes']))
        changes=[lambda x:x['root_waits'].clear(),
            lambda x:x['positive_nodes'][key]['initial'].__setitem__('command_raw_sha256','0'*64),
            lambda x:x['positive_nodes'][key].__setitem__('positive_parent_pid',int(key)),
            lambda x:x.__setitem__('scientific_credit',False),
            lambda x:x['positive_nodes'][key]['initial'].__setitem__('command_observed',False),
            lambda x:x['positive_nodes'][key]['last'].__setitem__('command_raw_hex','00')]
        for change in changes:
            changed=deepcopy(source);change(changed)
            with self.subTest(change=repr(change)),self.assertRaises((ValueError,KeyError)):
                runtime.validate_scope(changed,budget,scope.observer,scope.owner)
    def test_own_double_orphan_is_reaped_before_failed_scope_returns(self):
        p=plan();budget=runtime.Budget(runtime.origin(p,owner()),p)
        code='''import os,signal,sys,time
sys.stdin.buffer.read()
if os.fork()==0:
    if os.fork()==0:
        signal.signal(signal.SIGTERM,signal.SIG_IGN)
        os.close(1);os.close(2)
        time.sleep(4)
        os._exit(0)
    os._exit(0)
raise SystemExit(7)
'''
        with self.assertRaisesRegex(RuntimeError,'CPU-only requested failure'):
            with runtime.MetadataScope(p,budget) as scope:
                returned=scope.run(['/usr/bin/python3','-I','-S','-B','-c',code],input=b'release',timeout=5)
                self.assertEqual(returned.returncode,7)
                raise RuntimeError('CPU-only requested failure')
        self.assertTrue(scope.proof['terminal']['actual_echild'])
        self.assertTrue(scope.proof['all_metadata_descendants_exited'])
        self.assertFalse(scope.proof['successful_no_intervention'])
        self.assertGreaterEqual(len(scope.proof['positive_nodes']),2)
        with self.assertRaises(ValueError):runtime.validate_scope(scope.proof,budget,scope.observer,scope.owner)
    def test_command_phase_expiry_closes_only_its_actual_own_child(self):
        p=plan();budget=runtime.Budget(runtime.origin(p,owner()),p)
        with self.assertRaisesRegex(ValueError,'bounded phase limit'):
            with runtime.MetadataScope(p,budget) as scope:
                scope.run(['/usr/bin/python3','-I','-S','-B','-c','import sys,time;sys.stdin.buffer.read();time.sleep(10)'],
                    input=b'release',timeout=.01)
        self.assertTrue(scope.proof['terminal']['actual_echild'])
        self.assertTrue(scope.proof['all_metadata_descendants_exited'])
        self.assertFalse(scope.proof['successful_no_intervention'])
        self.assertEqual(len(scope.proof['spawned_roots']),1)
    def test_native_formal_profile_admission_stays_closed(self):
        source=(SCRIPTS/'direct_profile_prerequisites.py').read_bytes();tree=ast.parse(source)
        node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='require_admission')
        self.assertIsInstance(node.body[-1],ast.Raise)
        self.assertEqual(runtime.PROTOCOL['total_seconds'],300)
        self.assertIs(runtime.PROTOCOL['operational_admission'],False)

if __name__=='__main__':unittest.main()

