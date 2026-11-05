"""Check corrected child closure and release evidence using owned CPU fixtures only."""
import ast
import copy
import hashlib
import io
import json
import os
from pathlib import Path
import socket
import sys
import time
import unittest

W=Path(__file__).resolve().parents[1]
S=W/'10_jax_exploration_platform/scripts/atari_pixel_campaign';R=W/'.review_metadata'
F=R/'direct_nsight_producer_source_freeze_author_4_release_and_kernel.json'
frozen=json.loads(F.read_text())
assert hashlib.sha256(F.read_bytes()).hexdigest()=='aa16852710105c6e5adb55375906203780e8b508376eb90910fedc75c7d98f3a'
assert os.environ['SLURM_JOB_ID']=='6783003' and os.environ['SLURM_STEP_ID'].isdecimal()
assert all(hashlib.sha256((W/p).read_bytes()).hexdigest()==h for p,h in frozen['files'].items())
label='direct_nsight_corrected_independent_'+os.environ['SLURM_JOB_ID']+'_'+os.environ['SLURM_STEP_ID']
actual=R/('direct_tool_scope_independent_cpu_'+os.environ['SLURM_JOB_ID']+'_'+os.environ['SLURM_STEP_ID'])
actual.mkdir()
os.environ['DIRECT_CPU_SCOPE_REVIEW_OUTPUT']=str(actual)
sys.path.insert(0,str(S/'tests'));sys.path.insert(0,str(S))
import direct_tool_descendants as scope_module
import direct_profile_process as direct
import test_direct_profile_process as fixtures
from types import SimpleNamespace

started=time.time();output=io.StringIO();suite=unittest.TestSuite()
for name in ('test_direct_tool_descendants','test_direct_capture_release.DirectCaptureReleaseTests','test_direct_profile_process.DirectSavedProcessTests'):
    suite.addTests(unittest.defaultTestLoader.loadTestsFromName(name))
result=unittest.TextTestRunner(stream=output,verbosity=2).run(suite)

# Synthetic interleaving only. No operating-system PID below is inspected or signalled.
# A known non-producer parent can exit/reap after handle_exited returned false.
# Its numeric PID then denotes another same-UID process before child-list inspection.
synthetic=scope_module.OwnedToolScope.__new__(scope_module.OwnedToolScope)
synthetic.pid=100;synthetic.uid=888;synthetic.cgroup='synthetic common cgroup'
synthetic.handles={101:'old-parent-pidfd'}
synthetic.nodes={101:{'initial':{'process_start_time':'20'},'last':{},'positive_parent_pid':100}}
synthetic.events=[];synthetic.failed=False
synthetic.observer=SimpleNamespace(handle_exited=lambda handle:False,open_process_handle=lambda pid:'new-child-pidfd',close_process_handle=lambda handle:None)
observations=[]
def lists(pid):
    """Represent a parent PID reused after its earlier non-exited process-handle check."""
    if pid==100:return {101},[{'fixture':'producer positive child list before parent exit'}]
    if pid==101:
        observations.append({'parent_pid':101,'actual_parent_birth_now':'99','pinned_parent_birth':'20','foreign_same_uid_child':303})
        return {303},[{'fixture':'new unrelated same-UID parent child list'}]
    return set(),[{'fixture':'no children'}]
def snapshot(pid):
    """Keep the actual child UID/cgroup/parent checks true without a parent-birth check."""
    if pid==101:return {'pid':101,'parent_pid':100,'process_start_time':'20','state':'S'}
    return {'pid':303,'parent_pid':101,'process_start_time':'100','state':'S','uids':[888]*4,'cgroup':synthetic.cgroup}
synthetic.child_lists=lists;synthetic.snapshot=snapshot
synthetic.scan()
parent_birth_control={'kind':'synthetic_parent_pid_reuse_interleaving','actual_processes_touched':0,
    'observations':observations,'unrelated_same_uid_child_was_pinned':303 in synthetic.handles,
    'source_requires_expected_parent_birth_in_child_lists':False,
    'scope':'A source-level interleaving control, not a claim that actual PID reuse happened in this CPU step.'}

fixture=fixtures.DirectSavedProcessTests();fixture.setUp()
try:
    unit,plan=fixture.saved()
    path=unit/'tool_descendant_scope_restored.json'
    value=fixture.original(plan).read_json(path)
    value['restore']['value']=1
    path.unlink();fixture.original(plan).publish(path,value)
    proof=direct.validate_saved(unit,plan)
    restore_control={'kind':'synthetic_inconsistent_restored_prctl_parameter','previous_value':0,
        'restore_set_value':1,'verified_get_value':0,'saved_reader_accepted':proof.get('passed') is True,
        'actual_process_control_performed':False}
finally:fixture.doCleanups()

unchanged=all(hashlib.sha256((W/p).read_bytes()).hexdigest()==h for p,h in frozen['files'].items())
raw=output.getvalue().encode();(R/(label+'_unittest.raw')).write_bytes(raw)
controls={'parent_birth_control':parent_birth_control,'restore_control':restore_control}
(R/(label+'_extra_controls.json')).write_text(json.dumps(controls,indent=2)+'\n')
files=[]
for p in sorted(actual.rglob('*')):
    if p.is_file():files.append({'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'size_bytes':p.stat().st_size})
record={'kind':'independent_corrected_direct_nsight_review','source_freeze':{'path':str(F),'sha256':hashlib.sha256(F.read_bytes()).hexdigest()},
    'base':frozen['base'],'job_id':os.environ['SLURM_JOB_ID'],'step_id':os.environ['SLURM_STEP_ID'],'hostname':socket.gethostname(),
    'pid':os.getpid(),'started_at_seconds':started,'finished_at_seconds':time.time(),'focused_tests':result.testsRun,
    'failures':len(result.failures),'errors':len(result.errors),'source_unchanged_before_and_after':unchanged,
    'unittest_output':{'path':str(R/(label+'_unittest.raw')),'sha256':hashlib.sha256(raw).hexdigest()},
    'extra_controls':{'path':str(R/(label+'_extra_controls.json')),'sha256':hashlib.sha256((R/(label+'_extra_controls.json')).read_bytes()).hexdigest()},
    'actual_cpu_artifacts':files,'actual_tool_or_gpu_execution':False,'standing_gpu_lock_opened':False,
    'gpu_operations':0,'scientific_credit':0,'admission_granted':False,'approved':False}
(R/(label+'.json')).write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps({'receipt':str(R/(label+'.json')),'tests':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),
    'source_unchanged':unchanged,'synthetic_foreign_child_pinned':parent_birth_control['unrelated_same_uid_child_was_pinned'],
    'inconsistent_restore_accepted':restore_control['saved_reader_accepted'],'actual_cpu_artifacts':len(files)}))
assert result.wasSuccessful() and unchanged
