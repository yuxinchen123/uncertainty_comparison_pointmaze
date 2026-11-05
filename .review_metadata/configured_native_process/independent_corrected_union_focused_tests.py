import hashlib,io,json,os,subprocess,sys,tempfile,time,unittest
from pathlib import Path
W=Path('/p/rlprojects/RND_worktrees/2026-11-05-06-20_a4500-native-process-control_python-3.12_total-seconds-300_work-seconds-240_cleanup-seconds-60_admission-closed')
D=W/'.review_metadata/configured_native_process';S=W/'10_jax_exploration_platform/scripts/atari_pixel_campaign'
R=Path('/p/rlprojects/RND_worktrees/merge-atari-cleanrl/10_jax_exploration_platform/benchmark_runs/2026-10-31-00-11_jaxatari_montezuma_pixels_cleanrl_ppo_rnd_environments-per-copy-128_rollout-steps-128_ideas-30-plus-20_steps-per-copy-2000000000_seed-start-1')
assert os.environ['SLURM_JOB_ID']=='6821437'
assert '6821437' in (R/'slurm/submitted_jobids_uva.txt').read_text().splitlines()
freeze=D/'corrected_union_source_freeze.json'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert sha(freeze)=='36524765c7fba07fad274b0ee170d03df05ead3ea2a1217ffbbfb22ecd84dd39'
f=json.loads(freeze.read_bytes())
def verify():
 for rel,row in f['files'].items():
  p=W/rel;assert sha(p)==row['sha256'];assert oct(p.stat().st_mode & 0o777)==row['mode']
verify();sys.path[:0]=[str(S),str(S/'tests')]
import test_configured_native_process as policy_tests
import test_configured_native_saved_roles as role_tests
import test_configured_native_profile_plumbing as profile_tests
class IndependentSavedRefusals(unittest.TestCase):
 def fixture(self):
  t=role_tests.ConfiguredNativeSavedRoleTests('runTest');t.setUp();self.addCleanup(t.doCleanups);return t
 def exit_event(self,t,index=-1):
  f=t.fixture;event=f.terminals['waiter']['provider']['descriptor_events'][index]
  raw=event['identity']['raw_fdinfo'];old=event['identity']['pid'];raw=raw.replace('Pid:\t'+str(old)+'\n','Pid:\t-1\n')
  event['identity'].update(pid=-1,raw_fdinfo=raw,fdinfo_sha256=hashlib.sha256(raw.encode()).hexdigest())
  f.save(role_tests.evidence.role_files(f.directory,'waiter')['terminal'].name,f.terminals['waiter']);return event
 def test_nonzero_actual_wait_cannot_justify_exited_capture_close(self):
  t=self.fixture();self.exit_event(t);t.fixture.waited['returncode']=1
  with self.assertRaises(ValueError):t.read_saved()
 def test_boolean_wait_status_cannot_justify_exited_capture_close(self):
  t=self.fixture();self.exit_event(t);t.fixture.waited['returncode']=False
  with self.assertRaises(ValueError):t.read_saved()
 def test_intervention_cannot_justify_exited_capture_close(self):
  t=self.fixture();self.exit_event(t);t.fixture.waited['interventions']=['synthetic intervention']
  with self.assertRaises(ValueError):t.read_saved()
 def test_exited_producer_watcher_stays_refused(self):
  t=self.fixture();self.exit_event(t,index=-2)
  with self.assertRaises(ValueError):t.read_saved()
 def test_exited_capture_open_stays_refused(self):
  t=self.fixture();f=t.fixture;events=f.terminals['waiter']['provider']['descriptor_events']
  index=next(i for i,event in enumerate(events) if event['operation']=='pidfd_open' and event['opened_pid']==f.owners['capture']['pid'])
  self.exit_event(t,index=index)
  with self.assertRaises(ValueError):t.read_saved()
 def test_sigterm_enters_actual_owned_cpu_child_cleanup(self):
  code='''import json,os,pathlib,signal,sys,time
sys.path.insert(0,SOURCE)
import nsight_process_identity as identity
import linux_pidfd_process_control as native
import direct_tool_descendants as tools
import diagnose_configured_native_process as diagnostic
root=pathlib.Path(OUTPUT)
contract,unused=native.current_contract();provider=native.NativePidfdProvider(contract)
observer=native.NativeProcessObserver(identity,provider)
owner=identity.ProcReader().snapshot(os.getpid(),os.getuid())
plan={'profile_tool':{'path':sys.executable},'configured_native_process_control':{'fixture_only':True}}
scope=tools.OwnedToolScope(root,plan,observer,owner,lambda:{'standing_gpu_descriptor_held':False,'explicit_cpu_fixture':True},grace_seconds=1,custody_kind='configured_allocation_process')
original=signal.getsignal(signal.SIGTERM)
try:
 with diagnostic.producer_interruptions(),scope:
  child=scope.spawn([sys.executable,'-I','-S','-B','-c','import time;time.sleep(20)'])
  os.kill(os.getpid(),signal.SIGTERM)
except InterruptedError:pass
else:raise AssertionError('real signal did not enter cleanup')
terminal=json.loads((root/'tool_descendant_scope_terminal.json').read_bytes())
assert terminal['terminal']['actual_echild'] is True
assert terminal['terminal']['all_actual_tool_roots_waited'] is True
assert terminal['interventions']
assert provider.evidence()['open_handle_count']==0
assert signal.getsignal(signal.SIGTERM)==original
print(json.dumps({'actual_echild':True,'actual_roots_waited':True,'retained_native_handles':0,'native_fixture_only':True,'tool_executed':False,'numerical_operations':False,'gpu_operations':False,'scientific_credit':0}))
'''.replace('SOURCE',repr(str(S)))
  with tempfile.TemporaryDirectory() as temporary:
   q=subprocess.run([sys.executable,'-I','-S','-B','-c',code.replace('OUTPUT',repr(temporary))],capture_output=True,text=True,timeout=30)
   self.assertEqual(q.returncode,0,q.stderr);record=json.loads(q.stdout)
   self.assertTrue(record['actual_echild']);self.assertEqual(record['retained_native_handles'],0)
suite=unittest.TestSuite()
for cls in (policy_tests.ConfiguredNativePolicyTests,role_tests.ConfiguredNativeSavedRoleTests,profile_tests.NativeProfilePlumbingTests,IndependentSavedRefusals):suite.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(cls))
start=time.monotonic();stream=io.StringIO();result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite);verify()
record={'kind':'independent_configured_native_union_focused_tests','passed':result.wasSuccessful(),'source_freeze':{'path':str(freeze),'sha256':sha(freeze)},'test_count':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),'output':stream.getvalue(),'seconds':time.monotonic()-start,'job_id':os.environ['SLURM_JOB_ID'],'step_id':os.environ['SLURM_STEP_ID'],'uid':os.getuid(),'pid':os.getpid(),'cgroup':Path('/proc/self/cgroup').read_text(),'actual_target_qualification':False,'formal_profile_admission':False,'scientific_credit':0,'scope':'pure metadata fixtures plus explicitly declared fresh own-CPU-child libc/prctl/pidfd closure fixtures; no tool, GPU, or numerical operation; no target actions'}
p=D/'independent_corrected_union_focused_tests.json'
with p.open('x') as out:json.dump(record,out,indent=2);out.write('\n');out.flush();os.fsync(out.fileno())
print(stream.getvalue());print(json.dumps({'review_path':str(p),'review_sha256':sha(p),'test_count':result.testsRun,'passed':result.wasSuccessful(),'seconds':record['seconds'],'job_id':record['job_id'],'step_id':record['step_id']}))
raise SystemExit(0 if result.wasSuccessful() else 1)
