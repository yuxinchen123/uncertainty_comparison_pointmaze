"""Check only the changed readiness/hash/namespace and preserved handlers; never submit."""
import ast,ctypes,datetime,fcntl,hashlib,importlib.abc,io,json,os,stat,subprocess,sys,unittest
from pathlib import Path
from unittest.mock import patch
P=Path(__file__).resolve().parent;R=P.parents[1]
V1=R/'remote_setup/2026-11-01-21-40_selected-a4500-profile-held-submission-and-release_copies-1-2_gpu-count-1_processor-count-8_host-memory-gib-64_time-limit-hours-2-3'
assert os.environ.get('SLURM_JOB_ID')=='6783003' and os.environ.get('SLURM_STEP_ID','').isdigit()
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
paths=[p for p in P.rglob('*') if p.is_file()]
before={str(p):sha(p) for p in paths};inputs=json.loads((P/'submission_inputs.json').read_bytes())
protected={row['path']:row['sha256'] for row in inputs['required_files']}
oldfreeze=json.loads((V1/'source_freeze.json').read_bytes())['files']
blocked=[]
class NoNumerical(importlib.abc.MetaPathFinder):
 def find_spec(self,name,path=None,target=None):
  if name.split('.')[0] in {'jax','jaxlib','numpy','torch','flax','optax','jaxtari','cupy','scipy','exploration_platform'}:
   blocked.append(name);raise RuntimeError('numerical import forbidden')
sys.meta_path.insert(0,NoNumerical())
real_popen=subprocess.Popen
class GitOnly(real_popen):
 def __init__(self,args,*a,**k):
  cmd=list(args) if isinstance(args,(list,tuple)) else []
  if not (len(cmd)>3 and cmd[:3]==['git','-C',str(R.parents[2])] and cmd[3] in {'show','cat-file','rev-parse','status','diff','ls-tree','merge-base','symbolic-ref'}):
   blocked.append(cmd);raise RuntimeError('real scheduler,network or child process forbidden')
  super().__init__(args,*a,**k)
subprocess.Popen=GitOnly
ctypes.CDLL=lambda *a,**k:(_ for _ in ()).throw(RuntimeError('shared-library loading forbidden'))
def audit(event,args):
 if event in {'socket.connect','socket.bind','socket.getaddrinfo','os.system','os.kill','os.killpg'}:
  blocked.append(event);raise RuntimeError('network or process action forbidden')
sys.addaudithook(audit)
sys.path.insert(0,str(P))
import test_submit_selected_profiles_held as held
import test_profile_publication as publication
import test_release_selected_profiles as release_tests
S=held.submit
class ReadinessRevisionTests(unittest.TestCase):
 def test_actual_new_readiness_and_both_profiles_pass(self):
  for copies in (1,2):
   raw,item,plan=held.REAL_VALIDATE_INPUTS(copies);ready=S.validate_readiness(raw)
   proof=json.loads(Path(ready['full_verification']['path']).read_bytes())
   self.assertEqual(proof['step_id'],'140')
   self.assertEqual(proof['checked_at_utc'],'2026-11-02T05:55:43.452701+00:00')
   self.assertEqual(plan['profile_updates'],1);self.assertIsNone(plan['gpu_uuid'])
 def test_expired_version_one_readiness_is_refused(self):
  old=json.loads((V1/'submission_inputs.json').read_bytes())
  with self.assertRaisesRegex(ValueError,'stale'):S.validate_readiness(old)
 def test_new_inputs_hash_and_handler_ast_match(self):
  self.assertEqual(S.INPUTS_SHA,sha(P/'submission_inputs.json'))
  def normalized(path):
   tree=ast.parse(path.read_bytes())
   for node in tree.body:
    if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='INPUTS_SHA' for t in node.targets):node.value=ast.Constant(value='input hash')
   return ast.dump(tree,include_attributes=False)
  self.assertEqual(normalized(V1/'submit_selected_profiles_held.py'),normalized(P/'submit_selected_profiles_held.py'))
  self.assertEqual((V1/'release_selected_profiles.py').read_bytes(),(P/'release_selected_profiles.py').read_bytes())
 def test_unbound_release_remains_refused_before_any_query(self):
  with patch.object(release_tests.release.socket,'gethostname',return_value='portal05'),patch.object(release_tests.release.os,'getuid',return_value=2618919),patch.object(release_tests.release.subprocess,'run') as query:
   with self.assertRaisesRegex(ValueError,'not frozen'):release_tests.release.checked(1)
   query.assert_not_called()
with (R/'queue/cpu_monitor_large_artifact_read.lock').open('r+') as lock:
 st=os.fstat(lock.fileno());assert stat.S_ISREG(st.st_mode) and st.st_uid==os.getuid();fcntl.flock(lock.fileno(),fcntl.LOCK_EX)
 assert all(sha(Path(p))==h for p,h in protected.items())
 suite=unittest.TestSuite([unittest.defaultTestLoader.loadTestsFromTestCase(ReadinessRevisionTests),unittest.defaultTestLoader.loadTestsFromTestCase(publication.PublicationTests)])
 output=io.StringIO();result=unittest.TextTestRunner(stream=output,verbosity=2).run(suite)
 assert all(sha(Path(p))==h for p,h in before.items())
 assert all(sha(Path(p))==v['sha256'] for p,v in oldfreeze.items())
 assert all(sha(Path(p))==h for p,h in protected.items()) and not blocked
prefix='readiness_revision_author_6783003_'+os.environ['SLURM_STEP_ID']
def put(name,raw):
 p=P/name
 with p.open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
 return {'path':str(p),'sha256':sha(p)}
raw=put(prefix+'.raw',output.getvalue().encode())
value={'schema_version':1,'kind':'selected_profile_readiness_only_revision_author_controls','passed':result.wasSuccessful(),
 'at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'job_id':'6783003','step_id':os.environ['SLURM_STEP_ID'],'pid':os.getpid(),
 'birth':Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19],'cgroup':Path('/proc/self/cgroup').read_text(),
 'tests':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),'raw':raw,
 'error_details':[{'test':test.id(),'traceback':message} for test,message in result.errors],
 'failure_details':[{'test':test.id(),'traceback':message} for test,message in result.failures],
 'source_before':before,'source_after':{str(p):sha(p) for p in paths},'protected_files':len(protected),'protected_bytes_unchanged':True,
 'V1_frozen_bytes_unchanged':True,'actual_gpu_operations':0,'actual_scheduler_queries':0,'actual_submission_calls':0,'actual_release_calls':0,
 'actual_current_monitor_step':'140','native_task':'213f2e00-68e6-40e5-8e91-439d6ebb8239','expires_at_utc':'2026-11-02T06:15:43.452701+00:00'}
ref=put(prefix+'.json',(json.dumps(value,indent=2,sort_keys=True)+'\n').encode())
print(json.dumps({'passed':value['passed'],'tests':result.testsRun,'receipt':ref,'step_id':os.environ['SLURM_STEP_ID']}))
raise SystemExit(0 if result.wasSuccessful() else 1)
