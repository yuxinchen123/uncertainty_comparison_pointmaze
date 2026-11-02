"""Test fixed actual release binding with private synthetic actions; never call a real scheduler."""
import ctypes,datetime,fcntl,hashlib,importlib.abc,io,json,os,stat,subprocess,sys,unittest
from pathlib import Path
from unittest.mock import patch
P=Path(__file__).resolve().parent;R=P.parents[1]
assert os.environ.get('SLURM_JOB_ID')=='6783003' and os.environ.get('SLURM_STEP_ID','').isdigit()
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
names=('release_selected_profiles.py','test_release_selected_profiles.py','release_frozen_inputs.json','release_author_review.json','run_author_controls.py')
before={name:sha(P/name) for name in names};frozen=json.loads((P/'release_frozen_inputs.json').read_bytes())
protected={**frozen['shared_files'],**frozen['jobs']['1']['files'],**frozen['jobs']['2']['files']}
roots={str(R.parents[2])}
for item in frozen['jobs'].values():
 plan=json.loads((Path(item['unit'])/'measurement_plan.json').read_bytes())
 roots.update((plan['source_root'],plan['helpers']['source_root']))
blocked=[]
class NoNumerical(importlib.abc.MetaPathFinder):
 def find_spec(self,name,path=None,target=None):
  if name.split('.')[0] in {'jax','jaxlib','numpy','torch','flax','optax','jaxtari','cupy','scipy','exploration_platform','pynvml'}:
   blocked.append(name);raise RuntimeError('numerical import forbidden')
sys.meta_path.insert(0,NoNumerical())
real_popen=subprocess.Popen
class GitOnly(real_popen):
 def __init__(self,args,*a,**k):
  cmd=list(args) if isinstance(args,(list,tuple)) else []
  if not (len(cmd)>3 and cmd[:2]==['git','-C'] and cmd[2] in roots and cmd[3] in {'show','cat-file','rev-parse','status','diff','ls-tree','merge-base','symbolic-ref'}):
   blocked.append(cmd);raise RuntimeError('real scheduler,network or child process forbidden')
  super().__init__(args,*a,**k)
subprocess.Popen=GitOnly
ctypes.CDLL=lambda *a,**k:(_ for _ in ()).throw(RuntimeError('shared-library load forbidden'))
def audit(event,args):
 if event in {'socket.connect','socket.bind','socket.getaddrinfo','os.system','os.kill','os.killpg'}:
  blocked.append(event);raise RuntimeError('network or process action forbidden')
sys.addaudithook(audit)
sys.path.insert(0,str(P))
import test_release_selected_profiles as tests
release=tests.release
class ActualBindingPublicationTests(unittest.TestCase):
 def test_both_actual_defaults_and_core_binding_pass_without_queries(self):
  with patch.object(release.socket,'gethostname',return_value='portal05'),patch.object(release.os,'getuid',return_value=2618919):
   for copies,job in ((1,'6783807'),(2,'6783808')):
    intent=release.checked(copies)
    self.assertEqual(intent['job_id'],job);self.assertEqual(intent['command'],['scontrol','release',job])
    self.assertFalse((release.PACKET/'release_intent.json').exists())
 def test_actual_initial_publication_refuses_missing_release_objects(self):
  original=release.subprocess.check_output;observed=[]
  def capture(command,*args,**kwargs):
   try:return original(command,*args,stderr=subprocess.PIPE,**kwargs)
   except subprocess.CalledProcessError as error:
    observed.append({'command':command,'returncode':error.returncode,'stdout':(error.output or b'').decode(errors='strict'),'stderr':(error.stderr or b'').decode(errors='strict')});raise
  with patch.object(release.socket,'gethostname',return_value='portal05'),patch.object(release.os,'getuid',return_value=2618919),patch.object(release.subprocess,'check_output',side_effect=capture):
   with self.assertRaises(subprocess.CalledProcessError):release.checked(1,release.INITIAL_PUBLICATION)
  self.assertEqual(len(observed),1);self.assertEqual(observed[0]['returncode'],128)
  output=P/('actual_unpublished_release_refusal_'+os.environ['SLURM_STEP_ID']+'.json')
  with output.open('x') as f:json.dump({'expected_refusal':True,'actual_read_only_Git_result':observed[0]},f,indent=2);f.write('\n')
 def test_exact_mock_publication_and_changed_object_refusal(self):
  with patch.object(release.socket,'gethostname',return_value='portal05'),patch.object(release.os,'getuid',return_value=2618919):
   release.checked(2)
   selected=release.SELECTED
   paths=[Path(p) for p in {**frozen['shared_files'],**selected['files']}]+[P/'release_selected_profiles.py',P/'release_frozen_inputs.json',P/'release_author_review.json',P/'test_release_selected_profiles.py']
   objects={str(path.relative_to(release.REPOSITORY)):path.read_bytes() for path in paths}
   calls=[]
   def object_bytes(command,*args,**kwargs):
    self.assertEqual(command[:4],['git','-C',str(release.REPOSITORY),'cat-file'])
    calls.append(command);return objects[command[-1].split(':',1)[1]]
   with patch.object(release,'core',return_value=object()),patch.object(release.subprocess,'run') as ancestry,patch.object(release.subprocess,'check_output',side_effect=object_bytes):
    intent=release.checked(2,'a'*40)
    self.assertEqual(intent['job_id'],'6783808')
   self.assertEqual(len(calls),len(paths));ancestry.assert_called_once()
   objects[next(iter(objects))]+=b'\n'
   with patch.object(release,'core',return_value=object()),patch.object(release.subprocess,'run'),patch.object(release.subprocess,'check_output',side_effect=object_bytes):
    with self.assertRaises(ValueError):release.checked(2,'a'*40)
with (R/'queue/cpu_monitor_large_artifact_read.lock').open('r+') as lock:
 st=os.fstat(lock.fileno());assert stat.S_ISREG(st.st_mode) and st.st_uid==os.getuid();fcntl.flock(lock.fileno(),fcntl.LOCK_EX)
 assert all(sha(Path(path))==record['sha256'] for path,record in protected.items())
 suite=unittest.TestSuite([unittest.defaultTestLoader.loadTestsFromModule(tests),unittest.defaultTestLoader.loadTestsFromTestCase(ActualBindingPublicationTests)])
 output=io.StringIO();result=unittest.TextTestRunner(stream=output,verbosity=2).run(suite)
 assert not blocked and before=={name:sha(P/name) for name in names}
 assert all(sha(Path(path))==record['sha256'] for path,record in protected.items())
prefix='release_author_controls_6783003_'+os.environ['SLURM_STEP_ID']
def put(name,raw):
 path=P/name
 with path.open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
 return {'path':str(path),'sha256':sha(path)}
raw=put(prefix+'.raw',output.getvalue().encode())
value={'schema_version':1,'kind':'actual_held_selected_profile_release_metadata_controls','passed':result.wasSuccessful(),
 'at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'job_id':'6783003','step_id':os.environ['SLURM_STEP_ID'],'pid':os.getpid(),
 'birth':Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19],'cgroup':Path('/proc/self/cgroup').read_text(),
 'tests':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),'raw':raw,
 'error_details':[{'test':test.id(),'traceback':message} for test,message in result.errors],
 'failure_details':[{'test':test.id(),'traceback':message} for test,message in result.failures],
 'source_before':before,'source_after':{name:sha(P/name) for name in names},'protected_files':len(protected),
 'protected_bytes_unchanged':True,'actual_local_defaults_and_real_staged_core_passed':True,
 'mock_publication_source_identity_stub_scope':'publication-only loop; actual source identity separately passed both real defaults',
 'actual_scheduler_queries':0,'actual_releases':0,'GPU_operations':0,'numerical_import_attempts':blocked,
 'expiry_utc':min(json.loads(Path(item['held_proof']['path']).read_bytes())['expires_at_utc'] for item in frozen['jobs'].values()),
 'scientific_credit':0,'admission_granted':False}
receipt=put(prefix+'.json',(json.dumps(value,indent=2,sort_keys=True)+'\n').encode())
print(json.dumps({'passed':value['passed'],'tests':result.testsRun,'receipt':receipt,'actual_step':os.environ['SLURM_STEP_ID'],'expiry':value['expiry_utc']}))
raise SystemExit(0 if result.wasSuccessful() else 1)
