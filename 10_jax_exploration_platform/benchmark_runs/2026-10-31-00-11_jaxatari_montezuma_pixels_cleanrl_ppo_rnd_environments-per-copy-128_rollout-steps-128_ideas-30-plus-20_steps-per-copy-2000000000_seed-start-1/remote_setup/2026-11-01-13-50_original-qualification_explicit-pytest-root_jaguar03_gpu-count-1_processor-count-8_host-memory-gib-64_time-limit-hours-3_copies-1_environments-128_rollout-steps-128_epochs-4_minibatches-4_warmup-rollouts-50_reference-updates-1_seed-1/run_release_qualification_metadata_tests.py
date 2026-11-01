"""Run only metadata controls on the existing owned CPU monitor allocation."""
import ctypes,datetime,hashlib,importlib.abc,json,os,pathlib,subprocess,sys,traceback,unittest
from unittest.mock import patch
P=pathlib.Path(__file__).resolve().parent
assert os.environ.get('SLURM_JOB_ID')=='6781231' and os.environ.get('SLURM_STEP_ID','').isdecimal()
names=['release_qualification_6782778.py','test_release_qualification_6782778.py','release_frozen_inputs_6782778.json','release_implementation_review_6782778.json']
images=lambda:{n:hashlib.sha256((P/n).read_bytes()).hexdigest() for n in names}
before=images();blocked=[]
initial_roster=P.parent/'2026-11-01-13-56_explicit-pytest-root-qualification-packet-publication.json'
initial=json.loads(initial_roster.read_text())
protected={row['absolute_path']:row['sha256'] for row in initial['files']}
old_packet=next(q for q in P.parent.iterdir() if q.name.startswith('2026-11-01-12-36_full-original-qualification_'))
old_unit=P.parent.parent/'data'/old_packet.name
for root in (old_packet,old_unit):
 for q in root.rglob('*'):
  if q.is_file() and not q.is_symlink(): protected[str(q)]=hashlib.sha256(q.read_bytes()).hexdigest()
protected_before={path:hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest() for path in protected}
assert protected_before==protected

class Block(importlib.abc.MetaPathFinder):
 def find_spec(self,name,path=None,target=None):
  if name.split('.')[0] in {'jax','jaxlib','numpy','torch','flax','optax','cupy','scipy'}:blocked.append(name);raise RuntimeError('numerical import forbidden')
sys.meta_path.insert(0,Block())
ctypes.CDLL=lambda *a,**k:(_ for _ in ()).throw(RuntimeError('shared-library loads forbidden'))
real_popen=subprocess.Popen
class GitOnly(real_popen):
 def __init__(self,args,*a,**k):
  cmd=list(args) if isinstance(args,(list,tuple)) else []
  assert len(cmd)>3 and cmd[:2]==['git','-C'] and pathlib.Path(cmd[2]).is_absolute() and cmd[3] in {'show','cat-file','rev-parse','status','symbolic-ref','ls-tree','merge-base'},'live command forbidden'
  super().__init__(args,*a,**k)
subprocess.Popen=GitOnly
sys.path.insert(0,str(P));import test_release_qualification_6782778 as tests
suite=unittest.defaultTestLoader.loadTestsFromModule(tests);result=unittest.TextTestRunner(verbosity=2).run(suite)
extra={}
try:
 with patch.object(tests.release.socket,'gethostname',return_value='portal05'):
  actual=tests.release.checked()
  extra['actual_local_default_check']=actual['job_id']=='6782778' and actual['publication_commit'] is None
  extra['actual_saved_held_shape']=tests.release.validate_held((P/'held_scontrol_stdout.raw').read_text()) is None
except BaseException:
 extra['actual_local_default_check']=False
 extra['traceback']=traceback.format_exc()
protected_after={path:hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest() for path in protected}
assert images()==before and protected_after==protected_before and not blocked
raw=pathlib.Path('/proc/self/stat').read_text();step=os.environ['SLURM_STEP_ID']
v={'schema_version':1,'kind':'qualification_release_author_metadata_tests','at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'job_id':'6781231','step_id':step,'pid':os.getpid(),'birth':raw[raw.rfind(')')+2:].split()[19],'cgroup':pathlib.Path('/proc/self/cgroup').read_text(),'cpu_affinity':sorted(os.sched_getaffinity(0)),'tests':result.testsRun,'errors':len(result.errors),'failures':len(result.failures),'error_details':[{'test':t.id(),'traceback':s} for t,s in result.errors],'failure_details':[{'test':t.id(),'traceback':s} for t,s in result.failures],'passed':result.wasSuccessful() and extra.get('actual_local_default_check') is True and extra.get('actual_saved_held_shape') is True,'extra':extra,'protected_file_count':len(protected),'protected_before':protected_before,'protected_after':protected_after,'source_before':before,'source_after':images(),'numerical_imports_blocked':True,'shared_library_loads_blocked':True,'only_real_subprocesses_are_read_only_git':True,'actual_scheduler_queries':0,'actual_release_calls':0,'final_owned_inputs_written':False}
f=P/('release_author_6781231_'+step+'.json')
with f.open('x') as stream:json.dump(v,stream,sort_keys=True,indent=2);stream.write('\n');stream.flush();os.fsync(stream.fileno())
print(json.dumps({'receipt':str(f),'sha256':hashlib.sha256(f.read_bytes()).hexdigest(),'passed':v['passed'],'tests':result.testsRun,'extra':extra}));sys.exit(0 if v['passed'] else 1)
