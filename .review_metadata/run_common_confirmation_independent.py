"""Run affected metadata tests on the existing CPU allocation with closed child commands."""
import ast,builtins,ctypes,datetime,hashlib,io,json,os
from pathlib import Path
import subprocess,sys,time,unittest
W=Path(__file__).absolute().parent.parent
S=W/'10_jax_exploration_platform/scripts/atari_pixel_campaign'
assert os.environ.get('SLURM_JOB_ID')=='6783003' and os.environ.get('SLURM_STEP_ID','').isdigit()
freeze=json.loads((W/'.review_metadata/common_confirmation_complete_source_freeze.json').read_bytes());expected=freeze['files']
images=lambda:{p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in expected}
before=images();assert before==expected
source=ast.parse((S/'tests/test_configured_measurement.py').read_bytes())
probe=next(ast.literal_eval(n.value) for n in source.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='ISOLATED_IMPORT_PROBE' for t in n.targets))
probes={probe,probe+'\nimport direct_measurement as direct\nfor name in direct.core().HELPERS: direct.core().adjacent(name[:-3])\nassert direct.shared().direct_support() is direct\n'}
NUM={'jax','jaxlib','numpy','torch','flax','optax','scipy','cupy','jaxtari','exploration_platform','pynvml','matplotlib'}
original=builtins.__import__;blocked=[];allowed=[]
def guarded(name,*args,**kwargs):
 if name.split('.')[0] in NUM:blocked.append(name);raise RuntimeError('numerical import forbidden: '+name)
 return original(name,*args,**kwargs)
builtins.__import__=guarded
real_popen=subprocess.Popen
class MetadataOnly(real_popen):
 def __init__(self,args,*other,**kwargs):
  command=list(args) if isinstance(args,(tuple,list)) else [];ok=False
  if command[:2]==['git','clone']:
   origin,target=map(Path,command[-2:]);ok='--no-hardlinks' in command and origin.is_absolute() and origin.resolve(strict=True)==origin and origin.is_relative_to('/tmp') and origin.stat().st_uid==os.getuid() and target.is_absolute() and target.is_relative_to('/tmp') and not target.exists()
  elif command[:2]==['git','-C']:
   root=Path(command[2]);rest=command[3:]
   while rest[:1]==['-c']:rest=rest[2:]
   read={'show','cat-file','rev-parse','status','symbolic-ref','ls-tree','merge-base','diff','ls-files','log'}
   fixture={'init','add','commit','config','checkout','branch','tag','update-ref'}
   ok=root.is_absolute() and bool(rest) and (rest[0] in read or root.is_relative_to('/tmp') and root.stat().st_uid==os.getuid() and rest[0] in fixture)
  elif command[:2]==['git','init']:
   target=Path(command[-1]);ok=target.is_absolute() and target.is_relative_to('/tmp')
  elif command and command[0]==sys.executable and '-c' in command:
   index=command.index('-c');tail=command[index+2:]
   ok=command[1:index] in (['-I','-B'],['-I','-S','-B']) and command[index+1] in probes and len(tail)==2 and Path(tail[0]).is_absolute() and Path(tail[0]).resolve().is_relative_to('/tmp')
  if not ok:blocked.append(command);raise RuntimeError('unapproved child command: '+repr(command))
  allowed.append(command);super().__init__(args,*other,**kwargs)
subprocess.Popen=MetadataOnly
sys.addaudithook(lambda event,args:(_ for _ in ()).throw(RuntimeError('live operation forbidden: '+event)) if event in {'socket.connect','socket.bind','ctypes.dlopen','os.system'} else None)
sys.path[:0]=[str(W/'.review_metadata'),str(S),str(S/'tests')]
modules=['independent_common_confirmation_controls','test_common_confirmation','test_common_confirmation_preparation','test_configured_measurement','test_measure_selected','test_profile_selected','test_direct_measurement','test_post_comparison_validation','test_validate_training_deployment','test_final_phase_deployment']
suite=unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromName(name) for name in modules);started=time.time();log=io.StringIO()
with unittest.mock.patch('sys.stderr',log):result=unittest.TextTestRunner(stream=log,verbosity=1).run(suite)
after=images();assert before==after
value={'schema_version':1,'kind':'common_confirmation_affected_independent_metadata_tests','job_id':'6783003','step_id':os.environ['SLURM_STEP_ID'],'pid':os.getpid(),'cgroup':Path('/proc/self/cgroup').read_text(),'at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'modules':modules,'tests':result.testsRun,'passed':result.wasSuccessful() and not blocked,'failures':[{'test':t.id(),'traceback':e} for t,e in result.failures],'errors':[{'test':t.id(),'traceback':e} for t,e in result.errors],'source_before':before,'source_after':after,'blocked_actions':blocked,'allowed_metadata_children':allowed,'duration_seconds':time.time()-started,'output':log.getvalue(),'numerical_work':False,'gpu_or_job_actions':0}
raw=(json.dumps(value,indent=2)+'\n').encode();p=W/'.review_metadata'/('common_confirmation_affected_independent_6783003_'+os.environ['SLURM_STEP_ID']+'.json')
with p.open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
print(json.dumps({'receipt':str(p),'sha256':hashlib.sha256(raw).hexdigest(),'passed':value['passed'],'tests':result.testsRun,'failures':value['failures'],'errors':value['errors'],'blocked_actions':blocked}));raise SystemExit(0 if value['passed'] else 1)
