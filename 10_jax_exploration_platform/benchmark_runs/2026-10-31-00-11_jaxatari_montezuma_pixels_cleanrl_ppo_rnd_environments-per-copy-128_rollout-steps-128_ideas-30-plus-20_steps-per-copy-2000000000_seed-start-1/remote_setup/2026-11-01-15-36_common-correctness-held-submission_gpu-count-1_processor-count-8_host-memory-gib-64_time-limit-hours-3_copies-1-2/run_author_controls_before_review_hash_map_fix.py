"""Run synthetic held-submission controls and real local defaults without live commands."""
from datetime import datetime,timezone
import hashlib,importlib.abc,io,json,os,pathlib,subprocess,sys,unittest
P=pathlib.Path(__file__).absolute().parent
assert os.environ.get('SLURM_JOB_ID')=='6782816' and os.environ.get('SLURM_STEP_ID','').isdecimal()
names=['submit_common_correctness_held.py','test_submit_common_correctness_held.py','submission_inputs.json']
images=lambda:{n:hashlib.sha256((P/n).read_bytes()).hexdigest() for n in names}
before=images();inputs=json.loads((P/'submission_inputs.json').read_bytes());protected={row['path']:row['sha256'] for row in inputs['required_files']};assert all(hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()==h for p,h in protected.items());blocked=[]
class NoNumerical(importlib.abc.MetaPathFinder):
 def find_spec(self,name,path=None,target=None):
  if name.split('.')[0] in {'jax','jaxlib','numpy','torch','flax','optax','jaxtari','cupy','scipy'}:blocked.append(name);raise RuntimeError('numerical import forbidden')
sys.meta_path.insert(0,NoNumerical())
real_popen=subprocess.Popen
class GitOnly(real_popen):
 def __init__(self,args,*a,**k):
  command=list(args) if isinstance(args,(list,tuple)) else []
  assert len(command)>3 and command[:2]==['git','-C'] and command[3] in {'show','cat-file','rev-parse','status','diff','ls-tree','merge-base','symbolic-ref'},'real scheduler/network process forbidden'
  super().__init__(args,*a,**k)
subprocess.Popen=GitOnly
sys.addaudithook(lambda event,args:(_ for _ in ()).throw(RuntimeError('GPU/library/network action forbidden')) if event in {'socket.connect','socket.bind','socket.getaddrinfo'} or event=='ctypes.dlopen' and args[0] is not None else None)
sys.path.insert(0,str(P));import test_submit_common_correctness_held as tests
result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(tests))
actual=[]
if result.wasSuccessful():
 for copies in [1,2]:
  raw,item,plan=tests.submit.validate_inputs(copies);actual.append({'copies':copies,'mode':plan['mode'],'fixed_command':item['command'],'scheduler_queries':0,'submission_calls':0,'plan_sha256':item['plan']['sha256']})
assert images()==before and not blocked and all(hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()==h for p,h in protected.items())
value={'schema_version':1,'kind':'common_correctness_one_shot_held_submission_author_controls','passed':result.wasSuccessful(),'at_utc':datetime.now(timezone.utc).isoformat(),'job_id':'6782816','step_id':os.environ['SLURM_STEP_ID'],'pid':os.getpid(),'birth':pathlib.Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19],'cgroup':pathlib.Path('/proc/self/cgroup').read_text(),'tests':result.testsRun,'errors':len(result.errors),'failures':len(result.failures),'error_details':[{'test':test.id(),'traceback':message} for test,message in result.errors],'failure_details':[{'test':test.id(),'traceback':message} for test,message in result.failures],'actual_packet_defaults':actual,'source_before':before,'source_after':images(),'protected_packet_and_independent_review_files':len(protected),'protected_bytes_unchanged':True,'actual_scheduler_queries':0,'actual_submissions':0,'release_actions':0,'network_actions':0,'GPU_operations':0,'numerical_import_attempts':blocked,'scientific_credit':0,'admission_granted':False}
f=P/('author_controls_6782816_'+os.environ['SLURM_STEP_ID']+'.json')
with f.open('x') as out:json.dump(value,out,sort_keys=True,indent=2);out.write('\n');out.flush();os.fsync(out.fileno())
print(json.dumps({'passed':value['passed'],'tests':result.testsRun,'receipt':str(f),'sha256':hashlib.sha256(f.read_bytes()).hexdigest(),'actual_step':os.environ['SLURM_STEP_ID']}));raise SystemExit(0 if result.wasSuccessful() else 1)
