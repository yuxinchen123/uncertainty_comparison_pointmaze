"""Check publication refusal and exact local defaults without a scheduler call."""
import hashlib,importlib.abc,importlib.util,io,json,os,subprocess,sys
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch
P=Path(__file__).absolute().parent
assert os.environ.get('SLURM_JOB_ID')=='6783003' and os.environ.get('SLURM_STEP_ID','').isdigit()
blocked=[]
class NoNumerical(importlib.abc.MetaPathFinder):
 def find_spec(self,name,path=None,target=None):
  if name.split('.')[0] in {'jax','jaxlib','numpy','torch','flax','optax','jaxtari','scipy','cupy'}:blocked.append(name);raise RuntimeError('numerical import forbidden')
sys.meta_path.insert(0,NoNumerical())
real=subprocess.Popen
class LocalGitOnly(real):
 def __init__(self,args,*a,**k):
  command=list(args);assert command[:2]==['git','-C'] and command[2]=='/p/rlprojects/RND_worktrees/merge-atari-cleanrl' and command[3] in {'merge-base','show'}
  super().__init__(args,*a,**k)
subprocess.Popen=LocalGitOnly
sys.addaudithook(lambda event,args:(_ for _ in ()).throw(RuntimeError('network/device/library operation forbidden')) if event in {'socket.connect','socket.bind','socket.getaddrinfo','ctypes.dlopen'} else None)
spec=importlib.util.spec_from_file_location('literal_common_submit',P/'submit_common_confirmation_held.py');s=importlib.util.module_from_spec(spec);spec.loader.exec_module(s)
source=s.receipt(P/'submit_common_confirmation_held.py')
actual=[]
for copies in (1,2):
 with redirect_stdout(io.StringIO()) as out:s.main(['--copies',str(copies)])
 value=json.loads(out.getvalue());assert value['scheduler_queries']==value['submissions']==value['releases']==0
 actual.append(value)
inputs,_,_=s.validate_inputs(1)
old_refused=False
try:s.validate_publication(inputs,'9fcc43a478b93d731df780805fd7c3c514cdf67f')
except (ValueError,subprocess.CalledProcessError) as error:old_refused=True;old_error={'type':type(error).__name__,'message':str(error)}
assert old_refused
commands=[]
def git_object(command,*args,**kwargs):
 commands.append(command);assert command[:3]==['git','-C',inputs['repository']] and command[3]=='show'
 relative=command[4].split(':',1)[1];return (Path(inputs['repository'])/relative).read_bytes()
with patch.object(s.subprocess,'run',return_value=None),patch.object(s.subprocess,'check_output',side_effect=git_object):
 assert s.validate_publication(inputs,'a'*40)=='a'*40
changed_refused=False
with patch.object(s.subprocess,'run',return_value=None),patch.object(s.subprocess,'check_output',return_value=b'changed'):
 try:s.validate_publication(inputs,'a'*40)
 except ValueError:changed_refused=True
assert changed_refused and source==s.receipt(P/'submit_common_confirmation_held.py') and not blocked
assert all(not(Path(item['packet'])/'submission_intent.json').exists() for item in inputs['packets'].values())
value={'kind':'common_confirmation_held_publication_and_actual_default_controls','passed':True,'job_id':'6783003','step_id':os.environ['SLURM_STEP_ID'],'source':source,'actual_defaults':actual,'actual_old_source_commit_refused':old_refused,'old_publication_error':old_error,'mocked_exact_publication_objects':len(commands),'mocked_changed_object_refused':changed_refused,'mocked_positive_is_not_publication_evidence':True,'numerical_import_attempts':blocked,'scheduler_queries':0,'actual_submissions':0,'releases':0,'admission_granted':False}
p=P/('publication_and_default_controls_6783003_'+os.environ['SLURM_STEP_ID']+'.json')
with p.open('x') as f:json.dump(value,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
print(json.dumps({'passed':True,'receipt':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}))

