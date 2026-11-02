"""Guard one exact published metadata preparation; do not initialize numerical libraries."""
import ctypes,hashlib,importlib.abc,json,os,runpy,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent;R=ROOT.parents[1]
command=json.loads(sys.argv[1]);S=Path(command[4]).parent
assert command[:4]==['/usr/bin/python3','-I','-S','-B'] and S.name=='atari_pixel_campaign' and command[5]=='--prepare-only'
assert os.environ.get('SLURM_JOB_ID')=='6783003' and os.environ.get('SLURM_STEP_ID','').isdigit()
NUM={'jax','jaxlib','numpy','torch','flax','optax','scipy','cupy','jaxtari','exploration_platform','pynvml'}
assert not NUM.intersection(sys.modules)
roots={str(S.parents[2])}
for prefix in ('20-35','20-37'):
 paths=list((R/'remote_setup').glob('2026-11-01-'+prefix+'_common-a4500-confirmation-independent-raw-audit*/review.json'));assert len(paths)==1
 roots.update(json.loads(paths[0].read_bytes())['closed_Git_roots'])
blocked=[];allowed=[]
class NoNumerical(importlib.abc.MetaPathFinder):
 def find_spec(self,name,path=None,target=None):
  if name.split('.')[0] in NUM:blocked.append(name);raise RuntimeError('numerical import forbidden: '+name)
sys.meta_path.insert(0,NoNumerical())
ctypes.CDLL=lambda *a,**k:(_ for _ in ()).throw(RuntimeError('real shared-library load forbidden'))
real_popen=subprocess.Popen
class MetadataOnly(real_popen):
 def __init__(self,argv,*a,**k):
  cmd=list(argv)
  ok=len(cmd)>3 and cmd[:2]==['git','-C'] and cmd[2] in roots and cmd[3] in {'show','cat-file','rev-parse','status','symbolic-ref','ls-tree','merge-base'}
  ok|=cmd==['/usr/bin/python3','-I','-S','-B',str(S/'common_confirmation.py'),'--plan-stdin']
  if not ok:blocked.append(cmd);raise RuntimeError('unapproved metadata child: '+repr(cmd))
  allowed.append(cmd);super().__init__(argv,*a,**k)
subprocess.Popen=MetadataOnly
def audit(event,values):
 if event in {'socket.connect','socket.bind','socket.getaddrinfo','os.system','os.kill','os.killpg'}:blocked.append(event);raise RuntimeError('network or process action forbidden')
sys.addaudithook(audit)
sys.path.insert(0,str(S));sys.argv=command[4:]
before={name:getattr(sys.modules.get(name),'__file__',None) for name in ('ideas','paired_benchmark','controller')}
runpy.run_path(command[4],run_name='__main__')
assert not blocked and not NUM.intersection(sys.modules)
assert before=={name:getattr(sys.modules.get(name),'__file__',None) for name in before}
packet=Path(command[command.index('--configuration')+1]).parent
value={'passed':True,'job_id':'6783003','step_id':os.environ['SLURM_STEP_ID'],'command':command,
 'allowed_read_only_metadata_children':allowed,'numerical_import_attempts':blocked,'parent_public_numerical_helpers_unchanged':True,
 'actual_gpu_operations':0,'actual_job_actions':0}
p=packet/'preparation_guard.json'
with p.open('x') as f:json.dump(value,f,indent=2,sort_keys=True);f.write('\n');f.flush();os.fsync(f.fileno())
