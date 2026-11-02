"""Run only the bounded packet metadata commands on the owned CPU allocation."""
import builtins,ctypes,hashlib,json,os,runpy,subprocess,sys,time
from pathlib import Path
W=Path("/p/rlprojects/RND_worktrees/2026-11-01-17-55_a4500-common-confirmation_prior-correctness-required_rounds-7_complete-updates-per-arm-10_copies-1-2")
H=Path("/p/rlprojects/RND_worktrees/2026-11-01-18-47_common-confirmation-source_commit-9fcc43a478b93d731df780805fd7c3c514cdf67f")
R=Path("/p/rlprojects/RND_worktrees/merge-atari-cleanrl/10_jax_exploration_platform/benchmark_runs/2026-10-31-00-11_jaxatari_montezuma_pixels_cleanrl_ppo_rnd_environments-per-copy-128_rollout-steps-128_ideas-30-plus-20_steps-per-copy-2000000000_seed-start-1")
assert os.environ.get('SLURM_JOB_ID')=='6783003' and os.environ.get('SLURM_STEP_ID','').isdigit()
NUM={'jax','jaxlib','numpy','torch','flax','optax','scipy','cupy','jaxtari','exploration_platform','pynvml','matplotlib'}
original_import=builtins.__import__; blocked=[];allowed=[]
def guarded(name,*args,**kwargs):
 if name.split('.')[0] in NUM:blocked.append(name);raise RuntimeError('numerical import forbidden: '+name)
 return original_import(name,*args,**kwargs)
builtins.__import__=guarded
real_popen=subprocess.Popen
class MetadataOnly(real_popen):
 def __init__(self,args,*other,**kwargs):
  command=list(args) if isinstance(args,(tuple,list)) else [];ok=False
  if command[:2]==['git','-C']:
   root=Path(command[2]);rest=command[3:]
   ok=root==H and rest and rest[0] in {'show','ls-tree','status','rev-parse','symbolic-ref','merge-base'}
  elif command[:4]==['/usr/bin/python3','-I','-S','-B']:
   p=Path(command[4]) if len(command)>4 else Path('.')
   if p==H/'10_jax_exploration_platform/scripts/atari_pixel_campaign/measure_selected.py':
    ok='--prepare-only' in command and '--mode' in command and command[command.index('--mode')+1]=='confirmation'
   elif p.is_absolute() and p.name in {'common_confirmation.py','test_wait_for_owned_inputs.py'}:
    ok=p.is_relative_to(R) and not p.is_symlink()
  elif command[:2]==['bash','-n']:ok=len(command)==3 and Path(command[2]).is_relative_to(R)
  if not ok:blocked.append(command);raise RuntimeError('unapproved metadata child: '+repr(command))
  allowed.append(command);super().__init__(args,*other,**kwargs)
subprocess.Popen=MetadataOnly
sys.addaudithook(lambda event,args:(_ for _ in ()).throw(RuntimeError('forbidden operation: '+event)) if event in {'socket.connect','socket.bind','ctypes.dlopen','os.system'} else None)
before=hashlib.sha256((W/'.review_metadata/prepare_common_confirmation_packets.py').read_bytes()).hexdigest()
sys.argv=[str(W/'.review_metadata/prepare_common_confirmation_packets.py'),'--helper-root',str(H),'--helper-commit','9fcc43a478b93d731df780805fd7c3c514cdf67f']
started=time.time();failure=None
try:runpy.run_path(sys.argv[0],run_name='__main__')
except BaseException as error:failure={'type':type(error).__name__,'message':str(error)};raise
finally:
 after=hashlib.sha256((W/'.review_metadata/prepare_common_confirmation_packets.py').read_bytes()).hexdigest()
 value={'kind':'actual_common_confirmation_packet_preparation_guard','job_id':'6783003','step_id':os.environ['SLURM_STEP_ID'],'pid':os.getpid(),'cgroup':Path('/proc/self/cgroup').read_text(),'passed':failure is None and not blocked and before==after,'failure':failure,'source_before':before,'source_after':after,'allowed_children':allowed,'blocked_actions':blocked,'seconds':time.time()-started,'numerical_work':False,'gpu_or_job_actions':0}
 p=W/'.review_metadata'/('common_confirmation_packet_preparation_6783003_'+os.environ['SLURM_STEP_ID']+'.json')
 with p.open('x') as stream:json.dump(value,stream,indent=2);stream.write('\n');stream.flush();os.fsync(stream.fileno())
 print(json.dumps({'guard_receipt':str(p),'passed':value['passed']}))

