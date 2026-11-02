RUN='/p/rlprojects/RND_worktrees/merge-atari-cleanrl/10_jax_exploration_platform/benchmark_runs/2026-10-31-00-11_jaxatari_montezuma_pixels_cleanrl_ppo_rnd_environments-per-copy-128_rollout-steps-128_ideas-30-plus-20_steps-per-copy-2000000000_seed-start-1'
OUTPUT='/p/rlprojects/RND_worktrees/merge-atari-cleanrl/10_jax_exploration_platform/benchmark_runs/2026-10-31-00-11_jaxatari_montezuma_pixels_cleanrl_ppo_rnd_environments-per-copy-128_rollout-steps-128_ideas-30-plus-20_steps-per-copy-2000000000_seed-start-1/remote_setup/2026-11-01-17-54-39_fresh-original-h100-qualification-independent-review_gpu-count-0'
"""Recompute formal source, history, software and asset gates without numerical or host work."""
from datetime import datetime,timezone
from importlib import metadata
import ctypes,fcntl,hashlib,importlib.abc,json,os,pathlib,re,stat,subprocess,sys,traceback
R=pathlib.Path(RUN);D=pathlib.Path(OUTPUT)
P=next(R.glob('remote_setup/*17-17_original-qualification_h100-nvl*'));U=R/'data'/P.name
JOB='6783003';PLAN_SHA='515e4089bec5fa65931b5f08fc018d716209243fd8311cc0f4740f954769472e'
SOURCE='6066d9ea35fbc9c045855144550971c2ac5895f9'
NUMERICAL={'jax','jaxlib','numpy','torch','flax','optax','cupy','scipy','jaxtari','exploration_platform'}
def require(condition,message):
 "Refuse a missing actual or frozen input."
 if not condition:raise ValueError(message)
def receipt(path):
 "Hash one source or metadata file with bounded memory."
 p=pathlib.Path(path);require(p.is_absolute() and p.resolve(strict=True)==p and stat.S_ISREG(p.lstat().st_mode),str(p))
 checksum=hashlib.sha256()
 with p.open('rb') as incoming:
  while block:=incoming.read(1024*1024):checksum.update(block)
 return {'path':str(p),'sha256':checksum.hexdigest()}
def doc(path):
 "Read a small exact JSON artifact."
 p=pathlib.Path(path);receipt(p);return json.loads(p.read_bytes())
def save(path,value):
 "Preserve a new durable reviewer record."
 with pathlib.Path(path).open('x') as outgoing:json.dump(value,outgoing,indent=2,sort_keys=True);outgoing.write('\n');outgoing.flush();os.fsync(outgoing.fileno())
def hash_paths(paths):
 "Keep source and failed-attempt byte evidence before and after the review."
 return {str(p):receipt(p)['sha256'] for p in paths}
require(os.environ.get('SLURM_JOB_ID')==JOB and os.environ.get('SLURM_STEP_ID','').isdecimal() and os.getuid()==os.geteuid()==2618919,'only current owned monitor step allowed')
for name in ['submitted_jobids.txt','submitted_jobids_uva.txt']:require((R/'slurm'/name).read_text().splitlines().count(JOB)==1,'exact own ID missing')
rows=[row.split('\t') for row in (R/'slurm/submissions.tsv').read_text().splitlines() if row.split('\t',1)[0]==JOB]
require(len(rows)==1 and rows[0][1:3]==['monitor','79a01e851ff92ddba7d0e12eaf8164cab2028878'],'exact monitor row changed')
command=['scontrol','show','job','-o',JOB];running=subprocess.run(command,capture_output=True,text=True,timeout=30)
save(D/'independent_formal_running_guard.json',{'command':command,'returncode':running.returncode,'stdout':running.stdout,'stderr':running.stderr,'job_id':JOB,'step_id':os.environ['SLURM_STEP_ID']})
require(running.returncode==0 and running.stderr=='' and 'JobId=6783003 ' in running.stdout and 'UserId=sl5nw(2618919) ' in running.stdout and 'JobState=RUNNING ' in running.stdout,'actual current CPU monitor is not running')
resource=R/'queue/cpu_monitor_large_artifact_read.lock';fd=os.open(resource,os.O_RDWR|os.O_CREAT|os.O_NOFOLLOW,0o600)
info=os.fstat(fd);require(stat.S_ISREG(info.st_mode) and info.st_uid==2618919,'ordinary owner CPU read lock required')
fcntl.flock(fd,fcntl.LOCK_EX);print('ACTUAL_OWNED_STEP='+JOB+'.'+os.environ['SLURM_STEP_ID'],flush=True)
blocked=[];queries=[]
class NoNumerical(importlib.abc.MetaPathFinder):
 def find_spec(self,name,path=None,target=None):
  if name.split('.')[0] in NUMERICAL:blocked.append(name);raise RuntimeError('numerical import forbidden')
sys.meta_path.insert(0,NoNumerical())
ctypes.CDLL=lambda *a,**k:(_ for _ in ()).throw(RuntimeError('shared-library loading forbidden'))
plan=doc(U/'measurement_plan.json');require(receipt(U/'measurement_plan.json')['sha256']==PLAN_SHA,'formal plan changed')
oldP=next(R.glob('remote_setup/*14-53_full-original-direct-qualification*'));oldU=R/'data'/oldP.name
protected=[U/'measurement_plan.json',U/'measurement_prepared.json',P/'preparation_request.json',P/'metadata_preparation_result.json',P/'shared_software_assets_staged_verification.json',R/'SOURCE_AND_GATES_READY.json']
protected += [oldU/name for name in ['failure.json','measurement_plan.json','direct_controller_outcome.json','direct_process_wait_return.json','code/measure_selected.py','code/serval02_card1_binding.py']]
protected += [pathlib.Path(plan['source_root'])/name for name in plan['core']['files']]
protected += [U/'code'/pathlib.Path(name).name for name in plan['helpers']['files']]
protected += [pathlib.Path(plan['helpers']['source_root'])/name for name in plan['helpers']['files']]
before=hash_paths(protected);real_popen=subprocess.Popen
allowed_roots={plan['source_root'],plan['helpers']['source_root'],'/p/rlprojects/RND/08_cleanrl_ppo_rnd/cleanrl','/p/rlprojects/RND/reference_repos/jaxtari_pixel_study'}
history_manifest=doc(U/'direct_history/evidence_manifest.json')
allowed_roots.update(str(U/'direct_history'/row['relative_path']) for row in history_manifest['git_roots'])
allowed_roots.update(row['original_root'] for row in history_manifest['git_roots'])
class ReadOnlyGit(real_popen):
 def __init__(self,command,*a,**k):
  require(isinstance(command,(list,tuple)) and len(command)>3 and command[0]=='git' and '-C' in command,'only local read-only Git is allowed')
  index=command.index('-C');root=command[index+1];operation=command[index+2]
  require(root in allowed_roots and operation in {'rev-parse','symbolic-ref','status','show','ls-tree','cat-file','merge-base','diff','diff-files','diff-index','ls-files','log'},'unexpected source root or Git operation: '+repr(command))
  queries.append(list(command));super().__init__(command,*a,**k)
subprocess.Popen=ReadOnlyGit
def audit(event,args):
 "Reject device, standing-lock, network and native-library access."
 if event in {'socket.connect','socket.bind','socket.getaddrinfo','ctypes.dlopen'}:raise RuntimeError('network or library action forbidden')
 if event=='open' and isinstance(args[0],str):
  require(not args[0].startswith('/dev/') and args[0]!='/tmp/sl5nw_serval02_card1.lock','device or GPU standing-lock access forbidden')
sys.addaudithook(audit)
try:
 source=pathlib.Path(plan['helpers']['source_root']);scripts=source/'10_jax_exploration_platform/scripts/atari_pixel_campaign';sys.path.insert(0,str(scripts))
 import direct_measurement as direct
 import measure_selected as selected
 import train_optimized as training
 require(pathlib.Path(direct.__file__).resolve()==scripts/'direct_measurement.py','wrong direct reader')
 require(plan['helpers']['source_commit']==SOURCE and len(plan['helpers']['files'])==25,'new exact 25-helper source required')
 history=direct.core().validate_history(plan)
 direct.verify_saved_sources(U,plan);direct.verify_qualification_source_receipts(plan)
 tests=direct.shared().test_collection_contract(plan);require(sum(tests.values())==48,'original full test definition count changed')
 require(plan['renderer_contract']['rooms']==33 and plan['renderer_contract']['cases']==132 and plan['renderer_contract']['synthetic_pixel_cases']==11,'renderer gate matrix changed')
 assets=training.verify_assets(plan['asset_record'],{})
 installed=selected.installed_backend_receipt(training.jaxtari_python_receipt(plan['jaxtari_source_root']),training.installed_gpu_library_receipts())
 reference=doc(plan['qualification_software']['installed_backend']['path'])
 require(installed==reference,'installed package/backend metadata differs from original')
 require(assets==doc(plan['qualification_software']['asset_receipt']['path']) and assets['file_count']==4531,'original asset bytes changed')
 require(metadata.version('torch')=='2.6.0+cpu','CPU Torch reference changed')
 author=doc(P/'shared_software_assets_staged_verification.json')
 require(author['passed'] is True and author['installed_library_images']==installed['gpu_libraries'] and author['helper_files']==plan['helpers']['files'] and author['assets']==assets,'author shared verification differs from actual recomputation')
 require(history['comparisons']==120 and history['stop']['jobs']==25,'original full history or exit job count changed')
 after=hash_paths(protected);require(after==before and not blocked,'a protected file or numerical-import boundary changed')
 require(not any((U/name).exists() for name in ['direct_current_launch.json','direct_device_binding.json','direct_process_metadata.json','direct_qualification_result.json','result.json','failure.json']),'preparation cannot invent or reuse actual execution')
 value={'schema_version':1,'kind':'independent_fresh_original_h100_formal_source_history_software_assets_review','passed':True,'at_utc':datetime.now(timezone.utc).isoformat(),'job_id':JOB,'step_id':os.environ['SLURM_STEP_ID'],'pid':os.getpid(),'cgroup':pathlib.Path('/proc/self/cgroup').read_text(),'packet':str(P),'plan':receipt(U/'measurement_plan.json'),'source_commit':SOURCE,'helper_files':plan['helpers']['files'],'history':history,'original_test_definitions':tests,'renderer_contract':plan['renderer_contract'],'assets':assets,'installed_library_images':installed['gpu_libraries'],'packages':installed['packages'],'torch_version':metadata.version('torch'),'protected_before':before,'protected_after':after,'local_read_only_git_commands':queries,'actual_cpu_memory_read_lock_held':True,'large_read_lock_device':info.st_dev,'large_read_lock_inode':info.st_ino,'actual_numerical_import_attempts':blocked,'gpu_operations':0,'host_contacts':0,'gpu_standing_lock_access':False,'case_credit':0,'idea_credit':0,'scientific_credit':0,'worker_admitted':False,'controller_and_freeze_review_still_pending':True}
 path=D/('formal_source_history_software_assets_'+JOB+'_'+os.environ['SLURM_STEP_ID']+'.json');save(path,value)
 print(json.dumps({'passed':True,'record':receipt(path),'comparisons':history['comparisons'],'jobs':history['stop']['jobs'],'original_tests':sum(tests.values()),'installed_images':len(installed['gpu_libraries']),'asset_files':assets['file_count']}),flush=True)
except BaseException as error:
 save(D/('formal_source_history_failure_'+JOB+'_'+os.environ['SLURM_STEP_ID']+'.json'),{'error_type':type(error).__name__,'error':str(error),'traceback':traceback.format_exc(),'actual_numerical_import_attempts':blocked,'local_read_only_git_commands':queries,'scientific_credit':0});raise
finally:os.close(fd)
