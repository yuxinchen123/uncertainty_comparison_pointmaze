# Independently read the actual native process component; never execute tools or native operations.
import argparse,ctypes,hashlib,importlib.abc,json,os,re,shlex,sys,time
from pathlib import Path
R=Path('/p/rlprojects/RND_worktrees/merge-atari-cleanrl/10_jax_exploration_platform/benchmark_runs/2026-10-31-00-11_jaxatari_montezuma_pixels_cleanrl_ppo_rnd_environments-per-copy-128_rollout-steps-128_ideas-30-plus-20_steps-per-copy-2000000000_seed-start-1')
NAME='2026-11-05-06-39_native-process-control_jaguar03_python-3.12_gpu-count-1_processor-count-1_host-memory-gib-2_total-seconds-300_work-seconds-240_cleanup-seconds-60_environment-transitions-0_training-updates-0'
P=R/'remote_setup'/NAME;U=R/'data'/NAME
HELPER_ROOT=Path('/p/rlprojects/RND_worktrees/2026-11-05-06-48_a4500-native-process-helper_source-c228cfc402b3_total-seconds-300_work-seconds-240_cleanup-seconds-60_gpu-count-0')
JOB='6825842';SOURCE='c228cfc402b357ad9cf14ef1a7e0134eebdcf35b'
PREFIX='10_jax_exploration_platform/scripts/atari_pixel_campaign/'
NUM={'jax','jaxlib','numpy','torch','flax','optax','cupy','scipy','jaxtari','exploration_platform','pynvml'}
checks=[]
def need(value,label):
 if not value:raise ValueError(label)
 checks.append(label)
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as stream:
  while True:
   raw=stream.read(1024*1024)
   if not raw:break
   h.update(raw)
 return h.hexdigest()
def ref(path):return {'path':str(path),'sha256':sha(path)}
def doc(path):
 path=Path(path);need(path.is_absolute() and path.resolve(strict=True)==path and path.is_file(),'exact ordinary document '+str(path))
 def unique(pairs):
  result={}
  for key,value in pairs:
   if key in result:raise ValueError('duplicate raw metadata key')
   result[key]=value
  return result
 return json.loads(path.read_bytes(),object_pairs_hook=unique,parse_constant=lambda v:(_ for _ in ()).throw(ValueError('nonfinite raw metadata')))
def checked(value):
 need(type(value) is dict and set(value)=={'path','sha256'} and ref(value['path'])==value,'actual referenced metadata bytes')
 return doc(value['path'])
def source():
 plan=doc(U/'measurement_plan.json')
 need(plan['helpers']['source_commit']==SOURCE and len(plan['helpers']['files'])==7,'exact c228 seven-helper diagnostic')
 for relative,expected in plan['helpers']['files'].items():
  need(sha(U/'code'/Path(relative).name)==sha(HELPER_ROOT/relative)==expected,'exact staged/canonical helper '+relative)
 need(doc(U/'measurement_prepared.json')=={'plan_sha256':sha(U/'measurement_plan.json')},'exact actual canonical prepared plan')
 need(plan['kind']=='configured_native_nsight_process_diagnostic' and plan['gpu_uuid'] is None and plan['campaign_run_dir']==str(R) and plan['unit_id']==NAME,'distinct unresolved diagnostic scope')
 need(plan['python']=='/p/rlprojects/RND/.venvs/atari_pixels_jax/bin/python' and plan['maximum_active_seconds']==240 and plan['diagnostic_time_budget']=={'schema_version':1,'work_seconds':240,'cleanup_seconds':60,'total_seconds':300},'actual selected Python and five-minute budget')
 return plan

def accounting(args,intent):
 need(sha(args.accounting_stdout)==args.accounting_stdout_sha256 and sha(args.accounting_return)==args.accounting_return_sha256,'actual owned terminal accounting bytes pinned')
 ret=doc(args.accounting_return);need(ret.get('returncode')==0 and type(ret['returncode']) is int,'actual owned accounting query succeeded')
 command=ret.get('command',ret.get('argv'));need(type(command) is list and Path(command[0]).name=='sacct','actual accounting is an exact sacct read')
 jobs=None
 for index,arg in enumerate(command):
  if arg in ('-j','--jobs') and index+1<len(command):jobs=command[index+1]
  elif arg.startswith('--jobs='):jobs=arg.split('=',1)[1]
 need(jobs==JOB,'accounting queried only this exact owned ID')
 lines=Path(args.accounting_stdout).read_text().splitlines();need(bool(lines),'actual accounting has header and rows')
 header=lines[0].split('|');rows=[dict(zip(header,line.split('|'))) for line in lines[1:] if line.strip()]
 need(all(key in header for key in ('JobIDRaw','State','ExitCode','AllocCPUS','AllocTRES','NodeList')),'actual accounting required fields present')
 selected={row['JobIDRaw']:row for row in rows};numeric=JOB+'.'+intent['step_id']
 need(JOB in selected and numeric in selected,'actual job and numeric step terminal accounting')
 for identity,row in selected.items():
  need(identity==JOB or identity.startswith(JOB+'.'),'no foreign accounting row')
  need(row['State']=='COMPLETED' and row['ExitCode']=='0:0','actual owned accounting completed normally '+identity)
 for identity in (JOB,numeric):
  row=selected[identity];need(row['NodeList']=='jaguar03' and row['AllocCPUS'].isdigit() and int(row['AllocCPUS'])>0,'actual owned target and allocated CPUs '+identity)
  counts=dict(item.split('=',1) for item in row['AllocTRES'].split(',') if '=' in item)
  need(counts.get('gres/gpu')=='1' or counts.get('gres/gpu:nvidia_rtx_a4500')=='1','actual one-GPU allocation '+identity)
 return {'stdout':ref(args.accounting_stdout),'return':ref(args.accounting_return),'rows':rows,'numeric_step':numeric}

def execute(args,plan):
 need(os.environ.get('SLURM_JOB_ID')=='6821437' and os.environ.get('SLURM_STEP_ID','').isdecimal(),'actual current owned CPU review allocation')
 need(os.getresuid()==(2618919,)*3,'actual review owner')
 need((R/'slurm/submitted_jobids_uva.txt').read_text().splitlines().count('6821437')==1,'review allocation exact ownership')
 for filename in ('submitted_jobids.txt','submitted_jobids_uva.txt'):need((R/'slurm'/filename).read_text().splitlines().count(JOB)==1,'actual diagnostic exact ownership '+filename)
 need(not NUM.intersection(sys.modules),'no numerical modules before audit')
 bootstrap={'standard_library_ctypes_imported_before_audit':True,'native_library_calls_requested_by_audit':0}
 class NoNUM(importlib.abc.MetaPathFinder):
  def find_spec(self,name,path=None,target=None):
   if name.split('.')[0] in NUM:raise RuntimeError('no numerical imports in component raw review')
 sys.meta_path.insert(0,NoNUM());ctypes.CDLL=ctypes.PyDLL=lambda *a,**k:(_ for _ in ()).throw(RuntimeError('no native library load in component raw review'))
 def guard(event,values):
  if event in ('subprocess.Popen','socket.connect','socket.bind','socket.getaddrinfo','os.system','ctypes.dlopen','ctypes.dlsym'):raise RuntimeError('no processes/network/native operations in component raw review')
  if event=='open' and isinstance(values[0],(str,bytes)):
   path=os.fsdecode(values[0])
   if path.startswith(('/dev/nvidia','/dev/dri','/proc/driver/nvidia')) or path=='/tmp/sl5nw_serval02_card1.lock':raise RuntimeError('no device or standing descriptor access in component raw review')
 sys.addaudithook(guard);sys.path.insert(0,str(U/'code'))
 import configured_native_process as native
 import nsight_process_identity as identity
 native.policy(plan)
 result=doc(U/'result.json');need(result['kind']==native.DIAGNOSTIC and result['completed'] is True and result['plan']==ref(U/'measurement_plan.json') and result['source']==plan['helpers'],'actual completed component result exact source/plan')
 need(not any((U/name).exists() for name in ('failure.json','waiter_failure.json','capture_failure.json','tool_descendant_budget_inconclusive.json','native_process_failure.json')),'no failed or incomplete component record')
 zero=plan['diagnostic_boundary'];need(all(result.get(key)==value and type(result.get(key)) is type(value) for key,value in zero.items()),'all diagnostic zero-work/zero-credit boundaries exact')
 need(result['target_contract']==ref(U/'native_process_target_contract.json') and result['tool']==plan['profile_tool'],'actual target-contract and tool bytes bound')
 need(shlex.split((U/'command.txt').read_text())==native.producer_command(U,plan),'exact selected-Python producer command')
 owned=doc(U/'owned_launch_inputs.json');need(owned['job_id']==JOB and owned['plan']==ref(U/'measurement_plan.json'),'actual owned release binds exact prepared plan')
 row=[JOB,native.DIAGNOSTIC,SOURCE,str(P/'job.sbatch')]
 need(owned['submission_row']==row and (R/'slurm/submissions.tsv').read_text().splitlines().count('\t'.join(row))==1,'distinct exact fsynced diagnostic row')
 for key,kind in (('submission','owned_configured_native_process_diagnostic_submission'),('release','owned_configured_native_process_diagnostic_release')):
  recorded=checked(owned[key]);need(recorded['kind']==kind and recorded['job_id']==JOB and type(recorded['returncode']) is int and recorded['returncode']==0 and recorded['plan_sha256']==owned['plan']['sha256'] and recorded['data_unit']==str(U),'actual successful owned '+key)
 intent=identity.validate_saved_intent(U,plan,doc(U/'profile_process_intent.json'));need(intent['job_id']==JOB and intent['step_id'].isdecimal(),'exact actual diagnostic numeric step')
 claim=doc(U/'capture_identity.json');need(claim['schema_version']==1 and claim['protocol']==identity.PROTOCOL and claim['challenge']==intent['challenge'] and claim['intent']==ref(U/'profile_process_intent.json') and claim['source']==intent['source'] and claim['plan']==intent['plan'],'actual child blocked claim source/attempt')
 identity.validate_snapshot(claim['process'],2618919,intent['parent']['cgroup']);need(claim['pid']==claim['process']['pid'] and claim['process_start_time']==claim['process']['process_start_time'] and claim['process']['command']==intent['capture_command'],'actual capture claim raw UID/PID/birth/command')
 observation=doc(U/'capture_process_observation.json');rebuilt=identity.validate_observation(observation,claim,intent['capture_command'],intent['parent'])
 need(identity.stable(rebuilt['capture'])==identity.stable(claim['process']),'actual observed capture belongs to blocked claim')
 profiler=doc(U/'profiler_process.json');need(profiler['pid']==observation['root_pid'] and profiler['process_start_time']==observation['root_start_time'] and profiler['parent_pid']==intent['parent']['pid'] and profiler['parent_start_time']==intent['parent']['process_start_time'] and profiler['command']==intent['nsight_command']==doc(U/'executed_command.json'),'actual original tool Popen identity/command')
 identity.validate_root_wait(U,intent,profiler)
 launch=doc(U/'capture_actual_launch.json');waited=doc(U/'capture_actual_wait.json')
 need(launch['challenge']==waited['challenge']==intent['challenge'] and launch['claim']==ref(U/'capture_identity.json') and identity.stable(launch['child'])==identity.stable(claim['process']) and launch['command']==waited['command']==intent['capture_command'] and waited['launch']==ref(U/'capture_actual_launch.json') and waited['waited_pid']==claim['pid'] and type(waited['returncode']) is int and waited['returncode']==0 and waited['exited_normally'] is True and waited['interventions']==[],'actual exact parent child normal wait0')
 waiter=launch['waiter'];identity.validate_snapshot(waiter,2618919,claim['process']['cgroup']);need(waiter['command']==intent['waiter_command'] and claim['process']['parent_pid']==waiter['pid'] and identity.stable(observation['nodes'][str(waiter['pid'])])==identity.stable(waiter),'actual waiter raw ancestry/UID/source')
 release=doc(U/'capture_release.json');need(release=={**profiler,'capture_identity':claim,'process_observation':ref(U/'capture_process_observation.json')},'actual release exactly authorizes blocked observed child')
 acknowledged=doc(U/'capture_release_observed.json');need(acknowledged['release']==ref(U/'capture_release.json') and acknowledged['capture_identity']==ref(U/'capture_identity.json'),'actual child acknowledged exact immutable release')
 probe=doc(U/'probe_result.json');need(probe['completed'] is True and probe['pid']==claim['pid'] and probe['process_start_time']==claim['process_start_time'] and probe['numerical_modules']==[] and probe['cuda_profiler_api_called'] is False and all(probe.get(key)==value for key,value in zero.items()),'actual zero-environment zero-learner probe')
 tool=doc(U/'tool_version_return.json');need(tool['command']==[plan['profile_tool']['path'],'--version'] and type(tool['returncode']) is int and tool['returncode']==0 and tool['stderr']=='' and tool['stdout'].strip()==plan['profile_tool']['version'],'actual sole version command matches recorded tool')
 need(sha(plan['profile_tool']['path'])==plan['profile_tool']['sha256'],'actual shared tool bytes unchanged')
 deadline=native.saved_deadline(U,plan,identity);deadline.verify_snapshot(result['diagnostic_deadline'],phase='total',cleanup_required=True)
 proof=native.validate_saved(U,plan,identity,intent,observation,launch,waited,deadline)
 need(result['native_process_evidence']==proof,'actual complete three-role native/custody proof recomputed')
 terminal=doc(U/'tool_descendant_scope_terminal.json');need(terminal['terminal']['actual_echild'] is True and terminal['terminal']['all_actual_tool_roots_waited'] is True and terminal['all_tool_descendants_exited_before_allocation_custody_release'] is True and terminal['interventions']==[] and terminal['read_errors']==[],'actual all descendants waited/closed with fresh ECHILD')
 actual_accounting=accounting(args,intent)
 need(not NUM.intersection(sys.modules),'no numerical imports after raw review')
 review={'schema_version':1,'kind':'independent_configured_native_process_review','approved':True,'passed':True,'independent_of_producer':True,'reviewer_identity':'/root/review_bound_h100_timing_launch','plan':ref(U/'measurement_plan.json'),'result':ref(U/'result.json'),'target_contract':result['target_contract'],'actual_owned_numeric_step_verified':True,'all_tool_descendants_exited_and_waited':True,'exact_source_helpers_verified':7,'source_commit':SOURCE,'native_process_proof_recomputed':True,'actual_scheduler_accounting':actual_accounting,'checks':checks,'check_count':len(checks),'reviewed_at_seconds':time.time(),'job_id':os.environ['SLURM_JOB_ID'],'step_id':os.environ['SLURM_STEP_ID'],'uid':os.getuid(),'pid':os.getpid(),'cgroup':Path('/proc/self/cgroup').read_text(),'stdlib_bootstrap':bootstrap,'scientific_credit':0,'profile_admission_granted':False,'admission_granted':False,'worker_admitted':False,'GPU_numerical_qualification':False,'environment_transitions':0,'training_updates':0,'physical_gres_mapping_verified':False,'kernel_device_enforcement_verified':False,'whole_tool_library_or_device_activity_quantified':False,'unresolved_anomalies':[],'interpretation':'Process-provider and owned descendant component only. Every original numerical/GPU/source/configuration gate and a separate formal profile admission remain required.','audit_source':ref(Path(__file__).resolve())}
 out=P/'independent_actual_native_component_review.json'
 with out.open('x') as f:json.dump(review,f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
 print(json.dumps({'approved':True,'review':ref(out),'checks':len(checks),'scientific_credit':0,'formal_profile_admission':False}))
def main():
 parser=argparse.ArgumentParser();parser.add_argument('--execute-terminal-review',action='store_true')
 for key in ('accounting-stdout','accounting-return'):parser.add_argument('--'+key,type=Path);parser.add_argument('--'+key+'-sha256')
 args=parser.parse_args();plan=source()
 if not args.execute_terminal_review:
  need(not any(getattr(args,k) is not None for k in ('accounting_stdout','accounting_return','accounting_stdout_sha256','accounting_return_sha256')),'no unused execution inputs')
  print(json.dumps({'source_verified':True,'actual_result_reads':0,'terminal_review_executed':False,'source':ref(Path(__file__).resolve()),'requires':'actual final result, exact raw native roles/closure and successful owned terminal accounting'}));return
 need(all(getattr(args,k) is not None for k in ('accounting_stdout','accounting_return','accounting_stdout_sha256','accounting_return_sha256')),'actual exact terminal accounting inputs required')
 execute(args,plan)
if __name__=='__main__':main()
