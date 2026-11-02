"""Read actual saved correctness evidence without importing numerical packages."""
import argparse,ctypes,datetime,hashlib,importlib.abc,json,math,os,re,socket,stat,subprocess,sys,time,traceback
from pathlib import Path
OUT=Path(__file__).absolute().parent
R=OUT.parent.parent
NUM={'jax','jaxlib','numpy','torch','flax','optax','cupy','scipy','jaxtari','exploration_platform'}
args=argparse.ArgumentParser();args.add_argument('--copies',type=int,choices=(1,2),required=True);copies=args.parse_args().copies
job={1:'6782946',2:'6782947'}[copies]
name='2026-11-01-15-09_common_cache-room-backgrounds-true_render-final-two-only-true_correctness_jaguar03_copies-'+str(copies)+'_environments-128_rollout-steps-128_epochs-4_minibatches-4_warmup-rollouts-50_seed-start-1'
U=R/'data'/name;P=R/'remote_setup'/name
allowed_roots={str(R.parents[2]),'/p/rlprojects/RND_worktrees/atari-cleanrl-source-d28e673','/p/rlprojects/RND_worktrees/2026-11-01-13-50_full-original-qualification-explicit-pytest-root_source-commit-e5afbf6b51a3d1666b46e71c0631a45d47080cd8','/p/rlprojects/RND_worktrees/atari-allocator-retry-helpers-03b0f62','/p/rlprojects/RND/reference_repos/jaxtari_pixel_study','/p/rlprojects/RND/08_cleanrl_ppo_rnd/cleanrl'}
accounting_command=['sacct','--noheader','--parsable2','--jobs='+job,'--format=JobIDRaw,JobName,User,State,ExitCode,NodeList,Start,End,AllocCPUS,ReqMem,ReqTRES,AllocTRES']
commands=[]
def guard(event,values):
 if event=='import' and values[0].split('.')[0] in NUM:raise RuntimeError('numerical package import forbidden: '+values[0])
 if event in {'socket.connect','socket.bind','ctypes.dlopen'}:raise RuntimeError('network or library operation forbidden')
 if event=='open' and isinstance(values[0],(str,bytes)) and os.fsdecode(values[0]).startswith('/dev/nvidia'):raise RuntimeError('device operation forbidden')
 if event=='subprocess.Popen':
  cmd=values[1]
  ok=isinstance(cmd,list) and (cmd==accounting_command or len(cmd)>3 and cmd[:2]==['git','-C'] and cmd[2] in allowed_roots and cmd[3] in {'show','cat-file','rev-parse','status','symbolic-ref','ls-tree','merge-base'})
  if not ok:raise RuntimeError('unapproved command: '+repr(cmd))
  commands.append(cmd)
sys.addaudithook(guard)
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def json_file(path):
 def unique(pairs):
  out={}
  for k,v in pairs:
   assert k not in out,'duplicate JSON key';out[k]=v
  return out
 return json.loads(Path(path).read_bytes(),object_pairs_hook=unique,parse_constant=lambda v:(_ for _ in ()).throw(ValueError('nonfinite JSON')))
def put(name,value,raw=False):
 b=value if raw else (json.dumps(value,indent=2,allow_nan=False)+'\n').encode();q=OUT/('copies_'+str(copies)+'_'+name)
 with q.open('xb') as f:f.write(b);f.flush();os.fsync(f.fileno())
 return {'path':str(q),'sha256':hashlib.sha256(b).hexdigest()}
def snapshot():
 result={}
 for directory in (U,P):
  for p in sorted(directory.rglob('*')):
   assert not p.is_symlink(),str(p)
   if p.is_file():result[str(p)]={'sha256':sha(p),'bytes':p.stat().st_size,'mode':stat.S_IMODE(p.stat().st_mode)}
 return result

def main():
 assert os.environ.get('SLURM_JOB_ID')=='6783003' and os.environ.get('SLURM_STEP_ID','').isdigit() and os.getuid()==2618919
 assert '/job_6783003/step_'+os.environ['SLURM_STEP_ID']+'/' in Path('/proc/self/cgroup').read_text()
 step=os.environ['SLURM_STEP_ID'];print('ACTUAL_OWNED_STEP=6783003.'+step,flush=True)
 assert (R/'slurm/submitted_jobids.txt').read_text().splitlines().count(job)==1 and (R/'slurm/submitted_jobids_uva.txt').read_text().splitlines().count(job)==1
 before=snapshot();plan=json_file(U/'measurement_plan.json');result=json_file(U/'result.json');progress=json_file(U/'progress.json')
 assert not (U/'failure.json').exists() and progress['state']=='completed' and progress['completed'] is True and progress['mode']=='correctness' and progress['decision']=='correctness_passed' and progress['rounds']==0
 assert plan['kind']=='selected_configuration_measurement' and plan['mode']=='correctness' and plan['source_commit']=='d28e6735e79788f618acb0ebfde8609db1f776af' and plan['helpers']['source_commit']=='e5afbf6b51a3d1666b46e71c0631a45d47080cd8'
 assert plan['campaign_run_dir']==str(R) and plan['unit_id']==name and plan['gpu_uuid'] is None and plan['configured_isolation']['physical_identity']['gpu_uuid'] is None
 assert plan['copy_seeds']==result['copy_seeds']==list(range(1,copies+1)) and all(type(x) is int for x in result['copy_seeds']) and type(result['copies']) is int and result['copies']==copies
 assert result['candidate']==plan['configuration'] and result['reference']==plan['reference_configuration']
 changes=[]
 for scope in ('learner','environment','execution'):
  assert set(result['reference'][scope])==set(result['candidate'][scope])
  changes.extend((scope,key,result['reference'][scope][key],value) for key,value in result['candidate'][scope].items() if value!=result['reference'][scope][key])
 assert changes==[('environment','cache_room_backgrounds',False,True),('environment','render_final_two_only',False,True)],changes
 for configuration in (result['reference'],result['candidate']):
  assert configuration['learner']['n_copies']==copies and configuration['learner']['n_envs']==128 and configuration['learner']['num_steps']==128 and configuration['learner']['update_epochs']==4 and configuration['learner']['num_minibatches']==4 and configuration['learner']['observation_warmup_iterations']==50 and configuration['learner']['total_timesteps']==2000000000
 assert result['correctness_passed'] is True and result['correctness_attempted'] is True and result['implementation_distinct'] is True and result['measurement_valid'] is False and result['decision']=='correctness_passed' and result['measurement_mode']=='correctness' and not result.get('rounds')
 checks=result['correctness'];assert checks['passed'] is True and type(checks['complete_iterations']) is int and checks['complete_iterations']==1 and set(checks['supplied_inputs'])=={'preparation','fixed_batch','fixed_batch_update'}
 reports={'initialization':checks['initialization'],'complete_iteration':checks['complete_iteration'],**checks['supplied_inputs']}
 for key,report in reports.items():
  assert set(report)=={'leaves','largest_float32_absolute_error','largest_float64_absolute_error','float32_derived_reward_moment_fields'} and type(report['leaves']) is int and report['leaves']>0
  for error in ('largest_float32_absolute_error','largest_float64_absolute_error'):assert type(report[error]) in (int,float) and math.isfinite(report[error]) and report[error]==0
  expected=['[0].reward_statistics.mean','[0].reward_statistics.variance'] if key=='complete_iteration' else []
  assert report['float32_derived_reward_moment_fields']==expected
 for key in ('physical_gres_mapping_verified','kernel_device_enforcement_verified','installed_binary_matches_upstream_verified','worker_admitted'):assert result[key] is False
 sys.path.insert(0,str(U/'code'))
 import configured_measurement as support
 import post_comparison_validation as validation
 import train_optimized as training
 identity=support.core();metadata=json_file(U/'configured_isolation_metadata.json');binding=json_file(U/'resolved_device_binding.json');hardware=json_file(U/'hardware.json');backend=json_file(U/'installed_backend.json')
 identity.validate_plan(plan);support.verify_saved_sources(U,plan);identity.validate_binding(plan,metadata,binding,U)
 assert metadata['environment']['SLURM_JOB_ID']==job and metadata['environment']['SLURM_STEP_ID'].isdigit()
 assert result['resolved_device_binding']==binding and result['hardware']==hardware and result['installed_backend']==backend
 assert hardware=={**support.bound_hardware(plan,metadata,binding),'runtime_environment':plan['runtime_environment']}
 execution=validation.execution_identity(result);validation.correctness(result);validation._identity_fields(result,'jaguar03_rtx_a4500',plan['source_commit'],plan['runtime_environment']);validation.configuration(result['reference']);validation.configuration(result['candidate'])
 assert execution==(metadata['process']['hostname'],metadata['process']['pid'],metadata['process']['process_start_time']) and progress['pid']==execution[1]
 assert result['numerical_module_paths'] and {'exploration_platform.atari_pixels.config','exploration_platform.atari_pixels.environment','exploration_platform.atari_pixels.training'}<=set(result['numerical_module_paths'])
 for module,path in result['numerical_module_paths'].items():
  q=Path(path);assert q.resolve(strict=True)==q and q.is_relative_to(Path(plan['source_root'])/'10_jax_exploration_platform/src/exploration_platform/atari_pixels');expected=plan['core']['files'][str(q.relative_to(plan['source_root']))];assert sha(q)==expected['sha256']
 assert result['runtime_environment']==plan['runtime_environment']==hardware['runtime_environment']=={'JAX_DEFAULT_MATMUL_PRECISION':'highest','JAX_ENABLE_X64':'true','XLA_FLAGS':'--xla_gpu_exclude_nondeterministic_ops=true'}
 assets=training.verify_assets(plan['asset_record'],{});assert assets==json_file(U/'asset_receipt.json') and assets['file_count']==4531
 qualified=Path(plan['target_qualification_inputs']['result']['path']).parent;qualified_plan=json_file(qualified/'measurement_plan.json');support.verify_qualification_backend(qualified_plan,backend)
 loaded=result['loaded_gpu_libraries'];training.validate_gpu_library_receipt(loaded,{'installed_gpu_libraries':backend['gpu_libraries']});assert type(loaded['pid']) is int and loaded['pid']==execution[1] and binding['native']['driver_library'] in loaded['loaded_shared_libraries'] and loaded['platform'] in {'gpu','cuda'}
 loaded_hashes={}
 for path,record in loaded['verified_package_libraries'].items():assert sha(path)==record['sha256'];loaded_hashes[path]=record['sha256']
 admission=json_file(U/'assigned_process_admission.json');support.validate_process_sample(admission,binding['gpu_uuid'],execution[1]);assert admission['own']['bytes']>0 and admission['other_active_compute_rows']==0
 helper_modes={}
 for relative,digest in plan['helpers']['files'].items():
  staged=U/'code'/Path(relative).name;assert sha(staged)==digest
  raw_tree=subprocess.check_output(['git','-C',plan['helpers']['source_root'],'ls-tree',plan['helpers']['source_commit'],'--',relative],text=True);mode,kind,blob,path=raw_tree.strip().split(None,3);assert path==relative and kind=='blob';b=staged.read_bytes();assert hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()==blob;assert (staged.stat().st_mode&0o111)==(0o111 if mode=='100755' else 0);helper_modes[relative]={'mode':mode,'git_blob':blob,'sha256':digest}
 assert len(helper_modes)==20
 env=os.environ.copy();env['TZ']='UTC';accounting=subprocess.run(accounting_command,capture_output=True,timeout=20,env=env);assert accounting.returncode==0 and accounting.stderr==b''
 accounting_ref=put('terminal_accounting.raw',accounting.stdout,True);put('terminal_accounting_return.json',{'command':accounting_command,'environment_override':{'TZ':'UTC'},'returncode':accounting.returncode,'stderr_sha256':hashlib.sha256(accounting.stderr).hexdigest(),'at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()})
 rows=[line.split('|') for line in accounting.stdout.decode().splitlines() if line.strip()];byid={row[0]:row for row in rows};numeric=job+'.'+metadata['environment']['SLURM_STEP_ID']
 for selected in (job,job+'.batch',numeric):assert selected in byid and byid[selected][3:6]==['COMPLETED','0:0','jaguar03'],(selected,byid)
 assert byid[job][2]=='sl5nw' and byid[job][8]=='8' and byid[numeric][8]=='8'
 begin=datetime.datetime.fromisoformat(byid[job][6]).replace(tzinfo=datetime.timezone.utc).timestamp();end=datetime.datetime.fromisoformat(byid[job][7]).replace(tzinfo=datetime.timezone.utc).timestamp()+1
 assert begin<=result['execution_identity']['started_at_seconds']<result['execution_identity']['finished_at_seconds']<=end
 assert plan['protocol'] if 'protocol' in plan else True
 consumer_plan,consumer_binding=support.verify_saved_measurement(U,result);assert consumer_plan==plan and consumer_binding==binding
 after=snapshot();assert after==before
 assert not NUM.intersection(sys.modules)
 value={'schema_version':1,'kind':'independent_actual_common_a4500_correctness_raw_validation','passed':True,'copies':copies,'job_id':job,'numeric_step':numeric,'terminal_state':'COMPLETED','exit_code':'0:0','review_allocation':'6783003','review_step':step,'review_pid':os.getpid(),'review_cgroup':Path('/proc/self/cgroup').read_text(),'at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'unit':str(U),'packet':str(P),'result':{'path':str(U/'result.json'),'sha256':sha(U/'result.json')},'plan':{'path':str(U/'measurement_plan.json'),'sha256':sha(U/'measurement_plan.json')},'source_commit':plan['source_commit'],'helper_commit':plan['helpers']['source_commit'],'helper_files':helper_modes,'configuration':plan['configuration'],'reference_configuration':plan['reference_configuration'],'configuration_changes':changes,'copy_seeds':plan['copy_seeds'],'gpu_uuid':binding['gpu_uuid'],'pci_address':binding['pci_address'],'physical_minor':binding['physical_minor'],'original_cuda_visible_devices':metadata['environment']['CUDA_VISIBLE_DEVICES'],'configuration_trust_profile':binding['profile'],'resolved_binding':{'path':str(U/'resolved_device_binding.json'),'sha256':sha(U/'resolved_device_binding.json')},'execution_identity':result['execution_identity'],'agreement_reports':reports,'actual_assigned_process_admission':admission,'memory_evidence_scope':'One successful initial assigned-UUID own-only process-memory sample. Correctness mode has no whole-work sampling window and grants no measured-memory headroom or timing credit.','loaded_package_library_hashes':loaded_hashes,'asset_identity':assets,'unchanged_staged_consumer_passed':True,'terminal_accounting':accounting_ref,'terminal_accounting_rows':byid,'preserved_files_before':before,'preserved_files_after':after,'allowed_git_roots':sorted(allowed_roots),'commands':commands,'numerical_imports':[],'gpu_library_or_device_operations':0,'actual_submission_release_cancel_or_relaunch_actions':0,'timing_credit':0,'idea_credit':0,'final_training_credit':0,'admission_granted':False,'common_correctness_prerequisite_only':True}
 return value
try:value=main()
except BaseException:put('audit_failure.txt',traceback.format_exc().encode(),True);raise
ref=put('raw_audit.json',value);print(json.dumps({'passed':True,'copies':copies,'job_id':job,'receipt':ref,'review_step':value['review_step'],'result_sha256':value['result']['sha256']}))
