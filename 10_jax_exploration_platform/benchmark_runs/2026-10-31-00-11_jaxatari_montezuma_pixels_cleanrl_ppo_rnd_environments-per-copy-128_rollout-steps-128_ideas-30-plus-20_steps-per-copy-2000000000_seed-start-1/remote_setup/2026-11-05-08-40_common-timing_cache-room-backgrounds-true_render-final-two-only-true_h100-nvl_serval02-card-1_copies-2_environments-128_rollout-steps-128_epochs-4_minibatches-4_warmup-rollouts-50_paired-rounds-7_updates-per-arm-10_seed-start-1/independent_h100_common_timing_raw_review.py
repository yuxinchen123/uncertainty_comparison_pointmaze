# Independently read this one actual H100 timing unit. Never launch or recompute numerical work.
import argparse,ast,ctypes,datetime,fcntl,hashlib,importlib.abc,json,math,os,shlex,statistics,subprocess,sys,time
from pathlib import Path
R=Path('/p/rlprojects/RND_worktrees/merge-atari-cleanrl/10_jax_exploration_platform/benchmark_runs/2026-10-31-00-11_jaxatari_montezuma_pixels_cleanrl_ppo_rnd_environments-per-copy-128_rollout-steps-128_ideas-30-plus-20_steps-per-copy-2000000000_seed-start-1')
NAME='2026-11-05-08-40_common-timing_cache-room-backgrounds-true_render-final-two-only-true_h100-nvl_serval02-card-1_copies-2_environments-128_rollout-steps-128_epochs-4_minibatches-4_warmup-rollouts-50_paired-rounds-7_updates-per-arm-10_seed-start-1'
P=R/'remote_setup'/NAME;U=R/'data'/NAME;I=R.parents[2]
PLAN_SHA='40e85dafdecffaef7c3f9e28b91d8b9f5bea97a09750d822768b67d49a451da3'
HELPER_COMMIT='69f1b5075530418adbe02c1e433c5647848a61ba'
NUMERICAL_COMMIT='d28e6735e79788f618acb0ebfde8609db1f776af'
GPU='GPU-fed69954-5a71-dbb2-ee91-544e0eb6b8a3'
NUMERICAL={'jax','jaxlib','numpy','torch','flax','optax','scipy','cupy','jaxtari','exploration_platform','pynvml'}
checks=[];commands=[]
def need(value,label):
 if not value:raise ValueError(label)
 checks.append(label)
def digest(raw):return hashlib.sha256(raw).hexdigest()
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as stream:
  while True:
   part=stream.read(1024*1024)
   if not part:break
   h.update(part)
 return h.hexdigest()
def doc(path):
 path=Path(path);need(path.is_absolute() and path.resolve(strict=True)==path and path.is_file(),'ordinary exact document '+str(path))
 def unique(pairs):
  value={}
  for key,item in pairs:
   if key in value:raise ValueError('duplicate JSON key')
   value[key]=item
  return value
 return json.loads(path.read_bytes(),object_pairs_hook=unique,parse_constant=lambda v:(_ for _ in ()).throw(ValueError('nonfinite JSON')))
def ref(path):return {'path':str(path),'sha256':sha(path)}
def checked(value):
 need(type(value) is dict and set(value)=={'path','sha256'} and ref(value['path'])==value,'exact referenced document')
 return doc(value['path'])
def write_once(path,value):
 with Path(path).open('x') as out:json.dump(value,out,indent=2,allow_nan=False);out.write('\n');out.flush();os.fsync(out.fileno())
def source_only():
 need(sha(U/'measurement_plan.json')==PLAN_SHA,'exact current timing plan')
 plan=doc(U/'measurement_plan.json')
 need(plan['helpers']['source_commit']==HELPER_COMMIT and len(plan['helpers']['files'])==29,'actual current 29-helper source')
 need(plan['source_commit']==NUMERICAL_COMMIT,'actual unchanged numerical source')
 for label in ('core','paired_source','helpers'):
  value=plan[label]
  for name,row in value['files'].items():
   expected=row['sha256'] if type(row) is dict else row
   need(sha(Path(value['source_root'])/name)==expected,'actual pinned source '+name)
   if label=='helpers':need(sha(U/'code'/Path(name).name)==expected,'actual staged source '+name)
 return plan

def audit_context(plan):
 need(os.environ.get('SLURM_JOB_ID')=='6825983' and os.environ.get('SLURM_STEP_ID','').isdecimal(),'current owned CPU allocation only')
 need((R/'slurm/submitted_jobids_uva.txt').read_text().splitlines().count('6825983')==1,'exact current monitor ownership ledger')
 need(os.getresuid()==(2618919,)*3,'all effective and saved owner UIDs')
 expected='0::/system.slice/slurmstepd.scope/job_6825983/step_'+os.environ['SLURM_STEP_ID']+'/user/task_0'
 need(Path('/proc/self/cgroup').read_text().splitlines()==[expected],'actual owned numeric CPU step')
 bootstrap={'scope':'standard-library imports including ctypes occurred before audit hooks; no CDLL call was requested by this script','ctypes_loaded_before_guard':True,'numerical_imports_before_guard':sorted(NUMERICAL.intersection(sys.modules))}
 need(not bootstrap['numerical_imports_before_guard'],'no numerical imports before audit')
 class NoNumerical(importlib.abc.MetaPathFinder):
  def find_spec(self,name,path=None,target=None):
   if name.split('.')[0] in NUMERICAL:raise RuntimeError('numerical import prohibited in raw audit')
 sys.meta_path.insert(0,NoNumerical())
 ctypes.CDLL=ctypes.PyDLL=lambda *a,**k:(_ for _ in ()).throw(RuntimeError('native library load prohibited in raw audit'))
 roots={str(Path(plan[label]['source_root']).resolve()) for label in ('core','paired_source','helpers')};roots.add(str(I))
 def guard(event,args):
  if event in ('socket.connect','socket.bind','socket.getaddrinfo','os.system','ctypes.dlopen','ctypes.dlsym'):raise RuntimeError('network/native operation prohibited in raw audit')
  if event=='subprocess.Popen':
   argv=args[1]
   metadata_command=['/usr/bin/python3','-I','-S','-B',str(U/'code/direct_common_confirmation.py'),'--plan-stdin']
   exact_metadata=(type(argv) in (list,tuple) and list(argv)==metadata_command
    and args[0]=='/usr/bin/python3' and args[2] is None
    and args[3]==common.metadata_environment(os.environ)
    and sha(U/'code/direct_common_confirmation.py')==plan['helpers']['files']['10_jax_exploration_platform/scripts/atari_pixel_campaign/direct_common_confirmation.py'])
   source_git=(type(argv) in (list,tuple) and argv and argv[0]=='git' and len(argv)>3 and argv[1]=='-C' and str(Path(argv[2]).resolve()) in roots)
   if not(exact_metadata or source_git):raise RuntimeError('only exact local source Git reads or the pinned isolated correctness metadata child are allowed')
   commands.append(list(argv))
  if event=='open' and isinstance(args[0],(str,bytes)):
   path=os.fsdecode(args[0])
   if path.startswith(('/dev/nvidia','/dev/dri','/proc/driver/nvidia')) or path=='/tmp/sl5nw_serval02_card1.lock':raise RuntimeError('device/standing GPU lock access prohibited')
 sys.addaudithook(guard)
 sys.path.insert(0,str(U/'code'))
 import direct_measurement as direct
 import direct_common_confirmation as common
 import post_comparison_validation as validation
 import train_optimized as training
 need(all(Path(module.__file__).parent==U/'code' for module in (direct,common,validation,training)),'exact staged metadata reader namespace')
 return direct,common,validation,training,bootstrap

def actual_exit(path,expected_sha,process,controller):
 need(sha(path)==expected_sha,'actual root both-process exit observation hash')
 value=doc(path)
 need(value.get('host')=='serval02' and value.get('uid')==2618919 and value.get('all_original_processes_ended') is True and value.get('signals_sent')==0 and value.get('gpu_or_standing_lock_operations')==0 and value.get('global_process_enumeration') is False,'root observed exact normal process exits without interventions')
 expected={(controller['pid'],controller['process_start_time']),(process['pid'],process['process_start_time'])}
 rows=value.get('observations');need(type(rows) is list and len(rows)==2,'exact controller and numerical child exit observations')
 need({(row.get('pid'),row.get('expected_birth_counter')) for row in rows}==expected,'exit observations join exact PID and birth')
 need(all(row.get('original_process_ended') is True and (row.get('proc_entry_absent') is True or (type(row.get('actual_birth_counter')) is str and row['actual_birth_counter']!=row['expected_birth_counter'])) for row in rows),'both original processes actually ended')
 return ref(path)

def raw_review(args,plan,direct,common,validation,training,bootstrap):
 started=time.monotonic();result=doc(U/'result.json');metadata=doc(U/'direct_process_metadata.json');binding=doc(U/'direct_device_binding.json')
 need(not any((U/name).exists() for name in ('failure.json','direct_backend_capture_failure.json','instrumented_completion.json')),'no failure or instrumented result replaces timing evidence')
 need(result['source_commit']==NUMERICAL_COMMIT and result['helpers']==plan['helpers'] and result['copy_seeds']==[1,2] and result['runtime_environment']==plan['runtime_environment'],'actual result source helper seeds runtime exact')
 need(result['reference']==plan['reference_configuration'] and result['candidate']==plan['configuration'] and result['copies']==2,'actual full configurations exact')
 need(result['measurement_mode']=='confirmation' and result['measurement_valid'] is True and result['contaminated'] is False and result['implementation_distinct'] is True,'actual complete distinct uncontaminated timing record')
 need(result['protocol']=={'rounds':7,'requested_minimum_updates_per_round':10,'actual_updates_per_round':10},'actual seven pairs ten updates protocol')
 need(plan['configuration']['learner']['n_envs']==128 and plan['configuration']['learner']['num_steps']==128 and plan['configuration']['learner']['update_epochs']==4 and plan['configuration']['learner']['num_minibatches']==4 and plan['configuration']['learner']['observation_warmup_iterations']==50 and plan['configuration']['learner']['total_timesteps']==2000000000,'original full configuration and two-billion-step annealing horizon')
 direct.verify_saved_sources(U,plan)
 direct.core().validate_binding(plan,metadata,binding,U)
 need(binding['gpu_uuid']==GPU and result['resolved_device_binding']==binding and result['hardware']['gpu_uuid']==GPU,'same actual approved physical card1 UUID')
 process=metadata['after']['identity'];need(process['pid']==545091 and process['process_start_time']=='85908834' and process['parent_pid']==543087,'actual registered producer identity')
 need(validation.execution_identity(result)==tuple(process[key] for key in ('hostname','pid','process_start_time')),'actual numerical execution identity and finite chronological bounds')
 need(result['direct_qualification_inputs']==plan['direct_qualification_inputs'],'actual accepted original qualification exact reference')
 direct.verify_qualification_inputs(plan)
 common.contract(plan);common.verify_saved(U,plan,result)
 need(result['correctness_attempted'] is True and result['correctness_passed'] is True and result['correctness']['passed'] is True,'actual correctness reports passed')
 correctness=result['correctness'];reports={'initialization':correctness['initialization'],'complete_iteration':correctness['complete_iteration'],**correctness['supplied_inputs']}
 need(set(reports)=={'initialization','complete_iteration','preparation','fixed_batch','fixed_batch_update'} and correctness['complete_iterations']==1,'all exact fixed-input and full-iteration reports present')
 for name,row in reports.items():
  need(type(row['leaves']) is int and row['leaves']>0,'actual nonempty correctness leaves '+name)
  need(all(type(row[key]) in (int,float) and math.isfinite(row[key]) and row[key]>=0 for key in ('largest_float32_absolute_error','largest_float64_absolute_error')),'finite nonnegative observed correctness errors '+name)
  need(type(row['float32_derived_reward_moment_fields']) is list and all(type(field) is str and any(field.endswith('.reward_statistics.'+key) for key in ('mean','variance')) for field in row['float32_derived_reward_moment_fields']),'only unchanged source-declared derived reward moment fields '+name)
 validation.correctness(result);common.validate_pair_protocol(plan,result)
 shared=direct.shared();memory_summaries=[];ratios=[];previous_finish=None
 def window(memory,label):
  proof=memory['assigned_gpu_process_observation'];boundref=ref(U/'direct_device_binding.json')
  need(proof.get('direct_binding')==boundref and proof.get('startup_memory_observed') is False and proof.get('gpu_uuid')==GPU and proof.get('pid')==process['pid'],'actual direct window identity '+label)
  expected={'custody':binding['custody_after'],'current_admission':metadata['current_admission'],'binding':boundref}
  need(direct.core().exact(proof.get('direct_before'),expected) and direct.core().exact(proof.get('direct_after'),expected),'actual direct window custody boundaries '+label)
  rebuilt=shared._validate_window_for_policy(proof,'direct_serval02_card_one')
  need(all(memory.get(key)==value for key,value in rebuilt.items()),'actual complete raw own-only window recomputation '+label)
  need(memory['gpu_uuid']==GPU and all(sample['free_bytes']*100>=sample['total_bytes']*15 for sample in proof['samples']),'actual measured physical memory headroom '+label)
  memory_summaries.append({'label':label,'sample_count':len(proof['samples']),'memory_sha256':digest(json.dumps(memory,sort_keys=True,allow_nan=False).encode()),'started_at_seconds':proof['window_started_at_seconds'],'finished_at_seconds':proof['window_finished_at_seconds']})
  return proof
 for pair in result['rounds']:
  for label in pair['order']:
   row=pair[label];window_proof=window(row['memory'],'round '+str(pair['round'])+' '+label)
   measured=validation.measured_round(row,2,10)
   need(row['iterations']==10 and row['seconds']<=window_proof['window_finished_at_seconds']-window_proof['window_started_at_seconds']+.001,'raw sampler covers complete measured round')
   need(math.isclose(row['transitions_per_second_total'],measured,rel_tol=1e-12,abs_tol=0) and math.isclose(row['transitions_per_second_per_copy'],measured/2,rel_tol=1e-12,abs_tol=0) and math.isclose(row['seconds_per_update'],row['seconds']/10,rel_tol=1e-12,abs_tol=0),'actual throughput and per-update arithmetic')
   need(previous_finish is None or window_proof['window_started_at_seconds']>=previous_finish,'actual alternating timed windows do not overlap or reverse')
   previous_finish=window_proof['window_finished_at_seconds']
  ratios.append(pair['reference']['seconds']/pair['candidate']['seconds'])
 whole=window(result['direct_whole_work_memory'],'whole numerical work')
 need(whole['window_started_at_seconds']<=memory_summaries[0]['started_at_seconds'] and whole['window_finished_at_seconds']>=max(row['finished_at_seconds'] for row in memory_summaries[:-1]),'actual whole-work window contains all fourteen timed windows')
 ideas_file=Path(plan['paired_source']['source_root'])/'10_jax_exploration_platform/scripts/atari_pixel_campaign/ideas.py'
 summary_node=next(node for node in ast.parse(ideas_file.read_bytes()).body if isinstance(node,ast.FunctionDef) and node.name=='summarize_pairs')
 summary_namespace={'math':math,'statistics':statistics}
 exec(compile(ast.Module(body=[summary_node],type_ignores=[]),str(ideas_file),'exec'),summary_namespace)
 summary=summary_namespace['summarize_pairs'](result['rounds'],correctness_passed=True,contaminated=False)
 need(all(result.get(key)==value for key,value in summary.items()),'all exact existing paired statistics and decision fields recomputed')
 median=summary['median_speedup'];probability=summary['one_sided_sign_probability']
 need(summary['decision']=='accepted','actual accepted intermediate common performance under existing unchanged rule')
 need(result['median_candidate_transitions_per_second_total']==statistics.median(row['candidate']['transitions_per_second_total'] for row in result['rounds']) and result['median_reference_transitions_per_second_total']==statistics.median(row['reference']['transitions_per_second_total'] for row in result['rounds']),'actual median per-arm transition rates')
 for arm in ('reference','candidate'):
  latency=result['synchronized_latency'][arm]
  need(latency['calls']==3 and latency['updates_per_call']==1 and len(latency['seconds_per_synchronized_call'])==3 and all(type(v) in (int,float) and math.isfinite(v) and v>0 for v in latency['seconds_per_synchronized_call']),'actual separate three-call synchronized latency '+arm)
  need(latency['median_seconds_per_synchronized_call']==statistics.median(latency['seconds_per_synchronized_call']) and latency['median_amortized_seconds_per_update']==latency['median_seconds_per_synchronized_call'],'actual synchronized latency arithmetic '+arm)
 assets=training.verify_assets(plan['asset_record'],dict(plan['runtime_environment']))
 need(assets==doc(U/'asset_receipt.json'),'actual full asset bytes and receipt')
 backend=doc(U/'installed_backend.json');need(result['installed_backend']==backend,'actual installed backend result matches saved metadata')
 loaded=result['loaded_gpu_libraries'];training.validate_gpu_library_receipt(loaded,{'installed_gpu_libraries':backend['gpu_libraries']})
 need(loaded['pid']==process['pid'] and binding['native']['driver_library'] in loaded['loaded_shared_libraries'],'actual current-process driver and CUDA/cuDNN loaded-library receipt')
 direct.validate_startup_checks(U,binding)
 backend_proof=direct.backend_support(plan).validate_capture(U,plan,binding,loaded,result)
 terminal=direct.terminal_process(U,metadata);controller_records=common.controller_evidence(direct,U,metadata)
 spawn=doc(U/'direct_controller_spawn.json');controller=spawn['controller']
 need(controller['pid']==543087 and controller['process_start_time']=='85900187','exact original controller birth')
 exits=actual_exit(args.exit_observation,args.exit_observation_sha256,process,controller)
 startup=doc(P/'independent_actual_startup_registration_review.json');need(sha(P/'independent_actual_startup_registration_review.json')=='5cb80736948f4764581c86cd4610d6b05f232e4f525e8095ff78b0be4dd9849d','genuine independent actual startup source/registration review unchanged')
 need(not NUMERICAL.intersection(sys.modules),'no numerical imports after raw review')
 record={'schema_version':1,'kind':'independent_actual_direct_common_configuration_confirmation_raw_first','approved':True,'passed':True,'result':ref(U/'result.json'),'plan':ref(U/'measurement_plan.json'),'source_commit':plan['source_commit'],'helper_commit':plan['helpers']['source_commit'],'configuration':plan['configuration'],'reference_configuration':plan['reference_configuration'],'copies':2,'copy_seeds':[1,2],'gpu_uuid':GPU,'resolved_binding':ref(U/'direct_device_binding.json'),'execution_identity':result['execution_identity'],'actual_returncode':0,'terminal_process':ref(U/'direct_terminal_process.json'),'exit_observation':exits,'controller_records':controller_records,'raw_window_recomputation_passed':True,'window_summaries':memory_summaries,'paired_rounds':7,'updates_per_arm_per_round':10,'paired_speedups':ratios,'median_speedup':median,'one_sided_sign_probability':probability,'decision':result['decision'],'correctness':correctness,'actual_assets_verified':assets,'backend_expected':ref(U/'direct_backend_image_expected.json'),'backend_capture_begin':ref(U/'direct_backend_capture_begin.json'),'captured_backend':ref(U/'direct_backend_capture.json'),'loaded_gpu_libraries_sha256':digest(json.dumps(loaded,sort_keys=True,indent=2,allow_nan=False).encode()+b'\n'),'startup_review':ref(P/'independent_actual_startup_registration_review.json'),'unchanged_staged_consumer_passed':False,'raw_approval_created_before_unchanged_consumer':True,'independent_of_producer':True,'reviewer_identity':'/root/correct_native_process_control','reviewed_at_seconds':time.time(),'unresolved_anomalies':[],'admission_granted':False,'worker_admitted':False,'scientific_new_idea_credit':0,'packing_point_credit':0,'final_training_credit':0,'qualification_credit':0,'job_id':os.environ['SLURM_JOB_ID'],'step_id':os.environ['SLURM_STEP_ID'],'uid':os.getuid(),'pid':os.getpid(),'cgroup':Path('/proc/self/cgroup').read_text(),'stdlib_bootstrap':bootstrap,'numerical_imports':sorted(NUMERICAL.intersection(sys.modules)),'gpu_or_device_or_library_operations_requested_by_audit':0,'checks':checks,'check_count':len(checks),'seconds':time.monotonic()-started,'actual_local_source_git_commands':commands,'audit_source':ref(Path(__file__).resolve())}
 out=P/'independent_actual_timing_raw_first.json';write_once(out,record)
 print(json.dumps({'passed':True,'review':ref(out),'checks':len(checks),'seconds':record['seconds'],'median_speedup':median,'one_sided_sign_probability':probability,'consumer_has_not_run':True}))

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--execute-terminal-review',action='store_true');parser.add_argument('--exit-observation',type=Path);parser.add_argument('--exit-observation-sha256');args=parser.parse_args()
 plan=source_only()
 if not args.execute_terminal_review:
  need(args.exit_observation is None and args.exit_observation_sha256 is None,'no unused execution inputs')
  print(json.dumps({'source_verified':True,'actual_result_reads':0,'terminal_review_run':False,'actual_source':ref(Path(__file__).resolve()),'source_checks':len(checks),'requires_actual_terminal_files_and_both_process_exits':True}));return
 need(args.exit_observation is not None and args.exit_observation_sha256 is not None,'actual hashed both-process exit observation is required')
 for name in ('result.json','direct_backend_capture.json','direct_terminal_process.json','direct_process_wait_return.json','direct_controller_outcome.json','direct_controller_output_hashes.json'):
  need((U/name).is_file(),'actual final artifact required '+name)
 lock_path=R/'queue/cpu_monitor_large_artifact_read.lock'
 need(lock_path.resolve(strict=True)==lock_path and lock_path.is_file(),'exact shared large-read lock')
 with lock_path.open('a+') as lock:
  fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
  modules=audit_context(plan);raw_review(args,plan,*modules)
if __name__=='__main__':main()
