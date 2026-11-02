"""Read actual historical full-update/compile costs and finish only new launch metadata."""
from datetime import datetime,timezone
import hashlib,json,math,os,re,shlex,subprocess
from pathlib import Path

INDEX=Path('/p/rlprojects/RND_worktrees/merge-atari-cleanrl/10_jax_exploration_platform/benchmark_runs/2026-10-31-00-11_jaxatari_montezuma_pixels_cleanrl_ppo_rnd_environments-per-copy-128_rollout-steps-128_ideas-30-plus-20_steps-per-copy-2000000000_seed-start-1/remote_setup/2026-11-01-15-09_unconfirmed-common-a4500-preparation_index.json');value=json.loads(INDEX.read_text());records=value['packets'];owner=Path(__file__).absolute().parent;run=owner.parent.parent
assert os.environ.get('SLURM_JOB_ID')=='6782816' and os.environ.get('SLURM_STEP_ID','').isdecimal()
def doc(path):return json.loads(Path(path).read_bytes())
def image(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def ref(path):return {'path':str(path),'sha256':image(path)}
def put(path,v):
 raw=v if isinstance(v,bytes) else (json.dumps(v,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
 with Path(path).open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
 return ref(path)

initial=doc(value['initial_completion']['actual_120_raw_audit']);costs={};allinputs={}
for copies in [1,2]:
 entries=[]
 for row in initial['rows']:
  if row['hardware_class']!='jaguar03_rtx_a4500' or row['copies']!=copies:continue
  path=Path(row['result_path']);result=doc(path);assert image(path)==row['result_sha256']
  rounds=result['rounds'];assert len(rounds)==7
  totals={arm:sum(float(pair[arm]['seconds']) for pair in rounds) for arm in ['reference','candidate']}
  per_update={arm:max(float(pair[arm]['seconds'])/int(pair[arm]['iterations']) for pair in rounds) for arm in ['reference','candidate']}
  assert all(type(pair[arm]['iterations']) is int and pair[arm]['iterations']>=1 for pair in rounds for arm in ['reference','candidate'])
  warmups={arm:sum(float(item['seconds'])/int(item['updates']) for item in result['warmup'][arm]) for arm in ['reference','candidate']}
  total=float(result['build_to_completion_seconds']);nonround=total-sum(totals.values());assert nonround>=0
  entry={'name':row['name'],'result':ref(path),'build_to_completion_seconds':total,'non_round_initialization_correctness_compile_warmup_seconds':nonround,'timed_one_update_rounds_seconds':totals,'max_seconds_per_full_update':per_update,'measured_one_update_warmup_seconds':warmups}
  entries.append(entry);allinputs[str(path)]=image(path)
 assert len(entries)==30
 max_ref=max(x['max_seconds_per_full_update']['reference'] for x in entries);max_candidate=max(x['max_seconds_per_full_update']['candidate'] for x in entries);overhead=max(x['non_round_initialization_correctness_compile_warmup_seconds'] for x in entries);warm=max(sum(x['measured_one_update_warmup_seconds'].values()) for x in entries)
 timed=7*10*(max_ref+max_candidate);extra_warm=9*warm
 estimate=2*(overhead+extra_warm+timed)+900
 hours=math.ceil(estimate/3600)
 costs[str(copies)]={'actual_initial_cases':entries,'max_reference_seconds_per_full_update':max_ref,'max_candidate_seconds_per_full_update':max_candidate,'maximum_actual_non_round_overhead_seconds':overhead,'additional_warmup_extrapolation_seconds':extra_warm,'seven_pairs_ten_updates_timing_extrapolation_seconds':timed,'planned_seconds_with_two_times_margin_and_900_seconds_startup':estimate,'recommended_future_request_hours':hours,'recommendation_only_not_a_formal_confirmation_plan':True,'must_recheck_against_new_actual_correctness_and_current_resources_before_formal_plan':True}
budget=put(owner/'historical_timing_budget.json',{'schema_version':1,'kind':'historical_full_update_and_compile_budget_for_intermediate_common_confirmation','at_utc':datetime.now(timezone.utc).isoformat(),'cpu_job':'6782816','cpu_step':os.environ['SLURM_STEP_ID'],'source_commit':'d28e6735e79788f618acb0ebfde8609db1f776af','copies':costs,'method':'Use all30 actual A4500 cases per copy count. Normalize each actual complete round by its recorded positive full-update count. Separate recorded total-minus-timed-round cost; add nine additional copies of normalized warmup cost for ten-update warmups; extrapolate70reference+70candidate full updates from worst observed seconds per update, double the combined cost and add900seconds. This is a conservative planning estimate, not a measured new-combination runtime.','actual_initial_result_hashes':allinputs,'formal_confirmation_plans_created':0,'numerical_experiments':0,'scientific_credit':0})
qualified_packet=Path('/p/rlprojects/RND_worktrees/merge-atari-cleanrl/10_jax_exploration_platform/benchmark_runs/2026-10-31-00-11_jaxatari_montezuma_pixels_cleanrl_ppo_rnd_environments-per-copy-128_rollout-steps-128_ideas-30-plus-20_steps-per-copy-2000000000_seed-start-1/remote_setup/2026-11-01-13-50_original-qualification_explicit-pytest-root_jaguar03_gpu-count-1_processor-count-8_host-memory-gib-64_time-limit-hours-3_copies-1_environments-128_rollout-steps-128_epochs-4_minibatches-4_warmup-rollouts-50_reference-updates-1_seed-1');oldscript=(qualified_packet/'job.sbatch').read_text();qplan=doc(qualified_packet.parent.parent/'data'/qualified_packet.name/'measurement_plan.json')
resource=doc(owner/'current_resource_observation.json');assert 'MaxTime=4-00:00:00' in resource['queries']['partition']['stdout']
created=[]
for item in records:
 packet=Path(item['packet']);unit=Path(item['data_unit']);copies=item['copies']
 if item['mode']!='correctness':
  put(packet/'historical_timing_budget.json',{'schema_version':1,'kind':'confirmation_request_timing_budget_supplement','source_budget':budget,'copies':copies,'recommended_future_request_hours':costs[str(copies)]['recommended_future_request_hours'],'formal_plan_present':False,'walltime_still_not_frozen_as_execution_plan':True,'actual_correctness_required_before_formal_creation':True,'no_future_correctness_hash_is_present':True})
  continue
 plan=doc(unit/'measurement_plan.json');context=doc(packet/'preparation_context.json')
 assert costs[str(copies)]['maximum_actual_non_round_overhead_seconds']*2+900<3*3600,'three-hour correctness packet cannot be supported by actual historical overhead'
 script=oldscript.replace(str(qualified_packet),str(packet)).replace(str(qualified_packet.parent.parent/'data'/qualified_packet.name),str(unit)).replace('pixel-original-qualification','pixel-common-correctness').replace(image(qualified_packet.parent.parent/'data'/qualified_packet.name/'measurement_plan.json'),image(unit/'measurement_plan.json'))
 put(packet/'job.sbatch',script.encode())
 command=['sbatch','--parsable','--hold','--no-requeue','--nodes=1','--ntasks=1','--cpus-per-task=8','--mem=64G','--time=03:00:00','--partition=gpu','--account=sl5nw_base','--qos=csresnolim','--nodelist=jaguar03','--reservation=sl5nw_168','--gres=gpu:nvidia_rtx_a4500:1',str(packet/'job.sbatch')]
 put(packet/'held_submission_command.json',{'schema_version':1,'kind':'prepared_unsubmitted_common_correctness_held_command','command':command,'job_id':None,'gpu_uuid':None,'no_submission_intent_or_receipt_exists':True})
 put(packet/'command.txt',(shlex.join(command)+'\n').encode())
 artifacts=[path for path in packet.iterdir() if path.is_file() and path.name not in {'job.sbatch'}]+[path for path in unit.rglob('*') if path.is_file()]
 # Include exact actual qualification and historical operational proof bytes; live growing ledgers are not pinned here.
 prerequisite=doc(packet/'launch_prerequisites.json')
 for key,record in prerequisite.items():
  if isinstance(record,dict) and set(record)=={'path','sha256'}:
   assert image(record['path'])==record['sha256'];artifacts.append(Path(record['path']))
 checksums={str(path):image(path) for path in artifacts}
 put(packet/'source_checksums.sha256',''.join(digest+'  '+path+'\n' for path,digest in sorted(checksums.items())).encode())
 ret=subprocess.run(['bash','-n',str(packet/'job.sbatch')],capture_output=True,text=True,check=False);assert ret.returncode==0
 staged={path.name:image(path) for path in (unit/'code').iterdir() if path.is_file()}
 assert len(staged)==20 and all(staged[Path(path).name]==digest for path,digest in plan['helpers']['files'].items())
 assert not any((packet/n).exists() for n in ['submission_intent.json','submission_receipt.json','held_binding.json','release_intent.json','release_receipt.json'])
 assert not any((unit/n).exists() for n in ['configured_isolation_pre_release_inputs.json','owned_allocation_inputs.json','configured_isolation_metadata.json','resolved_device_binding.json','result.json','failure.json'])
 result={'schema_version':1,'kind':'common_correctness_launch_packet_static_verification','passed':True,'at_utc':datetime.now(timezone.utc).isoformat(),'cpu_job':'6782816','cpu_step':os.environ['SLURM_STEP_ID'],'copies':copies,'mode':'correctness','plan':ref(unit/'measurement_plan.json'),'script':ref(packet/'job.sbatch'),'startup_barrier':ref(packet/'wait_for_owned_inputs.py'),'startup_barrier_tests':ref(packet/'barrier_tests_final_receipt.json'),'staged_helpers':staged,'helper_commit':plan['helpers']['source_commit'],'source_commit':plan['source_commit'],'target_qualification_inputs':plan['target_qualification_inputs'],'held_command':command,'shell_syntax_passed':True,'startup_barrier_max_seconds':180,'frozen_gpu_uuid':plan['gpu_uuid'],'frozen_physical_identity':plan['configured_isolation']['physical_identity'],'actual_job_identity_not_yet_exists':True,'source_checksum_files':len(checksums),'filename_components_all_at_most_255':all(len(part.encode())<=255 for path in [*artifacts,packet/'slurm-999999999.err',packet/'slurm-999999999.out'] for part in path.parts),'three_hour_correctness_budget_supported_by_historical_nonround_cost':True,'confirmation_budget':budget,'correctness_credit':0,'confirmation_credit':0,'scientific_credit':0,'case_credit':0,'idea_credit':0,'admission_granted':False,'jobs_submitted':0}
 assert result['filename_components_all_at_most_255']
 put(packet/'packet_verification.json',result);created.append({'packet':str(packet),'verification':ref(packet/'packet_verification.json'),'plan':ref(unit/'measurement_plan.json'),'script':ref(packet/'job.sbatch')})
print(json.dumps({'passed':True,'budget':budget,'recommended_confirmation_hours':{k:v['recommended_future_request_hours'] for k,v in costs.items()},'created':created,'cpu_step':os.environ['SLURM_STEP_ID']}))
