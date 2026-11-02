"""Derive per-job held proofs only from actual saved CPU, Cron and query records."""
import datetime,hashlib,json,os
from pathlib import Path
P=Path("/p/rlprojects/RND_worktrees/merge-atari-cleanrl/10_jax_exploration_platform/benchmark_runs/2026-10-31-00-11_jaxatari_montezuma_pixels_cleanrl_ppo_rnd_environments-per-copy-128_rollout-steps-128_ideas-30-plus-20_steps-per-copy-2000000000_seed-start-1/remote_setup/2026-11-01-19-24_common-confirmation-release_jobs-6783295-6783296_paired-rounds-7_updates-per-arm-10_copies-1-2_gpu-count-1_processor-count-8_host-memory-gib-64_time-limit-hours-2-3");R=Path("/p/rlprojects/RND_worktrees/merge-atari-cleanrl/10_jax_exploration_platform/benchmark_runs/2026-10-31-00-11_jaxatari_montezuma_pixels_cleanrl_ppo_rnd_environments-per-copy-128_rollout-steps-128_ideas-30-plus-20_steps-per-copy-2000000000_seed-start-1")
assert os.environ.get('SLURM_JOB_ID')=='6783003' and os.environ.get('SLURM_STEP_ID','').isdigit()
def ref(p):return {'path':str(p),'sha256':hashlib.sha256(Path(p).read_bytes()).hexdigest()}
def read(p):return json.loads(Path(p).read_bytes())
def put(p,value):
 b=(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
 with p.open('xb') as f:f.write(b);f.flush();os.fsync(f.fileno())
 return ref(p)
next_actions=read(R/'next_actions.json');monitor=next_actions['persistent_monitor'];M=Path(monitor['packet'])
V=M/'scheduled_check_2026-11-02-03-31-00_verification.json';S=M/'scheduled_check_2026-11-02-03-31-00_status.json';REPORT=M/'scheduled_check_2026-11-02-03-31-00_report.md'
verification,status=read(V),read(S);cron_path=R/'remote_setup/monitoring_task_2026-11-02-03-29.json';cron=read(cron_path)
assert verification['passed'] is True and verification['monitor_job_id']=='6783003' and verification['step_id']=='66' and verification['source_commit']=='79a01e851ff92ddba7d0e12eaf8164cab2028878'
assert verification['queue_counts']=={'pending':0,'running':0,'completed':120,'failed':0} and verification['report_sha256']==ref(REPORT)['sha256'] and verification['status_sha256']==ref(S)['sha256']
assert all(label in REPORT.read_text().splitlines() for label in ('TRAINING','WAITING','HELD','FINISHED','PLANNED'))
assert [row for row in status['jobs'] if row[0]=='6783003']==[['6783003','RUNNING','heartpiece','gres/gpu:0','heartpiece']]
assert cron['id']=='bd8f17a1-31a1-4ae5-9a30-af1132e14c3c' and cron['cron']=='*/20 * * * *' and cron['recurring'] is True
scopes=read(P/'draft_scope.json')['jobs'];queries=read(P/'fresh_held_query_index.json')['records'];results={}
for copies in (1,2):
 scope=scopes[str(copies)];packet,unit=Path(scope['packet']),Path(scope['unit']);job=scope['job_id']
 capture=next(row for row in queries if row['job_id']==job)
 for key in ('return','stdout','stderr'):assert ref(capture[key]['path'])==capture[key]
 query=read(capture['return']['path']);assert query['job_id']==job and query['command']==['scontrol','show','job','-o',job] and type(query['returncode']) is int and query['returncode']==0
 assert query['stdout']==capture['stdout'] and query['stderr']==capture['stderr'] and Path(capture['stderr']['path']).read_bytes()==b''
 row=[job,'PENDING','','gres/gpu:nvidia_rtx_a4500:1','(JobHeldUser)'];assert [r for r in status['jobs'] if r[0]==job]==[row]
 qtime=datetime.datetime.fromisoformat(query['at_utc']);mtime=datetime.datetime.fromisoformat(verification['checked_at_utc'])
 now=datetime.datetime.now(datetime.timezone.utc);assert 0<=(now-qtime).total_seconds()<1200 and 0<=(now-mtime).total_seconds()<1200
 proof={'schema_version':1,'kind':'owned_uva_configured_isolation_actual_full_held_monitor_proof','passed':True,'job_id':job,'copies':copies,'state':'PENDING','reason':'JobHeldUser','publication_commit':'4ef27e7e46bce2b64ea6b8050935f88986ded12e','helper_source_commit':'9fcc43a478b93d731df780805fd7c3c514cdf67f','at_utc':query['at_utc'],'derived_at_utc':now.isoformat(),'expires_at_utc':(min(qtime,mtime)+datetime.timedelta(seconds=1200)).isoformat(),'submission':ref(packet/'submission_receipt.json'),'plan':ref(unit/'measurement_plan.json'),'job_script':ref(packet/'job.sbatch'),'pre_release_inputs':ref(unit/'configured_isolation_pre_release_inputs.json'),'held_binding':ref(packet/'held_binding.json'),'owned_query_return':capture['return'],'owned_query_stdout':capture['stdout'],'owned_query_stderr':capture['stderr'],'root_resource_review':ref(packet/'root_actual_held_resource_review.json'),'query_capture_index':ref(P/'fresh_held_query_index.json'),'monitor':{'job_id':'6783003','source_commit':verification['source_commit'],'step_id':verification['step_id'],'cadence_seconds':1200,'persistent_running_verified':True,'runtime_and_first_full_report_review':ref(Path(monitor['runtime_and_first_full_report_review'])),'full_verification':ref(V),'full_status':ref(S),'full_report':ref(REPORT),'row':row},'recurring_session_task':{'id':cron['id'],'cron':cron['cron'],'cadence_seconds':1200,'exact_prompt_and_schedule_verified_before_old_deleted':True,'actual_record':ref(cron_path)},'frozen_physical_identity_remains_unresolved':True,'released':False,'scientific_credit':0,'case_credit':0,'idea_credit':0,'admission_granted':False,'root_source_for_cron_verification':'explicit parent message confirming actual CronList verification before deleting the predecessor','new_scheduler_calls':0,'proof_is_metadata_not_a_release':True}
 results[str(copies)]=put(P/('held_monitor_proof_copies_'+str(copies)+'.json'),proof)
print(json.dumps({'passed':True,'proofs':results,'expiry_utc':proof['expires_at_utc'],'job':'6783003','step':os.environ['SLURM_STEP_ID']}))

