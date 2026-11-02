"""Bind only actual fresh held proofs; never query, release or change the frozen plans."""
import argparse,datetime,hashlib,json,os
from pathlib import Path
P=Path("/p/rlprojects/RND_worktrees/merge-atari-cleanrl/10_jax_exploration_platform/benchmark_runs/2026-10-31-00-11_jaxatari_montezuma_pixels_cleanrl_ppo_rnd_environments-per-copy-128_rollout-steps-128_ideas-30-plus-20_steps-per-copy-2000000000_seed-start-1/remote_setup/2026-11-01-19-24_common-confirmation-release_jobs-6783295-6783296_paired-rounds-7_updates-per-arm-10_copies-1-2_gpu-count-1_processor-count-8_host-memory-gib-64_time-limit-hours-2-3");R=Path("/p/rlprojects/RND_worktrees/merge-atari-cleanrl/10_jax_exploration_platform/benchmark_runs/2026-10-31-00-11_jaxatari_montezuma_pixels_cleanrl_ppo_rnd_environments-per-copy-128_rollout-steps-128_ideas-30-plus-20_steps-per-copy-2000000000_seed-start-1")
def raw(p):return Path(p).read_bytes()
def ref(p):return {'path':str(p),'sha256':hashlib.sha256(raw(p)).hexdigest()}
def record(p):
 p=Path(p);assert p.is_absolute() and p.resolve(strict=True)==p and p.is_file()
 b=raw(p);assert len(b)<=64*1024*1024
 return {'sha256':hashlib.sha256(b).hexdigest(),'bytes':len(b)}
def put(name,value,bytes_only=False):
 p=P/name;b=value if bytes_only else (json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
 with p.open('xb') as f:f.write(b);f.flush();os.fsync(f.fileno())
 return ref(p)
parser=argparse.ArgumentParser();parser.add_argument('--proof-copies-1',type=Path,required=True);parser.add_argument('--proof-copies-2',type=Path,required=True);args=parser.parse_args()
assert os.environ.get('SLURM_JOB_ID')=='6783003' and os.environ.get('SLURM_STEP_ID','').isdigit()
draft=json.loads((P/'draft_scope.json').read_bytes());scopes=draft['jobs']
submit_dir=R/'remote_setup/2026-11-01-18-57_common-confirmation-held-submission_paired-rounds-7_updates-per-arm-10_copies-1-2_gpu-count-1_processor-count-8_host-memory-gib-64_time-limit-hours-2-3'
submit=json.loads((submit_dir/'submission_inputs.json').read_bytes())
progress={str(Path(scope['unit'])/'progress.json') for scope in scopes.values()}
shared={v['path']:record(v['path']) for v in submit['required_files'] if v['path'] not in progress}
assert all(ref(v['path'])==v for v in submit['required_files'])
assert len(submit['required_files'])-len(shared)==2
jobs={}
for copies,proof_path in ((1,args.proof_copies_1),(2,args.proof_copies_2)):
 scope=scopes[str(copies)];packet=Path(scope['packet']);unit=Path(scope['unit'])
 proof=json.loads(raw(proof_path));assert proof['passed'] is True and proof['job_id']==scope['job_id']
 assert proof['helper_source_commit']=='9fcc43a478b93d731df780805fd7c3c514cdf67f' and proof['publication_commit']=='4ef27e7e46bce2b64ea6b8050935f88986ded12e'
 observed=datetime.datetime.fromisoformat(proof['at_utc'].replace('Z','+00:00'));assert 0<=(datetime.datetime.now(datetime.timezone.utc)-observed).total_seconds()<1200
 assert proof['monitor']['job_id']=='6783003' and proof['monitor']['source_commit']=='79a01e851ff92ddba7d0e12eaf8164cab2028878'
 own={}
 def collect(value):
  if isinstance(value,dict):
   if set(value)=={'path','sha256'}:
    assert ref(value['path'])==value
    if value['path'] not in shared:own[value['path']]=record(value['path'])
   else:
    for v in value.values():collect(v)
  elif isinstance(value,list):
   for v in value:collect(v)
 collect(proof);own[str(proof_path)]=record(proof_path)
 for name in ('submission_intent.json','submission_stdout.raw','submission_stderr.raw','submission_return.json','submission_receipt.json','held_binding.json','root_actual_held_resource_review.json'):
  own[str(packet/name)]=record(packet/name)
 for name in ('progress.json','configured_isolation_owned_job_ids.txt','configured_isolation_owned_submissions.tsv','configured_isolation_pre_release_inputs.json'):
  own[str(unit/name)]=record(unit/name)
 own={p:v for p,v in own.items() if p not in shared}
 shape=json.loads((packet/'root_actual_held_resource_review.json').read_bytes())
 assert shape['passed'] is True and shape['job_id']==scope['job_id'] and shape['fields']['TimeLimit']=={1:'02:00:00',2:'03:00:00'}[copies]
 jobs[str(copies)]={**scope,'files':own,'plan':ref(unit/'measurement_plan.json'),'script':ref(packet/'job.sbatch'),
  'pre_release_inputs':ref(unit/'configured_isolation_pre_release_inputs.json'),'submission':ref(packet/'submission_receipt.json'),
  'held_binding':ref(packet/'held_binding.json'),'held_proof':ref(proof_path),'recurring_session_task_id':proof['recurring_session_task']['id']}
assert all(not (Path(scope['packet'])/'release_intent.json').exists() and not (Path(scope['unit'])/'owned_allocation_inputs.json').exists() for scope in scopes.values())
frozen=put('release_frozen_inputs.json',{'schema_version':1,'kind':'fixed_common_confirmation_actual_release_inputs','source_commit':'9fcc43a478b93d731df780805fd7c3c514cdf67f','initial_publication':'4ef27e7e46bce2b64ea6b8050935f88986ded12e','shared_files':shared,'jobs':jobs,'selected_target_only_progress_and_ownership':True,'maximum_startup_wait_seconds':180,'actual_releases':0,'admission_granted':False})
source=P/'release_common_confirmation.py';before=source.read_bytes();assert before.count(b'FROZEN_INPUTS_SHA = None')==1
put('unbound_release_source_before_actual_full_monitor_proofs.py',before,True)
after=before.replace(b'FROZEN_INPUTS_SHA = None',('FROZEN_INPUTS_SHA = '+repr(frozen['sha256'])).encode())
with source.open('wb') as f:f.write(after);f.flush();os.fsync(f.fileno())
put('author_review.json',{'schema_version':1,'kind':'fixed_common_confirmation_release_implementation_review','source_sha256':ref(source)['sha256'],'jobs':{k:v['job_id'] for k,v in jobs.items()},'release_executed':False,'maximum_startup_wait_seconds':180,'real_monitor_proofs':{k:v['held_proof'] for k,v in jobs.items()},'shared_static_file_count':len(shared),'selected_target_only_mutable_state':True,'frozen_inputs':frozen,'NUM_work':False,'admission_granted':False})
runner=(R/'remote_setup/2026-11-01-16-33_common-correctness-release-new-monitor_jobs-6782946-6782947_copies-1-2_gpu-count-1_processor-count-8_host-memory-gib-64_time-limit-hours-3'/'run_author_controls.py').read_text().replace('release_common_correctness','release_common_confirmation').replace('common_correctness_release','common_confirmation_release').replace('6782946','6783295').replace('6782947','6783296')
runner=runner.replace("frozen['files'].items()", "{**frozen['shared_files'],**frozen['jobs']['1']['files'],**frozen['jobs']['2']['files']}.items()")
runner=runner.replace("'/p/rlprojects/RND_worktrees/2026-11-01-13-50_full-original-qualification-explicit-pytest-root_source-commit-e5afbf6b51a3d1666b46e71c0631a45d47080cd8'","'/p/rlprojects/RND_worktrees/2026-11-01-18-47_common-confirmation-source_commit-9fcc43a478b93d731df780805fd7c3c514cdf67f'")
put('run_author_controls.py',runner.encode(),True)
print(json.dumps({'passed':True,'source':ref(source),'tests':ref(P/'test_release_common_confirmation.py'),'inputs':frozen,'shared_files':len(shared),'owned_files':{k:len(v['files']) for k,v in jobs.items()},'actual_releases':0}))

