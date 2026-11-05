import ast,hashlib,json,os,shlex,subprocess,time
from pathlib import Path
W=Path('/p/rlprojects/RND_worktrees/2026-11-05-06-20_a4500-native-process-control_python-3.12_total-seconds-300_work-seconds-240_cleanup-seconds-60_admission-closed')
D=W/'.review_metadata/configured_native_process';S=W/'10_jax_exploration_platform/scripts/atari_pixel_campaign'
R=Path('/p/rlprojects/RND_worktrees/merge-atari-cleanrl/10_jax_exploration_platform/benchmark_runs/2026-10-31-00-11_jaxatari_montezuma_pixels_cleanrl_ppo_rnd_environments-per-copy-128_rollout-steps-128_ideas-30-plus-20_steps-per-copy-2000000000_seed-start-1')
assert os.environ['SLURM_JOB_ID']=='6821437';assert '6821437' in (R/'slurm/submitted_jobids_uva.txt').read_text().splitlines()
started=time.monotonic();checks=[]
def check(label,predicate):
 assert predicate,label
 checks.append(label)
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
freeze=D/'corrected_union_source_freeze.json';check('exact held source freeze',sha(freeze)=='36524765c7fba07fad274b0ee170d03df05ead3ea2a1217ffbbfb22ecd84dd39')
f=json.loads(freeze.read_bytes())
for rel,row in f['files'].items():
 p=W/rel;check('held hash '+rel,sha(p)==row['sha256']);check('held mode '+rel,oct(p.stat().st_mode&0o777)==row['mode'])
focused=D/'independent_corrected_union_focused_tests.json';check('actual independent focused test receipt',sha(focused)=='f527f0489cfab312c733b9b91ca61b7f3be1d24668f0995e27f440f4684d0f63')
tested=json.loads(focused.read_bytes());check('all thirty affected tests passed without failure',tested['passed'] is True and tested['test_count']==30 and tested['failures']==tested['errors']==0)
profile_static=W/'.review_metadata/profile_selected_native_slice/independent_static_review.json'
check('actual profile static receipt',sha(profile_static)=='57b472c5d8f46c3c77abf8985b32c7bbca9636fad8d344f082f594a8c13d0573')
check('exact held profile preserved since static review',sha(S/'profile_selected.py')=='59aa781caeb66ceed38def4c279480d65f437a949676f7150056a693e272c22a')
for name in ('linux_pidfd_process_control.py','direct_profile_process.py'):
 rel='10_jax_exploration_platform/scripts/atari_pixel_campaign/'+name
 q=subprocess.run(['git','-C',str(W),'show',f['base']+':'+rel],capture_output=True,check=True)
 check('unchanged default native/process source '+name,q.stdout==(S/name).read_bytes())
H=next((R/'data').glob('2026-11-05-05-49_*'));plan=json.loads((H/'measurement_plan.json').read_bytes())
check('actual live numerical source unchanged name',plan['source_commit']=='d28e6735e79788f618acb0ebfde8609db1f776af')
check('actual live helper unchanged name',plan['helpers']['source_commit']=='69f1b5075530418adbe02c1e433c5647848a61ba')
for key in ('core','paired_source','helpers'):
 entry=plan[key]
 for rel,row in entry['files'].items():
  check('actual protected '+key+' '+rel,sha(Path(entry['source_root'])/rel)==(row['sha256'] if isinstance(row,dict) else row))
check('thirteen numerical and paired source files preserved',len(plan['core']['files'])+len(plan['paired_source']['files'])==13)
proposal=D/'qualification_packet_proposal';proposal_freeze=proposal/'proposal_freeze.json';pr=json.loads(proposal_freeze.read_bytes())
check('exact proposal freeze',sha(proposal_freeze).startswith('7551798251'))
for name,expected in pr['files'].items():check('proposal hash '+name,sha(proposal/name)==expected)
config=json.loads((proposal/'config_resolved.yaml').read_bytes());manifest=json.loads((proposal/'manifest.yaml').read_bytes());prep=json.loads((proposal/'preparation_arguments.json').read_bytes());procedure=json.loads((proposal/'root_procedure_proposal.json').read_bytes())
check('matching standard config and manifest',config==manifest)
check('uncreated proposed packet and unit',pr['actual_packet_not_created'] is True and pr['actual_data_unit_not_created'] is True and not Path(config['packet']).exists() and not Path(config['data_unit']).exists())
check('no invented future source or identities',all(config[k] is None for k in ('reviewed_helper_commit','actual_job_id','actual_step_id','gpu_uuid')) and prep['source_root'] is None)
check('exact selected python',config['selected_python']=='/p/rlprojects/RND/.venvs/atari_pixels_jax/bin/python')
check('five minute one GPU diagnostic resources',config['requested_gpu_count']==config['requested_cpu_count']==1 and config['host_memory_gib']==2 and config['maximum_active_seconds']==240 and config['diagnostic_time_budget']=={'schema_version':1,'work_seconds':240,'cleanup_seconds':60,'total_seconds':300})
check('zero credit and no mapping or descriptor claim',all(config[k]==0 for k in ('environment_transitions','training_updates','scientific_credit','profile_credit')) and all(config[k] is False for k in ('physical_gres_mapping_verified','kernel_device_enforcement_verified','admission_granted','standing_gpu_descriptor_held')))
check('actual tool metadata bytes',sha(config['profile_tool_record']['path'])==config['profile_tool_record']['sha256'])
command=shlex.split((proposal/'command.txt').read_text());check('exact isolated diagnostic producer command',command==[config['selected_python'],'-I','-S','-B',str(Path(config['data_unit'])/'code/diagnose_configured_native_process.py'),'--prepared-unit',config['data_unit']])
job=(proposal/'job.sbatch.proposed').read_text();check('frozen script ends in exact producer command',shlex.join(command) in job)
for text in ('#SBATCH --time=00:05:00','#SBATCH --cpus-per-task=1','#SBATCH --mem=2G','#SBATCH --nodelist=jaguar03','#SBATCH --reservation=sl5nw_168','#SBATCH --qos=csresnolim','#SBATCH --gres=gpu:nvidia_rtx_a4500:1','#SBATCH --no-requeue'):
 check('exact script resource '+text,text in job)
check('owned numeric step execution', 'exec srun --nodes=1 --ntasks=1 --cpus-per-task=1 --cpu-bind=cores --gres=gpu:nvidia_rtx_a4500:1 --kill-on-bad-exit=1 ' in job)
check('submission proposal held only with no guessed release id',procedure['submission_command'][:3]==['sbatch','--hold','--parsable'] and procedure['release_command_before_actual_id_exists'] is None)
check('fresh reservation and publication still required',config['fresh_exact_reservation_check_required'] is True and prep['no_actual_preparation_called'] is True and 'publish' in procedure['ordered_steps'][0].lower())
background=(proposal/'experiment_background.md').read_text()
for label in ('Purpose','Key hyperparameters','Code and config changes','Git state'):check('standard background section '+label,'## '+label+'\n' in background)
for rel,row in f['files'].items():check('unchanged source after final review '+rel,sha(W/rel)==row['sha256'])
review={'kind':'independent_configured_native_union_source_review','approved':True,'passed':True,'independent_of_author':True,'source_freeze':{'path':str(freeze),'sha256':sha(freeze)},'focused_tests':{'path':str(focused),'sha256':sha(focused),'methods':30},'profile_static_review':{'path':str(profile_static),'sha256':sha(profile_static),'checks':39},'proposal_freeze':{'path':str(proposal_freeze),'sha256':sha(proposal_freeze)},'checks':checks,'check_count':len(checks),'seconds':time.monotonic()-started,'job_id':os.environ['SLURM_JOB_ID'],'step_id':os.environ['SLURM_STEP_ID'],'uid':os.getuid(),'pid':os.getpid(),'cgroup':Path('/proc/self/cgroup').read_text(),'actual_target_qualification':False,'formal_profile_admission_granted':False,'scientific_credit':0,'approval_scope':'Reviewed source and unprepared five-minute qualification proposal only. Publish exact reviewed source and actual standard records before submission. Actual target process qualification and independent raw review remain required before a separate formal profiling admission change.','required_before_submission':['published reviewed source commit and actual clean source checkout','canonical preparation once with real generated plan and helper hashes','fresh exact reservation/resource check','independently reviewed and published held-submit and real-ID release utilities','real returned IDs and row fsynced before release','actual current monitor verification'],'preserved_failures':['independent helper rejection 29a950f3f53999e44635207ba0ba2e09971744f7b627323ba65307525eca95b3','original author metadata fixtures and unchanged historical fixture-schema failures','local independent test-writer syntax error before any test or source write'],'work_performed':'Static source/record reading only in this check. Earlier declared CPU native fixtures used own libc/prctl/pidfd operations and own CPU children only; no target, Nsight, numerical, GPU, device or scientific work.'}
p=D/'independent_corrected_union_source_and_proposal_review.json'
with p.open('x') as out:json.dump(review,out,indent=2);out.write('\n');out.flush();os.fsync(out.fileno())
print(json.dumps({'review_path':str(p),'review_sha256':sha(p),'check_count':len(checks),'seconds':review['seconds'],'job_id':review['job_id'],'step_id':review['step_id']}))
