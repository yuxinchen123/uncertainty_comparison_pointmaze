import ast,datetime,hashlib,json,os,pathlib,subprocess,time
D=pathlib.Path('/p/rlprojects/RND_worktrees/2026-11-05-10-14_profile-target-process_native-provider_separate-role-contracts_prior-abi-baseline-only_admission-closed/.review_metadata/task_roster_scope_independent_review')
W=pathlib.Path('/p/rlprojects/RND_worktrees/2026-11-05-15-11_profile-task-roster_known-live-parent_enumerated-thread-exit_fresh-complete-read_gpu-actions-0')
assert os.environ['SLURM_JOB_ID']=='6825983'
source=D/'preserved_initial_runtime_50c170/profile_target_tool_scope.py';freeze=D/'preserved_initial_runtime_50c170/runtime_source_freeze_1.json';control=D/'independent_initial_pin_chronology_control.json'
s=source.read_text();f=json.loads(freeze.read_text());r=json.loads(control.read_text());assert r['reproduced_gap']is True and r['source_sha256']==hashlib.sha256(source.read_bytes()).hexdigest()
env={k:v for k,v in os.environ.items()if not k.startswith('GIT_')};env['GIT_OPTIONAL_LOCKS']='0'
base=subprocess.check_output(['git','-C',str(W),'show','4bb3a61cb92fe48ab5ba68fb52f70b47736a1174:10_jax_exploration_platform/scripts/atari_pixel_campaign/profile_target_tool_scope.py'],env=env).decode()
funcs=lambda x:{n.name:ast.get_source_segment(x,n)for n in ast.parse(x).body if isinstance(n,ast.FunctionDef)}
before,after=funcs(base),funcs(s)
unchanged=[name for name in before if before[name]==after.get(name)]
assert 'validate_parent_exit_record'in unchanged and 'handle_state'in unchanged
other={}
for rel,sha in f['all_other_staged_runtime_sources_unchanged'].items():
 assert hashlib.sha256((W/rel).read_bytes()).hexdigest()==sha,rel
 other[rel]=sha
record=dict(schema_version=1,kind='independent_task_roster_initial_runtime_source_review',approved=False,passed=False,independent_of_author=True,source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),source_freeze_sha256=hashlib.sha256(freeze.read_bytes()).hexdigest(),preserved_source=str(source),preserved_source_freeze=str(freeze),actual_job_id=os.environ['SLURM_JOB_ID'],actual_step_id=os.environ['SLURM_STEP_ID'],reviewer_pid=os.getpid(),reviewer_uid=os.getuid(),reviewed_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),blocking_findings=[dict(finding='Fresh positive child pin work occurs after recorded retry elapsed/finish; no final same-live-parent proof follows the pins.',independently_reproduced=True,control={'path':str(control),'sha256':hashlib.sha256(control.read_bytes()).hexdigest()},required_change='Measure actual completion and elapsed after all pins, retain final same-parent live handle, and require saved child identity observations inside that completed window.')],unchanged_existing_top_level_function_source_texts=unchanged,other_runtime_source_hashes=other,actual_CPU_lifecycle_executed_by_this_review=False,actual_GPU_native_tool_or_remote_operations=False,actual_target_qualification=False,scientific_credit=0,admission_granted=False,remaining_review='Corrected immutable source and distinct actual own-CPU task-expiry fixture still require review; no approval of the initial source.')
out=D/'independent_runtime_1_block_review.json';raw=(json.dumps(record,indent=2,sort_keys=True)+'\n').encode()
with out.open('xb')as f:f.write(raw);f.flush();os.fsync(f.fileno())
print(json.dumps({'path':str(out),'sha256':hashlib.sha256(raw).hexdigest(),'step':record['actual_step_id'],'protected_other_sources':len(other)}))

