"""Check only corrected saved-proof boundaries; reuse the preserved CPU lifecycle."""
import ast,copy,hashlib,json,math,os,re,sys,tempfile,time
from pathlib import Path
from types import SimpleNamespace
W=Path('/p/rlprojects/RND_worktrees/2026-11-05-13-23_profile-tool-scope_pinned-parent-exit_fresh-producer-child-proof_unknown-parents-refused_gpu-actions-0');D=Path(__file__).resolve().parent;F=Path('/p/rlprojects/RND_worktrees/2026-11-05-13-23_profile-tool-scope_pinned-parent-exit_fresh-producer-child-proof_unknown-parents-refused_gpu-actions-0/.review_metadata/profile_parent_exit_fixture/actual_positive_cpu_fixture_metadata')
S=W/'10_jax_exploration_platform/scripts/atari_pixel_campaign';FREEZE=W/'.review_metadata/profile_parent_exit_scope/runtime_source_freeze_saved_joins.json'
start=time.monotonic();checks=[]
def need(value,label):
    if not value:raise ValueError(label)
    checks.append(label)
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def ref(path):return {'path':str(path),'sha256':sha(path)}
def read(path):return json.loads(Path(path).read_text())
def extract(path,names,env):
    nodes=[n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef) and n.name in names]
    need(len(nodes)==len(names),'exact pure function extraction '+str(names));exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),env)
need(os.environ.get('SLURM_JOB_ID')=='6825983' and os.environ.get('SLURM_STEP_ID','').isdecimal(),'current owned CPU')
print(json.dumps({'actual_job_id':os.environ['SLURM_JOB_ID'],'actual_step_id':os.environ['SLURM_STEP_ID']}),flush=True)
need(sha(FREEZE)=='f867af82a50eaffc3af112e944a3571106549a1578cb0f9b943c0b0813d179ae','exact corrected freeze')
freeze=read(FREEZE)
for name,item in freeze['files'].items():need(sha(item['path'])==item['sha256'],'held source before '+name)
old=ast.parse((D/'preserved_runtime/profile_target_tool_scope.py').read_text());new=ast.parse((S/'profile_target_tool_scope.py').read_text())
for tree in (old,new):tree.body=[n for n in tree.body if not isinstance(n,ast.FunctionDef) or n.name not in ('validate_parent_exit_record','validate_saved_scope')]
need(ast.dump(old,include_attributes=False)==ast.dump(new,include_attributes=False),'only two saved reader functions changed from preserved lifecycle source')
for name in ('profile_target_process.py','profile_target_direct_process.py'):
    need((S/name).read_bytes()==(D/'preserved_runtime'/name).read_bytes(),'AA/H live and saved caller wiring unchanged from initial '+name)
runtime_freeze=read(Path('/p/rlprojects/RND_worktrees/2026-11-05-09-42_profile-target-qualification_original-tests-48_environments-128_rollout-steps-128_warmup-rollouts-50_reference-updates-1_default-admission-closed/.review_metadata/profile_target_qualification/union_source_freeze_2_corrected_runtime_interfaces.json'))
for name,wanted in runtime_freeze['protected_existing_files'].items():
    path=W/name
    if not path.exists():path=Path('/p/rlprojects/RND_worktrees/merge-atari-cleanrl')/name
    need(sha(path)==wanted,'old AA7/other protected/13NUM '+name)
need(sha(S/'profile_selected.py')==runtime_freeze['runtime_and_test_files']['10_jax_exploration_platform/scripts/atari_pixel_campaign/profile_selected.py']['sha256'],'capture core containing source unchanged')
test=S/'tests/test_profile_target_tool_scope_parent_exit.py'
need(sha(test)=='58e565aac53636c00bcd3c24296214394e45408657ca2688b3b1b6b3f0e5e2b0','exact held fixture API update')
ast.parse(test.read_text());need("'open_generation':retained['generation']" in test.read_text() and 'producer_terminal_time=time.time()' in test.read_text(),'future fixture records actual close generation and updated API; branch not executed')
env={'hashlib':hashlib,'json':json,'math':math,'Path':Path}
tree=ast.parse((S/'profile_target_tool_scope.py').read_text())
for node in tree.body:
    if isinstance(node,ast.Assign):
        for target in node.targets:
            if isinstance(target,ast.Name) and target.id in ('PROTOCOL','NAME','PREFIX','EXIT_REFUSALS'):env[target.id]=ast.literal_eval(node.value)
extract(S/'profile_target_tool_scope.py',{'require','same','validate_parent_exit_record','validate_saved_scope'},env)
be={};extract(S/'direct_tool_descendants.py',{'require','parsed_identity'},be)
pe={'re':re,'os':os};extract(S/'linux_pidfd_process_control.py',{'require','parse_pidfd_fdinfo'},pe)
base=SimpleNamespace(MAX_EVENTS=100000,parsed_identity=be['parsed_identity']);parser=SimpleNamespace(parse_pidfd_fdinfo=pe['parse_pidfd_fdinfo'])
observer=SimpleNamespace(receipt=ref,read_json=read,parsed_uids=lambda raw:[int(v) for line in raw.splitlines() if line.startswith('Uid:') for v in line.split()[1:]])
terminal=read(F/'tool_descendant_scope_terminal.json');metadata=read(F/'fixture_observer_metadata.json');producer=metadata['owner']
original=read(F/'tool_parent_exit_observation_1.json');events=copy.deepcopy(metadata['open_events']);close_time=terminal['terminal']['observed_at_seconds']+.01;role_time=close_time+.01
for index,event in enumerate(events,1):event['open_generation']=index
opened=next(event for event in events if event['descriptor']==original['handle_state']['descriptor'] and event['opened_pid']==original['parent_pid'])
events.append({**copy.deepcopy(opened),'operation':'close','observed_at_seconds':close_time})
env['validate_parent_exit_record'](original,terminal,producer,base,observer,events,parser,producer_terminal_time=role_time)
need(True,'positive real raw CPU record with explicitly synthetic close/generation/role premises')
def rejects(record=original,ev=events,end=role_time,label='negative'):
    try:env['validate_parent_exit_record'](record,terminal,producer,base,observer,ev,parser,producer_terminal_time=end)
    except (ValueError,KeyError):need(True,label);return
    raise AssertionError('incorrectly accepted '+label)
late=copy.deepcopy(original);t=terminal['terminal']['observed_at_seconds']+10000
late['handle_state']['observed_at_seconds']=t;late['observed_at_seconds']=t+1;late['fresh_producer_children'].update(started_at_seconds=t+2,finished_at_seconds=t+3)
rejects(late,label='original post-terminal timing mutation rejected')
bad=copy.deepcopy(events);bad[-1]['open_generation']+=1;rejects(ev=bad,label='wrong close generation rejected')
rejects(ev=events[:-1],label='missing actual close rejected')
rejects(end=close_time-.001,label='role terminal before close rejected')
bad=copy.deepcopy(events);bad[-1]['observed_at_seconds']=terminal['terminal']['observed_at_seconds']-.001;rejects(ev=bad,label='close before scope closure rejected')
badrec=copy.deepcopy(original);badrec['fresh_producer_children']['finished_at_seconds']=terminal['terminal']['observed_at_seconds']+1;rejects(badrec,label='fresh adoption proof after terminal rejected')
for key,value in (('parent_birth','1'),):
    badrec=copy.deepcopy(original);badrec[key]=value;rejects(badrec,label='changed known parent birth rejected')
badrec=copy.deepcopy(original);badrec['refusal']['message']='generic or unknown parent failure';rejects(badrec,label='generic error is not accepted as known exit')
env['base_module']=lambda plan:base;env['adjacent']=lambda plan,name:parser
with tempfile.TemporaryDirectory(prefix='corrected_new_reader_only_') as temporary:
    directory=Path(temporary);observation=directory/'tool_parent_exit_observation_1.json';observation.write_text(json.dumps(original))
    terminal_copy=copy.deepcopy(terminal)
    terminal_events=[e for e in terminal_copy['events'] if e.get('kind')=='known_pinned_parent_exit_reconciled']
    need(len(terminal_events)==1,'one preserved actual reconciliation event')
    terminal_events[0]['observation']=ref(observation)
    tp=directory/'terminal.json';tp.write_text(json.dumps(terminal_copy))
    legacy={'tool_descendant_scope_begin':ref(F/'tool_descendant_scope_begin.json'),'tool_descendant_scope_terminal':ref(tp),'tool_descendant_scope_restored':ref(F/'tool_descendant_scope_restored.json')}
    base.validate_saved_scope=lambda *args,**kwargs:legacy
    manifest=read(F/'tool_parent_exit_observations.json');manifest.update(adapter_sha256=sha(S/'profile_target_tool_scope.py'),records=[ref(observation)],scope_terminal=legacy['tool_descendant_scope_terminal'],scope_restored=legacy['tool_descendant_scope_restored'])
    mp=directory/'tool_parent_exit_observations.json';mp.write_text(json.dumps(manifest))
    role={'role':'producer','completed':True,'kind':'actual_profile_target_process_role_terminal','provider':{'descriptor_events':events},'completed_at_seconds':role_time,'explicit_synthetic_fixture_premise':True}
    rp=directory/'profile_target_process_producer_terminal.json';rp.write_text(json.dumps(role))
    plan={'kind':'full_original_profile_source_target_qualification','helpers':{'files':{env['PREFIX']+'profile_target_tool_scope.py':manifest['adapter_sha256'],env['PREFIX']+'direct_tool_descendants.py':manifest['base_sha256']}}}
    env['validate_saved_scope'](directory,plan,observer,producer,0,'unused',lambda r:None);need(True,'positive new manifest with explicit passed legacy/role premises')
    observation.unlink();manifest['records']=[];mp.write_text(json.dumps(manifest))
    try:env['validate_saved_scope'](directory,plan,observer,producer,0,'unused',lambda r:None)
    except ValueError:need(True,'original omitted event observation rejected')
    else:raise AssertionError('omitted immutable terminal event was accepted')
    observation.write_text(json.dumps(original));manifest['records']=[ref(observation)];mp.write_text(json.dumps(manifest))
    original_event=copy.deepcopy(terminal_events[0])
    for key,value in (('pid',original['parent_pid']+1),('birth','1'),('discarded_child_list_valid',True),('at_seconds',terminal['terminal']['observed_at_seconds']+1)):
        terminal_events[0].update(original_event);terminal_events[0][key]=value;tp.write_text(json.dumps(terminal_copy));legacy['tool_descendant_scope_terminal']=ref(tp);manifest['scope_terminal']=legacy['tool_descendant_scope_terminal'];mp.write_text(json.dumps(manifest))
        try:env['validate_saved_scope'](directory,plan,observer,producer,0,'unused',lambda r:None)
        except ValueError:need(True,'event '+key+' mismatch rejected')
        else:raise AssertionError('bad event accepted '+key)
for name,item in freeze['files'].items():need(sha(item['path'])==item['sha256'],'held source after '+name)
need(time.monotonic()-start<285,'bounded review')
record={'schema_version':1,'kind':'independent_profile_parent_exit_scope_runtime_source_review','approved':True,'passed':True,'independent_of_author':True,
    'source_freeze':ref(FREEZE),'files':freeze['files'],'fixture_source':ref(test),'actual_job_id':os.environ['SLURM_JOB_ID'],'actual_step_id':os.environ['SLURM_STEP_ID'],
    'checks':checks,'check_count':len(checks),'prior_lifecycle_proof_reused':True,'lifecycle_repeated':False,'old_AA7_NUM_capture_core_unchanged':True,
    'saved_control_premises':'Preserved real CPU106 raw PID/birth/UID/cgroup/adoption/fdinfo/terminal bytes; synthetic open-generation/close/completed-role times and relocated private refs isolate NEW saved proof semantics. No close times were added to historical captures. Legacy/full native-role validation is an explicit fixture premise, not target qualification.',
    'actual_target_qualified':False,'GPU_or_native_process_operations_this_review':False,'future_fresh_H_and_AA_qualifications_required':True,
    'new_helper_counts':{'h100':45,'jaguar03_rtx_a4500':36},'scientific_credit':0,'admission_granted':False,'seconds':time.monotonic()-start}
path=D/'independent_corrected_saved_reader_review.json'
with path.open('x') as stream:json.dump(record,stream,indent=2,sort_keys=True);stream.write('\n');stream.flush();os.fsync(stream.fileno())
print(json.dumps({'approved':True,'review':ref(path),'step':record['actual_step_id'],'checks':len(checks),'seconds':record['seconds']}))
