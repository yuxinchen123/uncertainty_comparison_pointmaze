"""Independent frozen source and portable saved-CPU-metadata review; no lifecycle is executed."""
import ast,copy,ctypes,datetime,hashlib,importlib.util,json,math,os,pathlib,stat,subprocess,sys,time
START=time.monotonic();CHECKS=[];FILES={}
W=pathlib.Path('/p/rlprojects/RND_worktrees/2026-11-05-15-11_profile-task-roster_known-live-parent_enumerated-thread-exit_fresh-complete-read_gpu-actions-0');D=W/'.review_metadata/profile_task_roster';PREFIX='10_jax_exploration_platform/scripts/atari_pixel_campaign/'
OUTDIR=pathlib.Path('/p/rlprojects/RND_worktrees/2026-11-05-10-14_profile-target-process_native-provider_separate-role-contracts_prior-abi-baseline-only_admission-closed/.review_metadata/task_roster_scope_independent_review')
def check(v,label):
 if not v:raise ValueError(label)
 CHECKS.append(label)
def read(p):
 p=pathlib.Path(p);s=p.lstat();check(p.resolve(strict=True)==p and stat.S_ISREG(s.st_mode)and s.st_size<=2500000,'bounded ordinary '+str(p));raw=p.read_bytes();t=p.lstat();check((s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)==(t.st_dev,t.st_ino,t.st_size,t.st_mtime_ns,t.st_ctime_ns)and len(raw)==s.st_size,'stable exact '+str(p));ref={'path':str(p),'sha256':hashlib.sha256(raw).hexdigest(),'size_bytes':len(raw)};FILES[str(p)]=ref;return raw,ref
def document(p):
 raw,ref=read(p);return json.loads(raw),ref
def load(p,name):
 spec=importlib.util.spec_from_file_location(name,p);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module
def audit(event,args):
 if event=='import'and args[0].split('.')[0]in {'jax','jaxlib','numpy','torch','flax','optax','jaxtari','cupy','scipy','pynvml'}:raise ValueError('no numerical/GPU import')
 if event in ('ctypes.dlopen','ctypes.dlsym','os.system','os.exec','os.fork')or event.startswith('socket.'):raise ValueError('no native/network action')
 if event=='subprocess.Popen'and (not isinstance(args[1],(list,tuple))or args[1][0]!='git'):raise ValueError('only source Git')
 if event=='open'and isinstance(args[0],(str,bytes))and os.fsdecode(args[0]).startswith('/proc/'):raise ValueError('saved reader must not access live proc')
check(os.environ.get('SLURM_JOB_ID')=='6825983','exact current CPU allocation')
os.environ['GIT_OPTIONAL_LOCKS']='0';sys.addaudithook(audit)
freeze,FR=document(D/'combined_source_freeze.json')
check(FR['sha256']=='b59204c6ace7832a0d8bc88bf8c9e3d66ebfc7be39d397aeb806806bf8ab6462'and len(freeze['files'])==2,'held combined runtime and fixture freeze')
for rel,record in freeze['files'].items():
 raw,ref=read(W/rel);check(ref['sha256']==record['sha256']and len(raw)==record['size_bytes']and stat.S_IMODE((W/rel).stat().st_mode)==record['mode'],'exact held file '+rel)
runtime,RF=document(D/'runtime_source_freeze_2.json')
check(RF['sha256']=='c3d9473ea47b702fc72905188682a207fbf9fe341001d0bf1971ac9a6a637dc6','exact corrected runtime freeze')
for rel,sha in runtime['all_other_staged_runtime_sources_unchanged'].items():check(read(W/rel)[1]['sha256']==sha,'old protected runtime helper unchanged '+rel)
source=(W/PREFIX/'profile_target_tool_scope.py').read_text();tree=ast.parse(source)
environment={k:v for k,v in os.environ.items()if not k.startswith('GIT_')};environment['GIT_OPTIONAL_LOCKS']='0'
base_source=subprocess.check_output(['git','-C',str(W),'show','4bb3a61cb92fe48ab5ba68fb52f70b47736a1174:'+PREFIX+'profile_target_tool_scope.py'],env=environment).decode()
funcs=lambda s:{n.name:ast.get_source_segment(s,n)for n in ast.parse(s).body if isinstance(n,ast.FunctionDef)}
oldf,newf=funcs(base_source),funcs(source)
for name in ['require','same','source','adjacent','base_module','handle_state','create_scope','validate_parent_exit_record']:check(oldf[name]==newf[name],'old standalone function source unchanged '+name)
check(subprocess.check_output(['git','-C',str(W),'diff','--name-only','4bb3a61cb92fe48ab5ba68fb52f70b47736a1174','--','10_jax_exploration_platform'],env=environment).decode().splitlines()==[PREFIX+'profile_target_tool_scope.py'],'only the one tracked runtime file changes; old7/NUM/core unchanged')
fixture,FF=document(D/'actual_task_exit_fixture_handoff.json');check(FF['sha256']=='8d74d8048b5f7ed7b81a46b6c722e09c52a06cd167a619310fb748f73db4b203','actual fixture handoff retained')
proof,PR=document(D/'actual_task_exit_fixture_checks_2.json')
check(PR['sha256']=='4ba5d561a074f105f73f9a63871ae45cd04123bbf6c8b6ce598d4ff681c69ad8'and proof['passed']is True and proof['actual_job_id']=='6825983'and proof['actual_step_id']=='178'and proof['methods']==1 and proof['native_provider_role_claimed']is False,'actual distinct successful own CPU task-expiry evidence178')
for path,ref in fixture['actual_new_case_evidence'].items():check(read(path)[1]['sha256']==ref['sha256']and pathlib.Path(path).stat().st_size==ref['bytes'],'preserved actual CPU evidence exact '+path)
E=D/'actual_cpu_task_exit_evidence_2/fresh_complete_read'
control,_=document(E/'fixture_control.json');terminal,_=document(E/'tool_descendant_scope_terminal.json');begin,_=document(E/'tool_descendant_scope_begin.json');manifest,_=document(E/'tool_task_roster_observations.json');record,_=document(E/'tool_task_roster_observation_1.json')
actual_parent=control['actual_trigger'][0]['parent_pid'];actual_birth=control['actual_trigger'][0]['parent_birth'];actual_tid=control['actual_trigger'][0]['worker_tid']
check(control['native_provider_role_claimed']is False and len(control['actual_trigger'])==1 and control['actual_trigger'][0]['parent_handle_live']is True and control['actual_trigger'][0]['worker_join_acknowledged']is True,'real worker exit with same live own parent')
check(record['parent_pid']==actual_parent and record['parent_birth']==actual_birth and record['failed_task']==actual_tid and record['failure']['errno']==2 and record['fresh_complete_collection']['complete_attempt']==2,'actual precise enumerated-task ENOENT and complete retry')
check(terminal['terminal']['actual_echild']is terminal['terminal']['all_actual_tool_roots_waited']is terminal['terminal']['all_pinned_handles_exited']is True and terminal['read_errors']==terminal['interventions']==[],'actual closed CPU scope with exact ECHILD/no interventions')
wait,_=document(E/('tool_root_wait_'+str(actual_parent)+'.json'));check(wait['pid']==actual_parent and wait['birth']==actual_birth and type(wait['returncode'])is int and wait['returncode']==0,'actual exact original Popen normalwait0')
events=control['descriptor_events'];opens=[x for x in events if x['operation']=='pidfd_open'];closes=[x for x in events if x['operation']=='close']
check(len(opens)==len(closes)>0 and all(sum(y['descriptor']==x['descriptor']and y['opened_pid']==x['opened_pid']and y['open_generation']==x['open_generation']and y['identity']==x['identity']for y in closes)==1 for x in opens),'all actual stdlib pidfd generations have one same-identity close')
adapter=load(W/PREFIX/'profile_target_tool_scope.py','independent_saved_task_roster_adapter')
base=load(W/PREFIX/'direct_tool_descendants.py','independent_saved_task_roster_base')
observer=load(W/PREFIX/'nsight_process_identity.py','independent_saved_task_roster_observer')
parser=load(W/PREFIX/'linux_pidfd_process_control.py','independent_saved_task_roster_parser')
owner=begin['producer'];completed=control['actual_fixture_completed_at_seconds']
def validate(rec=record,ev=events,term=terminal,done=completed):return adapter.validate_task_roster_record(rec,term,begin,owner,base,observer,ev,parser,producer_terminal_time=done)
validate();check(True,'unchanged pure full task-record reader accepts actual saved CPU record and actual stdlib close/completion clocks')
negative=[]
def refusal(label,change):
 rec=copy.deepcopy(record);ev=copy.deepcopy(events);term=copy.deepcopy(terminal);done=[completed];change(rec,ev,term,done)
 try:adapter.validate_task_roster_record(rec,term,begin,owner,base,observer,ev,parser,producer_terminal_time=done[0])
 except (ValueError,KeyError,TypeError,IndexError):negative.append(label);check(True,'strict saved task record refuses '+label)
 else:raise ValueError('saved task mutation accepted '+label)
refusal('unknown parent',lambda r,e,t,c:r.update(parent_pid=actual_parent+1000000))
refusal('changed known parent birth',lambda r,e,t,c:r.update(parent_birth='1'))
refusal('wrong exact task-open phase',lambda r,e,t,c:r['failure'].update(operation='read_children'))
refusal('foreign errno',lambda r,e,t,c:r['failure'].update(errno=13))
refusal('noncanonical vanished TID',lambda r,e,t,c:r['failure'].update(filename='0'+str(actual_tid)))
refusal('failed task present again',lambda r,e,t,c:r['task_roster_after_failure'].append(str(actual_tid)))
refusal('discarded partial child union invented',lambda r,e,t,c:r['partial_child_pids'].append(999999))
refusal('partial sample credited',lambda r,e,t,c:r.update(child_list_sample_valid=True))
refusal('incomplete retry roster',lambda r,e,t,c:r['fresh_complete_collection'].update(task_roster_after=[]))
refusal('retry generation changed',lambda r,e,t,c:r['fresh_complete_collection'].update(parent_birth='1'))
refusal('late retry budget',lambda r,e,t,c:r['fresh_complete_collection'].update(bounded_retry_elapsed_seconds=.6))
refusal('boolean retry attempts',lambda r,e,t,c:r.update(maximum_attempts=True))
refusal('closed parent at retry',lambda r,e,t,c:r['fresh_complete_collection']['retained_live_handle'].update(handle_exited=True))
refusal('raw fdinfo hash changed',lambda r,e,t,c:r['retained_live_handle'].update(fdinfo_sha256='0'*64))
def wrong_generation(r,e,t,c):
 for x in e:
  if x['operation']=='close'and x['opened_pid']==actual_parent:x['open_generation']+=1
refusal('wrong actual close generation',wrong_generation)
def early_close(r,e,t,c):
 for x in e:
  if x['operation']=='close'and x['opened_pid']==actual_parent:x['observed_at_seconds']=r['retained_live_handle']['observed_at_seconds']-.1
refusal('same handle closes before sample',early_close)
refusal('record finishes after kernel closure',lambda r,e,t,c:r['fresh_complete_collection'].update(identities_finished_at_seconds=t['terminal']['observed_at_seconds']+1))
refusal('actual completed clock before close',lambda r,e,t,c:c.__setitem__(0,r['retained_live_handle']['observed_at_seconds']))
# Execute exact new terminal/manifest predicates from the held source, without faking a native role or directory.
saved_fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='validate_saved_scope')
requires=[n for n in ast.walk(saved_fn)if isinstance(n,ast.Call)and isinstance(n.func,ast.Name)and n.func.id=='require']
def predicate(message):
 return next(n.args[0]for n in requires if len(n.args)>1 and isinstance(n.args[1],ast.Constant)and n.args[1].value==message)
manifest_condition=compile(ast.Expression(predicate('every immutable terminal task-roster event must match its ordered manifest reference')),'held-manifest-predicate','eval')
event_condition=compile(ast.Expression(predicate('the terminal task-roster event changed its parent, task, validity or chronology')),'held-event-predicate','eval')
task_events=[x for x in terminal['events']if x.get('kind')=='known_parent_task_roster_reconciled']
scope={'task_events':task_events,'task_records':manifest['records']}
check(eval(manifest_condition,{},scope),'exact held terminal-event/manifest predicate accepts actual references')
for label,refs in [('omitted event record',[]),('duplicated event record',manifest['records']*2)]:
 check(not eval(manifest_condition,{},dict(scope,task_records=refs)),'exact held manifest predicate rejects '+label)
event=task_events[0];ctx=dict(event=event,record=record,terminal=terminal,same=adapter.same,math=math)
check(eval(event_condition,{},ctx),'exact held terminal event PID/birth/task/finish predicate accepts actual CPU event')
for label,change in [('PID',lambda x:x.update(pid=actual_parent+1)),('birth',lambda x:x.update(birth='1')),('task',lambda x:x.update(task=actual_tid+1)),('valid sample',lambda x:x.update(discarded_child_list_valid=True)),('before final identity completion',lambda x:x.update(at_seconds=record['fresh_complete_collection']['identities_finished_at_seconds']-.1)),('after closure',lambda x:x.update(at_seconds=terminal['terminal']['observed_at_seconds']+1))]:
 altered=copy.deepcopy(event);change(altered);check(not eval(event_condition,{},dict(ctx,event=altered)),'exact held terminal event predicate rejects '+label)
check(len(manifest['records'])==1 and manifest['records'][0]['sha256']==hashlib.sha256((E/'tool_task_roster_observation_1.json').read_bytes()).hexdigest()and pathlib.Path(manifest['records'][0]['path']).name=='tool_task_roster_observation_1.json','actual immutable temporary receipt matches preserved exact basename and SHA without restoring or rewriting its path')
check(manifest['adapter_sha256']==freeze['files'][PREFIX+'profile_target_tool_scope.py']['sha256']and manifest['failure_count']==0 and manifest['discarded_child_list_samples_accepted_as_valid']is False,'actual task manifest source and outcome')
pins,PC=document(OUTDIR/'independent_corrected_pin_timing_controls.json');check(pins['old174_gap_now_rejected']is True and len(pins['cases'])==4,'prior four affected extracted-method pin timing controls reused')
for rel,r in freeze['files'].items():check(read(W/rel)[1]['sha256']==r['sha256'],'final held source unchanged after saved review '+rel)
check(not any(n.split('.')[0]in {'jax','jaxlib','numpy','torch','flax','optax','jaxtari'}for n in sys.modules),'no numerical modules imported')
result=dict(schema_version=1,kind='independent_profile_task_roster_source_and_saved_CPU_evidence_review',approved=True,passed=True,independent_of_author=True,source_freeze=FR,runtime_freeze=RF,actual_fixture_handoff=FF,actual_fixture_proof=PR,corrected_pin_timing_control=PC,actual_job_id=os.environ['SLURM_JOB_ID'],actual_step_id=os.environ['SLURM_STEP_ID'],reviewer_pid=os.getpid(),reviewer_uid=os.getuid(),reviewed_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),seconds=time.monotonic()-START,check_count=len(CHECKS),checks=CHECKS,files=FILES,saved_record_refusals=negative,actual_lifecycle_reexecuted_by_reviewer=False,native_provider_role_qualification_claimed=False,full_saved_native_role_wrapper_executed=False,scope='Frozen source, actual178 own-CPU/std-pidfd task event, pure task-record reader and exact extracted manifest/event predicates only. Existing whole native-role/source/tool readers remain required on actual target qualification. No historical paths, times or bytes changed.',actual_GPU_native_process_tool_or_remote_operations=False,actual_target_qualification=False,scientific_credit=0,case_credit=0,idea_credit=0,profile_credit=0,worker_or_profile_admission=False,unresolved_anomalies=[])
out=D/'independent_corrected_source_and_saved_evidence_review.json';data=(json.dumps(result,indent=2,sort_keys=True)+'\n').encode()
with out.open('xb')as f:f.write(data);f.flush();os.fsync(f.fileno())
fd=os.open(D,os.O_RDONLY|os.O_DIRECTORY);os.fsync(fd);os.close(fd)
print(json.dumps({'path':str(out),'sha256':hashlib.sha256(data).hexdigest(),'step':result['actual_step_id'],'checks':len(CHECKS),'seconds':result['seconds']}))

