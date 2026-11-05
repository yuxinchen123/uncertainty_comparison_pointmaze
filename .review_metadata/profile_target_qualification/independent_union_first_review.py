"""Independent whole-union metadata review; never prepare or run a GPU target."""
import ast,contextlib,copy,ctypes,datetime,hashlib,importlib.util,io,json,os,re,stat,subprocess,sys,tempfile,time,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
W=Path('/p/rlprojects/RND_worktrees/2026-11-05-09-42_profile-target-qualification_original-tests-48_environments-128_rollout-steps-128_warmup-rollouts-50_reference-updates-1_default-admission-closed');D=W/'.review_metadata/profile_target_qualification';S=W/'10_jax_exploration_platform/scripts/atari_pixel_campaign';start=time.monotonic();checks=[]
def need(v,s):
 if not v:raise AssertionError(s)
 checks.append(s)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ref(p):return {'path':str(p),'sha256':sha(p)}
fpath=D/'union_source_freeze_1.json';need(sha(fpath)=='82e42ff2954295687df24b68f9821ad6196980cedd1dd2f153dc71c4cd16df0b','whole16file held source freeze');f=json.loads(fpath.read_text());need(os.getresuid()==(2618919,)*3 and os.environ['SLURM_JOB_ID']=='6825983','new owned CPU only');need('/job_6825983/step_'+os.environ['SLURM_STEP_ID']+'/' in Path('/proc/self/cgroup').read_text(),'exact numeric owned CPU step')
for n,row in f['runtime_and_test_files'].items():need(sha(W/n)==row['sha256'] and (W/n).stat().st_size==row['size_bytes'] and oct(stat.S_IMODE((W/n).stat().st_mode))==row['mode'],'exact full union file '+n)
for n,h in f['protected_existing_files'].items():need(sha(W/n)==h,'actual protected base source '+n)
raw=subprocess.run(['git','-C',str(W),'cat-file','--batch'],input=''.join(f['base_commit']+':'+n+'\n' for n in f['protected_existing_files']).encode(),stdout=subprocess.PIPE,stderr=subprocess.PIPE,check=True,timeout=30,env={**os.environ,'GIT_OPTIONAL_LOCKS':'0'});need(not raw.stderr,'exact local protected Git reads');stream=io.BytesIO(raw.stdout)
for n,h in f['protected_existing_files'].items():
 header=stream.readline().decode().split();need(len(header)==3 and header[1]=='blob','ordinary protected Git blob');need(hashlib.sha256(stream.read(int(header[2]))).hexdigest()==h and stream.read(1)==b'\n','actual published protected Git bytes '+n)
need(stream.read()==b'','complete protected Git output');ptree=ast.parse((S/'profile_selected.py').read_text());core=next(n for n in ptree.body if isinstance(n,ast.FunctionDef) and n.name=='_capture_core');need(hashlib.sha256(ast.dump(core,include_attributes=False).encode()).hexdigest()==f['capture_core_ast_sha256'],'capture core AST exact')
proposal=json.loads((D/'target_resource_proposal/resource_proposal.json').read_text());need(proposal['operational'] is False and proposal['source_commit'] is None and proposal['job_id'] is None and proposal['controller_pid'] is None and proposal['numerical_pid'] is None and proposal['actual_unit_prepared'] is False,'resource proposal retains all unknown actual bindings');need(proposal['qualification_budget']=={'schema_version':1,'work_seconds':10200,'cleanup_seconds':600,'total_seconds':10800},'resource proposal qualification-only3h budget');need(all(row['gpus']==1 and row['cpus']==8 and row['wall_time_hours']==3 for row in proposal['resources']),'resource proposal one GPU/eight CPUs/three hours per exact target')
# Load only ordinary source-test modules before the strict parent audit; stdlib ctypes self bootstrap is explicit.
loader=unittest.TestLoader();suite=unittest.TestSuite();suite.addTests(loader.discover(str(S/'tests'),pattern='test_profile_target_*.py'));suite.addTests(loader.discover(str(S/'tests'),pattern='test_profile_qualification_backend.py'))
def audit(event,args):
 if event in {'ctypes.dlopen','ctypes.dlsym','socket.connect','socket.bind','os.system'}:raise RuntimeError('native/network operation forbidden after own stdlib bootstrap')
 if event=='import' and args[0].split('.')[0] in {'jax','jaxlib','numpy','torch','flax','optax','jaxtari','exploration_platform'}:raise RuntimeError('numerical import forbidden')
 if event=='open' and isinstance(args[0],(str,bytes)) and os.fsdecode(args[0]).startswith(('/dev/nvidia','/dev/dri','/proc/driver/nvidia')):raise RuntimeError('GPU device metadata forbidden')
 if event=='subprocess.Popen':
  argv=args[1];allowed=(list(argv[:4])==['/usr/bin/python3','-I','-S','-B'] and len(argv)==6 and Path(argv[4]).name=='test_profile_target_real_metadata.py' and argv[5]=='-v' and args[3]=={'PATH':'/usr/bin:/bin','PROFILE_COLD_METADATA_FIXTURE':'1'})
  local_git=(list(argv[:4])==['git','-C',str(W),'show'] and len(argv)==5 and argv[4].startswith(f['base_commit']+':') and argv[4].split(':',1)[1] in set(f['protected_existing_files'])|{'10_jax_exploration_platform/scripts/atari_pixel_campaign/profile_selected.py'})
  own_cpu=list(argv)==['/usr/bin/python3','-I','-S','-B','-c','import time; time.sleep(.25)']
  if not (allowed or local_git or own_cpu):raise RuntimeError('only exact local sourceGit or own CPU metadata children are allowed')
os.environ['GIT_OPTIONAL_LOCKS']='0'
# Reuse the49 completed methods; run only the two guard-refused metadata methods.
def flat(suite):
 for item in suite:
  if isinstance(item,unittest.TestSuite):yield from flat(item)
  else:yield item
suite=unittest.TestSuite(item for item in flat(suite) if item._testMethodName in {'test_existing_component_helpers_and_numerical_capture_are_unchanged','test_actual_own_cpu_child_is_recorded_and_waited_without_gpu_work'})
sys.addaudithook(audit);log=io.StringIO();result=unittest.TextTestRunner(stream=log,verbosity=2).run(suite);(D/'independent_union_two_guard_refused_cases.raw').write_text(log.getvalue());(D/'independent_union_two_guard_refused_cases_result.json').write_text(json.dumps({'passed':result.wasSuccessful(),'tests':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),'job_id':os.environ['SLURM_JOB_ID'],'step_id':os.environ['SLURM_STEP_ID']},indent=2)+'\n');need(result.wasSuccessful() and result.testsRun==2,'two previously guard-refused metadata methods pass; prior49 completed cases reused')
# Enter the actual H producer/waiter/capture call sites with explicit metadata premises only.
spec=importlib.util.spec_from_file_location('independent_union_profile',S/'profile_selected.py');profile=importlib.util.module_from_spec(spec);spec.loader.exec_module(profile)
spec=importlib.util.spec_from_file_location('independent_union_process',S/'profile_target_process.py');process=importlib.util.module_from_spec(spec);spec.loader.exec_module(process)
findings=[];errors=[]
with tempfile.TemporaryDirectory() as temporary:
 unit=Path(temporary);(unit/'code').mkdir();plan={'kind':process.QUALIFICATION,'python':'/fixture/bin/python','diagnostic_time_budget':dict(process.QUALIFICATION_BUDGET),'hardware_class':'h100'};(unit/'measurement_plan.json').write_text(json.dumps(plan))
 commands={'producer':process.producer_command(unit,plan),'waiter':['/fixture/bin/python','waiter'],'capture':['/fixture/bin/python','capture']}
 metadata=SimpleNamespace(stable=lambda x:x,publish=lambda *args:None,receipt=lambda path:{'path':str(path),'sha256':'a'*64})
 facade=SimpleNamespace(generic=lambda p:metadata)
 def context(directory,plan,observer):return {'pid':os.getpid(),'uids':[2618919]*4,'command':commands[current_role[0]]}
 def unused_provider(plan):raise AssertionError('provider must never start in an omitted-deadline fixture')
 current_role=['producer']
 with patch.object(profile,'target_process_reader',return_value=process),patch.object(profile,'direct_process_reader',return_value=facade),patch.object(profile,'verify_prepared',return_value=None),patch.object(process,'policy',return_value=None),patch.object(process,'context',side_effect=context),patch.object(process,'role_commands',return_value=(commands['waiter'],commands['capture'])),patch.object(process,'load_provider',side_effect=unused_provider):
  for role,call in [('producer',lambda:profile.run_direct_target_prepared(unit,plan,{})),('waiter',lambda:profile.run_direct_owned_waiter(unit)),('capture',lambda:profile.capture_direct_owned_child(unit))]:
   current_role[0]=role
   try:call()
   except ValueError as error:
    if 'declared diagnostic deadline' in str(error):errors.append({'role':role,'actual_error':str(error)})
 need(len(errors)==3,'actual three H entrypoints reach real Runtime omitted-deadline refusal before native startup')
 findings.append({'name':'all_H_native_bootstrap_calls_omit_required_deadline','reproduced_roles':errors,'required':'Create/read the same purpose-bound deadline before bootstrap in each new H role and pass deadline= explicitly. Retain one producer origin and exact child role source/owner gates.'})
 # Actual unchanged legacy policy is incompatible with the new qualification plan.
 legacy=profile.configured_native_reader(selected=True);legacy_plan={'kind':process.QUALIFICATION,'hardware_class':'jaguar03_rtx_a4500','gpu_uuid':None,'maximum_active_seconds':10200,'diagnostic_time_budget':dict(process.QUALIFICATION_BUDGET)}
 legacy_error=None
 try:legacy.policy(legacy_plan)
 except ValueError as error:legacy_error=str(error)
 need(legacy_error is not None,'actual old configured-native policy refuses new qualification kind/budget')
 aa=next(n for n in ptree.body if isinstance(n,ast.FunctionDef) and n.name=='run_configured_native_prepared');need(any(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and isinstance(n.func.value,ast.Name) and n.func.value.id=='metadata' and n.func.attr=='validate_saved' for n in ast.walk(aa)),'new AA producer still calls unchanged raw observer whole saved reader')
 old_reader=ast.parse((S/'nsight_process_identity.py').read_text());validator=next(n for n in old_reader.body if isinstance(n,ast.FunctionDef) and n.name=='validate_saved');need("'configured_native_process_control' in plan" in ast.unparse(validator) and 'reader.validate_saved' in ast.unparse(validator),'unchanged observer saved reader invokes legacy native hook for that declaration')
 findings.append({'name':'AA_final_saved_reader_enters_rejecting_legacy_native_policy','actual_old_policy_error':legacy_error,'required':'Retain all unchanged raw saved ancestry/source/release/wait/result checks in an explicit new-target-only pure reader, then validate the new target proof; keep originalAA7 byte exact, no changed-kind projection or namespace mutation.'})

 # The old direct backend is not the distinct new qualification protocol, and its declaration conflicts with the new backend.
 spec=importlib.util.spec_from_file_location('independent_union_backend',S/'profile_qualification_backend.py');backend=importlib.util.module_from_spec(spec);spec.loader.exec_module(backend)
 old_backend=profile.measurement.producer_support if False else None
 spec=importlib.util.spec_from_file_location('independent_union_old_backend',S/'direct_backend_image.py');old_backend=importlib.util.module_from_spec(spec);spec.loader.exec_module(old_backend)
 need(old_backend.profile_plan({'kind':process.QUALIFICATION,'mode':'profile','direct_profile_protocol':'direct-h100-reviewed-common-profile-version-1','direct_binding':{}}) is False,'actual old backend rejects the distinct qualification kind')
 spec=importlib.util.spec_from_file_location('independent_union_backend_fixture',S/'tests/test_profile_qualification_backend.py');bt=importlib.util.module_from_spec(spec);spec.loader.exec_module(bt);case=bt.ProfileQualificationBackendTests('test_full_round_trip_on_aa_preserves_zero_credit_and_both_phase_scope');case.setUp()
 try:
  conflict=copy.deepcopy(case.plan);conflict['direct_backend_image']={};refusal=None
  try:backend.unit_directory(conflict)
  except ValueError as error:refusal=str(error)
  need(refusal is not None,'new backend rejects the legacy direct-backend declaration inserted by current configure_binding')
  findings.append({'name':'new_H_qualification_still_enters_old_direct_backend_hooks','actual_old_profile_plan_accepts_new_kind':False,'new_backend_legacy_flag_refusal':refusal,'required':'Dispatch the new qualifier during binding preparation, capture begin/completion and complete_capture/final validation to its separate backend records. Preserve old ordinary direct schema2 hooks and every _capture_core call unchanged; do not leave the conflicting old direct_backend_image declaration on a new qualifier.'})
  spec=importlib.util.spec_from_file_location('independent_union_qualifier',S/'profile_target_qualification.py');qual=importlib.util.module_from_spec(spec);spec.loader.exec_module(qual)
  source_tree=ast.parse((S/'configured_measurement.py').read_text());run=next(n for n in source_tree.body if isinstance(n,ast.FunctionDef) and n.name=='run_qualification');returned=next(n.value for n in ast.walk(run) if isinstance(n,ast.Return) and isinstance(n.value,ast.Dict) and any(isinstance(k,ast.Constant) and k.value=='executed_work' for k in n.value.keys));counter_node=next(v for k,v in zip(returned.keys,returned.values) if isinstance(k,ast.Constant) and k.value=='executed_work');counters=eval(compile(ast.Expression(body=counter_node),'actual_unchanged_reference_counter_dictionary','eval'),{'__builtins__':{}})
  now=time.time();captured={'compilation_warmup':[{'updates':1},{'updates':1}],'normalization_warmup_iterations':50,'profile_start_iteration':2,'profile_end_iteration':3,'compiled_signatures_before':[1,1],'compiled_signatures_after':[1,1],'execution_identity':{'hostname':'jaguar03','pid':os.getpid(),'process_start_time':case.process['process_start_time'],'started_at_seconds':now-4,'finished_at_seconds':now-3}};original={'reference_iteration':1,'reference_optimizer_step':16,'executed_work':counters};work=qual.controlled_work(case.plan,captured,original,now-2,now-1);counter_error=None
  try:backend.completed_work(case.plan,work)
  except ValueError as error:counter_error=str(error)
  need(counter_error is not None,'actual unchanged full_reference counter schema refuses current controlled_work forwarding')
  findings.append({'name':'actual_original_work_counter_schema_not_projected_to_backend_contract','actual_unchanged_gate_counter_keys':list(counters),'actual_backend_refusal':counter_error,'required':'Project exactly the four actual full_reference_* controlled values into the four backend original_reference keys, retain the complete original raw executed_work separately with unknown fixture totals unchanged. Never change old gate results or invent zeros.'})
 finally:case.doCleanups()

for n,row in f['runtime_and_test_files'].items():need(sha(W/n)==row['sha256'],'all16 union bytes held after independent checks '+n)
for n,h in f['protected_existing_files'].items():need(sha(W/n)==h,'all22 protected bytes held after independent checks '+n)
record={'schema_version':1,'kind':'independent_full_profile_target_qualification_union_first_source_review','passed':False,'approved':False,'source_freeze':ref(fpath),'independent_union_methods_passed_combining_prior49_and_current2':51,'methods_passed_current_step':2,'prior49_completed_cases_record':ref(D/'independent_union_first_unittest_result_attempt2.json'),'findings':findings,'all22_protected_and13NUM_capture_core_unchanged':True,'actual_hardware_qualification_or_admission_granted':False,'actual_GPU_tool_NUM_native_or_remote_operations':False,'test_premises':'Real cold copied source/accepted component metadata plus explicit metadata-only H context/source/owner premises. Actual unmodified Runtime deadline contract and legacy policy were exercised; no native provider initialized.','source_resource_proposal':ref(D/'target_resource_proposal/resource_proposal.json'),'checks':checks,'check_count':len(checks),'job_id':os.environ['SLURM_JOB_ID'],'step_id':os.environ['SLURM_STEP_ID'],'seconds':time.monotonic()-start,'at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
for name,data in [('independent_union_first_unittest.raw',log.getvalue().encode()),('independent_union_first_review.json',(json.dumps(record,indent=2,sort_keys=True)+'\n').encode())]:
 with (D/name).open('xb') as h:h.write(data);h.flush();os.fsync(h.fileno())
print(json.dumps({'passed':False,'review_sha256':sha(D/'independent_union_first_review.json'),'tests':result.testsRun,'findings':[x['name'] for x in findings],'seconds':record['seconds'],'step_id':record['step_id']}))
