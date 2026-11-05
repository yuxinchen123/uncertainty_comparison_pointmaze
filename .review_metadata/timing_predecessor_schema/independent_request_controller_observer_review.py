import ast,copy,hashlib,importlib.util,io,json,os,sys,time,unittest
from pathlib import Path
W=Path('/p/rlprojects/RND_worktrees/2026-11-05-07-20_h100-timing-predecessor_actual-confirmation-schema_two-copy-request_read-only-exits_gpu-actions-0');D=W/'.review_metadata/timing_predecessor_schema';S=W/'10_jax_exploration_platform/scripts/atari_pixel_campaign'
O=Path('/p/rlprojects/RND_worktrees/2026-11-05-07-23_prior-h100-exit-observation_controller-pid-386202_birth-84810007_numerical-pid-386669_birth-84815419_read-only-true_gpu-operations-0')
R=Path('/p/rlprojects/RND_worktrees/merge-atari-cleanrl/10_jax_exploration_platform/benchmark_runs/2026-10-31-00-11_jaxatari_montezuma_pixels_cleanrl_ppo_rnd_environments-per-copy-128_rollout-steps-128_ideas-30-plus-20_steps-per-copy-2000000000_seed-start-1')
assert os.environ['SLURM_JOB_ID']=='6821437';assert (R/'slurm/submitted_jobids_uva.txt').read_text().splitlines().count('6821437')==1
started=time.monotonic();checks=[]
def check(name,value):
 assert value,name
 checks.append(name)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def doc(p):return json.loads(Path(p).read_bytes())
freeze=D/'request_and_source_freeze.json';check('exact held combined request/source freeze',sha(freeze)=='ad9485a028e4dfc06acd8931d9083d93c8224fec695b0c0e357fe2292328338c');f=doc(freeze)
obs_freeze=O/'source_and_test_freeze.json';check('exact held observer source/test freeze',sha(obs_freeze)=='c2bdfc35c8f48512870b4fb12a74e402aaeadb029b474fbb5a66c6e3676a2019');of=doc(obs_freeze)
def verify():
 for path,expected in f['files'].items():check('held source/request '+path,sha(path)==expected)
 for name,row in of['files'].items():
  check('held observer '+name,sha(O/name)==row['sha256']);check('held observer mode '+name,oct((O/name).stat().st_mode&0o777)==row['mode'])
verify();proposal=D/'proposal';request=doc(proposal/'preparation_request.json');derivation=doc(proposal/'derivation.json');base=doc(derivation['base_request']['path']);expected=copy.deepcopy(base)
expected.pop('wrapper_source_commit',None)
expected.update(unit_id=request['unit_id'],packet=str(R/'remote_setup'/request['unit_id']),data_unit=str(R/'data'/request['unit_id']),serial_order=2,copy_seeds=[1,2],serial_predecessor_unit=str(R/'data'/Path(derivation['base_request']['path']).parent.name))
for key in ('configuration','reference_configuration'):expected[key]['learner']['n_copies']=2
expected['common_correctness_inputs']={key:base['accepted_correctness']['2'][key] for key in ('result','review')}
check('only exact intended request differences',request==expected)
check('actual new packet/unit still absent',not Path(request['packet']).exists() and not Path(request['data_unit']).exists())
check('actual original numerical source retained',request['numerical_source_commit']=='d28e6735e79788f618acb0ebfde8609db1f776af')
check('actual original selected helper source retained',request['helper_source_commit']=='69f1b5075530418adbe02c1e433c5647848a61ba' and request['expected_staged_helper_count']==29)
check('actual original full training configuration retained',request['configuration']['learner']['n_envs']==128 and request['configuration']['learner']['num_steps']==128 and request['configuration']['learner']['num_minibatches']==4 and request['configuration']['learner']['update_epochs']==4 and request['configuration']['learner']['observation_warmup_iterations']==50 and request['configuration']['learner']['total_timesteps']==2000000000)
for reference in request['common_correctness_inputs'].values():check('actual accepted copy-two correctness immutable ref',sha(reference['path'])==reference['sha256'])
review=doc(request['common_correctness_inputs']['review']['path']);check('genuine accepted copy-two correctness review',review['approved'] is True and review['passed'] is True and review['copies']==2 and review['copy_seeds']==[1,2] and review['result']==request['common_correctness_inputs']['result'])
h1=doc(Path(request['serial_predecessor_unit'])/'measurement_plan.json')
for label,entries in f['source_roster'].items():
 for relative,row in entries.items():
  if label=='protected_numerical_13':
   for reference in row.values():check('protected numerical actual hash '+relative,sha(reference['path'])==reference['sha256'])
  else:check('protected selected helper actual hash '+relative,sha(row['path'])==row['sha256']==h1['helpers']['files'][relative])
check('all thirteen numerical and twenty-nine helper entries present',len(f['source_roster']['protected_numerical_13'])==13 and len(f['source_roster']['selected_helper_29'])==29)
available=doc(proposal/'predecessor_available_records.json')
check('genuine accepted one-copy timing review',available['timing_review']['sha256']=='901a5805d7f768cd3b3159fe7e478fffdbee36280c2144f195ada1b3471bdf22')
for key,value in available.items():
 if type(value) is dict and set(value)=={'path','sha256'}:
  if key=='result':check('large result bound to existing genuine review without a second read',value==doc(available['timing_review']['path'])['result'])
  else:check('actual existing predecessor ref '+key,sha(value['path'])==value['sha256'])
procedure=doc(proposal/'planned_root_metadata_procedure.json')
check('no actual phase or future source/PID claimed',procedure['actual_phase_invocations']==0 and procedure['actual_root_packet_or_unit_created'] is False and procedure['entire_controller_source_unchanged_claim'] is False)
check('fresh root-only observer must run before immutable serial binding',procedure['fresh_exit_observer_root_invocation_not_performed'] is True and procedure['fresh_exit_observer_must_be_independently_approved_and_published'] is True and procedure['fresh_exit_observer_source']=={'path':str(O/'observe_exact_prior_h100_exits.py'),'sha256':'87bdf73927b03341ebe6b5b3d3600e8b77e62820b7f8c07f78c9ff5936ba9540'})
check('claim hash derives only from real preparation',procedure['claim_probe']['complete_command_not_formed_before_actual_prepare'] is True)
check('root/controller never own standing GPU descriptor',procedure['controller_owns_gpu_standing_descriptor'] is False and procedure['producer_owns_gpu_standing_descriptor'] is True)
for phase,command in procedure['phase_commands'].items():check('bounded exact current owned metadata phase '+phase,'--jobid=6821437' in command and '--mem=512M' in command and '--time=00:05:00' in command)
sys.path[:0]=[str(S),str(S/'tests'),str(O)]
import test_timing_predecessor_actual_schema as timing_tests
import test_observe_exact_prior_h100_exits as observer_tests
class AdditionalControls(unittest.TestCase):
 def test_no_global_size_limit_change_and_exact_changed_function(self):
  old=ast.parse((D/'controller_published_d225.py').read_bytes());new=ast.parse((S/'direct_common_timing_controller.py').read_bytes());functions=lambda tree:{n.name:ast.dump(n,include_attributes=False) for n in tree.body if isinstance(n,ast.FunctionDef)}
  a,b=functions(old),functions(new);self.assertEqual([name for name in a if a[name]!=b[name]],['check_serial_predecessor_inputs']);self.assertEqual(timing_tests.control.ordinary_bytes.__defaults__,(16*1024*1024,))
 def test_observer_only_absent_rows_satisfy_old_controller_schema(self):
  self.assertEqual(observer_tests.observer.IDENTITIES,{'controller':(386202,'84810007'),'numerical_child':(386669,'84815419')})
  text=(O/'observe_exact_prior_h100_exits.py').read_text();self.assertNotIn('os.kill(',text);self.assertNotIn('subprocess',text);self.assertNotIn('flock(',text)
suite=unittest.TestSuite()
for cls in (timing_tests.ActualTimingPredecessorSchemaTests,observer_tests.PriorExitObserverTests,AdditionalControls):suite.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(cls))
out=io.StringIO();result=unittest.TextTestRunner(stream=out,verbosity=2).run(suite);verify()
record={'schema_version':1,'kind':'independent_two_copy_timing_predecessor_and_exit_observer_source_review','approved':result.wasSuccessful(),'passed':result.wasSuccessful(),'independent_of_authors':True,'reviewer_identity':'/root/review_bound_h100_timing_launch','source_and_request_freeze':{'path':str(freeze),'sha256':sha(freeze)},'observer_source_and_test_freeze':{'path':str(obs_freeze),'sha256':sha(obs_freeze)},'check_count':len(checks),'checks':checks,'test_methods':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),'output':out.getvalue(),'seconds':time.monotonic()-started,'job_id':os.environ['SLURM_JOB_ID'],'step_id':os.environ['SLURM_STEP_ID'],'uid':os.getuid(),'pid':os.getpid(),'cgroup':Path('/proc/self/cgroup').read_text(),'source_scope':'Only check_serial_predecessor_inputs changes. Serial-two reads its exact accepted result at a 64MiB bound; global16MiB helpers, serial-one correctness branch, all other controller functions and scientific helpers/numerical sources remain unchanged.','approval_scope':'Publish exact corrected source, observer and private request/procedure. Root must collect fresh actual observer output, prepare/verify/probe/bind once, and obtain new actual bound and transport reviews before any remote numerical launch.','actual_remote_target_or_GPU_actions':0,'actual_production_phase_invocations':0,'historical_raw_collector_replayed':False,'final_training_or_packing_credit':0,'observer_test_scope':'Synthetic private proc rows plus actual own-CPU-process namespace check only; no recorded target PID queried.'}
p=D/'independent_request_controller_observer_review.json'
with p.open('x') as f:json.dump(record,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
print(out.getvalue());print(json.dumps({'approved':record['approved'],'review':str(p),'sha256':sha(p),'checks':len(checks),'methods':result.testsRun,'seconds':record['seconds'],'job_id':record['job_id'],'step_id':record['step_id']}))
raise SystemExit(0 if result.wasSuccessful() else 1)
