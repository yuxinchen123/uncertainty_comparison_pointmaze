import builtins,ctypes,hashlib,json,os,pathlib,sys,tempfile,threading,time,types,unittest
from unittest.mock import patch
W=pathlib.Path('/p/rlprojects/RND_worktrees/2026-11-01-10-44_selected-and-profile-configured-isolation_single-a4500_gpu-count-1_processor-count-8_host-memory-gib-64_unresolved-device-identity')
S=W/'10_jax_exploration_platform/scripts/atari_pixel_campaign'
freeze=W/'.review_metadata/configured_selected_profile_producer_source_freeze.json'
frozen=json.loads(freeze.read_text()); files={**frozen['own_files'],'uva_slurm_configured_isolation.py':frozen['borrowed_dependency']['sha256']}
before={n:hashlib.sha256((S/n).read_bytes()).hexdigest() for n in files}; assert before==files
assert os.environ.get('SLURM_JOB_ID')=='6781231'
blocked=[]; original=builtins.__import__
def imports(name,*args,**kwargs):
 if name.split('.')[0] in {'jax','jaxlib','numpy','torch','jaxtari','exploration_platform','pynvml','cupy','numba','scipy','matplotlib'}:
  blocked.append(name);raise AssertionError('numerical import forbidden')
 return original(name,*args,**kwargs)
builtins.__import__=imports
def audit(event,args):
 if event in {'subprocess.Popen','socket.connect','ctypes.dlopen'}:
  blocked.append(event);raise AssertionError('live action forbidden')
 if event=='open' and isinstance(args[0],str) and args[0].startswith('/dev/'):
  blocked.append(event);raise AssertionError('device open forbidden')
sys.addaudithook(audit);sys.path[:0]=[str(S),str(S/'tests')]
import configured_measurement as support
import test_configured_measurement as tests
findings={}
# These deliberately demonstrate reader weaknesses, not valid physical/source qualification.
class Reproductions(unittest.TestCase):
 def test_saved_qualification_accepts_invalid_semantics(self):
  identity=support.core(); root=S.parents[2]
  import diagnose_gpu_execution as diagnostic
  defaults=diagnostic.default_configuration(root)
  configuration={k:defaults[k] for k in ('learner','environment')}
  configuration['execution']={'compilation_mode':'whole_iteration','iterations_per_call':1,'report_readout':'none','report_every_iterations':10}
  with tempfile.TemporaryDirectory() as tmp:
   d=pathlib.Path(tmp)
   plan={'source_root':str(root),'source_commit':'d28e6735e79788f618acb0ebfde8609db1f776af','mode':'correctness','configuration':configuration,'reference_configuration':configuration,
     'python':sys.executable,'hardware_class':'jaguar03_rtx_a4500','runtime_environment':{},'configured_isolation':{'profile':{}},'target_qualification':support.QUALIFICATION,'helpers':{},'asset_record':{}}
   reports=[{'nodeid':'synthetic-one','phase':'setup','outcome':'passed','duration_seconds':0.}]
   report={'command':[sys.executable,'-m','pytest','-q',str(root/'10_jax_exploration_platform/tests/atari_pixels'),str(root/'10_jax_exploration_platform/tests/test_atari_pixel_environment.py')],'returncode':0,'reports':reports}
   binding={'gpu_uuid':'GPU-ffffffff-ffff-ffff-ffff-ffffffffffff','process':{'pid':4321}}
   docs={'measurement_plan.json':plan,'configured_isolation_metadata.json':{'inputs':{'job_id':'123'}},'resolved_device_binding.json':binding,
     'original_tests.json':report,'original_renderer.json':{'checks':{'cache':'failed'}}}
   for n,v in docs.items():(d/n).write_text(json.dumps(v))
   for n in ('original_tests.stdout','original_tests.stderr'):(d/n).write_text('')
   value={'kind':'uva_configured_isolation_full_original_qualification','completed':True,'qualification_protocol':support.QUALIFICATION,
    'source_commit':plan['source_commit'],'hardware_class':plan['hardware_class'],'runtime_environment':{},'configured_isolation':{'profile':{}},'asset_record':{},'all_original_gates_passed':True,
    'resolved_device_binding':binding,'helpers':{},'reference_iteration':1,'reference_optimizer_step':16,'frozen_target_unchanged':True,'policy_parameters_changed':True,
    'executed_work':{'full_reference_warmup_transitions':819200,'full_reference_training_transitions':16384,'full_reference_updates':1,'full_reference_optimizer_minibatch_updates':16},
    'qualification_memory':{'assigned_gpu_process_observation':tests.memory_window()},'original_test_reports':reports,
    'gate_artifacts':{n:identity.receipt(d/n) for n in ('original_tests.json','original_tests.stdout','original_tests.stderr','original_renderer.json')},
    'dtypes':{'parameters':['float16']},'finite_state_leaves':0,'renderer':{'checks':{'cache':'failed'}}}
   result=d/'target_qualification_result.json';result.write_text(json.dumps(value));receipt=identity.receipt(result)
   review=d/'review.json';review.write_text(json.dumps({'schema_version':1,'kind':'independent_full_original_target_qualification_review','result':receipt,'approved':True,'admission_granted':False,'scientific_credit':0,'job_id':'123','actual_terminal_state':'COMPLETED','actual_exit_code':'0:0'}))
   plan['target_qualification_inputs']={'result':receipt,'review':identity.receipt(review)}
   with patch.object(identity,'validate_binding'),patch.object(support,'core',return_value=identity),patch.object(support,'verify_saved_sources'):
    got=support.verify_qualification_inputs(plan)
   self.assertEqual(got,value)
   findings['invalid_qualification_semantics_accepted']={'wrong_precision':True,'zero_finite_state_leaves':True,'failed_renderer':True,'setup_only_report':True,'memory_pid_uuid_differs_from_binding':True,'loaded_library_fields_absent':True,'unrelated_checks_mocked':['physical binding','source receipts']}
 def test_large_successful_process_capture_is_accepted(self):
  raw=f'{tests.GPU}, {tests.PID}, '+(' '*support.QUERY_LIMIT)+'512\n'
  sample=support.process_sample(tests.GPU,tests.PID,lambda command:tests.raw(raw.encode()))
  self.assertTrue(sample['valid']);self.assertEqual(support.validate_process_sample(sample,tests.GPU,tests.PID),512*2**20)
  findings['oversized_complete_process_capture_accepted']={'bytes':len(raw.encode()),'limit':support.QUERY_LIMIT}
 def test_allocator_failure_discards_proof(self):
  with tempfile.TemporaryDirectory() as tmp:
   d=pathlib.Path(tmp)/'paired_memory';v=support.MemoryObservation.__new__(support.MemoryObservation)
   v.uuid,v.pid,v.directory=tests.GPU,tests.PID,d;v.window_started=1.5;v.samples=[tests.memory_window()['samples'][0]];v.errors=[];v.normal_stop=True;v.counter=0
   v.stop=threading.Event();v.thread=types.SimpleNamespace(is_alive=lambda:False,join=lambda timeout:None)
   v.sample=lambda:tests.memory_window()['samples'][-1]
   def bad():raise RuntimeError('synthetic allocator failure')
   v.device=types.SimpleNamespace(memory_stats=bad)
   with self.assertRaisesRegex(RuntimeError,'synthetic allocator failure'):v.finish()
   self.assertFalse(d.exists())
   findings['allocator_failure_before_raw_proof_save']=True
 def test_invalid_successful_device_row_discards_raw(self):
  with tempfile.TemporaryDirectory() as tmp:
   v=support.MemoryObservation.__new__(support.MemoryObservation);v.uuid=tests.GPU;v.pid=os.getpid();v.directory=pathlib.Path(tmp)/'paired_memory'
   birth=pathlib.Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19]
   v.binding={'process':{'process_start_time':birth},'original_environment':{}}
   with patch.object(support,'bounded_query',return_value=tests.raw(b'wrong, 20470, 1, 20469\n')):
    with self.assertRaises(ValueError):v.sample()
   self.assertFalse(v.directory.exists())
   findings['invalid_successful_device_row_before_raw_failure_save']=True
suite=unittest.TestSuite([unittest.defaultTestLoader.loadTestsFromTestCase(tests.AssignedProcessTests),unittest.defaultTestLoader.loadTestsFromTestCase(tests.CompleteWindowTests),unittest.defaultTestLoader.loadTestsFromTestCase(Reproductions)])
started=time.time();result=unittest.TextTestRunner(verbosity=1).run(suite)
after={n:hashlib.sha256((S/n).read_bytes()).hexdigest() for n in files}
record={'schema_version':1,'kind':'configured_producer_independent_initial_findings','freeze_sha256':hashlib.sha256(freeze.read_bytes()).hexdigest(),'job_id':os.environ['SLURM_JOB_ID'],'step_id':os.environ['SLURM_STEP_ID'],'pid':os.getpid(),'birth':pathlib.Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19],'cgroup':pathlib.Path('/proc/self/cgroup').read_text(),'tests':result.testsRun,'errors':[(str(t),x) for t,x in result.errors],'failures':[(str(t),x) for t,x in result.failures],'findings':findings,'before':before,'after':after,'unchanged':before==after,'blocked_actions':blocked,'duration_seconds':time.time()-started,'numerical_work':False,'job_actions':0,'limits':'Stage-local reader reproductions deliberately replace unrelated already-reviewed physical/source gates; this is not valid physical qualification.'}
out=W/'.review_metadata'/('configured_producer_independent_findings_'+os.environ['SLURM_JOB_ID']+'_'+os.environ['SLURM_STEP_ID']+'.json')
with out.open('x') as stream:json.dump(record,stream,indent=2);stream.write('\n');stream.flush();os.fsync(stream.fileno())
print(json.dumps({'receipt':str(out),'sha256':hashlib.sha256(out.read_bytes()).hexdigest(),'tests':result.testsRun,'findings':findings,'success':result.wasSuccessful(),'step_id':os.environ['SLURM_STEP_ID']}),flush=True)
raise SystemExit(0 if result.wasSuccessful() and before==after and not blocked else 1)
