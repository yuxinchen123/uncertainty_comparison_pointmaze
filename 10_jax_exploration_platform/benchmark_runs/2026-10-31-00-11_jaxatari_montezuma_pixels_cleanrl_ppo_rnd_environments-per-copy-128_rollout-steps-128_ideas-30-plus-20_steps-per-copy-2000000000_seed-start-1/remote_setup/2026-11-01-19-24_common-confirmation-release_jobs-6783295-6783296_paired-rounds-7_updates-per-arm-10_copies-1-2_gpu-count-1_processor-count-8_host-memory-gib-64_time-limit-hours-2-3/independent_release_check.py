RUN='/p/rlprojects/RND_worktrees/merge-atari-cleanrl/10_jax_exploration_platform/benchmark_runs/2026-10-31-00-11_jaxatari_montezuma_pixels_cleanrl_ppo_rnd_environments-per-copy-128_rollout-steps-128_ideas-30-plus-20_steps-per-copy-2000000000_seed-start-1'
PACKET='/p/rlprojects/RND_worktrees/merge-atari-cleanrl/10_jax_exploration_platform/benchmark_runs/2026-10-31-00-11_jaxatari_montezuma_pixels_cleanrl_ppo_rnd_environments-per-copy-128_rollout-steps-128_ideas-30-plus-20_steps-per-copy-2000000000_seed-start-1/remote_setup/2026-11-01-19-24_common-confirmation-release_jobs-6783295-6783296_paired-rounds-7_updates-per-arm-10_copies-1-2_gpu-count-1_processor-count-8_host-memory-gib-64_time-limit-hours-2-3'
import builtins,contextlib,copy,ctypes,datetime,hashlib,importlib.util,io,json,os,pathlib,subprocess,sys,unittest
from unittest import mock
R=pathlib.Path(RUN);P=pathlib.Path(PACKET)
O=R/'remote_setup'/(datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=-8))).strftime('%Y-%m-%d-%H-%M')+'_common-confirmation-release-independent-review_jobs-6783295-6783296_gpu-count-0')
O.mkdir()
assert os.environ['SLURM_JOB_ID']=='6783003' and os.environ['SLURM_STEP_ID'].isdigit()
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def ref(p):return {'path':str(p),'sha256':sha(p)}
def read(p):return json.loads(pathlib.Path(p).read_bytes())
fpath=P/'source_freeze.json';assert sha(fpath)=='d8eb9dd15e963af8fa981c9e9c1b6f02b8efd64ee9cd89ed7c3d9a77284415a0';f=read(fpath)
before={p:sha(p) for p in f['files']};assert before==f['files']
inputs=read(P/'release_frozen_inputs.json');assert sha(P/'release_frozen_inputs.json')=='ae959ea98a6daba71b0c7a651a8342e4c959c05d6b269d3af7c31ed6ed540a9b'
allfiles={**inputs['shared_files'],**inputs['jobs']['1']['files'],**inputs['jobs']['2']['files']}
images=lambda:{p:{'sha256':sha(p),'bytes':pathlib.Path(p).stat().st_size} for p in allfiles}
frozen_before=images();assert frozen_before==allfiles
NUM={'jax','jaxlib','numpy','torch','flax','optax','jaxtari','cupy','scipy','pynvml','exploration_platform'};orig=builtins.__import__;blocked=[];commands=[]
def guard(name,*a,**k):
 if name.split('.')[0] in NUM:blocked.append(name);raise RuntimeError('numerical import forbidden')
 return orig(name,*a,**k)
builtins.__import__=guard
real_popen=subprocess.Popen
class OnlyMetadata(real_popen):
 def __init__(self,cmd,*a,**k):
  ok=isinstance(cmd,list) and len(cmd)>3 and cmd[:2]==['git','-C'] and pathlib.Path(cmd[2]).is_absolute() and cmd[3] in {'show','cat-file','rev-parse','status','symbolic-ref','ls-tree','merge-base'}
  if not ok:blocked.append(cmd);raise RuntimeError('live child forbidden')
  commands.append(cmd);super().__init__(cmd,*a,**k)
subprocess.Popen=OnlyMetadata
sys.addaudithook(lambda e,a:(_ for _ in ()).throw(RuntimeError('live operation forbidden')) if e in {'socket.connect','socket.bind','os.system'} or e=='ctypes.dlopen' and a[0] is not None else None)
spec=importlib.util.spec_from_file_location('independent_release_tests',P/'test_release_common_confirmation.py');t=importlib.util.module_from_spec(spec);spec.loader.exec_module(t);m=t.release
class Extra(unittest.TestCase):
 def fixture(self):
  x=t.QualificationReleaseTests();x.setUp();self.addCleanup(x.doCleanups);return x
 def test_existing_partial_preflight_refuses_before_intent(self):
  x=self.fixture();p=x.packet/'release_preflight_stdout.raw';p.write_bytes(b'previous partial')
  with mock.patch.object(m.subprocess,'run') as call:
   with self.assertRaises(ValueError):m.release_once(x.intent)
   call.assert_not_called()
  self.assertEqual(p.read_bytes(),b'previous partial');self.assertFalse((x.packet/'release_intent.json').exists())
 def test_preflight_warning_blocks_release_and_repeat(self):
  x=self.fixture()
  with mock.patch.object(m.subprocess,'run',return_value=type('Return',(),{'returncode':0,'stdout':x.held().encode(),'stderr':b'warning'})()) as call:
   with self.assertRaises(ValueError):m.release_once(x.intent)
   self.assertEqual(call.call_count,1)
  self.assertFalse((x.packet/'release_receipt.json').exists())
  with mock.patch.object(m.subprocess,'run') as call:
   with self.assertRaises(ValueError):m.release_once(x.intent)
   call.assert_not_called()
 def test_release_timeout_keeps_return_and_blocks_repeat(self):
  x=self.fixture()
  def fake(cmd,**kw):
   if cmd[1]=='show':return x.fake_run(cmd,**kw)
   raise subprocess.TimeoutExpired(cmd,30,output=b'partial release response')
  with mock.patch.object(m.subprocess,'run',side_effect=fake):
   with self.assertRaises(subprocess.TimeoutExpired):m.release_once(x.intent)
  self.assertEqual((x.packet/'release_stdout.raw').read_bytes(),b'partial release response')
  self.assertIsNone(m.document(x.packet/'release_return.json')['returncode'])
  self.assertFalse((x.unit/'owned_allocation_inputs.json').exists())
  with mock.patch.object(m.subprocess,'run') as call:
   with self.assertRaises(ValueError):m.release_once(x.intent)
   call.assert_not_called()
 def test_actual_proof_cadence_and_foreign_schedule_refused(self):
  m.select(1);proof=m.document(m.SELECTED['held_proof']['path'])
  for key,value in [('monitor',True),('schedule','foreign')]:
   p=copy.deepcopy(proof)
   if key=='monitor':p['monitor']['cadence_seconds']=value
   else:p['recurring_session_task']['id']=value
   with self.subTest(key=key),self.assertRaises(ValueError):m.validate_proof(p)
log=io.StringIO();suite=unittest.TestSuite([unittest.defaultTestLoader.loadTestsFromModule(t),unittest.defaultTestLoader.loadTestsFromTestCase(Extra)])
with contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):result=unittest.TextTestRunner(stream=log,verbosity=2).run(suite)
assert result.wasSuccessful() and result.testsRun==24,log.getvalue()
defaults={};proof_expiry=datetime.datetime.fromisoformat(f['expires_at_utc'].replace('Z','+00:00'))
for copies in [1,2]:
 with mock.patch.object(m.socket,'gethostname',return_value='portal05'):
  intent=m.checked(copies)
 assert intent['job_id']=={1:'6783295',2:'6783296'}[copies] and intent['publication_commit'] is None
 defaults[str(copies)]={'passed':True,'portal_hostname_explicitly_simulated_on_owned_CPU':True,'actual_cached_proofs_source_ledgers_and_staged_core_checked':True,'job_id':intent['job_id'],'command':intent['command'],'scheduler_queries':0,'release_calls':0}
 own=inputs['jobs'][str(copies)]['files'];other=inputs['jobs'][str(3-copies)]
 other_progress=str(pathlib.Path(other['unit'])/'progress.json')
 assert other_progress not in inputs['shared_files'] and other_progress not in own
 assert str(pathlib.Path(inputs['jobs'][str(copies)]['unit'])/'progress.json') in own
for p,v in allfiles.items():assert sha(p)==v['sha256']
after={p:sha(p) for p in f['files']};assert before==after==f['files'] and images()==frozen_before and not blocked
now=datetime.datetime.now(datetime.timezone.utc);assert now<proof_expiry
v={'schema_version':1,'kind':'independent_actual_common_confirmation_release_review','approved':True,'passed':True,'at_utc':now.isoformat(),'cpu_job_id':'6783003','cpu_step_id':os.environ['SLURM_STEP_ID'],'source_freeze':ref(fpath),'source_before':before,'source_after':after,'frozen_inputs':ref(P/'release_frozen_inputs.json'),'frozen_prerequisites_before':frozen_before,'frozen_prerequisites_after':images(),'independent_test_methods':24,'author_test_methods_repeated':20,'extra_independent_methods':4,'test_output':log.getvalue(),'cached_metadata_checks':defaults,'proofs_expire_at_utc':f['expires_at_utc'],'proof_seconds_remaining':(proof_expiry-now).total_seconds(),'selected_target_mutable_state_isolation_verified':True,'dangling_pending_link_refused_before_any_intent_or_scheduler':True,'real_success_receipt_fsynced_before_atomic_final_envelope':True,'no_retry_after_partial_or_ambiguous_attempt':True,'publication_default_and_release_still_required_on_actual_portal05':True,'approved_only_for_literal_owned_jobs':['6783295','6783296'],'publish_exact_source_proofs_and_this_review_before_action':True,'release_permitted_only_before_frozen_proof_expiry_and_after_actual_publication_checks':True,'monitor_display_limit':ref(P/'monitor_unit_display_limit.json'),'actual_release_actions':0,'actual_scheduler_queries':0,'actual_gpu_operations':0,'numerical_work':False,'scientific_credit':0,'admission_granted':False,'allowed_read_only_metadata_commands':commands}
out=O/'review.json'
with out.open('x') as stream:json.dump(v,stream,indent=2,sort_keys=True,allow_nan=False);stream.write('\n');stream.flush();os.fsync(stream.fileno())
print(json.dumps({'passed':True,'approved':True,'review':ref(out),'tests':24,'expires_at_utc':f['expires_at_utc'],'seconds_remaining':v['proof_seconds_remaining']}))
