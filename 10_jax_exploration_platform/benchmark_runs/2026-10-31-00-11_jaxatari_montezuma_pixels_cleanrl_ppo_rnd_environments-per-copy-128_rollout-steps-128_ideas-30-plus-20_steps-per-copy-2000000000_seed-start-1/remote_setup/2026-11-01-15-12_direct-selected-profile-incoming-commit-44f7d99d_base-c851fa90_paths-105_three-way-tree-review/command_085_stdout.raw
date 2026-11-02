"""Isolate saved-result semantic joins with explicitly synthetic unrelated proof stubs; no qualification credit."""
import copy,ctypes,datetime,hashlib,importlib.abc,json,os,pathlib,subprocess,sys,unittest
from unittest.mock import patch
W=pathlib.Path(__file__).resolve().parents[1];S=W/'10_jax_exploration_platform/scripts/atari_pixel_campaign'
assert os.environ.get('SLURM_JOB_ID')=='6781231' and os.environ.get('SLURM_STEP_ID','').isdigit()
freeze=W/'.review_metadata/direct_adapter_fourth_source_freeze.json';spec=json.loads(freeze.read_text());sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest();before={n:sha(W/n) for n in spec['files']};assert before==spec['files']
blocked=[]
class Block(importlib.abc.MetaPathFinder):
 def find_spec(self,name,path=None,target=None):
  if name.split('.')[0] in {'jax','jaxlib','numpy','scipy','torch','flax','optax','jaxtari','pynvml','cupy'}:blocked.append(name);raise AssertionError('numerical import refused')
sys.meta_path.insert(0,Block());subprocess.Popen=lambda *a,**k:(_ for _ in ()).throw(AssertionError('real subprocess refused'));ctypes.CDLL=lambda *a,**k:(_ for _ in ()).throw(AssertionError('real shared library refused'))
sys.path[:0]=[str(S),str(S/'tests')]
import direct_measurement as direct
import configured_measurement as shared
import test_configured_measurement as fixtures
core=direct.core()
class SavedJoinReproductions(unittest.TestCase):
 def setUp(self):
  self.f=fixtures.QualificationReaderTests();self.f.setUp();self.addCleanup(self.f.doCleanups)
  f=self.f;f.plan.pop('configured_isolation');f.plan.pop('target_qualification');f.plan.update(direct_binding={'profile':copy.deepcopy(core.PROFILE),'consumer':'selected_measurement','current_launch_inputs_path':'synthetic saved launch'},direct_qualification=direct.QUALIFICATION,hardware_class='h100',copy_seeds=[1])
  f.result.update(kind='direct_serval02_card1_full_original_qualification',qualification_protocol=direct.QUALIFICATION,hardware_class='h100',direct_binding=copy.deepcopy(f.plan['direct_binding']),copy_seeds=[1],execution_identity={'hostname':'serval02','pid':fixtures.PID,'process_start_time':'100'})
  f.result.pop('configured_isolation');f.result.update(direct.BOUNDARY)
  self.process={'hostname':'serval02','pid':fixtures.PID,'process_start_time':'100'}
 def validate(self,mutation=None,outer_mutation=None):
  f=self.f;f.publish()
  for name,value in [('direct_process_metadata.json',{'after':{'identity':self.process}}),('direct_device_binding.json',f.binding),('direct_terminal_process.json',{'synthetic_unrelated_terminal_stub':True})]: (f.unit/name).write_text(json.dumps(value))
  result=copy.deepcopy(f.result)
  if mutation: mutation(result)
  path=f.unit/'direct_qualification_result.json';path.write_text(json.dumps(result))
  review={'schema_version':1,'kind':'independent_full_original_direct_qualification_review','result':core.receipt(path),'approved':True,'admission_granted':False,'scientific_credit':0,'independent_of_producer':True,'terminal_process':core.receipt(f.unit/'direct_terminal_process.json'),'actual_returncode':0,**self.process}
  ref=f.unit/'direct_review.json';ref.write_text(json.dumps(review));outer=copy.deepcopy(f.plan);outer['direct_qualification_inputs']={'result':core.receipt(path),'review':core.receipt(ref)}
  if outer_mutation:outer_mutation(outer)
  with patch.object(core,'validate_plan'),patch.object(core,'validate_binding'),patch.object(direct,'validate_original_qualification'),patch.object(direct,'verify_saved_sources'),patch.object(direct,'verify_qualification_source_receipts'),patch.object(direct,'validate_window',return_value=f.result['qualification_memory']),patch.object(direct,'validate_startup_checks'),patch.object(direct,'validate_backend_capture'),patch.object(direct,'terminal_process'):
   return direct.verify_qualification_inputs(outer)
 def test_synthetic_unrelated_stubs_leave_complete_saved_semantic_reader_positive(self):self.assertTrue(self.validate()['completed'])
 def test_missing_full_direct_binding_is_currently_accepted(self):self.assertEqual(self.validate(lambda r:r.update(direct_binding={'profile':core.PROFILE}))['direct_binding'],{'profile':core.PROFILE})
 def test_relabelled_result_seed_is_currently_accepted(self):self.assertEqual(self.validate(lambda r:r.update(copy_seeds=[999]))['copy_seeds'],[999])
 def test_relabelled_result_execution_identity_is_currently_accepted(self):self.assertEqual(self.validate(lambda r:r['execution_identity'].update(pid=999999))['execution_identity']['pid'],999999)
 def test_result_and_outer_changed_runtime_can_hide_unchanged_saved_runtime(self):
  changed={'synthetic_changed_runtime':'yes'};self.assertEqual(self.validate(lambda r:r.update(runtime_environment=changed),lambda p:p.update(runtime_environment=changed))['runtime_environment'],changed)
result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(SavedJoinReproductions));after={n:sha(W/n) for n in spec['files']};assert before==after and not blocked
receipt=W/'.review_metadata'/('direct_qualification_saved_join_reproductions_6781231_'+os.environ['SLURM_STEP_ID']+'.json')
value={'schema_version':1,'kind':'direct_qualification_reader_saved_join_defect_reproductions','at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'job':'6781231','step':os.environ['SLURM_STEP_ID'],'freeze_sha256':sha(freeze),'tests':result.testsRun,'failures':[(t.id(),v) for t,v in result.failures],'errors':[(t.id(),v) for t,v in result.errors],'before':before,'after':after,'scope':'Deliberately synthetic renderer/library/raw-test fixtures; source/native/lock/window/startup/backend/terminal proofs stubbed solely to isolate actual saved-result reader joins. This establishes no physical, scientific or qualification evidence.','defects_demonstrated':['full saved direct_binding can be reduced to profile only','result seed is not joined to saved plan','execution PID is not joined to saved process','result+outer runtime can differ from saved qualified plan'],'actual_GPU_work':False,'actual_direct_or_scheduler_operations':False,'qualification_credit':0,'independent_review_approved':False}
with receipt.open('x') as f:json.dump(value,f,indent=2,sort_keys=True);f.write('\n');f.flush();os.fsync(f.fileno())
print(json.dumps({'receipt':str(receipt),'sha256':sha(receipt),'tests':result.testsRun,'success':result.wasSuccessful()}),flush=True)
sys.exit(0 if result.wasSuccessful() else 1)
