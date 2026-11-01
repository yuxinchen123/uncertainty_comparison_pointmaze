"""Run the direct-core metadata suite only on the already owned CPU allocation."""
import ctypes,datetime,hashlib,importlib.abc,json,os,pathlib,subprocess,sys,unittest
W=pathlib.Path(__file__).resolve().parents[1];S=W/'10_jax_exploration_platform/scripts/atari_pixel_campaign'
def require(value,message):
 if not value:raise RuntimeError(message)
require(os.environ.get('SLURM_JOB_ID')=='6781231','only ownedCPU6781231 may run this suite')
step=os.environ.get('SLURM_STEP_ID','');require(step.isdecimal(),'an actual numeric step is required')
frozen=W/'.review_metadata/direct_serval02_core_fourth_source_freeze.json';expected=json.loads(frozen.read_text())['files']
images=lambda:{name:hashlib.sha256((W/name).read_bytes()).hexdigest() for name in expected}
before=images();require(before==expected,'source changed before tests')
blocked=[]
class Block(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname.split('.')[0] in {'jax','jaxlib','numpy','torch','flax','optax','cupy','scipy'}:
   blocked.append(fullname);raise RuntimeError('numerical import prohibited: '+fullname)
sys.meta_path.insert(0,Block())
real_popen=subprocess.Popen
class LocalGitOnly(real_popen):
 def __init__(self,args,*a,**kw):
  cmd=list(args) if isinstance(args,(tuple,list)) else []
  allowed=False
  if cmd[:2]==['git','clone']:
   allowed=pathlib.Path(cmd[-2])==W and pathlib.Path(cmd[-1]).is_absolute() and str(pathlib.Path(cmd[-1])).startswith('/tmp/')
  elif len(cmd)>3 and cmd[:2]==['git','-C']:
   root=pathlib.Path(cmd[2]);allowed=root.is_absolute() and (str(root).startswith('/tmp/') or root==W and cmd[3] in {'show','status','rev-parse','ls-tree','symbolic-ref','cat-file'})
  require(allowed and not any(x in {'fetch','push','pull','remote'} for x in cmd),'only local temporary Git metadata operations are permitted')
  super().__init__(args,*a,**kw)
subprocess.Popen=LocalGitOnly
ctypes.CDLL=lambda *a,**kw:(_ for _ in ()).throw(RuntimeError('shared-library loads prohibited'))
sys.path[:0]=[str(S),str(S/'tests')]
import copy
from unittest.mock import patch
import test_serval02_card1_binding as fixtures
import serval02_card1_binding as core
extra={}
class DirectReproductions(unittest.TestCase):
 @classmethod
 def setUpClass(cls):fixtures.DirectBindingTests.setUpClass()
 @classmethod
 def tearDownClass(cls):fixtures.DirectBindingTests.tearDownClass()
 def setUp(self):
  self.f=fixtures.DirectBindingTests();self.f.setUp();self.addCleanup(self.f.doCleanups)
 def test_live_claim_accepts_nonexistent_numerical_root(self):
  f=self.f;p=f.write(f.run/'SOURCE_AND_GATES_READY.json',{'synthetic':'original admission'});checksum=core.receipt(p)['sha256']
  claim={**f.metadata['current_admission'],'canonical_path':str(p),'frozen_admission':{'path':str(f.unit/'direct_history/admission.json'),'sha256':checksum}}
  self.assertFalse(pathlib.Path(claim['source_root']).exists())
  with patch.object(core,'ORIGINAL_ADMISSION_SHA256',checksum):self.assertEqual(core.check_live_admission(claim),claim)
  extra['missing_live_numerical_root_accepted']={'source_root':claim['source_root'],'scope':'stage-local active-source helper, synthetic admission hash only; no actual authority changed'}
 def test_metadata_and_binding_accept_external_output_directory(self):
  f=self.f;outside=f.run/'outside_unit';outside.mkdir()
  with patch.object(core,'current_process',return_value=f.process),patch.object(core,'check_live_admission'),patch.dict(os.environ,{'CUDA_VISIBLE_DEVICES':f.profile['gpu_uuid']},clear=True):
   core.collect_metadata(f.plan,outside,f.fd)
  self.assertTrue((outside/'direct_process_metadata.json').is_file())
  value=copy.deepcopy(f.binding);target=outside/'captured_driver'/pathlib.Path(value['native']['driver_library']).name;target.parent.mkdir();target.write_bytes(pathlib.Path(value['driver_capture']['captured']['path']).read_bytes())
  value['driver_capture']['captured']=core.receipt(target)
  core.validate_binding(f.plan,f.metadata,value,directory=outside)
  extra['external_metadata_and_binding_directory_accepted']=True
 def test_profile_parent_owner_can_substitute_for_child_binding(self):
  f=self.f;f.test_inherited_profile_metadata_proves_holder_not_child_ownership()
  value=copy.deepcopy(f.binding);value['plan']=core.receipt(f.unit/'measurement_plan.json');value['metadata_sha256']=core.digest(core.encoded(f.metadata))
  core.validate_binding(f.plan,f.metadata,value)
  self.assertEqual(f.plan['direct_binding']['consumer'],'component_profile');self.assertEqual(f.metadata['role'],'owner')
  extra['profile_parent_owner_binding_accepted']=True
 def test_frozen_exit_plan_can_omit_manifest_covered_child(self):
  f=self.f;f.history_patch.stop();original=core.validate_history;seen=[];tampered=[False]
  def wrapped(plan):
   if not tampered[0]:
    tampered[0]=True
    inputs_path=pathlib.Path(plan['direct_binding']['frozen_inputs']['path']);inputs=core.read_receipt(plan['direct_binding']['frozen_inputs'])
    exit_path=pathlib.Path(inputs['exit_plan']['path']);exit_plan=core.read_receipt(inputs['exit_plan'])
    missing='queue/leases/synthetic.json';self.assertIn(missing,exit_plan['inputs']['files'])
    manifest=core.read_receipt(inputs['evidence_manifest']);original_child_path=str(pathlib.Path(plan['campaign_run_dir'])/missing)
    self.assertTrue(any(row['original_path']==original_child_path for row in manifest['files']))
    exit_plan['inputs']['files'].pop(missing);exit_plan['inputs']['runners']=[]
    observations=core.read_receipt(inputs['exit_observations']);raw=json.loads(observations['direct_processes'][0]['stdout']);raw['observations']=[row for row in raw['observations'] if row['pid']!=79];observations['direct_processes'][0]['stdout']=json.dumps(raw)
    collector=core.adjacent('collect_worker_exit_receipts');jobs=exit_plan['inputs']['jobs'];stamp=observations['scheduler']['observed_at']
    rebuilt_jobs={identifier:{'job_id':identifier,'owner':'sl5nw','state':'COMPLETED','state_raw':'COMPLETED','exit_code':'0:0','node_list':'jaguar03','start':'start','end':'end','submission_row':jobs[identifier],'observed_at':stamp} for identifier in [line.split('|')[0] for line in observations['scheduler']['stdout'].splitlines()]}
    direct={('serval02',row['pid'],row['expected_birth']):{**row,'hostname':'serval02','observed_at':stamp} for row in raw['observations']}
    stop,waiting=collector.make_receipt(exit_plan,rebuilt_jobs,direct);self.assertEqual(waiting,[])
    status=core.read_receipt(inputs['exit_status']);status['counts']['runners']=0
    f.write(exit_path,exit_plan);status['prepared_plan_sha256']=core.receipt(exit_path)['sha256']
    inputs['exit_plan']=core.receipt(exit_path)
    for name,value in [('exit_observations',observations),('worker_stop',stop),('exit_status',status)]:
     path=pathlib.Path(inputs[name]['path']);f.write(path,value);inputs[name]=core.receipt(path)
    f.write(inputs_path,inputs);plan['direct_binding']['frozen_inputs']=core.receipt(inputs_path)
   result=original(plan);seen.append(result);return result
  with patch.object(core,'validate_history',side_effect=wrapped):f.test_real_existing_reader_rederives_frozen_synthetic_120_case_history()
  self.assertTrue(seen);self.assertEqual(seen[0]['comparisons'],120);self.assertEqual(seen[0]['stop']['direct_processes'],1)
  extra['omitted_manifest_child_accepted']={'validated_comparisons':120,'remaining_direct_processes':1,'removed_original_child_pid':79,'raw_lease_still_in_frozen_manifest':True,'scope':'all-source/gate constants explicitly synthetic as in author fixture; real unchanged cohort reader used'}
suite=unittest.defaultTestLoader.loadTestsFromTestCase(DirectReproductions)
result=unittest.TextTestRunner(verbosity=2).run(suite)
after=images();require(before==after,'source changed during tests');require(not blocked,'numerical imports were attempted')
raw=pathlib.Path('/proc/self/stat').read_text()
value={'schema_version':1,'kind':'direct_serval02_core_independent_initial_findings','extra':extra,'at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
 'job_id':'6781231','step_id':step,'pid':os.getpid(),'birth':raw[raw.rfind(')')+2:].split()[19],
 'cgroup':pathlib.Path('/proc/self/cgroup').read_text(),'cpu_affinity':sorted(os.sched_getaffinity(0)),
 'source_freeze_sha256':hashlib.sha256(frozen.read_bytes()).hexdigest(),'source_before':before,'source_after':after,
 'tests':result.testsRun,'errors':len(result.errors),'failures':len(result.failures),'passed':result.wasSuccessful(), 'error_details':[{'test':test.id(),'traceback':trace} for test,trace in result.errors], 'failure_details':[{'test':test.id(),'traceback':trace} for test,trace in result.failures],
 'blocked_numerical_imports':blocked,'process_calls_limited_to_local_git':True,'temporary_fork_test_uses_only_own_lock_metadata':True,
 'shared_library_loads_blocked':True,'actual_serval02_contact':False,'actual_gpu_operations':False,'scientific_credit':0}
p=W/'.review_metadata'/('direct_serval02_core_independent_findings_corrected_6781231_'+step+'.json')
with p.open('x') as f:json.dump(value,f,sort_keys=True,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
print(json.dumps({'receipt':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'passed':result.wasSuccessful(),'tests':result.testsRun}))
sys.exit(0 if result.wasSuccessful() else 1)
