"""Independent semantic controls for exact profile scope and current monitoring metadata."""
import copy,datetime,hashlib,importlib.util,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('profile_scope_production',HERE/'submit_selected_profiles_held.py')
submit=importlib.util.module_from_spec(spec);spec.loader.exec_module(submit)
INPUTS=json.loads((HERE/'submission_inputs.json').read_bytes())
class ProfileScopeTests(unittest.TestCase):
 def setUp(self):
  self.item=copy.deepcopy(INPUTS['packets']['1']);self.plan=json.loads((Path(self.item['unit'])/'measurement_plan.json').read_bytes())
 def refuse(self,plan=None,item=None):
  with self.assertRaises(ValueError):submit.profile_scope(plan or self.plan,1,item or self.item)
 def test_both_actual_profile_scopes_pass(self):
  for copies in (1,2):
   item=INPUTS['packets'][str(copies)];plan=json.loads((Path(item['unit'])/'measurement_plan.json').read_bytes())
   submit.profile_scope(plan,copies,item)
 def test_measurement_direct_final_and_common_plan_fields_refuse(self):
  for name,value in (('mode','confirmation'),('rounds',7),('direct_binding',{}),('target_qualification',{}),('final_confirmation_protocol',{}),('common_correctness_inputs',{})):
   plan=copy.deepcopy(self.plan);plan[name]=value;self.refuse(plan)
 def test_boolean_and_reduced_or_extended_capture_counts_refuse(self):
  for value in (True,0,2):
   plan=copy.deepcopy(self.plan);plan['profile_updates']=value;self.refuse(plan)
 def test_changed_seeds_configuration_tool_and_review_joins_refuse(self):
  for key,value in (('copy_seeds',[2]),('profile_tool',{**self.plan['profile_tool'],'sha256':'1'*64})):
   plan=copy.deepcopy(self.plan);plan[key]=value;self.refuse(plan)
  plan=copy.deepcopy(self.plan);plan['configuration']['environment']['cache_room_backgrounds']=False;self.refuse(plan)
  item=copy.deepcopy(self.item);item['accepted_confirmation']=dict(item['accepted_confirmation'],sha256='1'*64);self.refuse(item=item)
 def test_nonprofile_consumer_and_resolved_plan_uuid_refuse(self):
  for label in ('selected_measurement','unknown'):
   plan=copy.deepcopy(self.plan);plan['configured_isolation']['consumer']=label;self.refuse(plan)
  plan=copy.deepcopy(self.plan);plan['gpu_uuid']='GPU-old';self.refuse(plan)
 def test_missing_common_helper_cannot_be_replaced_with_another_name(self):
  plan=copy.deepcopy(self.plan);files=plan['helpers']['files'];key=next(k for k in files if k.endswith('/common_confirmation.py'))
  files[key.replace('common_confirmation.py','other.py')]=files.pop(key);self.refuse(plan)
 def test_wrong_profile_kind_and_boolean_configuration_copies_refuse(self):
  plan=copy.deepcopy(self.plan);plan['kind']='selected_configuration_measurement';self.refuse(plan)
  plan=copy.deepcopy(self.plan);plan['configuration']['learner']['n_copies']=True;self.refuse(plan)
 def test_current_actual_readiness_passes_without_scheduler(self):
  submit.validate_readiness(INPUTS)
 def test_unexpected_profile_attempt_file_or_empty_directory_refuses(self):
  with tempfile.TemporaryDirectory() as temporary:
   unit=Path(temporary);(unit/'static.json').write_text('{}\n');item={'initial_unit_paths':['static.json']}
   submit.require_profile_unstarted(unit,item)
   (unit/'trace').mkdir()
   with self.assertRaises(ValueError):submit.require_profile_unstarted(unit,item)
   (unit/'trace').rmdir();(unit/'profile.lock').touch()
   with self.assertRaises(ValueError):submit.require_profile_unstarted(unit,item)
 def test_dangling_link_refuses_before_any_profile_submission(self):
  with tempfile.TemporaryDirectory() as temporary:
   unit=Path(temporary);(unit/'missing').symlink_to(unit/'absent')
   with self.assertRaises(ValueError):submit.require_profile_unstarted(unit,{'initial_unit_paths':['missing']})

class SyntheticReadinessTests(unittest.TestCase):
 def setUp(self):
  self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)
  now=datetime.datetime.now(datetime.timezone.utc)
  self.documents={
   'report':{'text':'synthetic report'},'status':{'jobs':[['6783003','RUNNING','','','heartpiece']]},
   'runtime':{'passed':True,'job_id':'6783003','source_commit':'79a01e851ff92ddba7d0e12eaf8164cab2028878','gpu_count':0,'source_git_archive_sha_and_executable_files_verified':265},
   'native':{'cron':'*/20 * * * *','cadence_seconds':1200,'list_verified':True,'persistent_job_id':'6783003','expires_at':(now+datetime.timedelta(days=1)).isoformat()},
   'verification':{'checked_at_utc':now.isoformat(),'passed':True,'returncode':0,'monitor_job_id':'6783003','source_commit':'79a01e851ff92ddba7d0e12eaf8164cab2028878'}}
 def write(self,name,value):
  p=self.root/(name+'.json');p.write_text(json.dumps(value)+'\n');return {'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
 def inputs(self):
  refs={name:self.write(name,self.documents[name]) for name in ('report','status','runtime','native')}
  self.documents['verification'].update(report_sha256=refs['report']['sha256'],status_sha256=refs['status']['sha256'])
  refs['verification']=self.write('verification',self.documents['verification'])
  ready={'schema_version':1,'kind':'actual_selected_profile_submission_monitor_readiness','passed':True,'actual_actions':0,
   'full_verification':refs['verification'],'full_status':refs['status'],'full_report':refs['report'],'runtime_review':refs['runtime'],'native_record':refs['native']}
  return {'submission_readiness':self.write('ready',ready)}
 def test_good_synthetic_monitor_passes(self):submit.validate_readiness(self.inputs())
 def test_stale_future_or_non_utc_full_proof_refuses(self):
  now=datetime.datetime.now(datetime.timezone.utc)
  for observed in (now-datetime.timedelta(seconds=1201),now+datetime.timedelta(seconds=10),now.replace(tzinfo=None)):
   self.documents['verification']['checked_at_utc']=observed.isoformat()
   with self.assertRaises(ValueError):submit.validate_readiness(self.inputs())
 def test_wrong_source_boolean_return_and_missing_cpu_row_refuse(self):
  for key,value in (('source_commit','0'*40),('returncode',False)):
   saved=self.documents['verification'][key];self.documents['verification'][key]=value
   with self.assertRaises(ValueError):submit.validate_readiness(self.inputs())
   self.documents['verification'][key]=saved
  self.documents['status']['jobs']=[['foreign','RUNNING']]
  with self.assertRaises(ValueError):submit.validate_readiness(self.inputs())
 def test_missing_native_schedule_or_expired_schedule_refuses(self):
  for key,value in (('cron','7 * * * *'),('list_verified',False),('expires_at','2000-01-01T00:00:00+00:00')):
   saved=self.documents['native'][key];self.documents['native'][key]=value
   with self.assertRaises(ValueError):submit.validate_readiness(self.inputs())
   self.documents['native'][key]=saved

if __name__=='__main__':unittest.main()
