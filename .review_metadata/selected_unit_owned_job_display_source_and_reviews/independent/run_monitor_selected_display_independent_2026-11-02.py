"""Review the fixed selected-unit display using metadata only on the owned CPU allocation."""
import datetime,hashlib,importlib.abc,io,json,os
from pathlib import Path
import subprocess,sys,unittest
from unittest.mock import patch
W=Path('/p/rlprojects/RND_worktrees/2026-11-01-20-03_monitor-selected-unit-owned-job-display_prepared-held-running_campaign-ledger-and-receipt')
S=W/'10_jax_exploration_platform/scripts/atari_pixel_campaign'
R=Path('/p/rlprojects/RND_worktrees/merge-atari-cleanrl/10_jax_exploration_platform/benchmark_runs/2026-10-31-00-11_jaxatari_montezuma_pixels_cleanrl_ppo_rnd_environments-per-copy-128_rollout-steps-128_ideas-30-plus-20_steps-per-copy-2000000000_seed-start-1')
assert os.environ.get('SLURM_JOB_ID')=='6783003' and os.environ.get('SLURM_STEP_ID','').isdigit()
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
freeze=W/'.review_metadata/selected_unit_display_corrected_source_freeze.json'
assert sha(freeze)=='d62f28c26a5abbaf853b73517184d38fb64d8e33a6b4cf54f3cdf51c221090df'
paths=[S/'monitor.py',S/'tests/test_monitor_selected_unit_jobs.py']
expected=['dea8682097e23b4c6ecaff4801914017a7fb7e04672d4ece1cfb3f46de62ea84','01544a6f71434636c18a9f30abebe7f931f8a551815ba7a764eb1b8a186848ba']
images=lambda:{str(p):{'sha256':sha(p),'mode':p.stat().st_mode&0o777,'size_bytes':p.stat().st_size} for p in paths}
before=images();assert [v['sha256'] for v in before.values()]==expected
blocked=[]
class NoNumerical(importlib.abc.MetaPathFinder):
 def find_spec(self,name,path=None,target=None):
  if name.split('.')[0] in {'jax','jaxlib','numpy','torch','flax','optax','scipy','cupy','jaxtari','exploration_platform','pynvml','matplotlib'}:
   blocked.append(name);raise RuntimeError('numerical import forbidden: '+name)
sys.meta_path.insert(0,NoNumerical())
def no_child(*args,**kwargs):
 blocked.append(repr(args));raise RuntimeError('real child process forbidden')
subprocess.Popen=no_child
sys.addaudithook(lambda event,args:(_ for _ in ()).throw(RuntimeError('network/device/library action forbidden: '+event)) if event in {'socket.connect','socket.bind','socket.getaddrinfo','ctypes.dlopen','os.system'} else None)
sys.path[:0]=[str(S),str(S/'tests')]
import monitor,test_monitor_selected_unit_jobs as focused
class IndependentControls(unittest.TestCase):
 def fixture(self):
  case=focused.SelectedUnitDisplayTests();case.setUp();self.addCleanup(case.doCleanups);return case
 def test_profile_consumer_uses_same_owned_state_join(self):
  c=self.fixture();c.plan['kind']='selected_configuration_profile';c.plan['configured_isolation']['consumer']='component_profile';c.write(c.unit/'measurement_plan.json',c.plan);c.bind()
  self.assertEqual(c.row(c.jobs())['state'],'held')
 def test_wrong_profile_consumer_cannot_claim_running(self):
  c=self.fixture();c.plan['configured_isolation']['consumer']='component_profile';c.write(c.unit/'measurement_plan.json',c.plan);c.bind();c.unknown(c.jobs('RUNNING'))
 def test_rehashed_foreign_release_is_refused(self):
  c=self.fixture();c.bind();c.finish_release();c.returned['job_id']='12345002';c.write(c.packet/'release_receipt.json',c.returned);c.final['release']=c.ref(c.packet/'release_receipt.json');c.write(c.unit/'owned_allocation_inputs.json',c.final);c.unknown(c.jobs('RUNNING'))
 def test_live_ledger_symlink_cannot_supply_owned_state(self):
  c=self.fixture();c.bind();live=c.root/'slurm/submitted_jobids_uva.txt';target=c.root/'different_ids';target.write_bytes(live.read_bytes());live.unlink();live.symlink_to(target);c.unknown(c.jobs('RUNNING'))
 def test_boolean_returncode_does_not_count_as_submission_success(self):
  c=self.fixture();c.bind();c.submission['returncode']=False;c.write(c.packet/'submission_receipt.json',c.submission);c.pre['submission']=c.ref(c.packet/'submission_receipt.json');c.write(c.unit/'configured_isolation_pre_release_inputs.json',c.pre);c.unknown(c.jobs('RUNNING'))
 def test_terminal_failure_remains_terminal_with_metadata_error(self):
  c=self.fixture();c.bind();c.write(c.unit/'progress.json',{'state':'failed','error':'actual producer failure'});c.pre['job_id']='12345002';c.write(c.unit/'configured_isolation_pre_release_inputs.json',c.pre);row=c.row(c.jobs('RUNNING'));self.assertEqual(row['state'],'failed');self.assertEqual(row['slurm_job_id'],'');self.assertIn('metadata error',row['progress'])
 def test_unknown_selected_metadata_preserves_unrelated_held_row(self):
  c=self.fixture();c.bind();other='12345002';c.write(c.unit/'hardware.json',{'slurm_job_id':other,'hostname':'jaguar03'});jobs=c.jobs('RUNNING')+[[other,'PENDING','','gres/gpu:0','(JobHeldUser)']]
  with patch.object(monitor,'own_jobs',return_value=jobs), __import__('contextlib').redirect_stdout(io.StringIO()):monitor.tick(c.root)
  report=(c.root/'20_mins_monitoring/latest_report.md').read_text();self.assertIn('UNKNOWN',report);held=report.split('HELD\n')[1].split('FINISHED\n')[0];self.assertIn('held study job',held)
 def test_duplicate_scheduler_rows_are_refused_even_when_identical(self):
  c=self.fixture();c.bind();c.unknown(c.jobs('RUNNING')*2)
modules=['test_monitor_selected_unit_jobs','test_monitor','test_monitor_process_hostname']
suite=unittest.TestSuite([*(unittest.defaultTestLoader.loadTestsFromName(name) for name in modules),unittest.defaultTestLoader.loadTestsFromTestCase(IndependentControls)])
output=io.StringIO();result=unittest.TextTestRunner(stream=output,verbosity=2).run(suite)
after=images();assert before==after
value={'schema_version':1,'kind':'independent_selected_unit_owned_job_display_source_review','approved':result.wasSuccessful() and not blocked,'passed':result.wasSuccessful() and not blocked,'job_id':'6783003','step_id':os.environ['SLURM_STEP_ID'],'pid':os.getpid(),'at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'source_freeze':{'path':str(freeze),'sha256':sha(freeze)},'before':before,'after':after,'tests':result.testsRun,'errors':[{'test':t.id(),'traceback':e} for t,e in result.errors],'failures':[{'test':t.id(),'traceback':e} for t,e in result.failures],'output':output.getvalue(),'blocked_actions':blocked,'numerical_imports':[],'actual_scheduler_queries':0,'actual_job_actions':0,'GPU_or_library_or_network_operations':0,'source_review_scope':'Configured UVA selected measurement/profile display only; exact receipt, pre-release/final envelope, private/live owned ledgers, existing current scheduler rows. No scientific acceptance or admission. Direct and legacy paths remain unchanged. Invalid selected hardware cannot hide another generic owned running row.','admission_granted':False,'scientific_credit':0,'case_credit':0,'idea_credit':0}
p=R/'remote_setup'/('selected_unit_display_independent_6783003_'+os.environ['SLURM_STEP_ID']+'.json')
with p.open('x') as f:json.dump(value,f,sort_keys=True,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
print(json.dumps({'passed':value['passed'],'approved':value['approved'],'tests':value['tests'],'review':str(p),'sha256':sha(p),'errors':value['errors'],'failures':value['failures']}));raise SystemExit(0 if value['passed'] else 1)
