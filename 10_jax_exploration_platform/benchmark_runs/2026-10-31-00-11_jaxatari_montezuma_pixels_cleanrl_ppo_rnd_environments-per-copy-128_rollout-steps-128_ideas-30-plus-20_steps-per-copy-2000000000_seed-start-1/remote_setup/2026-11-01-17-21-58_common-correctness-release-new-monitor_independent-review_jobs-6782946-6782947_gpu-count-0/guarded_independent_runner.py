"""Run bounded metadata release tests on the existing owned CPU allocation."""
import ctypes,datetime,hashlib,importlib.abc,json,os,pathlib,socket,subprocess,sys,traceback,unittest
from unittest.mock import patch
P=pathlib.Path('/p/rlprojects/RND_worktrees/merge-atari-cleanrl/10_jax_exploration_platform/benchmark_runs/2026-10-31-00-11_jaxatari_montezuma_pixels_cleanrl_ppo_rnd_environments-per-copy-128_rollout-steps-128_ideas-30-plus-20_steps-per-copy-2000000000_seed-start-1/remote_setup/2026-11-01-16-33_common-correctness-release-new-monitor_jobs-6782946-6782947_copies-1-2_gpu-count-1_processor-count-8_host-memory-gib-64_time-limit-hours-3')
assert os.environ.get('SLURM_JOB_ID')=='6783003' and os.environ.get('SLURM_STEP_ID','').isdecimal()
step=os.environ['SLURM_STEP_ID'];print('ACTUAL_OWNED_STEP=6783003.'+step,flush=True)
names=['release_common_correctness.py','test_release_common_correctness.py','release_frozen_inputs.json','author_review.json']
images=lambda:{n:hashlib.sha256((P/n).read_bytes()).hexdigest() for n in names}
before=images();blocked=[]
assert before['release_common_correctness.py']=='3806a02a349fd3b2fd189a76b96deef5f3a0ece5b4c72c13efb7080a5f47400d'
assert before['release_frozen_inputs.json']=='e63805d76d005d74f41be82b9a263b0deeb4ccd171db72bcd4e7c9015815762e'
assert hashlib.sha256((P/'source_freeze.json').read_bytes()).hexdigest()=='d1cfbe8a49475979dcbe7b6b9f31ac129edbdbec3d19a10116468df4c8cf35f4'
frozen=json.loads((P/'release_frozen_inputs.json').read_text());protected={path:value['sha256'] for path,value in frozen['files'].items()}
protected_before={path:hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest() for path in protected};assert protected_before==protected
class Block(importlib.abc.MetaPathFinder):
 def find_spec(self,name,path=None,target=None):
  if name.split('.')[0] in {'jax','jaxlib','numpy','torch','flax','optax','cupy','scipy'}:blocked.append(name);raise RuntimeError('numerical import forbidden')
sys.meta_path.insert(0,Block())
ctypes.CDLL=lambda *a,**k:(_ for _ in ()).throw(RuntimeError('shared-library loads forbidden'))
socket.socket=lambda *a,**k:(_ for _ in ()).throw(RuntimeError('network action forbidden'))
real_popen=subprocess.Popen
allowed={str(P.parents[4]),'/p/rlprojects/RND_worktrees/atari-cleanrl-source-d28e673','/p/rlprojects/RND_worktrees/2026-11-01-13-50_full-original-qualification-explicit-pytest-root_source-commit-e5afbf6b51a3d1666b46e71c0631a45d47080cd8'}
class GitOnly(real_popen):
 def __init__(self,args,*a,**k):
  cmd=list(args) if isinstance(args,(list,tuple)) else []
  assert len(cmd)>3 and cmd[:2]==['git','-C'] and cmd[2] in allowed and cmd[3] in {'show','cat-file','rev-parse','status','symbolic-ref','ls-tree','merge-base'},'live command forbidden: '+repr(cmd)
  super().__init__(args,*a,**k)
subprocess.Popen=GitOnly
sys.path.insert(0,str(P));import test_release_common_correctness as tests
roster=json.loads((P/'publication_roster.json').read_bytes());assert len(roster['files'])==roster['file_count']==219
roster_before={item['absolute_path']:hashlib.sha256(pathlib.Path(item['absolute_path']).read_bytes()).hexdigest() for item in roster['files']}
assert roster_before=={item['absolute_path']:item['sha256'] for item in roster['files']}
def copy_two_actual_core(self):
 import shutil
 actual=pathlib.Path(tests.release.SCOPES['2']['unit'])
 self.plan=json.loads((actual/'measurement_plan.json').read_bytes());self.plan['campaign_run_dir']=str(self.run);self.plan['unit_id']=self.unit.name
 self.plan['configured_isolation']['owned_inputs_path']=str(self.unit/'owned_allocation_inputs.json')
 self.write(self.unit/'measurement_plan.json',self.plan)
 shutil.copytree(actual/'code',self.unit/'code')
 (self.run/'slurm').mkdir()
 identifier='6782947';row=[identifier,self.plan['kind'],tests.release.SOURCE,str(self.packet/'job.sbatch')]
 ledger=self.run/'slurm/submitted_jobids_uva.txt';ledger.write_text(identifier+'\n')
 (self.run/'slurm/submitted_jobids.txt').write_text(identifier+'\n')
 rows=self.run/'slurm/submissions.tsv';rows.write_text('\t'.join(row)+'\n')
 ids_snapshot=self.unit/'configured_isolation_owned_job_ids.txt';ids_snapshot.write_bytes(ledger.read_bytes())
 rows_snapshot=self.unit/'configured_isolation_owned_submissions.tsv';rows_snapshot.write_bytes(rows.read_bytes())
 submitted=self.packet/'submission_receipt.json'
 self.write(submitted,{'schema_version':1,'kind':'owned_uva_configured_isolation_submission','job_id':identifier,'cluster':'uva','plan_sha256':tests.release.ref(self.unit/'measurement_plan.json')['sha256'],'data_unit':str(self.unit),'returncode':0,'held_command':['sbatch','--hold','--no-requeue',str(self.packet/'job.sbatch')],'submission_row':row})
 self.pre={'schema_version':1,'kind':'uva_configured_isolation_pre_release_inputs','job_id':identifier,'plan':tests.release.ref(self.unit/'measurement_plan.json'),'ledger':{'source_path':str(ledger),'snapshot':tests.release.ref(ids_snapshot)},'submissions':{'source_path':str(rows),'snapshot':tests.release.ref(rows_snapshot)},'submission':tests.release.ref(submitted),'submission_row':row}
 self.write(self.unit/'configured_isolation_pre_release_inputs.json',self.pre)
 values={'JOB':identifier,'COMMAND':['scontrol','release',identifier],'PLAN_SHA':self.pre['plan']['sha256'],'SUBMISSION_SHA':self.pre['submission']['sha256'],'PRE_RELEASE_SHA':tests.release.ref(self.unit/'configured_isolation_pre_release_inputs.json')['sha256']}
 real_run=tests.release.subprocess.run
 def synthetic_or_git(command,**arguments):
  if command[0]=='git':return real_run(command,**arguments)
  return self.fake_run(command,**arguments)
 with tests.patch.multiple(tests.release,**values),tests.patch.object(tests.release.subprocess,'run',side_effect=synthetic_or_git):
  intent={**self.intent,'job_id':identifier,'command':['scontrol','release',identifier]}
  observed=tests.release.release_once(intent)
  self.assertTrue(observed['actual_core_reader_passed'])
  saved=tests.release.document(self.unit/'owned_allocation_inputs.json')
  self.assertEqual(saved['release'],tests.release.ref(self.packet/'release_receipt.json'))
  self.assertEqual(self.calls,[['scontrol','show','job','-o',identifier],['scontrol','release',identifier]])
tests.QualificationReleaseTests.test_independent_copy_two_final_release_chain_passes_actual_unchanged_core=copy_two_actual_core
def current_runtime_refusal(self):
 release=tests.release;release.select(1);proof=release.document(release.SELECTED['held_proof']['path']);runtime_ref=proof['monitor']['runtime_and_first_full_report_review'];original=release.checked_ref;baseline=original(runtime_ref)
 for key,value in [('job_id','6782816'),('source_commit','31fd9fe7cd96343741e8b48a18019efee9b4ac5c'),('gpu_count',True),('source_git_archive_sha_and_executable_files_verified',True),('numerical_or_gpu_operations',False)]:
  changed={**baseline,key:value}
  with self.subTest(field=key),patch.object(release,'checked_ref',side_effect=lambda ref:changed if ref==runtime_ref else original(ref)):
   with self.assertRaises(ValueError):release.validate_proof(proof)
tests.QualificationReleaseTests.test_independent_current_runtime_identity_and_boolean_refusals=current_runtime_refusal
suite=unittest.defaultTestLoader.loadTestsFromModule(tests);result=unittest.TextTestRunner(verbosity=2).run(suite)
extra={}
try:
 with patch.object(tests.release.socket,'gethostname',return_value='portal05'):
  for copies in (1,2):
   actual=tests.release.checked(copies)
   assert actual['job_id']=={'1':'6782946','2':'6782947'}[str(copies)] and actual['publication_commit'] is None
   extra[str(copies)]={'actual_local_default_passed':True,'job_id':actual['job_id'],'no_intent_written':not (tests.release.PACKET/'release_intent.json').exists()}
except BaseException:
 extra['traceback']=traceback.format_exc()
protected_after={path:hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest() for path in protected};assert images()==before and protected_after==protected_before and not blocked
roster_after={path:hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest() for path in roster_before};assert roster_before==roster_after
raw=pathlib.Path('/proc/self/stat').read_text()
v={'schema_version':1,'kind':'fixed_common_correctness_release_independent_metadata_tests','at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'job_id':'6783003','step_id':step,'pid':os.getpid(),'birth':raw[raw.rfind(')')+2:].split()[19],'cgroup':pathlib.Path('/proc/self/cgroup').read_text(),'cpu_affinity':sorted(os.sched_getaffinity(0)),'tests':result.testsRun,'errors':len(result.errors),'failures':len(result.failures),'error_details':[{'test':t.id(),'traceback':s} for t,s in result.errors],'failure_details':[{'test':t.id(),'traceback':s} for t,s in result.failures],'passed':result.wasSuccessful() and set(extra)=={'1','2'},'actual_defaults':extra,'protected_file_count':len(protected),'publication_roster219_before':roster_before,'publication_roster219_after':roster_after,'protected_before':protected_before,'protected_after':protected_after,'source_before':before,'source_after':images(),'numerical_import_attempts':blocked,'shared_library_loads_blocked':True,'only_real_subprocesses_are_read_only_git':True,'actual_scheduler_queries':0,'actual_release_calls':0,'final_owned_inputs_written':False}
f=pathlib.Path('/p/rlprojects/RND_worktrees/merge-atari-cleanrl/10_jax_exploration_platform/benchmark_runs/2026-10-31-00-11_jaxatari_montezuma_pixels_cleanrl_ppo_rnd_environments-per-copy-128_rollout-steps-128_ideas-30-plus-20_steps-per-copy-2000000000_seed-start-1/remote_setup/2026-11-01-17-21-58_common-correctness-release-new-monitor_independent-review_jobs-6782946-6782947_gpu-count-0')/('independent_controls_6783003_'+step+'.json')
with f.open('x') as stream:json.dump(v,stream,sort_keys=True,indent=2);stream.write('\n');stream.flush();os.fsync(stream.fileno())
print(json.dumps({'receipt':str(f),'sha256':hashlib.sha256(f.read_bytes()).hexdigest(),'passed':v['passed'],'tests':result.testsRun,'actual_defaults':extra}));sys.exit(0 if v['passed'] else 1)
