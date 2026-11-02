"""Run synthetic held-submission controls and real local defaults without live commands."""
from datetime import datetime,timezone
import contextlib,stat
import hashlib,importlib.abc,io,json,os,pathlib,subprocess,sys,unittest
P=pathlib.Path('/p/rlprojects/RND_worktrees/merge-atari-cleanrl/10_jax_exploration_platform/benchmark_runs/2026-10-31-00-11_jaxatari_montezuma_pixels_cleanrl_ppo_rnd_environments-per-copy-128_rollout-steps-128_ideas-30-plus-20_steps-per-copy-2000000000_seed-start-1/remote_setup/2026-11-01-15-36_common-correctness-held-submission_gpu-count-1_processor-count-8_host-memory-gib-64_time-limit-hours-3_copies-1-2')
assert os.environ.get('SLURM_JOB_ID')=='6782816' and os.environ.get('SLURM_STEP_ID','').isdecimal()
names=['submit_common_correctness_held.py','test_submit_common_correctness_held.py','submission_inputs.json']
images=lambda:{n:hashlib.sha256((P/n).read_bytes()).hexdigest() for n in names}
before=images();inputs=json.loads((P/'submission_inputs.json').read_bytes());protected={row['path']:row['sha256'] for row in inputs['required_files']};assert all(hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()==h for p,h in protected.items());blocked=[]
class NoNumerical(importlib.abc.MetaPathFinder):
 def find_spec(self,name,path=None,target=None):
  if name.split('.')[0] in {'jax','jaxlib','numpy','torch','flax','optax','jaxtari','cupy','scipy'}:blocked.append(name);raise RuntimeError('numerical import forbidden')
sys.meta_path.insert(0,NoNumerical())
real_popen=subprocess.Popen
class GitOnly(real_popen):
 def __init__(self,args,*a,**k):
  command=list(args) if isinstance(args,(list,tuple)) else []
  assert len(command)>3 and command[:2]==['git','-C'] and command[3] in {'show','cat-file','rev-parse','status','diff','ls-tree','merge-base','symbolic-ref'},'real scheduler/network process forbidden'
  super().__init__(args,*a,**k)
subprocess.Popen=GitOnly
sys.addaudithook(lambda event,args:(_ for _ in ()).throw(RuntimeError('GPU/library/network action forbidden')) if event in {'socket.connect','socket.bind','socket.getaddrinfo'} or event=='ctypes.dlopen' and args[0] is not None else None)
sys.path.insert(0,str(P));import test_submit_common_correctness_held as tests
roster=json.loads((P/'publication_roster.json').read_bytes());assert roster['file_count']==len(roster['files'])==185
frozen_paths={item['absolute_path']:item['sha256'] for item in roster['files']}
frozen_paths[str(P/'publication_roster.json')]=hashlib.sha256((P/'publication_roster.json').read_bytes()).hexdigest()
frozen_before={}
for path,expected in frozen_paths.items():
 value=pathlib.Path(path);assert value.resolve(strict=True)==value and stat.S_ISREG(value.lstat().st_mode)
 assert hashlib.sha256(value.read_bytes()).hexdigest()==expected
 frozen_before[path]={'sha256':expected,'mode':format(stat.S_IMODE(value.stat().st_mode),'04o'),'bytes':value.stat().st_size}
def copy_two_actual_core(self):
 with tests.patch.object(tests.submit.subprocess,'run',return_value=self.result(b'9910002\n')):self.invoke(2)
 unit=self.unit(2);pre=json.loads((unit/'configured_isolation_pre_release_inputs.json').read_text())
 self.assertEqual(pre['submission_row'][0],'9910002')
 release=unit/'synthetic_copy_two_release.json'
 value={'schema_version':1,'kind':'owned_uva_configured_isolation_release','job_id':'9910002','cluster':'uva','plan_sha256':pre['plan']['sha256'],'data_unit':str(unit),'returncode':0,'command':['scontrol','release','9910002'],'submission_receipt_sha256':pre['submission']['sha256']}
 release.write_text(json.dumps(value)+'\n')
 final={**pre,'kind':'uva_configured_isolation_owned_inputs','release':tests.submit.receipt(release)}
 (unit/'owned_allocation_inputs.json').write_text(json.dumps(final)+'\n')
 spec=tests.importlib.util.spec_from_file_location('independent_copy_two_owned_inputs',unit/'code/uva_slurm_configured_isolation.py');core=tests.importlib.util.module_from_spec(spec);spec.loader.exec_module(core)
 self.assertEqual(core.owned_inputs(self.contexts[2][1]),final)
 value['command']=['scontrol','release','9910001'];release.write_text(json.dumps(value)+'\n')
 final['release']=tests.submit.receipt(release);(unit/'owned_allocation_inputs.json').write_text(json.dumps(final)+'\n')
 with self.assertRaises(ValueError):core.owned_inputs(self.contexts[2][1])
def private_snapshot_failure(self):
 real_publish=tests.submit.publish_bytes
 def fail_private(path,raw):
  if path==self.unit(2)/'configured_isolation_owned_job_ids.txt':
   self.assertTrue(all(ledger.read_text().splitlines().count('9910002')==1 for ledger in self.ledgers[:2]))
   self.assertEqual(sum(line.split('\t')[0]=='9910002' for line in self.ledgers[2].read_text().splitlines()),1)
   raise OSError('synthetic private snapshot failure after complete global ledger append')
  return real_publish(path,raw)
 with tests.patch.object(tests.submit.subprocess,'run',return_value=self.result(b'9910002\n')) as scheduler,tests.patch.object(tests.submit,'publish_bytes',side_effect=fail_private):
  with self.assertRaises(OSError):self.invoke(2)
  scheduler.assert_called_once()
 self.assertTrue(all(ledger.read_text().splitlines().count('9910002')==1 for ledger in self.ledgers[:2]))
 failure=json.loads((self.path(2)/'submission_failure.json').read_text());self.assertEqual(failure['actual_returned_job_id_if_validated'],'9910002')
 self.assertFalse((self.path(2)/'submission_receipt.json').exists());self.assertFalse((self.unit(2)/'configured_isolation_pre_release_inputs.json').exists())
 before_retry=[ledger.read_bytes() for ledger in self.ledgers]
 with tests.patch.object(tests.submit.subprocess,'run') as scheduler:
  with self.assertRaises(ValueError):self.invoke(2)
  scheduler.assert_not_called()
 self.assertEqual(before_retry,[ledger.read_bytes() for ledger in self.ledgers])
tests.HeldSubmissionTests.test_independent_copy_two_actual_core_release_and_wrong_command_refusal=copy_two_actual_core
tests.HeldSubmissionTests.test_independent_private_snapshot_failure_after_global_append_never_retries=private_snapshot_failure
result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(tests))
actual=[]
if result.wasSuccessful():
 for copies in [1,2]:
  output=io.StringIO()
  with contextlib.redirect_stdout(output):tests.submit.main(['--copies',str(copies)])
  default=json.loads(output.getvalue());assert default['submissions']==default['scheduler_queries']==default['releases']==0
  raw,item,plan=tests.submit.validate_inputs(copies)
  try:tests.submit.main(['--copies',str(copies),'--publication-commit','c851fa90f7b25d6ced54b7bac31457106ac91aa3'])
  except (subprocess.CalledProcessError,ValueError) as error:old={'refused':True,'error_type':type(error).__name__,'error':str(error)}
  else:raise AssertionError('old publication accepted new preparation bytes')
  assert not (pathlib.Path(item['packet'])/'submission_intent.json').exists()
  actual.append({'copies':copies,'mode':plan['mode'],'fixed_command':item['command'],'scheduler_queries':0,'submission_calls':0,'plan_sha256':item['plan']['sha256'],'actual_default':default,'old_publication':old})
assert images()==before and not blocked and all(hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()==h for p,h in protected.items())
frozen_after={path:{'sha256':hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest(),'mode':format(stat.S_IMODE(pathlib.Path(path).stat().st_mode),'04o'),'bytes':pathlib.Path(path).stat().st_size} for path in frozen_before}
assert frozen_before==frozen_after
value={'schema_version':1,'kind':'common_correctness_one_shot_held_submission_independent_controls','passed':result.wasSuccessful(),'at_utc':datetime.now(timezone.utc).isoformat(),'job_id':'6782816','step_id':os.environ['SLURM_STEP_ID'],'pid':os.getpid(),'birth':pathlib.Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19],'cgroup':pathlib.Path('/proc/self/cgroup').read_text(),'tests':result.testsRun,'errors':len(result.errors),'failures':len(result.failures),'error_details':[{'test':test.id(),'traceback':message} for test,message in result.errors],'failure_details':[{'test':test.id(),'traceback':message} for test,message in result.failures],'actual_packet_defaults':actual,'frozen186_before':frozen_before,'frozen186_after':frozen_after,'source_before':before,'source_after':images(),'protected_packet_and_independent_review_files':len(protected),'protected_bytes_unchanged':True,'actual_scheduler_queries':0,'actual_submissions':0,'release_actions':0,'network_actions':0,'GPU_operations':0,'numerical_import_attempts':blocked,'scientific_credit':0,'admission_granted':False}
f=pathlib.Path('/p/rlprojects/RND_worktrees/merge-atari-cleanrl/10_jax_exploration_platform/benchmark_runs/2026-10-31-00-11_jaxatari_montezuma_pixels_cleanrl_ppo_rnd_environments-per-copy-128_rollout-steps-128_ideas-30-plus-20_steps-per-copy-2000000000_seed-start-1/remote_setup/2026-11-01-15-55-16_common-correctness-held-submission_independent-review_copies-1-2_gpu-count-0')/('independent_controls_6782816_'+os.environ['SLURM_STEP_ID']+'.json')
with f.open('x') as out:json.dump(value,out,sort_keys=True,indent=2);out.write('\n');out.flush();os.fsync(out.fileno())
print(json.dumps({'passed':value['passed'],'tests':result.testsRun,'receipt':str(f),'sha256':hashlib.sha256(f.read_bytes()).hexdigest(),'actual_step':os.environ['SLURM_STEP_ID']}));raise SystemExit(0 if result.wasSuccessful() else 1)
