import pathlib,sys,json,hashlib,importlib.util,unittest,io,subprocess,datetime,os
R=pathlib.Path(sys.argv[1]);M=pathlib.Path(sys.argv[2]);D=R/'remote_setup/2026-11-01-14-34_monitor-held-submission-root-review_source-31fd9fe7_gpu-count-0_processor-count-1'
D.mkdir(exist_ok=False)
def save(name,value):
 raw=value if isinstance(value,bytes) else (json.dumps(value,indent=2,allow_nan=False)+'\n').encode()
 with (D/name).open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
 return {'path':str(D/name),'sha256':hashlib.sha256(raw).hexdigest()}
assert os.environ.get('SLURM_JOB_ID')=='6781231' and os.environ.get('SLURM_STEP_ID','').isdecimal()
freeze=json.loads((M/'held_submission_source_freeze.json').read_bytes())
for path,digest in freeze['files'].items():assert hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()==digest
before={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [M/'submit_monitor_held.py',M/'test_submit_monitor_held.py',M/'held_submission_inputs.json',R/'slurm/submitted_jobids.txt',R/'slurm/submitted_jobids_uva.txt',R/'slurm/submissions.tsv']}
def guard(event,args):
 if event=='import' and str(args[0]).split('.')[0] in {'numpy','jax','jaxlib','torch','flax','optax','jaxtari'}:raise RuntimeError('numerical import refused')
 if event=='subprocess.Popen':
  argv=args[1]
  if not isinstance(argv,(list,tuple)) or argv[:3]!=['git','-C',str(R.parent.parent.parent)] or not any(x in argv for x in ['show','merge-base']):raise RuntimeError('actual child refused')
 if event.startswith('socket.') and event!='socket.gethostname':raise RuntimeError('network refused')
sys.addaudithook(guard)
spec=importlib.util.spec_from_file_location('held_tests',M/'test_submit_monitor_held.py');t=importlib.util.module_from_spec(spec);spec.loader.exec_module(t)
class IndependentHeldSubmissionControls(t.HeldMonitorSubmissionTests):
 def test_broken_symlink_attempt_cannot_be_repeated(self):
  p=self.packet/'submission_intent.json';p.symlink_to(self.packet/'missing')
  with t.patch.object(t.submit.subprocess,'run') as child:
   with self.assertRaises(ValueError):t.submit.submit_once('a'*40)
   child.assert_not_called()
  self.assertTrue(p.is_symlink())
 def test_partial_ledger_failure_retains_intent_and_blocks_second_submission(self):
  def partial(campaign,job,row):
   with self.paths[0].open('a') as f:f.write(job+'\n');f.flush();os.fsync(f.fileno())
   raise OSError('independent simulated second-ledger failure')
  with t.patch.object(t.submit,'append_ledgers',side_effect=partial),t.patch.object(t.submit.subprocess,'run',return_value=self.result()) as child:
   with self.assertRaises(OSError):t.submit.submit_once('a'*40)
   self.assertEqual(child.call_count,1)
  self.assertEqual(self.paths[0].read_text().splitlines()[-1],'9910001')
  self.assertTrue((self.packet/'submission_return.json').exists())
  self.assertTrue((self.packet/'submission_failure.json').exists())
  self.assertFalse((self.packet/'submission_receipt.json').exists())
  with t.patch.object(t.submit.subprocess,'run') as child:
   with self.assertRaises(ValueError):t.submit.submit_once('a'*40)
   child.assert_not_called()
output=io.StringIO();result=unittest.TextTestRunner(stream=output,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(IndependentHeldSubmissionControls));save('tests_stdout.raw',output.getvalue().encode());assert result.wasSuccessful()
inputs,plan=t.submit.validate_inputs();assert len(inputs['required_files'])==38
try:t.submit.validate_publication(inputs,'550425fd8ec68f11e60fb93f873bd77ee7b06f0b')
except (ValueError,subprocess.CalledProcessError) as e:negative={'rejected':True,'type':type(e).__name__}
else:raise AssertionError('old publication incorrectly accepted')
after={p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest() for p in before};assert before==after
assert not any((M/name).exists() or (M/name).is_symlink() for name in t.submit.ATTEMPTS)
review={'schema_version':1,'kind':'independent_monitor_held_submission_source_and_controls_review','approved':True,'at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'job_id':os.environ['SLURM_JOB_ID'],'step_id':os.environ['SLURM_STEP_ID'],'source_read_in_full':True,'source_sha256':before[str(M/'submit_monitor_held.py')],'tests_run':result.testsRun,'errors':len(result.errors),'failures':len(result.failures),'actual_required_files_verified':38,'actual_old_publication_rejected':negative,'original_before':before,'original_after':after,'old_monitor_preserved':'6781231','scheduler_calls':0,'submission_calls':0,'release_calls':0,'new_job_id':None,'scientific_credit':0,'admission_granted':False}
print(json.dumps({'review':save('review.json',review),'tests':result.testsRun,'actual_step':os.environ['SLURM_STEP_ID']}))
