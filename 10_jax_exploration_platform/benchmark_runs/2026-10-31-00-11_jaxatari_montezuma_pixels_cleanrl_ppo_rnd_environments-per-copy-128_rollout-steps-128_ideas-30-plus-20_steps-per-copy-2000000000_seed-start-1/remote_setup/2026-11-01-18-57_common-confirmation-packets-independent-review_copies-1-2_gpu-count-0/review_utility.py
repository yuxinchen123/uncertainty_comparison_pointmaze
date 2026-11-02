UTILITY='/p/rlprojects/RND_worktrees/merge-atari-cleanrl/10_jax_exploration_platform/benchmark_runs/2026-10-31-00-11_jaxatari_montezuma_pixels_cleanrl_ppo_rnd_environments-per-copy-128_rollout-steps-128_ideas-30-plus-20_steps-per-copy-2000000000_seed-start-1/remote_setup/2026-11-01-18-57_common-confirmation-held-submission_paired-rounds-7_updates-per-arm-10_copies-1-2_gpu-count-1_processor-count-8_host-memory-gib-64_time-limit-hours-2-3'
OUTPUT='/p/rlprojects/RND_worktrees/merge-atari-cleanrl/10_jax_exploration_platform/benchmark_runs/2026-10-31-00-11_jaxatari_montezuma_pixels_cleanrl_ppo_rnd_environments-per-copy-128_rollout-steps-128_ideas-30-plus-20_steps-per-copy-2000000000_seed-start-1/remote_setup/2026-11-01-18-57_common-confirmation-packets-independent-review_copies-1-2_gpu-count-0'
RUN='/p/rlprojects/RND_worktrees/merge-atari-cleanrl/10_jax_exploration_platform/benchmark_runs/2026-10-31-00-11_jaxatari_montezuma_pixels_cleanrl_ppo_rnd_environments-per-copy-128_rollout-steps-128_ideas-30-plus-20_steps-per-copy-2000000000_seed-start-1'
"""Review the final literal held-submission utility without real scheduler actions."""
import ast,builtins,contextlib,ctypes,datetime,hashlib,importlib.util,io,json,os,pathlib,stat,subprocess,sys,unittest
from unittest import mock
U=pathlib.Path(UTILITY);O=pathlib.Path(OUTPUT);R=pathlib.Path(RUN)
assert os.environ['SLURM_JOB_ID']=='6783003' and os.environ['SLURM_STEP_ID'].isdigit()
def sha(raw):return hashlib.sha256(raw).hexdigest()
def ref(p):return {'path':str(p),'sha256':sha(p.read_bytes())}
def read(p):return json.loads(p.read_bytes())
def save(name,value):
 p=O/name;raw=value if isinstance(value,bytes) else (json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n').encode()
 with p.open('xb') as stream:stream.write(raw);stream.flush();os.fsync(stream.fileno())
 return ref(p)
freeze=U/'source_freeze.json';assert ref(freeze)['sha256']=='7a73dc6b9cbe5a91ea172b50faa90dbd2b93006fa6c5b7bcf3927522995c4e09'
f=read(freeze);source_before={p:sha(pathlib.Path(p).read_bytes()) for p in f['files']};assert source_before==f['files']
pf=R/'remote_setup/2026-11-01-19-00_common-confirmation_prepared-packets_copies-1-2_rounds-7_updates-per-arm-10.json';packets=read(pf)
def packet_images():
 return {p:{'sha256':sha(pathlib.Path(p).read_bytes()),'bytes':pathlib.Path(p).stat().st_size,'mode':stat.S_IMODE(pathlib.Path(p).stat().st_mode)} for p in packets['files']}
before=packet_images();assert before==packets['files']
NUM={'jax','jaxlib','numpy','torch','cupy','flax','optax','scipy','jaxtari','exploration_platform','pynvml'}
orig_import=builtins.__import__;blocked=[];allowed=[]
def guarded(name,*args,**kwargs):
 if name.split('.')[0] in NUM:blocked.append(name);raise RuntimeError('real numerical import forbidden')
 return orig_import(name,*args,**kwargs)
builtins.__import__=guarded
real_popen=subprocess.Popen
class MetadataOnly(real_popen):
 def __init__(self,args,*other,**kwargs):
  cmd=list(args) if isinstance(args,(tuple,list)) else [];ok=False
  if cmd[:2]==['git','-C'] and len(cmd)>3:ok=pathlib.Path(cmd[2]).is_absolute() and cmd[3] in {'show','cat-file','rev-parse','merge-base','status','symbolic-ref','ls-tree'}
  elif cmd[:5]==['/usr/bin/python3','-I','-S','-B',str(U/'submit_common_confirmation_held.py')]:ok='--submit-held' not in cmd
  if not ok:blocked.append(cmd);raise RuntimeError('unexpected child: '+repr(cmd))
  allowed.append(cmd);super().__init__(args,*other,**kwargs)
subprocess.Popen=MetadataOnly
sys.addaudithook(lambda event,args:(_ for _ in ()).throw(RuntimeError('live operation forbidden')) if event in {'socket.connect','socket.bind','os.system'} or event=='ctypes.dlopen' and args[0] is not None else None)
spec=importlib.util.spec_from_file_location('independent_final_submit_tests',U/'test_submit_common_confirmation_held.py');tests=importlib.util.module_from_spec(spec);spec.loader.exec_module(tests);submit=tests.submit
inputs=read(U/'submission_inputs.json');assert ref(U/'submission_inputs.json')['sha256']==submit.INPUTS_SHA=='035353595780d30bd12b5ecb7ed3989af42e005dcd65473dfbee8b41210139e2'
assert inputs['independent_packet_review'] in inputs['required_files'] and inputs['packet_freeze'] in inputs['required_files']
assert inputs['independent_packet_review']['sha256']=='5dfe5004468178855440873e007c52c78ad18c7e0108dc6b572b4601f787a8ad'
real_validate=submit.validate_inputs
class ExtraControls(unittest.TestCase):
 def fixture(self):
  item=tests.HeldSubmissionTests();item.setUp();self.addCleanup(item.doCleanups);return item
 def test_success_with_stderr_refused_and_not_repeated(self):
  item=self.fixture();before=[p.read_bytes() for p in item.ledgers]
  with mock.patch.object(submit.subprocess,'run',return_value=item.result(stderr=b'synthetic warning')) as call:
   with self.assertRaises(ValueError):item.invoke()
  self.assertEqual(call.call_count,1);self.assertEqual(before,[p.read_bytes() for p in item.ledgers])
  self.assertEqual((item.path()/'submission_stdout.raw').read_bytes(),b'9910001\n')
  with mock.patch.object(submit.subprocess,'run') as call:
   with self.assertRaises(ValueError):item.invoke()
   call.assert_not_called()
 def test_third_ledger_fsync_failure_keeps_row_and_blocks_retry(self):
  item=self.fixture();original=submit.os.fsync;hit=[]
  def fail(fd):
   target=pathlib.Path(submit.os.readlink('/proc/self/fd/'+str(fd)))
   if target==item.ledgers[2] and not hit:hit.append(str(target));raise OSError('synthetic third ledger fsync failure')
   return original(fd)
  with mock.patch.object(submit.subprocess,'run',return_value=item.result()),mock.patch.object(submit.os,'fsync',side_effect=fail):
   with self.assertRaises(OSError):item.invoke()
  self.assertEqual(len(hit),1);self.assertTrue(all(p.read_text().splitlines()[-1]=='9910001' for p in item.ledgers[:2]))
  self.assertEqual(item.ledgers[2].read_text().splitlines()[-1].split('\t')[0],'9910001')
  self.assertFalse((item.path()/'submission_receipt.json').exists())
  with mock.patch.object(submit.subprocess,'run') as call:
   with self.assertRaises(ValueError):item.invoke()
   call.assert_not_called()
 def test_dangling_private_ownership_sidecar_refused(self):
  item=self.fixture();(item.unit()/'configured_isolation_pre_release_inputs.json').symlink_to(item.root/'missing')
  with mock.patch.object(submit.subprocess,'run') as call:
   with self.assertRaises(ValueError):item.invoke()
   call.assert_not_called()
 def test_unsafe_override_cli_is_rejected_before_child_or_write(self):
  with mock.patch.object(submit.subprocess,'run') as call,contextlib.redirect_stderr(io.StringIO()):
   with self.assertRaises(SystemExit):submit.main(['--copies','1','--gpu-uuid','GPU-other'])
   call.assert_not_called()
 def test_boolean_and_other_copy_counts_are_refused_before_file_access(self):
  for copies in [True,0,3]:
   with self.subTest(copies=copies),mock.patch.object(submit,'receipt') as receipt:
    with self.assertRaises(ValueError):real_validate(copies)
    receipt.assert_not_called()
log=io.StringIO();suite=unittest.TestSuite([unittest.defaultTestLoader.loadTestsFromTestCase(tests.HeldSubmissionTests),unittest.defaultTestLoader.loadTestsFromTestCase(ExtraControls)])
with contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):result=unittest.TextTestRunner(stream=log,verbosity=2).run(suite)
assert result.wasSuccessful() and result.testsRun==22,log.getvalue()
default_results={}
for copies in [1,2]:
 capture=io.StringIO()
 with contextlib.redirect_stdout(capture):submit.main(['--copies',str(copies)])
 record=json.loads(capture.getvalue());assert record['validated'] is True and record['submissions']==record['releases']==record['scheduler_queries']==0 and record['owned_launch_inputs_written'] is False
 default_results[str(copies)]=record
# Real earlier publication lacks these new packets and must be refused without a held submission.
old=subprocess.run(['/usr/bin/python3','-I','-S','-B',str(U/'submit_common_confirmation_held.py'),'--copies','1','--publication-commit','9fcc43a478b93d731df780805fd7c3c514cdf67f'],capture_output=True)
old_out=save('utility_old_publication.stdout.raw',old.stdout);old_err=save('utility_old_publication.stderr.raw',old.stderr)
assert old.returncode!=0 and b'does not exist in' in old.stderr
# Test every exact published object lookup without creating a commit or changing any source.
required=[*inputs['required_files'],submit.receipt(U/'submission_inputs.json'),submit.receipt(U/'submit_common_confirmation_held.py'),submit.receipt(U/'test_submit_common_confirmation_held.py'),submit.receipt(U/'author_review.json')]
lookups=[];commit='a'*40;repository=pathlib.Path(inputs['repository'])
objects={commit+':'+str(pathlib.Path(rec['path']).relative_to(repository)):pathlib.Path(rec['path']).read_bytes() for rec in required}
def exact_object(command,*a,**k):
 assert command[:4]==['git','-C',str(repository),'show'] and len(command)==5
 lookups.append(command[-1]);return objects[command[-1]]
with mock.patch.object(submit.subprocess,'run',return_value=subprocess.CompletedProcess([],0)),mock.patch.object(submit.subprocess,'check_output',side_effect=exact_object):
 assert submit.validate_publication(inputs,commit)==commit
assert len(lookups)==len(required)==159
first=required[0];key=commit+':'+str(pathlib.Path(first['path']).relative_to(repository));objects[key]+=b'changed'
with mock.patch.object(submit.subprocess,'run',return_value=subprocess.CompletedProcess([],0)),mock.patch.object(submit.subprocess,'check_output',side_effect=exact_object):
 try:submit.validate_publication(inputs,commit)
 except ValueError:pass
 else:raise AssertionError('changed published object accepted')
# Only metadata names changed in the reviewed durable submission mechanism.
old_source=R/'remote_setup/2026-11-01-15-36_common-correctness-held-submission_gpu-count-1_processor-count-8_host-memory-gib-64_time-limit-hours-3_copies-1-2/submit_common_correctness_held.py'
old_ast=ast.parse(old_source.read_bytes());new_ast=ast.parse((U/'submit_common_confirmation_held.py').read_bytes())
def functions(tree):return {n.name:n for n in tree.body if isinstance(n,ast.FunctionDef)}
a,b=functions(old_ast),functions(new_ast)
class Labels(ast.NodeTransformer):
 def visit_Constant(self,node):
  if isinstance(node.value,str):node.value=node.value.replace('correctness','confirmation')
  return node
for name in a:
 if name=='validate_inputs':continue
 assert ast.dump(Labels().visit(a[name]),include_attributes=False)==ast.dump(b[name],include_attributes=False),name
after=packet_images();source_after={p:sha(pathlib.Path(p).read_bytes()) for p in f['files']}
assert before==after==packets['files'] and source_before==source_after==f['files'] and not blocked
for item in packets['packets']:
 p=pathlib.Path(item['packet']);u=pathlib.Path(item['unit'])
 assert not any(os.path.lexists(p/name) for name in submit.ATTEMPTS)
 assert not os.path.lexists(u/'owned_allocation_inputs.json') and not os.path.lexists(u/'configured_isolation_pre_release_inputs.json')
roster=U/'publication_roster.json';r=read(roster);assert ref(roster)['sha256']=='302f271486c7fca28815659153c63bed441e1b9566b660b8194176161d16ea9f'
value={'schema_version':1,'kind':'independent_common_a4500_confirmation_literal_held_submission_utility_review','approved':True,'passed':True,'at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'cpu_job_id':'6783003','cpu_step_id':os.environ['SLURM_STEP_ID'],'source_freeze':ref(freeze),'source_before':source_before,'source_after':source_after,'packet_freeze':ref(pf),'packet_review':inputs['independent_packet_review'],'frozen_packet_before':before,'frozen_packet_after':after,'independent_tests':22,'repeated_author_controls':17,'extra_independent_controls':5,'test_output':log.getvalue(),'actual_default_checks':default_results,'actual_earlier_publication_refused':{'returncode':old.returncode,'stdout':old_out,'stderr':old_err},'mock_exact_published_object_count':159,'mock_changed_published_object_refused':True,'unchanged_submission_mechanism_AST_after_literal_correctness_to_confirmation_names':True,'publication_roster':ref(roster),'publication_requires_roster_itself_and_this_genuine_review_as_additional_paths':True,'submission_approved_only_after_root_publishes_all_exact_bytes_and_passes_actual_publication_defaults':True,'release_approved':False,'new_jobs_submitted':0,'actual_job_queries':0,'actual_gpu_operations':0,'numerical_work':False,'scientific_credit':0,'case_credit':0,'idea_credit':0,'admission_granted':False,'worker_admitted':False,'allowed_metadata_children':allowed,'blocked_parent_operations':blocked}
final=save('utility_review.json',value);print(json.dumps({'passed':True,'approved':True,'review':final,'tests':22,'published_objects_checked':159,'frozen_packet_files':len(after)}))
