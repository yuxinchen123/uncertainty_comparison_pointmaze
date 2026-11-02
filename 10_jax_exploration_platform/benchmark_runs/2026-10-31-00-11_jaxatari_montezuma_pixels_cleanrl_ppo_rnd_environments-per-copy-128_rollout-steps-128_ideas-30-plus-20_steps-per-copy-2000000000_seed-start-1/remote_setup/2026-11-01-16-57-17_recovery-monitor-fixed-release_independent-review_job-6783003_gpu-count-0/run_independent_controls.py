import contextlib,hashlib,importlib.util,io,json,os,socket,subprocess,sys,tempfile,unittest
from datetime import datetime,timezone
from pathlib import Path
from unittest.mock import patch
P=Path('/p/rlprojects/RND_worktrees/merge-atari-cleanrl/10_jax_exploration_platform/benchmark_runs/2026-10-31-00-11_jaxatari_montezuma_pixels_cleanrl_ppo_rnd_environments-per-copy-128_rollout-steps-128_ideas-30-plus-20_steps-per-copy-2000000000_seed-start-1/remote_setup/2026-11-01-16-33_pixel-study-monitor-recovery_source-commit-79a01e851ff92ddba7d0e12eaf8164cab2028878_processor-count-1_host-memory-gib-2_time-limit-days-7_poll-interval-seconds-1200_partition-nolim_account-sl5nw-base_gpu-count-0')
OUT=Path(__file__).absolute().parent
NUM={'jax','jaxlib','numpy','torch','flax','optax','cupy','scipy'}
blocked=[]
allowed=[]
def guard(event,args):
 if event=='import' and args[0].split('.')[0] in NUM:blocked.append(event);raise RuntimeError('numerical import forbidden')
 if event in {'socket.connect','socket.bind','ctypes.dlopen'}:blocked.append(event);raise RuntimeError('network or shared library operation forbidden')
 if event=='open' and isinstance(args[0],(str,bytes)) and os.fsdecode(args[0]).startswith('/dev/nvidia'):blocked.append(event);raise RuntimeError('device access forbidden')
 if event=='subprocess.Popen':
  command=args[1]
  if not isinstance(command,list) or command[:3]!=['git','-C','/p/rlprojects/RND_worktrees/merge-atari-cleanrl'] or len(command)<5 or command[3]!='cat-file':blocked.append(event);raise RuntimeError('only pinned local Git reads permitted')
  allowed.append(command)
sys.addaudithook(guard)
assert socket.gethostname().split('.')[0]=='portal05' and not os.environ.get('SLURM_JOB_ID') and os.getuid()==2618919
expected_source='8505d609a53f2f1dcc4b76a97d7fbd571ae6298e0d8f00c44084af8e463b476e'
expected_test='a5384ef5f0db18b91af7b0e254b5039c1ec64f43229f13a242eff5c42b64c28d'
expected_freeze='38b0eef4895f6d688e0837581105f07aad868406573e055d4c69887d1abff955'
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
assert sha(P/'release_monitor_6783003.py')==expected_source
assert sha(P/'test_release_monitor_6783003.py')==expected_test
assert sha(P/'release_6783003_source_freeze.json')==expected_freeze
maps={}
for name in ('static_source_freeze.json','release_6783003_source_freeze.json'):
 value=json.loads((P/name).read_bytes());maps[name]=value['files']
for values in maps.values():
 for name,digest in values.items():
  path=Path(name);assert not path.is_symlink() and path.resolve(strict=True)==path and path.is_file() and sha(path)==digest,name
paths=set().union(*(set(v) for v in maps.values()))|{str(P/'static_source_freeze.json'),str(P/'release_6783003_source_freeze.json')}
before={name:sha(Path(name)) for name in sorted(paths)}
spec=importlib.util.spec_from_file_location('independent_controls',P/'test_release_monitor_6783003.py');tests=importlib.util.module_from_spec(spec);spec.loader.exec_module(tests);m=tests.m
mock_reads=[]
class Extra(unittest.TestCase):
 def test_successful_private_attempt_cannot_repeat_or_query_again(self):
  with tempfile.TemporaryDirectory() as tmp,patch.object(m,'PACKET',Path(tmp)):
   raw=tests.RAW.replace(str(P/'monitor.sbatch'),str(Path(tmp)/'monitor.sbatch')).encode();seen=[]
   def fake(command,**kwargs):
    seen.append(command);return subprocess.CompletedProcess(command,0,raw if len(seen)==1 else b'',b'')
   with patch.object(m.subprocess,'run',side_effect=fake):
    receipt=m.release_once(tests.intent())
    with self.assertRaisesRegex(ValueError,'preserve every prior release attempt'):m.release_once(tests.intent())
   self.assertEqual(len(seen),2);self.assertEqual(receipt['job_id'],'6783003');self.assertEqual(json.loads((Path(tmp)/'release_receipt.json').read_bytes()),receipt)
 def test_exact_mocked_publication_passes_and_changed_object_refuses(self):
  repo=m.RUN.parents[2];publication='a'*40
  def read_blob(command,**kwargs):
   self.assertEqual(command[:4],['git','-C',str(repo),'cat-file']);self.assertEqual(command[4],'blob')
   commit,relative=command[5].split(':',1);self.assertIn(commit,{m.INITIAL_PUBLICATION,publication});path=repo/relative;mock_reads.append(str(path));return path.read_bytes()
  with patch.object(m.subprocess,'check_output',side_effect=read_blob),patch.object(m.subprocess,'run',return_value=subprocess.CompletedProcess([],0,b'',b'')) as process:
   value=m.checked(publication)
   self.assertEqual(value['published_release_commit'],publication);self.assertEqual(process.call_count,1)
   self.assertEqual(process.call_args.args[0],['git','-C',str(repo),'merge-base','--is-ancestor',publication,'refs/remotes/origin/merge-atari-cleanrl'])
  def changed(command,**kwargs):
   raw=read_blob(command,**kwargs)
   return raw+b'changed' if command[5]==publication+':'+str((P/'release_monitor_6783003.py').relative_to(repo)) else raw
  with patch.object(m.subprocess,'check_output',side_effect=changed),patch.object(m.subprocess,'run',return_value=subprocess.CompletedProcess([],0,b'',b'')):
   with self.assertRaisesRegex(ValueError,'release prerequisite was not published'):m.checked(publication)
 def test_actual_default_cli_creates_no_attempt_and_queries_no_scheduler(self):
  with patch.object(m.sys,'argv',['release']),contextlib.redirect_stdout(io.StringIO()) as output:m.main()
  value=json.loads(output.getvalue());self.assertEqual(value['scheduler_queries'],0);self.assertEqual(value['release_calls'],0)
  self.assertFalse((P/'release_intent.json').exists());self.assertFalse((P/'release_receipt.json').exists())
suite=unittest.TestSuite([unittest.defaultTestLoader.loadTestsFromModule(tests),unittest.defaultTestLoader.loadTestsFromTestCase(Extra)])
log=io.StringIO();result=unittest.TextTestRunner(stream=log,verbosity=2).run(suite)
after={name:sha(Path(name)) for name in sorted(paths)}
assert before==after
assert not NUM.intersection(sys.modules)
value={'schema_version':1,'kind':'independent_fixed_cpu_monitor_release_metadata_review','job_id':'6783003','source_commit':m.SOURCE,'checked_at_utc':datetime.now(timezone.utc).isoformat(),'metadata_host':socket.gethostname(),'metadata_allocation':None,'pid':os.getpid(),'uid':os.getuid(),'cgroup':Path('/proc/self/cgroup').read_text(),'proc_stat':Path('/proc/self/stat').read_text(),'tests':result.testsRun,'passed':result.wasSuccessful(),'test_output':log.getvalue(),'source_sha256':expected_source,'test_sha256':expected_test,'freeze_sha256':expected_freeze,'preserved_file_count':len(before),'preserved_files_before':before,'preserved_files_after':after,'real_local_git_reads':allowed,'mocked_publication_reads':mock_reads,'blocked_actions':blocked,'live_scheduler_network_device_library_or_numerical_operations':0,'actual_release_calls':0,'repeat_attempt_refusal_verified':True,'post_start_actual_runtime_source_first_full_report_review_required':True,'scientific_credit':0,'worker_admission':False}
raw=(json.dumps(value,indent=2,allow_nan=False)+'\n').encode();target=OUT/'independent_controls.json'
with target.open('xb') as stream:stream.write(raw);stream.flush();os.fsync(stream.fileno())
fd=os.open(OUT,os.O_RDONLY|os.O_DIRECTORY);os.fsync(fd);os.close(fd)
print(json.dumps({'receipt':str(target),'sha256':hashlib.sha256(raw).hexdigest(),'tests':result.testsRun,'passed':result.wasSuccessful(),'preserved_files':len(before),'blocked_actions':blocked,'output':log.getvalue()}))
raise SystemExit(0 if result.wasSuccessful() else 1)
