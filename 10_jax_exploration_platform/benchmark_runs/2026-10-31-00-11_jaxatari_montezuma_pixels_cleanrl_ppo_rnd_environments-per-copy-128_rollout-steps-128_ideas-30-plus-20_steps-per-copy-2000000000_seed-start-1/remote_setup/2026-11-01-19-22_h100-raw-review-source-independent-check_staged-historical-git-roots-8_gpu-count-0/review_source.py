NEW_PACKET='/p/rlprojects/RND_worktrees/merge-atari-cleanrl/10_jax_exploration_platform/benchmark_runs/2026-10-31-00-11_jaxatari_montezuma_pixels_cleanrl_ppo_rnd_environments-per-copy-128_rollout-steps-128_ideas-30-plus-20_steps-per-copy-2000000000_seed-start-1/remote_setup/2026-11-01-19-13_fresh-original-h100-qualification-review_staged-historical-git-roots-8_source-6066d9ea_gpu-count-0'
OLD_PACKET='/p/rlprojects/RND_worktrees/merge-atari-cleanrl/10_jax_exploration_platform/benchmark_runs/2026-10-31-00-11_jaxatari_montezuma_pixels_cleanrl_ppo_rnd_environments-per-copy-128_rollout-steps-128_ideas-30-plus-20_steps-per-copy-2000000000_seed-start-1/remote_setup/2026-11-01-18-25_fresh-original-h100-qualification-raw-review-preparation_source-6066d9ea_gpu-count-0'
OUTPUT='/p/rlprojects/RND_worktrees/merge-atari-cleanrl/10_jax_exploration_platform/benchmark_runs/2026-10-31-00-11_jaxatari_montezuma_pixels_cleanrl_ppo_rnd_environments-per-copy-128_rollout-steps-128_ideas-30-plus-20_steps-per-copy-2000000000_seed-start-1/remote_setup/2026-11-01-19-22_h100-raw-review-source-independent-check_staged-historical-git-roots-8_gpu-count-0'
"""Check the exact historical Git namespace correction without terminal scientific review."""
import ast,builtins,contextlib,ctypes,datetime,hashlib,importlib.util,io,json,os,pathlib,stat,subprocess,sys,unittest
from unittest import mock
P=pathlib.Path(NEW_PACKET);OLD=pathlib.Path(OLD_PACKET);O=pathlib.Path(OUTPUT)
assert os.environ['SLURM_JOB_ID']=='6783003' and os.environ['SLURM_STEP_ID'].isdigit()
def sha(raw):return hashlib.sha256(raw).hexdigest()
def ref(p):return {'path':str(p),'sha256':sha(p.read_bytes())}
def read(p):return json.loads(p.read_bytes())
def save(name,value):
 p=O/name
 with p.open('x') as f:json.dump(value,f,sort_keys=True,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
 return ref(p)
F=P/'source_freeze.json';assert ref(F)['sha256']=='ef2d27f59ecb6231a8b8e8fc379b45dceca846c192ac82814259e71cb03514f5'
frozen=read(F);assert len(frozen['files'])==12
before={p:sha(pathlib.Path(p).read_bytes()) for p in frozen['files']};assert before==frozen['files']
def snapshot(root):
 out={}
 for p in root.rglob('*'):
  if p.is_file():
   s=p.lstat();assert stat.S_ISREG(s.st_mode) and p.resolve(strict=True)==p
   out[str(p)]={'sha256':sha(p.read_bytes()),'mode':stat.S_IMODE(s.st_mode),'uid':s.st_uid,'device':s.st_dev,'inode':s.st_ino}
 return out
old_before=snapshot(OLD)
assert not (OLD/'terminal_full_review/independent_actual_review.json').exists() and not (P/'terminal_full_review').exists()
NUM={'jax','jaxlib','numpy','torch','flax','optax','jaxtari','cupy','scipy','exploration_platform','pynvml'}
original_import=builtins.__import__;blocked=[]
def guarded(name,*args,**kwargs):
 if name.split('.')[0] in NUM:blocked.append(name);raise RuntimeError('numerical import forbidden')
 return original_import(name,*args,**kwargs)
builtins.__import__=guarded
forbidden_names={'direct_qualification_result.json','direct_process_metadata.json','direct_device_binding.json','direct_terminal_process.json'}
def audit(event,args):
 if event in {'socket.connect','socket.bind','os.system'} or event=='ctypes.dlopen' and args[0] is not None:raise RuntimeError('live operation forbidden')
 if event=='open' and isinstance(args[0],(str,bytes)):
  path=os.fsdecode(args[0])
  if path.startswith(('/dev/nvidia','/dev/dri')):raise RuntimeError('device access forbidden')
  if pathlib.Path(path).name in forbidden_names:raise RuntimeError('actual terminal evidence read forbidden in source review')
sys.addaudithook(audit)
spec=importlib.util.spec_from_file_location('independent_historical_review_tests',P/'test_fresh_direct_raw_review.py');tests=importlib.util.module_from_spec(spec);spec.loader.exec_module(tests);review=tests.review
assert ref(P/'review_direct_raw_then_consumer.py')['sha256']=='a57885c4976d960df1fcc984678e7fd50bf1a2075a9f8b49361035f334d34a42'
assert ref(P/'raw_direct_qualification_checks.py')['sha256']=='96c8f15609dba406300f57b8dadaee47789a56acf72c1e4fa96b8ef35156a32d'
assert (P/'raw_direct_qualification_checks.py').read_bytes()==(OLD/'raw_direct_qualification_checks.py').read_bytes()
# Every old function except the one metadata-source prerequisite and command guard remains identical.
def functions(path):return {n.name:n for n in ast.parse(path.read_bytes()).body if isinstance(n,ast.FunctionDef)}
a,b=functions(OLD/'review_direct_raw_then_consumer.py'),functions(P/'review_direct_raw_then_consumer.py')
assert set(b)-set(a)=={'canonical_original','pinned_history_git_roots','allowed_git_command'}
for name in set(a)-{'verify_source_contract','main'}:assert ast.dump(a[name],include_attributes=False)==ast.dump(b[name],include_attributes=False),name
assert len(b['verify_source_contract'].body)==len(a['verify_source_contract'].body)+1
# Independently compare the exact raw 50 statements with the immutable staged 52-statement consumer.
contract=read(P/'raw_review_source_contract.json')
source_path=pathlib.Path(contract['staged_source']['path']);assert ref(source_path)['sha256']=='beead126531ee254a7999adae2e2e0e7292f18154a2e697e8a22ce1e93a33294'
original=functions(source_path)['verify_qualification_inputs'];raw=functions(P/'raw_direct_qualification_checks.py')['verify_unreviewed_actual_raw_qualification']
assert len(original.body)==52 and len(raw.body)==50
assert [ast.dump(n,include_attributes=False) for i,n in enumerate(original.body) if i not in (4,47)]==[ast.dump(n,include_attributes=False) for n in raw.body]
plan=review.doc(review.UNIT/'measurement_plan.json');roots=review.pinned_history_git_roots(plan)
assert len(roots)==8
inputs=review.doc(plan['direct_binding']['frozen_inputs']['path']);manifest_ref=inputs['evidence_manifest'];manifest=review.doc(manifest_ref['path'])
assert manifest_ref['sha256'].startswith('cdc371c6')
# Source-only Git queries are restricted to the eight exact new roots.
real_popen=subprocess.Popen;git_calls=[]
class ReadOnlyGit(real_popen):
 def __init__(self,command,*args,**kwargs):
  assert review.allowed_git_command(command,roots) and not kwargs.get('shell',False),repr(command)
  git_calls.append(command);super().__init__(command,*args,**kwargs)
subprocess.Popen=ReadOnlyGit
actual_roots=[]
staging=pathlib.Path(manifest_ref['path']).parent
for row in manifest['git_roots']:
 root=str(staging/row['relative_path'])
 head=subprocess.check_output(['git','-C',root,'rev-parse','HEAD']).decode().strip()
 top=subprocess.check_output(['git','-C',root,'rev-parse','--show-toplevel']).decode().strip()
 assert head==row['commit'] and top==root
 actual_roots.append({**row,'staged_root':root,'actual_HEAD':head,'actual_top_level':top})
class ExtraControls(unittest.TestCase):
 def fixture(self):
  obj=tests.MappedHistoricalGitRootTests();obj.setUp();self.addCleanup(obj.doCleanups);return obj
 def test_changed_inputs_hash_refused(self):
  obj=self.fixture();p=obj.history/'inputs.json';p.write_bytes(p.read_bytes()+b' ')
  with self.assertRaises(ValueError):obj.roots()
 def test_alternate_manifest_path_refused_even_with_valid_new_hash(self):
  obj=self.fixture();alternate=obj.history/'another_manifest.json';alternate.write_bytes((obj.history/'evidence_manifest.json').read_bytes())
  obj.write(obj.history/'inputs.json',{'evidence_manifest':review.ref(alternate)})
  obj.plan['direct_binding']['frozen_inputs']=review.ref(obj.history/'inputs.json')
  with self.assertRaises(ValueError):obj.roots()
 def test_extra_mapping_fields_and_boolean_schema_refused(self):
  for change in [lambda obj:obj.manifest['git_roots'][0].update(extra='unapproved'),lambda obj:obj.manifest.update(schema_version=True)]:
   obj=self.fixture();change(obj);obj.refresh()
   with self.assertRaises(ValueError):obj.roots()
 def test_symlink_parent_cannot_select_different_directory(self):
  obj=self.fixture();parent=obj.history/'git_sources';actual=obj.history/'actual_sources';parent.rename(actual);parent.symlink_to(actual,target_is_directory=True)
  with self.assertRaises(ValueError):obj.roots()
 def test_original_namespace_aliases_refused(self):
  for path in ['//synthetic/original','/synthetic//original','/synthetic/./original','/synthetic/original/','/synthetic/../original','relative/original','/synthetic/\0original']:
   with self.subTest(path=path),self.assertRaises(ValueError):review.canonical_original(path)
 def test_duplicate_declared_originals_and_bad_commits_refused(self):
  for change in [lambda obj:obj.preparation['git_sources'][1].update(original_root=obj.preparation['git_sources'][0]['original_root']),lambda obj:obj.preparation['git_sources'][0].update(commit=True),lambda obj:obj.preparation['git_sources'][0].update(commit='b'*40)]:
   obj=self.fixture();change(obj);obj.write(obj.packet/'history_preparation_inputs.json',obj.preparation)
   with self.assertRaises(ValueError):obj.roots()
 def test_command_roots_require_exact_string_identity(self):
  one=next(iter(roots))
  for alias in [one+'/',one+'/.',one+'/..',pathlib.Path(one),'/unlisted/root']:
   self.assertFalse(review.allowed_git_command(['git','-C',alias,'status'],roots))
  self.assertFalse(review.allowed_git_command(['git','-C',one,'checkout','other'],roots))
 def test_nested_readonly_guard_refuses_shell_before_any_parent_call(self):
  klass=next(n for n in ast.walk(b['main']) if isinstance(n,ast.ClassDef) and n.name=='ReadOnlyGit')
  calls=[]
  class DummyBase:
   def __init__(self,*args,**kwargs):calls.append((args,kwargs))
  namespace={'real_popen':DummyBase,'allowed_roots':roots,'allowed_git_command':review.allowed_git_command,'require':review.require,'__name__':'independent_guard_fixture'}
  exec(compile(ast.fix_missing_locations(ast.Module(body=[klass],type_ignores=[])),'<fixed-read-only-git-guard>','exec'),namespace)
  command=['git','-C',next(iter(roots)),'status']
  with self.assertRaises(ValueError):namespace['ReadOnlyGit'](command,shell=True)
  self.assertFalse(calls)
  namespace['ReadOnlyGit'](command);self.assertEqual(len(calls),1)
log=io.StringIO();suite=unittest.TestSuite([unittest.defaultTestLoader.loadTestsFromModule(tests),unittest.defaultTestLoader.loadTestsFromTestCase(ExtraControls)])
with contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):result=unittest.TextTestRunner(stream=log,verbosity=2).run(suite)
assert result.wasSuccessful() and result.testsRun==22,log.getvalue()
default=io.StringIO()
with contextlib.redirect_stdout(default):review.main([])
default_record=json.loads(default.getvalue());assert default_record=={'source_contract_passed':True,'raw_statements_retained':50,'scheduler_queries':0,'actual_result_reads':0,'independent_review_written':False,'formal_consumer_called':False}
after={p:sha(pathlib.Path(p).read_bytes()) for p in frozen['files']};old_after=snapshot(OLD)
assert before==after==frozen['files'] and old_before==old_after and not blocked
assert not (P/'terminal_full_review').exists() and not (OLD/'terminal_full_review/independent_actual_review.json').exists()
value={'schema_version':1,'kind':'independent_h100_raw_review_exact_staged_historical_git_namespace_source_review','approved':True,'passed':True,'at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'cpu_job_id':'6783003','cpu_step_id':os.environ['SLURM_STEP_ID'],'source_freeze':ref(F),'source_before':before,'source_after':after,'previous_failed_packet':str(OLD),'previous_failed_packet_before':old_before,'previous_failed_packet_after':old_after,'previous_qualification_approval_exists':False,'source_contract':ref(P/'raw_review_source_contract.json'),'raw_reader_sha256':ref(P/'raw_direct_qualification_checks.py')['sha256'],'staged_full_consumer_sha256':ref(source_path)['sha256'],'all50_raw_statements_unchanged':True,'only_prior_review_load_and_approval_predicate_omitted':True,'existing_startup_capture_terminal_gates_retained':True,'default_source_only_result':default_record,'independent_test_methods':22,'author_test_methods_repeated':14,'extra_independent_test_methods':8,'test_output':log.getvalue(),'frozen_portable_manifest':manifest_ref,'exact_eight_staged_git_roots':actual_roots,'actual_read_only_git_queries':git_calls,'change_scope':'Only add the exact eight already pinned staged historical Git roots to the existing read-only command namespace. No scientific validator or tolerance changes.','actual_terminal_review_attempted':False,'actual_numerical_result_read':False,'publish_exact_reviewed_source_before_one_terminal_audit':True,'actual_GPU_qualification_verified_by_this_source_review':False,'numerical_work':False,'actual_gpu_operations':0,'actual_job_queries':0,'scientific_credit':0,'case_credit':0,'idea_credit':0,'admission_granted':False,'worker_admitted':False}
proof=save('review.json',value);print(json.dumps({'passed':True,'approved':True,'review':proof,'methods':22,'new_exact_git_roots':len(roots),'unchanged_raw_statements':50,'preserved_old_files':len(old_before)}))
