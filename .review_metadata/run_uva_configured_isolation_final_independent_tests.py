"""Run only guarded metadata tests inside the already owned CPU monitor allocation."""
import copy,ctypes,datetime,hashlib,importlib.abc,json,os,pathlib,subprocess,sys,unittest
from unittest.mock import patch
W=pathlib.Path(__file__).resolve().parents[1]
S=W/'10_jax_exploration_platform/scripts/atari_pixel_campaign'
require=lambda condition,message: condition or (_ for _ in ()).throw(RuntimeError(message))
require(os.environ.get('SLURM_JOB_ID')=='6781231','only owned CPU 6781231 may run these tests')
step=os.environ.get('SLURM_STEP_ID','');require(step.isdecimal(),'an actual numeric step is required')
freeze=W/'.review_metadata/uva_configured_isolation_final_source_freeze.json'
expected=json.loads(freeze.read_text())['files']
check=lambda:{path:hashlib.sha256((W/path).read_bytes()).hexdigest() for path in expected}
before=check();require(before==expected,'source differs before metadata tests')
blocked=[]
class BlockNumericalImports(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname.split('.')[0] in {'jax','jaxlib','numpy','torch','flax','optax','cupy','scipy'}:
   blocked.append(fullname);raise RuntimeError('numerical import prohibited: '+fullname)
sys.meta_path.insert(0,BlockNumericalImports())
real_popen=subprocess.Popen
class LocalGitOnly(real_popen):
 def __init__(self,args,*a,**kw):
  cmd=list(args) if isinstance(args,(tuple,list)) else []
  if cmd == globals().get('allowed_cold_probe'):
   super().__init__(args,*a,**kw);return
  require(len(cmd)>3 and cmd[0]=='git' and cmd[1]=='-C' and pathlib.Path(cmd[2]).is_absolute()
    and str(pathlib.Path(cmd[2])).startswith('/tmp/'),'only local temporary Git metadata commands are permitted')
  require(not any(x in {'clone','fetch','push','pull','remote'} for x in cmd),'network Git is prohibited')
  super().__init__(args,*a,**kw)
subprocess.Popen=LocalGitOnly
ctypes.CDLL=lambda *a,**kw:(_ for _ in ()).throw(RuntimeError('shared-library loads are prohibited'))
sys.path[:0]=[str(S),str(S/'tests')]
suite=unittest.defaultTestLoader.loadTestsFromName('test_uva_slurm_configured_isolation')
result=unittest.TextTestRunner(verbosity=2).run(suite)

import test_uva_slurm_configured_isolation as fixtures
from test_passive_installed_slurm_observation import SyntheticQuery
core=fixtures.core
findings=[]
fixtures.ConfiguredIsolationTests.setUpClass()
try:
 case=fixtures.ConfiguredIsolationTests('runTest');case.setUp()
 try:
  reloaded=json.loads(core.encoded(case.metadata))
  core.validate_metadata(case.plan,reloaded)
  findings.append({'name':'published_sorted_metadata_roundtrip_passes','verified':True})
  changed=copy.deepcopy(case.configuration)
  first=next(iter(changed['files']))
  changed['files'][first]['settings']=[{'line':1,'name':'ConstrainDevices','value':'no'}]
  try:core.validate_configuration(changed)
  except ValueError:rejected=True
  else:rejected=False
  findings.append({'name':'changed_allowed_excerpts_reject_under_unchanged_full_file_sha','verified':rejected})
  changed=copy.deepcopy(case.metadata)
  changed['job'].update(stdout_full_sha256='0'*64,stdout_captured_bytes=1)
  try:core.validate_metadata(case.plan,changed)
  except ValueError:rejected=True
  else:rejected=False
  findings.append({'name':'own_query_raw_capture_hash_count_mutation_rejects','verified':rejected})
 finally:case.doCleanups()
finally:fixtures.ConfiguredIsolationTests.tearDownClass()
child=SyntheticQuery(stdout=b'JobId=123\xff')
with patch.object(core.subprocess,'Popen',return_value=child):
 captured=core.bounded_query(['scontrol','show','job','-d','123'])
findings.append({'name':'invalid_utf8_raw_query_is_explicit_failure','verified':captured['capture_status']=='invalid_utf8'})
require(all(x['verified'] for x in findings),'one final regression did not pass')
# Independently derive the five excerpt constants from the actual immutable passive evidence.
actual_path=pathlib.Path('/p/rlprojects/RND_worktrees/merge-atari-cleanrl/10_jax_exploration_platform/benchmark_runs/2026-10-31-00-11_jaxatari_montezuma_pixels_cleanrl_ppo_rnd_environments-per-copy-128_rollout-steps-128_ideas-30-plus-20_steps-per-copy-2000000000_seed-start-1/data/2026-11-01-10-00_passive-installed-slurm-observation_jaguar03_gpu-count-0_processor-count-1_host-memory-gib-2_time-limit-minutes-5_qos-csresnolim/passive_observation.json')
actual_raw=actual_path.read_bytes();require(hashlib.sha256(actual_raw).hexdigest()=='b52907bbe47942918eb147e1e403c89c2114f69b1aa3970423cda5f81fa61f16','actual passive basis changed')
actual=json.loads(actual_raw)
rows=[row for tree in actual['configuration_trees'] for row in tree['files']]+actual['gres_and_cgroup_files']
by_path={row.get('path',row['requested_path']):row for row in rows}
actual_excerpt_proof={}
for path,full_hash in core.CONFIG_FILES.items():
 row=by_path[path];require(row['sha256']==full_hash,'actual full-file baseline differs')
 excerpt=core.digest(core.encoded(row['assignments']))
 require(excerpt==core.CONFIG_EXCERPT_HASHES[path],'actual excerpt constant differs')
 actual_excerpt_proof[path]={'full_sha256':full_hash,'excerpt_sha256':excerpt,'rows':len(row['assignments'])}
# Cold staged import: no ambient project path and no numerical/library/device actions.
import tempfile,shutil
cold_results=[]
with tempfile.TemporaryDirectory() as temporary:
 stage=pathlib.Path(temporary)/'stage';stage.mkdir()
 for name in core.HELPERS:shutil.copyfile(S/name,stage/name)
 probe=pathlib.Path(temporary)/'cold_probe.py'
 probe.write_text("import ctypes,hashlib,importlib.abc,json,os,pathlib,sys\n"
  "stage=pathlib.Path(sys.argv[1]);blocked=[]\n"
  "class Guard(importlib.abc.MetaPathFinder):\n def find_spec(self,name,path=None,target=None):\n  if name.split('.')[0] in {'jax','jaxlib','numpy','torch','flax','optax','cupy','scipy'}: raise RuntimeError('numerical import prohibited')\n"
  "def audit(event,args):\n if event in {'subprocess.Popen','socket.connect','socket.getaddrinfo','ctypes.dlopen'}:raise RuntimeError('live operation prohibited')\n if event=='open' and args and isinstance(args[0],str) and args[0].startswith(('/dev/nvidia','/dev/dri')):raise RuntimeError('device read prohibited')\n"
  "sys.addaudithook(audit);sys.meta_path.insert(0,Guard());ctypes.CDLL=lambda *a,**k:(_ for _ in ()).throw(RuntimeError('library prohibited'))\n"
  "sys.path.insert(0,str(stage));import uva_slurm_configured_isolation as core\n"
  "loaded={}\nfor name in core.HELPERS:\n module=core.adjacent(pathlib.Path(name).stem);path=pathlib.Path(module.__file__).resolve();assert path.parent==stage;loaded[name]=hashlib.sha256(path.read_bytes()).hexdigest()\n"
  "import uva_active_configuration_observation as owner;assert pathlib.Path(owner.__file__).resolve().parent==stage\n"
  "print(json.dumps({'passed':True,'loaded':loaded,'owner':str(pathlib.Path(owner.__file__).resolve())}))\n")
 allowed_cold_probe=[sys.executable,'-I','-B',str(probe),str(stage)]
 cold=subprocess.run(allowed_cold_probe,capture_output=True,text=True,timeout=20)
 require(cold.returncode==0,'complete cold staged import failed: '+cold.stderr)
 cold_results.append({'kind':'complete_staged_closure','returncode':cold.returncode,'result':json.loads(cold.stdout)})
 (stage/'uva_active_configuration_observation.py').unlink()
 missing=subprocess.run(allowed_cold_probe,capture_output=True,text=True,timeout=20)
 require(missing.returncode!=0,'missing staged dependency used an ambient fallback')
 cold_results.append({'kind':'missing_staged_dependency_refused','returncode':missing.returncode,'stderr_sha256':hashlib.sha256(missing.stderr.encode()).hexdigest()})
 cold_probe_sha256=hashlib.sha256(probe.read_bytes()).hexdigest()
allowed_cold_probe=None
after=check();require(after==before,'source changed during tests')
require(not blocked,'numerical imports were attempted')
raw=pathlib.Path('/proc/self/stat').read_text()
record={'schema_version':1,'kind':'uva_configured_isolation_final_independent_metadata_review','at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
 'job_id':'6781231','step_id':step,'pid':os.getpid(),'birth':raw[raw.rfind(')')+2:].split()[19],
 'cgroup':pathlib.Path('/proc/self/cgroup').read_text(),'cpu_affinity':sorted(os.sched_getaffinity(0)),
 'source_freeze_sha256':hashlib.sha256(freeze.read_bytes()).hexdigest(),'source_before':before,'source_after':after,
 'verified_regressions':findings,'actual_excerpt_proof':actual_excerpt_proof,'cold_staged_imports':cold_results,'cold_probe_sha256':cold_probe_sha256,'source_approved':True,'operational_admission':False,'limits':['metadata/source approval only; no actual GPU qualification','combined producer parent/child and default/helper-union checks remain required','initial f9/c3 physical source-byte archive was unavailable and is not claimed recovered'],'tests_run':result.testsRun,'errors':len(result.errors),'failures':len(result.failures),'passed':result.wasSuccessful(),
 'numerical_imports_attempted':blocked,'all_process_commands_restricted_to_local_temporary_git':True,'shared_library_loading_prohibited':True,
 'scientific_credit':0,'gpu_qualification':False}
p=W/'.review_metadata'/('uva_configured_isolation_final_independent_6781231_'+step+'.json')
with p.open('x') as f:json.dump(record,f,sort_keys=True,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
print(json.dumps({'receipt':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'passed':result.wasSuccessful(),'tests':result.testsRun,'verified_regressions':len(findings),'cold_staged_checks':len(cold_results)}))
sys.exit(0 if result.wasSuccessful() else 1)
