"""Run only guarded metadata tests inside the already owned CPU monitor allocation."""
import copy,ctypes,datetime,hashlib,importlib.abc,json,os,pathlib,subprocess,sys,unittest
from unittest.mock import patch
W=pathlib.Path(__file__).resolve().parents[1]
S=W/'10_jax_exploration_platform/scripts/atari_pixel_campaign'
require=lambda condition,message: condition or (_ for _ in ()).throw(RuntimeError(message))
require(os.environ.get('SLURM_JOB_ID')=='6781231','only owned CPU 6781231 may run these tests')
step=os.environ.get('SLURM_STEP_ID','');require(step.isdecimal(),'an actual numeric step is required')
freeze=W/'.review_metadata/uva_configured_isolation_corrected_source_freeze.json'
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
  try:core.validate_metadata(case.plan,reloaded)
  except ValueError as error:rejected='configuration scope' in str(error)
  else:rejected=False
  findings.append({'name':'published_sorted_metadata_roundtrip_is_rejected_by_file_map_order','reproduced':rejected})
  changed=copy.deepcopy(case.configuration)
  first=next(iter(changed['files']))
  changed['files'][first]['settings']=[{'line':1,'name':'ConstrainDevices','value':'no'}]
  core.validate_configuration(changed)
  findings.append({'name':'changed_allowed_excerpts_pass_with_unchanged_full_file_sha','reproduced':True})
  changed=copy.deepcopy(case.metadata)
  changed['job'].update(stdout_full_sha256='0'*64,stdout_captured_bytes=1)
  core.validate_metadata(case.plan,changed)
  findings.append({'name':'own_query_raw_capture_hash_and_count_are_not_verified','reproduced':True})
 finally:case.doCleanups()
finally:fixtures.ConfiguredIsolationTests.tearDownClass()
child=SyntheticQuery(stdout=b'JobId=123\xff')
with patch.object(core.subprocess,'Popen',return_value=child):
 captured=core.bounded_query(['scontrol','show','job','-d','123'])
findings.append({'name':'invalid_utf8_raw_query_is_marked_completed_with_replacement_text','reproduced':captured['capture_status']=='completed' and '\ufffd' in captured['stdout']})
require(all(x['reproduced'] for x in findings),'one expected corrected-source finding did not reproduce')
after=check();require(after==before,'source changed during tests')
require(not blocked,'numerical imports were attempted')
raw=pathlib.Path('/proc/self/stat').read_text()
record={'schema_version':1,'kind':'uva_configured_isolation_second_independent_findings','at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
 'job_id':'6781231','step_id':step,'pid':os.getpid(),'birth':raw[raw.rfind(')')+2:].split()[19],
 'cgroup':pathlib.Path('/proc/self/cgroup').read_text(),'cpu_affinity':sorted(os.sched_getaffinity(0)),
 'source_freeze_sha256':hashlib.sha256(freeze.read_bytes()).hexdigest(),'source_before':before,'source_after':after,
 'findings':findings,'source_approved':False,'tests_run':result.testsRun,'errors':len(result.errors),'failures':len(result.failures),'passed':result.wasSuccessful(),
 'numerical_imports_attempted':blocked,'all_process_commands_restricted_to_local_temporary_git':True,'shared_library_loading_prohibited':True,
 'scientific_credit':0,'gpu_qualification':False}
p=W/'.review_metadata'/('uva_configured_isolation_second_independent_6781231_'+step+'.json')
with p.open('x') as f:json.dump(record,f,sort_keys=True,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
print(json.dumps({'receipt':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'passed':result.wasSuccessful(),'tests':result.testsRun,'reproduced_findings':len(findings)}))
sys.exit(0 if result.wasSuccessful() else 1)
