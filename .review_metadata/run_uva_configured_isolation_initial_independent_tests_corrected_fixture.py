"""Run only guarded metadata tests inside the already owned CPU monitor allocation."""
import copy,ctypes,datetime,hashlib,importlib.abc,json,os,pathlib,socket,subprocess,sys,unittest
from types import SimpleNamespace
from unittest.mock import patch
W=pathlib.Path(__file__).resolve().parents[1]
S=W/'10_jax_exploration_platform/scripts/atari_pixel_campaign'
require=lambda condition,message: condition or (_ for _ in ()).throw(RuntimeError(message))
require(os.environ.get('SLURM_JOB_ID')=='6781231','only owned CPU 6781231 may run these tests')
step=os.environ.get('SLURM_STEP_ID','');require(step.isdecimal(),'an actual numeric step is required')
freeze=W/'.review_metadata/uva_configured_isolation_initial_source_freeze.json'
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
core=fixtures.core
findings=[]
fixtures.ConfiguredIsolationTests.setUpClass()
try:
 case=fixtures.ConfiguredIsolationTests('runTest');case.setUp()
 try:
  changed=copy.deepcopy(case.metadata)
  changed['step']['stdout']=changed['step']['stdout'].replace('mem=64G','mem=128G')
  core.validate_metadata(case.plan,changed)
  findings.append({'name':'step_memory_above_job_limit_is_accepted','reproduced':True})
 finally:case.doCleanups()
 case=fixtures.ConfiguredIsolationTests('runTest');case.setUp()
 try:
  altered=copy.deepcopy(case.inputs)
  live=pathlib.Path(altered['ledger']['source_path'])
  altered['ledger']['snapshot']=core.receipt(live)
  case.write(pathlib.Path(case.plan['configured_isolation']['owned_inputs_path']),altered)
  core.owned_inputs(case.plan)
  live.write_text(live.read_text()+'99999999\n')
  try:core.owned_inputs(case.plan)
  except ValueError:later_rejected=True
  else:later_rejected=False
  findings.append({'name':'live_ledger_is_accepted_as_immutable_snapshot_then_legitimate_append_invalidates_it','reproduced':later_rejected})
 finally:case.doCleanups()
 case=fixtures.ConfiguredIsolationTests('runTest');case.setUp()
 try:
  foreign=copy.deepcopy(case.process);foreign['uid']=999
  helper=SimpleNamespace(process_identity=lambda:foreign,v2_paths=lambda *args:None,
      capture_v2_ownership=lambda *args:copy.deepcopy(case.metadata['cgroup']),
      UVA_CGROUP_V2_OWNERSHIP_PROFILE=copy.deepcopy(case.metadata['cgroup']['ownership_profile']))
  calls=[];secret='synthetic_private_configuration_value_418'
  configuration=copy.deepcopy(case.configuration)
  first=next(iter(configuration['files']))
  configuration['files'][first]['raw']='AuthInfo='+secret+'\n'
  configuration['files'][first]['sha256']=core.digest(configuration['files'][first]['raw'].encode())
  def query(command,**kwargs):
   calls.append(command)
   key='job' if command[2]=='job' else 'step' if command[2]=='step' else 'running_configuration'
   return subprocess.CompletedProcess(command,0,case.metadata[key]['stdout'],'')
  account=SimpleNamespace(pw_name='slurm',pw_uid=64030,pw_gid=64030)
  with patch.object(core,'source_identity',return_value=case.plan['helpers']),patch.object(core,'owned_inputs',return_value=case.inputs),patch.object(core,'adjacent',return_value=helper),patch.object(core.subprocess,'run',side_effect=query),patch.object(core.pwd,'getpwnam',return_value=account),patch.object(core.os.path,'lexists',return_value=False),patch.object(core,'config_file',side_effect=lambda path:configuration['files'][path]),patch.dict(os.environ,case.metadata['environment'],clear=True):
   try:core.collect_metadata(case.plan,case.unit)
   except ValueError:rejected=True
   else:rejected=False
  capture=(case.unit/'configured_isolation_metadata.json').read_text()
  findings.append({'name':'foreign_current_UID_rejected_only_after_three_scheduler_queries','reproduced':rejected and len(calls)==3,'synthetic_query_count':len(calls)})
  findings.append({'name':'failed_metadata_saves_changed_unfiltered_private_configuration_text','reproduced':secret in capture})
 finally:case.doCleanups()
finally:fixtures.ConfiguredIsolationTests.tearDownClass()
require(all(item['reproduced'] for item in findings),'one expected initial-source finding did not reproduce')
after=check();require(after==before,'source changed during tests')
require(not blocked,'numerical imports were attempted')
raw=pathlib.Path('/proc/self/stat').read_text()
record={'schema_version':1,'kind':'uva_configured_isolation_initial_independent_findings','at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
 'job_id':'6781231','step_id':step,'pid':os.getpid(),'birth':raw[raw.rfind(')')+2:].split()[19],
 'cgroup':pathlib.Path('/proc/self/cgroup').read_text(),'cpu_affinity':sorted(os.sched_getaffinity(0)),
 'source_freeze_sha256':hashlib.sha256(freeze.read_bytes()).hexdigest(),'source_before':before,'source_after':after,
 'findings':findings,'source_approved':False,'additional_source_findings':['read descriptor opened before pinned regular-file proof','native binding lacks current-context API checks','native/binding failure can leave no partial native receipt','scheduler capture is unbounded','plan digest uses canonical encoding rather than exact frozen bytes'],'tests_run':result.testsRun,'errors':len(result.errors),'failures':len(result.failures),'passed':result.wasSuccessful(),
 'numerical_imports_attempted':blocked,'all_process_commands_restricted_to_local_temporary_git':True,'shared_library_loading_prohibited':True,
 'scientific_credit':0,'gpu_qualification':False}
p=W/'.review_metadata'/('uva_configured_isolation_initial_independent_6781231_'+step+'.json')
with p.open('x') as f:json.dump(record,f,sort_keys=True,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
print(json.dumps({'receipt':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'passed':result.wasSuccessful(),'tests':result.testsRun,'reproduced_findings':len(findings)}))
sys.exit(0 if result.wasSuccessful() else 1)
