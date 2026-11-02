"""Freeze the candidate monitor after actual metadata-only ownership and source checks."""
import hashlib,importlib.abc,importlib.util,json,os,stat,sys
from pathlib import Path
W=Path(__file__).resolve().parent.parent
S=W/'10_jax_exploration_platform/scripts/atari_pixel_campaign'
R=Path("/p/rlprojects/RND_worktrees/merge-atari-cleanrl/10_jax_exploration_platform/benchmark_runs/2026-10-31-00-11_jaxatari_montezuma_pixels_cleanrl_ppo_rnd_environments-per-copy-128_rollout-steps-128_ideas-30-plus-20_steps-per-copy-2000000000_seed-start-1")
assert os.environ.get('SLURM_JOB_ID')=='6783003' and os.environ.get('SLURM_STEP_ID','').isdigit()
print('ACTUAL_OWNED_STEP=6783003.'+os.environ['SLURM_STEP_ID'],flush=True)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ref(p):return {'path':str(p),'sha256':sha(p)}
def put(name,value):
 p=W/'.review_metadata'/name;b=(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n').encode()
 with p.open('xb') as f:f.write(b);f.flush();os.fsync(f.fileno())
 return ref(p)
protected={};units=[]
for copies in (1,2):
 paths=list((R/'data').glob('2026-11-01-18-54_common*confirmation_paired-rounds-7*copies-'+str(copies)+'_*'));assert len(paths)==1
 unit=paths[0];plan=json.loads((unit/'measurement_plan.json').read_bytes());units.append((unit,plan))
 for key in ('core','paired_source'):
  receipt=plan[key]
  for relative,value in receipt['files'].items():
   expected=value['sha256'] if key=='core' else value
   path=Path(receipt['source_root'])/relative;assert sha(path)==expected;protected[str(path)]=expected
 for relative,expected in plan['helpers']['files'].items():
  for path in (Path(plan['helpers']['source_root'])/relative,unit/'code'/Path(relative).name):
   assert sha(path)==expected;protected[str(path)]=expected
actions=json.loads((R/'next_actions.json').read_bytes());M=Path(actions['persistent_monitor']['packet'])
manifest=json.loads((M/'manifest.yaml').read_bytes());monitor_plan=json.loads(Path(manifest['plan']['path']).read_bytes())
source_manifest=json.loads(Path(monitor_plan['source_manifest']['path']).read_bytes())
assert source_manifest['source_commit']=='79a01e851ff92ddba7d0e12eaf8164cab2028878' and len(source_manifest['files'])==265
snapshot=Path(monitor_plan['source']).parent
for relative,value in source_manifest['files'].items():
 path=snapshot/relative;assert sha(path)==value['sha256'];protected[str(path)]=value['sha256']
before={p:sha(p) for p in protected};assert before==protected
blocked=[]
class NoNumerical(importlib.abc.MetaPathFinder):
 def find_spec(self,name,path=None,target=None):
  if name.split('.')[0] in {'jax','jaxlib','numpy','torch','flax','optax','scipy','jaxtari','exploration_platform','cupy','pynvml'}:
   blocked.append(name);raise RuntimeError('numerical import forbidden')
sys.meta_path.insert(0,NoNumerical())
spec=importlib.util.spec_from_file_location('candidate_monitor_owned_unit_metadata',S/'monitor.py');monitor=importlib.util.module_from_spec(spec);spec.loader.exec_module(monitor)
def no_query(*args,**kwargs):blocked.append(str(args));raise RuntimeError('no scheduler or child process allowed')
monitor.subprocess.Popen=no_query
actual=[]
for unit,plan in units:
 observed=monitor.selected_unit_job(R,unit,())
 assert observed['job_id']=={'1':'6783295','2':'6783296'}[str(plan['configuration']['learner']['n_copies'])]
 assert observed['state']=='waiting' and 'no current scheduler observation' in observed['progress']
 actual.append({'unit':str(unit),'job_id':observed['job_id'],'current_scheduler_observation_supplied':False,
   'actual_owned_receipt_envelope_and_ledger_match':True,'no_current_run_state_inferred':True})
after={p:sha(p) for p in protected};assert before==after and not blocked
source_paths=[S/'monitor.py',S/'tests/test_monitor_selected_unit_jobs.py']
source={str(p):sha(p) for p in source_paths}
author=W/'.review_metadata/selected_unit_display_author_6783003_87.json';assert json.loads(author.read_bytes())['passed']
proof=put('selected_unit_display_actual_metadata_and_protected_sources_6783003_'+os.environ['SLURM_STEP_ID']+'.json',{
 'schema_version':1,'kind':'selected_unit_display_actual_owned_metadata_and_source_preservation','passed':True,
 'job_id':'6783003','step_id':os.environ['SLURM_STEP_ID'],'actual_owned_metadata':actual,
 'protected_before':before,'protected_after':after,'protected_source_files':len(protected),
 'numerical_source_files':13,'active_monitor_source_files':265,'active_a4500_staged_helpers':42,
 'read_only_scheduler_calls':0,'job_actions':0,'NUM_or_GPU_work':False,'scientific_approval_granted':False})
freeze=put('selected_unit_display_corrected_source_freeze.json',{'schema_version':1,'kind':'selected_unit_display_source_and_tests_freeze',
 'base_commit':'e02f72b93a6705853c5d35b72d81fbc3f5310679','files':source,'author_controls':ref(author),
 'actual_metadata_and_source_preservation':proof,'tests':42,'active_cpu_job':'6783003',
 'numerical_or_live_helper_source_changed':False,'current_monitor_source_changed':False,
 'old_fixture_failure':ref(W/'.review_metadata/selected_unit_display_author_6783003_79.json'),
 'source_scope':'configured A4500 selected measurement/profile metadata display only; direct and legacy unchanged',
 'job_actions':0,'NUM_or_GPU_work':False,'scientific_approval_granted':False})
archive=W/'.review_metadata/final_frozen_source';archive.mkdir()
for p in source_paths:
 target=archive/p.name
 with target.open('xb') as f:f.write(p.read_bytes());f.flush();os.fsync(f.fileno())
 target.chmod(0o444);assert sha(target)==sha(p)
print(json.dumps({'passed':True,'freeze':freeze,'source':source,'protected_sources':len(protected),'actual_metadata':actual,'actual_step':os.environ['SLURM_STEP_ID']}))

