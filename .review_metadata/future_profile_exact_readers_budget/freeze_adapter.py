"""Freeze the six-path closed adapter and verify untouched numerical/live source bytes."""
import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
M=Path(__file__).resolve().parent;W=M.parents[1]
PREFIX='10_jax_exploration_platform/scripts/atari_pixel_campaign/'
BASE='27f9b5c9276de2405c9aa7225b2c4c6b99e5c50d'
def sha(raw):return hashlib.sha256(raw).hexdigest()
def source(path):
    raw=path.read_bytes();info=path.stat()
    return {'path':str(path),'sha256':sha(raw),'bytes':len(raw),'mode':info.st_mode&0o777}
def syntax(raw):
    return {n.name:sha(ast.dump(n,include_attributes=False).encode()) for n in ast.parse(raw).body
        if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
def git(*args):return subprocess.check_output(['git','-C',str(W),*args])
head=git('rev-parse','HEAD').decode().strip();assert head==BASE
copy=json.loads((M/'copied_draft_manifest.json').read_bytes());old_root=Path(copy['source_freeze']['path']).parents[2]
old_freeze=json.loads(Path(copy['source_freeze']['path']).read_bytes())
assert sha(Path(copy['source_freeze']['path']).read_bytes())==copy['source_freeze']['sha256']
old_files={}
for relative,value in copy['explicit_copied_paths'].items():
    raw=Path(value['copied_from']).read_bytes();assert sha(raw)==value['copied_sha256']
    old_files[relative]=sha(raw)
runtime_paths=[PREFIX+name for name in ('direct_profile_prerequisites.py','direct_profile_saved_consumers.py','profile_metadata_runtime.py')]
test_paths=[PREFIX+'tests/'+name for name in ('test_direct_profile_prerequisites.py','test_direct_profile_saved_consumers.py','test_profile_metadata_runtime.py')]
files={relative:source(W/relative) for relative in runtime_paths+test_paths}
for relative in files:ast.parse((W/relative).read_bytes())
expected_changed={PREFIX+'direct_profile_prerequisites.py',PREFIX+'tests/test_direct_profile_prerequisites.py'}
assert set(git('diff','--name-only',BASE).decode().splitlines())==expected_changed
tracked=git('ls-tree','-r','--name-only',BASE,'--','10_jax_exploration_platform').decode().splitlines()
protected={}
for relative in tracked:
    if Path(relative).suffix not in ('.py','.sh') or relative in expected_changed:continue
    actual=(W/relative).read_bytes();original=git('show',BASE+':'+relative)
    assert actual==original,relative
    protected[relative]=sha(actual)
assert len(protected)>100
original_numeric={}
for path,expected in old_freeze['unchanged_original_numerical_files'].items():
    assert sha(Path(path).read_bytes())==expected,path
    original_numeric[path]=expected
assert len(original_numeric)==13
old_saved=(old_root/PREFIX/'direct_profile_saved_consumers.py').read_bytes()
new_saved=(W/PREFIX/'direct_profile_saved_consumers.py').read_bytes()
old_nodes=syntax(old_saved);new_nodes=syntax(new_saved)
unchanged=['ExactSavedLoader','install_sources','source_roster','validate_common_metadata','source_closure',
    'statistic_source_proof','combine_records_with_verified_readers','combine','compute_context','read','document']
for name in unchanged:assert old_nodes[name]==new_nodes[name],name
old_pre=(old_root/PREFIX/'direct_profile_prerequisites.py').read_bytes()
new_pre=(W/PREFIX/'direct_profile_prerequisites.py').read_bytes()
old_pre_nodes=syntax(old_pre);new_pre_nodes=syntax(new_pre)
assert old_pre_nodes['validate_initial_cases']==new_pre_nodes['validate_initial_cases']
assert old_pre_nodes['require_admission']==new_pre_nodes['require_admission']
def function(raw,name):return next(n for n in ast.parse(raw).body if isinstance(n,ast.FunctionDef) and n.name==name)
old_history=function(old_pre,'validate_current_history');new_history=function(new_pre,'validate_current_history')
assert ast.dump(ast.Module(body=old_history.body[-4:],type_ignores=[]),include_attributes=False)==ast.dump(ast.Module(body=new_history.body[-4:],type_ignores=[]),include_attributes=False)
admission=function(new_pre,'require_admission');assert isinstance(admission.body[-1],ast.Raise)
for raw in (new_pre,new_saved):
    assert not any(isinstance(n,ast.keyword) and n.arg=='timeout' and isinstance(n.value,ast.Constant) and n.value.value==600 for n in ast.walk(ast.parse(raw)))
runtime_tree=ast.parse((W/PREFIX/'profile_metadata_runtime.py').read_bytes())
protocol=ast.literal_eval(next(n.value for n in runtime_tree.body if isinstance(n,ast.Assign) and any(isinstance(x,ast.Name) and x.id=='PROTOCOL' for x in n.targets)))
assert protocol['work_seconds']==240 and protocol['cleanup_seconds']==60 and protocol['total_seconds']==300 and protocol['operational_admission'] is False
core=syntax((W/PREFIX/'profile_selected.py').read_bytes())['_capture_core']
artifacts={str(p.relative_to(W)):source(p) for p in M.iterdir() if p.is_file() and p.suffix in ('.py','.json','.txt') and p.name not in ('source_freeze.json','author_handoff.json')}
result={'schema_version':1,'kind':'closed_future_profile_exact_readers_and_single_metadata_budget_source_freeze',
    'base_commit':BASE,'source_and_tests':files,'artifacts':artifacts,'original_four_path_draft_preserved':old_files,
    'protected_tracked_source_files':protected,'protected_tracked_source_count':len(protected),
    'protected_original_numerical_files':original_numeric,'capture_core_ast_sha256':core,
    'unchanged_saved_functions':{name:new_nodes[name] for name in unchanged},
    'unchanged_initial_case_validator_ast':new_pre_nodes['validate_initial_cases'],
    'old28_history_raw_tail_unchanged':True,'old55_raw_consumers_unchanged':True,
    'metadata_protocol':protocol,'current_profile_helper_count':34,
    'new29_qualification_and_correctness_mode':'exact reviewed snapshot facts; no old raw gate replay',
    'new29_confirmation_mode':'exact staged current common consumer; genuine result/review still required',
    'source_implementation_ready_for_independent_review':True,'independent_full_source_review_passed':False,
    'new_operational_unit_prepared':False,'native_formal_profile_support':False,'current_target_qualification':False,
    'scientific_credit':0,'operational_admission':False,'job_id':os.environ['SLURM_JOB_ID'],
    'step_id':os.environ['SLURM_STEP_ID'],'created_at_seconds':time.time()}
target=M/'source_freeze.json'
with target.open('x') as handle:
    json.dump(result,handle,indent=2,sort_keys=True);handle.write('\n');handle.flush();os.fsync(handle.fileno())
print(str(target));print(sha(target.read_bytes()));print('protected tracked source files:',len(protected))
