"""Seal the affected8paths and exact published admission behavior; no broad science replay."""
import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
M=Path(__file__).resolve().parent;W=M.parents[1];P='10_jax_exploration_platform/scripts/atari_pixel_campaign/'
def sha(raw):return hashlib.sha256(raw).hexdigest()
def ref(path):return {'path':str(path),'sha256':sha(path.read_bytes())}
def nodes(raw):return {node.name:node for node in ast.parse(raw).body if isinstance(node,(ast.FunctionDef,ast.ClassDef))}
def dump(node):return ast.dump(node,include_attributes=False)
def function_bytes(raw,name):
    node=nodes(raw)[name];return b''.join(raw.splitlines(keepends=True)[node.lineno-1:node.end_lineno])
old_path=W/'.review_metadata/future_profile_exact_readers_budget/source_freeze.json';old=json.loads(old_path.read_bytes())
assert sha(old_path.read_bytes())=='b23cf4c912d8207f55b1671ce89b620f30800c62782b2334cbcffb1a040a69f5'
preserved=W/'.review_metadata/independent_full_profile_adapter_review/preserved_b23_integration_failure'
manifest=preserved/'preservation_manifest.json';assert sha(manifest.read_bytes())=='140aaf1ec7a6157c0b70ca062d6fe0cd92cccea4ae3393a32f780a3f25fd384b'
for relative,value in old['source_and_tests'].items():assert sha((preserved/relative).read_bytes())==value['sha256']
source_paths=list(old['source_and_tests'])+[P+'profile_selected.py',P+'serval02_card1_binding.py']
files={}
for relative in source_paths:
    path=W/relative;raw=path.read_bytes();ast.parse(raw)
    files[relative]={'path':str(path),'sha256':sha(raw),'bytes':len(raw),'mode':path.stat().st_mode&0o777}
for relative in (P+'direct_profile_saved_consumers.py',P+'profile_metadata_runtime.py',P+'tests/test_direct_profile_saved_consumers.py',P+'tests/test_profile_metadata_runtime.py'):
    assert files[relative]['sha256']==old['source_and_tests'][relative]['sha256']
git=lambda *args:subprocess.check_output(['git','-C',str(W),*args])
expected={P+'direct_profile_prerequisites.py',P+'tests/test_direct_profile_prerequisites.py',P+'profile_selected.py',P+'serval02_card1_binding.py'}
assert set(git('diff','--name-only',old['base_commit']).decode().splitlines())==expected
scope={}
for filename,allowed in (('profile_selected.py',{'prepare','required_helpers'}),('serval02_card1_binding.py',{'validate_history','current_admission_claim'})):
    current=(W/P/filename).read_bytes();base=git('show',old['base_commit']+':'+P+filename)
    before,after=nodes(base),nodes(current);assert set(before)==set(after)
    changed={name for name in before if dump(before[name])!=dump(after[name])};assert changed==allowed,(filename,changed)
    scope[filename]={'changed_functions':sorted(changed),'all_other_existing_functions_unchanged':True}
    if filename=='serval02_card1_binding.py':
        assert dump(ast.Module(body=before['validate_history'].body[1:],type_ignores=[]))==dump(ast.Module(body=after['validate_history'].body[2:],type_ignores=[]))
    else:assert sha(dump(after['_capture_core']).encode())==old['capture_core_ast_sha256']
published=Path('/p/rlprojects/RND_worktrees/2026-11-05-04-02_common-timing-helper_source-69f1b507_helpers-29_numerical-source-d28e6735_gpu-count-0')/P/'serval02_card1_binding.py'
published_raw=published.read_bytes();assert sha(published_raw)=='af72d3de3e84f7b8be08a73f90f5becf143b1d32c90083c1773b10343751361b'
binding=(W/P/'serval02_card1_binding.py').read_bytes()
for name in ('current_admission_claim','check_live_admission'):
    assert function_bytes(binding,name)==function_bytes(published_raw,name),name
    scope['serval02_card1_binding.py'][name+'_published_69f_function_sha256']=sha(function_bytes(binding,name))
pre=(W/P/'direct_profile_prerequisites.py').read_bytes()
oldpre=(preserved/P/'direct_profile_prerequisites.py').read_bytes()
assert dump(nodes(pre)['require_admission'])==dump(nodes(oldpre)['require_admission'])
assert dump(nodes(pre)['validate_initial_cases'])==dump(nodes(oldpre)['validate_initial_cases'])
for path,expected_hash in old['protected_original_numerical_files'].items():assert sha(Path(path).read_bytes())==expected_hash
receipts={path.name:ref(path) for path in M.glob('*_6821437_*.json')}
artifacts={str(path.relative_to(W)):ref(path) for path in M.iterdir() if path.is_file() and path.suffix in ('.py','.json') and path.name not in ('source_freeze.json','author_handoff.json')}
record={'schema_version':1,'kind':'closed_future_profile_production_history_and_original_admission_integration_source_freeze',
    'base_commit':old['base_commit'],'rejected_b23_freeze':ref(old_path),'preserved_b23_manifest':ref(manifest),
    'source_and_tests':files,'scope':scope,'published_69f_binding':ref(published),
    'original_current_admission_and_live_source_behavior_preserved':True,'original_28_55_raw_history_body_preserved':True,
    'protected_13_numerical_files':old['protected_original_numerical_files'],'capture_core_ast_sha256':old['capture_core_ast_sha256'],
    'prior_3002_protected_source_proof':ref(old_path),'unmodified_tracked_paths_outside_explicit_four_file_delta':True,
    'receipts':receipts,'artifacts':artifacts,'synthetic_upstream_review_source_and_backend_premises_explicit':True,
    'no_actual_target_backend_preparation_claim':True,'actual_profile_unit_prepared':False,
    'formal_profile_and_native_admission_closed':True,'scientific_credit':0,'operational_admission':False,
    'job_id':os.environ['SLURM_JOB_ID'],'step_id':os.environ['SLURM_STEP_ID'],'created_at_seconds':time.time()}
target=M/'source_freeze.json'
with target.open('x') as handle:json.dump(record,handle,indent=2,sort_keys=True);handle.write('\n');handle.flush();os.fsync(handle.fileno())
print(str(target));print(sha(target.read_bytes()))
