"""Seal the disjoint A4500 profile-only author slice; no tool or target code runs."""
import ast
import hashlib
import json
import os
from pathlib import Path
import time
M=Path(__file__).resolve().parent;W=M.parents[1];S=W/'10_jax_exploration_platform/scripts/atari_pixel_campaign'
def sha(raw):return hashlib.sha256(raw).hexdigest()
baseline=(M/'profile_selected_published_e206.py').read_bytes();assert sha(baseline)=='f73918863bcde274c833ad08702936714a2a4cde7ea875e32d319201150f4c57'
before={n.name:n for n in ast.parse(baseline).body if isinstance(n,ast.FunctionDef)}
current=(S/'profile_selected.py').read_bytes();after={n.name:n for n in ast.parse(current).body if isinstance(n,ast.FunctionDef)}
dump=lambda n:ast.dump(n,include_attributes=False)
for name in ('_capture_core','require_reviewed_profile_process_admission','capture_direct_owned_processes','run_direct_owned_waiter','capture_direct_owned_child'):
    assert dump(before[name])==dump(after[name]),name
old_h=next(n for n in before['required_helpers'].body if isinstance(n,ast.If) and isinstance(n.test,ast.Name) and n.test.id=='direct_profile')
new_h=next(n for n in after['required_helpers'].body if isinstance(n,ast.If) and isinstance(n.test,ast.Name) and n.test.id=='direct_profile')
assert dump(old_h)==dump(new_h)
paths=[S/'profile_selected.py',S/'tests/test_configured_native_profile_plumbing.py']
files={str(path.relative_to(W)):{'sha256':sha(path.read_bytes()),'bytes':path.stat().st_size,'mode':path.stat().st_mode&0o777} for path in paths}
for path in paths:ast.parse(path.read_bytes())
record={'kind':'configured_native_profile_disjoint_author_slice_freeze','files':files,
    'published_e206_baseline':{'path':str(M/'profile_selected_published_e206.py'),'sha256':sha(baseline)},
    'unchanged_capture_core_ast_sha256':sha(dump(after['_capture_core']).encode()),
    'h100_direct_branch_and_process_functions_unchanged':True,'formal_profile_admission_closed':True,
    'new_cli_qualification_fields':'configured native actual diagnostic result and independent review only',
    'producer_one_scope':'before first version through capture/postversion/export and complete closure',
    'scientific_credit':0,'actual_target_qualification':False,'operational_admission':False,
    'job_id':os.environ['SLURM_JOB_ID'],'step_id':os.environ['SLURM_STEP_ID'],'created_at_seconds':time.time()}
path=M/'source_freeze_author_1.json'
with path.open('x') as handle:json.dump(record,handle,indent=2,sort_keys=True);handle.write('\n');handle.flush();os.fsync(handle.fileno())
print(str(path));print(sha(path.read_bytes()))
