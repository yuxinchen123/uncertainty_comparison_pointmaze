"""Read genuine reviewed admission metadata only; no operational profile unit is created."""
import hashlib
from copy import deepcopy
import importlib.util
import json
import os
from pathlib import Path
import time
M=Path(__file__).resolve().parent;W=M.parents[1];S=W/'10_jax_exploration_platform/scripts/atari_pixel_campaign'
spec=importlib.util.spec_from_file_location('_profile_original_authority_probe',S/'direct_profile_prerequisites.py')
profile=importlib.util.module_from_spec(spec);spec.loader.exec_module(profile)
peer=json.loads((W/'.review_metadata/independent_full_profile_adapter_review/independent_dependency_6821437_50.json').read_bytes())
ref=peer['profile_pointer']['common_plan'];raw=Path(ref['path']).read_bytes();assert hashlib.sha256(raw).hexdigest()==ref['sha256']
common=json.loads(raw);plan={**common,'kind':'selected_configuration_profile','unit_id':'nonoperational_source_authority_metadata_probe'}
started=time.time();admission,provenance=profile.original_admission_metadata(plan,common,live=True)
assert hashlib.sha256(admission).hexdigest()==profile.ORIGINAL_ADMISSION_SHA256
assert provenance['profile_worker_admitted'] is False and provenance['current_target_qualification'] is False
pointer=profile.history_pointer(plan,common);assert pointer['admission']['sha256']==profile.ORIGINAL_ADMISSION_SHA256
rejected=[]
for key,replacement in (('source_root','/synthetic/DIFFERENT_NUMERICAL_SOURCE'),('source_commit','0'*40),('copy_seeds',[2])):
    changed=deepcopy(plan);changed[key]=replacement
    try:profile.original_admission_metadata(changed,common)
    except ValueError:rejected.append(key)
    else:raise AssertionError('changed original-source authority field passed:'+key)
changed=deepcopy(plan);changed['configuration']['learner']['num_steps']=64
try:profile.original_admission_metadata(changed,common)
except ValueError:rejected.append('configuration')
else:raise AssertionError('changed configuration passed original authority')
target=M/('original_authority_'+os.environ['SLURM_JOB_ID']+'_'+os.environ['SLURM_STEP_ID']+'.json')
value={'passed':True,'common_plan':ref,'admission_sha256':hashlib.sha256(admission).hexdigest(),'provenance':provenance,
    'pointer':pointer,'profile_unit_created':False,'NUM_or_GPU_operations':0,'source_scope_only':True,
    'changed_source_config_seed_fields_rejected':rejected,
    'elapsed_seconds':time.time()-started,'operational_admission':False}
with target.open('x') as handle:json.dump(value,handle,indent=2,sort_keys=True);handle.write('\n');handle.flush();os.fsync(handle.fileno())
print(str(target))
