"""Finish the already verified corrected freeze in a fresh archive without replacing any prior bytes."""
import hashlib,json,os
from pathlib import Path
W=Path(__file__).resolve().parent.parent
assert os.environ.get('SLURM_JOB_ID')=='6783003' and os.environ.get('SLURM_STEP_ID','').isdigit()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ref(p):return {'path':str(p),'sha256':sha(p)}
freeze=W/'.review_metadata/selected_unit_display_corrected_source_freeze.json';value=json.loads(freeze.read_bytes())
assert value['tests']==42
for p,digest in value['files'].items():assert sha(p)==digest
proof=json.loads(Path(value['actual_metadata_and_source_preservation']['path']).read_bytes())
assert proof['passed'] and proof['protected_before']==proof['protected_after'] and proof['protected_source_files']==341
archive=W/'.review_metadata/final_corrected_frozen_source';archive.mkdir()
for p,digest in value['files'].items():
 p=Path(p);target=archive/p.name
 with target.open('xb') as f:f.write(p.read_bytes());f.flush();os.fsync(f.fileno())
 target.chmod(0o444);assert sha(target)==digest
out=W/'.review_metadata'/('selected_unit_display_corrected_freeze_completion_6783003_'+os.environ['SLURM_STEP_ID']+'.json')
result={'schema_version':1,'kind':'completed_corrected_selected_unit_display_freeze','passed':True,'job_id':'6783003','step_id':os.environ['SLURM_STEP_ID'],'freeze':ref(freeze),'source':value['files'],'actual_metadata_source_proof':value['actual_metadata_and_source_preservation'],'protected_files':341,'tests':42,'prior_wrapper_failure':{'actual_step':'6783003.88','exception':'FileExistsError','path':str(W/'.review_metadata/final_frozen_source'),'cause':'corrected wrapper retained the first freeze archive directory name','source_or_existing_archive_changed':False},'fresh_archive':str(archive),'existing_freeze_reused_only_after_exact_source_and_proof_checks':True,'job_or_GPU_actions':0}
b=(json.dumps(result,sort_keys=True,indent=2)+'\n').encode()
with out.open('xb') as f:f.write(b);f.flush();os.fsync(f.fileno())
print(json.dumps({'passed':True,'completion':ref(out),'freeze':ref(freeze),'source':value['files'],'actual_step':os.environ['SLURM_STEP_ID']}))

