"""Roster only the reviewed8paths and bounded UTF8 metadata for a detached local commit."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import time
M=Path(__file__).resolve().parent;W=M.parents[1]
def sha(raw):return hashlib.sha256(raw).hexdigest()
def row(path):
    raw=path.read_bytes();info=path.stat()
    return {'sha256':sha(raw),'bytes':len(raw),'mode':info.st_mode&0o777}
freeze_path=M/'source_freeze.json';assert sha(freeze_path.read_bytes())=='562bc51e25645f31113602c082280bff39882ab51bc1c8b593ca2b6363a42d79'
freeze=json.loads(freeze_path.read_bytes())
approval_path=W/'.review_metadata/root_history_integration_review/independent_review.json'
assert sha(approval_path.read_bytes())=='a3a89936567afd5f4c5de45aad455e8e8a8ab6d6fedf52f2ff6024de8369b811'
approval=json.loads(approval_path.read_bytes());assert approval['approved_for_closed_source_integration'] is True and approval['operational_admission'] is False
assert len(freeze['source_and_tests'])==8
for relative,expected in freeze['source_and_tests'].items():assert row(W/relative)=={k:expected[k] for k in ('sha256','bytes','mode')},relative
copied=json.loads((W/'.review_metadata/future_profile_exact_readers_budget/copied_draft_manifest.json').read_bytes())
prior=M/'prior_accepted_component_544';prior.mkdir(exist_ok=False)
prior_sources={}
for key,name in (('source_freeze','source_freeze.json'),('prior_component_review','independent_component_summary.json')):
    reference=copied[key];raw=Path(reference['path']).read_bytes();assert sha(raw)==reference['sha256']
    target=prior/name
    with target.open('xb') as handle:handle.write(raw);handle.flush();os.fsync(handle.fileno())
    prior_sources[name]={'original_reference':reference,'archived_path':str(target.relative_to(W)),'sha256':sha(raw)}
with (prior/'archive_provenance.json').open('x') as handle:
    json.dump(prior_sources,handle,indent=2,sort_keys=True);handle.write('\n');handle.flush();os.fsync(handle.fileno())
directories=['future_profile_exact_readers_budget','future_profile_history_integration','independent_full_profile_adapter_review','root_history_integration_review']
roster_path=M/'prepared_local_commit_roster.json';paths_path=M/'local_commit_paths.txt'
metadata=[]
for name in directories:
    root=W/'.review_metadata'/name
    for path in sorted(root.rglob('*')):
        if path.is_dir():continue
        info=path.lstat();assert stat.S_ISREG(info.st_mode) and info.st_uid==2618919 and not path.is_symlink(),str(path)
        assert path.suffix in ('.py','.json','.txt') and info.st_size<=2*1024*1024,str(path)
        raw=path.read_bytes();raw.decode('utf8');assert b'\0' not in raw,str(path)
        if path.suffix=='.json':json.loads(raw)
        metadata.append(str(path.relative_to(W)))
paths=sorted(set(freeze['source_and_tests'])|set(metadata)|{str(roster_path.relative_to(W)),str(paths_path.relative_to(W))})
assert all('\n' not in value and '\r' not in value for value in paths)
with paths_path.open('x') as handle:handle.write('\n'.join(paths)+'\n');handle.flush();os.fsync(handle.fileno())
files={relative:row(W/relative) for relative in paths if relative!=str(roster_path.relative_to(W))}
record={'schema_version':1,'kind':'exact_reviewed_detached_local_commit_roster','parent_commit':freeze['base_commit'],
    'source_freeze':{'path':str(freeze_path),'sha256':sha(freeze_path.read_bytes())},
    'independent_approval':{'path':str(approval_path),'sha256':sha(approval_path.read_bytes())},
    'runtime_source_and_test_paths':list(freeze['source_and_tests']),'runtime_source_and_test_count':8,
    'paths':paths,'files_excluding_this_roster':files,'roster_self_sha256_omitted':True,
    'text_metadata_paths_count':len(paths)-8,'total_bytes_excluding_roster':sum(value['bytes'] for value in files.values()),
    'captured_binary_or_large_payload_files':0,'prior_accepted_component_metadata_archived':prior_sources,
    'operational_admission':False,'actual_target_qualification':False,'scientific_credit':0,
    'job_id':os.environ['SLURM_JOB_ID'],'step_id':os.environ['SLURM_STEP_ID'],'created_at_seconds':time.time()}
with roster_path.open('x') as handle:json.dump(record,handle,indent=2,sort_keys=True);handle.write('\n');handle.flush();os.fsync(handle.fileno())
print(str(roster_path));print(sha(roster_path.read_bytes()));print(json.dumps({'paths':len(paths),'source_tests':8,'text_metadata':len(paths)-8,'bytes':record['total_bytes_excluding_roster']}))
