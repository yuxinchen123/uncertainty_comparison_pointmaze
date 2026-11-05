"""Read only the reviewed accepted-history metadata and exact helper Git sources."""
import importlib.util
import json
import os
from pathlib import Path
import sys
M=Path(__file__).resolve().parent;S=M.parents[1]/'10_jax_exploration_platform/scripts/atari_pixel_campaign'
def load(name):
    spec=importlib.util.spec_from_file_location('_pure_policy_'+name,S/(name+'.py'))
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module
saved=load('direct_profile_saved_consumers');runtime=load('profile_metadata_runtime')
payload=json.loads(sys.stdin.buffer.read());p=payload['runtime_plan'];budget=runtime.Budget(payload['metadata_origin'],p)
path=Path(payload['common_plan']);common=json.loads(path.read_bytes())
captured=json.loads(Path(common['accepted_evidence_snapshot_inputs']['proof']['path']).read_bytes())
blocked=set()
for accepted in (captured['accepted']['qualification'],*captured['accepted']['correctness'].values()):
    manifest_path=Path(accepted['historical_manifest']['path']);manifest=json.loads(manifest_path.read_bytes())
    blocked.update(str(manifest_path.parent/row['relative_path']) for row in manifest['files'])
def no_old_payload_bytes(event,values):
    if event=='open' and isinstance(values[0],(str,bytes)):
        name=os.fsdecode(values[0])
        if name in blocked or '.so' in Path(name).name:
            raise AssertionError('historical captured payload/driver bytes are forbidden in fact-reuse fixture')
sys.addaudithook(no_old_payload_bytes)
facts=[]
with runtime.MetadataScope(p,budget) as scope:
    with runtime.work_alarm(budget):
        guard,env,bootstrap,proof=saved.common_policy(common,common,path.parent,scope)
        sources=saved.source_closure(common,path.parent,guard,env,common=True)
        snapshot=saved.reviewed_snapshot(common,path.parent)
        for role,accepted in (('qualification',snapshot.qual),('correctness',snapshot.correctness)):
            previous=snapshot.files.json(accepted['plan'])
            returned,fact=saved.accepted_history_fact(common,role,accepted['result'],previous,snapshot)
            assert fact['old_saved_consumer_reexecuted'] is False and fact['historical_payloads_rehashed'] is False
            assert fact['accepted']['review']==accepted['review']
            assert fact['returned_contract_sha256']==saved.digest(saved.encoded(returned))
            wrong=dict(accepted['result']);wrong['sha256']='0'*64
            try:saved.accepted_history_fact(common,role,wrong,previous,snapshot)
            except ValueError:pass
            else:raise AssertionError('changed accepted result hash passed')
            facts.append(fact)
        try:saved.accepted_history_fact(common,'confirmation',snapshot.correctness['result'],previous,snapshot)
        except ValueError:pass
        else:raise AssertionError('historical reuse replaced new confirmation')
        configuration=guard.finish_repository_configurations()
runtime.validate_scope(scope.proof,budget,scope.observer,scope.owner)
assert 'direct_measurement' not in sys.modules
assert proof['history_policy']['historical_payloads_rehashed'] is False
print(json.dumps({'passed':True,'policy':proof,'source_count':len(sources),'repository_configuration':configuration,
    'accepted_history_facts':facts,'captured_payload_paths_open_blocked':len(blocked),
    'scope':scope.proof,'original_cases_reexecuted':False,'scientific_credit':0,'operational_admission':False}))
