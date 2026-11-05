"""One fresh metadata-only reader: helper source and plan reads, no saved results."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

M=Path(__file__).resolve().parent
S=M.parents[1]/'10_jax_exploration_platform/scripts/atari_pixel_campaign'
spec=importlib.util.spec_from_file_location('_current_profile_saved_reader_fixture',S/'direct_profile_saved_consumers.py')
saved=importlib.util.module_from_spec(spec);spec.loader.exec_module(saved)
payload=json.loads(sys.stdin.buffer.read())
path=Path(payload['plan_path']);raw=path.read_bytes()
saved.require(hashlib.sha256(raw).hexdigest()==payload['plan_sha256'],'actual metadata plan bytes changed')
plan=json.loads(raw);unit=path.parent
def no_scientific_result(event,values):
    if event=='open' and isinstance(values[0],(str,bytes)):
        name=Path(values[0].decode() if isinstance(values[0],bytes) else values[0]).name
        if name in ('result.json','direct_qualification_result.json','failure.json','snapshot.json'):
            raise AssertionError('scientific result/payload read forbidden in namespace fixture')
    if event=='import' and values[0].split('.')[0] in saved.NUMERICAL:
        raise AssertionError('numerical import forbidden in namespace fixture')
sys.addaudithook(no_scientific_result)
sources={Path(relative).stem:{'path':str(unit/'code'/Path(relative).name),'sha256':sha,
    'raw':(unit/'code'/Path(relative).name).read_bytes()} for relative,sha in plan['helpers']['files'].items()}
for source in sources.values():saved.require(saved.digest(source['raw'])==source['sha256'],'staged source hash mismatch')
common=payload['role']=='confirmation'
saved.source_roster(sources,common,accepted_snapshot=common and 'accepted_evidence_snapshot_protocol' in plan)
if common:saved.validate_common_metadata(plan,unit)
loader=saved.install_sources(sources)
if common:
    import direct_common_confirmation as guard
    guard.contract(plan)
import direct_measurement as direct
saved.require(Path(direct.__file__)==unit/'code/direct_measurement.py','ambient direct helper entered namespace')
saved.require(not saved.NUMERICAL.intersection(sys.modules),'numerical module entered metadata fixture')
print(json.dumps({'role':payload['role'],'plan':{'path':str(path),'sha256':payload['plan_sha256']},
    'source_count':len(sources),'loaded':loader.loaded,'actual_namespace_contract_passed':True,
    'scientific_results_opened':0,'GPU_or_numerical_operations':0,'operational_admission':False}))
