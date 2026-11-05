"""Pure saved-reader controls using an actual own-CPU record and explicit legacy premises.

No native process API, tool, device or live PID is used. The omission test isolates
the NEW manifest reader with synthetic premises for already-passed legacy scope
and producer-role validation; it does not claim native-provider qualification.
"""
import ast
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile
import time
from types import SimpleNamespace

D=Path(__file__).resolve().parent;S=D/'preserved_runtime'
F=Path('/p/rlprojects/RND_worktrees/2026-11-05-13-23_profile-tool-scope_pinned-parent-exit_fresh-producer-child-proof_unknown-parents-refused_gpu-actions-0/.review_metadata/profile_parent_exit_fixture/actual_positive_cpu_fixture_metadata')
assert os.environ.get('SLURM_JOB_ID')=='6825983' and os.environ.get('SLURM_STEP_ID','').isdecimal()
print(json.dumps({'actual_job_id':os.environ['SLURM_JOB_ID'],'actual_step_id':os.environ['SLURM_STEP_ID']}),flush=True)
start=time.monotonic()
def extract(path,names,namespace):
    tree=ast.parse(path.read_text());nodes=[node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name in names]
    assert len(nodes)==len(names);exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),namespace)
def read(path):return json.loads(Path(path).read_text())
def ref(path):return {'path':str(path),'sha256':hashlib.sha256(Path(path).read_bytes()).hexdigest()}
namespace={'hashlib':hashlib,'json':json,'math':math,'Path':Path,'PROTOCOL':'profile-target-pinned-parent-exit-observation-version-1',
    'NAME':'profile_target_tool_scope.py','PREFIX':'10_jax_exploration_platform/scripts/atari_pixel_campaign/'}
extract(S/'profile_target_tool_scope.py',{'require','same','validate_parent_exit_record','validate_saved_scope'},namespace)
for node in ast.parse((S/'profile_target_tool_scope.py').read_text()).body:
    if isinstance(node,ast.Assign) and any(isinstance(target,ast.Name) and target.id=='EXIT_REFUSALS' for target in node.targets):
        namespace['EXIT_REFUSALS']=ast.literal_eval(node.value)
base_namespace={};extract(S/'direct_tool_descendants.py',{'require','parsed_identity'},base_namespace)
parser_namespace={'re':re,'os':os};extract(S/'linux_pidfd_process_control.py',{'require','parse_pidfd_fdinfo'},parser_namespace)
base=SimpleNamespace(MAX_EVENTS=100000,parsed_identity=base_namespace['parsed_identity'])
parser=SimpleNamespace(parse_pidfd_fdinfo=parser_namespace['parse_pidfd_fdinfo'])
observer=SimpleNamespace(receipt=ref,read_json=read,parsed_uids=lambda raw:[int(value) for line in raw.splitlines()
    if line.startswith('Uid:') for value in line.split()[1:]])
terminal=read(F/'tool_descendant_scope_terminal.json');metadata=read(F/'fixture_observer_metadata.json')
producer=metadata['owner'];opened=metadata['open_events'];record=read(F/'tool_parent_exit_observation_1.json')
namespace['validate_parent_exit_record'](record,terminal,producer,base,observer,opened,parser)
mutated=copy.deepcopy(record)
late=terminal['terminal']['observed_at_seconds']+10000
mutated['handle_state']['observed_at_seconds']=late
mutated['observed_at_seconds']=late+1
mutated['fresh_producer_children']['started_at_seconds']=late+2
mutated['fresh_producer_children']['finished_at_seconds']=late+3
namespace['validate_parent_exit_record'](mutated,terminal,producer,base,observer,opened,parser)
actual_events=[event for event in terminal['events'] if event.get('kind')=='known_pinned_parent_exit_reconciled']
assert len(actual_events)==1 and actual_events[0]['observation']['sha256']==ref(F/'tool_parent_exit_observation_1.json')['sha256']
legacy_result={key:ref(F/name) for key,name in (('tool_descendant_scope_begin','tool_descendant_scope_begin.json'),
    ('tool_descendant_scope_terminal','tool_descendant_scope_terminal.json'),('tool_descendant_scope_restored','tool_descendant_scope_restored.json'))}
base.validate_saved_scope=lambda *args,**kwargs:legacy_result
namespace['base_module']=lambda plan:base
namespace['adjacent']=lambda plan,name:parser
manifest=read(F/'tool_parent_exit_observations.json');manifest['records']=[]
manifest['scope_terminal']=legacy_result['tool_descendant_scope_terminal']
manifest['scope_restored']=legacy_result['tool_descendant_scope_restored']
plan={'kind':'full_original_profile_source_target_qualification','helpers':{'files':{
    namespace['PREFIX']+'profile_target_tool_scope.py':manifest['adapter_sha256'],
    namespace['PREFIX']+'direct_tool_descendants.py':manifest['base_sha256']}}}
with tempfile.TemporaryDirectory(prefix='isolated_new_manifest_reader_') as temporary:
    directory=Path(temporary)
    (directory/'tool_parent_exit_observations.json').write_text(json.dumps(manifest))
    (directory/'profile_target_process_producer_terminal.json').write_text(json.dumps({
        'role':'producer','completed':True,'kind':'actual_profile_target_process_role_terminal',
        'provider':{'descriptor_events':opened},'explicit_synthetic_fixture_premise':True}))
    accepted=namespace['validate_saved_scope'](directory,plan,observer,producer,0,'unused',lambda value:None)
    assert accepted['discarded_parent_samples_used_as_child_lists'] is False
review={'schema_version':1,'kind':'independent_initial_profile_parent_exit_scope_saved_reader_review','decision':'BLOCK',
    'approved':False,'passed':False,'source_freeze_sha256':'e2a4b5e77e5229ef2c8107afa0e6d989c3927dcde63e2102bbd1469fbc131b35',
    'preservation_manifest':ref(D/'preservation_manifest.json'),'job_id':os.environ['SLURM_JOB_ID'],'step_id':os.environ['SLURM_STEP_ID'],
    'actual_positive_CPU_fixture':{'directory':str(F),'original_observation':ref(F/'tool_parent_exit_observation_1.json'),
        'native_process_provider':False,'scope':'own actual system-Python stdlib pidfds; no current target/GPU qualification'},
    'unchanged_positive_record_accepted':True,'post_terminal_observation_incorrectly_accepted':True,
    'omitted_manifest_record_with_immutable_terminal_event_incorrectly_accepted':True,
    'omission_control_premises':'Only the NEW manifest function is under test. Legacy full-scope and completed producer role are explicit synthetic premises, with actual own-CPU terminal/event bytes and actual stdlib open-event identities. This is not full source/target acceptance.',
    'actual_exit_event_retained':actual_events[0],'mutated_observation_times':{
        'scope_terminal':terminal['terminal']['observed_at_seconds'],'handle_state':late,'record':late+1,'fresh_start':late+2,'fresh_end':late+3},
    'required_corrections':['Bind ordered manifest observations exactly to immutable terminal exit events, including ref/PID/birth.',
        'Bind observation/fresh proof to the actual same open generation and close event and scope/prod terminal time window.'],
    'existing_runtime_algorithm_or_old_AA7_modified':False,'GPU_device_library_or_live_process_actions':False,'seconds':time.monotonic()-start}
path=D/'independent_initial_saved_reader_review.json'
with path.open('x') as stream:json.dump(review,stream,indent=2,sort_keys=True);stream.write('\n');stream.flush();os.fsync(stream.fileno())
print(json.dumps({'decision':'BLOCK','review':ref(path),'step':review['step_id'],'seconds':review['seconds'],
    'positive':True,'omitted_record_incorrectly_accepted':True,'post_terminal_times_incorrectly_accepted':True}))
