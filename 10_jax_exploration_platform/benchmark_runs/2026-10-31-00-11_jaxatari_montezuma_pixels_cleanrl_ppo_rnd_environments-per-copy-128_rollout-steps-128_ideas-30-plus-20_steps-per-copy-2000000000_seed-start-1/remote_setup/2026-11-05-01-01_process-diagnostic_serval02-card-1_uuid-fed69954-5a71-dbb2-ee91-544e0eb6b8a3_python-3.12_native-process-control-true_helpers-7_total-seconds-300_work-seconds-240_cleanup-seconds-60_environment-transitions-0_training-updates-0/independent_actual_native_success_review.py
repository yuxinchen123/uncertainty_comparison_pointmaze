"""Review actual saved process-component success; no live/native/tool/numerical operations."""
import ctypes
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

R = Path('/p/rlprojects/RND_worktrees/merge-atari-cleanrl/10_jax_exploration_platform/benchmark_runs/2026-10-31-00-11_jaxatari_montezuma_pixels_cleanrl_ppo_rnd_environments-per-copy-128_rollout-steps-128_ideas-30-plus-20_steps-per-copy-2000000000_seed-start-1')
N = '2026-11-05-01-01_process-diagnostic_serval02-card-1_uuid-fed69954-5a71-dbb2-ee91-544e0eb6b8a3_python-3.12_native-process-control-true_helpers-7_total-seconds-300_work-seconds-240_cleanup-seconds-60_environment-transitions-0_training-updates-0'
U, P = R/'data'/N, R/'remote_setup'/N
START = time.monotonic()
STEP = os.environ['SLURM_STEP_ID']
assert os.environ['SLURM_JOB_ID'] == '6783003' and STEP.isdecimal()
assert os.getresuid() == (2618919,) * 3
assert Path('/proc/self/cgroup').read_text().splitlines() == ['0::/system.slice/slurmstepd.scope/job_6783003/step_' + STEP + '/user/task_0']
FORBIDDEN = {'jax','jaxlib','numpy','torch','flax','optax','scipy','cupy','jaxtari','exploration_platform'}
def audit(event,args):
    if event=='import' and args[0].split('.')[0] in FORBIDDEN:
        raise RuntimeError('numerical import refused')
    if event in {'subprocess.Popen','socket.connect','socket.bind','socket.getaddrinfo','ctypes.dlopen','ctypes.dlsym','os.system'}:
        raise RuntimeError('process/native/library/network operation refused in saved review')
    if event=='open' and isinstance(args[0],(str,bytes)):
        path=os.fsdecode(args[0])
        if path.startswith(('/dev/nvidia','/dev/dri','/tmp/sl5nw_serval02_card1.lock')):
            raise RuntimeError('device or standing lock refused')
        if path.startswith('/proc/') and not path.startswith('/proc/self/'):
            raise RuntimeError('historical live process query refused')
        if path.startswith(('/usr/lib/','/lib/')) and any(name in path.lower() for name in ('cuda','nvidia','cudnn')):
            raise RuntimeError('GPU image reading refused')
sys.addaudithook(audit)
def sha(raw): return hashlib.sha256(raw).hexdigest()
def load(path): return json.loads(path.read_bytes())
def ref(path): return {'path':str(path),'sha256':sha(path.read_bytes())}
paths=sorted([*U.glob('*.json'),*U.glob('*.raw'),*U.glob('*.log'),*U.glob('code/*.py')])
before={str(path):sha(path.read_bytes()) for path in paths}
assert all(path.stat().st_size<=8*1024*1024 for path in paths)
plan=load(U/'measurement_plan.json')
assert ref(U/'measurement_plan.json')['sha256']=='b0182335219034ea345207a586e8b64e601df3aa942eee3f7a017f8c62cc3b30'
assert plan['kind']=='direct_nsight_process_descriptor_diagnostic' and len(plan['helpers']['files'])==7
assert plan['helpers']['source_commit'].startswith('ba1d6cfe')
assert plan['native_process_control']['expected_target_contract'] is None and plan['native_process_control']['qualification'] is False
prefix='10_jax_exploration_platform/scripts/atari_pixel_campaign/'
for relative,expected in plan['helpers']['files'].items():
    assert relative.startswith(prefix)
    staged=U/'code'/relative[len(prefix):]
    assert sha(staged.read_bytes())==expected==sha((Path(plan['helpers']['source_root'])/relative).read_bytes())
root_return=load(P/'root_actual_target_launch_return.json')
assert type(root_return['returncode']) is int and root_return['returncode']==0
assert root_return['local_wait_timed_out'] is False and root_return['no_signal_sent'] is True and root_return['no_retry'] is True
assert 0<=root_return['elapsed_seconds']<300
assert not (U/'failure.json').exists()
assert not list(U.glob('native_process_*_failure.json'))
result=load(U/'result.json')
assert result['kind']==plan['kind'] and result['completed'] is True
ZERO={'scientific_credit':0,'profile_credit':0,'timing_credit':0,'case_credit':0,'idea_credit':0,
    'environment_transitions':0,'training_updates':0,'admission_granted':False,'worker_admitted':False,
    'cuda_qualification_established':False,'physical_binding_established':False,
    'whole_tool_library_or_device_activity_quantified':False}
assert all(type(result[k]) is type(v) and result[k]==v for k,v in ZERO.items())
assert result['plan']==ref(U/'measurement_plan.json')
spec=importlib.util.spec_from_file_location('independent_actual_saved_native_success_direct',U/'code/direct_profile_process.py')
direct=importlib.util.module_from_spec(spec);spec.loader.exec_module(direct)
observer=direct.generic(plan)
# This is the exact staged saved consumer. No live method, provider bootstrap or
# ctypes symbol lookup is invoked; its captured public CPU images are hashed as
# data solely by the unchanged contract validator.
proof=direct.validate_saved(U,plan)
assert proof==result['process_proof'] and proof['passed'] is True
assert proof['admission_granted'] is False and proof['scientific_credit']==0
assert proof['native_process_evidence']['passed'] is True
assert proof['native_process_evidence']['qualification'] is False
payload=load(U/'probe_result.json')
assert result['payload']==ref(U/'probe_result.json')
assert payload['completed'] is True and payload['numerical_modules']==[] and payload['cuda_profiler_api_called'] is False
assert all(type(payload[k]) is type(v) and payload[k]==v for k,v in ZERO.items())
deadline=direct.validate_diagnostic_origin(U,plan,observer)
wait=load(U/'controller_wait.json')
intent=load(U/'direct_profile_process_intent.json')
assert type(wait['returncode']) is int and wait['returncode']==0 and wait['exited_normally'] is True and wait['interventions']==[]
assert wait['waited_pid']==intent['producer']['pid']
assert wait['controller_no_standing_descriptor_after']['standing_descriptor_count']==0
observer.validate_snapshot(wait['controller_after'],direct.OWNER,intent['producer']['cgroup'])
deadline.verify_snapshot(wait['diagnostic_deadline'],phase='total')
controller_spawn=load(U/'controller_spawn.json')
assert controller_spawn['producer']['pid']==intent['producer']['pid']
assert controller_spawn['producer']['process_start_time']==intent['producer']['process_start_time']
producer_release=load(U/'producer_release.json')
assert producer_release['controller_spawn']==ref(U/'controller_spawn.json')
assert producer_release['launch_review']==ref(U/'diagnostic_launch_review.json')
assert producer_release['ledger_append']==ref(U/'diagnostic_process_ledger_append.json')
assert producer_release['diagnostic_origin']==ref(U/'diagnostic_controller_time_origin.json')
ledger_append=load(U/'diagnostic_process_ledger_append.json')
assert producer_release['producer_row']==ledger_append['row']
assert ledger_append['row']['pid']==intent['producer']['pid']
assert ledger_append['row']['start_time']==intent['producer']['process_start_time']
assert ledger_append['row']['producer_owns_standing_descriptor'] is True
assert ledger_append['row']['controller_owns_standing_descriptor'] is False
assert ledger_append['row']['gpu_uuid']=='GPU-fed69954-5a71-dbb2-ee91-544e0eb6b8a3'
scope=load(U/'tool_descendant_scope_terminal.json')
assert scope['terminal']['actual_echild'] is True and scope['terminal']['all_pinned_handles_exited'] is True
assert scope['terminal']['all_actual_tool_roots_waited'] is True and scope['all_tool_descendants_exited_before_descriptor_release'] is True
assert scope['within_diagnostic_budget'] is True and scope['read_errors']==[] and scope['interventions']==[]
assert all(row['raw'].strip()=='' for row in scope['terminal']['fresh_own_child_lists'])
assert all(type(row['returncode']) is int and row['returncode']==0 for row in scope['actual_root_waits'].values())
custody=load(U/'producer_lock_before_release.json')
direct.validate_descriptor(plan,custody,intent['producer'])
assert custody['producer_after']['observed_at_seconds']<=scope['standing_descriptor_at_terminal']['producer_before']['observed_at_seconds']
version=load(U/'tool_version_return.json')
assert result['tool_version']==ref(U/'tool_version_return.json')
assert version['returncode']==0 and version['stderr']=='' and version['stdout'].strip()==plan['profile_tool']['version']
contract=load(U/'native_process_target_contract.json')
assert contract['actual_target_observation_only'] is True and contract['qualification'] is False and contract['admission_granted'] is False
role_details={}
for role in ('producer','waiter','capture'):
    terminal=load(U/('native_process_'+role+'_terminal.json'))
    begin=load(U/('native_process_'+role+'_begin.json'))
    assert terminal['begin']==ref(U/('native_process_'+role+'_begin.json'))
    assert terminal['provider']['calls'][:3]==begin['provider']['calls']
    assert len(begin['provider']['calls'])==3 and terminal['provider']['open_handle_count']==0
    assert all(not (c['name']=='pidfd_send_signal' and c['arguments']['signal'] in (15,9)) for c in terminal['provider']['calls'])
    role_details[role]={'pid':terminal['owner']['pid'],'birth':terminal['owner']['process_start_time'],
        'calls':len(terminal['provider']['calls']),'self_preflight':True,'remaining_open_handles':0,
        'python_version':terminal['provider']['bootstrap']['sys_version'],
        'native_call_failure':terminal['provider']['native_call_failure']}
assert all(sha(Path(path).read_bytes())==expected for path,expected in before.items())
receipt={'schema_version':1,'kind':'independent_actual_direct_native_process_component_review',
    'reviewer':'/root/implement_direct_h100_profiles/review_direct_profile_source','approved':True,'passed':True,
    'review_scope':'Actual zero-environment native process-control diagnostic only; not numerical/current-CUDA/formal-profile qualification',
    'actual_cpu':{'job_id':'6783003','step_id':STEP,'pid':os.getpid(),'cgroup':Path('/proc/self/cgroup').read_text()},
    'elapsed_seconds':time.monotonic()-START,'packet':str(P),'unit':str(U),'plan':ref(U/'measurement_plan.json'),
    'result':ref(U/'result.json'),'source_commit':plan['helpers']['source_commit'],'all_seven_helper_hashes':plan['helpers']['files'],
    'exact_unchanged_staged_consumer':'direct_profile_process.validate_saved','saved_consumer_matches_actual_producer_proof':True,
    'verified_process_proof':proof,'native_roles':role_details,'actual_launch_return':ref(P/'root_actual_target_launch_return.json'),
    'actual_controller_wait':ref(U/'controller_wait.json'),
    'all_native_role_failures_absent':True,'normal_tool_and_payload_waits':True,'all_tool_children_ECHILD_and_empty_lists':True,
    'typed_subreaper_restore_verified':True,'single_producer_FLOCK_custody_across_waits_and_closure':True,
    'exact_release_ack_precedes_payload':True,'same_origin_240_work_60_cleanup_300_total_budget_verified':True,
    'actual_contract_CPU_executable_libc_header_bytes_verified_as_data':True,'actual_contract_observation_only':True,
    'raw_sources_and_metadata_pre_post_unchanged':before,
    'no_old_failed_unit_reclassification':True,'numerical_imports':sorted(FORBIDDEN.intersection(sys.modules)),
    'actual_native_syscalls_in_review':0,'GPU_library_loads_or_device_ops':0,'native_library_symbol_lookups_in_review':0,
    'tool_or_remote_operations_in_review':0,'actual_live_PID_queries_in_review':0,
    'physical_mapping_verified':False,'kernel_device_enforcement_verified':False,
    'current_CUDA_context_qualified':False,'numerical_or_dtype_qualification':False,
    'formal_profile_admission':False,'worker_admission':False,'all_scientific_case_idea_timing_packing_final_credit':0,
    'whole_tool_native_or_device_activity_quantified':False,
    'next_step':'Separate reviewed placement/profile/source admission is still required; no ordinary profile enabled by this diagnostic.'}
out=P/('independent_actual_native_success_review_'+STEP+'.json')
with out.open('xb') as stream:
    stream.write((json.dumps(receipt,indent=2,sort_keys=True)+'\n').encode());stream.flush();os.fsync(stream.fileno())
print(json.dumps({'receipt':ref(out),'passed':True,'CPU_step':'6783003.'+STEP,'elapsed_seconds':receipt['elapsed_seconds'],
    'native_roles':role_details,'formal_profile_admission':False},sort_keys=True))

