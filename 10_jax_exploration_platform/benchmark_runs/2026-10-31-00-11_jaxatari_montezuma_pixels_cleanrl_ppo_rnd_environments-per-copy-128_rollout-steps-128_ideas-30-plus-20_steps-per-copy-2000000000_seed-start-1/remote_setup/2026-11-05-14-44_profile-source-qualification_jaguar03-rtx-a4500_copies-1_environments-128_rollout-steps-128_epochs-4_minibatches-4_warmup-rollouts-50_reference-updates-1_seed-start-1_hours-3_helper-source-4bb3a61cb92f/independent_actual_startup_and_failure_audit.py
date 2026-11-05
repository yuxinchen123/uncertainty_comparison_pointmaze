"""Audit actual startup/release/restoration and preserve the later failed process scope as failed."""
import datetime
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import sys
import time

P=Path(__file__).resolve().parent
R=P.parent.parent
U=R/'data'/P.name
C=R/'data'/(P.name+'_controller')


def main():
    """Read only saved owned metadata, without source admission, native tools or live PID queries."""
    started=time.monotonic();checks=[];bootstrap=[];sealed=False
    def hook(event,args):
        if event.startswith('ctypes.'):
            if sealed:raise RuntimeError('late native operation refused')
            bootstrap.append({'event':event,'own_process_only':bool(args and args[0] is None)})
        if sealed and event in ('subprocess.Popen','socket.connect','socket.bind','os.chmod'):
            raise RuntimeError('process/network/permission action refused')
        if sealed and event=='import' and args[0].split('.')[0] in {'jax','jaxlib','numpy','flax','optax','torch','jaxtari','pynvml'}:
            raise RuntimeError('numerical/device import refused')
    sys.addaudithook(hook)
    import ctypes
    sealed=True
    spec=importlib.util.spec_from_file_location('actual_startup_metadata_only',P/'submit_profile_qualification_held_once.py')
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    def need(value,message):
        if not value:raise ValueError(message)
        checks.append(message)
    need(os.environ.get('SLURM_JOB_ID')=='6825983' and os.environ.get('SLURM_STEP_ID','').isdecimal(),
        'actual current owned CPU metadata allocation and numeric step')
    plan=m.document(U/'measurement_plan.json');parent=m.document(C/'measurement_plan.json')
    binding=m.document(C/'prepared_child_binding.json');spawn=m.document(C/'controller_spawn.json')
    startup=m.document(U/'profile_target_child_startup_observations.json')
    release=m.document(U/'profile_target_controller_release.json')
    need(m.reference(U/'code/uva_slurm_configured_isolation.py')['sha256']==plan['helpers']['files'][m.PREFIX+'uva_slurm_configured_isolation.py'],
        'exact actual staged owned-input reader source')
    spec=importlib.util.spec_from_file_location('actual_new_qualification_owned_reader',U/'code/uva_slurm_configured_isolation.py')
    owned=importlib.util.module_from_spec(spec);spec.loader.exec_module(owned)
    envelope=owned.owned_inputs(plan)
    need(envelope['job_id']=='6826574' and envelope['plan']==binding['child_plan'],'unchanged actual staged owned-input consumer accepts the real envelope')
    receipt=m.document(P/'release_receipt.json');returned=m.document(P/'actual_release_return.json');completed=m.document(P/'release_completed.json')
    native_intent=m.document(P/'release_intent.json')
    first=m.document(P/'root_actual_first_native_release_after_publication_proof_intent.json')
    first_return=m.document(P/'root_actual_first_native_release_after_publication_proof_return.json')
    failed=m.document(P/'root_actual_release_return.json')
    need(failed['returncode']==1 and first_return['returncode']==0
        and first['kind']=='first_native_release_after_failed_publication_preflight'
        and first['previous_failed_preflight_preserved'] is True and first['automatic_retry_permitted'] is False
        and first['independent_readiness_review']==m.reference(P/'independent_first_native_release_readiness_review.json')
        and native_intent['job_id']=='6826574' and native_intent['automatic_retry_permitted'] is False
        and receipt['publication']==native_intent['publication']=='1f8155f8934ff74804d7e605c71c13fbfdf60497'
        and completed['released_once'] is True,
        'one actual native release followed the separately preserved failed publication preflight')
    need(plan['helpers']['source_commit']=='4bb3a61cb92fe48ab5ba68fb52f70b47736a1174'
        and len(plan['helpers']['files'])==36,'actual new qualification uses the exact reviewed current runtime and thirty-six helpers')
    actual=m.owned_submission(plan)
    need(actual['job_id']=='6826574' and actual['submission_row']==envelope['submission_row']
        and all(value==envelope[key] for key,value in m.snapshot_binding('6826574',plan).items()),
        'the actual release uses exactly one unchanged owned job/row and its immutable unit snapshots')
    need(receipt['command']==returned['command']==['scontrol','release','6826574'] and receipt['returncode']==returned['returncode']==0
        and completed['release']==m.reference(P/'release_receipt.json') and completed['owned_allocation_inputs']==m.reference(U/'owned_allocation_inputs.json'),
        'actual successful one-shot release is joined to its atomic final envelope')
    utc=lambda value:datetime.datetime.fromisoformat(value)
    need(utc(returned['finished_at_utc'])<=utc(receipt['at_utc'])<=utc(completed['at_utc'])
        <=utc(spawn['controller']['observed_at_utc'])<=utc(spawn['child']['observed_at_utc']),
        'release return, durable receipt, final envelope and startup observations have the actual order')
    need(spawn['actual_popen_pid']==2511708 and spawn['child_plan']==binding['child_plan']
        and spawn['command']==binding['command'] and release['controller_spawn']==m.reference(C/'controller_spawn.json')
        and release['child']==spawn['child'] and release['controller']==spawn['controller']
        and release['parent_plan']==binding['controller_plan'] and release['child_plan']==binding['child_plan'],
        'actual exact child Popen identity is bound before the controller release')
    for identity,expected in ((spawn['controller'],['/usr/bin/python3','-I','-S','-B',str(C/'code/profile_target_qualification.py'),'--prepared-unit',str(C)]),
            (spawn['child'],binding['command'])):
        raw=bytes.fromhex(identity['command_bytes_hex'])
        need(raw==b'\0'.join(value.encode() for value in expected)+b'\0'
            and hashlib.sha256(raw).hexdigest()==identity['command_sha256'] and identity['command']==expected,
            'exact actual raw null-terminated command and digest')
        uid=identity['uid_bytes'].encode()
        need([int(value) for value in identity['uid_bytes'].split()[1:]]==[2618919]*4
            and hashlib.sha256(uid).hexdigest()==identity['uid_sha256'] and identity['hostname']=='jaguar03',
            'all four actual process user identifiers and host are exact')
        stat=identity['stat'];fields=stat[stat.rfind(')')+2:].split()
        need(int(stat.split(' ',1)[0])==identity['pid'] and fields[19]==identity['process_start_time']
            and int(fields[1])==identity['parent_pid'] and hashlib.sha256(stat.encode()).hexdigest()==identity['stat_sha256'],
            'saved actual PID/birth/parent/stat bytes agree')
    need(spawn['child']['parent_pid']==spawn['controller']['pid']==2511706
        and spawn['child']['affinity']==startup['selected_affinity']==spawn['controller']['affinity']
        and len(startup['selected_affinity'])==8,'actual child parent and eight-CPU affinity agree')
    need(startup['exact_command_ready'] is True and startup['failure'] is None and startup['actual_popen_pid']==2511708
        and startup['observed_birth']==spawn['child']['process_start_time'] and startup['expected_parent_pid']==2511706
        and startup['finished_monotonic']<=startup['deadline_monotonic']==startup['started_monotonic']+30,
        'actual bounded startup completed without weakening exact identity')
    for row in startup['observations']:
        need(all(value is True for value in row['predicates'].values()) and row['expected_raw_command'] is True
            and row['actual_popen_returncode'] is None,'actual startup observations retained every original identity predicate')
    protection=m.document(P/'metadata_permission_protection.json');restored=m.document(P/'metadata_permission_restoration.json')
    need(m.reference(P/'metadata_permission_restoration.json')['sha256']=='97d930800d7d2d85e7470cc47c8ae68ae39c9e5680c925b4030cb5029d33ebed'
        and restored['completed'] is True and restored['hostname']==protection['hostname']=='portal05'
        and restored['original_files']==protection['original_files'],'actual root permission restoration uses its same-host original snapshots')
    for original,protected,after in zip(protection['original_files'],protection['actual_files_after'],restored['actual_files_after']):
        keys=('path','hostname','device','inode','uid','gid','nlink','bytes','mtime_ns','sha256')
        need(all(original[key]==protected[key]==after[key] for key in keys)
            and original['mode']==after['mode']==0o660 and protected['mode']==0o640,
            'root restored exact same file identity and bytes from0640 to original0660')
    evidence=restored['restoration_evidence']
    need(len(evidence)==1 and evidence[0]['mode']=='child_released'
        and evidence[0]['spawn']==m.reference(C/'controller_spawn.json')
        and evidence[0]['startup']==m.reference(U/'profile_target_child_startup_observations.json')
        and evidence[0]['release']==m.reference(U/'profile_target_controller_release.json'),
        'permission restoration consumed these actual durable startup/release records')
    failure=m.document(U/'failure.json');progress=m.document(U/'progress.json');wait=m.document(C/'controller_wait.json')
    need(progress['state']=='failed' and wait['waited_pid']==2511708 and wait['returncode']==1
        and failure['error']=="FileNotFoundError(2, 'No such file or directory')",
        'the later actual run is failed, not qualified or successful')
    need(utc(progress['time'])<utc(restored['at_utc']),'actual tool-scope failure preceded restoration; later permission restoration did not cause it')
    scope=m.document(U/'tool_descendant_scope_terminal.json')
    terminal=scope['terminal'];need(terminal['actual_echild'] is True and terminal['errno']==10
        and terminal['all_pinned_handles_exited'] is True and terminal['all_actual_tool_roots_waited'] is True,
        'saved cleanup includes actual ECHILD, exited pinned handles and exact root waits')
    for row in terminal['fresh_own_child_lists']:
        need(row['raw'].strip()=='' and hashlib.sha256(row['raw'].encode()).hexdigest()==row['sha256'],
            'actual final own child-list bytes are empty and hashed')
    for pid,row in scope['actual_root_waits'].items():
        need(row==m.document(U/('tool_root_wait_'+pid+'.json')),'actual saved tool Popen wait is unchanged')
    need(scope['actual_root_waits']['2511761']['returncode']==0 and scope['actual_root_waits']['2511765']['returncode']==-15
        and len(scope['interventions'])==6 and scope['read_errors']==[],
        'version returned zero, profile root was terminated during cleanup, and ordinary successful-scope gates must refuse')
    live_failure=m.document(U/'tool_parent_exit_failure_1.json');manifest=m.document(U/'tool_parent_exit_observations.json')
    need(live_failure['outcome']=='unverified_parent_observation' and live_failure['error_type']=='FileNotFoundError'
        and live_failure['parent_known'] is True and live_failure['retained_handle_exited'] is False
        and live_failure['parent_pid']==2511841 and live_failure['child_list_sample_valid'] is False
        and "'2511842'" in live_failure['error'] and str(live_failure['parent_pid']) in scope['positive_nodes'],
        'this actual failed attempt records a known live parent and unavailable enumerated task; incomplete child lists were refused')
    node=scope['positive_nodes']['2511841']['initial']
    need(node['uids']==[2618919]*4 and node['cgroup']==spawn['child']['cgroup'] if 'cgroup' in spawn['child'] else node['uids']==[2618919]*4,
        'the recorded failed parent retains its positive own-process identity')
    need(manifest['failure_count']==1 and manifest['discarded_child_list_samples_accepted_as_valid'] is False,
        'the new process scope retains its actual failed sample and grants no successful-scope credit')
    need('line 192, in child_lists' in failure['traceback'] and "os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=task_directory)" in failure['traceback'],
        'the saved traceback identifies task-directory open after enumeration in the unchanged base helper')
    role=m.document(U/'profile_target_process_producer_failure.json')
    need(role['completed'] is False and role['provider']['open_handle_count']==0
        and role['provider']['native_call_failure'] is False,'the failed native role retains actual closed-handle evidence without claiming completion')
    accounting_prefix=P/'root_owned_terminal_accounting_2026-11-05-23-10-13'
    accounting=m.document(str(accounting_prefix)+'_return.json')
    need(accounting['command']==['sacct','-j','6826574','--parsable2',
        '--format=JobIDRaw,State,ExitCode,Start,End,Elapsed,NodeList,AllocTRES','--noheader']
        and accounting['returncode']==0 and m.ordinary_bytes(Path(str(accounting_prefix)+'_stderr.raw'))==b'',
        'saved actual terminal accounting queries only this exact owned qualification identifier')
    rows=[row.split('|') for row in m.ordinary_bytes(Path(str(accounting_prefix)+'_stdout.raw')).decode().splitlines()]
    need({row[0] for row in rows}=={'6826574','6826574.batch','6826574.0'}
        and all(row[1:3]==['FAILED','1:0'] and row[4]=='2026-11-05T23:06:46'
            and row[5]=='00:00:23' and row[6]=='jaguar03' for row in rows),
        'actual job, batch and numerical step ended failed with exact normal accounting records')
    need(not any((U/name).exists() for name in ('result.json','profile_target_qualification_result.json','capture_release.json')),
        'no completed profile qualification or capture-release evidence exists')
    record={'schema_version':2,'kind':'independent_actual_qualification_startup_release_restoration_and_failure_review',
        'approved':True,'passed':True,'scope':'Startup, release/envelope and permission workflow only; failed qualification stays failed',
        'independent_of_producer':True,'job_id':'6826574','controller_pid':2511706,'controller_birth':spawn['controller']['process_start_time'],
        'producer_pid':2511708,'producer_birth':spawn['child']['process_start_time'],'checks':checks,'check_count':len(checks),
        'owned_inputs':m.reference(U/'owned_allocation_inputs.json'),'controller_spawn':m.reference(C/'controller_spawn.json'),
        'startup':m.reference(U/'profile_target_child_startup_observations.json'),'controller_release':m.reference(U/'profile_target_controller_release.json'),
        'permission_restoration':m.reference(P/'metadata_permission_restoration.json'),'failure':m.reference(U/'failure.json'),
        'tool_scope_terminal':m.reference(U/'tool_descendant_scope_terminal.json'),'capture_log':m.reference(U/'capture.log'),
        'run_failed':True,'qualification_passed':False,'native_release_count':1,'failed_publication_preflight_preserved':True,
        'actual_failed_parent_pid':2511841,'actual_failed_parent_birth':node['process_start_time'],
        'actual_unavailable_task_directory_name':'2511842','unavailable_task_birth_known':False,
        'prior_attempt_failure_identity_or_cause_inferred':False,'historical_thread_exit_mechanism_proven':False,
        'actual_failed_scope_sample':m.reference(U/'tool_parent_exit_failure_1.json'),
        'actual_native_provider_failure':m.reference(U/'profile_target_process_producer_failure.json'),
        'terminal_accounting':m.reference(Path(str(accounting_prefix)+'_return.json')),
        'ordinary_saved_scope_would_refuse_interventions_and_nonzero_root_wait':True,
        'native_tool_work_quantified':False,'numerical_completion_proven':False,'actual_cpu_job_id':os.environ['SLURM_JOB_ID'],
        'actual_cpu_step_id':os.environ['SLURM_STEP_ID'],'seconds':time.monotonic()-started,'stdlib_own_process_bootstrap':bootstrap,
        'reviewer_scheduler_queries':0,'reviewer_permission_changes':0,'reviewer_native_or_GPU_operations_after_guard':0,
        'scientific_credit':0,'profile_credit':0,'case_credit':0,'idea_credit':0,'final_training_credit':0,'worker_admitted':False,'admission_granted':False}
    result=m.publish(P/'independent_actual_startup_release_restoration_failure_review.json',record)
    print(json.dumps({'passed':True,'review':result,'actual_cpu_step_id':os.environ['SLURM_STEP_ID'],'checks':len(checks),'seconds':record['seconds'],'qualification_passed':False}))


if __name__=='__main__':main()
