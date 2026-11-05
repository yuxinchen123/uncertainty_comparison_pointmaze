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
    plan=m.document(U/'measurement_plan.json');parent=m.document(C/'measurement_plan.json')
    binding=m.document(C/'prepared_child_binding.json');spawn=m.document(C/'controller_spawn.json')
    startup=m.document(U/'profile_target_child_startup_observations.json')
    release=m.document(U/'profile_target_controller_release.json')
    need(m.reference(U/'code/uva_slurm_configured_isolation.py')['sha256']==plan['helpers']['files'][m.PREFIX+'uva_slurm_configured_isolation.py'],
        'exact actual staged owned-input reader source')
    spec=importlib.util.spec_from_file_location('actual_new_qualification_owned_reader',U/'code/uva_slurm_configured_isolation.py')
    owned=importlib.util.module_from_spec(spec);spec.loader.exec_module(owned)
    envelope=owned.owned_inputs(plan)
    need(envelope['job_id']=='6826372' and envelope['plan']==binding['child_plan'],'unchanged actual staged owned-input consumer accepts the real envelope')
    receipt=m.document(P/'release_receipt.json');returned=m.document(P/'actual_release_return.json');completed=m.document(P/'release_completed.json')
    need(receipt['command']==returned['command']==['scontrol','release','6826372'] and receipt['returncode']==returned['returncode']==0
        and completed['release']==m.reference(P/'release_receipt.json') and completed['owned_allocation_inputs']==m.reference(U/'owned_allocation_inputs.json'),
        'actual successful one-shot release is joined to its atomic final envelope')
    utc=lambda value:datetime.datetime.fromisoformat(value)
    need(utc(returned['finished_at_utc'])<=utc(receipt['at_utc'])<=utc(completed['at_utc'])
        <=utc(spawn['controller']['observed_at_utc'])<=utc(spawn['child']['observed_at_utc']),
        'release return, durable receipt, final envelope and startup observations have the actual order')
    need(spawn['actual_popen_pid']==2503108 and spawn['child_plan']==binding['child_plan']
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
    need(spawn['child']['parent_pid']==spawn['controller']['pid']==2503106
        and spawn['child']['affinity']==startup['selected_affinity']==spawn['controller']['affinity']
        and len(startup['selected_affinity'])==8,'actual child parent and eight-CPU affinity agree')
    need(startup['exact_command_ready'] is True and startup['failure'] is None and startup['actual_popen_pid']==2503108
        and startup['observed_birth']==spawn['child']['process_start_time'] and startup['expected_parent_pid']==2503106
        and startup['finished_monotonic']<=startup['deadline_monotonic']==startup['started_monotonic']+30,
        'actual bounded startup completed without weakening exact identity')
    for row in startup['observations']:
        need(all(value is True for value in row['predicates'].values()) and row['expected_raw_command'] is True
            and row['actual_popen_returncode'] is None,'actual startup observations retained every original identity predicate')
    protection=m.document(P/'metadata_permission_protection.json');restored=m.document(P/'metadata_permission_restoration.json')
    need(m.reference(P/'metadata_permission_restoration.json')['sha256']=='d13e43eecd5c048473c250926fcbadfad989b56563bd743a03713eaeacef71d6'
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
    need(progress['state']=='failed' and wait['waited_pid']==2503108 and wait['returncode']==1
        and failure['error']=="ValueError('the known parent exited before its child lists could be accepted')",
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
    need(scope['actual_root_waits']['2503326']['returncode']==0 and scope['actual_root_waits']['2503336']['returncode']==-15
        and len(scope['interventions'])==7 and scope['read_errors']==[],
        'version returned zero, profile root was terminated during cleanup, and ordinary successful-scope gates must refuse')
    need(not any((U/name).exists() for name in ('result.json','profile_target_qualification_result.json','capture_identity.json','capture_release.json')),
        'no completed profile qualification or capture-release evidence exists')
    record={'schema_version':2,'kind':'independent_actual_qualification_startup_release_restoration_and_failure_review',
        'approved':True,'passed':True,'scope':'Startup, release/envelope and permission workflow only; failed qualification stays failed',
        'independent_of_producer':True,'job_id':'6826372','controller_pid':2503106,'controller_birth':spawn['controller']['process_start_time'],
        'producer_pid':2503108,'producer_birth':spawn['child']['process_start_time'],'checks':checks,'check_count':len(checks),
        'owned_inputs':m.reference(U/'owned_allocation_inputs.json'),'controller_spawn':m.reference(C/'controller_spawn.json'),
        'startup':m.reference(U/'profile_target_child_startup_observations.json'),'controller_release':m.reference(U/'profile_target_controller_release.json'),
        'permission_restoration':m.reference(P/'metadata_permission_restoration.json'),'failure':m.reference(U/'failure.json'),
        'tool_scope_terminal':m.reference(U/'tool_descendant_scope_terminal.json'),'capture_log':m.reference(U/'capture.log'),
        'run_failed':True,'qualification_passed':False,'failed_parent_pid_recorded':False,'historical_race_cause_established':False,
        'source_exit_between_scan_and_parent_identity_is_possible':True,'ordinary_saved_scope_would_refuse_interventions_and_nonzero_root_wait':True,
        'native_tool_work_quantified':False,'numerical_completion_proven':False,'actual_cpu_job_id':os.environ['SLURM_JOB_ID'],
        'actual_cpu_step_id':os.environ['SLURM_STEP_ID'],'seconds':time.monotonic()-started,'stdlib_own_process_bootstrap':bootstrap,
        'reviewer_scheduler_queries':0,'reviewer_permission_changes':0,'reviewer_native_or_GPU_operations_after_guard':0,
        'scientific_credit':0,'profile_credit':0,'case_credit':0,'idea_credit':0,'final_training_credit':0,'worker_admitted':False,'admission_granted':False}
    result=m.publish(P/'independent_actual_startup_release_restoration_failure_review.json',record)
    print(json.dumps({'passed':True,'review':result,'actual_cpu_step_id':os.environ['SLURM_STEP_ID'],'checks':len(checks),'seconds':record['seconds'],'qualification_passed':False}))


if __name__=='__main__':main()
