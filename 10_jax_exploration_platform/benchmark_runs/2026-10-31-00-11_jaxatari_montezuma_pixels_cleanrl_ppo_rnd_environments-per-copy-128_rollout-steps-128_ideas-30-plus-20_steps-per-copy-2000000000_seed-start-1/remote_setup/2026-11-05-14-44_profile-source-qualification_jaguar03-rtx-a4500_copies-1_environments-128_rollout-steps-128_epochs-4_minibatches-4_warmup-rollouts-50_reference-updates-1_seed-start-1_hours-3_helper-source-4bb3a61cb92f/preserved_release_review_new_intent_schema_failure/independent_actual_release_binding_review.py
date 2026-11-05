"""Review one actual held A4500 release binding using recorded local evidence only."""
import datetime
import importlib.util
import json
import os
from pathlib import Path
import re
import shlex
import sys
import time

P=Path(__file__).resolve().parent
R=P.parent.parent
U=R/'data'/P.name
JOB='6826574'
BINDING='4b191f70e332b26c9c0cfa4ef8a4abfe9d07d1f0a96b7c8da692984a00c53012'


def main():
    """Verify exact ownership, resources and chronology without another scheduler query."""
    started=time.monotonic();checks=[]
    def need(value,message):
        if not value:raise ValueError(message)
        checks.append(message)
    need(os.environ.get('SLURM_JOB_ID')=='6825983' and os.environ.get('SLURM_STEP_ID','').isdecimal(),
        'actual current owned CPU allocation and numeric step')
    def guard(event,args):
        if event.startswith('ctypes.') or event in ('subprocess.Popen','socket.connect','socket.bind'):
            raise RuntimeError('scheduler/native-library/network action refused')
        if event=='import' and args[0].split('.')[0] in {'jax','jaxlib','numpy','flax','optax','torch','jaxtari','pynvml'}:
            raise RuntimeError('numerical/device import refused')
    sys.addaudithook(guard)
    path=P/'submit_profile_qualification_held_once.py'
    spec=importlib.util.spec_from_file_location('actual_fixed_aa_release_review',path)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    need(m.reference(path)['sha256']=='5a12373711096753ed9f80eb2c9e49c8eceb220b5e96083ca32103ddc6b7f85b',
        'unchanged independently approved release utility source')
    def forbidden(*args,**kwargs):raise RuntimeError('no query, publication check or scheduler call in this review')
    m.scheduler=forbidden;m.check_publication=forbidden;m.git=forbidden
    m.new_path(P/'release_independent_review.json')
    binding_ref=m.reference(P/'release_binding.json');binding=m.document(P/'release_binding.json')
    need(binding_ref['sha256']==BINDING and binding['job_id']==JOB and binding['command']==['scontrol','release',JOB]
        and binding['actual_release_performed'] is False,'the actual fixed-ID release binding is exact and unexecuted')
    plan=m.document(U/'measurement_plan.json')
    submission=m.owned_submission(plan)
    need(submission['job_id']==JOB and submission['submitted_held'] is True
        and submission['publication']=='6bcc60b4115cc696dfe4d03ed3fa8bbc78cced62',
        'actual submitted identifier and prior published source are exact')
    need(binding['submission']==m.reference(P/'submission_receipt.json') and binding['plan']==m.reference(U/'measurement_plan.json'),
        'binding joins the actual successful receipt and immutable capture plan')
    need(all(binding[key]==value for key,value in m.snapshot_binding(JOB,plan).items()),
        'actual owned ledger snapshots and fixed source paths are exact')
    synced=m.document(P/'submission_ledgers_fsynced.json')
    need(synced['kind']=='actual_profile_target_qualification_three_ledgers_fsynced'
        and synced['job_id']==JOB and synced['submission_row']==submission['submission_row']
        and [row['path'] for row in synced['ledgers']]==[str(path) for path in m.ledgers()],
        'all three actual campaign ledger references are present in their fsync receipt')
    for row in synced['ledgers']:
        need(m.reference(Path(row['path']))==row,'actual ledger bytes still match the fsynced submission record: '+row['path'])
    actual=m.document(P/'actual_submission_return.json')
    output=m.ordinary_bytes(P/'actual_submission_stdout.raw').decode().strip()
    need(re.fullmatch(JOB+r'(?:;[A-Za-z0-9_.-]+)?',output) is not None and actual['command']==submission['held_command']
        and type(actual['returncode']) is int and actual['returncode']==0 and actual['automatic_retry_permitted'] is False,
        'actual returned native identifier, command and successful no-retry return agree')
    query={key:m.checked(value) if key=='return' else m.ordinary_bytes(Path(value['path']))
        for key,value in binding['held_query'].items()}
    for value in binding['held_query'].values():need(m.reference(Path(value['path']))==value,'exact cached held-query reference')
    need(query['return']['command']==['scontrol','show','job','-o',JOB]
        and type(query['return']['returncode']) is int and query['return']['returncode']==0 and query['stderr']==b'',
        'cached query addresses only the exact owned held identifier')
    fields=m.native_fields(query['stdout'])
    expected={'JobId':JOB,'UserId':'sl5nw(2618919)','JobState':'PENDING','Reason':'JobHeldUser','Requeue':'0','Restarts':'0',
        'RunTime':'00:00:00','TimeLimit':'03:00:00','Partition':'gpu','Account':'sl5nw_base','QOS':'csresnolim',
        'ReqNodeList':'jaguar03','NumTasks':'1','CPUs/Task':'8','Reservation':'sl5nw_168','Command':str(P/'job.sbatch')}
    need(all(fields.get(key)==value for key,value in expected.items()) and fields['NumNodes'] in ('1','1-1')
        and fields['NumCPUs']=='8' and fields['TresPerNode']=='gres/gpu:nvidia_rtx_a4500:1',
        'actual held owner, state and exact requested one-A4500/eight-CPU/three-hour shape agree')
    pairs=[row.split('=',1) for row in fields['ReqTRES'].split(',')]
    need(all(len(row)==2 for row in pairs) and len({row[0] for row in pairs})==len(pairs),'requested-resource fields are unambiguous')
    resources=dict(pairs)
    need(all(resources.get(key)==value for key,value in {'cpu':'8','mem':'64G','node':'1','gres/gpu':'1'}.items()),
        'actual requested TRES preserve eight CPUs,64GiB,one node and one GPU')
    utc=lambda value:datetime.datetime.fromisoformat(value)
    need(utc(actual['finished_at_utc'])<=utc(synced['at_utc'])<=utc(submission['at_utc'])
        <=utc(query['return']['started_at_utc'])<=utc(query['return']['finished_at_utc'])<=utc(binding['at_utc']),
        'actual successful submission precedes all-ledger fsync, receipt, held query and release binding')
    reservation=submission['reservation_observation']
    reserve={key:m.checked(value) if key=='return' else m.ordinary_bytes(Path(value['path'])) for key,value in reservation.items()}
    for value in reservation.values():need(m.reference(Path(value['path']))==value,'exact recorded UTC reservation evidence')
    reservation_fields=m.native_fields(reserve['stdout'])
    need(reserve['return']['command']==['scontrol','show','reservation','sl5nw_168','-o'] and reserve['return']['returncode']==0
        and reserve['stderr']==b'' and reservation_fields['ReservationName']=='sl5nw_168'
        and reservation_fields['State']=='ACTIVE' and 'jaguar03' in reservation_fields['Nodes'].split(','),
        'actual submission observed the correct active reservation and node')
    as_utc=lambda value:datetime.datetime.fromisoformat(value).replace(tzinfo=datetime.timezone.utc)
    need(as_utc(reservation_fields['StartTime'])<=utc(actual['started_at_utc'])
        and as_utc(reservation_fields['EndTime'])>=utc(actual['started_at_utc'])+datetime.timedelta(seconds=10800),
        'the actual recorded UTC reservation covered the complete requested wall time')
    need(shlex.split(m.ordinary_bytes(R/'slurm/jobs'/JOB/'command.txt').decode())==submission['held_command'],
        'actual per-job command records the real held submission argv')
    need(m.reference(P/'prepared_launch_binding.json')['sha256']=='b54c95787eb0e5728ed000485ecb27c493eefabc30d2f9337803734c541c7837'
        and m.reference(P/'launcher_independent_review.json')['sha256']=='76d61309606eb4b3958e22f90e1b9fd199e8f27314dc33e0c3a4c876428b0a70',
        'prior actual prepared binding and independent review remain unchanged')
    need(plan['helpers']['source_commit']=='4bb3a61cb92fe48ab5ba68fb52f70b47736a1174'
        and len(plan['helpers']['files'])==36
        and '10_jax_exploration_platform/scripts/atari_pixel_campaign/profile_target_tool_scope.py' in plan['helpers']['files']
        and m.document(P/'launcher_independent_review.json')['publication_freeze_count']==64,
        'the exact already reviewed new runtime and complete helper closure remain bound')
    need(not any(os.path.lexists(P/name) for name in ('release_intent.json','release_receipt.json','release_completed.json'))
        and not os.path.lexists(U/'owned_allocation_inputs.json'),'no release or final ownership envelope has been attempted')
    protection_ref=m.reference(P/'metadata_permission_protection.json')
    protection=m.document(P/'metadata_permission_protection.json')
    prepared=m.document(P/'prepared_launch_binding.json')
    need(protection_ref['sha256']=='038d60789be4cbc7e7c7758ba72ef2e4bb57ab780924409ca261767f2560345f'
        and protection['completed'] is True and protection['hostname']=='portal05'
        and len(protection['controllers'])==1 and protection['controllers'][0]['controller_plan']==prepared['parent_plan']
        and protection['controllers'][0]['child_plan']==binding['plan'],'actual root protection is complete and bound to this fresh controller')
    protected=protection['actual_files_after']
    need({row['path'] for row in protected}=={str(R/'queue/metadata.lock'),str(R/'processes.lock')}
        and all(row['mode']==0o640 and row['uid']==2618919 and row['nlink']==1 for row in protected),
        'root protected exactly the two owned metadata files to0640')
    for row in protected:
        status=Path(row['path']).lstat()
        need(status.st_uid==2618919 and status.st_mode&0o777==0o640
            and m.reference(Path(row['path']))['sha256']==row['sha256'],
            'current CPU observation confirms protected mode and unchanged bytes without cross-host device identity assertion')
    preflight_return=m.document(P/'root_actual_protected_metadata_preflight_return.json')
    preflight_intent=m.document(P/'root_actual_protected_metadata_preflight_intent.json')
    preflight=m.document(P/'root_actual_protected_metadata_preflight_stdout.raw')
    need(preflight_return['returncode']==0 and preflight['passed'] is True and preflight['action_performed'] is False
        and preflight['controller_count']==1 and preflight['files']==protected
        and m.ordinary_bytes(P/'root_actual_protected_metadata_preflight_stderr.raw')==b''
        and preflight_intent['one_attempt'] is True
        and utc(protection['at_utc'])<=utc(preflight_intent['started_at_utc'])
        <=utc(preflight_return['finished_at_utc'])<=utc(actual['started_at_utc']),
        'actual successful protected preflight preceded held submission')
    need(time.monotonic()-started<285,'whole fixed-ID metadata review stays below five minutes')
    record={'schema_version':1,'kind':'independent_profile_target_qualification_release_review',
        'approved':True,'passed':True,'independent_of_producer':True,'binding':binding_ref,'job_id':JOB,
        'submission':binding['submission'],'plan':binding['plan'],'held_query':binding['held_query'],
        'submission_ledgers_fsynced':m.reference(P/'submission_ledgers_fsynced.json'),
        'metadata_permission_protection':protection_ref,'protected_metadata_preflight_return':m.reference(P/'root_actual_protected_metadata_preflight_return.json'),
        'protected_metadata_preflight_stdout':m.reference(P/'root_actual_protected_metadata_preflight_stdout.raw'),
        'protected_metadata_preflight_intent':m.reference(P/'root_actual_protected_metadata_preflight_intent.json'),
        'cross_host_device_identity_asserted':False,'fixed_command':['scontrol','release',JOB],'checks':checks,'check_count':len(checks),
        'actual_cpu_job_id':os.environ['SLURM_JOB_ID'],'actual_cpu_step_id':os.environ['SLURM_STEP_ID'],
        'actual_target_gpu_uuid':None,'seconds':time.monotonic()-started,'cached_records_only':True,
        'new_scheduler_queries':0,'release_performed_by_reviewer':False,
        'fresh_final_held_query_and_reservation_required_before_root_release':True,
        'GPU_numerical_qualification':False,'current_physical_identity_verified':False,'unresolved_anomalies':[],**m.ZERO}
    receipt=m.publish(P/'release_independent_review.json',record)
    print(json.dumps({'passed':True,'review':receipt,'actual_cpu_step_id':os.environ['SLURM_STEP_ID'],
        'binding':binding_ref,'checks':len(checks),'seconds':record['seconds']}))


if __name__=='__main__':main()
