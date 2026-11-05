"""Prove the failed publication preflight made no native release, then check the complete publication read-only."""
import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

P=Path(__file__).resolve().parent;R=P.parent.parent;U=R/'data'/P.name;C=R/'data'/(P.name+'_controller')
JOB='6826574';OLD='15af4d1f7b24627a126d969a678e500ce6970be4';COMPLETE='4fcaf5c1cf92e31e5efc33b61f5f55f0da8c0746'
CHECKS=[]
def need(value,message):
    """Preserve each readiness fact only after its actual record and source predicate passes."""
    if not value:raise ValueError(message)
    CHECKS.append(message)

def main():
    """Perform local Git and bounded owned-file reads only; never invoke an action entrypoint."""
    started=time.monotonic();job=os.environ.get('SLURM_JOB_ID');step=os.environ.get('SLURM_STEP_ID')
    need(job=='6825983' and step and step.isdecimal() and '/job_'+job+'/step_'+step+'/' in Path('/proc/self/cgroup').read_text(),
        'exact current owned CPU metadata allocation and step')
    def audit(event,args):
        if event.startswith('ctypes.') or event in ('socket.connect','socket.bind','os.chmod'):raise ValueError('native/network/permission action refused')
        if event=='import' and args[0].split('.')[0] in {'jax','jaxlib','numpy','flax','optax','torch','jaxtari','cupy','pynvml'}:raise ValueError('numerical/device import refused')
        if event=='subprocess.Popen':
            command=args[1]
            if not isinstance(command,(list,tuple)) or command[0]!='git':raise ValueError('only read-only local Git commands are allowed')
            if time.monotonic()-started>265:raise ValueError('whole metadata review deadline reached before another Git read')
    sys.addaudithook(audit)
    source=P/'submit_profile_qualification_held_once.py'
    spec=importlib.util.spec_from_file_location('independent_first_release_readiness_metadata',source)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    need(m.reference(source)['sha256']=='5a12373711096753ed9f80eb2c9e49c8eceb220b5e96083ca32103ddc6b7f85b',
        'the actual unchanged approved action source is exact')
    def forbidden(*args,**kwargs):raise ValueError('scheduler, query, release, submission or reservation action is forbidden in this review')
    m.scheduler=forbidden;m.query_held=forbidden;m.reservation_observation=forbidden
    plan,unused=m.checked_packet();binding_ref=m.reference(P/'release_binding.json');binding=m.document(P/'release_binding.json')
    need(binding_ref['sha256']=='4b191f70e332b26c9c0cfa4ef8a4abfe9d07d1f0a96b7c8da692984a00c53012'
        and binding['job_id']==JOB and binding['command']==['scontrol','release',JOB] and binding['actual_release_performed'] is False,
        'the same immutable fixed-ID release binding remains unexecuted')
    prior_ref=m.reference(P/'release_independent_review.json');prior=m.document(P/'release_independent_review.json')
    need(prior_ref['sha256']=='f6139711425ef217eddbc6d03a8d6b13f85cc2dc3ede2bbcc73e322d24ad45c2'
        and prior['approved'] is prior['passed'] is prior['independent_of_producer'] is True
        and prior['binding']==binding_ref,'the original genuine release review is unchanged')
    failed_names=('root_actual_release_intent.json','root_actual_release_stdout.raw','root_actual_release_stderr.raw','root_actual_release_return.json')
    failed_refs={name:m.reference(P/name) for name in failed_names}
    intent=m.document(P/failed_names[0]);returned=m.document(P/failed_names[3]);stderr=m.ordinary_bytes(P/failed_names[2]).decode()
    need(intent['command']==['/usr/bin/python3','-I','-S','-B',str(source),'--release-once','--publication-commit',OLD]
        and intent['automatic_retry_permitted'] is False and returned['returncode']==1
        and m.ordinary_bytes(P/failed_names[1])==b'',
        'the preserved high-level call failed and is not retagged as successful')
    need('line 557, in release_once' in stderr and 'check_publication(commit,release=True)' in stderr
        and 'line 409, in check_publication' in stderr and 'returned non-zero exit status 128' in stderr
        and str(Path(binding['ledger']['snapshot']['path']).relative_to(m.INTEGRATION)) in stderr,
        'the actual traceback stopped at the missing published unit identifier snapshot')
    tree=ast.parse(m.ordinary_bytes(source));function=next(node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name=='release_once')
    calls=[node for node in ast.walk(function) if isinstance(node,ast.Call) and isinstance(node.func,ast.Name)]
    line=lambda name:min(node.lineno for node in calls if node.func.id==name)
    need(line('check_publication')<line('query_held')<line('reservation_observation')<line('publish')<line('scheduler')
        and not any(isinstance(node,ast.Try) for node in ast.walk(function)),
        'unchanged source cannot reach any native query, action marker or native release after this publication exception')
    absent=[P/name for name in ('release_intent.json','release_receipt.json','release_completed.json',
        'actual_release_return.json','actual_release_stdout.raw','actual_release_stderr.raw',
        'release_final_query_return.json','release_final_query_stdout.raw','release_final_query_stderr.raw',
        'release_reservation_return.json','release_reservation_stdout.raw','release_reservation_stderr.raw')]
    absent+=[U/'owned_allocation_inputs.json']+[C/name for name in ('controller_intent.json','controller_spawn.json','controller_wait.json','controller_failure.json','controller_inconclusive.json')]
    need(all(not os.path.lexists(path) for path in absent),'all canonical query/release/envelope/controller action markers remain absent')
    actual=m.owned_submission(plan)
    need(actual['job_id']==JOB and actual['submitted_held'] is True and all(binding[key]==value for key,value in m.snapshot_binding(JOB,plan).items()),
        'same held submission, real unit snapshots and exact original row are unchanged')
    snapshots={key:binding[key]['snapshot'] for key in ('ledger','submissions')};missing=[]
    for key,reference in snapshots.items():
        need(m.reference(Path(reference['path']))==reference,'actual original fsynced unit snapshot is unchanged: '+key)
        relative=str(Path(reference['path']).relative_to(m.INTEGRATION))
        try:m.git('cat-file','blob',OLD+':'+relative)
        except subprocess.CalledProcessError as error:
            need(error.returncode==128,'old publication snapshot absence has its actual Git return');missing.append(reference)
        else:raise ValueError('the recorded incomplete publication unexpectedly contains this unit snapshot')
        need(hashlib.sha256(m.git('cat-file','blob',COMPLETE+':'+relative)).hexdigest()==reference['sha256'],
            'complete publication contains the exact unchanged unit snapshot: '+key)
    need(len(missing)==2,'both previously omitted unit snapshot files are now explicitly accounted for')
    # This is the real unchanged publication gate; action/query functions remain forbidden.
    m.check_publication(COMPLETE,release=True)
    need(True,'unchanged check_publication with release=True passes for the actual complete publication')
    protection=m.document(P/'metadata_permission_protection.json')
    for row in protection['actual_files_after']:
        status=Path(row['path']).lstat()
        need(status.st_uid==2618919 and status.st_mode&0o777==0o640
            and m.reference(Path(row['path']))['sha256']==row['sha256'],
            'exact owned metadata file remains protected with unchanged bytes')
    need(all(m.reference(P/name)==reference for name,reference in failed_refs.items()),'all original failed root records remain unchanged')
    need(m.reference(P/'release_binding.json')==binding_ref and m.reference(P/'release_independent_review.json')==prior_ref
        and all(not os.path.lexists(path) for path in absent),'read-only completion check made no action and changed no binding/review')
    need(time.monotonic()-started<285,'whole independent readiness check is below five minutes')
    record={'kind':'independent_first_native_profile_qualification_release_readiness_review','approved':True,'passed':True,
        'independent_of_producer':True,'job_id':JOB,'binding':binding_ref,'original_release_review':prior_ref,
        'incomplete_publication':OLD,'complete_publication':COMPLETE,'original_failed_root_records':failed_refs,
        'previously_omitted_unit_snapshots':missing,'unchanged_publication_check_release_true_passed':True,
        'failed_high_level_preflight_stays_failed':True,'previous_native_queries_or_release_attempts':0,
        'canonical_action_markers_absent':[str(path) for path in absent],
        'first_native_release_ready':True,'fixed_native_command':['scontrol','release',JOB],
        'approval_scope':'The first actual native release may run only through the unchanged utility after publication of this review and its complete file closure; its own fresh held query and reservation check remain mandatory.',
        'automatic_retry_authorized':False,'resubmit_rebind_or_binding_rewrite_authorized':False,
        'reviewer_scheduler_queries_or_actions':0,'permission_changes':0,'actual_native_release_performed':False,
        'actual_cpu_job_id':job,'actual_cpu_step_id':step,'checks':CHECKS,'check_count':len(CHECKS),
        'seconds':time.monotonic()-started,'scientific_credit':0,'target_qualification':False,'admission_granted':False}
    reference=m.publish(P/'independent_first_native_release_readiness_review.json',record)
    print(json.dumps({'passed':True,'review':reference,'step_id':step,'checks':len(CHECKS),'seconds':record['seconds']}))

if __name__=='__main__':
    try:main()
    except BaseException as error:
        path=P/'independent_first_native_release_readiness_failure.json'
        with path.open('x') as stream:json.dump({'passed':False,'error':repr(error),'traceback':traceback.format_exc(),
            'actual_cpu_job_id':os.environ.get('SLURM_JOB_ID'),'actual_cpu_step_id':os.environ.get('SLURM_STEP_ID'),'scheduler_actions':0},stream,indent=2)
        raise
