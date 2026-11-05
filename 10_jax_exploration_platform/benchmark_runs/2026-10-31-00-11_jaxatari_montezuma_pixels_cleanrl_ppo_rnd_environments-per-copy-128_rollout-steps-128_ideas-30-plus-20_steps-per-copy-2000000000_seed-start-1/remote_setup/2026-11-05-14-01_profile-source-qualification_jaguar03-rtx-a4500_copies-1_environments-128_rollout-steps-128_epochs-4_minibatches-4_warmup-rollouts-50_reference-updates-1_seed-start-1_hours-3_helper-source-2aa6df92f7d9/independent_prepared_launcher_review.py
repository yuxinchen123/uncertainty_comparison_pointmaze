"""Review the actual late-bound A4500 launch packet, without any scheduler action."""
import hashlib
import argparse
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--packet',type=Path,required=True)
parser.add_argument('--expected-binding-sha256',required=True)
parser.add_argument('--controller-review-sha256',required=True)
arguments=parser.parse_args()
P=arguments.packet.absolute()
if P.resolve(strict=True)!=P:raise ValueError('supply the exact actual successfully bound packet path')
R=P.parent.parent
U=R/'data'/P.name
C=R/'data'/(P.name+'_controller')
EXPECTED_BINDING=arguments.expected_binding_sha256
EXPECTED_CONTROLLER_REVIEW=arguments.controller_review_sha256
EXPECTED_SOURCE='5a12373711096753ed9f80eb2c9e49c8eceb220b5e96083ca32103ddc6b7f85b'
EXPECTED_TEST='c7e8314e225f7e3bd8fe61c74e757bc2a6c1cebc2559c0afc5c14ee560c42a08'


def main():
    """Use the unchanged prepared reader, then freeze every mandatory actual publication input."""
    started=time.monotonic();checks=[]
    def need(value,message):
        if not value:raise ValueError(message)
        checks.append(message)
    need(os.environ.get('SLURM_JOB_ID')=='6825983' and os.environ.get('SLURM_STEP_ID','').isdecimal(),
        'actual current owned CPU allocation and step')
    def guard(event,args):
        if event.startswith('ctypes.') or event in ('socket.connect','socket.bind'):
            raise RuntimeError('native-library/device/network action refused')
        if event=='import' and args[0].split('.')[0] in {'jax','jaxlib','numpy','flax','optax','torch','jaxtari','pynvml'}:
            raise RuntimeError('numerical/device import refused')
        if event=='subprocess.Popen':
            command=args[1]
            if not isinstance(command,(tuple,list)) or len(command)<5 or command[0]!='git' \
                    or command[3:5]!=['cat-file','blob']:
                raise RuntimeError('only exact local source Git object reads are allowed')
    sys.addaudithook(guard)
    source=P/'submit_profile_qualification_held_once.py'
    specification=importlib.util.spec_from_file_location('independent_actual_aa_launcher',source)
    launcher=importlib.util.module_from_spec(specification);specification.loader.exec_module(launcher)
    need(launcher.reference(source)['sha256']==EXPECTED_SOURCE
        and launcher.reference(P/'test_submit_profile_qualification_held_once.py')['sha256']==EXPECTED_TEST,
        'exact independently approved utility and test bytes')
    def no_scheduler(*args,**kwargs):raise RuntimeError('scheduler action forbidden in independent preparation review')
    launcher.scheduler=no_scheduler
    for name in ('packet_static_verification.json','launcher_source_freeze.json','launcher_independent_review.json'):
        launcher.new_path(P/name)
    binding_ref=launcher.reference(P/'prepared_launch_binding.json')
    need(binding_ref['sha256']==EXPECTED_BINDING,'actual one-shot prepared binding is exact')
    plan,held=launcher.checked_packet()
    binding=launcher.document(P/'prepared_launch_binding.json')
    need(binding['actual_job_id'] is None and binding['actual_release_performed'] is False
        and plan.get('gpu_uuid') is None and plan.get('job_id') is None,'job and physical GPU identities remain unresolved')
    need(binding['source_commit']=='2aa6df92f7d9da20e2fbc1d15a20df16ba10ba1e'
        and len(plan['helpers']['files'])==36 and plan['copy_seeds']==[1]
        and '10_jax_exploration_platform/scripts/atari_pixel_campaign/profile_target_tool_scope.py' in plan['helpers']['files'],
        'exact thirty-six-helper one-copy current source with the new scope')
    need(launcher.same(binding['resources'],launcher.resources('reservation'))
        and held==launcher.held_command(launcher.resources('reservation')),
        'actual one-A4500 eight-CPU sixty-four-GiB three-hour reserved shape')
    prior_ref=launcher.reference(C/'controller_launch_independent_review.json')
    need(prior_ref['sha256']==EXPECTED_CONTROLLER_REVIEW,
        'genuine actual prepared parent/capture review remains unchanged')
    prior=launcher.document(C/'controller_launch_independent_review.json')
    need(prior['approved'] is prior['passed'] is True and prior['controller_plan']==binding['parent_plan']
        and prior['child_plan']==binding['plan'],'actual prepared review joins both exact plans')
    inventory=launcher.document(P/'immutable_artifacts.json')['files']
    need(len(inventory)==59,'complete actual immutable launch inventory has fifty-nine files')
    for folder in (P,C,U):
        for name in launcher.STANDARD:
            path=folder/name
            need(str(path) in inventory and launcher.reference(path)['sha256']==prior['source_and_metadata_inventory'][str(path)]['sha256'],
                'one of twelve existing standard records is unchanged: '+str(path))
    need(plan['configured_profile_prerequisite_inputs']['proof']['sha256']=='e75b78e9a9b3fd4a857c2d7c5148de15c8800ef14398c5b56b13f5c78db4d3c7'
        and plan['configured_profile_prerequisite_inputs']['review']['sha256']=='5e19415c03e8063275422d6651c5496de9e94fc63c3f1d4d12df27e11a71298d'
        and plan['profile_qualification_backend']['expected']['sha256']=='19e752ef9b93999e8bec28a0706947935c31f4229a3b33556597de8f86575412',
        'genuine four-context fact, independent review and fresh backend preparation remain exact')
    need(all(launcher.same(binding.get(key),value) for key,value in launcher.ZERO.items()),'all scientific/worker/device authority remains false or zero')
    static={'schema_version':1,'kind':'actual_profile_target_qualification_packet_static_verification',
        'passed':True,'prepared_binding':binding_ref,'controller_review':prior_ref,
        'immutable_artifact_count':len(inventory),'staged_capture_helpers':36,'parent_helpers':2,
        'checks':list(checks),'actual_job_id':os.environ['SLURM_JOB_ID'],'actual_step_id':os.environ['SLURM_STEP_ID'],
        'scheduler_calls':0,'publication_check_called_before_publication':False,**launcher.ZERO}
    launcher.publish(P/'packet_static_verification.json',static)
    paths=set(inventory)|{str(P/name) for name in ('immutable_artifacts.json','prepared_launch_binding.json',
        'packet_static_verification.json','submit_profile_qualification_held_once.py','test_submit_profile_qualification_held_once.py')}
    frozen={str(path):launcher.reference(Path(path))['sha256'] for path in sorted(paths)}
    need(len(frozen)==64,'all sixty-four mandatory actual inputs are included in the publication freeze')
    source_freeze=launcher.publish(P/'launcher_source_freeze.json',{
        'schema_version':1,'kind':'actual_profile_target_qualification_launcher_source_freeze','files':frozen,
        'prepared_binding':binding_ref,'runtime_source_commit':binding['source_commit'],
        'actual_job_id_unknown':True,'physical_gpu_uuid_unknown':True,**launcher.ZERO})
    need(time.monotonic()-started<285,'whole metadata review remains under five minutes')
    review={'schema_version':1,'kind':'independent_profile_target_qualification_prepared_launcher_review',
        'approved':True,'passed':True,'independent_of_producer':True,'source_freeze':source_freeze,
        'prepared_binding':binding_ref,'controller_review':prior_ref,'source_review':{
            'path':'/p/rlprojects/RND_worktrees/2026-11-05-11-30_common-performance-prerequisite-fact_contexts-4_paired-rounds-7_updates-per-arm-10_profile-qualification-operational-preparation_gpu-actions-0/.review_metadata/profile_target_operational_preparation/aa_independent_corrected_source_review.json',
            'sha256':'a40c26618bc3321950e40aaae0257c55b79d85099d383db8b0ac7dea1e359ec7'},
        'actual_immutable_artifact_count':59,'publication_freeze_count':64,'checks':list(checks),'check_count':len(checks),
        'review_scope':'Actual prepared runtime and unchanged dynamic launcher metadata only; operational pinning source has a separate independent approval and is not self-approved here.',
        'actual_cpu_job_id':os.environ['SLURM_JOB_ID'],'actual_cpu_step_id':os.environ['SLURM_STEP_ID'],
        'actual_target_job_id':None,'actual_target_gpu_uuid':None,'seconds':time.monotonic()-started,
        'audit_source':launcher.reference(Path(__file__).resolve()),'publication_check_not_called':True,
        'fresh_reservation_check_still_required_before_actual_action':True,
        'GPU_numerical_qualification':False,'current_physical_identity_verified':False,
        'scheduler_calls':0,'unresolved_anomalies':[],**launcher.ZERO}
    result=launcher.publish(P/'launcher_independent_review.json',review)
    print(json.dumps({'passed':True,'review':result,'source_freeze':source_freeze,'prepared_binding':binding_ref,
        'publication_files':64,'checks':len(checks),'seconds':review['seconds']}))


if __name__=='__main__':main()
