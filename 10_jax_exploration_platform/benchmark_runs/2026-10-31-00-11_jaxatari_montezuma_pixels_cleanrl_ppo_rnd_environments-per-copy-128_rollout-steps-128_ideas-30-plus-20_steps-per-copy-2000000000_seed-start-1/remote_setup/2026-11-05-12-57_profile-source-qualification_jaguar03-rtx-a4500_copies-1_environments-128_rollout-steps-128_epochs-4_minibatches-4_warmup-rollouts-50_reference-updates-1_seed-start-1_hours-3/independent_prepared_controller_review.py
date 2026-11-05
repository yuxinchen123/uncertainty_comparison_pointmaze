"""Review only actual prepared A4500 controller/capture metadata, without a target action."""
import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shlex
import stat
import subprocess
import sys
import time
import traceback

P=Path(__file__).resolve().parent
R=P.parent.parent
U=R/'data'/P.name
C=R/'data'/(P.name+'_controller')
SOURCE='d28e6735e79788f618acb0ebfde8609db1f776af'
HELPER='d1469fd063a44167ec56760729eb0beea7ecf7d3'
PREFIX='10_jax_exploration_platform/scripts/atari_pixel_campaign/'
STANDARD=('manifest.yaml','config_resolved.yaml','command.txt','experiment_background.md')
FORBIDDEN={'jax','jaxlib','numpy','flax','optax','torch','jaxtari','cupy','pynvml'}
RECORDS={}
CHECKS=[]


def need(value,message):
    """Record a passed check only after its exact actual predicate holds."""
    if not value:raise ValueError(message)
    CHECKS.append(message)


def raw(path):
    """Capture bounded stable owned source/metadata; never open a scientific result or image."""
    path=Path(path).absolute();before=path.lstat()
    need(path.resolve(strict=True)==path and stat.S_ISREG(before.st_mode)
        and before.st_uid==os.getuid() and before.st_size<=4*1024*1024,'bounded owned ordinary file: '+str(path))
    descriptor=os.open(path,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK)
    try:
        identity=lambda item:(item.st_dev,item.st_ino,item.st_size,item.st_mtime_ns,item.st_ctime_ns,item.st_mode,item.st_uid,item.st_gid)
        need(identity(os.fstat(descriptor))==identity(before),'pinned metadata identity before read: '+str(path))
        with os.fdopen(descriptor,'rb',closefd=False) as stream:value=stream.read(4*1024*1024+1)
        need(len(value)==before.st_size and identity(os.fstat(descriptor))==identity(before)==identity(path.lstat()),
            'stable metadata bytes: '+str(path))
    finally:os.close(descriptor)
    RECORDS[str(path)]={'sha256':hashlib.sha256(value).hexdigest(),'bytes':len(value),'mode':oct(stat.S_IMODE(before.st_mode))}
    return value


def ref(path):
    """Return the exact actual two-field reference after bounded capture."""
    raw(path);return {'path':str(path),'sha256':RECORDS[str(path)]['sha256']}


def document(path):
    """Reject duplicate keys and nonfinite values in every current record."""
    def unique(items):
        result={}
        for key,value in items:
            if key in result:raise ValueError('duplicate metadata key')
            result[key]=value
        return result
    return json.loads(raw(path),object_pairs_hook=unique,
        parse_constant=lambda value:(_ for _ in ()).throw(ValueError('nonfinite metadata')))


def checked(value):
    """Require an exact recorded path/hash without following another type of record."""
    need(type(value) is dict and set(value)=={'path','sha256'} and ref(Path(value['path']))==value,'exact immutable input reference')
    return document(value['path'])


def git_blob(root,commit,relative):
    """Read one fixed historical source object without fetching or updating an index."""
    environment={key:value for key,value in os.environ.items() if not key.startswith('GIT_')}
    environment.update(GIT_OPTIONAL_LOCKS='0',GIT_TERMINAL_PROMPT='0')
    result=subprocess.run(['git','-C',str(root),'cat-file','blob',commit+':'+relative],capture_output=True,
        check=True,timeout=15,env=environment)
    need(result.stderr==b'' and len(result.stdout)<=4*1024*1024,'bounded actual committed source response')
    return result.stdout


def save(path,value):
    """Fsync one new actual independent result without replacing any earlier record."""
    with path.open('x') as stream:
        json.dump(value,stream,indent=2,sort_keys=True,allow_nan=False);stream.write('\n');stream.flush();os.fsync(stream.fileno())
    descriptor=os.open(path.parent,os.O_RDONLY|os.O_DIRECTORY)
    try:os.fsync(descriptor)
    finally:os.close(descriptor)


def main():
    """Check actual staged source and approved metadata, leaving all runtime admission closed."""
    started=time.monotonic()
    need(os.environ.get('SLURM_JOB_ID')=='6825983' and os.environ.get('SLURM_STEP_ID','').isdecimal(),
        'actual current owned CPU allocation and numeric step')
    def audit(event,args):
        if event=='import' and args[0].split('.')[0] in FORBIDDEN:raise RuntimeError('numerical or device import refused')
        if event.startswith('ctypes.') or event in ('socket.connect','socket.bind'):raise RuntimeError('native-library/device/network action refused')
        if event=='open' and isinstance(args[0],(str,bytes)):
            name=os.fsdecode(args[0])
            if name.startswith('/dev/nvidia') or name.endswith(('.so','.so.1','.so.12')):raise RuntimeError('device or library-image read refused')
    sys.addaudithook(audit)
    parent,child=document(C/'measurement_plan.json'),document(U/'measurement_plan.json')
    binding=document(C/'prepared_child_binding.json')
    need(parent['kind']=='profile_target_qualification_controller' and child['kind']=='full_original_profile_source_target_qualification'
        and parent['source_commit']==child['source_commit']==SOURCE and parent['profile_unit']==str(U)
        and parent['campaign_run_dir']==child['campaign_run_dir']==str(R),'exact new controller and capture identities')
    helper=child['helpers'];root=Path(helper['source_root'])
    need(helper['source_commit']==HELPER and len(helper['files'])==35 and parent['target_fields']['helpers']==helper,
        'exact full thirty-five-helper source contract')
    for relative,expected in helper['files'].items():
        need(relative==PREFIX+Path(relative).name and relative.endswith('.py'),'single exact staged helper namespace')
        staged=raw(U/'code'/Path(relative).name)
        need(hashlib.sha256(staged).hexdigest()==expected and raw(root/relative)==staged
            and git_blob(root,HELPER,relative)==staged,'staged, clean-root and committed helper bytes agree: '+relative)
    for name in ('profile_target_qualification.py','direct_common_timing_controller.py'):
        need(raw(C/'code'/name)==raw(U/'code'/name),'actual parent source matches capture source: '+name)
    for receipt in (child['core'],child['paired_source']):
        need(receipt['source_commit']==SOURCE and receipt['source_root']==child['source_root'],'unchanged numerical source receipt')
        for relative,item in receipt['files'].items():
            expected=item['sha256'] if isinstance(item,dict) else item
            value=raw(Path(child['source_root'])/relative)
            need(hashlib.sha256(value).hexdigest()==expected,'actual protected numerical source bytes: '+relative)
    for folder in (P,C,U):
        for name in STANDARD:need(bool(raw(folder/name)),'present nonempty standard record: '+str(folder/name))
    need(binding['controller_plan']==ref(C/'measurement_plan.json') and binding['child_plan']==ref(U/'measurement_plan.json')
        and binding['controller_source']==ref(C/'code/profile_target_qualification.py') and binding['helpers']==helper,
        'actual prepared controller-to-capture source binding')
    parent_command=['/usr/bin/python3','-I','-S','-B',str(C/'code/profile_target_qualification.py'),'--prepared-unit',str(C)]
    child_command=[child['python'],str(U/'code/profile_selected.py'),'--prepared-unit',str(U)]
    need(shlex.split(raw(C/'command.txt').decode())==parent_command
        and shlex.split(raw(U/'command.txt').decode())==binding['command']==child_command,
        'exact isolated metadata parent and unchanged selected-Python capture argv')
    need(document(U/'measurement_prepared.json')=={'plan_sha256':ref(U/'measurement_plan.json')['sha256']},'actual child preparation receipt')
    spec=importlib.util.spec_from_file_location('independent_actual_aa_target_qualifier',C/'code/profile_target_qualification.py')
    qualifier=importlib.util.module_from_spec(spec);spec.loader.exec_module(qualifier)
    need(qualifier.contract(child)==parent,'actual staged pure target qualification contract')
    original=qualifier.original_gate_plan(child)
    old=checked(parent['original_qualification_plan'])
    for key in ('qualification_sources','cleanrl_reference','qualification_software','renderer_contract'):
        need(original[key]==old[key],'unchanged original gate declaration: '+key)
    need(original['configuration']==original['reference_configuration']==old['configuration']
        and original['copy_seeds']==[1] and original['configuration']['learner']['n_copies']==1,
        'separate one-copy original full reference projection')
    for relative,expected in old['qualification_sources']['files'].items():
        value=raw(Path(child['source_root'])/relative)
        need(hashlib.sha256(value).hexdigest()==expected,'unchanged original gate/test source: '+relative)
    need(child['configuration']==parent['target_fields']['configuration'] and child['copy_seeds']==[1]
        and type(child['copy_seeds'][0]) is int and child['configuration']['environment']['num_envs']==128,
        'full candidate configuration and stable single-copy seed')
    proof=checked(parent['four_context_proof']);review=checked(parent['four_context_review'])
    need(parent['four_context_proof']['sha256']=='e75b78e9a9b3fd4a857c2d7c5148de15c8800ef14398c5b56b13f5c78db4d3c7'
        and parent['four_context_review']['sha256']=='5e19415c03e8063275422d6651c5496de9e94fc63c3f1d4d12df27e11a71298d'
        and review['approved'] is review['passed'] is True and review['proof']==parent['four_context_proof']
        and {(row['hardware_class'],row['copies']) for row in proof['contexts']}=={('h100',1),('h100',2),('jaguar03_rtx_a4500',1),('jaguar03_rtx_a4500',2)},
        'reuse exact genuine independently reviewed four-context fact; no new fact review')
    native=child['profile_target_process_control'];component=checked(native['prior_abi_baseline']['independent_review'])
    need(native['usage']=='qualification' and native['prior_component_qualifies_current_adapter'] is False
        and native['prior_abi_baseline']['independent_review']['sha256']=='4264288c209606dccc15c883724ffeccaf5a480c943c5e29dd9b1c409c814da5'
        and component['approved'] is component['passed'] is True and native['admission_granted'] is False,
        'actual accepted process-only A4500 component is a prior baseline, not new numerical admission')
    expected=child['profile_qualification_backend']['expected']
    need(expected['sha256']==parent['backend_image_record']['sha256']=='19e752ef9b93999e8bec28a0706947935c31f4229a3b33556597de8f86575412'
        and raw(expected['path'])==raw(parent['backend_image_record']['path']),'fresh new-schema backend metadata copied exactly')
    for name in ('gpu_uuid','job_id'):need(child.get(name) is None,'actual '+name+' remains unresolved before submission')
    need(child['configured_isolation']['physical_identity']=={'schema_version':1,'kind':'unresolved_owned_cuda_device','gpu_uuid':None,
        'pci_address':None,'physical_minor':None} and child['configured_isolation']['owned_inputs_path']==str(U/'owned_allocation_inputs.json'),
        'logical configured trust is separate from unresolved actual device identity')
    for folder,names in ((C,('controller_intent.json','controller_spawn.json','controller_wait.json','controller_failure.json','controller_inconclusive.json')),
            (U,('owned_allocation_inputs.json','progress.json','result.json','failure.json','profile_target_qualification_result.json'))):
        need(not any(os.path.lexists(folder/name) for name in names),'no attempted controller/capture runtime records: '+str(folder))
    need(not FORBIDDEN.intersection(sys.modules),'no numerical/device modules imported by this review')
    before=dict(RECORDS)
    for path,value in before.items():need(hashlib.sha256(raw(path)).hexdigest()==value['sha256'],'reviewed source/metadata unchanged: '+path)
    need(time.monotonic()-started<285,'whole metadata review stays below five minutes')
    record={'schema_version':1,'kind':'independent_profile_target_qualification_controller_launch_review',
        'approved':True,'passed':True,'independent_of_producer':True,'reviewer':'/root',
        'controller_plan':ref(C/'measurement_plan.json'),'child_plan':ref(U/'measurement_plan.json'),
        'prepared_child_binding':ref(C/'prepared_child_binding.json'),'runtime_helper_commit':HELPER,
        'staged_capture_helper_count':35,'parent_source_count':2,'standard_record_count':12,
        'four_context_fact':parent['four_context_proof'],'four_context_review':parent['four_context_review'],
        'prior_process_review':native['prior_abi_baseline']['independent_review'],'backend_image_record':parent['backend_image_record'],
        'source_and_metadata_inventory':RECORDS,'checks':CHECKS,'check_count':len(CHECKS),
        'original_48_tests_renderer_full_reference_gate_declarations_preserved':True,
        'actual_job_id':os.environ['SLURM_JOB_ID'],'actual_step_id':os.environ['SLURM_STEP_ID'],'seconds':time.monotonic()-started,
        'actual_target_job_id':None,'actual_target_gpu_uuid':None,'actual_runtime_launched':False,
        'GPU_numerical_qualification':False,'current_physical_identity_verified':False,'physical_gres_mapping_verified':False,
        'kernel_device_enforcement_verified':False,'scientific_credit':0,'profile_credit':0,'case_credit':0,'idea_credit':0,
        'packing_credit':0,'final_training_credit':0,'worker_admitted':False,'admission_granted':False,'unresolved_anomalies':[]}
    destination=C/'controller_launch_independent_review.json';save(destination,record)
    print(json.dumps({'passed':True,'review':ref(destination),'checks':len(CHECKS),'seconds':record['seconds']}))


if __name__=='__main__':
    try:main()
    except BaseException as error:
        path=P/'independent_prepared_controller_review_failure.json'
        if not path.exists():save(path,{'passed':False,'error':repr(error),'traceback':traceback.format_exc(),'actual_job_id':os.environ.get('SLURM_JOB_ID'),
            'actual_step_id':os.environ.get('SLURM_STEP_ID'),'target_operations':0,'admission_granted':False})
        raise
