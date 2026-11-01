"""Prepare the exact published full-original qualification without importing numerical packages."""
import builtins
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time
import traceback

PACKET=Path(__file__).resolve().parent
CONTEXT=json.loads((PACKET/'preparation_context.json').read_text())
SOURCE=Path(CONTEXT['helper_root'])
SCRIPTS=SOURCE/'10_jax_exploration_platform/scripts/atari_pixel_campaign'
OBSERVED=[]
BLOCKED=[]

def audit(event,args):
    if event=='import' and args[0].split('.')[0] in {'jax','jaxlib','numpy','scipy','torch','flax','optax','jaxtari','pynvml'}:
        BLOCKED.append([event,args[0]]);raise RuntimeError('numerical import forbidden during metadata preparation')
    if event in {'ctypes.dlopen','socket.connect','os.system'}:
        BLOCKED.append([event,str(args)[:200]]);raise RuntimeError('device/library/network operation forbidden during preparation')
    if event=='subprocess.Popen':
        argv=list(args[1]);name=Path(str(argv[0])).name
        allowed=name=='git' or argv in [['scontrol','show','node','jaguar03','-o'],['scontrol','show','partition','gpu','-o'],['scontrol','show','reservation','sl5nw_168','-o']]
        if not allowed:
            BLOCKED.append([event,argv]);raise RuntimeError('unapproved command during metadata preparation: '+repr(argv))
        OBSERVED.append(argv)

sys.addaudithook(audit)
sys.path.insert(0,str(SCRIPTS))

def write(path,value):
    raw=(json.dumps(value,indent=2,allow_nan=False)+'\n').encode()
    with path.open('xb') as stream:stream.write(raw);stream.flush();os.fsync(stream.fileno())

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    import diagnose_gpu_execution as diagnostic
    import measure_selected as measure
    import uva_slurm_configured_isolation as core
    import configured_measurement as support
    assert os.environ['SLURM_JOB_ID']=='6781231' and os.environ['SLURM_STEP_ID'].isdigit()
    assert subprocess.check_output(['git','-C',str(SOURCE),'rev-parse','HEAD'],text=True).strip()==CONTEXT['helper_commit']
    assert not subprocess.check_output(['git','-C',str(SOURCE),'status','--porcelain','--untracked-files=all'],text=True).strip()
    refs=CONTEXT['references']
    assert sha(refs['reference_packet_plan'])==refs['reference_packet_plan_sha256']
    for item in refs['shared_software'].values():assert sha(item['path'])==item['sha256']
    assert sha(refs['cleanrl_reference']['file']['path'])==refs['cleanrl_reference']['file']['sha256']
    old=json.loads(Path(refs['reference_packet_plan']).read_text())
    backend=json.loads(Path(refs['shared_software']['installed_backend']['path']).read_text())
    assets=old['asset_record']
    assert sha(assets['manifest'])==assets['sha256']
    manifest=json.loads(Path(assets['manifest']).read_text())
    jaxtari=backend['jaxtari_source']
    assert jaxtari['source_commit']==manifest['source_commit']
    assert subprocess.check_output(['git','-C',jaxtari['source_root'],'rev-parse','HEAD'],text=True).strip()==manifest['source_commit']
    full=diagnostic.default_configuration(Path(CONTEXT['numerical_root']))
    full['execution']={'compilation_mode':'whole_iteration','iterations_per_call':1,'report_readout':'none','report_every_iterations':10}
    original=old['configuration'] if 'configuration' in old else {'learner':old['learner'],'environment':old['environment']}
    assert json.loads(json.dumps(full['learner']))==original['learner']
    assert json.loads(json.dumps(full['environment']))==original['environment']
    runtime={key:old['runtime_environment'][key] for key in ('JAX_ENABLE_X64','JAX_DEFAULT_MATMUL_PRECISION','XLA_FLAGS')}
    for name,value in [('full_original.json',full),('runtime.json',runtime),('assets.json',assets),('profile.json',core.PROFILE)]:write(PACKET/name,value)
    resources={}
    for key,argv in [('node',['scontrol','show','node','jaguar03','-o']),('partition',['scontrol','show','partition','gpu','-o']),('reservation',['scontrol','show','reservation','sl5nw_168','-o'])]:
        run=subprocess.run(argv,capture_output=True,text=True,timeout=20,env={**os.environ,'TZ':'UTC'})
        resources[key]={'command':argv,'returncode':run.returncode,'stdout':run.stdout,'stderr':run.stderr}
        assert run.returncode==0
    write(PACKET/'resource_observation.json',{'schema_version':1,'kind':'qualification_preparation_resource_observation','time_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'resource_basis':CONTEXT['resource_basis'],'observations':resources,'other_jobs_queried':False,'gpu_queries':0})
    argv=['--prepare-only','--mode','correctness','--configured-isolation-profile',str(PACKET/'profile.json'),'--target-qualification','--cleanrl-source-root',refs['cleanrl_reference']['source_root'],'--qualification-software-result',refs['shared_software']['result']['path'],'--source-root',CONTEXT['numerical_root'],'--source-commit',CONTEXT['numerical_commit'],'--configuration',str(PACKET/'full_original.json'),'--reference-configuration',str(PACKET/'full_original.json'),'--runtime-environment',str(PACKET/'runtime.json'),'--asset-record',str(PACKET/'assets.json'),'--jaxtari-source-root',jaxtari['source_root'],'--run-dir',CONTEXT['campaign'],'--unit-id',CONTEXT['unit_id'],'--hardware-class','jaguar03_rtx_a4500','--seed-start','1','--python',refs['python']]
    command=['/usr/bin/python3','-I','-B',str(SCRIPTS/'measure_selected.py'),*argv]
    write(PACKET/'preparation_command.json',{'argv':command,'execution':'same exact published parser/main called by this metadata guard on owned compute'})
    with (PACKET/'command.txt').open('x') as stream:stream.write(shlex.join(command)+'\n')
    result=measure.prepare(measure.parse_arguments(argv))
    unit=Path(CONTEXT['unit'])
    plan=json.loads((unit/'measurement_plan.json').read_text())
    assert plan['gpu_uuid'] is None and plan['helpers']['source_commit']==CONTEXT['helper_commit']
    support.verify_qualification_source_receipts(plan)
    matrix=support.test_collection_contract(plan)
    assert sum(matrix.values())==48
    assert {key:plan['renderer_contract'][key] for key in ('rooms','cases','synthetic_pixel_cases')}=={'rooms':33,'cases':132,'synthetic_pixel_cases':11}
    for relative,digest in plan['helpers']['files'].items():
        assert sha(unit/'code'/Path(relative).name)==digest
        assert sha(SOURCE/relative)==digest
    assert not any((unit/name).exists() for name in ('owned_allocation_inputs.json','configured_isolation_pre_release_inputs.json','resolved_device_binding.json','target_qualification_result.json','result.json'))
    assert not BLOCKED
    record={'schema_version':1,'kind':'full_original_qualification_metadata_preparation','job_id':os.environ['SLURM_JOB_ID'],'step_id':os.environ['SLURM_STEP_ID'],'pid':os.getpid(),'helper_commit':CONTEXT['helper_commit'],'helper_root':str(SOURCE),'unit':str(unit),'plan_sha256':sha(unit/'measurement_plan.json'),'staged_helpers':len(plan['helpers']['files']),'original_pytest_cases':sum(matrix.values()),'original_pytest_roster':matrix,'renderer_contract':plan['renderer_contract'],'configuration_matches_original':True,'runtime_fields':runtime,'source_receipts_rebuilt':True,'unknown_gpu_uuid':plan['gpu_uuid'],'job_identity_in_plan':False,'gpu_operations':0,'numerical_imports':0,'commands':OBSERVED,'blocked_attempts':BLOCKED,'source_status_after':subprocess.check_output(['git','-C',str(SOURCE),'status','--porcelain','--untracked-files=all'],text=True),'no_submission':True,'returncode':0}
    write(PACKET/'preparation_receipt.json',record)
    print(json.dumps({k:record[k] for k in ('job_id','step_id','plan_sha256','staged_helpers','original_pytest_cases','returncode')}))

if __name__=='__main__':
    try:main()
    except BaseException as error:
        write(PACKET/'preparation_failure.json',{'schema_version':1,'job_id':os.environ.get('SLURM_JOB_ID'),'step_id':os.environ.get('SLURM_STEP_ID'),'error_type':type(error).__name__,'error':str(error),'traceback':traceback.format_exc(),'commands':OBSERVED,'blocked_attempts':BLOCKED,'no_submission':True,'gpu_operations':0})
        raise
