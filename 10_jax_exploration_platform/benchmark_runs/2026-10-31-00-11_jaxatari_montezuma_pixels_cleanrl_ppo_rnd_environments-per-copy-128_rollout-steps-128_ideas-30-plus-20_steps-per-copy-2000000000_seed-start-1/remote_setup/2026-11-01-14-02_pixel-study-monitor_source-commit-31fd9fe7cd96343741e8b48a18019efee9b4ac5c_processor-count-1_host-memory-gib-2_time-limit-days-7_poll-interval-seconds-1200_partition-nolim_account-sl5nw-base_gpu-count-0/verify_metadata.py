"""Check cap arithmetic, unchanged resource guards and cold frozen metadata imports without a monitor tick."""
import ast
from datetime import datetime,timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback

P=Path(__file__).resolve().parent
R=P.parent.parent
CONTEXT=json.loads((P/'preparation_context.json').read_text())
MANIFEST=json.loads((P/'source_snapshot.json').read_text())
SOURCE=Path(MANIFEST['source_root'])
SCRIPTS=SOURCE/'10_jax_exploration_platform/scripts/atari_pixel_campaign'
OLD=next((R/'remote_setup').glob('2026-11-01-08-35_pixel-study-monitor_*'))
PREFIX='10_jax_exploration_platform/scripts/atari_pixel_campaign/'

def sha(path):
    """Hash bounded file chunks without retaining a captured driver image in memory."""
    value=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):value.update(chunk)
    return value.hexdigest()

def write(path,value):
    """Keep actual verification evidence immutable and durable."""
    with path.open('x') as stream:json.dump(value,stream,indent=2);stream.write('\n');stream.flush();os.fsync(stream.fileno())

def load(name,path):
    """Read one exact new preparation helper without executing its CLI entry."""
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module

def main():
    """Verify real cap evidence and cold source visibility while proving all protected bytes are retained."""
    assert os.environ['SLURM_JOB_ID']=='6781231' and os.environ['SLURM_STEP_ID'].isdigit()
    print('owned metadata step '+os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],flush=True)
    original_old_plan=R/'data/2026-11-01-12-36_full-original-qualification_jaguar03_gpu-count-1_processor-count-8_host-memory-gib-64_time-limit-hours-3_policies-1_environments-128_rollout-steps-128_epochs-4_minibatches-4_warmup-rollouts-50_reference-updates-1_seed-1_driver-580.159.04/measurement_plan.json'
    old_unit=original_old_plan.parent
    old_packet=R/'remote_setup'/old_unit.name
    fresh_roster=R/'remote_setup/2026-11-01-13-56_explicit-pytest-root-qualification-packet-publication.json'
    fresh=json.loads(fresh_roster.read_text());assert sha(fresh_roster)=='858fa178a67886e18da307a1a0d0366f6250521442d93ac267f2709f2e0a9423'
    protected=[R/'SOURCE_AND_GATES_READY.json',*[R/'data/source_checks/d28e6735e79788f618acb0ebfde8609db1f776af'/filename for filename in ('h100.json','jaguar03_rtx_a4500.json')]]
    protected+=[Path(item['absolute_path']) for item in fresh['files']]
    protected+=[path for folder in (old_packet,old_unit) for path in folder.rglob('*') if path.is_file()]
    numerical=Path('/p/rlprojects/RND_worktrees/atari-cleanrl-source-d28e673')
    protected+=list((numerical/'10_jax_exploration_platform/src/exploration_platform/atari_pixels').glob('*.py'))
    protected+=[numerical/PREFIX/name for name in ('ideas.py','paired_benchmark.py','check_rendering.py')]
    protected=sorted(set(protected),key=str);before={str(path):sha(path) for path in protected}
    source_before={name:sha(SOURCE/name) for name in MANIFEST['files']}
    cap=load('prepared_monitor_duration',P/'prepare_packet.py')
    duration_checks=[]
    for text,expected in [('20-00:00:00',1728000),('7-00:00:00',604800),('6-23:59:59',604799),('UNLIMITED',None)]:
        assert cap.duration_seconds(text)==expected;duration_checks.append({'input':text,'seconds':expected})
    assert cap.duration_seconds('7-00:00:00')<=cap.duration_seconds('20-00:00:00')
    assert cap.duration_seconds('7-00:00:00')>cap.duration_seconds('6-23:59:59')
    for text in ('20-24:00:00','nonsense'):
        try:cap.duration_seconds(text)
        except ValueError:duration_checks.append({'input':text,'refused':True})
        else:raise AssertionError('bad duration accepted')
    guard=load('new_monitor_guard',P/'allocation_guard.py')
    old_ast={node.name:ast.dump(node,include_attributes=False) for node in ast.parse((OLD/'allocation_guard.py').read_text()).body if isinstance(node,ast.FunctionDef)}
    new_ast={node.name:ast.dump(node,include_attributes=False) for node in ast.parse((P/'allocation_guard.py').read_text()).body if isinstance(node,ast.FunctionDef)}
    assert old_ast==new_ast
    fields={'CPUs/Task':'1','NumNodes':'1','MinMemoryNode':'2G','ReqTRES':'cpu=1,mem=2G,node=1','NumCPUs':'1','AllocTRES':'cpu=1,mem=2G,node=1','NodeList':'synthetic-cpu-node'}
    node={'NodeName':'synthetic-cpu-node','ThreadsPerCore':'2'};guard.validate_resources(fields,node)
    two={**fields,'NumCPUs':'2','AllocTRES':'cpu=2,mem=2G,node=1'};guard.validate_resources(two,node)
    bad=[({**fields,'AllocTRES':'cpu=1,mem=2G,node=1,gres/gpu=1'},node),({**fields,'ReqTRES':'cpu=2,mem=2G,node=1'},node),(two,{**node,'ThreadsPerCore':'1'}),(fields,{**node,'NodeName':'another-node'})]
    for values,topology in bad:
        try:guard.validate_resources(values,topology)
        except ValueError:pass
        else:raise AssertionError('invalid resource shape accepted')
    names=['monitor','controller','configured_measurement','uva_slurm_configured_isolation','measure_selected','profile_selected','post_comparison_validation','portable_evidence','device_memory_limit','h100_nvl_slurm_site_validation','uva_active_configuration_observation','passive_installed_slurm_observation']
    probe=r"""
import ctypes,hashlib,importlib,json,os,pathlib,sys
packet=pathlib.Path(sys.argv[1]);manifest=json.loads((packet/'source_snapshot.json').read_text());source=pathlib.Path(manifest['source_root']);scripts=source/'10_jax_exploration_platform/scripts/atari_pixel_campaign';names=json.loads(sys.argv[2]);blocked=[]
assert not any(n.split('.')[0] in {'jax','jaxlib','numpy','torch','flax','optax','jaxtari','scipy','pynvml'} for n in sys.modules)
def guard(event,args):
 deny=event=='import' and args[0].split('.')[0] in {'jax','jaxlib','numpy','torch','flax','optax','jaxtari','scipy','pynvml'}
 deny=deny or event in {'subprocess.Popen','socket.connect','os.system'} or event=='ctypes.dlopen' and args[0] is not None
 deny=deny or event=='open' and isinstance(args[0],str) and args[0].startswith('/dev/nvidia')
 if deny:blocked.append(event);raise RuntimeError('cold metadata import forbids live operation:'+event)
sys.addaudithook(guard);sys.path.insert(0,str(scripts));seen={}
for name in names:
 module=importlib.import_module(name);path=pathlib.Path(module.__file__).resolve();assert path==scripts/(name+'.py');relative=path.relative_to(source).as_posix();assert hashlib.sha256(path.read_bytes()).hexdigest()==manifest['files'][relative]['sha256'];seen[name]=str(path)
core=sys.modules['configured_measurement'].core();assert pathlib.Path(core.__file__).resolve()==scripts/'uva_slurm_configured_isolation.py'
for filename in core.HELPERS:
 if filename=='uva_slurm_configured_isolation.py':continue
 module=core.adjacent(filename[:-3]);path=pathlib.Path(module.__file__).resolve();assert path==scripts/filename;assert hashlib.sha256(path.read_bytes()).hexdigest()==manifest['files'][path.relative_to(source).as_posix()]['sha256']
assert not blocked
print(json.dumps({'imported':seen,'numerical_imports':False,'blocked_actions':blocked,'full_tick_executed':False}))
"""
    argv=['/usr/bin/python3','-I','-S','-B','-c',probe,str(P),json.dumps(names)]
    result=subprocess.run(argv,capture_output=True,text=True,timeout=30)
    write(P/'cold_import_return.json',{'command':argv,'returncode':result.returncode,'stdout':result.stdout,'stderr':result.stderr})
    assert result.returncode==0,result.stderr
    ast.parse((P/'candidate_full_tick.py').read_text());subprocess.run(['bash','-n',str(P/'monitor.sbatch')],check=True)
    after={str(path):sha(path) for path in protected};assert before==after
    source_after={name:sha(SOURCE/name) for name in MANIFEST['files']};assert source_before==source_after
    assert not any((P/name).exists() for name in ('submission_intent.json','submission_receipt.json','release_intent.json','release_receipt.json','candidate_full_tick_verification.json'))
    value={'schema_version':1,'kind':'replacement_monitor_cap_resource_cold_import_metadata_verification','job_id':os.environ['SLURM_JOB_ID'],'step_id':os.environ['SLURM_STEP_ID'],'created_at_utc':datetime.now(timezone.utc).isoformat(),'source_commit':MANIFEST['source_commit'],'source_files':len(MANIFEST['files']),'duration_cap_proof':{'path':str(P/'duration_cap_proof.json'),'sha256':sha(P/'duration_cap_proof.json')},'duration_checks':duration_checks,'resource_cases':{'valid':2,'refused':4},'old_and_new_resource_guard_function_asts_equal':True,'cold_metadata_import':{'path':str(P/'cold_import_return.json'),'sha256':sha(P/'cold_import_return.json')},'protected_before':before,'protected_after':after,'protected_paths':len(protected),'source_hashes_before':source_before,'source_hashes_after':source_after,'candidate_script_syntax_passed':True,'shell_syntax_passed':True,'candidate_full_tick_executed':False,'supports_corrected_a4500_readers':True,'future_direct_adapter_supported':False,'new_jobs_submitted':0,'jobs_released_or_cancelled':0,'worker_admission':False,'scientific_credit':0}
    write(P/'metadata_verification.json',value)
    static=[path for path in P.iterdir() if path.is_file()]+[path for path in (R/'data'/P.name).iterdir() if path.is_file()]
    write(P/'preparation_file_roster.json',{'schema_version':1,'kind':'prepared_replacement_monitor_static_file_roster','source_manifest':{'path':str(P/'source_snapshot.json'),'sha256':sha(P/'source_snapshot.json')},'files':[{'path':str(path),'sha256':sha(path)} for path in sorted(static,key=str)],'source_file_count':len(MANIFEST['files']),'full_tick_executed':False,'job_actions':0})
    print(json.dumps({'packet':str(P),'source_files':len(MANIFEST['files']),'protected_unchanged':len(protected),'metadata_verification_sha256':sha(P/'metadata_verification.json'),'plan_sha256':sha(P/'monitor_replacement_plan.json'),'manifest_sha256':sha(P/'manifest.yaml'),'script_sha256':sha(P/'monitor.sbatch'),'roster_sha256':sha(P/'preparation_file_roster.json')}),flush=True)

if __name__=='__main__':
    try:main()
    except BaseException as error:
        write(P/'metadata_verification_failure.json',{'job_id':os.environ.get('SLURM_JOB_ID'),'step_id':os.environ.get('SLURM_STEP_ID'),'error':str(error),'traceback':traceback.format_exc(),'new_job_actions':0,'no_candidate_tick':True})
        raise
