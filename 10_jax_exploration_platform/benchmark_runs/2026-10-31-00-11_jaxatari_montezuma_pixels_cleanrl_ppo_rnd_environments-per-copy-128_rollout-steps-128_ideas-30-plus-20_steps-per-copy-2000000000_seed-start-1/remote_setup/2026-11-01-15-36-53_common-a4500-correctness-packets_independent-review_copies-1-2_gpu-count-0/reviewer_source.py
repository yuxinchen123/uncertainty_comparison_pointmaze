"""Independently review two frozen A4500 correctness packets using metadata only."""
import ast
import copy
from datetime import datetime, timezone
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import re
import shlex
import stat
import subprocess
import sys
import traceback
import unittest
from zoneinfo import ZoneInfo

W = Path(__file__).resolve().parents[1]
MONITOR = json.loads((W / '.review_metadata/monitor_7fd_preparation_inputs.json').read_text())
R = Path(MONITOR['packet']).parent.parent
INDEX = R / 'remote_setup/2026-11-01-15-09_unconfirmed-common-a4500-preparation_index.json'
value = json.loads(INDEX.read_bytes())
P = Path(value['packets'][0]['packet'])
OUT = R / 'remote_setup' / (datetime.now(ZoneInfo('America/Los_Angeles')).strftime('%Y-%m-%d-%H-%M-%S') + '_common-a4500-correctness-packets_independent-review_copies-1-2_gpu-count-0')
OUT.mkdir()
BLOCKED = []
FORBIDDEN = {'jax','jaxlib','numpy','scipy','torch','flax','optax','jaxtari','pynvml','cupy'}
ROOTS = {item['helper_root'] for item in value['packets']} | {item['numerical_root'] for item in value['packets']}
assert os.environ.get('SLURM_JOB_ID') == '6782816' and os.environ.get('SLURM_STEP_ID','').isdigit()
print('independent AA metadata step6782816.' + os.environ['SLURM_STEP_ID'], flush=True)


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024), b''):result.update(chunk)
    return result.hexdigest()


def record(path):
    path = Path(path); mode = path.lstat().st_mode
    assert stat.S_ISREG(mode) and path.resolve(strict=True) == path and not path.is_symlink(), str(path)
    return {'path':str(path),'sha256':digest(path),'bytes':path.stat().st_size,'mode':format(stat.S_IMODE(mode),'04o')}


def write(name, document):
    raw = document if isinstance(document, bytes) else (json.dumps(document, indent=2, allow_nan=False)+'\n').encode()
    with (OUT / name).open('xb') as stream:stream.write(raw);stream.flush();os.fsync(stream.fileno())
    fd=os.open(OUT,os.O_RDONLY|os.O_DIRECTORY);os.fsync(fd);os.close(fd)
    return record(OUT/name)


def read(path):return json.loads(Path(path).read_bytes())


def scope(event, arguments):
    denied = event == 'import' and arguments[0].split('.')[0] in FORBIDDEN
    denied = denied or event in {'socket.connect','os.system'} or event == 'ctypes.dlopen' and arguments[0] is not None
    denied = denied or event == 'open' and isinstance(arguments[0],str) and arguments[0].startswith('/dev/nvidia')
    if event == 'subprocess.Popen':
        argv = arguments[1]; denied = True
        if isinstance(argv,(list,tuple)) and argv:
            if Path(argv[0]).name == 'git' and '-C' in argv:
                root = argv[argv.index('-C')+1]
                denied = root not in ROOTS or not any(item in argv for item in ('show','rev-parse','status','diff','ls-files','symbolic-ref','ls-tree'))
            elif list(argv[:4]) == ['/usr/bin/python3','-I','-S','-B']:
                if len(argv)>4 and argv[4] == '-c':
                    denied = argv[5] not in ALLOWED_PROGRAMS
                elif len(argv)>4 and argv[4] in GATES:
                    denied = False
            elif list(argv[:2]) == ['bash','-n']:
                denied = len(argv)!=3 or argv[2] not in SCRIPTS
    if denied:
        BLOCKED.append({'event':event,'arguments':repr(arguments)[:2000]})
        raise RuntimeError('Independent metadata review refused '+event)


PROBE = r'''
import hashlib,importlib,json,os,pathlib,subprocess,sys
unit=pathlib.Path(sys.argv[1]);plan=json.loads((unit/'measurement_plan.json').read_bytes());roots={plan['source_root'],plan['helpers']['source_root']};blocked=[]
qualified_plan=json.loads((pathlib.Path(plan['target_qualification_inputs']['result']['path']).parent/'measurement_plan.json').read_bytes());roots.add(qualified_plan['cleanrl_reference']['source_root'])
roots.update(json.loads(sys.argv[2]))
forbidden={'jax','jaxlib','numpy','scipy','torch','flax','optax','jaxtari','pynvml','cupy'}
def guard(event,args):
 denied=event=='import' and args[0].split('.')[0] in forbidden
 denied=denied or event in {'socket.connect','os.system'} or event=='ctypes.dlopen' and args[0] is not None
 denied=denied or event=='open' and isinstance(args[0],str) and args[0].startswith('/dev/nvidia')
 if event=='subprocess.Popen':
  argv=args[1];denied=not(isinstance(argv,(list,tuple)) and pathlib.Path(argv[0]).name=='git' and '-C' in argv and argv[argv.index('-C')+1] in roots and any(x in argv for x in ('show','rev-parse','status','diff','ls-files','symbolic-ref','ls-tree')))
 if denied:blocked.append(event);raise RuntimeError('cold saved-evidence review forbids '+event)
sys.addaudithook(guard);sys.path.insert(0,str(unit/'code'))
import measure_selected as measurement
import configured_measurement as support
measurement.verify_prepared(unit,plan)
qualified=support.verify_qualification_inputs(plan)
assets=measurement.training.verify_assets(plan['asset_record'],{'XDG_DATA_HOME':plan['asset_record']['root']})
assert assets['file_count']==4531
assert len(plan['helpers']['files'])==20
loaded={}
for name in ('measure_selected','configured_measurement','uva_slurm_configured_isolation','diagnose_gpu_execution','post_comparison_validation','train_optimized'):
 module=sys.modules.get(name)
 if module is None and name=='uva_slurm_configured_isolation':module=support.core()
 path=pathlib.Path(module.__file__).resolve();assert path==unit/'code'/(name+'.py');loaded[name]=str(path)
assert not blocked and not any(x.split('.')[0] in forbidden for x in sys.modules)
print(json.dumps({'passed':True,'copies':plan['configuration']['learner']['n_copies'],'helpers':20,'assets':assets,'actual_qualification_kind':qualified['kind'],'actual_qualification_result_sha256':plan['target_qualification_inputs']['result']['sha256'],'actual_qualification_review_sha256':plan['target_qualification_inputs']['review']['sha256'],'cold_imports':loaded,'numerical_imports':False,'blocked':blocked,'qualification_raw_reader_replayed':True}))
'''
GATES = {str(Path(item['packet'])/'wait_for_owned_inputs.py') for item in value['packets'] if item['mode']=='correctness'}
SCRIPTS = {str(Path(item['packet'])/'job.sbatch') for item in value['packets'] if item['mode']=='correctness'}
ALLOWED_PROGRAMS = {PROBE}
for gate in GATES:
    ALLOWED_PROGRAMS.add("import importlib.util,subprocess; s=importlib.util.spec_from_file_location('barrier',"+repr(gate)+"); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); observed=m.install_metadata_guard(); blocked=False\ntry:\n subprocess.run(['scontrol','show','job','123456789'])\nexcept RuntimeError:\n blocked=True\nassert blocked and observed['forbidden_operation_attempts']==['subprocess.Popen']\nprint('scheduler blocked before execution')\n")
sys.addaudithook(scope)


def historical_git_roots(plan):
    """Read only this qualification's hash-bound software approval chain before allowing local Git reads."""
    qualified=read(Path(plan['target_qualification_inputs']['result']['path']).parent/'measurement_plan.json')
    base=Path(qualified['qualification_software']['result']['path']).parent
    pending=[read(base/name) for name in ('case.json','infrastructure_retry_execution.json','region_preparation.json')]
    roots=set();seen={}
    def inspect(document):
        if isinstance(document,dict):
            if isinstance(document.get('source_root'),str) and re.fullmatch(r'[0-9a-f]{40}',str(document.get('source_commit',''))):roots.add(document['source_root'])
            if isinstance(document.get('path'),str) and document['path'].endswith('.json') and re.fullmatch(r'[0-9a-f]{64}',str(document.get('sha256',''))):
                path=Path(document['path'])
                assert path.is_relative_to(R) and path.stat().st_size<32*1024*1024
                if str(path) not in seen:
                    assert digest(path)==document['sha256'];seen[str(path)]=document['sha256'];pending.append(read(path))
            for child in document.values():inspect(child)
        elif isinstance(document,list):
            for child in document:inspect(child)
    while pending:
        assert len(seen)<256
        inspect(pending.pop())
    return sorted(roots),seen


def main():
    summary=P/'refrozen_preparation_summary_after_submission_log_correction.json'
    roster=P/'publication_roster_after_submission_log_correction.json'
    assert digest(summary)=='1f3ee3cd2407c4ccc6a6395b76718a43f1193ea4fa3bf9af4f77cd509eca74d9'
    assert digest(roster)=='a4d86b128c21b0b5b85bc5d281d4fe9d773c1afa8f8b3937eaeacbcc851e55b5'
    listed=read(roster)['files'];assert len(listed)==162 and len({item['absolute_path'] for item in listed})==162
    before={item['absolute_path']:record(item['absolute_path']) for item in listed}
    assert all(before[item['absolute_path']]['sha256']==item['sha256'] and before[item['absolute_path']]['bytes']==item['bytes'] for item in listed)
    old=read(P/'first_150_file_freeze_preservation_after_log_path_correction.json')['files'];assert len(old)==150
    for item in old.values():assert digest(item['preserved_path'])==item['sha256'] and Path(item['preserved_path']).stat().st_size==item['original_bytes']
    source_hashes={};facts=[];tests=[];probes=[]
    for index,item in enumerate(value['packets']):
        packet,unit=Path(item['packet']),Path(item['data_unit'])
        for folder in (packet,unit):
            assert folder.resolve(strict=True)==folder and not folder.is_symlink()
            assert all(not p.is_symlink() and len(p.name.encode())<=255 for p in folder.rglob('*'))
            assert len(folder.name.encode())<=255
            assert all((folder/name).is_file() for name in ('experiment_background.md','manifest.yaml','config_resolved.yaml','command.txt'))
        assert not any((folder/name).exists() for folder in (packet,unit) for name in ('submission_intent.json','submission_receipt.json','release_intent.json','release_receipt.json','owned_allocation_inputs.json','configured_isolation_pre_release_inputs.json','result.json','failure.json','resolved_device_binding.json'))
        if item['mode']!='correctness':
            request=read(packet/'confirmation_request.json')
            assert request['formal_confirmation_plan_present'] is False and request['correctness_receipt_not_yet_available'] is True
            assert request['job_id'] is request['gpu_uuid'] is None and request['timing_protocol']['rounds']==7 and request['timing_protocol']['full_updates_per_arm_per_round']==10
            assert not any((unit/name).exists() for name in ('measurement_plan.json','measurement_prepared.json','code'))
            assert not any((packet/name).exists() for name in ('held_submission_command.json','job.sbatch'))
            facts.append({'copies':item['copies'],'scope':'generic confirmation request only','formal_plan_present':False});continue
        plan=read(unit/'measurement_plan.json');assert plan==read(packet/'resolved_measurement_plan.json')
        assert plan['mode']=='correctness' and plan['rounds']==7 and plan['iterations']==10
        assert plan['gpu_uuid'] is None and all(plan['configured_isolation']['physical_identity'][key] is None for key in ('gpu_uuid','pci_address','physical_minor'))
        assert plan['copy_seeds']==list(range(1,item['copies']+1))
        candidate,reference=plan['configuration'],plan['reference_configuration']
        expected=copy.deepcopy(reference);expected['environment'].update(cache_room_backgrounds=True,render_final_two_only=True);assert candidate==expected
        assert reference['environment']['cache_room_backgrounds'] is reference['environment']['render_final_two_only'] is False
        for config in (candidate,reference):
            learner=config['learner'];assert (learner['n_copies'],learner['n_envs'],learner['num_steps'],learner['update_epochs'],learner['num_minibatches'],learner['observation_warmup_iterations'],learner['total_timesteps'])==(item['copies'],128,128,4,4,50,2000000000)
        helper=Path(plan['helpers']['source_root']);source=Path(plan['source_root'])
        for root,commit in ((helper,item['helper_commit']),(source,item['numerical_commit'])):
            command=['git','--no-optional-locks','-C',str(root),'status','--porcelain=v1','--untracked-files=all'];result=subprocess.run(command,capture_output=True,text=True,timeout=20);assert result.returncode==0 and not result.stdout
            assert subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip()==commit
            detached=subprocess.run(['git','-C',str(root),'symbolic-ref','--quiet','HEAD'],capture_output=True);assert detached.returncode==1
        assert len(plan['helpers']['files'])==20
        for relative,checksum in plan['helpers']['files'].items():
            raw=subprocess.check_output(['git','-C',str(helper),'show',item['helper_commit']+':'+relative])
            assert hashlib.sha256(raw).hexdigest()==digest(helper/relative)==digest(unit/'code'/Path(relative).name)==checksum
            source_hashes[str(helper/relative)]=checksum
        for group in ('core','paired_source'):
            for relative,checksum in plan[group]['files'].items():
                if isinstance(checksum,dict):checksum=checksum['sha256']
                assert digest(source/relative)==checksum;source_hashes[str(source/relative)]=checksum
        command=read(packet/'held_submission_command.json');assert command['job_id'] is command['gpu_uuid'] is None
        expected_command=['sbatch','--parsable','--hold','--no-requeue','--nodes=1','--ntasks=1','--cpus-per-task=8','--mem=64G','--time=03:00:00','--partition=gpu','--account=sl5nw_base','--qos=csresnolim','--nodelist=jaguar03','--reservation=sl5nw_168','--gres=gpu:nvidia_rtx_a4500:1',str(packet/'job.sbatch')]
        assert command['command']==expected_command
        current=(packet/'job.sbatch').read_text().splitlines();original=(packet/'job_before_campaign_submission_log_paths.sbatch').read_text().splitlines()
        assert len(current)==len(original)
        changes=[(a,b) for a,b in zip(original,current) if a!=b]
        assert len(changes)==2 and all(a.startswith('#SBATCH --output=') or a.startswith('#SBATCH --error=') for a,b in changes)
        assert all(str(R/'slurm/jobs')+'/' in b for a,b in changes)
        gate_lines=[line for line in current if line.startswith('/usr/bin/python3 -I -S -B ')]
        srun_lines=[line for line in current if line.startswith('exec srun ')]
        assert len(gate_lines)==len(srun_lines)==1 and current.index(gate_lines[0])<current.index(srun_lines[0])
        gate_argv=shlex.split(gate_lines[0]);assert gate_argv==['/usr/bin/python3','-I','-S','-B',str(packet/'wait_for_owned_inputs.py'),'--unit',str(unit),'--plan-sha256',digest(unit/'measurement_plan.json'),'--helper-commit',item['helper_commit']]
        assert 'CUDA_VISIBLE_DEVICES' not in '\n'.join(current) and 'SLURM_JOB_GPUS=' not in '\n'.join(current)
        subprocess.run(['bash','-n',str(packet/'job.sbatch')],check=True)
        for line in (packet/'source_checksums.sha256').read_text().splitlines():
            checksum,path=line.split(None,1);assert digest(path.strip())==checksum
        spec=importlib.util.spec_from_file_location('barrier_independent_'+str(index),packet/'test_wait_for_owned_inputs.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        suite=unittest.defaultTestLoader.loadTestsFromTestCase(module.BarrierTests);output=io.StringIO();result=unittest.TextTestRunner(stream=output,verbosity=2).run(suite)
        log=write('barrier_copies_'+str(item['copies'])+'.log',output.getvalue().encode());assert result.wasSuccessful(),output.getvalue()
        tests.append({'copies':item['copies'],'tests_run':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),'log':log})
        historical,chain=historical_git_roots(plan)
        write('historical_git_allowlist_copies_'+str(item['copies'])+'.json',{'roots':historical,'hash_bound_chain':chain,'read_only_git_operations_only':True})
        argv=['/usr/bin/python3','-I','-S','-B','-c',PROBE,str(unit),json.dumps(historical)]
        result=subprocess.run(argv,capture_output=True,text=True,timeout=120)
        returned=write('cold_record_gate_copies_'+str(item['copies'])+'.json',{'command':argv,'returncode':result.returncode,'stdout':result.stdout,'stderr':result.stderr})
        assert result.returncode==0,result.stderr
        probes.append({'copies':item['copies'],'receipt':returned,'observed':json.loads(result.stdout)})
        facts.append({'copies':item['copies'],'plan_sha256':digest(unit/'measurement_plan.json'),'helper_files':20,'only_two_environment_flags_changed':True,'full_original_learner_configuration':True,'job_id':None,'gpu_uuid':None,'held_shape':expected_command[1:-1],'barrier_before_numeric_srun':True,'release_receipt_validated_before_numeric_work':True})
    after={item['absolute_path']:record(item['absolute_path']) for item in listed};assert before==after
    assert all(digest(path)==checksum for path,checksum in source_hashes.items())
    assert not BLOCKED and not any(name.split('.')[0] in FORBIDDEN for name in sys.modules)
    final={'schema_version':1,'kind':'independent_common_a4500_correctness_packets_and_startup_review','approved':True,
        'actual_job_id':os.environ['SLURM_JOB_ID'],'actual_step_id':os.environ['SLURM_STEP_ID'],'at_utc':datetime.now(timezone.utc).isoformat(),
        'index':record(INDEX),'summary':record(summary),'publication_roster':record(roster),'frozen_paths_checked':len(listed),
        'original_payloads_preserved':150,'before':before,'after':after,'source_hashes_unchanged':source_hashes,
        'packet_facts':facts,'barrier_tests':tests,'cold_actual_record_and_asset_checks':probes,'blocked_operations':BLOCKED,
        'actual_current_physical_gpu_binding_proven':False,'actual_job_attempts':0,'numerical_imports':False,'gpu_operations':0,
        'new_jobs_submitted':0,'jobs_released_or_cancelled':0,'scientific_credit':0,'case_credit':0,'idea_credit':0,'admission_granted':False,
        'limitations':['This review authorizes packet preparation only; root owns publication and one held submission.',
            'Every actual launch still requires exact owned submission/release sidecars and same-process physical binding.',
            'Confirmation requests contain no executable formal plan; actual correctness and review remain required.']}
    receipt=write('review.json',final)
    print(json.dumps({'approved':True,'receipt':receipt,'tests':sum(row['tests_run'] for row in tests),'cold_full_record_and_asset_probes':len(probes),'frozen_files':len(listed)}),flush=True)


try:main()
except BaseException as error:
    write('failure.json',{'error':repr(error),'traceback':traceback.format_exc(),'actual_job_id':os.environ.get('SLURM_JOB_ID'),'actual_step_id':os.environ.get('SLURM_STEP_ID'),'blocked':BLOCKED,'gpu_operations':0,'job_actions':0})
    raise
