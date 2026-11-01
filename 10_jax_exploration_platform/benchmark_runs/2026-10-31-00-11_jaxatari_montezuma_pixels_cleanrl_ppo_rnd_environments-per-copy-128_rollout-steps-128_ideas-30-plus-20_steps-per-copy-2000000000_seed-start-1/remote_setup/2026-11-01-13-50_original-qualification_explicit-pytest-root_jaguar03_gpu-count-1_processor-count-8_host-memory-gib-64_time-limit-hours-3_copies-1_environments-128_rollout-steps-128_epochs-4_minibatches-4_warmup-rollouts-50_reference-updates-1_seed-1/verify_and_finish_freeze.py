"""Freeze standard launch records after successful owned-compute metadata preparation."""
import ast
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import time

PACKET=Path(__file__).resolve().parent
C=json.loads((PACKET/'preparation_context.json').read_text())
UNIT=Path(C['unit']);SOURCE=Path(C['helper_root'])

def write(path,value):
    data=(json.dumps(value,indent=2,allow_nan=False)+'\n').encode()
    if path.exists():
        assert path.is_file() and not path.is_symlink() and path.read_bytes()==data, 'existing immutable static record differs: '+str(path)
        return
    with path.open('xb') as stream:stream.write(data);stream.flush();os.fsync(stream.fileno())

def receipt(path):return {'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'bytes':path.stat().st_size}

def fields(text):return dict(re.findall(r'(?:^|\s)([A-Za-z][A-Za-z0-9_]*)=([^\s]*)',text))

assert os.environ['SLURM_JOB_ID']=='6781231' and os.environ['SLURM_STEP_ID'].isdigit()
plan=json.loads((UNIT/'measurement_plan.json').read_text())
planref=receipt(UNIT/'measurement_plan.json')
assert plan['helpers']['source_commit']==C['helper_commit'] and plan['gpu_uuid'] is None
assert plan['configuration']==plan['reference_configuration']
assert plan['configuration']['learner']['n_copies']==1 and plan['configuration']['learner']['n_envs']==128
assert plan['configuration']['learner']['num_steps']==128 and plan['configuration']['learner']['observation_warmup_iterations']==50
assert plan['configuration']['learner']['total_timesteps']==2000000000
assert plan['configuration']['learner']['update_epochs']==4 and plan['configuration']['learner']['num_minibatches']==4
observed=json.loads((PACKET/'resource_observation.json').read_text())
node=fields(observed['observations']['node']['stdout']);partition=fields(observed['observations']['partition']['stdout']);reservation=fields(observed['observations']['reservation']['stdout'])
assert node['NodeName']=='jaguar03' and not any(word in node['State'] for word in ('DOWN','DRAIN','FAIL','MAINT'))
assert int(node['CPUEfctv'])-int(node['CPUAlloc'])>=8
assert int(node['RealMemory'])-int(node['AllocMem'])-int(node['MemSpecLimit'])>=64*1024
assert 'gpu:nvidia_rtx_a4500:8' in node['Gres']
allocated_pairs=[part.split('=',1) for part in node.get('AllocTRES','').split(',') if part]
assert len(allocated_pairs)==len({pair[0] for pair in allocated_pairs})
allocated=dict(allocated_pairs)
configured=dict(part.split('=',1) for part in node['CfgTRES'].split(','))
assert int(configured['gres/gpu'])-int(allocated.get('gres/gpu','0'))>=1
assert partition['State']=='UP' and partition['MaxTime']=='4-00:00:00'
assert reservation['ReservationName']=='sl5nw_168' and reservation['State']=='ACTIVE' and 'sl5nw' in reservation['Users'].split(',') and 'jaguar03' in reservation['Nodes'].split(',')
end=datetime.fromisoformat(reservation['EndTime']).replace(tzinfo=timezone.utc)
assert (end-datetime.now(timezone.utc)).total_seconds()>3*3600
import sys
sys.path.insert(0,str(UNIT/'code'))
import configured_measurement as support
pytest_command=[plan['python'],'-m','pytest',*support.qualification_pytest_arguments(plan)]
assert pytest_command[3:5]==['--rootdir',str(Path(plan['source_root'])/'10_jax_exploration_platform')]
q=shlex.quote
startup={
 'PYTHONNOUSERSITE':'1','PYTHONDONTWRITEBYTECODE':'1','PYTHONUNBUFFERED':'1',
 'JAX_ENABLE_X64':'true','JAX_DEFAULT_MATMUL_PRECISION':'highest','XLA_FLAGS':'--xla_gpu_exclude_nondeterministic_ops=true',
 'JAXTARI_SKIP_SPRITE_UPDATE':'1','XLA_PYTHON_CLIENT_PREALLOCATE':'false','XLA_PYTHON_CLIENT_MEM_FRACTION':'0.85',
 'OMP_NUM_THREADS':'8','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1','NUMEXPR_NUM_THREADS':'1',
 'XDG_DATA_HOME':plan['asset_record']['root'],'SLURM_CONF':'/etc/slurm/slurm.conf',
 'JAX_ENABLE_COMPILATION_CACHE':'false','JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES':''}
barrier=['/usr/bin/python3','-I','-S','-B',str(PACKET/'wait_for_owned_inputs.py'),'--unit',str(UNIT),'--plan-sha256',planref['sha256'],'--helper-commit',C['helper_commit']]
run=['srun','--nodes=1','--ntasks=1','--cpus-per-task=8','--cpu-bind=cores','--gres=gpu:nvidia_rtx_a4500:1','--kill-on-bad-exit=1',C['references']['python'],'-B',str(UNIT/'code/measure_selected.py'),'--prepared-unit',str(UNIT)]
job='\n'.join(['#!/bin/bash','#SBATCH --job-name=pixel-original-qualification','#SBATCH --output='+str(PACKET/'slurm-%j.out'),'#SBATCH --error='+str(PACKET/'slurm-%j.err'),'set -euo pipefail','umask 077','cd '+q(C['campaign']),*['export '+key+'='+q(value) for key,value in startup.items()],
 'unset JAX_COMPILATION_CACHE_DIR JAX_COMPILATION_CACHE_MAX_SIZE JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES',
 'unset PYTEST_ADDOPTS PYTEST_PLUGINS PYTEST_CURRENT_TEST PYTHONOPTIMIZE CLEANRL_REFERENCE_FILE',
 'export LD_LIBRARY_PATH="/usr/lib/nvidia:${LD_LIBRARY_PATH:-}"',
 'sha256sum --check --strict '+q(str(PACKET/'source_checksums.sha256')),
 shlex.join(barrier),'exec '+shlex.join(run),''])
assert (PACKET/'job.sbatch').read_text()==job, 'existing immutable job script differs'
command=['sbatch','--parsable','--hold','--no-requeue','--nodes=1','--ntasks=1','--cpus-per-task=8','--mem=64G','--time=03:00:00','--partition=gpu','--account=sl5nw_base','--qos=csresnolim','--nodelist=jaguar03','--reservation=sl5nw_168','--gres=gpu:nvidia_rtx_a4500:1',str(PACKET/'job.sbatch')]
write(PACKET/'held_submission_command.json',{'schema_version':1,'kind':'full_original_qualification_held_submission_command','argv':command,'job_id':None,'held':True,'released':False,'no_requeue':True,'submitted':False})
write(PACKET/'config_resolved.yaml',{'schema_version':1,'kind':'full_original_target_qualification_launch_configuration','configuration':plan['configuration'],'runtime_environment':plan['runtime_environment'],'startup_environment':startup,'driver_library_path_prefix':'/usr/lib/nvidia','preserve_scheduler_and_cuda_selectors':True,'shape':C['shape'],'frozen_original_pytest_command':pytest_command,'explicit_original_pytest_root':str(Path(plan['source_root'])/'10_jax_exploration_platform'),'startup_barrier':{'maximum_wait_seconds':180,'wait_only_for_final_absence':True,'interpreter':['/usr/bin/python3','-I','-S','-B'],'pre_release_filename':'configured_isolation_pre_release_inputs.json','final_filename':'owned_allocation_inputs.json','future_actual_job_id':None,'future_actual_gpu_uuid':None},'barrier_command':barrier,'numeric_step_command':run,'checkpoints':False,'scientific_credit':0,'case_credit':0,'idea_credit':0,'packing_credit':0,'admission_granted':False})
write(PACKET/'manifest.yaml',{'schema_version':1,'kind':'zero_credit_full_original_target_qualification_packet','unit_id':C['unit_id'],'campaign':C['campaign'],'data_unit':str(UNIT),'measurement_plan':planref,'frozen_original_pytest_command':pytest_command,'explicit_original_pytest_root':str(Path(plan['source_root'])/'10_jax_exploration_platform'),'previous_failed_job_preserved':'6782640','helper_source':{'root':C['helper_root'],'commit':C['helper_commit']},'numerical_source':{'root':C['numerical_root'],'commit':C['numerical_commit']},'resource_basis':C['resource_basis'],'actual_resource_observation':receipt(PACKET/'resource_observation.json'),'job_id':None,'gpu_uuid':None,'reservation_at_submission':'sl5nw_168','held_command':command,'qualification_result_filename':'target_qualification_result.json','controlled_full_original_work':{'normalization_warmup_rollouts':50,'agent_transitions_per_rollout':128*128,'normalization_warmup_agent_transitions':50*128*128,'full_reference_updates':1,'full_reference_agent_transitions':128*128},'test_fixture_work':'unquantified separate synthetic and partial fixture work; full commands, collection and outcomes are retained; no whole-job transition total claimed','scientific_credit':0,'case_credit':0,'idea_credit':0,'packing_credit':0,'throughput_credit':0,'admission_granted':False,'actual_independent_review_required_before_future_use':True})
for file in (PACKET/'wait_for_owned_inputs.py',PACKET/'test_wait_for_owned_inputs.py',PACKET/'prepare_packet.py',Path(__file__)):
 ast.parse(file.read_text(),filename=str(file))
syntax=subprocess.run(['bash','-n',str(PACKET/'job.sbatch')],capture_output=True,text=True)
assert syntax.returncode==0,syntax.stderr
assert subprocess.check_output(['git','-C',str(SOURCE),'rev-parse','HEAD'],text=True).strip()==C['helper_commit']
assert not subprocess.check_output(['git','-C',str(SOURCE),'status','--porcelain','--untracked-files=all'],text=True).strip()
assert not any((UNIT/name).exists() for name in ('owned_allocation_inputs.json','configured_isolation_pre_release_inputs.json','target_qualification_result.json','result.json'))
for folder in (PACKET,UNIT):
 for path in folder.rglob('*'):
  assert all(len(part.encode())<=255 for part in path.parts)
  assert path.resolve()==path and not path.is_symlink()
files=sorted([p for p in PACKET.rglob('*') if p.is_file()]+[p for p in UNIT.rglob('*') if p.is_file()],key=str)
records=[receipt(path) for path in files]
with (PACKET/'source_checksums.sha256').open('x') as stream:
 for item in records:stream.write(item['sha256']+'  '+item['path']+'\n')
write(PACKET/'immutable_artifacts.json',{'schema_version':1,'kind':'full_original_qualification_frozen_preparation_artifacts','files':records,'source_checksums':receipt(PACKET/'source_checksums.sha256'),'plan':planref,'runtime_owned_inputs_absent':True,'no_submission':True})
write(PACKET/'packet_verification.json',{'schema_version':1,'kind':'full_original_qualification_packet_author_verification','job_id':os.environ['SLURM_JOB_ID'],'step_id':os.environ['SLURM_STEP_ID'],'helper_commit':C['helper_commit'],'helper_clean':True,'plan':planref,'packet_manifest':receipt(PACKET/'manifest.yaml'),'artifact_manifest':receipt(PACKET/'immutable_artifacts.json'),'job_script':receipt(PACKET/'job.sbatch'),'startup_barrier':receipt(PACKET/'wait_for_owned_inputs.py'),'barrier_tests':receipt(PACKET/'barrier_tests_receipt.json'),'held_command':receipt(PACKET/'held_submission_command.json'),'immutable_file_count':len(records),'staged_helper_count':len(plan['helpers']['files']),'full_original_pytest_cases':48,'frozen_original_pytest_command':pytest_command,'explicit_original_pytest_root':str(Path(plan['source_root'])/'10_jax_exploration_platform'),'full_renderer':{'rooms':33,'cases':132,'synthetic_pixel_cases':11},'partition_up':True,'remaining_cpu_fits':True,'remaining_host_memory_fits':True,'one_available_typed_gpu_fits':True,'reservation_active_with_three_hours_remaining':True,'node_observed_state':node['State'],'maintenance_reason_observed':node.get('Reason'),'all_component_name_bytes_at_most_255':True,'longest_component_bytes':max(len(part.encode()) for file in files for part in file.parts),'shell_syntax_returncode':syntax.returncode,'no_actual_job_or_gpu_identity':True,'no_submission':True,'no_gpu_operations':True,'returncode':0})
print(json.dumps({'step':os.environ['SLURM_STEP_ID'],'plan_sha256':planref['sha256'],'manifest_sha256':receipt(PACKET/'immutable_artifacts.json')['sha256'],'job_script_sha256':receipt(PACKET/'job.sbatch')['sha256'],'immutable_files':len(records)}))
