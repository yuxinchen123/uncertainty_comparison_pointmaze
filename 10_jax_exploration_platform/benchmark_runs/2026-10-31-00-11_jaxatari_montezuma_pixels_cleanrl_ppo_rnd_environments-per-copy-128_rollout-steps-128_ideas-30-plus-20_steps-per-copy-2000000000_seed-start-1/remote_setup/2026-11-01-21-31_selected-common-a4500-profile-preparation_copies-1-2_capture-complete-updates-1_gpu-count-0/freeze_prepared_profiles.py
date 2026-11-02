"""Freeze actual prepared profile records and verify the staged source/configuration contracts."""
import ast,datetime,hashlib,json,os,stat,subprocess
from pathlib import Path
P=Path(__file__).resolve().parent;R=P.parents[1]
value=json.loads((P/'result.json').read_bytes())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
ref=lambda p:{'path':str(p),'sha256':sha(p)}
helper=Path('/p/rlprojects/RND_worktrees/2026-11-01-21-25_selected-a4500-profile-source_commit-fdb35f886ad01c6d4ed387dc55b172d0221a9853')
prefix='10_jax_exploration_platform/scripts/atari_pixel_campaign/'
commit='fdb35f886ad01c6d4ed387dc55b172d0221a9853'
old='9fcc43a478b93d731df780805fd7c3c514cdf67f'
assert subprocess.check_output(['git','-C',str(helper),'rev-parse','HEAD'],text=True).strip()==commit
assert not subprocess.check_output(['git','-C',str(helper),'status','--porcelain=v1','--untracked-files=all'])
source=(helper/prefix/'profile_selected.py').read_text()
def function(text,name):
 tree=ast.parse(text)
 matches=[node for node in tree.body if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef)) and node.name==name]
 assert len(matches)==1
 return ast.dump(matches[0],include_attributes=False)
old_source=subprocess.check_output(['git','-C',str(helper),'show',old+':'+prefix+'profile_selected.py'],text=True)
assert function(source,'_capture_core')==function(old_source,'_capture_core')
checks=[]
for row in value['records']:
 packet=Path(row['packet']);unit=Path(row['unit'])
 plan=json.loads((unit/'measurement_plan.json').read_bytes());manifest=json.loads((packet/'manifest.yaml').read_bytes())
 assert plan['helpers']['source_commit']==commit and len(plan['helpers']['files'])==22
 assert ref(unit/'measurement_plan.json')==row['plan']
 assert plan['configuration']['learner']['n_copies']==row['copies'] and plan['copy_seeds']==list(range(1,row['copies']+1))
 assert plan['profile_updates']==plan['minimum_profile_updates']==1 and plan['maximum_active_seconds']==3600
 assert plan['gpu_uuid'] is None and manifest['job_id'] is None and manifest['gpu_uuid'] is None and manifest['submitted'] is False
 assert plan['accepted_confirmation_requires_common_reader'] is True and 'common_correctness_inputs' not in plan
 assert all((unit/n).is_file() for n in ('experiment_background.md','manifest.yaml','config_resolved.yaml','command.txt','progress.json'))
 assert not any(os.path.lexists(unit/n) for n in ('owned_allocation_inputs.json','configured_isolation_pre_release_inputs.json','resolved_device_binding.json','hardware.json','result.json','failure.json','capture_result.json','trace'))
 assert not any(os.path.lexists(packet/n) for n in ('submission_intent.json','submission_receipt.json','release_intent.json','release_receipt.json'))
 tests=json.loads((packet/'barrier_tests_final_receipt.json').read_bytes())
 assert tests['tests_run']==22 and tests['failures']==tests['errors']==0 and tests['returncode']==0
 assert tests['step_id']=='131' and tests['job_id']=='6783003'
 declared=json.loads((packet/'immutable_artifacts.json').read_bytes())['files']
 for path,record in declared.items():
  p=Path(path);assert p.is_file() and not p.is_symlink() and sha(p)==record['sha256'] and stat.S_IMODE(p.stat().st_mode)==record['mode']
 for path in list(packet.rglob('*'))+list(unit.rglob('*')):
  assert len(path.name.encode())<=255 and not path.is_symlink()
 held=json.loads((packet/'held_submission_command.json').read_bytes())['command']
 assert held[:4]==['sbatch','--parsable','--hold','--no-requeue'] and '--cpus-per-task=8' in held and '--mem=64G' in held and '--gres=gpu:nvidia_rtx_a4500:1' in held
 script=(packet/'job.sbatch').read_text()
 assert script.index('/usr/bin/python3 -I -S -B')<script.index('exec srun')
 assert 'code/profile_selected.py --prepared-unit' in script and 'code/measure_selected.py' not in script
 assert all(word not in script for word in ('CUDA_VISIBLE_DEVICES=','SLURM_JOB_GPUS=','SLURM_STEP_GPUS='))
 checks.append({'copies':row['copies'],'plan':row['plan'],'startup_controls':22,'five_data_standard_records':True,
  'held_command':ref(packet/'held_submission_command.json'),'job_script':ref(packet/'job.sbatch'),'no_launch_artifacts':True,
  'helper_files':22,'basename_bytes':len(packet.name.encode())})
files={str(p):{'sha256':sha(p),'bytes':p.stat().st_size,'mode':stat.S_IMODE(p.stat().st_mode)}
 for folder in [P]+[Path(row[key]) for row in value['records'] for key in ('packet','unit')]
 for p in sorted(folder.rglob('*')) if p.is_file()}
out=P/'profile_packets_source_freeze.json'
record={'schema_version':1,'kind':'prepared_selected_common_a4500_profile_packet_freeze','passed':True,
 'at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'helper_commit':commit,'files':files,
 'checks':checks,'capture_core_AST_equal_to_published_9fcc':True,'preparation_cpu_job':'6783003','preparation_cpu_step':'131',
 'scientific_credit':0,'actual_gpu_or_profile_actions':0,'actual_job_actions':0,'admission_granted':False,
 'limits':['no capture was executed','tool target-host hash/version must be rechecked by the actual executor','fresh owned scheduler/resource/reservation evidence is still required before actual submission','the H100 common baseline and both-hardware profile/packing phase are not complete']}
with out.open('x') as f:json.dump(record,f,indent=2,sort_keys=True);f.write('\n');f.flush();os.fsync(f.fileno())
roster=P/'profile_packets_publication_roster.json'
all_paths=[Path(p) for p in files]+[out]
r={'schema_version':1,'kind':'selected_common_profile_packet_publication_roster','paths':[
 {'path':str(p.relative_to(R.parents[2])),'absolute_path':str(p),'sha256':sha(p),'bytes':p.stat().st_size,'mode':stat.S_IMODE(p.stat().st_mode)} for p in sorted(all_paths)],'include_all_five_data_records':True}
with roster.open('x') as f:json.dump(r,f,indent=2,sort_keys=True);f.write('\n');f.flush();os.fsync(f.fileno())
print(json.dumps({'passed':True,'freeze':ref(out),'roster':ref(roster),'roster_paths_plus_itself':len(all_paths)+1,'checks':checks}))
