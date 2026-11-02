"""Prepare two fresh profile units from genuine accepted A4500 confirmations; never execute them."""
import ast,ctypes,datetime,fcntl,hashlib,importlib.abc,importlib.util,json,os,shlex,stat,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
R=ROOT.parents[1]
COMMIT='fdb35f886ad01c6d4ed387dc55b172d0221a9853'
HELPER=Path('/p/rlprojects/RND_worktrees/2026-11-01-21-25_selected-a4500-profile-source_commit-'+COMMIT)
PREFIX='10_jax_exploration_platform/scripts/atari_pixel_campaign/'
S=HELPER/PREFIX
OLD=Path('/p/rlprojects/RND_worktrees/2026-11-01-17-55_a4500-common-confirmation_prior-correctness-required_rounds-7_complete-updates-per-arm-10_copies-1-2/.review_metadata/prepare_common_confirmation_packets.py')
spec=importlib.util.spec_from_file_location('prior_metadata_packet_functions',OLD);base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
read,sha,ref,put,capture=base.read,base.sha,base.ref,base.put,base.capture
TOOL=R/'remote_setup/2026-11-01-21-17_nsight-systems-tool-version-and-hash_existing-cpu-allocation-6783003_processor-count-1_host-memory-mib-256_gpu-count-0_shared-read-lock/tool_record.json'
D28=Path('/p/rlprojects/RND_worktrees/atari-cleanrl-source-d28e673')
NUM={'jax','jaxlib','numpy','torch','flax','optax','scipy','cupy','jaxtari','exploration_platform','pynvml'}

def startup(packet,unit,plan,old,review):
    """Retain the exact absent-only wait and actual release reader, binding profile fields."""
    source=(old/'wait_for_owned_inputs.py').read_text()
    first=source.index('MAX_WAIT_SECONDS =');last=source.index('PREFIX = ',first)
    constants='MAX_WAIT_SECONDS = 180\nEXPECTED_QUALIFICATION_INPUTS = '+repr(plan['target_qualification_inputs'])+'\nEXPECTED_PLAN_INPUTS = '+repr(plan['inputs'])+'\nEXPECTED_BASELINE = '+repr(plan['accepted_baseline_specification'])+'\nEXPECTED_REVIEW = '+repr(ref(review))+'\n'
    source=source[:first]+constants+source[last:]
    first=source.index('    require(digest == expected_plan');last=source.index('    pre, pre_sha = read_json(',first)
    checks='''    require(digest == expected_plan and ready == {"plan_sha256": expected_plan}
        and plan["helpers"]["source_commit"] == expected_helper and plan["gpu_uuid"] is None
        and plan.get("kind") == "selected_configuration_profile"
        and "mode" not in plan and "rounds" not in plan and "direct_binding" not in plan
        and "target_qualification" not in plan and "final_confirmation_protocol" not in plan
        and "completed_study" not in plan and "common_correctness_inputs" not in plan
        and plan.get("target_qualification_inputs") == EXPECTED_QUALIFICATION_INPUTS
        and plan.get("inputs") == EXPECTED_PLAN_INPUTS
        and plan.get("accepted_baseline_specification") == EXPECTED_BASELINE
        and plan.get("accepted_confirmation_requires_common_reader") is True
        and type(plan.get("profile_updates")) is int and plan["profile_updates"] == 1
        and type(plan.get("minimum_profile_updates")) is int and plan["minimum_profile_updates"] == 1
        and type(plan.get("maximum_active_seconds")) is float and plan["maximum_active_seconds"] == 3600.0,
        "frozen selected-baseline profile identity changed")
    for name, field in (("configuration", "configuration"), ("runtime_environment", "runtime_environment"),
                        ("asset_record", "asset_record"), ("configured_isolation_profile", None),
                        ("profile_tool_record", "profile_tool"), ("accepted_baseline_specification", "accepted_baseline_specification")):
        record = EXPECTED_PLAN_INPUTS[name]
        value, actual_sha = read_json(record["path"])
        require(actual_sha == record["sha256"] and value == (plan[field] if field is not None else plan["configured_isolation"]["profile"]),
                "a full planned profile input changed")
    for record in (*EXPECTED_PLAN_INPUTS.values(), *EXPECTED_QUALIFICATION_INPUTS.values()):
        unused, actual_sha = read_json(record["path"])
        require(actual_sha == record["sha256"], "an exact accepted profile prerequisite changed")
    accepted, unused = read_json(EXPECTED_PLAN_INPUTS["accepted_confirmation"]["path"])
    approved, review_sha = read_json(EXPECTED_REVIEW["path"])
    require(review_sha == EXPECTED_REVIEW["sha256"] and approved.get("approved") is True
        and approved.get("passed") is True and approved.get("result") == EXPECTED_PLAN_INPUTS["accepted_confirmation"]
        and approved.get("actual_terminal_state") == "COMPLETED" and approved.get("actual_exit_code") == "0:0"
        and accepted.get("measurement_mode") == "confirmation" and accepted.get("decision") == "accepted",
        "the actual accepted baseline has no successful independent terminal review")
'''
    source=source[:first]+checks+source[last:]
    source=source.replace('common_candidate_confirmation_owned_input','selected_baseline_profile_owned_input')
    put(packet/'wait_for_owned_inputs.py',source.encode(),True)
    tests=(old/'test_wait_for_owned_inputs.py').read_text()
    tests=tests.replace("self.plan['common_correctness_inputs']['result']['sha256']='1'*64","self.plan['accepted_baseline_specification']['confirmation_record']='/changed/result.json'")
    tests=tests.replace('common_confirmation_startup_barrier_metadata_tests','selected_profile_startup_barrier_metadata_tests')
    tests=tests.replace('test_correctness_cannot_use_confirmation_startup','test_measurement_cannot_use_profile_startup')
    tests=tests.replace('test_coherent_changed_common_correctness_reference_refused','test_coherent_changed_accepted_confirmation_refused')
    put(packet/'test_wait_for_owned_inputs.py',tests.encode(),True)

def prepare_one(copies,approval):
    """Create a new metadata packet and stage the published profile's full helper closure."""
    approved=read(approval);previous_unit=Path(approved['result']['path']).parent
    old=R/'remote_setup'/previous_unit.name;previous=read(previous_unit/'measurement_plan.json')
    assert approved['approved'] is True and approved['passed'] is True and approved['actual_terminal_state']=='COMPLETED' and approved['actual_exit_code']=='0:0'
    assert ref(previous_unit/'result.json')==approved['result']
    hours={1:2,2:3}[copies]
    name='2026-11-01-21-31_selected-common-a4500-kernel-profile_cache-backgrounds-true_render-final-two-only-true_copies-'+str(copies)+'_environments-128_rollout-steps-128_epochs-4_minibatches-4_warmup-rollouts-50_capture-updates-1_seed-start-1_hours-'+str(hours)
    assert len(name.encode())<=255
    packet,unit=R/'remote_setup'/name,R/'data'/name
    assert not os.path.lexists(packet) and not os.path.lexists(unit)
    packet.mkdir()
    put(packet/'preparation_intent.json',{'kind':'selected_profile_metadata_preparation','helper_commit':COMMIT,'unit':str(unit),'accepted_result':approved['result'],'actual_review':ref(approval),'new_jobs':0,'retry_permitted':False})
    for filename,key in (('candidate.json','configuration'),('runtime.json','runtime_environment'),('assets.json','asset_record'),('profile.json','configured_isolation_profile')):
        source=Path(previous['inputs'][key]['path']);assert ref(source)==previous['inputs'][key]
        put(packet/filename,source.read_bytes(),True)
    put(packet/'study_registry.json',(R/'study_registry.json').read_bytes(),True)
    put(packet/'profile_tool.json',TOOL.read_bytes(),True)
    baseline={'source_commit':previous['source_commit'],'hardware_class':previous['hardware_class'],'runtime_environment':previous['runtime_environment'],'correctness_record':previous['common_correctness_inputs']['result']['path'],'confirmation_record':str(previous_unit/'result.json'),'configured_isolation_profile':previous['configured_isolation']['profile'],'independent_confirmation_review':ref(approval)}
    put(packet/'accepted_baseline.json',baseline)
    command=['/usr/bin/python3','-I','-S','-B',str(S/'profile_selected.py'),'--prepare-only','--source-root',str(D28),'--source-commit',previous['source_commit'],'--configuration',str(packet/'candidate.json'),'--runtime-environment',str(packet/'runtime.json'),'--asset-record',str(packet/'assets.json'),'--study-registry',str(packet/'study_registry.json'),'--accepted-baseline-specification',str(packet/'accepted_baseline.json'),'--profile-tool-record',str(packet/'profile_tool.json'),'--jaxtari-source-root',previous['jaxtari_source_root'],'--run-dir',str(R),'--unit-id',name,'--hardware-class','jaguar03_rtx_a4500','--configured-isolation-profile',str(packet/'profile.json'),'--target-qualification-record',previous['target_qualification_inputs']['result']['path'],'--target-qualification-review',previous['target_qualification_inputs']['review']['path'],'--seed-start','1','--minimum-updates','1','--maximum-active-seconds','3600','--python',previous['python']]
    held=read(old/'held_submission_command.json')['command']
    held=[str(packet/'job.sbatch') if value==str(old/'job.sbatch') else value for value in held]
    put(packet/'held_submission_command.json',{'schema_version':1,'kind':'prepared_unsubmitted_selected_profile_held_command','command':held,'job_id':None,'gpu_uuid':None,'new_jobs_submitted':0})
    put(packet/'command.txt',(shlex.join(held)+'\n').encode(),True)
    config={'configuration':previous['configuration'],'copy_seeds':list(range(1,copies+1)),'runtime_environment':previous['runtime_environment'],'capture_complete_updates':1,'profile_active_seconds_limit':3600,'requested_hours':hours,'requested_processors':8,'requested_host_memory_gib':64,'requested_typed_a4500_gpus':1}
    put(packet/'config_resolved.yaml',config)
    put(packet/'manifest.yaml',{'schema_version':1,'kind':'selected_common_a4500_baseline_profile_packet','run_id':name,'data_unit':str(unit),'helper_commit':COMMIT,'helper_root':str(HELPER),'source_commit':previous['source_commit'],'accepted_confirmation':approved['result'],'accepted_confirmation_review':ref(approval),'target_qualification_inputs':previous['target_qualification_inputs'],'tool_record':ref(packet/'profile_tool.json'),'requested_hours':hours,'requested_cpus':8,'requested_host_memory_gib':64,'requested_gpu_type':'nvidia_rtx_a4500','requested_gpu_count':1,'job_id':None,'gpu_uuid':None,'submitted':False,'capture_complete_updates':1,'case_credit':0,'idea_credit':0,'packing_credit':0,'final_training_credit':0,'admission_granted':False,'both_hardware_common_phase_complete':False})
    text='## Purpose\n\nProfile the independently accepted common A4500 baseline. This is local hardware profiling before new adaptive ideas; it does not establish the H100 common baseline or choose new ideas.\n\n## Key hyperparameters\n\n'+str(copies)+' independent policy copies, seeds '+str(list(range(1,copies+1)))+', 128 environments per policy, 128 rollout steps, four epochs, four minibatches, fifty normalization-warmup rollouts and the full two billion transition annealing schedule. Capture one complete update after compilation stabilizes. Request eight processors, 64 GiB and '+str(hours)+' hours on one typed A4500. The existing active Nsight limit is 3600 seconds.\n\n## Code and config changes\n\nOnly cached room backgrounds and rendering the last two frames differ from the original baseline. The 22-file profile closure includes the accepted common-confirmation reader. Numerical source and the capture function stay unchanged. The immutable plan has no future job or physical GPU identity; a 180-second absent-only startup gate requires the real successful release receipt before numerical srun. Positive native kernel and API summaries remain required.\n\n## Git state\n\nHelper source: '+COMMIT+'. Numerical source: '+previous['source_commit']+'. Exact source, configuration, assets, tool and accepted evidence hashes are recorded separately.\n'
    put(packet/'experiment_background.md',text.encode(),True)
    with (R/'queue/cpu_monitor_large_artifact_read.lock').open('r+') as lock:
        st=os.fstat(lock.fileno());assert stat.S_ISREG(st.st_mode) and st.st_uid==os.getuid()
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX)
        capture(packet,'actual_preparation',['/usr/bin/python3','-I','-S','-B',str(ROOT/'guarded_prepare_child.py'),json.dumps(command)])
    plan=read(unit/'measurement_plan.json')
    assert plan['configuration']==previous['configuration'] and plan['copy_seeds']==list(range(1,copies+1))
    assert plan['helpers']['source_commit']==COMMIT and len(plan['helpers']['files'])==22 and plan['gpu_uuid'] is None
    assert plan['accepted_confirmation_requires_common_reader'] is True and 'common_correctness_inputs' not in plan
    assert plan['profile_updates']==1 and plan['configured_isolation']['consumer']=='component_profile'
    put(unit/'progress.json',{'record':'selected_configuration_profile','state':'prepared_not_submitted','completed':False,'submitted':False,'profile_updates':1,'completed_profile_updates':0,'job_id':None,'gpu_uuid':None})
    put(packet/'resolved_measurement_plan.json',(unit/'measurement_plan.json').read_bytes(),True)
    put(packet/'preparation_context.json',{'unit':str(unit),'packet':str(packet),'helper_root':str(HELPER),'helper_commit':COMMIT,'copies':copies,'plan_sha256':sha(unit/'measurement_plan.json')})
    result=read(previous_unit/'result.json');identity=result['execution_identity']
    put(packet/'profile_budget.json',{'kind':'conservative_profile_preparation_walltime_basis','accepted_result':approved['result'],'accepted_review':ref(approval),'actual_prior_build_to_completion_seconds':result['build_to_completion_seconds'],'actual_prior_execution_seconds':identity['finished_at_seconds']-identity['started_at_seconds'],'actual_prior_candidate_seconds_per_update':[row['candidate']['seconds_per_update'] for row in result['rounds']],'actual_prior_reference_seconds_per_update':[row['reference']['seconds_per_update'] for row in result['rounds']],'prior_requested_job_hours':hours,'requested_job_hours':hours,'profile_active_seconds_limit':3600,'complete_capture_updates':1,'basis':'retain the earlier conservative whole-job limit; the prior confirmation contained both configurations and seven ten-update pairs','new_profile_duration_prediction':False})
    startup(packet,unit,plan,old,approval)
    capture(packet,'startup_barrier_tests',['/usr/bin/python3','-I','-S','-B',str(packet/'test_wait_for_owned_inputs.py')])
    helper_files={}
    for relative,expected in plan['helpers']['files'].items():
        source=HELPER/relative;staged=unit/'code'/Path(relative).name
        blob=subprocess.check_output(['git','-C',str(HELPER),'show',COMMIT+':'+relative]);entry=subprocess.check_output(['git','-C',str(HELPER),'ls-tree',COMMIT,'--',relative],text=True).strip().split()
        assert sha(source)==sha(staged)==hashlib.sha256(blob).hexdigest()==expected
        assert entry[0] in {'100644','100755'} and bool(source.stat().st_mode&0o111)==bool(int(entry[0],8)&0o111)
        helper_files[relative]={'sha256':expected,'git_blob':entry[2],'git_mode':entry[0],'staged_path':str(staged)}
    put(packet/'committed_helper_verification.json',{'passed':True,'helper_commit':COMMIT,'helper_root':str(HELPER),'count':22,'files':helper_files})
    script=(old/'job.sbatch').read_text().replace(str(old),str(packet)).replace(str(previous_unit),str(unit)).replace(sha(previous_unit/'measurement_plan.json'),sha(unit/'measurement_plan.json')).replace(previous['helpers']['source_commit'],COMMIT)
    script=script.replace('pixel-common-confirmation','pixel-selected-profile').replace('code/measure_selected.py','code/profile_selected.py').replace('2026-11-01-18-54_common-confirmation','2026-11-01-21-31_selected-profile')
    put(packet/'job.sbatch',script.encode(),True)
    immutable=[p for p in unit.rglob('*') if p.is_file() and p.name!='progress.json']+[packet/n for n in ('candidate.json','runtime.json','assets.json','profile.json','study_registry.json','accepted_baseline.json','profile_tool.json','wait_for_owned_inputs.py','job.sbatch')]
    put(packet/'source_checksums.sha256',''.join(sha(p)+'  '+str(p)+'\n' for p in sorted(immutable)).encode(),True)
    capture(packet,'shell_syntax',['bash','-n',str(packet/'job.sbatch')])
    assert not any(os.path.lexists(packet/n) for n in ('submission_intent.json','submission_receipt.json','release_intent.json','release_receipt.json')) and not os.path.lexists(unit/'owned_allocation_inputs.json')
    put(packet/'preparation_verification.json',{'kind':'actual_selected_common_profile_metadata_verification','passed':True,'cpu_job_id':'6783003','cpu_step_id':os.environ['SLURM_STEP_ID'],'helper_root':str(HELPER),'helper_commit':COMMIT,'plan':ref(unit/'measurement_plan.json'),'staged_helpers':22,'accepted_confirmation':approved['result'],'accepted_confirmation_review':ref(approval),'tool_record':ref(packet/'profile_tool.json'),'startup_barrier_tests':ref(packet/'barrier_tests_final_receipt.json'),'complete_capture_updates':1,'requested_job_hours':hours,'actual_gpu_operations':0,'actual_jobs_submitted':0,'scientific_credit':0,'admission_granted':False})
    files={str(p):{'sha256':sha(p),'bytes':p.stat().st_size,'mode':stat.S_IMODE(p.stat().st_mode)} for folder in (packet,unit) for p in sorted(folder.rglob('*')) if p.is_file()}
    put(packet/'immutable_artifacts.json',{'kind':'unsubmitted_selected_profile_static_artifacts','files':files,'job_id':None,'gpu_uuid':None,'submitted':False})
    return {'packet':str(packet),'unit':str(unit),'plan':ref(unit/'measurement_plan.json'),'verification':ref(packet/'preparation_verification.json'),'immutable_artifacts':ref(packet/'immutable_artifacts.json'),'copies':copies,'staged_helpers':22,'requested_hours':hours}

def main():
    """Require an owned CPU step and the clean exact published helper commit."""
    assert os.environ.get('SLURM_JOB_ID')=='6783003' and os.environ.get('SLURM_STEP_ID','').isdigit() and not NUM.intersection(sys.modules)
    assert subprocess.check_output(['git','-C',str(HELPER),'rev-parse','HEAD'],text=True).strip()==COMMIT
    assert not subprocess.check_output(['git','-C',str(HELPER),'status','--porcelain=v1','--untracked-files=all'])
    assert subprocess.run(['git','-C',str(HELPER),'symbolic-ref','-q','HEAD'],capture_output=True).returncode==1
    put(ROOT/'actual_allocation.json',{'job_id':'6783003','step_id':os.environ['SLURM_STEP_ID'],'pid':os.getpid(),'process_start_time':Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19],'cpu_affinity':sorted(os.sched_getaffinity(0)),'cgroup':Path('/proc/self/cgroup').read_text(),'requested_cpus':1,'requested_host_memory_mib':1024,'requested_gpus':0})
    records=[]
    for copies,prefix,expected in ((1,'20-35','c949d41603813be5344bd8d39a3a6ea357fb8cc45211e0e3cb9b5af80783794f'),(2,'20-37','6bd2243057e95c295279e725e17c9bca622f1e95061b5737643f842541fdc306')):
        matches=list((R/'remote_setup').glob('2026-11-01-'+prefix+'_common-a4500-confirmation-independent-raw-audit_copies-'+str(copies)+'*/review.json'))
        assert len(matches)==1 and sha(matches[0])==expected
        records.append(prepare_one(copies,matches[0]))
    assert not subprocess.check_output(['git','-C',str(HELPER),'status','--porcelain=v1','--untracked-files=all'])
    value={'passed':True,'records':records,'cpu_job_id':'6783003','cpu_step_id':os.environ['SLURM_STEP_ID'],'helper_commit':COMMIT,'metadata_only':True,'actual_jobs_submitted':0,'actual_gpu_operations':0}
    put(ROOT/'result.json',value);print(json.dumps(value))

if __name__=='__main__':main()
