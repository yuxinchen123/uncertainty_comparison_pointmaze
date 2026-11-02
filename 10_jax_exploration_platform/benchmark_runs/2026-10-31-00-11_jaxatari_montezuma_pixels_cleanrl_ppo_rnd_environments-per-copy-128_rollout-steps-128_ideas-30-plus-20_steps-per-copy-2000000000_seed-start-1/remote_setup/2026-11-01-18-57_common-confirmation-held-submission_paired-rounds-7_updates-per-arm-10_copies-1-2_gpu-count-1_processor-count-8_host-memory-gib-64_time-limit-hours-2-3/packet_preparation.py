"""Prepare two fresh intermediate confirmation packets; never submit or execute GPU work."""
import argparse,datetime,fcntl,hashlib,json,os,shlex,stat,subprocess,sys,time
from pathlib import Path
from zoneinfo import ZoneInfo

R=Path('/p/rlprojects/RND_worktrees/merge-atari-cleanrl/10_jax_exploration_platform/benchmark_runs/2026-10-31-00-11_jaxatari_montezuma_pixels_cleanrl_ppo_rnd_environments-per-copy-128_rollout-steps-128_ideas-30-plus-20_steps-per-copy-2000000000_seed-start-1')
W=Path(__file__).absolute().parent.parent
D28=Path('/p/rlprojects/RND_worktrees/atari-cleanrl-source-d28e673')
PREFIX='10_jax_exploration_platform/scripts/atari_pixel_campaign/'
CORRECTNESS_REVIEW=R/'remote_setup/2026-11-01-17-44-40_common-a4500-actual-correctness_independent-audit_jobs-6782946-6782947_copies-1-2_gpu-count-0'
BUDGET=W/'.review_metadata/common_confirmation_walltime_budget_6783003_35.json'
NUM={'jax','jaxlib','numpy','torch','flax','optax','scipy','cupy','jaxtari','exploration_platform'}

def read(path):
    """Read exact ordinary metadata without choosing between path aliases."""
    path=Path(path);assert path.is_absolute() and path.resolve(strict=True)==path and path.is_file()
    return json.loads(path.read_bytes())

def sha(path):
    """Hash source and evidence bytes incrementally."""
    value=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):value.update(block)
    return value.hexdigest()

def ref(path):
    """Keep the exact path and current content hash of an existing artifact."""
    return {'path':str(path),'sha256':sha(path)}

def put(path,value,raw=False):
    """Create new durable packet records and refuse every existing target."""
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    assert path.parent.resolve(strict=True)==path.parent and not path.exists() and not path.is_symlink()
    data=value if raw else (json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
    fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
    with os.fdopen(fd,'wb') as stream:stream.write(data);stream.flush();os.fsync(stream.fileno())
    fd=os.open(path.parent,os.O_RDONLY|os.O_DIRECTORY);os.fsync(fd);os.close(fd)
    return ref(path)

def capture(packet,label,command,input_bytes=None):
    """Retain every metadata command's actual output and status before parsing."""
    started=datetime.datetime.now(datetime.timezone.utc).isoformat()
    try:result=subprocess.run(command,input=input_bytes,capture_output=True,timeout=360,check=False)
    except BaseException as error:
        put(packet/(label+'.stdout'),getattr(error,'stdout',None) or b'',True)
        put(packet/(label+'.stderr'),getattr(error,'stderr',None) or b'',True)
        put(packet/(label+'_return.json'),{'command':command,'started_at_utc':started,'returncode':None,'error_type':type(error).__name__,'error':str(error)})
        raise
    stdout=put(packet/(label+'.stdout'),result.stdout,True);stderr=put(packet/(label+'.stderr'),result.stderr,True)
    put(packet/(label+'_return.json'),{'command':command,'started_at_utc':started,'returncode':result.returncode,'stdout':stdout,'stderr':stderr})
    assert result.returncode==0,(label,result.stderr.decode(errors='replace'))
    return result.stdout

def barrier(packet,unit,plan):
    """Extend only the proven absent-final-file barrier with frozen confirmation prerequisites."""
    old=R/'remote_setup/2026-11-01-15-09_common_cache-room-backgrounds-true_render-final-two-only-true_correctness_jaguar03_copies-1_environments-128_rollout-steps-128_epochs-4_minibatches-4_warmup-rollouts-50_seed-start-1'
    text=(old/'wait_for_owned_inputs.py').read_text();text=text.replace('EXPECTED_MODE = "correctness"','EXPECTED_MODE = "confirmation"')
    text=text.replace('common_candidate_correctness_owned_input','common_candidate_confirmation_owned_input').replace('frozen common-candidate correctness identity changed','frozen common-candidate confirmation identity changed')
    lines=[]
    for line in text.splitlines():
        if line.startswith('EXPECTED_QUALIFICATION_INPUTS = '):line='EXPECTED_QUALIFICATION_INPUTS = '+repr(plan['target_qualification_inputs'])
        if line.startswith('EXPECTED_PLAN_INPUTS = '):
            line='EXPECTED_PLAN_INPUTS = '+repr(plan['inputs'])+'\nEXPECTED_COMMON_INPUTS = '+repr(plan['common_correctness_inputs'])
        lines.append(line)
    text='\n'.join(lines)+'\n'
    oldline='        and plan.get("inputs") == EXPECTED_PLAN_INPUTS, "frozen common-candidate confirmation identity changed")'
    newline='''        and plan.get("inputs") == EXPECTED_PLAN_INPUTS
        and plan.get("common_correctness_inputs") == EXPECTED_COMMON_INPUTS
        and all(type(plan.get(key)) is int and plan[key] == value for key, value in (
            ("rounds", 7), ("minimum_updates_per_round", 10), ("iterations", 10))),
        "frozen common-candidate confirmation identity changed")'''
    assert oldline in text;text=text.replace(oldline,newline)
    anchor='    pre, pre_sha = read_json(unit / "configured_isolation_pre_release_inputs.json")'
    addition='''    for name, record in EXPECTED_COMMON_INPUTS.items():
        value, actual_sha = read_json(record["path"])
        require(actual_sha == record["sha256"], "actual common correctness or independent review changed")
        require(value.get("correctness_passed") is True if name == "result" else
            value.get("approved") is True and value.get("result") == EXPECTED_COMMON_INPUTS["result"]
            and value.get("actual_terminal_state") == "COMPLETED" and value.get("actual_exit_code") == "0:0",
            "the common correctness prerequisite has no successful independent terminal review")
'''
    assert anchor in text;text=text.replace(anchor,addition+anchor)
    put(packet/'wait_for_owned_inputs.py',text.encode(),True)
    tests=(old/'test_wait_for_owned_inputs.py').read_text().replace("=='6782816'","=='6783003'")
    tests=tests.replace('test_confirmation_cannot_use_correctness_startup','test_correctness_cannot_use_confirmation_startup').replace("self.plan['mode']='confirmation';self.rebind_changed_plan()","self.plan['mode']='correctness';self.rebind_changed_plan()")
    tests=tests.replace('common_correctness_startup_barrier_metadata_tests','common_confirmation_startup_barrier_metadata_tests')
    anchor="    def test_standalone_qualification_cannot_use_ordinary_startup(self):"
    added='''    def test_coherent_changed_common_correctness_reference_refused(self):
        self.plan['common_correctness_inputs']['result']['sha256']='1'*64;self.rebind_changed_plan()
        with self.assertRaises(ValueError):self.run_gate()
    def test_coherent_final_or_reduced_rounds_refused(self):
        self.plan['rounds']=11;self.rebind_changed_plan()
        with self.assertRaises(ValueError):self.run_gate()
'''
    assert anchor in tests;tests=tests.replace(anchor,added+anchor)
    put(packet/'test_wait_for_owned_inputs.py',tests.encode(),True)

def prepare_one(copies,helper_root,commit,stamp):
    """Freeze one actual-source packet after revalidating its real prior correctness inputs."""
    prior_name='2026-11-01-15-09_common_cache-room-backgrounds-true_render-final-two-only-true_correctness_jaguar03_copies-'+str(copies)+'_environments-128_rollout-steps-128_epochs-4_minibatches-4_warmup-rollouts-50_seed-start-1'
    prior_packet=R/'remote_setup'/prior_name;prior_unit=R/'data'/prior_name;previous=read(prior_unit/'measurement_plan.json')
    review=CORRECTNESS_REVIEW/('copies_'+str(copies)+'_independent_review.json');approved=read(review)
    assert approved['approved'] and approved['result']==ref(prior_unit/'result.json') and approved['actual_terminal_state']=='COMPLETED' and approved['actual_exit_code']=='0:0'
    budget=read(BUDGET)['copies'][str(copies)];hours=budget['requested_hours'];assert hours=={1:2,2:3}[copies]
    name=stamp+'_common_cache-room-backgrounds-true_render-final-two-only-true_confirmation_paired-rounds-7_updates-per-arm-10_copies-'+str(copies)+'_environments-128_rollout-steps-128_epochs-4_minibatches-4_warmup-rollouts-50_seed-start-1_hours-'+str(hours)+'_jaguar03'
    assert len(name.encode())<=255
    packet=R/'remote_setup'/name;unit=R/'data'/name
    assert not packet.exists() and not unit.exists();packet.mkdir();assert packet.resolve()==packet
    put(packet/'preparation_intent.json',{'kind':'new_intermediate_common_confirmation_metadata_preparation','helper_root':str(helper_root),'helper_commit':commit,'copies':copies,'unit':str(unit),'correctness':approved['result'],'correctness_review':ref(review),'new_jobs':0,'retry_permitted':False})
    copies_from={}
    for filename,key in (('candidate.json','configuration'),('reference.json','reference_configuration'),('runtime.json','runtime_environment'),('assets.json','asset_record'),('profile.json','configured_isolation_profile')):
        source=Path(previous['inputs'][key]['path']);assert ref(source)==previous['inputs'][key]
        copies_from[filename]=ref(source);put(packet/filename,source.read_bytes(),True)
    command=['/usr/bin/python3','-I','-S','-B',str(helper_root/PREFIX/'measure_selected.py'),'--prepare-only','--mode','confirmation','--source-root',str(D28),'--source-commit',previous['source_commit'],'--configuration',str(packet/'candidate.json'),'--reference-configuration',str(packet/'reference.json'),'--runtime-environment',str(packet/'runtime.json'),'--asset-record',str(packet/'assets.json'),'--jaxtari-source-root',previous['jaxtari_source_root'],'--run-dir',str(R),'--unit-id',name,'--hardware-class','jaguar03_rtx_a4500','--configured-isolation-profile',str(packet/'profile.json'),'--target-qualification-record',previous['target_qualification_inputs']['result']['path'],'--target-qualification-review',previous['target_qualification_inputs']['review']['path'],'--common-correctness-record',str(prior_unit/'result.json'),'--common-correctness-review',str(review),'--seed-start','1','--rounds','7','--minimum-updates-per-round','10','--python',previous['python']]
    with (R/'queue/cpu_monitor_large_artifact_read.lock').open('r+') as lock:
        assert os.fstat(lock.fileno()).st_uid==os.getuid();fcntl.flock(lock.fileno(),fcntl.LOCK_EX)
        capture(packet,'actual_preparation',command)
    plan=read(unit/'measurement_plan.json');plan_sha=sha(unit/'measurement_plan.json')
    with (R/'queue/cpu_monitor_large_artifact_read.lock').open('r+') as lock:
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX)
        proof_bytes=capture(packet,'actual_staged_common_correctness_preflight',['/usr/bin/python3','-I','-S','-B',str(unit/'code/common_confirmation.py'),'--plan-stdin'],(unit/'measurement_plan.json').read_bytes())
    preflight=json.loads(proof_bytes);assert preflight['passed'] and preflight['plan_sha256']==plan_sha
    put(packet/'common_correctness_preflight.json',preflight)
    assert plan['helpers']['source_commit']==commit and plan['common_correctness_inputs']=={'result':ref(prior_unit/'result.json'),'review':ref(review)}
    assert plan['configuration']==previous['configuration'] and plan['reference_configuration']==previous['reference_configuration'] and plan['copy_seeds']==list(range(1,copies+1))
    assert (plan['rounds'],plan['iterations'],plan['minimum_updates_per_round'])==(7,10,10) and plan['gpu_uuid'] is None and len(plan['helpers']['files'])==21
    put(packet/'resolved_measurement_plan.json',(unit/'measurement_plan.json').read_bytes(),True)
    put(packet/'preparation_context.json',{'unit':str(unit),'packet':str(packet),'helper_root':str(helper_root),'helper_commit':commit,'copies':copies,'plan_sha256':plan_sha,'target_qualification_inputs':plan['target_qualification_inputs'],'common_correctness_inputs':plan['common_correctness_inputs'],'formal_mode':'confirmation'})
    scope_names=('qualification_reuse_scope_6783003_26.json','common_confirmation_independent_source_review.json','common_confirmation_complete_source_freeze.json')
    scope={name:ref(W/'.review_metadata'/name) for name in scope_names}
    put(packet/'qualification_reuse_scope.json',{'kind':'reviewed_unchanged_a4500_numerical_qualification_scope','evidence':scope,'qualified_helper_commit':previous['helpers']['source_commit'],'current_helper_commit':commit,'only_metadata_prerequisite_added':True,'numerical_source_commit':previous['source_commit'],'runtime_environment':plan['runtime_environment'],'asset_record':plan['asset_record'],'new_qualification_claim':False})
    helper_checks={}
    for relative,digest in plan['helpers']['files'].items():
        source=helper_root/relative;staged=unit/'code'/Path(relative).name
        blob=subprocess.check_output(['git','-C',str(helper_root),'show',commit+':'+relative])
        entry=subprocess.check_output(['git','-C',str(helper_root),'ls-tree',commit,'--',relative],text=True).strip().split()
        assert sha(source)==sha(staged)==hashlib.sha256(blob).hexdigest()==digest and entry[0] in {'100644','100755'}
        assert bool(source.stat().st_mode & 0o111)==bool(int(entry[0],8)&0o111)
        helper_checks[relative]={'sha256':digest,'git_blob':entry[2],'git_mode':entry[0],'staged':str(staged)}
    put(packet/'committed_helper_verification.json',{'passed':True,'helper_root':str(helper_root),'helper_commit':commit,'files':helper_checks,'count':len(helper_checks)})
    barrier(packet,unit,plan)
    put(packet/'formal_timing_budget.json',{'source_budget':ref(BUDGET),'copies':copies,**{key:value for key,value in budget.items() if key!='actual_all_thirty_case_cost_records'}})
    old_script=(prior_packet/'job.sbatch').read_text().replace(str(prior_packet),str(packet)).replace(str(prior_unit),str(unit)).replace(sha(prior_unit/'measurement_plan.json'),plan_sha).replace(previous['helpers']['source_commit'],commit)
    old_script=old_script.replace('pixel-common-correctness','pixel-common-confirmation').replace('2026-11-01-15-09_common-correctness',stamp+'_common-confirmation').replace('umask 077\n','umask 077\nunset PYTHONPATH PYTHONHOME\n')
    put(packet/'job.sbatch',old_script.encode(),True)
    held=read(prior_packet/'held_submission_command.json')['command'];held=[str(packet/'job.sbatch') if value==str(prior_packet/'job.sbatch') else '--time='+str(hours).zfill(2)+':00:00' if value.startswith('--time=') else value for value in held]
    put(packet/'held_submission_command.json',{'schema_version':1,'kind':'prepared_unsubmitted_reviewed_common_confirmation_held_command','command':held,'gpu_uuid':None,'job_id':None,'new_jobs_submitted':0})
    put(packet/'command.txt',(shlex.join(held)+'\n').encode(),True)
    put(packet/'config_resolved.yaml',{'reference':plan['reference_configuration'],'candidate':plan['configuration'],'copy_seeds':plan['copy_seeds'],'runtime_environment':plan['runtime_environment'],'common_correctness_inputs':plan['common_correctness_inputs'],'target_qualification_inputs':plan['target_qualification_inputs'],'paired_rounds':7,'complete_updates_per_arm_per_round':10,'requested_hours':hours,'requested_processors':8,'requested_host_memory_gib':64,'requested_typed_gpus':1,'fixed_training_schedule_timesteps_per_copy':2000000000})
    put(packet/'manifest.yaml',{'schema_version':1,'kind':'intermediate_reviewed_common_configuration_confirmation_packet','run_id':name,'mode':'confirmation','source_commit':previous['source_commit'],'helper_commit':commit,'helper_root':str(helper_root),'data_unit':str(unit),'plan':ref(unit/'measurement_plan.json'),'correctness_result':approved['result'],'correctness_review':ref(review),'target_qualification_inputs':plan['target_qualification_inputs'],'historical_walltime_budget':ref(BUDGET),'requested_hours':hours,'requested_cpus':8,'requested_host_memory_gib':64,'requested_gpu_type':'nvidia_rtx_a4500','requested_gpu_count':1,'paired_rounds':7,'updates_per_arm_per_round':10,'pair_orders':[['reference','candidate'] if index%2==0 else ['candidate','reference'] for index in range(7)],'job_id':None,'gpu_uuid':None,'submitted':False,'final_protocol':False,'confirmation_credit':0,'case_credit':0,'idea_credit':0,'final_training_credit':0,'admission_granted':False,'copied_exact_input_sources':copies_from})
    background='## Purpose\n\nConfirm the independently checked common candidate on one A4500 using seven alternating paired rounds against the full original reference. This is intermediate evidence before adaptive research.\n\n## Key hyperparameters\n\n'+str(copies)+' independent policy copies, 128 environments per copy, 128 rollout steps, four epochs, four minibatches, fifty normalization-warmup rollouts, seeds '+str(plan['copy_seeds'])+', and ten complete updates per arm in every paired round. Retain the full two billion transition annealing schedule. Request eight processors, 64 GiB and '+str(hours)+' hours on one typed A4500.\n\n## Code and config changes\n\nOnly cache_room_backgrounds and render_final_two_only differ from the full original numerical reference. A separate standard-library child rechecks the actual correctness result and independent review before numerical initialization. The immutable plan leaves current job and physical GPU identity unresolved; a 180-second absent-only metadata barrier requires the real successful release receipt before numeric srun. No final training, idea, case or packing credit is granted by preparation.\n\n## Git state\n\nNumerical source: '+previous['source_commit']+'. Committed helper source: '+commit+'. The helper checkout is clean and detached. The exact helper, source, configuration, assets and input hashes are recorded in the plan and checksums.\n'
    put(packet/'experiment_background.md',background.encode(),True)
    immutable=[p for p in (unit).rglob('*') if p.is_file() and p.name!='progress.json']+[packet/name for name in ('candidate.json','reference.json','runtime.json','assets.json','profile.json','wait_for_owned_inputs.py','job.sbatch')]
    checksums=''.join(sha(path)+'  '+str(path)+'\n' for path in sorted(immutable))
    put(packet/'source_checksums.sha256',checksums.encode(),True)
    capture(packet,'shell_syntax',['bash','-n',str(packet/'job.sbatch')])
    capture(packet,'startup_barrier_tests',['/usr/bin/python3','-I','-S','-B',str(packet/'test_wait_for_owned_inputs.py')])
    assert not any((packet/name).exists() for name in ('submission_intent.json','submission_receipt.json','release_intent.json','release_receipt.json')) and not (unit/'owned_allocation_inputs.json').exists()
    put(packet/'preparation_verification.json',{'schema_version':1,'kind':'actual_prepared_reviewed_common_confirmation_verification','passed':True,'cpu_job_id':'6783003','cpu_step_id':os.environ['SLURM_STEP_ID'],'helper_commit':commit,'helper_root':str(helper_root),'plan':ref(unit/'measurement_plan.json'),'job_script':ref(packet/'job.sbatch'),'barrier':ref(packet/'wait_for_owned_inputs.py'),'barrier_tests':ref(packet/'barrier_tests_final_receipt.json'),'copied_input_sources':copies_from,'staged_helpers':len(plan['helpers']['files']),'requested_hours':hours,'formal_plan_mode':'confirmation','final_protocol':False,'actual_jobs_submitted':0,'gpu_operations':0,'scientific_credit':0,'admission_granted':False})
    files={str(path):{'sha256':sha(path),'bytes':path.stat().st_size,'mode':stat.S_IMODE(path.stat().st_mode)} for directory in (packet,unit) for path in sorted(directory.rglob('*')) if path.is_file()}
    put(packet/'immutable_artifacts.json',{'kind':'unsubmitted_common_confirmation_static_artifacts','files':files,'job_id':None,'gpu_uuid':None,'submitted':False})
    return {'packet':str(packet),'unit':str(unit),'plan':ref(unit/'measurement_plan.json'),'verification':ref(packet/'preparation_verification.json'),'immutable_artifacts':ref(packet/'immutable_artifacts.json'),'copies':copies,'requested_hours':hours,'helper_commit':commit}

def main():
    """Require actual published clean source and metadata CPU ownership before preparing fresh units."""
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--helper-root',type=Path,required=True);parser.add_argument('--helper-commit',required=True);args=parser.parse_args()
    assert os.environ.get('SLURM_JOB_ID')=='6783003' and os.environ.get('SLURM_STEP_ID','').isdigit() and not NUM.intersection(sys.modules)
    root=args.helper_root.absolute();assert root.resolve(strict=True)==root
    assert subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip()==args.helper_commit
    assert not subprocess.check_output(['git','-C',str(root),'status','--porcelain=v1','--untracked-files=all'])
    assert subprocess.run(['git','-C',str(root),'symbolic-ref','-q','HEAD'],capture_output=True).returncode==1
    subprocess.run(['git','-C',str(root),'merge-base','--is-ancestor','b1c7405c2dce76270554973bdf81c87271216cd4',args.helper_commit],check=True)
    for absolute,digest in read(W/'.review_metadata/common_confirmation_complete_source_freeze.json')['files'].items():assert sha(root/Path(absolute).relative_to(W))==digest
    assert sha(BUDGET)=='316f6e02afba386b104217283c7fe9ad6ea1461f4d8a45980433a240f8d63874'
    stamp=datetime.datetime.now(ZoneInfo('America/Los_Angeles')).strftime('%Y-%m-%d-%H-%M')
    records=[prepare_one(copies,root,args.helper_commit,stamp) for copies in (1,2)]
    assert not subprocess.check_output(['git','-C',str(root),'status','--porcelain=v1','--untracked-files=all'])
    print(json.dumps({'passed':True,'records':records,'actual_jobs_submitted':0,'gpu_operations':0}))

if __name__=='__main__':main()
