WORKTREE='/p/rlprojects/RND_worktrees/2026-11-01-17-55_a4500-common-confirmation_prior-correctness-required_rounds-7_complete-updates-per-arm-10_copies-1-2'
"""Finalize independent source review from actual completed metadata checks."""
import datetime,hashlib,json,os,pathlib
w=pathlib.Path(WORKTREE)
assert os.environ['SLURM_JOB_ID']=='6783003' and os.environ['SLURM_STEP_ID'].isdigit()
def ref(p):return {'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
def read(p):return json.loads(p.read_bytes())
freeze=w/'.review_metadata/common_confirmation_complete_source_freeze.json'
assert ref(freeze)['sha256']=='27281757454d61f4717e8d335a30f770829907c2f1d6a6a4d698c723c709cb6c'
actual={p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest() for p in read(freeze)['files']}
assert actual==read(freeze)['files']
tests=w/'.review_metadata/common_confirmation_affected_independent_6783003_32.json';t=read(tests)
assert ref(tests)['sha256']=='f34534dc7d6ab2fd50aba24c3f7b2a7102fd83236312bc7663f836d929f786aa'
assert t['passed'] is True and t['tests']==180 and not t['blocked_actions'] and not t['failures'] and not t['errors'] and t['source_before']==t['source_after']==actual
scope=w/'.review_metadata/independent_qualification_reuse_scope_6783003_34.json';z=read(scope)
assert z['passed'] is True and ref(scope)['sha256']=='f2823d14917f15093db66086d8d9fab2129aa3cdc7af7edea5fef56ce1fe1b4c'
cold_return=w/'.review_metadata/common_confirmation_independent_cold_return.json';cr=read(cold_return)
assert ref(cold_return)['sha256']=='f620403b65c4fcf06bdd3ac637fc6609f13f7395146234d3c72aeeb70e958b34' and cr['actual_srun_return_code']==0 and cr['passed'] is True
coldpath=pathlib.Path(cr['stdout']['path']);assert ref(coldpath)==cr['stdout'];c=read(coldpath)
assert c['passed'] is True and c['review_job_id']=='6783003' and c['review_step']=='36' and c['copies']==1
assert c['actual_prior_correctness_consumer_passed'] is True and c['parent_import_paths_unchanged_by_metadata_child'] is True and c['parent_ideas_and_controller_absent_after_metadata_child'] is True
assert c['staged_helper_count']==21 and not c['real_numerical_import_attempts'] and c['actual_gpu_operations']==c['actual_job_actions']==0
assert pathlib.Path(cr['stderr']['path']).read_bytes()==b''
author=w/'.review_metadata/common_confirmation_affected_author_6783003_30.json';a=read(author)
assert a['passed'] is True and a['tests']==172 and a['source_before']==a['source_after']==actual
value={'schema_version':1,'kind':'independent_common_confirmation_source_review','approved':True,'passed':True,
 'at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'review_job_id':'6783003','review_step':os.environ['SLURM_STEP_ID'],
 'reviewer':'common_confirmation_review','base_commit':'6066d9ea35fbc9c045855144550971c2ac5895f9','source_freeze':ref(freeze),
 'frozen_source_paths':actual,'author_affected_tests':ref(author),'author_test_count':172,'independent_affected_tests':ref(tests),'independent_test_count':180,'extra_independent_controls':8,
 'independent_qualification_reuse_scope':ref(scope),'independent_cold_actual_return':ref(cold_return),'independent_cold_actual_stdout':ref(coldpath),
 'source_findings':[],
 'reviewed_contract':{'prior_actual_correctness_and_independent_raw_review_required':True,'source_configuration_runtime_assets_seeds_physical_binding_terminal_joins':True,
 'isolated_metadata_child_before_parent_NUM_and_card_admission':True,'parent_sys_path_ideas_and_controller_preserved':True,
 'durable_child_return_stdout_stderr_proof_and_saved_result_binding':True,'saved_reader_revalidates_exact_current_prior_raw_evidence':True,
 'rounds':7,'full_updates_per_arm_per_round':10,'paired_order':'alternating reference/candidate then candidate/reference',
 'copies':[1,2],'copy_seeds':{'1':[1],'2':[1,2]},'environments_per_copy':128,'rollout_steps':128,'epochs':4,'minibatches':4,'normalization_warmup_rollouts':50,'annealing_horizon_requested_transitions':2000000000,
 'candidate_changes':['environment.cache_room_backgrounds=false to true','environment.render_final_two_only=false to true'],
 'only_opt_in_A4500_common_confirmation_adds_21st_helper':True,'default_and_direct_closures_unchanged':True,'final_completed_fifty_and_eleven_by_twenty_paths_unchanged':True},
 'qualification_reuse':{'approved_with_exact_scope':True,'qualified_original_helper_commit':'e5afbf6b51a3d1666b46e71c0631a45d47080cd8',
 'qualified_original_result_sha256':'0c4f94c2d58fdd7277a261481d7a52c4bb5b19e42612a8eae67dff6890c62cd1',
 'new_compute_AST_identical_to_6066':True,'UVA_specialized_compute_AST_identical_to_E5':True,'whole_E5_compute_AST_identical':False,'all13_NUM_and_Executor_files_unchanged':True,
 'unchanged_UVA_core_train_optimized_and_short_learning':True,'restriction':'Only this exact reviewed metadata change, the original D28 numerical source, unchanged runtime/assets and copies one/two common configuration. A changed numerical, target, runtime or source scope requires a separate review and qualification.'},
 'monitor_scope':{'current_monitor_source':'79a01e851ff92ddba7d0e12eaf8164cab2028878','current_monitor_job':'6783003',
 'progress_and_full_status_remain_usable':True,'new_common_scientific_acceptance_from_old_monitor_row':False,
 'scientific_acceptance_requires':'Exact newly staged reviewed consumer run independently on owned compute, or a separately reviewed upgraded frozen monitor.'},
 'next_authorized_steps':['Commit and publish exact reviewed source and records.','Prepare fresh formal copies-one and copies-two intermediate confirmation packets with actual correctness result and review references.','Independently review exact packets and actual source/monitor/ownership gates before any one-time submission.'],
 'formal_packet_approval':False,'actual_GPU_qualification_of_new_producer':False,'numerical_work':False,'actual_gpu_or_job_actions':0,'scientific_credit':0,'case_credit':0,'idea_credit':0,'admission_granted':False,'worker_admitted':False}
out=w/'.review_metadata/common_confirmation_independent_source_review.json'
with out.open('x') as stream:json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n');stream.flush();os.fsync(stream.fileno())
print(json.dumps({'passed':True,'approved':True,'receipt':str(out),'sha256':ref(out)['sha256']}))
