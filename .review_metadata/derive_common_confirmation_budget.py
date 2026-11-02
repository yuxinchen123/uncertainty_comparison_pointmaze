"""Recheck recorded full-update costs and actual new correctness durations for bounded requests."""
import datetime,hashlib,json,math,os
from pathlib import Path
W=Path(__file__).absolute().parent.parent
R=Path('/p/rlprojects/RND_worktrees/merge-atari-cleanrl/10_jax_exploration_platform/benchmark_runs/2026-10-31-00-11_jaxatari_montezuma_pixels_cleanrl_ppo_rnd_environments-per-copy-128_rollout-steps-128_ideas-30-plus-20_steps-per-copy-2000000000_seed-start-1')
assert os.environ.get('SLURM_JOB_ID')=='6783003' and os.environ.get('SLURM_STEP_ID','').isdigit()
def read(path):
    """Read one actual saved metadata record and retain its exact digest."""
    raw=Path(path).read_bytes();return json.loads(raw),{'path':str(path),'sha256':hashlib.sha256(raw).hexdigest()}
def positive(value):
    """Use only finite positive recorded durations and complete update counts."""
    assert type(value) in (int,float) and math.isfinite(value) and value>0
    return value
old_path=R/'remote_setup/2026-11-01-15-09_common_cache-room-backgrounds-true_render-final-two-only-true_correctness_jaguar03_copies-1_environments-128_rollout-steps-128_epochs-4_minibatches-4_warmup-rollouts-50_seed-start-1/historical_timing_budget.json'
old,old_ref=read(old_path);records={};inputs={};methods={}
for copies in (1,2):
    entries=[]
    for item in old['copies'][str(copies)]['actual_initial_cases']:
        result,ref=read(item['result']['path']);assert ref==item['result']
        assert result['source_commit']=='d28e6735e79788f618acb0ebfde8609db1f776af' and result['copies']==copies and result['hardware_class']=='jaguar03_rtx_a4500' and len(result['rounds'])==7
        maximum={arm:max(positive(pair[arm]['seconds'])/positive(pair[arm]['iterations']) for pair in result['rounds']) for arm in ('reference','candidate')}
        sums={arm:sum(positive(pair[arm]['seconds']) for pair in result['rounds']) for arm in ('reference','candidate')}
        warmups={arm:sum(positive(row['seconds'])/positive(row['updates']) for row in result['warmup'][arm]) for arm in ('reference','candidate')}
        overhead=positive(result['build_to_completion_seconds'])-sum(sums.values());assert overhead>=0
        entries.append({'name':item['name'],'result':ref,'maximum_seconds_per_complete_update':maximum,'normalized_two_warmup_call_seconds':warmups,'actual_non_round_overhead_seconds':overhead,'round_iterations':{arm:[pair[arm]['iterations'] for pair in result['rounds']] for arm in ('reference','candidate')}})
        inputs[ref['path']]=ref['sha256']
    assert len(entries)==30
    actual_path=R/'remote_setup/2026-11-01-17-44-40_common-a4500-actual-correctness_independent-audit_jobs-6782946-6782947_copies-1-2_gpu-count-0'/('final_copies_'+str(copies)+'_raw_audit.json')
    audit,audit_ref=read(actual_path);assert audit['passed'] and audit['terminal_state']=='COMPLETED' and audit['exit_code']=='0:0'
    row=audit['terminal_accounting_rows'][audit['job_id']]
    start=datetime.datetime.fromisoformat(row[6]);finish=datetime.datetime.fromisoformat(row[7]);actual_wall=positive((finish-start).total_seconds())
    measured=positive(audit['execution_identity']['finished_at_seconds']-audit['execution_identity']['started_at_seconds'])
    max_ref=max(v['maximum_seconds_per_complete_update']['reference'] for v in entries)
    max_candidate=max(v['maximum_seconds_per_complete_update']['candidate'] for v in entries)
    historical_overhead=max(v['actual_non_round_overhead_seconds'] for v in entries)
    overhead=max(historical_overhead,actual_wall)
    extra_warm=9*max(sum(v['normalized_two_warmup_call_seconds'].values()) for v in entries)
    timed=70*(max_ref+max_candidate)
    estimate=2*(overhead+extra_warm+timed)+900
    hours=math.ceil(estimate/3600);assert hours=={1:2,2:3}[copies]
    records[str(copies)]={'requested_hours':hours,'requested_seconds':hours*3600,'seven_pairs_ten_complete_updates_per_arm':True,'reference_complete_updates':70,'candidate_complete_updates':70,'maximum_observed_reference_seconds_per_update':max_ref,'maximum_observed_candidate_seconds_per_update':max_candidate,'historical_maximum_non_round_overhead_seconds':historical_overhead,'actual_new_correctness_job_wall_seconds':actual_wall,'actual_new_correctness_execution_seconds':measured,'actual_correctness_raw_audit':audit_ref,'chosen_non_round_allowance_seconds':overhead,'additional_nine_warmup_update_extrapolation_seconds':extra_warm,'measured_update_cost_extrapolation_seconds':timed,'predeclared_margin_factor':2,'startup_allowance_seconds':900,'planned_total_seconds':estimate,'remaining_requested_margin_seconds':hours*3600-estimate,'historical_complete_ten_update_cases':[v for v in entries if all(min(v['round_iterations'][arm])>=10 for arm in ('reference','candidate'))],'actual_all_thirty_case_cost_records':entries,'shape_fit_basis':'The same full-config actual common correctness completed both independent variants and one full update with the selected copy count. Historical full ten-update paired windows also completed. This is not measured peak memory or15percent headroom proof.','measured_new_combination_runtime':False,'timing_credit':0}
value={'schema_version':1,'kind':'actual_evidence_based_intermediate_common_confirmation_walltime_budget','passed':True,'allocation':'6783003','step':os.environ['SLURM_STEP_ID'],'at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'source_commit':'d28e6735e79788f618acb0ebfde8609db1f776af','prior_budget':old_ref,'copies':records,'actual_initial_result_hashes':inputs,'method':'For each copy count re-read all30 actual A4500 records and normalize complete-update times. Use the larger of historical maximum non-round cost and the new actual common-correctness job wall duration. Add nine extra normalized warmup updates per arm and70reference+70candidate complete timed updates. Retain the already declared factor2margin plus900seconds for startup and metadata barriers. Round upward to whole hours.','arbitrary_stalls_are_not_bounded':True,'numerical_experiments':0,'formal_packets_prepared':0,'scientific_credit':0,'admission_granted':False}
raw=(json.dumps(value,indent=2)+'\n').encode();out=W/'.review_metadata'/('common_confirmation_walltime_budget_6783003_'+os.environ['SLURM_STEP_ID']+'.json')
with out.open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
print(json.dumps({'receipt':str(out),'sha256':hashlib.sha256(raw).hexdigest(),'hours':{k:v['requested_hours'] for k,v in records.items()},'planned_seconds':{k:v['planned_total_seconds'] for k,v in records.items()},'actual_new_correctness_job_wall_seconds':{k:v['actual_new_correctness_job_wall_seconds'] for k,v in records.items()}}))
