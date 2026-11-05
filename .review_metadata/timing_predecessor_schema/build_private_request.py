"""Derive a private two-copy metadata request; never import or invoke a preparer/controller."""
import copy,datetime,hashlib,json,os
from pathlib import Path
from zoneinfo import ZoneInfo
W=Path(__file__).resolve().parents[2];D=Path(__file__).resolve().parent;OUT=D/'proposal'
R=Path('/p/rlprojects/RND_worktrees/merge-atari-cleanrl/10_jax_exploration_platform/benchmark_runs/2026-10-31-00-11_jaxatari_montezuma_pixels_cleanrl_ppo_rnd_environments-per-copy-128_rollout-steps-128_ideas-30-plus-20_steps-per-copy-2000000000_seed-start-1')
P1=R/'remote_setup/2026-11-05-05-49_common-timing_cache-room-backgrounds-true_render-final-two-only-true_h100-nvl_serval02-card-1_copies-1_environments-128_rollout-steps-128_epochs-4_minibatches-4_warmup-rollouts-50_paired-rounds-7_updates-per-arm-10_seed-start-1';U1=R/'data'/P1.name
name='2026-11-05-07-20_common-timing_cache-room-backgrounds-true_render-final-two-only-true_h100-nvl_serval02-card-1_copies-2_environments-128_rollout-steps-128_epochs-4_minibatches-4_warmup-rollouts-50_paired-rounds-7_updates-per-arm-10_seed-start-1'
P2=R/'remote_setup'/name;U2=R/'data'/name
if os.path.lexists(P2) or os.path.lexists(U2):raise ValueError('the proposed root paths are not fresh')
def ref(p):return {'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
def write(name,value):
 p=OUT/name
 with p.open('x') as f:json.dump(value,f,indent=2,sort_keys=True);f.write('\n');f.flush();os.fsync(f.fileno())
base=json.loads((P1/'preparation_request.json').read_bytes());req=copy.deepcopy(base)
req.update(unit_id=name,packet=str(P2),data_unit=str(U2),serial_order=2,serial_predecessor_unit=str(U1),copy_seeds=[1,2])
for key in ('configuration','reference_configuration'):req[key]['learner']['n_copies']=2
req['common_correctness_inputs']={key:req['accepted_correctness']['2'][key] for key in ('result','review')}
# The optional old provenance label cannot claim an as-yet-uncommitted corrected source.
req.pop('wrapper_source_commit',None)
write('preparation_request.json',req)
review_path=P1/'independent_actual_common_timing_review.json';review=json.loads(review_path.read_bytes())
if ref(review_path)['sha256']!='901a5805d7f768cd3b3159fe7e478fffdbee36280c2144f195ada1b3471bdf22':raise ValueError('actual accepted timing review changed')
prior={'schema_version':1,'kind':'actual_two_copy_timing_predecessor_available_records','predecessor_unit':str(U1),
 'plan':review['plan'],'result':review['result'],'timing_review':ref(review_path),
 **{key:ref(path) for key,path in {'terminal':U1/'direct_terminal_process.json','wait':U1/'direct_process_wait_return.json',
 'outcome':U1/'direct_controller_outcome.json','metadata':U1/'direct_process_metadata.json','controller_spawn':P1/'root_actual_controller_spawn.json'}.items()},
 'accepted_review_exit_observation':review['exit_observation'],
 'new_required_role_labelled_exit_observation_not_yet_observed':True}
write('predecessor_available_records.json',prior)
S=W/'10_jax_exploration_platform/scripts/atari_pixel_campaign';prefix=['srun','--jobid=6821437','--overlap','--exact','-n1','-c1','--mem=512M','--time=00:05:00','/usr/bin/python3','-I','-S','-B']
canonical=S/'prepare_direct_common_timing.py';probe=Path(req['helper_source_root'])/'10_jax_exploration_platform/scripts/atari_pixel_campaign/probe_direct_current_admission.py'
observer=Path('/p/rlprojects/RND_worktrees/2026-11-05-07-23_prior-h100-exit-observation_controller-pid-386202_birth-84810007_numerical-pid-386669_birth-84815419_read-only-true_gpu-operations-0/observe_exact_prior_h100_exits.py')
procedure={'kind':'planned_root_only_two_copy_common_timing_metadata_procedure','actual_phase_invocations':0,'actual_root_packet_or_unit_created':False,
 'created_at_pacific':datetime.datetime.now(ZoneInfo('America/Los_Angeles')).isoformat(),'request_after_root_promotion':str(P2/'preparation_request.json'),
 'canonical_preparer':ref(canonical),'controller':ref(S/'direct_common_timing_controller.py'),'controller_baseline_commit':'d2253fa853290cb8fcd0a9212e332dd33a63f2f9',
 'corrected_controller_source_freeze':ref(D/'author_source_freeze_1.json'),'numerical_source_unchanged':req['numerical_source_commit'],'helper_source_unchanged':req['helper_source_commit'],
 'phase_commands':{'prepare':[*prefix,str(canonical),'prepare','--request',str(P2/'preparation_request.json')],
 'verify':[*prefix,str(canonical),'verify','--packet',str(P2)],'bind':[*prefix,str(canonical),'bind','--packet',str(P2)]},
 'claim_probe':{'command_prefix':[*prefix,str(probe),'--prepared-unit',str(U2),'--expected-plan-sha256'],
 'required_next_argument_from_actual_file':str(U2/'measurement_prepared.json')+' -> plan_sha256','command_suffix':['--check-canonical-source'],
 'source':ref(probe),'complete_command_not_formed_before_actual_prepare':True},
 'fresh_exit_observer_source':ref(observer),'fresh_exit_observer_must_be_independently_approved_and_published':True,
 'fresh_exit_observer_root_invocation_not_performed':True,'new_required_actual_exit_output':str(P2/'root_exact_serial_predecessor_exit_observation.json'),
 'serial_predecessor_inputs_root_publication':{'target':str(P2/'serial_predecessor_exit_inputs.json'),
 'kind':'actual_owned_direct_serial_predecessor_exit_inputs','existing_inputs_source':ref(OUT/'predecessor_available_records.json'),
 'new_exit_ref_must_use_real_successful_observer_output_hash_only':True,'timing_review_required':prior['timing_review']},
 'ordered_root_steps':['Independently review and publish corrected controller/request/procedure and fresh observer source; preserve d225 and all prior outcomes.',
 'Promote this exact request to a fresh actual packet, leaving the future data unit absent. Do not fill optional source provenance with a future commit.',
 'Invoke the separate approved read-only observer exactly once on serval02 as root actor; save its actual return/stdout and real role-labelled observer record. No GPU/lock operations.',
 'Invoke canonical prepare once on owned CPU with five-minute step/240-second metadata bound; record successful actual plan and helper hashes. Preserve failure and never reuse an attempted unit.',
 'Invoke canonical verify once on owned CPU, preserving the actual cold-child proof; no historical raw replay or new scientific credit.',
 'Derive the exact claim-probe hash argument only from actual measurement_prepared.json, then invoke that distinct bounded metadata phase once.',
 'Assemble immutable serial_predecessor_exit_inputs.json from the actual available-record refs plus the new real observer output; include the exact accepted timing_review ref. Preserve old accepted exit as separate corroboration, never relabel it.',
 'Invoke canonical bind once. Independently review actual bound unit, all29 staged source helpers, current claim proof, serial waits/exits and packet/source freeze.',
 'Root publishes inner actual bound records. Only afterward derive the exact actual inner-publication launch command and root starter from the previously reviewed normal transport mechanism; do not copy an old hard-coded publication literal.',
 'Independently review and publish root transport/outer records before one SSH. Root alone invokes it; actual PID/birth/registration are recorded from actual processes, never planned values.',
 'Verify current monitor reporting, then collect all seven alternating ten-update pairs and exact normal wait/exits; independent raw-first review and unchanged staged consumer are required before accepting this two-copy timing.'],
 'only_device':'serval02 physical card1 GPU-fed69954-5a71-dbb2-ee91-544e0eb6b8a3 under /tmp/sl5nw_serval02_card1.lock',
 'controller_owns_gpu_standing_descriptor':False,'producer_owns_gpu_standing_descriptor':True,
 'hypothesis_count_contribution':0,'final_confirmation_credit':0,'packing_credit':0,'scientific_credit_at_preparation':0,
 'other_controller_functions_unchanged_claim_requires_exact_roster':True,'entire_controller_source_unchanged_claim':False}
write('planned_root_metadata_procedure.json',procedure)
write('derivation.json',{'kind':'private_two_copy_common_timing_request_derivation','base_request':ref(P1/'preparation_request.json'),
 'accepted_one_copy_review':ref(review_path),'actual_h1_result_referenced_without_reread':review['result'],'accepted_c2_correctness':req['accepted_correctness']['2'],
 'source_claim_note':'d225 is preserved baseline; only serial-predecessor timing schema predicate is corrected and separately frozen. No future publication commit is guessed.',
 'only_request_semantic_changes':['n_copies1->2 in both arms','copy_seeds[1]->[1,2]','serial_order1->2','serial predecessor genuine H1 timing','common correctness genuine C2','fresh Pacific unit/packet paths'],
 'optional_old_wrapper_source_commit_label_removed':True,'current_cuda_backend_runtime_and_numerical_contracts_unchanged':True,'actual_prepare_verify_claim_bind_or_launch_invocations':0})
print(json.dumps({'private_proposal':str(OUT),'future_packet':str(P2),'actual_unit_created':False},sort_keys=True))
