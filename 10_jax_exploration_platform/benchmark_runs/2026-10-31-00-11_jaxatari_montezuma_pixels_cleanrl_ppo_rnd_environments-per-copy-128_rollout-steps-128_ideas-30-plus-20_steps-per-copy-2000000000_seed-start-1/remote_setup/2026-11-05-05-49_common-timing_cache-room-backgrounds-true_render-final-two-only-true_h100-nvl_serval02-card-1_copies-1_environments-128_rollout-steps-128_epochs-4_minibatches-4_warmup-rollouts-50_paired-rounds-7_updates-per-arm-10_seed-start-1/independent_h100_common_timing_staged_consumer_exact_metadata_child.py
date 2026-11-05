# Run the exact staged H100 saved-evidence consumer only after independent raw approval.
import argparse,fcntl,hashlib,importlib.util,json,os,time,traceback
from pathlib import Path
R=Path('/p/rlprojects/RND_worktrees/merge-atari-cleanrl/10_jax_exploration_platform/benchmark_runs/2026-10-31-00-11_jaxatari_montezuma_pixels_cleanrl_ppo_rnd_environments-per-copy-128_rollout-steps-128_ideas-30-plus-20_steps-per-copy-2000000000_seed-start-1')
NAME='2026-11-05-05-49_common-timing_cache-room-backgrounds-true_render-final-two-only-true_h100-nvl_serval02-card-1_copies-1_environments-128_rollout-steps-128_epochs-4_minibatches-4_warmup-rollouts-50_paired-rounds-7_updates-per-arm-10_seed-start-1'
P=R/'remote_setup'/NAME;U=R/'data'/NAME
RAW_SOURCE_SHA='a247a3b3e05a320c0fdde513d3f94190918f80beae37b793ea9a35bc73f6519b'
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  while True:
   raw=f.read(1024*1024)
   if not raw:break
   h.update(raw)
 return h.hexdigest()
def ref(path):return {'path':str(path),'sha256':sha(path)}
def write(path,value):
 with Path(path).open('x') as out:json.dump(value,out,indent=2,allow_nan=False);out.write('\n');out.flush();os.fsync(out.fileno())
def main():
 parser=argparse.ArgumentParser();parser.add_argument('--execute-staged-consumer',action='store_true');parser.add_argument('--raw-first-sha256');args=parser.parse_args()
 raw_source=P/'independent_h100_common_timing_raw_review_exact_metadata_child.py';assert sha(raw_source)==RAW_SOURCE_SHA
 if not args.execute_staged_consumer:
  assert args.raw_first_sha256 is None
  print(json.dumps({'actual_result_reads':0,'consumer_run':False,'required_prior_raw_approval':True,'raw_source':ref(raw_source),'consumer_entry':'exact unchanged U/code/direct_measurement.py::verify_saved_measurement(U,result)'}));return
 raw_path=P/'independent_actual_timing_raw_first.json';assert args.raw_first_sha256 is not None and sha(raw_path)==args.raw_first_sha256
 raw=json.loads(raw_path.read_bytes());assert raw['approved'] is True and raw['passed'] is True and raw['independent_of_producer'] is True and raw['raw_window_recomputation_passed'] is True and raw['unchanged_staged_consumer_passed'] is False
 assert raw['result']==ref(U/'result.json') and raw['plan']==ref(U/'measurement_plan.json') and raw['audit_source']==ref(raw_source) and raw['exit_observation']==ref(raw['exit_observation']['path'])
 spec=importlib.util.spec_from_file_location('independent_actual_h100_raw_reader',raw_source);audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)
 plan=audit.source_only();lock_path=R/'queue/cpu_monitor_large_artifact_read.lock'
 with lock_path.open('a+') as lock:
  fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
  direct,common,validation,training,bootstrap=audit.audit_context(plan)
  result=audit.doc(U/'result.json');start=time.time();started=time.monotonic()
  try:
   saved_plan,binding=direct.verify_saved_measurement(U,result)
   assert saved_plan==plan and binding==result['resolved_device_binding'] and raw['result']==ref(U/'result.json') and raw['plan']==ref(U/'measurement_plan.json')
   consumer={'schema_version':1,'kind':'exact_staged_direct_common_timing_saved_consumer_result','passed':True,'result':ref(U/'result.json'),'plan':ref(U/'measurement_plan.json'),'source':ref(U/'code/direct_measurement.py'),'entry_point':'verify_saved_measurement','raw_first_approval':ref(raw_path),'actual_started_at_seconds':start,'finished_at_seconds':time.time(),'seconds':time.monotonic()-started,'source_helper_commit':plan['helpers']['source_commit'],'source_helper_count':len(plan['helpers']['files']),'job_id':os.environ['SLURM_JOB_ID'],'step_id':os.environ['SLURM_STEP_ID'],'stdlib_bootstrap':bootstrap,'numerical_imports':sorted(audit.NUMERICAL.intersection(audit.sys.modules)),'gpu_device_or_library_operations_requested_by_audit':0}
   assert raw['reviewed_at_seconds']<=start and not consumer['numerical_imports']
  except BaseException as error:
   failure={'kind':'exact_staged_direct_common_timing_saved_consumer_result','passed':False,'error':repr(error),'traceback':traceback.format_exc(),'raw_first_approval':ref(raw_path),'actual_started_at_seconds':start,'finished_at_seconds':time.time(),'seconds':time.monotonic()-started,'job_id':os.environ.get('SLURM_JOB_ID'),'step_id':os.environ.get('SLURM_STEP_ID'),'final_review_created':False}
   write(P/'independent_actual_staged_timing_consumer_failure.json',failure);raise
  consumer_path=P/'independent_actual_staged_timing_consumer.json';write(consumer_path,consumer)
  combined={**raw,'kind':'independent_actual_direct_common_configuration_confirmation_raw_audit','unchanged_staged_consumer_passed':True,'raw_first_approval':ref(raw_path),'unchanged_staged_consumer':ref(consumer_path),'completed_at_seconds':time.time(),'finalizer_source':ref(Path(__file__).resolve())}
  combined_path=P/'independent_actual_timing_raw_audit.json';write(combined_path,combined)
  final={key:raw[key] for key in ('schema_version','source_commit','helper_commit','configuration','reference_configuration','copies','copy_seeds','gpu_uuid','result','plan','paired_rounds','updates_per_arm_per_round','decision','median_speedup','one_sided_sign_probability','terminal_process','exit_observation')}
  final.update(kind='independent_actual_direct_common_configuration_confirmation_review',approved=True,passed=True,independent_of_producer=True,reviewer_identity='/root/review_bound_h100_timing_launch',reviewed_at_seconds=time.time(),unresolved_anomalies=[],admission_granted=False,worker_admitted=False,raw_window_recomputation_passed=True,raw_audit=ref(combined_path),raw_first_approval=ref(raw_path),unchanged_staged_consumer=ref(consumer_path),normal_actual_child_returncode=0,actual_controller_and_numerical_child_exits_verified=True,scientific_new_idea_credit=0,packing_point_credit=0,final_training_credit=0,interpretation='Accepted intermediate one-copy common performance on the approved H100. This does not qualify final confirmation, packing, additional ideas, another target, or final training.')
  final_path=P/'independent_actual_common_timing_review.json';write(final_path,final)
  print(json.dumps({'passed':True,'consumer':ref(consumer_path),'combined_raw_audit':ref(combined_path),'final_review':ref(final_path),'median_speedup':raw['median_speedup'],'one_sided_sign_probability':raw['one_sided_sign_probability'],'seconds':consumer['seconds']}))
if __name__=='__main__':main()
