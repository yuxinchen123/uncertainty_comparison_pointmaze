import argparse,datetime,hashlib,json,os,re,socket,subprocess,sys
from pathlib import Path
packet=Path('/p/rlprojects/RND_worktrees/merge-atari-cleanrl/10_jax_exploration_platform/benchmark_runs/2026-10-31-00-11_jaxatari_montezuma_pixels_cleanrl_ppo_rnd_environments-per-copy-128_rollout-steps-128_ideas-30-plus-20_steps-per-copy-2000000000_seed-start-1/remote_setup/2026-10-31-23-24_pixel-study-monitor_source-commit-11ceab5c640b484d18ca40e05785bbf55c6de86f_processor-count-1_host-memory-gib-2_time-limit-days-7_poll-interval-seconds-1200_partition-nolim_account-sl5nw-base_gpu-count-0')
run=Path('/p/rlprojects/RND_worktrees/merge-atari-cleanrl/10_jax_exploration_platform/benchmark_runs/2026-10-31-00-11_jaxatari_montezuma_pixels_cleanrl_ppo_rnd_environments-per-copy-128_rollout-steps-128_ideas-30-plus-20_steps-per-copy-2000000000_seed-start-1')
source=Path('/p/rlprojects/RND_worktrees/merge-atari-cleanrl/10_jax_exploration_platform/benchmark_runs/2026-10-31-00-11_jaxatari_montezuma_pixels_cleanrl_ppo_rnd_environments-per-copy-128_rollout-steps-128_ideas-30-plus-20_steps-per-copy-2000000000_seed-start-1/data/code_snapshots/11ceab5c640b484d18ca40e05785bbf55c6de86f/10_jax_exploration_platform')
job=os.environ.get('SLURM_JOB_ID','');step_name=os.environ.get('SLURM_STEP_ID','');step=job+'.'+step_name
cgroup=Path('/proc/self/cgroup').read_text()
if job!='6779550' or not re.fullmatch('[0-9]+',step_name) or job not in (run/'slurm/submitted_jobids_uva.txt').read_text().splitlines() or '/job_'+job+'/' not in cgroup:raise ValueError('candidate requires the exact existing owned CPU allocation')
parser=argparse.ArgumentParser();parser.add_argument('--aces-packet',type=Path,required=True);args=parser.parse_args()
aces=args.aces_packet.absolute()
if aces.parent!=run/'remote_setup' or aces.is_symlink():raise ValueError('candidate requires the exact local mirrored ACES packet')
sys.path.insert(0,str(source/'scripts/atari_pixel_campaign'))
from monitor import aces_allocation_observation_rows
rows,records=aces_allocation_observation_rows(run)
selected=[value for value in records if value['packet']==str(aces)]
if len(selected)!=1 or selected[0]['row']['state']!='held':raise ValueError('candidate is not ready: an actual fresh owned ACES JobHeldUser snapshot is required')
sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
identifier=selected[0]['row']['slurm_job_id']
admission=run/'SOURCE_AND_GATES_READY.json';admission_before=sha(admission)
before={'at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'campaign':json.loads((run/'campaign_state.json').read_text()),'aces_packet':str(aces),'aces_job_id':identifier,'aces_snapshot_sha256':sha(aces/'remote_monitor_snapshot.json'),'aces_submission_receipt_sha256':sha(aces/'submission_receipt.json'),'source_admission_sha256':admission_before}
(packet/'candidate_before.json').write_text(json.dumps(before,indent=2)+'\n')
started=datetime.datetime.now(datetime.timezone.utc).isoformat()
with (packet/'candidate_stdout.log').open('x') as output,(packet/'candidate_stderr.log').open('x') as errors:
 result=subprocess.run(['/usr/bin/python3','-B',str(source/'scripts/atari_pixel_campaign/monitor.py'),'--run-dir',str(run)],stdout=output,stderr=errors)
if result.returncode:raise RuntimeError('candidate monitor failed; retain its raw output')
report=(run/'20_mins_monitoring/latest_report.md').read_bytes();status_bytes=(run/'20_mins_monitoring/latest_status.json').read_bytes();status=json.loads(status_bytes)
(packet/'candidate_report.md').write_bytes(report);(packet/'candidate_status.json').write_bytes(status_bytes)
actual=[value for value in status['aces_allocation_observations'] if value['packet']==str(aces)]
if len(actual)!=1 or actual[0]['row']['slurm_job_id']!=identifier or actual[0]['row']['state']!='held':raise ValueError('complete candidate report did not retain the exact real held ACES job')
if actual[0]['row']['scientific_hypothesis_credit']!=0 or actual[0]['row']['workers_permitted'] is not False:raise ValueError('an observer cannot grant scientific credit or admission')
if sha(admission)!=admission_before:raise ValueError('candidate changed the original source admission')
if status['campaign']['phase']!='initial_benchmarks':raise ValueError('review the new phase before replacing this initial-study monitor')
if (packet/'candidate_stdout.log').read_text().count('ACES allocation observation')!=1:raise ValueError('candidate printed a duplicate or missing ACES operational row')
def proof(path):return {'path':str(path),'sha256':sha(path)}
record={'schema_version':1,'kind':'owned_candidate_monitor_tick','source_commit':'11ceab5c640b484d18ca40e05785bbf55c6de86f','source_snapshot':proof(packet/'source_snapshot.json'),'job_id':job,'step_id':step,'hostname':socket.gethostname(),'cgroup':cgroup,'started_at_utc':started,'completed_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'exit_code':result.returncode,'stdout':proof(packet/'candidate_stdout.log'),'stderr':proof(packet/'candidate_stderr.log'),'report':proof(packet/'candidate_report.md'),'status':proof(packet/'candidate_status.json'),'actual_aces_packet':str(aces),'actual_aces_job_id':identifier,'actual_aces_snapshot_sha256':before['aces_snapshot_sha256'],'new_job_submitted':False,'observer_scientific_credit':0,'source_admission_unchanged':True,'counts_before':before['campaign']['queue_counts'],'counts_after':status['campaign']['queue_counts'],'count_change_rule':'Any increased scientific count must identify actual newly indexed comparison result paths and bytes; this observer grants zero credit.'}
(packet/'candidate_tick.json').write_text(json.dumps(record,indent=2)+'\n')
verification={'schema_version':1,'kind':'verified_owned_candidate_monitor_tick','candidate':proof(packet/'candidate_tick.json'),'actual_aces_state':'held','actual_aces_job_id':identifier,'one_aces_row':True,'observer_scientific_credit':0,'source_admission_unchanged':True,'new_monitor_submitted':False}
(packet/'candidate_tick_verification.json').write_text(json.dumps(verification,indent=2)+'\n')
print(json.dumps({'candidate_step':step,'exit_code':result.returncode,'packet':str(packet),'aces_job_id':identifier,'counts':status['campaign']['queue_counts']}))
