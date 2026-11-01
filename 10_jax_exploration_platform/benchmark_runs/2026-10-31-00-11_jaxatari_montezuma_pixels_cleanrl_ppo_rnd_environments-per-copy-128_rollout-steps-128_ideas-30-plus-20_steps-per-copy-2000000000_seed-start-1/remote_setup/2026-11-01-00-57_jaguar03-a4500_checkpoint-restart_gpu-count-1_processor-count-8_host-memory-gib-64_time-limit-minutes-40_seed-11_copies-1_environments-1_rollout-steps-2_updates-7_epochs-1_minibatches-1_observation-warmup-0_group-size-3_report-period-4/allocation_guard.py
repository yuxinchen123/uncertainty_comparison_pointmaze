"""Verify only this component job's owned allocation before numerical imports."""
import getpass,json,os,re,socket,subprocess
from datetime import datetime,timezone
from pathlib import Path
EXPECTED = {'owner': 'sl5nw', 'uid': 2618919, 'account': 'sl5nw_base', 'partition': 'gpu', 'qos': 'csresnolim', 'reservation': 'sl5nw_168', 'node': 'jaguar03', 'cpus': 8, 'memory': '64G', 'time_limit': '00:40:00', 'gres': 'gpu:nvidia_rtx_a4500:1', 'campaign_run_dir': '/p/rlprojects/RND_worktrees/merge-atari-cleanrl/10_jax_exploration_platform/benchmark_runs/2026-10-31-00-11_jaxatari_montezuma_pixels_cleanrl_ppo_rnd_environments-per-copy-128_rollout-steps-128_ideas-30-plus-20_steps-per-copy-2000000000_seed-start-1', 'unit_id': '2026-11-01-00-57_jaguar03-a4500_checkpoint-restart_gpu-count-1_processor-count-8_host-memory-gib-64_time-limit-minutes-40_seed-11_copies-1_environments-1_rollout-steps-2_updates-7_epochs-1_minibatches-1_observation-warmup-0_group-size-3_report-period-4', 'source_commit': '8a96999408d4e214b212074d932f0eeabcbb5eea', 'submission_kind': 'checkpoint-restart-component-validation', 'job_script': '/p/rlprojects/RND_worktrees/merge-atari-cleanrl/10_jax_exploration_platform/benchmark_runs/2026-10-31-00-11_jaxatari_montezuma_pixels_cleanrl_ppo_rnd_environments-per-copy-128_rollout-steps-128_ideas-30-plus-20_steps-per-copy-2000000000_seed-start-1/remote_setup/2026-11-01-00-57_jaguar03-a4500_checkpoint-restart_gpu-count-1_processor-count-8_host-memory-gib-64_time-limit-minutes-40_seed-11_copies-1_environments-1_rollout-steps-2_updates-7_epochs-1_minibatches-1_observation-warmup-0_group-size-3_report-period-4/job.sbatch'}

def validate(fields, job, hostname, cgroup):
    """Require the exact owned account, resource shape, host and current job cgroup."""
    required = {'JobId':job,'UserId':EXPECTED['owner']+'('+str(EXPECTED['uid'])+')','Account':EXPECTED['account'],
        'Partition':EXPECTED['partition'],'QOS':EXPECTED['qos'],'Reservation':EXPECTED['reservation'],'JobState':'RUNNING',
        'NodeList':EXPECTED['node'],'NumNodes':'1','NumCPUs':str(EXPECTED['cpus']),'TimeLimit':EXPECTED['time_limit'],'MinMemoryNode':EXPECTED['memory']}
    if any(fields.get(key)!=value for key,value in required.items()):
        raise ValueError('component allocation differs from its exact owned requested shape')
    if hostname.split('.')[0]!=EXPECTED['node'] or '/job_'+job+'/' not in cgroup:
        raise ValueError('component process is outside its exact allocated compute host/job cgroup')
    if fields.get('TresPerNode') not in {'gres:gpu:nvidia_rtx_a4500:1','gres/gpu:nvidia_rtx_a4500:1'}:
        raise ValueError('component allocation does not request exactly one typed A4500')
    allocated=dict(item.split('=',1) for item in fields.get('AllocTRES','').split(',') if '=' in item)
    if allocated.get('gres/gpu')!='1' or allocated.get('cpu')!='8' or allocated.get('mem')!='64G':
        raise ValueError('component allocated resources differ from the one-card/eight-processor/64GiB shape')

def main():
    """Check literal campaign ledger ownership before querying only this process's exact job."""
    job=os.environ.get('SLURM_JOB_ID','');run=Path(EXPECTED['campaign_run_dir'])
    if not job.isdecimal() or int(job)<1 or os.getuid()!=EXPECTED['uid'] or getpass.getuser()!=EXPECTED['owner']:
        raise ValueError('component requires the exact owned Slurm compute user and job')
    ledger=run/'slurm/submitted_jobids_uva.txt';rows=run/'slurm/submissions.tsv'
    if ledger.is_symlink() or rows.is_symlink() or ledger.read_text().splitlines().count(job)!=1:
        raise ValueError('component job is absent or duplicated in the exact UVA ledger')
    matching=[line.split('\t') for line in rows.read_text().splitlines() if line.split('\t')[0]==job]
    if matching!=[[job,EXPECTED['submission_kind'],EXPECTED['source_commit'],EXPECTED['job_script']]]:
        raise ValueError('component job lacks its exact owned source/script submission row')
    command=['scontrol','show','job',job,'-o']
    raw=subprocess.run(command,capture_output=True,text=True,check=True,timeout=20,env={**os.environ,'TZ':'UTC'})
    if len(raw.stdout.splitlines())!=1:
        raise ValueError('component allocation query must return exactly one current job row')
    fields=dict(token.split('=',1) for token in raw.stdout.split() if '=' in token)
    cgroup=Path('/proc/self/cgroup').read_text();validate(fields,job,socket.gethostname(),cgroup)
    directory=run/'data'/EXPECTED['unit_id'];directory.mkdir(parents=True,exist_ok=True)
    record={'kind':'owned_component_allocation','expected':EXPECTED,'job_id':job,'pid':os.getpid(),'uid':os.getuid(),
        'hostname':socket.gethostname(),'cgroup':cgroup,'command':command,'returncode':raw.returncode,'stdout':raw.stdout,
        'stderr':raw.stderr,'observed_at_utc':datetime.now(timezone.utc).isoformat(),'numerical_imports':False}
    path=directory/'allocation_guard.json'
    with path.open('x') as stream:
        json.dump(record,stream,indent=2);stream.write('\n');stream.flush();os.fsync(stream.fileno())
if __name__=='__main__':main()
