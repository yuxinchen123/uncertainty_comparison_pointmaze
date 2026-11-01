"""Validate this monitor's exact owned CPU request and actual SMT allocation before its full loop."""
import getpass,json,os,socket,subprocess
from datetime import datetime,timezone
from pathlib import Path
EXPECTED={'owner': 'sl5nw', 'uid': 2618919, 'run': '/p/rlprojects/RND_worktrees/merge-atari-cleanrl/10_jax_exploration_platform/benchmark_runs/2026-10-31-00-11_jaxatari_montezuma_pixels_cleanrl_ppo_rnd_environments-per-copy-128_rollout-steps-128_ideas-30-plus-20_steps-per-copy-2000000000_seed-start-1', 'source_commit': '31fd9fe7cd96343741e8b48a18019efee9b4ac5c', 'script': '/p/rlprojects/RND_worktrees/merge-atari-cleanrl/10_jax_exploration_platform/benchmark_runs/2026-10-31-00-11_jaxatari_montezuma_pixels_cleanrl_ppo_rnd_environments-per-copy-128_rollout-steps-128_ideas-30-plus-20_steps-per-copy-2000000000_seed-start-1/remote_setup/2026-11-01-14-02_pixel-study-monitor_source-commit-31fd9fe7cd96343741e8b48a18019efee9b4ac5c_processor-count-1_host-memory-gib-2_time-limit-days-7_poll-interval-seconds-1200_partition-nolim_account-sl5nw-base_gpu-count-0/monitor.sbatch'}

def tres(value):
    """Read the explicit requested or allocated resource values without dropping duplicate keys."""
    pairs=[item.split('=',1) for item in value.split(',') if '=' in item]
    if len({key for key,value in pairs})!=len(pairs):raise ValueError('monitor resource record repeats a TRES key')
    return dict(pairs)

def validate_resources(fields,node):
    """Bind the one-CPU request to one or two consistent allocated logical CPUs on the actual node."""
    requested=tres(fields.get('ReqTRES',''));allocated=tres(fields.get('AllocTRES',''))
    if fields.get('CPUs/Task')!='1' or fields.get('NumNodes')!='1' or fields.get('MinMemoryNode')!='2G':
        raise ValueError('monitor must request one CPU per task, one node and two GiB')
    if requested.get('cpu')!='1' or requested.get('node')!='1' or requested.get('mem')!='2G':
        raise ValueError('monitor requested TRES differs from one CPU, one node and two GiB')
    count=fields.get('NumCPUs')
    if count not in {'1','2'} or allocated.get('cpu')!=count or allocated.get('node')!='1' or allocated.get('mem')!='2G':
        raise ValueError('monitor allocated CPU/node/memory records are inconsistent or exceed the reviewed shape')
    if any(key.startswith('gres/gpu') and value!='0' for record in (requested,allocated) for key,value in record.items()):
        raise ValueError('monitor must request and hold zero GPUs')
    if node.get('NodeName')!=fields.get('NodeList'):
        raise ValueError('monitor topology record belongs to another node')
    if count=='2' and node.get('ThreadsPerCore')!='2':
        raise ValueError('two allocated logical CPUs require actual two-thread-per-core topology')

def validate(fields,job,host,cgroup,node):
    """Require the exact owner, CPU-only job shape and actual allocated host/cgroup."""
    wanted={'JobId':job,'UserId':'sl5nw(2618919)','Account':'sl5nw_base','Partition':'nolim','QOS':'normal','JobState':'RUNNING',
            'TimeLimit':'7-00:00:00','Requeue':'0'}
    if any(fields.get(key)!=value for key,value in wanted.items()):raise ValueError('monitor CPU allocation differs from the reviewed owned job')
    if fields.get('NodeList')!=host.split('.')[0] or '/job_'+job+'/' not in cgroup or any(word in host.split('.')[0] for word in ('portal','login')):
        raise ValueError('monitor process is outside its exact allocated compute host/job cgroup')
    validate_resources(fields,node)

def main():
    """Check exact own ledger rows before querying only this job and its actual node topology."""
    run=Path(EXPECTED['run']);job=os.environ.get('SLURM_JOB_ID','')
    if not job.isdecimal() or int(job)<1 or os.getuid()!=EXPECTED['uid'] or getpass.getuser()!=EXPECTED['owner']:
        raise ValueError('monitor requires the recorded owner and a real Slurm job')
    ledger=run/'slurm/submitted_jobids_uva.txt';submissions=run/'slurm/submissions.tsv'
    if ledger.is_symlink() or submissions.is_symlink() or ledger.read_text().splitlines().count(job)!=1:
        raise ValueError('monitor job is absent or duplicated in its exact UVA ledger')
    rows=[line.split('\t') for line in submissions.read_text().splitlines() if line.split('\t')[0]==job]
    if rows!=[[job,'monitor',EXPECTED['source_commit'],EXPECTED['script']]]:raise ValueError('monitor submission row differs from the exact source/script')
    command=['scontrol','show','job',job,'-o'];raw=subprocess.run(command,capture_output=True,text=True,check=True,timeout=20,env={**os.environ,'TZ':'UTC'})
    if len(raw.stdout.splitlines())!=1:raise ValueError('monitor job query did not return exactly one owned row')
    fields=dict(token.split('=',1) for token in raw.stdout.split() if '=' in token);host=socket.gethostname();cgroup=Path('/proc/self/cgroup').read_text()
    if fields.get('NodeList')!=host.split('.')[0]:raise ValueError('monitor node query must name this exact allocated host')
    node_command=['scontrol','show','node',host.split('.')[0],'-o'];node_raw=subprocess.run(node_command,capture_output=True,text=True,check=True,timeout=20)
    if len(node_raw.stdout.splitlines())!=1:raise ValueError('monitor node query did not return exactly one topology row')
    node=dict(token.split('=',1) for token in node_raw.stdout.split() if '=' in token);validate(fields,job,host,cgroup,node)
    record={'kind':'owned_cpu_monitor_allocation','job_id':job,'hostname':host,'uid':os.getuid(),'pid':os.getpid(),'cgroup':cgroup,
            'command':command,'returncode':raw.returncode,'stdout':raw.stdout,'stderr':raw.stderr,
            'node_observation':{'command':node_command,'returncode':node_raw.returncode,'stdout':node_raw.stdout,'stderr':node_raw.stderr},
            'observed_at_utc':datetime.now(timezone.utc).isoformat(),'scientific_hypothesis_credit':0,'worker_admission':False}
    path=Path(__file__).parent/'runtime_allocation.json'
    with path.open('x') as stream:json.dump(record,stream,indent=2);stream.write('\n');stream.flush();os.fsync(stream.fileno())
if __name__=='__main__':main()
