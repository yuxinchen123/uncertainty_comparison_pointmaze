import datetime,hashlib,io,json,os,re,socket,stat,subprocess,sys,tarfile,traceback
from pathlib import Path
from zoneinfo import ZoneInfo
OUT=Path(__file__).absolute().parent
P=Path('/p/rlprojects/RND_worktrees/merge-atari-cleanrl/10_jax_exploration_platform/benchmark_runs/2026-10-31-00-11_jaxatari_montezuma_pixels_cleanrl_ppo_rnd_environments-per-copy-128_rollout-steps-128_ideas-30-plus-20_steps-per-copy-2000000000_seed-start-1/remote_setup/2026-11-01-16-33_pixel-study-monitor-recovery_source-commit-79a01e851ff92ddba7d0e12eaf8164cab2028878_processor-count-1_host-memory-gib-2_time-limit-days-7_poll-interval-seconds-1200_partition-nolim_account-sl5nw-base_gpu-count-0')
R=P.parent.parent
REPO=R.parents[2]
JOB='6783003';SOURCE='79a01e851ff92ddba7d0e12eaf8164cab2028878'
NUM={'jax','jaxlib','numpy','torch','flax','optax','cupy','scipy'}
def guard(event,args):
 if event=='import' and args[0].split('.')[0] in NUM:raise RuntimeError('numerical import forbidden')
 if event in {'socket.connect','socket.bind','ctypes.dlopen'}:raise RuntimeError('network or library operation forbidden')
 if event=='open' and isinstance(args[0],(str,bytes)) and os.fsdecode(args[0]).startswith('/dev/nvidia'):raise RuntimeError('device access forbidden')
 if event=='subprocess.Popen':
  cmd=args[1]
  if not isinstance(cmd,list) or cmd[:3]!=['git','-C',str(REPO)] or cmd[3]!='ls-tree':raise RuntimeError('only exact local source tree query permitted')
sys.addaudithook(guard)
def sha(raw):return hashlib.sha256(raw).hexdigest()
def put(name,raw):
 p=OUT/name
 with p.open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
 return {'path':str(p),'sha256':sha(raw),'bytes':len(raw)}
def field(raw,name):
 v=re.findall(r'(?:^|\s)'+re.escape(name)+r'=([^\s]*)',raw);assert len(v)==1,(name,v);return v[0]
def identity(pid):
 root=Path('/proc')/str(pid);s=(root/'status').read_text();u=re.search(r'^Uid:\s+(.+)$',s,re.M);assert u and [int(x) for x in u.group(1).split()]==[2618919]*4
 st=(root/'stat').read_text();cg=(root/'cgroup').read_text();assert '/job_6783003/step_batch/' in cg
 return {'pid':pid,'birth':int(st.rsplit(')',1)[1].split()[19]),'uids':[int(x) for x in u.group(1).split()],'cgroup':cg,'cmdline':(root/'cmdline').read_bytes().split(b'\0')[:-1],'fd1':os.readlink(root/'fd/1'),'fd2':os.readlink(root/'fd/2'),'stat':st}
def main():
 assert os.environ.get('SLURM_JOB_ID')==JOB and os.environ.get('SLURM_STEP_ID','').isdigit() and os.getuid()==2618919
 own_cgroup=Path('/proc/self/cgroup').read_text();assert '/job_6783003/step_'+os.environ['SLURM_STEP_ID']+'/' in own_cgroup
 assert socket.gethostname().split('.')[0]=='heartpiece'
 allocation_raw=(P/'runtime_allocation.json').read_bytes();a=json.loads(allocation_raw);assert a['job_id']==JOB and a['returncode']==0 and a['stderr']=='' and a['uid']==2618919 and a['hostname']=='heartpiece'
 raw=a['stdout'];assert field(raw,'JobId')==JOB and field(raw,'JobState')=='RUNNING' and field(raw,'UserId')=='sl5nw(2618919)' and field(raw,'NodeList')==field(raw,'BatchHost')=='heartpiece'
 for key,value in {'Account':'sl5nw_base','Partition':'nolim','QOS':'normal','Requeue':'0','Restarts':'0','NumTasks':'1','CPUs/Task':'1','NumNodes':'1','MinMemoryNode':'2G','TresPerNode':'gres/gpu:0','TimeLimit':'7-00:00:00','Command':str(P/'monitor.sbatch')}.items():assert field(raw,key)==value,(key,field(raw,key))
 req=dict(x.split('=',1) for x in field(raw,'ReqTRES').split(','));alloc=dict(x.split('=',1) for x in field(raw,'AllocTRES').split(','));assert req['cpu']=='1' and alloc['cpu'] in {'1','2'} and field(raw,'NumCPUs')==alloc['cpu']
 for resources in (req,alloc):
  assert resources['mem']=='2G' and resources['node']=='1'
  assert all(v=='0' for k,v in resources.items() if k.startswith('gres/gpu'))
 n=a['node_observation'];assert n['command']==['scontrol','show','node','heartpiece','-o'] and n['returncode']==0 and n['stderr']==''
 assert field(n['stdout'],'NodeName')==field(n['stdout'],'NodeHostName')=='heartpiece'
 if alloc['cpu']=='2':assert field(n['stdout'],'ThreadsPerCore')=='2'
 manifest_raw=(P/'source_snapshot.json').read_bytes();manifest=json.loads(manifest_raw);assert manifest['source_commit']==SOURCE and len(manifest['files'])==265
 source=Path(manifest['source_root']);files=manifest['files'];assert source.resolve(strict=True)==source
 tree_cmd=['git','-C',str(REPO),'ls-tree','-r','-z',SOURCE,'--']+manifest['source_prefixes'];tree_raw=subprocess.check_output(tree_cmd);tree={}
 for row in tree_raw.split(b'\0'):
  if not row:continue
  head,name=row.split(b'\t',1);mode,kind,blob=head.decode().split();assert kind=='blob';tree[name.decode()]=(mode,blob)
 assert set(tree)==set(files)
 assert {q.relative_to(source).as_posix() for q in source.rglob('*') if q.is_file()}==set(files)
 assert not any(q.is_symlink() for q in source.rglob('*'))
 checks={}
 for name,record in files.items():
  q=source/name;b=q.read_bytes();blob=hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest();mode=stat.S_IMODE(q.stat().st_mode)
  assert sha(b)==record['sha256'] and blob==record['git_blob'] and tree[name]==(record['git_mode'],blob) and len(b)==record['bytes']
  assert (mode&0o111)==(0o111 if record['git_mode']=='100755' else 0)
  checks[name]={'sha256':sha(b),'git_blob':blob,'git_mode':record['git_mode'],'physical_mode':mode,'bytes':len(b)}
 archive_path=P/'source_git_archive.tar';archive_raw=archive_path.read_bytes();assert sha(archive_raw)==manifest['archive']['sha256'];archive_checks={}
 with tarfile.open(fileobj=io.BytesIO(archive_raw),mode='r:') as tf:
  for item in tf.getmembers():
   if item.isdir():continue
   assert item.isfile() and item.name.startswith(SOURCE+'/')
   name=item.name[len(SOURCE)+1:];assert name in files and name not in archive_checks
   b=tf.extractfile(item).read();rec=files[name];assert sha(b)==rec['sha256'] and (item.mode&0o111)==(0o111 if rec['git_mode']=='100755' else 0);archive_checks[name]=sha(b)
 assert set(archive_checks)==set(files)
 expected=[b'/usr/bin/python3',b'-B',str(source/'10_jax_exploration_platform/scripts/atari_pixel_campaign/monitor.py').encode(),b'--run-dir',str(R).encode(),b'--loop',b'--interval',b'1200']
 batch='/system.slice/slurmstepd.scope/job_6783003/step_batch/user/task_0'
 members_raw=(Path('/sys/fs/cgroup')/batch.lstrip('/')/'cgroup.procs').read_bytes();members=[int(x) for x in members_raw.split()];matches=[]
 for pid in members:
  v=identity(pid)
  if v['cmdline']==expected:matches.append(v)
 assert len(matches)==1,[(v['pid'],v['cmdline']) for v in matches]
 monitor=matches[0];assert monitor['fd1']==monitor['fd2']==str(R/'slurm/jobs/monitor_6783003.log')
 report_raw=(R/'20_mins_monitoring/latest_report.md').read_bytes();status_raw=(R/'20_mins_monitoring/latest_status.json').read_bytes();log_raw=(R/'slurm/jobs/monitor_6783003.log').read_bytes();s=json.loads(status_raw);report=report_raw.decode();log=log_raw.decode();header,body=log.split('\n',1);initial=json.loads(header);assert initial=={'source_commit':SOURCE,'files_verified':265,'full_tick_executed':False,'numerical_imports':False}
 table=report.split('```text\n',1)[1].rsplit('```',1)[0];assert body==table+'\n',(len(body),len(table))
 expected_sections=['TRAINING','WAITING','HELD','FINISHED','PLANNED'];sections=[line for line in table.splitlines() if line in expected_sections];assert sections==expected_sections
 time=datetime.datetime.strptime(s['time'],'%Y-%m-%d %H:%M:%S PT').replace(tzinfo=ZoneInfo('America/Los_Angeles')).astimezone(datetime.timezone.utc)
 released=json.loads((P/'release_receipt.json').read_bytes());released_time=datetime.datetime.fromisoformat(released['at_utc']);assert time>=datetime.datetime.fromisoformat(a['observed_at_utc'])>released_time and datetime.datetime.now(datetime.timezone.utc)-time<datetime.timedelta(minutes=10)
 assert s['time'] in report and s['campaign']['queue_counts']=={'pending':0,'running':0,'completed':120,'failed':0}
 assert s['campaign'].get('authorization_error') is None and s['campaign'].get('final_verification_error') is None
 jobs={j[0]:j for j in s['jobs']};assert jobs[JOB][1:3]==['RUNNING','heartpiece']
 for job in ('6782946','6782947'):assert jobs[job][1]=='PENDING' and 'Held' in jobs[job][4]
 failed=[u for u in s['units'] if u['run'].startswith('2026-11-01-14-53_full-original-direct-qualification')];assert len(failed)==1 and failed[0]['state']=='failed' and 'imported numerical helper' in failed[0]['progress'] and 'ideas' in failed[0]['progress']
 assert s['final_training']['state']=='disabled'
 after=identity(monitor['pid']);assert {k:after[k] for k in ('pid','birth','uids','cgroup','cmdline','fd1','fd2')}=={k:monitor[k] for k in ('pid','birth','uids','cgroup','cmdline','fd1','fd2')}
 monitor['cmdline']=[x.decode() for x in monitor['cmdline']]
 copies={name:put(name,raw) for name,raw in [('first_report.md',report_raw),('first_status.json',status_raw),('monitor_6783003.log',log_raw),('runtime_allocation.json',allocation_raw),('batch_cgroup_procs.raw',members_raw),('actual_git_source_tree.raw',tree_raw)]}
 assert not NUM.intersection(sys.modules)
 return {'schema_version':1,'kind':'independent_actual_cpu_monitor_runtime_and_first_full_report','passed':True,'job_id':JOB,'step_id':os.environ['SLURM_STEP_ID'],'hostname':socket.gethostname(),'uid':os.getuid(),'pid':os.getpid(),'cgroup':own_cgroup,'proc_stat':Path('/proc/self/stat').read_text(),'observed_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'source_commit':SOURCE,'source_file_count':265,'source_manifest_sha256':sha(manifest_raw),'archive_sha256':sha(archive_raw),'git_tree_query':tree_cmd,'source_files':checks,'archive_files':archive_checks,'requested_cpus':1,'allocated_logical_cpus':int(alloc['cpu']),'actual_threads_per_core':int(field(n['stdout'],'ThreadsPerCore')),'host_memory_gib':2,'gpu_count':0,'monitor_process':monitor,'snapshots':copies,'report_sections':sections,'campaign_counts':s['campaign']['queue_counts'],'held_jobs':[jobs[j] for j in ('6782946','6782947')],'actual_failed_direct_unit':failed[0],'final_training_state':s['final_training']['state'],'stale_auxiliary_progress_rows':[u for u in s['units'] if u['state']=='running' and 'pixel-study-monitor_source-commit-fb3071f' in u['run']],'numerical_modules_present':sorted(NUM.intersection(sys.modules)),'full_tick_executed_by_reviewer':False,'actual_scheduler_job_actions':0,'scientific_credit':0,'worker_admission':False}
put('actual_metadata_step_identity.json',(json.dumps({'job_id':os.environ.get('SLURM_JOB_ID'),'step_id':os.environ.get('SLURM_STEP_ID'),'pid':os.getpid(),'uid':os.getuid(),'hostname':socket.gethostname(),'cgroup':Path('/proc/self/cgroup').read_text(),'proc_stat':Path('/proc/self/stat').read_text(),'at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()},indent=2)+'\n').encode())
try:
 result=main()
except BaseException:
 put('runtime_review_second_failure.txt',traceback.format_exc().encode());raise
raw=(json.dumps(result,indent=2,allow_nan=False)+'\n').encode();q=put('runtime_and_first_report.json',raw);print(json.dumps({'receipt':q,'passed':True,'step_id':result['step_id'],'source_files':265,'campaign_counts':result['campaign_counts']}))
