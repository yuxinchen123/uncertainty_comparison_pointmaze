"""Verify the actual new monitor's source, allocation, own first log and later attributed full snapshot."""
import ast
from datetime import datetime,timezone
import hashlib
import importlib.abc
import importlib.util
import io
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
import tarfile
import traceback
from zoneinfo import ZoneInfo

AUDIT=Path(__file__).absolute().parent
PACKET=Path('/p/rlprojects/RND_worktrees/merge-atari-cleanrl/10_jax_exploration_platform/benchmark_runs/2026-10-31-00-11_jaxatari_montezuma_pixels_cleanrl_ppo_rnd_environments-per-copy-128_rollout-steps-128_ideas-30-plus-20_steps-per-copy-2000000000_seed-start-1/remote_setup/2026-11-01-14-02_pixel-study-monitor_source-commit-31fd9fe7cd96343741e8b48a18019efee9b4ac5c_processor-count-1_host-memory-gib-2_time-limit-days-7_poll-interval-seconds-1200_partition-nolim_account-sl5nw-base_gpu-count-0')
RUN=PACKET.parent.parent
COMMIT='31fd9fe7cd96343741e8b48a18019efee9b4ac5c'
CHECKS=[]
BLOCKED=[]
assert os.environ.get('SLURM_JOB_ID')=='6781231' and os.environ.get('SLURM_STEP_ID','').isdecimal()

def sha(raw):return hashlib.sha256(raw).hexdigest()
def image(path):return sha(Path(path).read_bytes())
def ref(path):return {'path':str(path),'sha256':image(path)}
def doc(path):return json.loads(Path(path).read_bytes())
def check(value,message):assert value,message;CHECKS.append(message)
def save(name,value):
 raw=value if isinstance(value,bytes) else (json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
 with (AUDIT/name).open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
 return ref(AUDIT/name)
def fields(raw):
 return dict(re.findall(r'(?:^|\s)([^\s=]+)=([^\s]+)',raw))
class NoNumerical(importlib.abc.MetaPathFinder):
 def find_spec(self,name,path=None,target=None):
  if name.split('.')[0] in {'jax','jaxlib','numpy','torch','flax','optax','jaxtari','cupy','scipy'}:BLOCKED.append(name);raise RuntimeError('numerical import forbidden')
sys.meta_path.insert(0,NoNumerical())
sys.addaudithook(lambda event,args: (_ for _ in ()).throw(RuntimeError('GPU/library/network operation forbidden')) if (event in {'socket.connect','socket.bind','socket.getaddrinfo'} or event=='ctypes.dlopen' and args[0] is not None) else None)

try:
 capture=doc(AUDIT/'capture.json');plan=doc(PACKET/'monitor_replacement_plan.json');snapshot=doc(PACKET/'source_snapshot.json')
 protected={row['path']:row['sha256'] for row in doc(PACKET/'preparation_file_roster.json')['files']}
 check(all(image(path)==digest for path,digest in protected.items()),'all27 static packet/unit bytes retain exact reviewed hashes')
 check(snapshot['source_commit']==COMMIT and plan['helper_commit']==COMMIT and len(snapshot['files'])==260,'exact260source roster and monitor commit agree')
 frozen=Path(snapshot['source_root']);gitroot=Path(snapshot['clean_detached_source_root'])
 check(subprocess.check_output(['git','-C',str(gitroot),'rev-parse','HEAD'],text=True).strip()==COMMIT,'clean source Git HEAD equals exact31fd')
 branch=subprocess.run(['git','-C',str(gitroot),'symbolic-ref','--quiet','HEAD'],capture_output=True)
 check(branch.returncode==1,'exact source Git checkout remains detached')
 check(subprocess.check_output(['git','-C',str(gitroot),'status','--porcelain','--untracked-files=all'],text=True).strip()=='','detached source checkout remains completely clean')
 roster=subprocess.check_output(['git','-C',str(gitroot),'ls-tree','-rz',COMMIT,'--',*snapshot['source_prefixes']])
 git_entries={}
 for item in roster.split(b'\0'):
  if item:
   header,path=item.split(b'\t',1);mode,kind,blob=header.decode().split();check(kind=='blob','Git roster entry is an ordinary blob')
   git_entries[path.decode()]={'mode':mode,'blob':blob}
 check(set(git_entries)==set(snapshot['files']),'all260 committed Git paths exactly match archive manifest')
 request=''.join(COMMIT+':'+path+'\n' for path in snapshot['files']).encode()
 blobs=subprocess.check_output(['git','-C',str(gitroot),'cat-file','--batch'],input=request);cursor=0;source_rows={}
 for path,entry in snapshot['files'].items():
  newline=blobs.index(b'\n',cursor);header=blobs[cursor:newline].decode().split();cursor=newline+1
  check(len(header)==3 and header[1]=='blob','batch Git object is a blob')
  length=int(header[2]);raw=blobs[cursor:cursor+length];cursor+=length;check(blobs[cursor:cursor+1]==b'\n','batch Git blob boundary intact');cursor+=1
  expected=git_entries[path]
  check(header[0]==expected['blob']==entry['git_blob'] and expected['mode']==entry['git_mode'],'exact Git blob identity and mode '+path)
  check(raw==(frozen/path).read_bytes()==(gitroot/path).read_bytes() and sha(raw)==entry['sha256'] and len(raw)==entry['bytes'],'exact Git/archive-snapshot/detached-source bytes '+path)
  executable=0o111 if expected['mode']=='100755' else 0
  check(bool((frozen/path).stat().st_mode&0o111)==bool(executable) and bool((gitroot/path).stat().st_mode&0o111)==bool(executable),'actual executable mode matches Git '+path)
  source_rows[path]={'git_blob':header[0],'git_mode':expected['mode'],'sha256':sha(raw),'bytes':len(raw)}
 check(cursor==len(blobs),'all Git batch bytes consumed exactly')
 archive=Path(snapshot['archive']['path']);check(image(archive)==snapshot['archive']['sha256'],'actual immutable Git archive SHA matches manifest')
 with tarfile.open(archive,'r:') as stream:
  members=stream.getmembers();file_members=[m for m in members if m.isfile()]
  check(all(m.name.startswith(COMMIT+'/') for m in file_members),'every archive file has the exact declared commit prefix')
  regular={m.name[len(COMMIT)+1:]:m for m in file_members}
  check(all(m.isfile() or m.isdir() for m in members) and set(regular)==set(snapshot['files']),'archive has only exact260ordinary source files and directories')
  for path,entry in snapshot['files'].items():
   member=regular[path];raw=stream.extractfile(member).read()
   check(sha(raw)==entry['sha256'] and bool(member.mode&0o111)==(entry['git_mode']=='100755'),'actual archived source bytes and executable mode '+path)
 runtime=doc(AUDIT/'runtime_allocation.json');release=doc(AUDIT/'release_receipt.json');submission=doc(AUDIT/'submission_receipt.json')
 check(runtime['job_id']=='6782816' and runtime['hostname']=='heartpiece' and runtime['uid']==2618919 and runtime['returncode']==0 and runtime['stderr']=='','actual new runtime owner/job/host query success')
 check(release['job_id']=='6782816' and release['command']==['scontrol','release','6782816'] and release['returncode']==0 and release['release_source_sha256']==image(PACKET/'release_monitor_6782816.py')=='95be614c6acff984d4c2f92f3bb113df735653045fc82d388ec3a5497d2a6fae','actual one-shot release names6782816;95be is release-source SHA,31fd is monitor source commit')
 check(release['submission_receipt_sha256']==image(AUDIT/'submission_receipt.json') and release['old_monitor_stopped'] is False,'actual release binds exact durable submission and preserves old monitor')
 for name in ['submitted_jobids_uva.txt','submitted_jobids.txt']:check((RUN/'slurm'/name).read_text().splitlines().count('6782816')==1,'actual new ID occurs once in '+name)
 expected_row=['6782816','monitor',COMMIT,str(PACKET/'monitor.sbatch')]
 check([line.split('\t') for line in (RUN/'slurm/submissions.tsv').read_text().splitlines() if line.split('\t')[0]=='6782816']==[expected_row],'actual new submission row binds31fd and exact frozen script')
 spec=importlib.util.spec_from_file_location('new_actual_guard',PACKET/'allocation_guard.py');guard=importlib.util.module_from_spec(spec);spec.loader.exec_module(guard)
 actual,topology=fields(runtime['stdout']),fields(runtime['node_observation']['stdout'])
 guard.validate(actual,'6782816',runtime['hostname'],runtime['cgroup'],topology)
 check(actual['NumCPUs']=='2' and actual['CPUs/Task']=='1' and topology['ThreadsPerCore']=='2','requested oneCPU and actual two logical threads are justified by actualSMT2')
 check(actual['ReqTRES'].startswith('cpu=1,mem=2G,node=1,') and actual['AllocTRES'].startswith('cpu=2,mem=2G,node=1,') and actual['TresPerNode']=='gres/gpu:0' and actual['TimeLimit']=='7-00:00:00' and actual['Requeue']=='0','actual requested/allocated2GiB zeroGPU seven-day no-requeue shape matches reviewed packet')
 log=(AUDIT/'new_monitor.log').read_bytes();firstline,body=log.split(b'\n',1);started=json.loads(firstline)
 check(started=={'source_commit':COMMIT,'files_verified':260,'full_tick_executed':False,'numerical_imports':False},'actual new own log begins successful exact260source startup verification')
 check(body.startswith(b'run or job') and body.endswith(b'\n\n') and not body.endswith(b'\n\n\n'),'actual first own-log table is complete and contains exactly the print-added final newline')
 for heading in ['TRAINING','WAITING','HELD','FINISHED','PLANNED']:check(body.splitlines().count(heading.encode())==1,'actual first own-log has one '+heading+' section')
 check(b'Traceback' not in log,'actual new first own-log contains no traceback')
 observed=datetime.fromisoformat(runtime['observed_at_utc']);log_end=datetime.fromisoformat(capture['files']['new_monitor.log']['origin_mtime_utc'])
 check(observed<=log_end,'first own-log completion bound follows actual runtime guard observation')
 old_status=doc(AUDIT/'first_status.json')
 check(old_status['time']=='2026-11-01 14:47:24 PT','overwritten latest snapshot is explicitly identified as oldmonitor time; not claimed as new')
 attribution={'source':'actual new own log after source verification','runtime_guard_observed_at_utc':runtime['observed_at_utc'],'first_own_log_mtime_utc':capture['files']['new_monitor.log']['origin_mtime_utc'],'stdout_itself_has_no_timestamp':True,'first_new_status_snapshot_saved':False,'captured_latest_report_and_status_are_from_old_monitor_time':old_status['time']}
 check((AUDIT/'first_new_own_table.stdout.raw').read_bytes()==body,'preserved first own-table bytes remain exact')
 # Only pure formatting is imported; no monitor.tick, controller or scheduler query is invoked here.
 monitor_path=frozen/'10_jax_exploration_platform/scripts/atari_pixel_campaign/monitor.py'
 sys.path.insert(0,str(monitor_path.parent));spec=importlib.util.spec_from_file_location('format_actual_new_table',monitor_path);monitor=importlib.util.module_from_spec(spec);spec.loader.exec_module(monitor)
 audit_path=RUN/'remote_setup/2026-11-01-12-50_initial-comparisons-completion_ideas-30_gpu-types-2_copies-1-2_comparisons-120_source-d28e6735e797/initial_comparisons_audit.json'
 cohort=doc(audit_path);check(cohort['passed'] is True and cohort['comparisons']==120,'exact historical initial120audit is available as the terminal case roster')
 initial=[]
 check(b'0 awaiting current runs' in body,'first own-log table reports zero awaiting comparisons; exact120count verified in later attributed status')
 qualification=Path('/p/rlprojects/RND_worktrees/merge-atari-cleanrl/10_jax_exploration_platform/benchmark_runs/2026-10-31-00-11_jaxatari_montezuma_pixels_cleanrl_ppo_rnd_environments-per-copy-128_rollout-steps-128_ideas-30-plus-20_steps-per-copy-2000000000_seed-start-1/remote_setup/2026-11-01-13-50_original-qualification_explicit-pytest-root_jaguar03_gpu-count-1_processor-count-8_host-memory-gib-64_time-limit-hours-3_copies-1_environments-128_rollout-steps-128_epochs-4_minibatches-4_warmup-rollouts-50_reference-updates-1_seed-1').name;rows=[x for x in old_status['units'] if x['run']==qualification];check(len(rows)==1 and rows[0]['state']=='completed','actual completed qualification has one terminal unit template')
 row=rows[0];rendered=monitor.table_row(row['run'],row['pool'],monitor.gpu_label(row['gpu'][0]),'FINISHED',row['progress']).encode()
 check(body.count(rendered)==1,'actual first new own-log contains the current6782778qualification asFINISHED')
 new_proof=PACKET/'first_independent_full_tick_verification.json';check(image(new_proof)=='e1146c18c651566a70bf41f58beb3df66983da70e6c87d6836b58aa7e4633dca','later actual new-allocation full-snapshot proof exact hash')
 proof=doc(new_proof);check(proof['passed'] is True and proof['monitor_job_id']=='6782816' and proof['actual_step_id']=='0' and proof['source_commit']==COMMIT and proof['source_returncode']==proof['returncode']==0,'later fullsnapshot is actual new6782816numeric step0 with exact31fd source')
 for key in ['report','status']:check(ref(proof[key]['path'])==proof[key],'later fullsnapshot '+key+' raw hash exact')
 stdout=(PACKET/'first_independent_full_tick_stdout.raw').read_bytes();stderr=(PACKET/'first_independent_full_tick_stderr.raw').read_bytes();report=Path(proof['report']['path']).read_bytes();status=doc(proof['status']['path'])
 check(stdout.endswith(b'\n\n') and not stdout.endswith(b'\n\n\n'),'later fullsnapshot stdout has exact declared table/print newline shape')
 expected=b'# Pixel Montezuma study status\n\n'+status['time'].encode()+b'\n\n```text\n'+stdout[:-1]+b'```\n'
 check(report==expected and stderr==b'','later attributed fullsnapshot has exact framing and no stderr')
 for heading in ['TRAINING','WAITING','HELD','FINISHED','PLANNED']:check(stdout.splitlines().count(heading.encode())==1,'later attributed fullsnapshot contains one '+heading+' section')
 check(status['campaign']['queue_counts']==proof['queue_counts']=={'completed':120,'pending':0,'running':0,'failed':0},'later actual newfullsnapshot has120complete and zero other initial cases')
 check([row for row in status['jobs'] if row[0]=='6782816'][0][1:3]==['RUNNING','heartpiece'],'later actual newfullsnapshot reports new monitorRUNNING on actualheartpiece')
 check([x for x in status['units'] if x['run']==qualification][0]['state']=='completed','later actual newfullsnapshot also preserves completed6782778qualification')
 check(all(k in status for k in ['site_validation','allocation_observations','aces_allocation_observations','aces_full_site_checks']),'later fullsnapshot retains UVA and ACES operational fields')
 status_time=datetime.strptime(status['time'],'%Y-%m-%d %H:%M:%S PT').replace(tzinfo=ZoneInfo('America/Los_Angeles')).astimezone(timezone.utc)
 check(observed<=status_time<=datetime.fromisoformat(proof['checked_at_utc']),'later fullsnapshot status timestamp lies after actual startup and before its real verification')
 source_stdout=doc(PACKET/'first_independent_full_tick_source_stdout.raw')
 check(source_stdout==started and (PACKET/'first_independent_full_tick_source_stderr.raw').read_bytes()==b'','later attributed source verifier repeats exact260success without stderr')
 command=['sacct','-n','-P','-j','6782816','--format=JobIDRaw,State,ExitCode,NodeList,Elapsed'];accounting=subprocess.run(command,capture_output=True,timeout=30,check=False)
 save('new_monitor_accounting.stdout.raw',accounting.stdout);save('new_monitor_accounting.stderr.raw',accounting.stderr)
 check(accounting.returncode==0,'only exact newmonitor accounting query returned0')
 accounting_rows=[line.split('|') for line in accounting.stdout.decode().splitlines() if line.strip()]
 check([row for row in accounting_rows if row[0]=='6782816'][0][1]=='RUNNING' and [row for row in accounting_rows if row[0]=='6782816.0'][0][1:3]==['COMPLETED','0:0'],'actual newmonitor remainsRUNNING and its attributed numeric0 tick completed0:0')
 check(all(image(path)==digest for path,digest in protected.items()),'all frozen static packet/unit bytes unchanged throughout independent actual verification')
 check(not BLOCKED,'no numerical import attempted')
 result={'schema_version':1,'kind':'independent_new_persistent_monitor_runtime_first_log_and_attributed_full_snapshot_review','passed':True,'at_utc':datetime.now(timezone.utc).isoformat(),'audit_job_id':'6781231','audit_step_id':os.environ['SLURM_STEP_ID'],'new_monitor_job_id':'6782816','monitor_source_commit':COMMIT,'release_utility_source_sha256':'95be614c6acff984d4c2f92f3bb113df735653045fc82d388ec3a5497d2a6fae','checks':CHECKS,'check_count':len(CHECKS),'source_files':source_rows,'archive':ref(archive),'actual_runtime':ref(AUDIT/'runtime_allocation.json'),'requested_cpu':1,'allocated_logical_cpu':2,'actual_threads_per_core':2,'actual_host_memory':'2G','gpus':0,'time_limit':'7-00:00:00','requeue':0,'first_own_log':ref(AUDIT/'new_monitor.log'),'first_table_attribution':attribution,'initial_case_rows_displayed_individually':False,'initial_count_authority':'later explicitly attributed new6782816.0 full status, not invented first-status bytes','later_attributed_full_snapshot_proof':ref(new_proof),'later_actual_full_snapshot_time':status['time'],'later_actual_full_snapshot_counts':status['campaign']['queue_counts'],'actual_newmonitor_accounting_command':command,'actual_newmonitor_accounting_returncode':accounting.returncode,'old_monitor_preserved':True,'old_monitor_cancelled_by_reviewer':False,'all_auxiliary_checks_from_this_reviewer_finished_at_return':True,'numerical_imports':0,'network_calls':0,'gpu_actions':0,'job_submissions':0,'release_calls':0,'cancellation_calls':0,'canonical_writes':0,'scientific_credit':0,'case_credit':0,'idea_credit':0,'admission_granted':False,'worker_admitted':False}
 print(json.dumps({'passed':True,'receipt':save('review.json',result),'audit_step':os.environ['SLURM_STEP_ID'],'checks':len(CHECKS)}))
except BaseException:
 save('failure_final.json',{'traceback':traceback.format_exc(),'checks_before_failure':CHECKS,'audit_job_id':'6781231','audit_step_id':os.environ['SLURM_STEP_ID'],'canonical_writes':0,'job_actions':0});raise
