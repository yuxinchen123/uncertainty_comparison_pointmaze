"""Release the one saved replacement CPU monitor, with no repeat or old-job action."""
import argparse,hashlib,json,os,re,socket,subprocess,sys
from datetime import datetime,timezone
from pathlib import Path
PACKET=Path(__file__).absolute().parent
RUN=PACKET.parent.parent
JOB='6782816'
SOURCE='31fd9fe7cd96343741e8b48a18019efee9b4ac5c'
SUBMISSION_SHA='74685cccf6f349244aed89d53481583ec25c92f8be76ab45148f52f943778d2d'
INITIAL_PUBLICATION='f299d33f6fe6ec4dede1e839ab6025b41a97d369'
PINS={'monitor_replacement_plan.json': '909a59d9ad51c0000858ad04c2e464d5a42efbcfbf162d945a91a1fddbc9025a', 'monitor.sbatch': 'c35675a54e110293d8f8e51fc30c11fe7e8dcb541724d64494eed66799878f21', 'root_packet_and_actual_candidate_review.json': '922b7a42220be06aa48b828e88caefdb6d26857792c40d9580871976c3c921f6'}
PROOF_SHA='43cf45d0eca313fc9a82e860360929e2d48d35b75163b2801550f9a5761688ce'
BINDING_SHA='c37da9b048395dfd366cbb93fb978bc9a8d968e74161805172c2f1371f1ffa3b'
COMMAND=['scontrol','release',JOB]
NUM={'jax','jaxlib','numpy','torch','flax','optax','cupy','scipy'}
def require(value,message):
 if not value:raise ValueError(message)
def sha(raw):return hashlib.sha256(raw).hexdigest()
def encoded(value):return (json.dumps(value,indent=2,allow_nan=False)+'\n').encode()
def contents(path):
 require(path.is_absolute() and path.resolve(strict=True)==path and path.is_file() and not path.is_symlink(),'exact ordinary artifact required')
 raw=path.read_bytes();require(len(raw)<=4*1024*1024,'metadata exceeds bound');return raw
def read(path):return json.loads(contents(path))
def put(path,raw):
 require(path.parent.resolve(strict=True)==path.parent and not path.exists() and not path.is_symlink(),'preserve every prior release attempt')
 fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
 with os.fdopen(fd,'wb') as out:out.write(raw);out.flush();os.fsync(out.fileno())
 fd=os.open(path.parent,os.O_RDONLY|os.O_DIRECTORY);os.fsync(fd);os.close(fd)
def field(raw,name):
 values=re.findall(r'(?:^|\s)'+re.escape(name)+r'=([^\s]*)',raw);require(len(values)==1,'missing or repeated owned field: '+name);return values[0]
def validate_held(raw):
 expected={'JobId':JOB,'UserId':'sl5nw(2618919)','Account':'sl5nw_base','Partition':'nolim','QOS':'normal','JobState':'PENDING','Reason':'JobHeldUser','Requeue':'0','Restarts':'0','NumCPUs':'1','NumTasks':'1','CPUs/Task':'1','MinMemoryNode':'2G','TresPerNode':'gres/gpu:0','TimeLimit':'7-00:00:00','Command':str(PACKET/'monitor.sbatch')}
 require(all(field(raw,k)==v for k,v in expected.items()) and field(raw,'NumNodes') in {'1','1-1'} and field(raw,'NodeList') in {'','(null)'} and field(raw,'AllocTRES')=='(null)','actual owned held CPU monitor request changed')
 pairs=[value.split('=',1) for value in field(raw,'ReqTRES').split(',')]
 require(all(len(p)==2 for p in pairs) and len({p[0] for p in pairs})==len(pairs),'invalid or duplicate requested resources')
 resources=dict(pairs)
 require(resources.get('cpu')=='1' and resources.get('mem')=='2G' and resources.get('node')=='1'
  and all((key=='gres/gpu' or key.startswith('gres/gpu:')) and value=='0' for key,value in resources.items() if key.startswith('gres/')),'requested processors, memory, nodes or zero GPU shape changed')
def checked(publication=None):
 require(socket.gethostname().split('.')[0]=='portal05' and os.getuid()==2618919 and not NUM.intersection(sys.modules),'wrong local host/owner or numerical import')
 repo=RUN.parents[2]
 for name,digest in PINS.items():
  raw=contents(PACKET/name);require(sha(raw)==digest and raw==subprocess.check_output(['git','-C',str(repo),'cat-file','blob',INITIAL_PUBLICATION+':'+str((PACKET/name).relative_to(repo))]),'published preparation changed: '+name)
 plan=read(PACKET/'monitor_replacement_plan.json');receipt_raw=contents(PACKET/'submission_receipt.json');receipt=json.loads(receipt_raw)
 require(sha(receipt_raw)==SUBMISSION_SHA and receipt['kind']=='owned_uva_persistent_monitor_submission' and receipt['cluster']=='uva' and receipt['job_id']==JOB and receipt['source_commit']==SOURCE and receipt['held'] is True and receipt['released'] is False,'exact saved held submission changed')
 row=[JOB,'monitor',SOURCE,str(PACKET/'monitor.sbatch')]
 require(receipt['submission_row']==row and receipt['held_command']==plan['submission_command'],'exact submission row or request changed')
 for name in ['submitted_jobids.txt','submitted_jobids_uva.txt']:require(contents(RUN/'slurm'/name).decode().splitlines().count(JOB)==1,'exact owned CPU monitor ID missing or duplicated')
 require([r for r in contents(RUN/'slurm/submissions.tsv').decode().splitlines() if r.split('\t',1)[0]==JOB]==['\t'.join(row)],'exact owned CPU monitor TSV row changed')
 binding_raw=contents(PACKET/'held_binding.json');binding=json.loads(binding_raw)
 require(sha(binding_raw)==BINDING_SHA and type(binding['schema_version']) is int and binding['schema_version']==1 and binding['kind']=='held_owned_cpu_monitor_binding' and binding['job_id']==JOB and binding['source_commit']==SOURCE and binding['submission_receipt']=={'path':str(PACKET/'submission_receipt.json'),'sha256':SUBMISSION_SHA} and binding['submission_row']==row and binding['command']==plan['submission_command'] and binding['released'] is False and binding['requested_gpu_count']==0 and binding['worker_admitted'] is False,'actual held binding changed')
 proof_raw=contents(PACKET/'actual_held_monitor_proof.json');proof=json.loads(proof_raw)
 require(sha(proof_raw)==PROOF_SHA and type(proof['schema_version']) is int and proof['schema_version']==1 and proof['kind']=='actual_full_owned_cpu_monitor_held_proof' and proof['target_job_id']==JOB and proof['passed'] is True and proof['monitor_job_id']=='6781231' and proof['monitor_source_commit']=='fb3071f3772ed836144f5876dbdda1113a5339d5' and proof['native_cron_id']=='7cbb1ee6-54aa-4e69-ab3d-a70fca25b3e5' and proof['cadence_seconds']==1200 and type(proof['release_calls']) is int and proof['release_calls']==0,'actual held full-monitor/Cron proof required')
 for key in ['shape','verification','status','report']:require(sha(contents(Path(proof[key]['path'])))==proof[key]['sha256'],'saved full-monitor proof bytes changed')
 shape=read(Path(proof['shape']['path']));query=read(PACKET/'held_scontrol_return.json');raw=contents(PACKET/'held_scontrol_stdout.raw')
 require(shape['passed'] is True and shape['job_id']==JOB and shape['raw_stdout_sha256']==sha(raw) and shape['raw_observation']==str(PACKET/'held_scontrol_return.json') and shape['unallocated_node_unknown'] is True,'actual held shape evidence changed')
 require(query['command'] in (['scontrol','show','job',JOB,'-o'],['scontrol','show','job','-o',JOB]) and type(query['returncode']) is int and query['returncode']==0 and contents(PACKET/'held_scontrol_stderr.raw')==b'','saved exact owned held query changed');validate_held(raw.decode('utf-8',errors='strict'))
 verification=read(Path(proof['verification']['path']));status=read(Path(proof['status']['path']));report=contents(Path(proof['report']['path'])).decode('utf-8',errors='strict')
 require(verification['passed'] is True and verification['monitor_job_id']=='6781231' and verification['source_commit']=='fb3071f3772ed836144f5876dbdda1113a5339d5' and verification['step_id']==proof['monitor_step'] and type(verification['returncode']) is int and verification['returncode']==0 and verification['status_sha256']==proof['status']['sha256'] and verification['report_sha256']==proof['report']['sha256'] and verification['queue_counts']=={'pending':0,'running':0,'completed':120,'failed':0},'old monitor full report verification changed')
 require(proof['held_row']==[JOB,'PENDING','','gres/gpu:0','(JobHeldUser)'] and status['jobs'].count(proof['held_row'])==1 and len([row for row in status['jobs'] if row[0]=='6781231' and row[1]=='RUNNING'])==1 and status['campaign']['queue_counts']==verification['queue_counts'] and all(('\n'+section+'\n') in report for section in ['TRAINING','WAITING','HELD','FINISHED','PLANNED']),'actual old running/new held rows or complete report changed')
 observed=datetime.fromisoformat(proof['checked_at_utc'].replace('Z','+00:00'));expires=datetime.fromisoformat(proof['expires_at_utc'].replace('Z','+00:00'))
 require(observed.tzinfo is not None and observed.utcoffset().total_seconds()==0 and expires.tzinfo is not None and expires.utcoffset().total_seconds()==0 and (expires-observed).total_seconds()==1200 and 0 <= (datetime.now(timezone.utc)-observed).total_seconds() < 1200,'held monitor proof is stale or not UTC')
 review=read(PACKET/'release_implementation_review_6782816.json')
 require(review['job_id']==JOB and review['source_sha256']==sha(contents(Path(__file__).absolute())) and review['release_executed'] is False and review['held_proof_sha256']==PROOF_SHA and review['submission_receipt_sha256']==SUBMISSION_SHA,'release source review changed')
 for name in ['release_intent.json','release_receipt.json','release_return.json','release_preflight_return.json','release_owned_job_before.raw','release_owned_job_before_stderr.raw','release_stdout.raw','release_stderr.raw']:
  require(not (PACKET/name).exists() and not (PACKET/name).is_symlink(),'a prior release attempt exists; never retry')
 if publication is not None:
  require(re.fullmatch(r'[0-9a-f]{40}',publication) is not None,'full published release commit required')
  subprocess.run(['git','-C',str(repo),'merge-base','--is-ancestor',publication,'refs/remotes/origin/merge-atari-cleanrl'],check=True)
  for path in [Path(__file__).absolute(),PACKET/'release_implementation_review_6782816.json',PACKET/'actual_held_monitor_proof.json',PACKET/'submission_receipt.json',PACKET/'held_binding.json',PACKET/'held_scontrol_return.json',PACKET/'held_scontrol_stdout.raw',PACKET/'held_scontrol_stderr.raw',PACKET/'test_release_monitor_6782816.py']+[Path(proof[key]['path']) for key in ['shape','verification','status','report']]+[PACKET/name for name in PINS]:
   require(contents(path)==subprocess.check_output(['git','-C',str(repo),'cat-file','blob',publication+':'+str(path.relative_to(repo))]),'release prerequisite was not published: '+str(path))
 return {'kind':'owned_uva_persistent_monitor_release_intent','cluster':'uva','job_id':JOB,'command':COMMAND,'at_utc':datetime.now(timezone.utc).isoformat(),'published_release_commit':publication,'submission_receipt_sha256':SUBMISSION_SHA,'held_monitor_review_sha256':sha(proof_raw),'release_review_sha256':sha(contents(PACKET/'release_implementation_review_6782816.json')),'release_source_sha256':sha(contents(Path(__file__).absolute())),'automatic_retry_permitted':False,'old_monitor_job_id':'6781231','old_monitor_stopped':False,'scientific_hypothesis_credit':0,'worker_admission':False}
def capture(label,command):
 try:result=subprocess.run(command,capture_output=True,timeout=30,check=False)
 except BaseException as error:
  put(PACKET/(label+'_stdout.raw'),getattr(error,'stdout',None) or b'');put(PACKET/(label+'_stderr.raw'),getattr(error,'stderr',None) or b'')
  put(PACKET/(label+'_return.json'),encoded({'command':command,'returncode':None,'error_type':type(error).__name__,'error':str(error),'automatic_retry_permitted':False}));raise
 put(PACKET/(label+'_stdout.raw'),result.stdout);put(PACKET/(label+'_stderr.raw'),result.stderr)
 put(PACKET/(label+'_return.json'),encoded({'command':command,'returncode':result.returncode,'automatic_retry_permitted':False}))
 require(result.returncode==0 and result.stderr==b'','owned scheduler command failed; preserve evidence and do not retry')
 return result.stdout

def release_once(value):
 require(value['published_release_commit'] is not None,'published release required before action')
 put(PACKET/'release_intent.json',encoded(value))
 before=capture('release_preflight',['scontrol','show','job','-o',JOB])
 put(PACKET/'release_owned_job_before.raw',before)
 validate_held(before.decode())
 stdout=capture('release',COMMAND)
 receipt={'kind':'owned_uva_persistent_monitor_release','cluster':'uva','schema_version':1,'job_id':JOB,'command':COMMAND,'returncode':0,'stdout':stdout.decode(),'stderr':'','at_utc':datetime.now(timezone.utc).isoformat(),'submission_receipt_sha256':SUBMISSION_SHA,'release_review_sha256':value['release_review_sha256'],'held_monitor_review_sha256':value['held_monitor_review_sha256'],'published_release_commit':value['published_release_commit'],'release_source_sha256':value['release_source_sha256'],'automatic_retry_permitted':False,'old_monitor_job_id':'6781231','old_monitor_stopped':False,'scientific_hypothesis_credit':0,'worker_admission':False}
 put(PACKET/'release_receipt.json',encoded(receipt));return receipt

def main():
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--publication-commit');parser.add_argument('--release',action='store_true');args=parser.parse_args()
 require(not args.release or args.publication_commit is not None,'publication commit required before release')
 value=checked(args.publication_commit)
 print(json.dumps(release_once(value) if args.release else {'validated':True,'job_id':JOB,'command':COMMAND,'scheduler_queries':0,'release_calls':0,'old_monitor_unchanged':True}))
if __name__=='__main__':main()
