"""Release the one saved replacement CPU monitor, with no repeat or old-job action."""
import argparse,hashlib,json,os,re,socket,subprocess,sys
from datetime import datetime,timezone
from pathlib import Path
PACKET=Path(__file__).absolute().parent
RUN=PACKET.parent.parent
JOB='6781231'
SOURCE='fb3071f3772ed836144f5876dbdda1113a5339d5'
SUBMISSION_SHA='0d5faa27c218dec7fa56aa2f7f960aa194e3e268c6dda21c37babaa31172cc6c'
INITIAL_PUBLICATION='d40b75ee7b79c48cb52881158e1c28754e927bfd'
PINS={'monitor_replacement_plan.json':'f5b79f3113959b3fa14a50acf3ab5945aede66bdd3c45dee5a3e27c1d8dacedf','monitor.sbatch':'1062f826cce0649376a72b9ce04058b3aec33dd44dddadf2d1917ac69d0c47bf','root_independent_preparation_review.json':'f5eb37c6f505218643bdd8cc6113cf0eb3785dd8f9e6ebaad2bc5564731ef3aa'}
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
 values=re.findall(r'(?:^|\s)'+re.escape(name)+r'=([^\s]+)',raw);require(len(values)==1,'missing or repeated owned field: '+name);return values[0]
def validate_held(raw):
 expected={'JobId':JOB,'UserId':'sl5nw(2618919)','Account':'sl5nw_base','Partition':'nolim','QOS':'normal','JobState':'PENDING','Reason':'JobHeldUser','Requeue':'0','Restarts':'0','NumCPUs':'1','NumTasks':'1','CPUs/Task':'1','MinMemoryNode':'2G','TresPerNode':'gres/gpu:0','TimeLimit':'7-00:00:00','Command':str(PACKET/'monitor.sbatch')}
 require(all(field(raw,k)==v for k,v in expected.items()) and field(raw,'NumNodes') in {'1','1-1'},'actual owned held CPU monitor request changed')
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
 proof_raw=contents(PACKET/'held_monitor_review.json');proof=json.loads(proof_raw)
 require(proof['job_id']==JOB and proof['passed'] is True and proof['source_commit']==SOURCE and proof['submission_receipt_sha256']==SUBMISSION_SHA and proof['old_monitor_job_id']=='6781087' and proof['session_cron_verified'] is True and proof['cadence_seconds']==1200,'actual held full-monitor review required')
 query=proof['query'];require(query['command'] in (['scontrol','show','job',JOB,'-o'],['scontrol','show','job','-o',JOB]) and type(query['returncode']) is int and query['returncode']==0 and query['stderr']=='','saved exact owned held query changed');validate_held(query['stdout'])
 require(sha(contents(Path(proof['full_monitor_verification'])))==proof['full_monitor_verification_sha256'],'saved full-monitor verification bytes changed')
 observed=datetime.fromisoformat(proof['at_utc'].replace('Z','+00:00'))
 require(observed.tzinfo is not None and observed.utcoffset().total_seconds()==0 and 0 <= (datetime.now(timezone.utc)-observed).total_seconds() < 1200,'held monitor review is stale or not UTC')
 review=read(PACKET/'release_implementation_review_6781231.json')
 require(review['job_id']==JOB and review['source_sha256']==sha(contents(Path(__file__).absolute())) and review['release_executed'] is False,'release source review changed')
 for name in ['release_intent.json','release_receipt.json','release_return.json','release_preflight_return.json','release_owned_job_before.raw','release_owned_job_before_stderr.raw','release_stdout.raw','release_stderr.raw']:
  require(not (PACKET/name).exists() and not (PACKET/name).is_symlink(),'a prior release attempt exists; never retry')
 if publication is not None:
  require(re.fullmatch(r'[0-9a-f]{40}',publication) is not None,'full published release commit required')
  subprocess.run(['git','-C',str(repo),'merge-base','--is-ancestor',publication,'refs/remotes/origin/merge-atari-cleanrl'],check=True)
  for path in [Path(__file__).absolute(),PACKET/'release_implementation_review_6781231.json',PACKET/'held_monitor_review.json',PACKET/'submission_receipt.json']+[PACKET/name for name in PINS]:
   require(contents(path)==subprocess.check_output(['git','-C',str(repo),'cat-file','blob',publication+':'+str(path.relative_to(repo))]),'release prerequisite was not published: '+str(path))
 return {'kind':'owned_uva_persistent_monitor_release_intent','cluster':'uva','job_id':JOB,'command':COMMAND,'at_utc':datetime.now(timezone.utc).isoformat(),'published_release_commit':publication,'submission_receipt_sha256':SUBMISSION_SHA,'held_monitor_review_sha256':sha(proof_raw),'release_review_sha256':sha(contents(PACKET/'release_implementation_review_6781231.json')),'release_source_sha256':sha(contents(Path(__file__).absolute())),'automatic_retry_permitted':False,'old_monitor_job_id':'6781087','old_monitor_stopped':False,'scientific_hypothesis_credit':0,'worker_admission':False}
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
 receipt={'kind':'owned_uva_persistent_monitor_release','cluster':'uva','job_id':JOB,'command':COMMAND,'returncode':0,'stdout':stdout.decode(),'stderr':'','at_utc':datetime.now(timezone.utc).isoformat(),'submission_receipt_sha256':SUBMISSION_SHA,'release_review_sha256':value['release_review_sha256'],'held_monitor_review_sha256':value['held_monitor_review_sha256'],'published_release_commit':value['published_release_commit'],'release_source_sha256':value['release_source_sha256'],'automatic_retry_permitted':False,'old_monitor_job_id':'6781087','old_monitor_stopped':False,'scientific_hypothesis_credit':0,'worker_admission':False}
 put(PACKET/'release_receipt.json',encoded(receipt));return receipt

def main():
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--publication-commit');parser.add_argument('--release',action='store_true');args=parser.parse_args()
 require(not args.release or args.publication_commit is not None,'publication commit required before release')
 value=checked(args.publication_commit)
 print(json.dumps(release_once(value) if args.release else {'validated':True,'job_id':JOB,'command':COMMAND,'scheduler_queries':0,'release_calls':0,'old_monitor_unchanged':True}))
if __name__=='__main__':main()
