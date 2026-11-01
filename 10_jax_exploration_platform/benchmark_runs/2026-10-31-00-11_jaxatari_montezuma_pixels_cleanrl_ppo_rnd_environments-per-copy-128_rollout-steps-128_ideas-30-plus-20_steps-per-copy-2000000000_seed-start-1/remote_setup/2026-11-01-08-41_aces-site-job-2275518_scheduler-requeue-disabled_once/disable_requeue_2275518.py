"""Prepare or perform one exact-owned Requeue=0 update; never restart a job."""
import argparse,base64,hashlib,inspect,json,os,re,shlex,socket,subprocess,sys
from datetime import datetime,timezone
from pathlib import Path
RUN=Path('/p/rlprojects/RND_worktrees/merge-atari-cleanrl/10_jax_exploration_platform/benchmark_runs/2026-10-31-00-11_jaxatari_montezuma_pixels_cleanrl_ppo_rnd_environments-per-copy-128_rollout-steps-128_ideas-30-plus-20_steps-per-copy-2000000000_seed-start-1')
ACTION=Path(__file__).absolute().parent
WRAPPER=RUN/'remote_setup/2026-11-01-07-50_aces-autodetect-held-submission_helper-bad2ee3736a1_gpu-count-1_processor-count-8_host-memory-gib-32_time-limit-hours-3'
JOB='2275518'
SUBMISSION_SHA='358f16cc1eaa369638153ee7dc090572cca49652ed5ae1595714e2617ef8735f'
RELEASE_SHA='9b9ab540b0ab616dbeff749eee5a6ea28a2dd16925008d793caca14d02d796c1'
BOUNDARY={'workers_permitted':False,'campaign_claims_permitted':False,'scientific_hypothesis_credit':0,'comparison_credit':0,'idea_credit':0,'admission_credit':0}
COMMAND=['scontrol','update','JobId='+JOB,'Requeue=0']
NAMES={'intent.json','pre_job.stdout.raw','pre_job.stderr.raw','pre_job.return.json','update.stdout.raw','update.stderr.raw','update.return.json','post_job.stdout.raw','post_job.stderr.raw','post_job.return.json','receipt.json'}
NUM={'jax','jaxlib','numpy','torch','flax','optax','cupy','scipy'}
def require(value,message):
 if not value:raise ValueError(message)
def sha(raw):return hashlib.sha256(raw).hexdigest()
def encoded(value):return (json.dumps(value,indent=2,allow_nan=False)+'\n').encode()
def ordinary(path):
 require(path.is_absolute() and path.resolve(strict=True)==path and path.is_file() and not path.is_symlink(),'ordinary exact artifact required')
 raw=path.read_bytes();require(len(raw)<=4*1024*1024,'metadata exceeds bound');return raw
def read(path):return json.loads(ordinary(path))
def put(path,raw):
 require(path.parent.resolve(strict=True)==path.parent and not path.exists() and not path.is_symlink(),'preserve prior attempt')
 descriptor=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
 with os.fdopen(descriptor,'wb') as output:output.write(raw);output.flush();os.fsync(output.fileno())
 descriptor=os.open(path.parent,os.O_RDONLY|os.O_DIRECTORY);os.fsync(descriptor);os.close(descriptor)
def field(raw,name):
 rows=re.findall(r'(?:^|\s)'+re.escape(name)+r'=([^\s]+)',raw)
 require(len(rows)==1,'missing or repeated exact owned field: '+name);return rows[0]
def validate_job(raw,packet,post=False):
 expected={'JobId':'2275518','UserId':'u.sl359075(56626)','Account':'158590754315','Partition':'gpu','QOS':'normal','NumCPUs':'8','NumTasks':'1','CPUs/Task':'8','MinMemoryNode':'32G','TimeLimit':'03:00:00','Restarts':'0','Command':str(packet/'job.sbatch'),'TresPerNode':'gres/gpu:h100:1'}
 require(all(field(raw,key)==value for key,value in expected.items()),'owned job identity, request or restart count changed')
 require(field(raw,'JobState') in {'PENDING','RUNNING'} and field(raw,'NumNodes') in {'1','1-1'},'job is terminal or has unsupported state')
 parts=[x.split('=',1) for x in field(raw,'ReqTRES').split(',')]
 require(all(len(x)==2 for x in parts) and len({x[0] for x in parts})==len(parts),'duplicate or invalid requested resources')
 values=dict(parts);expected_tres={'cpu':'8','mem':'32G','node':'1','gres/gpu':'1','gres/gpu:h100':'1'}
 require(all(values.get(k)==v for k,v in expected_tres.items()) and all(k in expected_tres for k in values if k.startswith('gres/')),'exact typed H100 request changed')
 require(field(raw,'Requeue') in ({'0'} if post else {'0','1'}),'scheduler requeue setting is not verified disabled' if post else 'unknown scheduler requeue setting')
 return field(raw,'Requeue')

REMOTE_PREFIX=r'''
import base64,datetime,hashlib,json,os,pathlib,pwd,re,socket,subprocess,sys
Path=pathlib.Path
JOB='2275518'
SUBMISSION_SHA='358f16cc1eaa369638153ee7dc090572cca49652ed5ae1595714e2617ef8735f'
RELEASE_SHA='9b9ab540b0ab616dbeff749eee5a6ea28a2dd16925008d793caca14d02d796c1'
COMMAND=['scontrol','update','JobId='+JOB,'Requeue=0']
BOUNDARY={'workers_permitted':False,'campaign_claims_permitted':False,'scientific_hypothesis_credit':0,'comparison_credit':0,'idea_credit':0,'admission_credit':0}
'''
REMOTE_SUFFIX=r'''
def capture(directory,label,command):
 try:result=subprocess.run(command,capture_output=True,timeout=30,check=False)
 except BaseException as error:
  put(directory/(label+'.stdout.raw'),getattr(error,'stdout',None) or b'');put(directory/(label+'.stderr.raw'),getattr(error,'stderr',None) or b'')
  put(directory/(label+'.return.json'),encoded({'command':command,'returncode':None,'error_type':type(error).__name__,'error':str(error),'automatic_retry_permitted':False}));raise
 put(directory/(label+'.stdout.raw'),result.stdout);put(directory/(label+'.stderr.raw'),result.stderr)
 put(directory/(label+'.return.json'),encoded({'command':command,'returncode':result.returncode,'automatic_retry_permitted':False}))
 require(result.returncode==0 and result.stderr==b'','owned command failed; preserve evidence and never retry')
 return result.stdout.decode()
def native_main(x):
 require(os.getuid()==56626 and pwd.getpwuid(os.getuid()).pw_name=='u.sl359075' and socket.gethostname().split('.')[0]=='alogin1','wrong native owner or host')
 require(not {'jax','jaxlib','numpy','torch','flax','optax','cupy','scipy'}.intersection(sys.modules),'numerical modules forbidden')
 require(type(x.get('schema_version')) is int and x['schema_version']==1 and x.get('kind')=='owned_aces_requeue_disable_intent' and x.get('job_id')==JOB and x.get('command')==COMMAND and x.get('automatic_retry_permitted') is False,'wrong exact update scope')
 require(all(type(x.get(k)) is type(v) and x[k]==v for k,v in BOUNDARY.items()),'nonzero claim flags')
 require(re.fullmatch(r'[0-9a-f]{40}',x['publication_commit']) is not None,'published action required')
 run=Path(x['run']);packet=Path(x['packet']);action=Path(x['native_action_directory'])
 require(run.resolve(strict=True)==run and packet.resolve(strict=True)==packet and packet.parent==run/'remote_setup' and action.parent==run/'remote_setup' and action.name==x['action_name'],'wrong native packet/action path')
 release=ordinary(packet/'release_receipt.json');submission=ordinary(packet/'submission_receipt.json')
 require(sha(release)==RELEASE_SHA and sha(submission)==SUBMISSION_SHA,'immutable native release/submission changed')
 s=json.loads(submission);r=json.loads(release)
 require(s['job_id']==r['job_id']==JOB and s['submission_row']==[JOB,packet.name,'bad2ee3736a1451094d5d1644b36b8cfc2d91c38',str(run)] and s['submission_row']==x['submission_row'],'native own job or source changed')
 require(r['command']==['scontrol','release',JOB] and type(r['returncode']) is int and r['returncode']==0,'native single release record invalid')
 require(ordinary(run/'slurm/submitted_jobids_aces.txt').decode().splitlines().count(JOB)==1 and [row for row in ordinary(run/'slurm/submissions_aces.tsv').decode().splitlines() if row.split('\t',1)[0]==JOB]==['\t'.join(s['submission_row'])],'native exact owned ledgers changed')
 for name,expected in x['native_packet_hashes'].items():require(sha(ordinary(packet/name))==expected,'native immutable packet bytes changed')
 require(not action.exists() and not action.is_symlink(),'prior native action exists; no retry')
 action.mkdir(mode=0o700);descriptor=os.open(action.parent,os.O_RDONLY|os.O_DIRECTORY);os.fsync(descriptor);os.close(descriptor)
 put(action/'intent.json',encoded(x))
 before=capture(action,'pre_job',['scontrol','show','job','-o',JOB]);setting=validate_job(before,packet)
 changed=setting=='1'
 if changed:capture(action,'update',COMMAND)
 else:
  put(action/'update.stdout.raw',b'');put(action/'update.stderr.raw',b'');put(action/'update.return.json',encoded({'command':COMMAND,'returncode':None,'performed':False,'reason':'already Requeue=0','automatic_retry_permitted':False}))
 after=capture(action,'post_job',['scontrol','show','job','-o',JOB]);validate_job(after,packet,post=True)
 receipt={'schema_version':1,'kind':'owned_aces_requeue_disabled_receipt','cluster':'aces','job_id':JOB,'command':COMMAND,'update_performed':changed,'requeue_before':setting,'requeue_after':'0','restart_count_before':'0','restart_count_after':'0','preflight_owner_verified':True,'submission_receipt_sha256':SUBMISSION_SHA,'release_receipt_sha256':RELEASE_SHA,'publication_commit':x['publication_commit'],'source_sha256':x['source_sha256'],'completed_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'automatic_retry_permitted':False,**BOUNDARY}
 put(action/'receipt.json',encoded(receipt))
 names=['intent.json','pre_job.stdout.raw','pre_job.stderr.raw','pre_job.return.json','update.stdout.raw','update.stderr.raw','update.return.json','post_job.stdout.raw','post_job.stderr.raw','post_job.return.json','receipt.json']
 print(json.dumps({'files':{name:base64.b64encode(ordinary(action/name)).decode() for name in names}}))
if __name__=='__main__':native_main(json.load(sys.stdin))
'''
def remote_program():
 return REMOTE_PREFIX+'\n'.join(inspect.getsource(f) for f in (require,sha,encoded,ordinary,put,field,validate_job))+REMOTE_SUFFIX

def local_payload(publication=None):
 require(socket.gethostname().split('.')[0]=='portal05' and os.getuid()==2618919 and not NUM.intersection(sys.modules),'local owner/host or numerical modules changed')
 plan=read(WRAPPER/'aces_full_site_held_submission_plan.json');packet=Path(plan['local_packet']);s=read(packet/'submission_receipt.json');r=read(packet/'release_receipt.json')
 require(sha(ordinary(packet/'submission_receipt.json'))==SUBMISSION_SHA and sha(ordinary(packet/'release_receipt.json'))==RELEASE_SHA,'exact adopted/released site required')
 require(s['job_id']==r['job_id']==JOB and r['command']==['scontrol','release',JOB] and type(r['returncode']) is int and r['returncode']==0,'wrong release identity')
 require(read(packet/'submission_adoption_receipt.json')['local_ledgers_fsynced_before_receipt'] is True,'durable local adoption required')
 require(ordinary(RUN/'slurm/submitted_jobids_aces.txt').decode().splitlines().count(JOB)==1 and [x for x in ordinary(RUN/'slurm/submissions_aces.tsv').decode().splitlines() if x.split('\t',1)[0]==JOB]==['\t'.join(s['submission_row'])],'exact local ACES ownership required')
 require(field(r['owned_job_detail_before_release'],'Requeue')=='1' and field(r['owned_job_detail_before_release'],'Restarts')=='0','saved need for exact update changed')
 require(not (ACTION/'local_intent.json').exists() and not (ACTION/'local_intent.json').is_symlink(),'prior local action exists; no retry')
 paths=[Path(__file__).absolute(),packet/'submission_receipt.json',packet/'release_receipt.json',packet/'site_validation_plan.json',packet/'manifest.yaml',packet/'job.sbatch']
 if publication is not None:
  require(re.fullmatch(r'[0-9a-f]{40}',publication) is not None,'full published action commit required');repo=RUN.parents[2]
  subprocess.run(['git','-C',str(repo),'merge-base','--is-ancestor',publication,plan['publication_ref']],check=True)
  for path in paths:require(ordinary(path)==subprocess.check_output(['git','-C',str(repo),'cat-file','blob',publication+':'+str(path.relative_to(repo))]),'action input not published: '+str(path))
 return {'schema_version':1,'kind':'owned_aces_requeue_disable_intent','job_id':JOB,'command':COMMAND,'run':plan['native_run'],'packet':plan['native_packet'],'native_action_directory':str(Path(plan['native_run'])/'remote_setup'/ACTION.name),'action_name':ACTION.name,'submission_row':s['submission_row'],'native_packet_hashes':{name:sha(ordinary(packet/name)) for name in ['site_validation_plan.json','manifest.yaml','job.sbatch']},'private_python':plan['private_python'],'publication_commit':publication,'source_sha256':sha(ordinary(Path(__file__).absolute())),'at_utc':datetime.now(timezone.utc).isoformat(),'automatic_retry_permitted':False,**BOUNDARY}

def accept_response(raw,value):
 response=json.loads(raw);require(set(response)=={'files'} and set(response['files'])==NAMES,'wrong returned evidence roster')
 files={name:base64.b64decode(raw,validate=True) for name,raw in response['files'].items()};require(all(len(v)<=4*1024*1024 for v in files.values()),'returned metadata exceeds bound')
 require(json.loads(files['intent.json'])==value,'native intent differs');r=json.loads(files['receipt.json'])
 require(type(r.get('schema_version')) is int and r['schema_version']==1 and r['kind']=='owned_aces_requeue_disabled_receipt' and r['cluster']=='aces' and r['job_id']==JOB and r['command']==COMMAND and r['requeue_after']=='0' and r['restart_count_before']==r['restart_count_after']=='0','wrong native update identity/outcome')
 require(r['publication_commit']==value['publication_commit'] and r['source_sha256']==value['source_sha256'] and r['submission_receipt_sha256']==SUBMISSION_SHA and r['release_receipt_sha256']==RELEASE_SHA and r['automatic_retry_permitted'] is False and all(type(r.get(k)) is type(v) and r[k]==v for k,v in BOUNDARY.items()),'changed native proof/credit')
 packet=Path(value['packet']);before=validate_job(files['pre_job.stdout.raw'].decode(),packet);validate_job(files['post_job.stdout.raw'].decode(),packet,True)
 require(type(r['update_performed']) is bool and r['update_performed']==(before=='1') and r['requeue_before']==before,'wrong update occurrence')
 for label in ['pre_job','post_job']:
  q=json.loads(files[label+'.return.json']);require(q['command']==['scontrol','show','job','-o',JOB] and type(q['returncode']) is int and q['returncode']==0 and q['automatic_retry_permitted'] is False and files[label+'.stderr.raw']==b'','wrong own query return')
 update=json.loads(files['update.return.json']);require(update['command']==COMMAND and update['automatic_retry_permitted'] is False and files['update.stderr.raw']==b'','wrong update return')
 require((type(update['returncode']) is int and update['returncode']==0) if r['update_performed'] else (update['returncode'] is None and update['performed'] is False),'wrong update execution proof')
 return files

def execute_once(value):
 require(value['publication_commit'] is not None,'publication required before action');put(ACTION/'local_intent.json',encoded(value))
 command=['ssh','-T','-o','ControlMaster=no','-o','ProxyCommand=false','-o','BatchMode=yes','-o','NumberOfPasswordPrompts=0','-o','ConnectionAttempts=1','-o','ConnectTimeout=10','aces','exec '+shlex.join([value['private_python'],'-I','-B','-c',remote_program()])]
 with (ACTION/'transport.stdout.raw').open('xb') as out,(ACTION/'transport.stderr.raw').open('xb') as err:
  try:result=subprocess.run(command,input=encoded(value),stdout=out,stderr=err,timeout=120,check=False)
  except BaseException as error:
   out.flush();os.fsync(out.fileno());err.flush();os.fsync(err.fileno());put(ACTION/'transport.return.json',encoded({'returncode':None,'error_type':type(error).__name__,'error':str(error),'automatic_retry_permitted':False}));raise
  out.flush();os.fsync(out.fileno());err.flush();os.fsync(err.fileno())
 put(ACTION/'transport.return.json',encoded({'returncode':result.returncode,'automatic_retry_permitted':False}));require(result.returncode==0,'transport failed; never retry')
 files=accept_response(ordinary(ACTION/'transport.stdout.raw'),value);destination=ACTION/'received';destination.mkdir()
 for name in sorted(files):
  if name!='receipt.json':put(destination/name,files[name])
 put(destination/'receipt.json',files['receipt.json']);return json.loads(files['receipt.json'])

def main():
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--publication-commit');parser.add_argument('--execute',action='store_true');args=parser.parse_args()
 require(not args.execute or args.publication_commit is not None,'publication mandatory for one update');value=local_payload(args.publication_commit)
 print(json.dumps(execute_once(value) if args.execute else {'validated':True,'job_id':JOB,'action':COMMAND,'needed_from_saved_release':True,'ssh_calls':0,'updates':0}))
if __name__=='__main__':main()
