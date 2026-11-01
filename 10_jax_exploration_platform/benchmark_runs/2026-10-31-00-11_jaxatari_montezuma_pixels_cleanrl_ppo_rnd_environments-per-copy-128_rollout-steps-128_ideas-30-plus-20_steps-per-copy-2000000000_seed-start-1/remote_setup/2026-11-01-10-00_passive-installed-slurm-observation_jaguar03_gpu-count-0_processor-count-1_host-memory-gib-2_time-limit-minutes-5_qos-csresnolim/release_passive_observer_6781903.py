"""Release only the saved passive metadata observation, without running its collector here."""
import argparse,hashlib,json,os,re,socket,subprocess,sys
from datetime import datetime,timezone
from pathlib import Path
PACKET=Path(__file__).absolute().parent
RUN=PACKET.parent.parent
JOB='6781903'
SOURCE='fa7b0a291dde4e8be0acc5ab1b7ddf71ed1e2947'
SUBMISSION_SHA='43251c8f2d5341ab8c97b55b97c3b9f587370106e46b5cf1c50876a4d649d687'
INITIAL_PUBLICATION='efe539dd151ffbb6435dd27256893517823ad275'
FROZEN_INPUTS_SHA='d053777060efc639e75d5b2cdf3d66e43decbd42cf45a7e7f15b49e425f09cc6'
UNIT=RUN/'data'/PACKET.name
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
 expected={'JobId':JOB,'UserId':'sl5nw(2618919)','Account':'sl5nw_base','Partition':'gpu','QOS':'csresnolim','JobState':'PENDING','Reason':'JobHeldUser','ReqNodeList':'jaguar03','Reservation':'sl5nw_168','Requeue':'0','Restarts':'0','NumCPUs':'1','NumTasks':'1','CPUs/Task':'1','MinMemoryNode':'2G','TimeLimit':'00:05:00','Command':str(PACKET/'job.sbatch')}
 require(all(field(raw,k)==v for k,v in expected.items()) and field(raw,'NumNodes') in {'1','1-1'},'actual held passive observer identity or request changed')
 require(re.search(r'(?:^|\s)TresPerNode=',raw) is None,'--gres=none must retain the actual absent TresPerNode field')
 pairs=[value.split('=',1) for value in field(raw,'ReqTRES').split(',')]
 require(all(len(p)==2 for p in pairs) and len({p[0] for p in pairs})==len(pairs),'invalid or repeated requested resources')
 resources=dict(pairs)
 require(resources.get('cpu')=='1' and resources.get('mem')=='2G' and resources.get('node')=='1' and not any(k.startswith('gres/') for k in resources),'actual passive request must contain no GRES')

def checked(publication=None):
 require(socket.gethostname().split('.')[0]=='portal05' and os.getuid()==2618919 and not NUM.intersection(sys.modules),'wrong local host/owner or numerical import')
 repo=RUN.parents[2];snapshot_raw=contents(PACKET/'release_frozen_inputs_6781903.json')
 require(sha(snapshot_raw)==FROZEN_INPUTS_SHA,'exact immutable release input snapshot changed')
 frozen=json.loads(snapshot_raw)
 require(frozen['job_id']==JOB and frozen['source_commit']==SOURCE and frozen['original_unit_artifact_count']==11 and frozen['packet']==str(PACKET) and frozen['unit']==str(UNIT),'wrong immutable passive input scope')
 for path,expected in frozen['files'].items():
  value=contents(Path(path));require(sha(value)==expected['sha256'] and len(value)==expected['bytes'],'immutable passive input changed: '+path)
 checks=read(PACKET/'packet_checksums.json')
 require(len(checks['unit_immutable_files'])==11,'all eleven original unit artifacts required')
 for name,hashed in checks['unit_immutable_files'].items():require(sha(contents(UNIT/name))==hashed,'original unit artifact changed')
 for name,hashed in checks['packet_files'].items():require(sha(contents(PACKET/name))==hashed,'original packet artifact changed')
 for name in ['observation_plan.json','passive_plan.json']:
  plan=read(UNIT/name);require(all(plan[k] is None for k in ['job_id','reservation','gpu_uuid']),'frozen plans must retain unresolved identities')
 passive=read(UNIT/'passive_plan.json');source=passive['source'];require(source['source_commit']==SOURCE,'wrong passive source')
 for relative,hashed in source['files'].items():
  original=contents(Path(source['source_root'])/relative);committed=subprocess.check_output(['git','-C',str(repo),'cat-file','blob',SOURCE+':'+relative])
  require(sha(original)==hashed and original==committed and contents(UNIT/'code'/Path(relative).name)==committed,'actual committed/staged passive source changed')
 root_review=read(PACKET/'root_pre_submission_review.json');static=read(PACKET/'static_verification.json');verification=read(PACKET/'binding_verification.json')
 require(root_review['passed'] is True and root_review['source_commit']==SOURCE and static['all_source_hashes_verified'] is True and static['null_static_identities'] is True and verification['observed_job_id']==JOB and verification['source_commit']==SOURCE and verification['all11_original_unit_artifacts_unchanged'] is True and verification['null_frozen_identities_preserved'] is True,'actual preparation/static/binding reviews changed')
 receipt_raw=contents(PACKET/'submission_receipt.json');receipt=json.loads(receipt_raw)
 require(sha(receipt_raw)==SUBMISSION_SHA and type(receipt['schema_version']) is int and receipt['schema_version']==1 and receipt['kind']=='passive_installed_slurm_observation_submission_receipt' and receipt['cluster']=='uva' and receipt['job_id']==JOB and receipt['source_commit']==SOURCE and receipt['publication_commit']==INITIAL_PUBLICATION and receipt['held_requested'] is True and receipt['released'] is False and receipt['ledgers_fsynced_before_receipt'] is True,'exact passive held submission changed')
 require(receipt['command']==root_review['exact_command'] and receipt['plan_sha256']==sha(contents(UNIT/'passive_plan.json')) and receipt['script_sha256']==sha(contents(PACKET/'job.sbatch')),'submitted passive source/plan/script/request changed')
 row='\t'.join([JOB,'passive_installed_slurm_observation',SOURCE,str(PACKET/'job.sbatch')]);require(receipt['submission_row']==row,'exact submitted TSV row changed')
 for name in ['submitted_jobids.txt','submitted_jobids_uva.txt']:require(contents(RUN/'slurm'/name).decode().splitlines().count(JOB)==1,'exact owned passive ID missing or duplicated')
 require([r for r in contents(RUN/'slurm/submissions.tsv').decode().splitlines() if r.split('\t',1)[0]==JOB]==[row],'exact live owned passive TSV row changed')
 ownership=read(UNIT/'held_identifier_binding.json');binding=read(UNIT/'passive_held_identifier_binding.json');ledger=contents(UNIT/'held_owned_job_ledger_snapshot.txt')
 require(ownership['job_id']==binding['job_id']==JOB and ownership['owner']=={'username':'sl5nw','uid':2618919} and ownership['cluster']=='uva' and ownership['reservation']==binding['reservation']=='sl5nw_168' and ownership['plan_sha256']==sha(contents(UNIT/'observation_plan.json')) and ownership['owned_ledger_sha256']==sha(ledger) and ledger.decode().splitlines().count(JOB)==1,'strict ownership binding or frozen ledger changed')
 require(binding['passive_plan_sha256']==sha(contents(UNIT/'passive_plan.json')) and binding['ownership_binding_sha256']==sha(contents(UNIT/'held_identifier_binding.json')) and binding['source']==source,'passive binding no longer matches original strict binding and source')
 for name,hashed in verification['binding_files'].items():require(sha(contents(UNIT/name))==hashed,'verified actual binding changed')
 proof_raw=contents(PACKET/'held_monitor_review.json');proof=json.loads(proof_raw)
 require(proof['kind']=='owned_passive_observer_actual_held_monitor_and_binding_review' and proof['passed'] is True and proof['job_id']==JOB and proof['monitor_job_id']=='6781231' and proof['monitor_source_commit']=='fb3071f3772ed836144f5876dbdda1113a5339d5' and proof['session_cron_verified'] is True and proof['cadence_seconds']==1200 and proof['all_credit_zero'] is True and proof['source_admission_changed'] is False and type(proof['requested_gpu_count']) is int and proof['requested_gpu_count']==0 and proof['tres_per_node_absent'] is True,'actual held full-monitor/zero-GPU binding proof changed')
 require(proof['submission_receipt_sha256']==SUBMISSION_SHA and proof['passive_plan_sha256']==receipt['plan_sha256'] and proof['script_sha256']==receipt['script_sha256'] and proof['binding_verification_sha256']==sha(contents(PACKET/'binding_verification.json')),'held proof prerequisite hashes changed')
 for name,hashed in proof['binding_sha256'].items():require(sha(contents(UNIT/name))==hashed,'held proof binding bytes changed')
 query=proof['query'];require(query['command']==['scontrol','show','job','-o',JOB] and type(query['returncode']) is int and query['returncode']==0 and query['stderr']=='','exact saved own held query changed');validate_held(query['stdout'])
 require(sha(contents(Path(proof['monitor_verification'])))==proof['monitor_verification_sha256'],'actual full-monitor verification bytes changed')
 observed=datetime.fromisoformat(proof['at_utc'].replace('Z','+00:00'));require(observed.tzinfo is not None and observed.utcoffset().total_seconds()==0 and 0<=(datetime.now(timezone.utc)-observed).total_seconds()<1200,'held full-monitor proof is stale or not UTC')
 review=read(PACKET/'release_implementation_review_6781903.json');require(review['job_id']==JOB and review['source_sha256']==sha(contents(Path(__file__).absolute())) and review['release_executed'] is False,'release source review changed')
 for name in ['release_intent.json','release_receipt.json','release_return.json','release_preflight_return.json','release_owned_job_before.raw','release_owned_job_before_stderr.raw','release_stdout.raw','release_stderr.raw']:
  require(not (PACKET/name).exists() and not (PACKET/name).is_symlink(),'prior release attempt exists; never retry')
 if publication is not None:
  require(re.fullmatch(r'[0-9a-f]{40}',publication) is not None,'full published release commit required')
  subprocess.run(['git','-C',str(repo),'merge-base','--is-ancestor',publication,'refs/remotes/origin/merge-atari-cleanrl'],check=True)
  paths=[Path(__file__).absolute(),PACKET/'release_implementation_review_6781903.json',PACKET/'release_frozen_inputs_6781903.json',PACKET/'held_monitor_review.json']+[Path(p) for p in frozen['files'] if Path(p).is_relative_to(repo)]
  for path in paths:require(contents(path)==subprocess.check_output(['git','-C',str(repo),'cat-file','blob',publication+':'+str(path.relative_to(repo))]),'release input was not published: '+str(path))
 return {'schema_version':1,'kind':'owned_passive_observer_release_intent','cluster':'uva','job_id':JOB,'command':COMMAND,'at_utc':datetime.now(timezone.utc).isoformat(),'published_release_commit':publication,'submission_receipt_sha256':SUBMISSION_SHA,'held_monitor_review_sha256':sha(proof_raw),'release_review_sha256':sha(contents(PACKET/'release_implementation_review_6781903.json')),'release_source_sha256':sha(contents(Path(__file__).absolute())),'frozen_inputs_sha256':FROZEN_INPUTS_SHA,'source_commit':SOURCE,'automatic_retry_permitted':False,'scientific_credit':0,'case_credit':0,'idea_credit':0,'site_qualification_credit':0,'admission_granted':False}
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
 receipt={'schema_version':1,'kind':'owned_zero_gpu_passive_observer_release','cluster':'uva','job_id':JOB,'command':COMMAND,'returncode':0,'stdout':stdout.decode(),'stderr':'','at_utc':datetime.now(timezone.utc).isoformat(),'submission_receipt_sha256':SUBMISSION_SHA,'release_review_sha256':value['release_review_sha256'],'held_monitor_review_sha256':value['held_monitor_review_sha256'],'published_release_commit':value['published_release_commit'],'release_source_sha256':value['release_source_sha256'],'frozen_inputs_sha256':FROZEN_INPUTS_SHA,'source_commit':SOURCE,'automatic_retry_permitted':False,'scientific_credit':0,'case_credit':0,'idea_credit':0,'site_qualification_credit':0,'admission_granted':False}
 put(PACKET/'release_receipt.json',encoded(receipt));return receipt

def main():
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--publication-commit');parser.add_argument('--release',action='store_true');args=parser.parse_args()
 require(not args.release or args.publication_commit is not None,'publication commit required before release')
 value=checked(args.publication_commit)
 print(json.dumps(release_once(value) if args.release else {'validated':True,'job_id':JOB,'command':COMMAND,'scheduler_queries':0,'release_calls':0}))
if __name__=='__main__':main()
