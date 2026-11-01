"""Verify saved release/startup evidence using only owned CPU metadata reads."""
import ctypes
from datetime import datetime, timezone
import hashlib
import importlib.abc
import importlib.util
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
import traceback

AUDIT=Path(__file__).absolute().parent
PACKET=Path('/p/rlprojects/RND_worktrees/merge-atari-cleanrl/10_jax_exploration_platform/benchmark_runs/2026-10-31-00-11_jaxatari_montezuma_pixels_cleanrl_ppo_rnd_environments-per-copy-128_rollout-steps-128_ideas-30-plus-20_steps-per-copy-2000000000_seed-start-1/remote_setup/2026-11-01-13-50_original-qualification_explicit-pytest-root_jaguar03_gpu-count-1_processor-count-8_host-memory-gib-64_time-limit-hours-3_copies-1_environments-128_rollout-steps-128_epochs-4_minibatches-4_warmup-rollouts-50_reference-updates-1_seed-1')
RUN=PACKET.parent.parent
UNIT=RUN/'data'/PACKET.name
REPOSITORY=RUN.parents[2]
JOB='6782778'
PUBLICATION='550425fd8ec68f11e60fb93f873bd77ee7b06f0b'
CHECKS=[]
BLOCKED=[]
assert os.environ.get('SLURM_JOB_ID')=='6781231' and os.environ.get('SLURM_STEP_ID','').isdecimal()

def save(name,raw):
 if not isinstance(raw,bytes):raw=(json.dumps(raw,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
 with (AUDIT/name).open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
 return {'path':str(AUDIT/name),'sha256':hashlib.sha256(raw).hexdigest()}

def checked(value,message):
 assert value,message
 CHECKS.append(message)

def image(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def doc(path):return json.loads(Path(path).read_bytes())
def ref(path):return {'path':str(path),'sha256':image(path)}

class NoNumerical(importlib.abc.MetaPathFinder):
 def find_spec(self,name,path=None,target=None):
  if name.split('.')[0] in {'jax','jaxlib','numpy','torch','flax','optax','jaxtari','scipy','cupy'}:
   BLOCKED.append(name);raise RuntimeError('numerical import forbidden')
sys.meta_path.insert(0,NoNumerical())
ctypes.CDLL=lambda *a,**k:(_ for _ in ()).throw(RuntimeError('shared-library load forbidden'))
socket.socket=lambda *a,**k:(_ for _ in ()).throw(RuntimeError('network forbidden'))
real_run=subprocess.run

def run(command,**options):
 allowed=command[:2]==['git','-C'] and command[2]==str(REPOSITORY) and command[3] in {'cat-file','show','rev-parse','status','diff','ls-tree','merge-base','symbolic-ref'}
 allowed=allowed or command==['squeue','-h','-j','6782778,6781231','-o','%i|%u|%T|%N|%b|%R']
 allowed=allowed or command==['sacct','-n','-P','-j',JOB,'--format=JobIDRaw,User,State,ExitCode,NodeList,AllocCPUS,ReqMem,ReqTRES,AllocTRES,Elapsed']
 assert allowed,'unowned or mutating command forbidden'
 return real_run(command,**options)
subprocess.run=run

try:
 frozen=doc(PACKET/'release_frozen_inputs_6782778.json')
 initial=doc(RUN/'remote_setup/2026-11-01-13-56_explicit-pytest-root-qualification-packet-publication.json')
 protected={path:entry['sha256'] for path,entry in frozen['files'].items()}
 old=next(p for p in (RUN/'remote_setup').iterdir() if p.name.startswith('2026-11-01-12-36_full-original-qualification_'))
 for root in (old,RUN/'data'/old.name):
  for path in root.rglob('*'):
   if path.is_file() and not path.is_symlink():protected[str(path)]=image(path)
 before={path:image(path) for path in protected}
 checked(before==protected,'70 exact release prerequisites and all old failed bytes match their preserved inputs')
 checked(all(image(row['absolute_path'])==row['sha256'] for row in initial['files']),'all50 immutable preparation paths unchanged')
 ledger=(RUN/'slurm/submitted_jobids_uva.txt').read_text().splitlines()
 global_ids=(RUN/'slurm/submitted_jobids.txt').read_text().splitlines()
 checked(all(ledger.count(j)==1 for j in (JOB,'6781231')) and global_ids.count(JOB)==1,'actual target and monitor IDs occur exactly once in owned ledgers')
 plan=doc(UNIT/'measurement_plan.json');pre=doc(UNIT/'configured_isolation_pre_release_inputs.json');final=doc(UNIT/'owned_allocation_inputs.json')
 checked(image(PACKET/'release_receipt.json')=='093f9e720988d1ff585a253aa67552d2ac267ece0375e7d7a73c108c78e199d0','exact successful release receipt hash')
 checked(image(UNIT/'owned_allocation_inputs.json')=='9e12c41186974ac21251ecb11e848792477cee3d4296a2cd13147f60cef37bd9','exact final owned inputs hash')
 receipt=doc(PACKET/'release_receipt.json');intent=doc(PACKET/'release_intent.json')
 checked(receipt['job_id']==JOB and receipt['command']==['scontrol','release',JOB] and type(receipt['returncode']) is int and receipt['returncode']==0 and receipt['publication_commit']==PUBLICATION,'one successful exact fixed-ID release record')
 checked(intent['job_id']==JOB and intent['publication_commit']==PUBLICATION and intent['automatic_retry_permitted'] is False,'exclusive intent binds the actual publication and forbids retry')
 for label,command in [('release_preflight',['scontrol','show','job','-o',JOB]),('release',['scontrol','release',JOB])]:
  result=doc(PACKET/(label+'_return.json'))
  checked(result['command']==command and type(result['returncode']) is int and result['returncode']==0,'saved '+label+' command completed0')
  checked((PACKET/(label+'_stderr.raw')).read_bytes()==b'','saved '+label+' stderr is empty')
 checked((PACKET/'release_stdout.raw').read_bytes()==receipt['stdout'].encode() and (PACKET/'release_stderr.raw').read_bytes()==receipt['stderr'].encode(),'receipt equals exact saved raw release bytes')
 checked(final['kind']=='uva_configured_isolation_owned_inputs' and final['release']==ref(PACKET/'release_receipt.json') and {k:v for k,v in final.items() if k not in {'release','kind'}}=={k:v for k,v in pre.items() if k!='kind'},'final envelope adds only actual release reference and final kind')
 row=[JOB,plan['kind'],plan['helpers']['source_commit'],str(PACKET/'job.sbatch')]
 checked((RUN/'slurm/submissions.tsv').read_text().splitlines().count('\t'.join(row))==1 and pre['submission_row']==row,'actual canonical submission row matches source and exact script')
 source=PACKET/'release_qualification_6782778.py'
 checked(image(source)=='7240b11f232a329f3cd09471f7caedcf831d6492c24a0ee2598981265b4bdf9e' and receipt['release_source_sha256']==image(source),'executed utility is exact independently reviewed source')
 for path in list(frozen['files'])+[str(source),str(PACKET/'release_frozen_inputs_6782778.json'),str(PACKET/'release_implementation_review_6782778.json')]:
  output=run(['git','-C',str(REPOSITORY),'cat-file','blob',PUBLICATION+':'+str(Path(path).relative_to(REPOSITORY))],capture_output=True,check=True)
  checked(output.stdout==Path(path).read_bytes(),'published exact prerequisite '+str(Path(path).relative_to(REPOSITORY)))
 sys.path.insert(0,str(UNIT/'code'))
 spec=importlib.util.spec_from_file_location('audit_actual_configured_core',UNIT/'code/uva_slurm_configured_isolation.py');core=importlib.util.module_from_spec(spec);spec.loader.exec_module(core)
 core.validate_plan(plan)
 checked(core.owned_inputs(plan)==final,'actual staged core rebuilds the successful release chain')
 wait=doc(UNIT/'owned_inputs_wait_completed.json')
 checked(wait['job_id']==JOB and wait['helper_commit']==plan['helpers']['source_commit'] and wait['final_inputs_sha256']==image(UNIT/'owned_allocation_inputs.json') and wait['actual_release_receipt']==ref(PACKET/'release_receipt.json'),'actual job startup consumed this exact successful release')
 checked(wait['maximum_wait_seconds']==180 and 0<=wait['waited_seconds']<=180 and wait['numerical_imports']==wait['gpu_operations']==0 and wait['interpreter_flags']=={'isolated':1,'no_site':1,'dont_write_bytecode':True} and not wait['observed_actions']['numerical_import_attempts'] and not wait['observed_actions']['forbidden_operation_attempts'],'actual startup barrier is bounded isolated stdlib with no NUM/device attempt')
 checked(not any(n.split('.')[0] in {'jax','jaxlib','numpy','torch','flax','optax','jaxtari','cupy','scipy'} for n in wait['imported_module_names']),'actual startup import roster contains no numerical package')
 metadata=doc(UNIT/'configured_isolation_metadata.json');core.validate_metadata(plan,metadata)
 checked(metadata['environment']['SLURM_JOB_ID']==JOB and metadata['environment']['SLURM_STEP_ID'].isdecimal() and metadata['process']['hostname']=='jaguar03','actual metadata validates own numeric target step before learner')
 snapshots={}
 for path in sorted(UNIT.iterdir()):
  if path.is_file() and path.suffix in {'.json','.yaml'} and path.name not in {'measurement_plan.json','manifest.yaml'}:
   snapshots[path.name]=save('startup_'+path.name,path.read_bytes())
 for suffix in ['out','err']:
  path=PACKET/('slurm-'+JOB+'.'+suffix)
  if path.is_file():snapshots[path.name]=save(path.name+'.snapshot',path.read_bytes())
 binding_file=UNIT/'resolved_device_binding.json';binding_status='not yet present at audit'
 if binding_file.exists():
  binding=doc(binding_file);core.validate_binding(plan,metadata,binding,directory=UNIT);binding_status='actual same-process physical binding validated from saved evidence'
  checked(True,binding_status)
 query_evidence={}
 for name,command in [('queue',['squeue','-h','-j','6782778,6781231','-o','%i|%u|%T|%N|%b|%R']),('accounting',['sacct','-n','-P','-j',JOB,'--format=JobIDRaw,User,State,ExitCode,NodeList,AllocCPUS,ReqMem,ReqTRES,AllocTRES,Elapsed'])]:
  result=run(command,capture_output=True,timeout=30,check=False)
  query_evidence[name]={'command':command,'returncode':result.returncode,'stdout':save(name+'.stdout.raw',result.stdout),'stderr':save(name+'.stderr.raw',result.stderr)}
  checked(result.returncode==0,'exact owned '+name+' query returned0')
 after={path:image(path) for path in protected};checked(before==after,'all protected release and old failed bytes stayed unchanged during audit')
 checked(not BLOCKED,'no numerical import was attempted')
 result={'schema_version':1,'kind':'actual_qualification_release_and_startup_independent_review','passed':True,'at_utc':datetime.now(timezone.utc).isoformat(),'audit_job_id':'6781231','audit_step_id':os.environ['SLURM_STEP_ID'],'audit_pid':os.getpid(),'audit_birth':Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19],'audit_cgroup':Path('/proc/self/cgroup').read_text(),'job_id':JOB,'publication_commit':PUBLICATION,'checks':CHECKS,'check_count':len(CHECKS),'release_receipt':ref(PACKET/'release_receipt.json'),'final_inputs':ref(UNIT/'owned_allocation_inputs.json'),'actual_staged_core_reader_passed':True,'actual_startup_barrier_passed':True,'metadata_process':metadata['process'],'binding_status':binding_status,'startup_snapshots':snapshots,'owned_queries':query_evidence,'protected_before':before,'protected_after':after,'qualification_complete':False,'partial_test_fixture_work':'not quantified; no whole-job transition total claimed','scientific_credit':0,'case_credit':0,'idea_credit':0,'site_qualification_credit':0,'admission_granted':False,'worker_admitted':False,'new_jobs':0,'release_calls_by_audit':0,'cancellation_calls_by_audit':0,'network_calls':0,'numerical_imports':0}
 proof=save('review.json',result);print(json.dumps({'passed':True,'receipt':proof,'check_count':len(CHECKS),'actual_audit_step':os.environ['SLURM_STEP_ID'],'binding_status':binding_status}))
except BaseException:
 save('failure.json',{'traceback':traceback.format_exc(),'checks_before_failure':CHECKS,'audit_job':'6781231','audit_step':os.environ['SLURM_STEP_ID'],'release_calls':0,'worker_admission':False});raise
