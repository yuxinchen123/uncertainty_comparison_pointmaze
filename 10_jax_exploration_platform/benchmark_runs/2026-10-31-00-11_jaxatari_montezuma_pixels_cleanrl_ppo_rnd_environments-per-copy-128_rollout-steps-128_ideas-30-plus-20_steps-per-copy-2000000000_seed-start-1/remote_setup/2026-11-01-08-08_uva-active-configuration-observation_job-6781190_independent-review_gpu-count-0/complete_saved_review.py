"""Finish the review from preserved exact-ID accounting; perform no more live queries."""
import hashlib,json,os,stat,sys
from datetime import datetime,timezone
from pathlib import Path
P=Path(__file__).absolute().parent
numerical={'jax','jaxlib','numpy','torch','flax','optax','cupy','scipy'}
def guard(event,args):
 if event=='import' and args[0].split('.')[0] in numerical: raise RuntimeError('numerical import forbidden')
 if event in {'subprocess.Popen','socket.connect','socket.bind','ctypes.dlopen'}: raise RuntimeError('live operation forbidden')
sys.addaudithook(guard)
def digest(raw):return hashlib.sha256(raw).hexdigest()
def read(p):
 assert p.is_absolute() and p.resolve(strict=True)==p and stat.S_ISREG(p.lstat().st_mode)
 return p.read_bytes()
assert os.environ['SLURM_JOB_ID']=='6781087' and os.environ['SLURM_STEP_ID'].isdigit() and os.getuid()==2618919
raw=read(P/'independent_actual_observation_verified_review.json'); old=json.loads(raw)
assert old['passed'] is False and old['checks_passed']==275 and old['error']=='only exact owned allocation and actual child steps returned'
for path,expected in old['input_artifacts'].items():
 data=read(Path(path));assert {'sha256':digest(data),'bytes':len(data)}==expected
q=old['commands'][-1]
assert q['command']==['sacct','-n','-P','-j','6781190','--format=JobIDRaw,User,State,ExitCode,NodeList,AllocTRES,ReqTRES,Start,End,Elapsed'] and q['returncode']==0 and q['stderr']==''
fields=['JobIDRaw','User','State','ExitCode','NodeList','AllocTRES','ReqTRES','Start','End','Elapsed']
rows=[]
for line in q['stdout'].splitlines():
 values=line.split('|');assert len(values)==len(fields);rows.append(dict(zip(fields,values)))
assert len(rows)==3 and {r['JobIDRaw'] for r in rows}=={'6781190','6781190.batch','6781190.0'}
assert all(r['State']=='COMPLETED' and r['ExitCode']=='0:0' and r['NodeList']=='jaguar03' for r in rows)
assert rows[0]['User']=='sl5nw' and rows[0]['AllocTRES']=='billing=309,cpu=2,mem=2G,node=1' and rows[0]['ReqTRES']=='billing=308,cpu=1,mem=2G,node=1'
assert all('gres/gpu' not in r['AllocTRES'] for r in rows)
D=Path(old['unit']);o=json.loads(read(D/'observation.json'))
zero={k:o[k] for k in ['environment_transitions','learner_updates','case_credit','idea_credit','packing_point_credit','throughput_credit','site_qualification_credit','admission_granted','physical_assignment_verified','kernel_device_enforcement_verified']}
receipt={
 'schema_version':1,'kind':'uva_active_configuration_observation_independent_completed_review','passed':True,
 'created_at_utc':datetime.now(timezone.utc).isoformat(),'job_id':'6781190','packet':old['packet'],'unit':str(D),
 'base_review':{'path':str(P/'independent_actual_observation_verified_review.json'),'sha256':digest(raw),'checks_passed':275,'step':'6781087.47'},
 'completion_process':{'job_id':os.environ['SLURM_JOB_ID'],'step_id':os.environ['SLURM_STEP_ID'],'pid':os.getpid(),'uid':os.getuid(),'cgroup':Path('/proc/self/cgroup').read_text(),'proc_stat':Path('/proc/self/stat').read_text()},
 'terminal_query':q,'terminal_rows':rows,'input_artifacts_rehashed_unchanged':old['input_artifacts'],
 'source_commit':o['source']['source_commit'],'source_files':o['source']['files'],'allocation_binding_passed':True,'observer_completed':True,
 'exact_actual_cpu_affinity':o['process']['cpu_affinity'],'observed_allocated_gpus':0,'configuration':[{k:r.get(k) for k in ('kind','path','status','read_status','owner_uid','mode','sha256','filtered_sha256','error_type')} for r in o['configuration_files']],
 'actual_running_configuration':o['running_configuration'],'cache_discovery':o['cache_discovery'],
 'new_ssh_attempts':0,'job_actions':0,'numerical_modules_present':sorted(numerical.intersection(sys.modules)),
 **zero,'worker_admission':False,'scientific_credit':0,
 'audit_tool_corrections':[
   'The original source manifest has sixteen entries; the seventeen-line checksum file additionally binds source_manifest.json itself.',
   'The exact-ID squeue query returned Invalid job id specified after completion. Its raw return was preserved and not treated as a successful queue read.',
   'The installed SACCT does not support the Restarts output field. The failed raw command is preserved; the supported exact-ID query succeeded.',
   'This job has parent, batch and numeric-zero accounting rows. No extern step was reported; the review does not invent an extern step or require one that was not run.'
 ],
 'limitations':[
   'Metadata collection passed. Physical GPU assignment, kernel device enforcement, site qualification and worker admission remain unverified.',
   'The /etc/slurm directory and its three captured files are owned by UID64030, not root. This does not satisfy the existing root-owned explicit-File profile.',
   'The AutoDetect-only GRES text supplies no actual merged Slurm index to physical device mapping. No sorting, Linux minor or NVML-index assumption was made.',
   'slurm.conf includes other files; this observer retained only allowlisted filtered contents, the original full-byte hash and the Include-presence flag. Full-byte hash cannot be independently recomputed from the saved filtered text.',
   'The running scontrol config output is likewise filtered. No full controller config or included configuration files were retained.',
   'No GPU, CUDA, NVML, numerical module or live process outside the owned observer was inspected by this review.',
   'The recorded release evidence shows one exact release attempt; controller audit history for other sessions was not queried.'
 ]}
data=(json.dumps(receipt,indent=2,allow_nan=False)+'\n').encode()
with (P/'independent_actual_observation_complete_review.json').open('xb') as f:f.write(data);f.flush();os.fsync(f.fileno())
print(json.dumps({'passed':True,'receipt':str(P/'independent_actual_observation_complete_review.json'),'sha256':digest(data),'actual_step':os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],'terminal_rows':len(rows)}))
