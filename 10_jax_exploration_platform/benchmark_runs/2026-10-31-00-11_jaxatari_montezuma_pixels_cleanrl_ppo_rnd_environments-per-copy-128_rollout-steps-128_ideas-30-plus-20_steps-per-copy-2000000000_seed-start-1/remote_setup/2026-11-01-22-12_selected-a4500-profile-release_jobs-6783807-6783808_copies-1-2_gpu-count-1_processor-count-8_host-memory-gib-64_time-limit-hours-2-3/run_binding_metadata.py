"""Save actual CPU binding outputs and failures before any parsing."""
import datetime,hashlib,json,os,subprocess,sys
from pathlib import Path
P=Path(__file__).resolve().parent
assert os.environ.get('SLURM_JOB_ID')=='6783003' and os.environ.get('SLURM_STEP_ID','').isdigit()
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
command=['/usr/bin/python3','-I','-S','-B',str(P/'bind_actual_release.py')]
prefix='actual_binding_6783003_'+os.environ['SLURM_STEP_ID']
started=datetime.datetime.now(datetime.timezone.utc).isoformat()
try:
 completed=subprocess.run(command,capture_output=True,timeout=120)
 stdout,stderr=completed.stdout,completed.stderr;returned={'completed':True,'returncode':completed.returncode}
except BaseException as error:
 stdout,stderr=getattr(error,'stdout',None) or b'',getattr(error,'stderr',None) or b''
 returned={'completed':False,'returncode':None,'error_type':type(error).__name__,'error':str(error)}
def put(name,raw):
 path=P/name
 with path.open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
 return {'path':str(path),'sha256':sha(path)}
returned.update(command=command,started_at_utc=started,finished_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
 job_id='6783003',step_id=os.environ['SLURM_STEP_ID'],pid=os.getpid(),
 birth=Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19],cgroup=Path('/proc/self/cgroup').read_text(),
 stdout=put(prefix+'.stdout.raw',stdout),stderr=put(prefix+'.stderr.raw',stderr))
reference=put(prefix+'_return.json',(json.dumps(returned,sort_keys=True,indent=2)+'\n').encode())
assert returned['completed'] and returned['returncode']==0,reference
value=json.loads(stdout);assert value['passed'] is True
print(json.dumps({'passed':True,'actual_step':os.environ['SLURM_STEP_ID'],'return_record':reference,'binding':value}))
