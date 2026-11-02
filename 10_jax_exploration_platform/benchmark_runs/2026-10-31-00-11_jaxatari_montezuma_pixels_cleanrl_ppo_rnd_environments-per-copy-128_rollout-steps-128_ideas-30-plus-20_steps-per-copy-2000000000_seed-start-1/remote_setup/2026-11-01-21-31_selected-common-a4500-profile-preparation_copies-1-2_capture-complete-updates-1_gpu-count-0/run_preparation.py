"""Retain the actual preparation command, raw outputs and status before parsing."""
import hashlib,json,os,subprocess,sys,datetime
from pathlib import Path
P=Path(__file__).resolve().parent
assert os.environ.get('SLURM_JOB_ID')=='6783003' and os.environ.get('SLURM_STEP_ID','').isdigit()
paths=[P/'prepare_profiles.py',P/'guarded_prepare_child.py',P/'run_preparation.py']
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
before={str(p):sha(p) for p in paths}
command=['/usr/bin/python3','-I','-S','-B',str(P/'prepare_profiles.py')]
started=datetime.datetime.now(datetime.timezone.utc).isoformat()
try:
 run=subprocess.run(command,capture_output=True,timeout=1400)
 out,err=run.stdout,run.stderr;returned={'completed':True,'returncode':run.returncode}
except BaseException as error:
 out,err=getattr(error,'stdout',None) or b'',getattr(error,'stderr',None) or b''
 returned={'completed':False,'returncode':None,'error_type':type(error).__name__,'error':str(error)}
def put(name,raw):
 p=P/name
 with p.open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
 return {'path':str(p),'sha256':sha(p)}
stdout=put('actual_preparation.stdout.raw',out);stderr=put('actual_preparation.stderr.raw',err)
returned.update(command=command,started_at_utc=started,finished_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
 job_id='6783003',step_id=os.environ['SLURM_STEP_ID'],source_before=before,source_after={str(p):sha(p) for p in paths},stdout=stdout,stderr=stderr)
receipt=put('actual_preparation_return.json',(json.dumps(returned,sort_keys=True,indent=2)+'\n').encode())
assert returned['completed'] and returned['returncode']==0 and before==returned['source_after'],receipt
value=json.loads(out);assert value['passed'] is True
print(json.dumps({'passed':True,'actual_step':os.environ['SLURM_STEP_ID'],'return_record':receipt,'records':value['records']}))
