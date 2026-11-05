"""Run bounded metadata tests in the exact existing owned CPU allocation."""
import hashlib,json,os,socket,subprocess,sys,time
from pathlib import Path
w=Path(__file__).resolve().parents[1]
r=w/'.review_metadata'
frozen=r/'direct_h100_profile_source_freeze_2026-11-04_author_6.json'
f=json.loads(frozen.read_text())
assert all(hashlib.sha256((w/p).read_bytes()).hexdigest()==h for p,h in f['files'].items())
command=['/usr/bin/python3','-B','-m','unittest','test_profile_selected','test_profile_owned_processes','test_profile_common_confirmation_helpers','test_direct_common_confirmation','test_direct_measurement']
started=time.time()
completed=subprocess.run(command,cwd=w/'10_jax_exploration_platform/scripts/atari_pixel_campaign/tests',capture_output=True,check=False)
label='direct_h100_profile_existing_metadata_'+os.environ['SLURM_JOB_ID']+'_'+os.environ['SLURM_STEP_ID']
for suffix,raw in (('_stdout.raw',completed.stdout),('_stderr.raw',completed.stderr)):
    (r/(label+suffix)).write_bytes(raw)
assert all(hashlib.sha256((w/p).read_bytes()).hexdigest()==h for p,h in f['files'].items())
record={'command':command,'job_id':os.environ['SLURM_JOB_ID'],'step_id':os.environ['SLURM_STEP_ID'],'hostname':socket.gethostname(),'pid':os.getpid(),'started_at_seconds':started,'finished_at_seconds':time.time(),'returncode':completed.returncode,'source_freeze':{'path':str(frozen),'sha256':hashlib.sha256(frozen.read_bytes()).hexdigest()},'source_unchanged_before_and_after':True,'stdout_sha256':hashlib.sha256(completed.stdout).hexdigest(),'stderr_sha256':hashlib.sha256(completed.stderr).hexdigest(),'gpu_operations':0,'scientific_credit':0,'admission_granted':False}
out=r/(label+'.json');out.write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps({'receipt':str(out),'returncode':completed.returncode,'stderr_tail':completed.stderr.decode()[-2000:]}))
sys.exit(completed.returncode)
