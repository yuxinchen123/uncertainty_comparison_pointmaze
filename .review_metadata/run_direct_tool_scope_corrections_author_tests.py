"""Run only the newly reproduced CPU process/lock and exact release-join corrections."""
import hashlib,json,os,socket,subprocess,sys,time
from pathlib import Path
w=Path(__file__).resolve().parents[1];r=w/'.review_metadata';frozen=r/'direct_nsight_producer_source_freeze_author_4_release_and_kernel.json';f=json.loads(frozen.read_text())
assert all(hashlib.sha256((w/p).read_bytes()).hexdigest()==h for p,h in f['files'].items())
command=['/usr/bin/python3','-B','-m','unittest','test_direct_tool_descendants','test_direct_capture_release.DirectCaptureReleaseTests','test_direct_profile_process.DirectSavedProcessTests']
start=time.time();os.environ['DIRECT_CPU_SCOPE_REVIEW_OUTPUT']=str(r/('direct_tool_scope_actual_cpu_'+os.environ['SLURM_JOB_ID']+'_'+os.environ['SLURM_STEP_ID']));c=subprocess.run(command,cwd=w/'10_jax_exploration_platform/scripts/atari_pixel_campaign/tests',capture_output=True,check=False)
label='direct_tool_scope_corrections_author_'+os.environ['SLURM_JOB_ID']+'_'+os.environ['SLURM_STEP_ID']
for suffix,raw in (('_stdout.raw',c.stdout),('_stderr.raw',c.stderr)):(r/(label+suffix)).write_bytes(raw)
assert all(hashlib.sha256((w/p).read_bytes()).hexdigest()==h for p,h in f['files'].items())
v={'command':command,'job_id':os.environ['SLURM_JOB_ID'],'step_id':os.environ['SLURM_STEP_ID'],'hostname':socket.gethostname(),'pid':os.getpid(),'started_at_seconds':start,'finished_at_seconds':time.time(),'returncode':c.returncode,'source_freeze':{'path':str(frozen),'sha256':hashlib.sha256(frozen.read_bytes()).hexdigest()},'source_unchanged_before_and_after':True,'stdout_sha256':hashlib.sha256(c.stdout).hexdigest(),'stderr_sha256':hashlib.sha256(c.stderr).hexdigest(),'actual_tool_or_gpu_execution':False,'native_process_control_tested':True,'standing_gpu_lock_opened':False,'scientific_credit':0,'admission_granted':False}
out=r/(label+'.json');out.write_text(json.dumps(v,indent=2)+'\n');print(json.dumps({'receipt':str(out),'returncode':c.returncode,'stderr_tail':c.stderr.decode()[-5500:]}));sys.exit(c.returncode)
