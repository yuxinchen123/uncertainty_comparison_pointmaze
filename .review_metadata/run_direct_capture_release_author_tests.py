"""Run only the release acknowledgment metadata fixtures in an existing owned CPU allocation."""
import hashlib,json,os,socket,subprocess,sys,time
from pathlib import Path
w=Path(__file__).resolve().parents[1];r=w/'.review_metadata';s=w/'10_jax_exploration_platform/scripts/atari_pixel_campaign'
paths=[s/'direct_capture_release.py',s/'tests/test_direct_capture_release.py']
freeze={str(p.relative_to(w)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
assert freeze=={'10_jax_exploration_platform/scripts/atari_pixel_campaign/direct_capture_release.py':'8d7623b1b5cf8bf7a8b65321a33ec1cf551bf99f63d445145a638a9011982e42','10_jax_exploration_platform/scripts/atari_pixel_campaign/tests/test_direct_capture_release.py':'b18bf8c96ff4c60a072b3fa63958ff87adfbcf530c82f815aa4e211fb90cb925'}
command=['/usr/bin/python3','-B','-m','unittest','test_direct_capture_release.DirectCaptureReleaseTests'];started=time.time()
result=subprocess.run(command,cwd=s/'tests',capture_output=True,check=False)
label='direct_capture_release_author_'+os.environ['SLURM_JOB_ID']+'_'+os.environ['SLURM_STEP_ID']
for suffix,raw in (('_stdout.raw',result.stdout),('_stderr.raw',result.stderr)):(r/(label+suffix)).write_bytes(raw)
assert all(hashlib.sha256((w/p).read_bytes()).hexdigest()==v for p,v in freeze.items())
record={'kind':'direct_capture_release_author_metadata_tests','command':command,'cwd':str(s/'tests'),'job_id':os.environ['SLURM_JOB_ID'],'step_id':os.environ['SLURM_STEP_ID'],'hostname':socket.gethostname(),'pid':os.getpid(),'started_at_seconds':started,'finished_at_seconds':time.time(),'source_files':freeze,'source_unchanged_before_and_after':True,'returncode':result.returncode,'stdout_sha256':hashlib.sha256(result.stdout).hexdigest(),'stderr_sha256':hashlib.sha256(result.stderr).hexdigest(),'actual_tool_execution':False,'actual_gpu_execution':False,'standing_lock_opened':False,'scientific_credit':0,'admission_granted':False}
out=r/(label+'.json');out.write_text(json.dumps(record,indent=2)+'\n');print(json.dumps({'receipt':str(out),'returncode':result.returncode,'output':result.stderr.decode()}));sys.exit(result.returncode)
