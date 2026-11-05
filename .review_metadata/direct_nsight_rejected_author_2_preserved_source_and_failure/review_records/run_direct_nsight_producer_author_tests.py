"""Run only bounded CPU source/process fixtures in the existing owned monitor allocation."""
import hashlib,json,os,socket,subprocess,sys,time
from pathlib import Path
w=Path(__file__).resolve().parents[1];r=w/'.review_metadata';frozen=r/'direct_nsight_producer_source_freeze_author_2.json';f=json.loads(frozen.read_text())
assert all(hashlib.sha256((w/p).read_bytes()).hexdigest()==h for p,h in f['files'].items())
command=['/usr/bin/python3','-B','-m','unittest','test_direct_profile_process.ActualTemporaryDescriptorTests','test_direct_profile_process.DirectSavedProcessTests','test_direct_profile_process.DirectDiagnosticSourceTests','test_direct_profile_prerequisites.DirectProfileMetadataTests','test_direct_profile_prerequisites.DirectProfileBackendTests','test_profile_selected','test_profile_owned_processes','test_profile_common_confirmation_helpers','test_direct_common_confirmation','test_direct_measurement']
started=time.time();c=subprocess.run(command,cwd=w/'10_jax_exploration_platform/scripts/atari_pixel_campaign/tests',capture_output=True,check=False)
label='direct_nsight_producer_author_'+os.environ['SLURM_JOB_ID']+'_'+os.environ['SLURM_STEP_ID']
for suffix,raw in (('_stdout.raw',c.stdout),('_stderr.raw',c.stderr)):(r/(label+suffix)).write_bytes(raw)
assert all(hashlib.sha256((w/p).read_bytes()).hexdigest()==h for p,h in f['files'].items())
record={'command':command,'job_id':os.environ['SLURM_JOB_ID'],'step_id':os.environ['SLURM_STEP_ID'],'hostname':socket.gethostname(),'pid':os.getpid(),'started_at_seconds':started,'finished_at_seconds':time.time(),'returncode':c.returncode,'source_freeze':{'path':str(frozen),'sha256':hashlib.sha256(frozen.read_bytes()).hexdigest()},'source_unchanged_before_and_after':True,'stdout_sha256':hashlib.sha256(c.stdout).hexdigest(),'stderr_sha256':hashlib.sha256(c.stderr).hexdigest(),'actual_nsight_execution':False,'actual_gpu_execution':False,'environment_transitions':0,'scientific_credit':0,'admission_granted':False}
out=r/(label+'.json');out.write_text(json.dumps(record,indent=2)+'\n');print(json.dumps({'receipt':str(out),'returncode':c.returncode,'stderr_tail':c.stderr.decode()[-3500:]}));sys.exit(c.returncode)
