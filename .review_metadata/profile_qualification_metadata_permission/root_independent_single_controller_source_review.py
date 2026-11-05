"""Independently check private-file permission actions and refuse the unprotected actual campaign."""
from pathlib import Path
import subprocess,json,os,hashlib,time,datetime,stat
D=Path(__file__).resolve().parent
R=Path('/p/rlprojects/RND_worktrees/merge-atari-cleanrl/10_jax_exploration_platform/benchmark_runs/2026-10-31-00-11_jaxatari_montezuma_pixels_cleanrl_ppo_rnd_environments-per-copy-128_rollout-steps-128_ideas-30-plus-20_steps-per-copy-2000000000_seed-start-1')
P=R/'remote_setup/2026-11-05-12-57_profile-source-qualification_jaguar03-rtx-a4500_copies-1_environments-128_rollout-steps-128_epochs-4_minibatches-4_warmup-rollouts-50_reference-updates-1_seed-start-1_hours-3'
# Test only private chmods; actual campaign files are observed without mutation.
assert os.environ.get('SLURM_JOB_ID')=='6825983' and os.environ.get('SLURM_STEP_ID','').isdecimal()
freeze=D/'author_source_freeze_2_single_controller.json';assert hashlib.sha256(freeze.read_bytes()).hexdigest()=='392948151d40756fe3bc34a96d2fc285a99dd98ced2f8f690b817286bab7eba1'
files=json.loads(freeze.read_text())['files']
for row in files:assert hashlib.sha256(Path(row['path']).read_bytes()).hexdigest()==row['sha256']
locks=[R/'queue/metadata.lock',R/'processes.lock']
def observed(path):
    """Preserve the exact real owned metadata identity and bytes through all checks."""
    s=path.lstat();return [s.st_dev,s.st_ino,s.st_mode,s.st_uid,s.st_gid,s.st_size,s.st_mtime_ns,s.st_ctime_ns,hashlib.sha256(path.read_bytes()).hexdigest()]
before=[observed(p) for p in locks];started=time.monotonic()
# The nine fixtures exercise actual private-file chmod/fsync, refusing pending or changed evidence.
q=subprocess.run(['/usr/bin/python3','-I','-S','-B',str(D/'test_metadata_permission_once.py')],capture_output=True,timeout=280)
(D/'root_independent_single_controller_private_tests_stdout.raw').write_bytes(q.stdout);(D/'root_independent_single_controller_private_tests_stderr.raw').write_bytes(q.stderr)
assert q.returncode==0 and b'Ran 9 tests' in q.stderr and b'OK' in q.stderr
# This real-input no-action check must refuse the actual group-writable locks before submission.
cmd=['/usr/bin/python3','-I','-S','-B',str(D/'metadata_permission_once.py'),'--run-dir',str(R),'--packet',str(P),'--controller-unit',str(R/'data'/(P.name+'_controller'))]
a=subprocess.run(cmd,capture_output=True,timeout=15)
(D/'root_independent_single_controller_actual_preflight_stdout.raw').write_bytes(a.stdout);(D/'root_independent_single_controller_actual_preflight_stderr.raw').write_bytes(a.stderr)
assert a.returncode==1 and b'protect both exact owned metadata files to0640' in a.stderr
assert [observed(p) for p in locks]==before
for row in files:assert hashlib.sha256(Path(row['path']).read_bytes()).hexdigest()==row['sha256']
# Root read the full source and confirms the capability/publication/same-file/no-timeout constraints.
r={'schema_version':2,'kind':'independent_metadata_permission_utility_source_review','approved':True,'passed':True,'independent_of_author':True,'reviewer':'/root','source_sha256':files[0]['sha256'],'test_source_sha256':files[1]['sha256'],'source_freeze':{'path':str(freeze),'sha256':hashlib.sha256(freeze.read_bytes()).hexdigest()},'private_file_fixtures':9,'actual_campaign_read_only_preflight_refused':True,'actual_campaign_files_unchanged':True,'root_reviewed_full_source':True,'actual_cpu_job':os.environ['SLURM_JOB_ID'],'actual_cpu_step':os.environ['SLURM_STEP_ID'],'seconds':time.monotonic()-started,'actual_campaign_permission_changes':0,'standing_GPU_lock_accesses':0,'scientific_credit':0,'admission_granted':False,'scope_limit':'No-submit abort marker forbids reuse through this required root preflight/workflow; it is not a new runtime check in the unchanged scheduler utility.','at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
with (D/'root_independent_single_controller_source_review.json').open('x') as f:json.dump(r,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
print(json.dumps(r))
