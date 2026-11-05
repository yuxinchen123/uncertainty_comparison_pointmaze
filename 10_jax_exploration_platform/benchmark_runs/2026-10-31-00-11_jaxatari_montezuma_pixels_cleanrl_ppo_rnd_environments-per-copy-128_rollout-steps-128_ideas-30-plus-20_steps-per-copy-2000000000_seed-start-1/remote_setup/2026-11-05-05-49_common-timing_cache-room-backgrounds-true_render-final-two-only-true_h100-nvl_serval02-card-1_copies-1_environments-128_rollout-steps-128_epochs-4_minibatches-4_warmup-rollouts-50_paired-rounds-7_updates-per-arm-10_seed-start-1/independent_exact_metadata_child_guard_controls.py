import hashlib,importlib.util,json,os,sys,time
from pathlib import Path
R=Path('/p/rlprojects/RND_worktrees/merge-atari-cleanrl/10_jax_exploration_platform/benchmark_runs/2026-10-31-00-11_jaxatari_montezuma_pixels_cleanrl_ppo_rnd_environments-per-copy-128_rollout-steps-128_ideas-30-plus-20_steps-per-copy-2000000000_seed-start-1');P=next((R/'remote_setup').glob('2026-11-05-05-49_*'));U=R/'data'/P.name
path=P/'independent_h100_common_timing_raw_review_exact_metadata_child.py'
assert hashlib.sha256(path.read_bytes()).hexdigest()=='a247a3b3e05a320c0fdde513d3f94190918f80beae37b793ea9a35bc73f6519b'
spec=importlib.util.spec_from_file_location('exact_child_review_fixture',path);audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)
plan=audit.source_only();direct,common,validation,training,bootstrap=audit.audit_context(plan)
command=['/usr/bin/python3','-I','-S','-B',str(U/'code/direct_common_confirmation.py'),'--plan-stdin'];environment=common.metadata_environment(os.environ);start=time.monotonic();checks=[]
sys.audit('subprocess.Popen','/usr/bin/python3',command,None,environment);checks.append('exact frozen metadata command/environment admitted as a simulated event only')
negative=[('/other/python',command,None,environment),('/usr/bin/python3',command[:-1]+['--other'],None,environment),('/usr/bin/python3',command[:4]+['/other/source.py',command[-1]],None,environment),('/usr/bin/python3',command,str(U),environment),('/usr/bin/python3',command,None,{**environment,'UNDECLARED':'1'}),('nvidia-smi',['nvidia-smi','-i',audit.GPU],None,environment)]
for index,event in enumerate(negative):
 try:sys.audit('subprocess.Popen',*event)
 except RuntimeError:checks.append('refused exact negative simulated event '+str(index))
 else:raise AssertionError('unexpected child permission '+str(index))
original=audit.sha;audit.sha=lambda p:'0'*64
try:
 try:sys.audit('subprocess.Popen','/usr/bin/python3',command,None,environment)
 except RuntimeError:checks.append('changed metadata source hash refused')
 else:raise AssertionError('changed metadata source accepted')
finally:audit.sha=original
record={'kind':'exact_metadata_child_audit_guard_focused_controls','passed':True,'checks':checks,'check_count':len(checks),'simulated_audit_events_only':True,'actual_child_processes_created':0,'actual_result_reads':0,'GPU_native_or_numerical_operations':0,'source':{'path':str(path),'sha256':original(path)},'job_id':os.environ['SLURM_JOB_ID'],'step_id':os.environ['SLURM_STEP_ID'],'seconds':time.monotonic()-start}
p=P/'independent_exact_metadata_child_guard_controls.json'
with p.open('x') as f:json.dump(record,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
print(json.dumps({'record':str(p),'sha256':original(p),'checks':len(checks),'seconds':record['seconds'],'job_id':record['job_id'],'step_id':record['step_id']}))
