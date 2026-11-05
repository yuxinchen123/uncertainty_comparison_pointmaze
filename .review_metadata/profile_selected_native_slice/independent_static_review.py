import ast,hashlib,json,os,time
from pathlib import Path
W=Path("/p/rlprojects/RND_worktrees/2026-11-05-06-20_a4500-native-process-control_python-3.12_total-seconds-300_work-seconds-240_cleanup-seconds-60_admission-closed")
D=W/".review_metadata/profile_selected_native_slice"
R=Path("/p/rlprojects/RND_worktrees/merge-atari-cleanrl/10_jax_exploration_platform/benchmark_runs/2026-10-31-00-11_jaxatari_montezuma_pixels_cleanrl_ppo_rnd_environments-per-copy-128_rollout-steps-128_ideas-30-plus-20_steps-per-copy-2000000000_seed-start-1")
assert os.environ['SLURM_JOB_ID']=='6821437'
assert '6821437' in (R/'slurm/submitted_jobids_uva.txt').read_text().split()
start=time.monotonic();checks=[]
def check(label,value):
 assert value,label
 checks.append(label)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
freeze=D/'source_freeze_author_1.json'
check('frozen author profile slice',sha(freeze)=='5e5325ed5bdc28f2573dad66b4b1328392f07d28a24ec11d68396884f7a16d0d')
f=json.loads(freeze.read_bytes())
for rel,row in f['files'].items():
 path=W/rel
 check('frozen byte hash '+rel,sha(path)==row['sha256'])
 check('frozen byte size '+rel,path.stat().st_size==row['bytes'])
 check('frozen file mode '+rel,path.stat().st_mode & 0o777==row['mode'])
base=Path(f['published_e206_baseline']['path']);check('published reference bytes',sha(base)==f['published_e206_baseline']['sha256'])
current=W/'10_jax_exploration_platform/scripts/atari_pixel_campaign/profile_selected.py'
def funcs(p):return {n.name:n for n in ast.parse(p.read_bytes()).body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))}
a,b=funcs(base),funcs(current)
unchanged=('process_birth','process_parent','profile_updates','tool_record','direct_profile_reader','direct_process_reader','owned_processes','require_reviewed_profile_process_admission','process_reader','accepted_confirmation_common_reader','initial_summaries_complete','validate_measured_baseline','child_command','stop_own_child','interruption_cleanup','_capture_core','capture_direct_owned_processes','run_direct_owned_waiter','capture_direct_owned_child')
for name in unchanged:
 check('unchanged existing function '+name,ast.dump(a[name],include_attributes=False)==ast.dump(b[name],include_attributes=False))
for name in ('configured_native_tool_record','capture_configured_native_processes'):
 text=ast.unparse(b[name]);check('scope command present '+name,('scope.run(' if name=='configured_native_tool_record' else 'run_command=scope.run') in text)
for name in ('run_configured_native_prepared','capture_configured_native_processes','run_configured_native_waiter','capture_configured_native_child'):
 text=ast.unparse(b[name])
 check('no direct numeric signal or process group calls '+name,all(v not in text for v in ('os.kill(', 'os.killpg(', '.terminate(', '.kill(', 'os.pidfd_open(', 'signal.pidfd_send_signal(')))
producer=ast.unparse(b['run_configured_native_prepared'])
check('native producer begins scope before first version',producer.index('with native.tool_scope(')<producer.index('configured_native_tool_record('))
check('native completion after scope body',next(n for n in ast.walk(b['run_configured_native_prepared']) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='complete').lineno>next(n for n in ast.walk(b['run_configured_native_prepared']) if isinstance(n,ast.With) and any('tool_scope' in ast.unparse(i.context_expr) for i in n.items)).end_lineno)
waiter=ast.unparse(b['run_configured_native_waiter'])
check('actual capture wait precedes actual wait-finished timestamp',waiter.index('child.wait(')<waiter.index('actual_wait_finished = time.time()'))
check('wait timestamp saved before capture handle close',waiter.index("'actual_wait_finished_at_seconds'")<waiter.index('observer.close_process_handle(child_handle)'))
check('formal admission remains unconditional refusal',len(b['require_reviewed_profile_process_admission'].body)==2 and isinstance(b['require_reviewed_profile_process_admission'].body[-1],ast.Raise))
check('capture core exact published AST',hashlib.sha256(ast.dump(b['_capture_core'],include_attributes=False).encode()).hexdigest()==f['unchanged_capture_core_ast_sha256'])
record={'kind':'independent_profile_slice_static_review','approved_for_final_union_review':True,'formal_admission':False,'actual_target_qualification':False,'scientific_credit':0,'helper_fixes_still_required':True,'tested_scope':'static profile slice only; no changing helper bytes read; no tools, native operations, numerical operations, devices, or target actions','job_id':os.environ['SLURM_JOB_ID'],'step_id':os.environ['SLURM_STEP_ID'],'pid':os.getpid(),'uid':os.getuid(),'cgroup':Path('/proc/self/cgroup').read_text(),'freeze':{'path':str(freeze),'sha256':sha(freeze)},'checks':checks,'check_count':len(checks),'seconds':time.monotonic()-start}
p=D/'independent_static_review.json'
with p.open('x') as out:json.dump(record,out,indent=2);out.write('\n');out.flush();os.fsync(out.fileno())
print(json.dumps({'check_count':len(checks),'seconds':record['seconds'],'review_path':str(p),'review_sha256':sha(p),'job_id':record['job_id'],'step_id':record['step_id']}))
