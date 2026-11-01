"""Run synthetic release controls on the current owned CPU allocation only."""
import hashlib,importlib.util,io,json,os,socket,sys,time,unittest
from pathlib import Path
W=Path(__file__).absolute().parent
NUM={'jax','jaxlib','numpy','torch','flax','optax','cupy','scipy'}
I='/p/rlprojects/RND_worktrees/merge-atari-cleanrl'
def guard(event,args):
 if event=='import' and args[0].split('.')[0] in NUM:raise RuntimeError('numerical import forbidden')
 if event in {'socket.connect','socket.bind','ctypes.dlopen'}:raise RuntimeError('network/shared-library operation forbidden')
 if event=='subprocess.Popen':
  command=args[1]
  if not isinstance(command,list) or command[:3]!=['git','-C',I] or command[3] not in {'cat-file','merge-base'}:raise RuntimeError('only pinned local Git reads allowed: '+repr(command))
sys.addaudithook(guard)
assert os.environ.get('SLURM_JOB_ID')=='6781087' and os.environ.get('SLURM_STEP_ID','').isdigit() and os.getuid()==2618919
paths=[W/'release_aces_full_site_2275518.py',W/'test_release_aces_full_site_2275518.py']
def hashes():return {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
before=hashes();log=io.StringIO();started=time.monotonic()
spec=importlib.util.spec_from_file_location('release_tests',paths[1]);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
result=unittest.TextTestRunner(stream=log,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(module))
after=hashes();assert before==after
value={'kind':'aces_2275518_release_author_metadata_verification','job_id':os.environ['SLURM_JOB_ID'],'step_id':os.environ['SLURM_STEP_ID'],'pid':os.getpid(),'uid':os.getuid(),'hostname':socket.gethostname(),'cgroup':Path('/proc/self/cgroup').read_text(),'proc_stat':Path('/proc/self/stat').read_text(),'tests_run':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),'passed':result.wasSuccessful(),'elapsed_seconds':time.monotonic()-started,'source_before':before,'source_after':after,'test_output':log.getvalue(),'numerical_modules_present':sorted(NUM.intersection(sys.modules)),'network_or_scheduler_actions':0,'synthetic_fixture_note':'Held shape derives only job ID and packet path from saved older ACES held metadata. It does not claim a new actual allocation or execute a cluster command.'}
raw=(json.dumps(value,indent=2)+'\n').encode();target=W/('release_2275518_author_'+os.environ['SLURM_JOB_ID']+'_'+os.environ['SLURM_STEP_ID']+'.json')
with target.open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
print(json.dumps({'receipt':str(target),'sha256':hashlib.sha256(raw).hexdigest(),'tests_run':result.testsRun,'passed':result.wasSuccessful(),'output':log.getvalue()}));raise SystemExit(0 if result.wasSuccessful() else 1)
