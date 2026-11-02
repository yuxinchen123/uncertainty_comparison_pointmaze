import ctypes,datetime,hashlib,importlib.util,io,json,os,socket,sys,unittest
from pathlib import Path
W=Path('/p/rlprojects/RND_worktrees/2026-11-01-17-55_a4500-common-confirmation_prior-correctness-required_rounds-7_complete-updates-per-arm-10_copies-1-2');S=W/'10_jax_exploration_platform/scripts/atari_pixel_campaign';OUT=Path(__file__).parent
assert os.environ.get('SLURM_JOB_ID')=='6783003' and os.environ.get('SLURM_STEP_ID','').isdigit()
NUM={'jax','jaxlib','numpy','torch','flax','optax','scipy','cupy','jaxtari','exploration_platform'}
def guard(event,args):
 if event=='import' and args[0].split('.')[0] in NUM:raise RuntimeError('numerical import forbidden')
 if event in {'socket.connect','socket.bind','ctypes.dlopen','subprocess.Popen','os.system'}:raise RuntimeError('live operation forbidden: '+event)
sys.addaudithook(guard)
files=[S/'common_confirmation.py',S/'measure_selected.py',S/'configured_measurement.py',S/'tests/test_common_confirmation.py']
hashes=lambda:{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
before=hashes();sys.path.insert(0,str(S));spec=importlib.util.spec_from_file_location('common_confirmation_tests',files[-1]);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);output=io.StringIO();result=unittest.TextTestRunner(stream=output,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(module));after=hashes();assert before==after
value={'schema_version':1,'kind':'common_confirmation_first_author_metadata_tests','allocation':'6783003','step':os.environ['SLURM_STEP_ID'],'pid':os.getpid(),'cgroup':Path('/proc/self/cgroup').read_text(),'at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'tests':result.testsRun,'passed':result.wasSuccessful(),'source_before':before,'source_after':after,'output':output.getvalue(),'real_numerical_imports':sorted(NUM.intersection(sys.modules)),'actual_gpu_library_network_or_job_operations':0}
raw=(json.dumps(value,indent=2)+'\n').encode();p=OUT/('author_6783003_'+os.environ['SLURM_STEP_ID']+'.json')
with p.open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
print(json.dumps({'receipt':str(p),'sha256':hashlib.sha256(raw).hexdigest(),'tests':result.testsRun,'passed':result.wasSuccessful(),'output':output.getvalue()}));raise SystemExit(0 if result.wasSuccessful() else 1)
