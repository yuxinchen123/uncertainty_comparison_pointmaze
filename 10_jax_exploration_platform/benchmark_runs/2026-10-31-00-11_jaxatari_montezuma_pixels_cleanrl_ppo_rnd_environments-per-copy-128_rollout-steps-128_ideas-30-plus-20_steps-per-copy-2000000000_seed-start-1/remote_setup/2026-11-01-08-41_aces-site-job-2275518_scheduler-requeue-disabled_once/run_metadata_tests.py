import hashlib,importlib.util,io,json,os,socket,sys,time,unittest
from pathlib import Path
P=Path(__file__).absolute().parent
NUM={'jax','jaxlib','numpy','torch','flax','optax','cupy','scipy'}
def guard(event,args):
 if event=='import' and args[0].split('.')[0] in NUM:raise RuntimeError('numerical import forbidden')
 if event in {'subprocess.Popen','socket.connect','socket.bind','ctypes.dlopen'}:raise RuntimeError('live command, network or shared library forbidden')
sys.addaudithook(guard)
assert os.environ.get('SLURM_JOB_ID')=='6781087' and os.environ.get('SLURM_STEP_ID','').isdigit() and os.getuid()==2618919
files=[P/'disable_requeue_2275518.py',P/'test_disable_requeue_2275518.py'];before={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
log=io.StringIO();start=time.monotonic();spec=importlib.util.spec_from_file_location('tests',files[1]);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);result=unittest.TextTestRunner(stream=log,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(m))
after={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files};assert before==after
v={'kind':'aces_2275518_requeue_disable_author_metadata_verification','actual_job_id':os.environ['SLURM_JOB_ID'],'actual_step_id':os.environ['SLURM_STEP_ID'],'pid':os.getpid(),'uid':os.getuid(),'host':socket.gethostname(),'cgroup':Path('/proc/self/cgroup').read_text(),'proc_stat':Path('/proc/self/stat').read_text(),'source_before':before,'source_after':after,'test_count':result.testsRun,'passed':result.wasSuccessful(),'elapsed_seconds':time.monotonic()-start,'test_output':log.getvalue(),'live_commands':0,'network_calls':0,'numerical_modules_present':sorted(NUM.intersection(sys.modules))}
raw=(json.dumps(v,indent=2)+'\n').encode();p=P/('author_metadata_6781087_'+os.environ['SLURM_STEP_ID']+'.json')
with p.open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
print(json.dumps({'receipt':str(p),'sha256':hashlib.sha256(raw).hexdigest(),'passed':result.wasSuccessful(),'tests':result.testsRun,'output':log.getvalue()}));raise SystemExit(0 if result.wasSuccessful() else 1)
