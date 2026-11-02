import hashlib,importlib.util,io,json,os,socket,sys,unittest
from pathlib import Path
P=Path(__file__).absolute().parent;NUM={'jax','jaxlib','numpy','torch','flax','optax','cupy','scipy'}
def guard(event,args):
 if event=='import' and args[0].split('.')[0] in NUM:raise RuntimeError('numerical import forbidden')
 if event in {'socket.connect','socket.bind','ctypes.dlopen'}:raise RuntimeError('network/shared-library operation forbidden')
 if event=='subprocess.Popen':
  command=args[1]
  if not isinstance(command,list) or command[:3]!=['git','-C','/p/rlprojects/RND_worktrees/merge-atari-cleanrl'] or command[3]!='cat-file':raise RuntimeError('only pinned local Git read allowed')
sys.addaudithook(guard);assert socket.gethostname().split('.')[0]=='portal05' and not os.environ.get('SLURM_JOB_ID') and os.getuid()==2618919
files=[P/'release_monitor_6783003.py',P/'test_release_monitor_6783003.py',P/'release_implementation_review_6783003.json']
def hashes():return {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
before=hashes();log=io.StringIO();spec=importlib.util.spec_from_file_location('tests',files[1]);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);r=unittest.TextTestRunner(stream=log,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(m));after=hashes();assert before==after
v={'kind':'cpu_monitor_6783003_release_author_metadata_verification','metadata_allocation':None,'metadata_host':'portal05','pid':os.getpid(),'uid':os.getuid(),'hostname':socket.gethostname(),'cgroup':Path('/proc/self/cgroup').read_text(),'proc_stat':Path('/proc/self/stat').read_text(),'tests':r.testsRun,'passed':r.wasSuccessful(),'source_before':before,'source_after':after,'test_output':log.getvalue(),'live_scheduler_or_network_actions':0,'numerical_modules_present':sorted(NUM.intersection(sys.modules))}
raw=(json.dumps(v,indent=2)+'\n').encode();target=P/'release_6783003_author_portal_metadata.json'
with target.open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
print(json.dumps({'receipt':str(target),'sha256':hashlib.sha256(raw).hexdigest(),'passed':r.wasSuccessful(),'tests':r.testsRun,'output':log.getvalue()}));raise SystemExit(0 if r.wasSuccessful() else 1)
