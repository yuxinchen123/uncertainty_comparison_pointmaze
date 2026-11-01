"""Run existing producer metadata contracts on the already owned CPU allocation."""
import ast,ctypes,datetime,hashlib,importlib.abc,json,os,pathlib,subprocess,sys,unittest
W=pathlib.Path(__file__).resolve().parents[1];S=W/'10_jax_exploration_platform/scripts/atari_pixel_campaign'
assert os.environ.get('SLURM_JOB_ID')=='6781231' and os.environ.get('SLURM_STEP_ID','').isdecimal()
f=W/'.review_metadata/direct_adapter_fifth_source_freeze.json';expected=json.loads(f.read_bytes())['files']
images=lambda:{n:hashlib.sha256((W/n).read_bytes()).hexdigest() for n in expected}
before=images();assert before==expected;assert hashlib.sha256(f.read_bytes()).hexdigest()=='5dd2f095d336de79b7e1e943f5741af979f16fae87ae71d58af99376075c1c95';protected=json.loads(f.read_bytes())['protected'];assert all(hashlib.sha256((W/n).read_bytes()).hexdigest()==v for n,v in protected.items());blocked=[]
class Block(importlib.abc.MetaPathFinder):
 def find_spec(self,name,path=None,target=None):
  if name.split('.')[0] in {'jax','jaxlib','numpy','torch','flax','optax','cupy','scipy','jaxtari','pynvml','numba'}:blocked.append(name);raise RuntimeError('numerical import forbidden')
sys.meta_path.insert(0,Block());ctypes.CDLL=lambda *a,**k:(_ for _ in ()).throw(RuntimeError('shared-library loads forbidden'))
real_popen=subprocess.Popen;commands=[]
tree=ast.parse((S/'tests/test_configured_measurement.py').read_text())
base_probe=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='ISOLATED_IMPORT_PROBE' for t in n.targets))
probes={base_probe}
for node in ast.walk(ast.parse((S/'tests/test_direct_measurement.py').read_text())):
 if isinstance(node,ast.BinOp) and isinstance(node.op,ast.Add) and isinstance(node.right,ast.Constant) and isinstance(node.right.value,str) and 'import direct_measurement as direct' in node.right.value:probes.add(base_probe+node.right.value)
def owned_temporary(value):
 path=pathlib.Path(value)
 return path.is_absolute() and str(path).startswith('/tmp/') and path.resolve(strict=True)==path and path.stat().st_uid==os.getuid()
class LocalOnly(real_popen):
 def __init__(self,args,*a,**k):
  cmd=list(args) if isinstance(args,(tuple,list)) else [];allowed=False
  if cmd[:2]==['git','clone']:
   allowed=pathlib.Path(cmd[-2]).is_absolute() and not any(x in cmd[-2] for x in [':','@']) and pathlib.Path(cmd[-1]).is_absolute() and str(cmd[-1]).startswith('/tmp/') and pathlib.Path(cmd[-1]).parent.stat().st_uid==os.getuid()
  elif cmd[:2]==['git','-C']:
   root=pathlib.Path(cmd[2]);read_only=cmd[3] in {'show','cat-file','rev-parse','status','symbolic-ref','ls-tree','diff','ls-files','merge-base'}
   allowed=root.is_absolute() and (owned_temporary(root) or str(root).startswith('/p/rlprojects/RND_worktrees/') and read_only)
  elif cmd and cmd[0]==sys.executable and '-I' in cmd and '-c' in cmd:
   index=cmd.index('-c');allowed=cmd[index+1] in probes and len(cmd)>index+2 and owned_temporary(cmd[index+2]) and set(cmd[1:index])<={'-I','-S','-B'}
  assert allowed,'live process forbidden: '+repr(cmd)
  commands.append(cmd);super().__init__(args,*a,**k)
def extra_guard(event,args):
 if event=='socket.connect' or event=='ctypes.dlopen' and args[0] is not None or event=='open' and isinstance(args[0],str) and args[0].startswith('/dev/nvidia'):blocked.append(event);raise AssertionError('real network/library/device refused')
sys.addaudithook(extra_guard)
subprocess.Popen=LocalOnly;sys.path[:0]=[str(S),str(S/'tests')]
names=['test_direct_qualification_reader','test_direct_measurement','test_configured_measurement','test_measure_selected','test_profile_selected','test_qualification_pytest_root'];suite=unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromName(n) for n in names)
result=unittest.TextTestRunner(verbosity=2).run(suite);after=images();assert before==after and not blocked;assert all(hashlib.sha256((W/n).read_bytes()).hexdigest()==v for n,v in protected.items())
step=os.environ['SLURM_STEP_ID'];raw=pathlib.Path('/proc/self/stat').read_text()
v={'schema_version':1,'kind':'direct_adapter_fifth_independent_metadata_test_result','at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'job_id':'6781231','step_id':step,'pid':os.getpid(),'birth':raw[raw.rfind(')')+2:].split()[19],'cgroup':pathlib.Path('/proc/self/cgroup').read_text(),'tests':result.testsRun,'errors':len(result.errors),'failures':len(result.failures),'passed':result.wasSuccessful(),'error_details':[{'test':t.id(),'traceback':s} for t,s in result.errors],'failure_details':[{'test':t.id(),'traceback':s} for t,s in result.failures],'source_before':before,'source_after':after,'numerical_import_attempts':blocked,'actual_subprocesses':commands,'strict_local_command_policy':True,'actual_gpu_operations':False,'actual_host_contact':False,'scientific_credit':0,'scope':'Synthetic direct adapter plus default/UVA/root-directory metadata checks; no GPU or qualification credit; metadata source review; actual direct qualification remains required.'}
p=W/'.review_metadata'/('direct_adapter_fifth_independent_6781231_'+step+'.json')
with p.open('x') as out:json.dump(v,out,sort_keys=True,indent=2);out.write('\n');out.flush();os.fsync(out.fileno())
print(json.dumps({'receipt':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'tests':result.testsRun,'passed':result.wasSuccessful()}));sys.exit(0 if result.wasSuccessful() else 1)
