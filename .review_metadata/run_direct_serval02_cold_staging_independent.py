import ast,datetime,hashlib,json,os,pathlib,shutil,subprocess,tempfile,sys
W=pathlib.Path(__file__).resolve().parents[1];S=W/'10_jax_exploration_platform/scripts/atari_pixel_campaign'
assert os.environ.get('SLURM_JOB_ID')=='6781231' and os.environ.get('SLURM_STEP_ID','').isdigit()
sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
f=W/'.review_metadata/direct_serval02_core_sixth_source_freeze.json';assert sha(f)=='af07c0e781890e438fb4235ba9a24e475041e17a1fa4147ae3f4f00a67a7a6a0'
expected=json.loads(f.read_text())['files'];assert all(sha(W/p)==v for p,v in expected.items())
tree=ast.parse((S/'serval02_card1_binding.py').read_text());names=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='HELPERS' for t in n.targets))
before={n:sha(S/n) for n in names}
PROBE=r'''
import builtins,ctypes,hashlib,importlib,json,pathlib,sys
root=pathlib.Path(sys.argv[1]).resolve(strict=True);expected=json.loads(sys.argv[2]);NUM={'jax','jaxlib','numpy','scipy','torch','flax','optax','jaxtari','pynvml','cupy','numba','exploration_platform'}
original=builtins.__import__
def guard(name,*a,**kw):
 if name.split('.')[0] in NUM:raise AssertionError('numerical import prohibited')
 return original(name,*a,**kw)
builtins.__import__=guard
observed=[]
def audit(event,args):
 if event in {'subprocess.Popen','socket.connect','os.system'} or event=='ctypes.dlopen' and args[0] is not None:raise AssertionError('live operation prohibited')
 if event=='open' and isinstance(args[0],str) and args[0].startswith('/dev/nvidia'):raise AssertionError('GPU device open prohibited')
sys.addaudithook(audit);sys.path.insert(0,str(root))
for filename,digest in expected.items():
 module=importlib.import_module(filename[:-3]);path=pathlib.Path(module.__file__).resolve(strict=True)
 assert path==root/filename and hashlib.sha256(path.read_bytes()).hexdigest()==digest
 observed.append(filename)
module=sys.modules['serval02_card1_binding'];assert pathlib.Path(module.shared_portable_module().__file__).resolve(strict=True)==root/'portable_evidence.py'
assert not any(name.split('.')[0] in NUM for name in sys.modules)
print(json.dumps({'loaded_helpers':observed,'module_paths_exact':True,'numerical_imports':False,'isolated':sys.flags.isolated,'no_site':sys.flags.no_site,'GPU_operations':0}))
'''
rows=[]
for missing in (None,'uva_active_configuration_observation.py'):
 with tempfile.TemporaryDirectory(prefix='direct-cold-staged-helpers-') as tmp:
  root=pathlib.Path(tmp)
  for name in names:
   if name!=missing:shutil.copyfile(S/name,root/name)
  command=['/usr/bin/python3','-I','-S','-B','-c',PROBE,str(root),json.dumps(before)]
  result=subprocess.run(command,capture_output=True,text=True,timeout=30)
  if missing is None:
   assert result.returncode==0,result.stderr;value=json.loads(result.stdout);assert len(value['loaded_helpers'])==len(names) and value['isolated']==value['no_site']==1
  else:assert result.returncode!=0 and 'uva_active_configuration_observation' in result.stderr
  rows.append({'missing':missing,'returncode':result.returncode,'stdout':result.stdout,'stderr':result.stderr,'probe_sha256':hashlib.sha256(PROBE.encode()).hexdigest(),'assertion_passed':True})
after={n:sha(S/n) for n in names};assert before==after and all(sha(W/p)==v for p,v in expected.items())
value={'schema_version':1,'kind':'independent_direct_core_cold_staging_review','at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'job_id':os.environ['SLURM_JOB_ID'],'step_id':os.environ['SLURM_STEP_ID'],'pid':os.getpid(),'birth':pathlib.Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19],'cgroup':pathlib.Path('/proc/self/cgroup').read_text(),'freeze_sha256':sha(f),'helpers_before':before,'helpers_after':after,'unchanged':before==after,'checks_passed':2,'checks':rows,'NUM_work':False,'direct_queries':0,'GPU_work':False,'job_actions':0}
out=W/'.review_metadata'/('direct_serval02_core_cold_staging_independent_6781231_'+os.environ['SLURM_STEP_ID']+'.json')
with out.open('x') as stream:json.dump(value,stream,indent=2);stream.write('\n');stream.flush();os.fsync(stream.fileno())
print(json.dumps({'receipt':str(out),'sha256':sha(out),'checks_passed':2,'helper_count':len(names)}))
