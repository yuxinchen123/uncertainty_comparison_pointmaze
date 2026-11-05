"""Exercise exact cold metadata-to-numerical import ordering without loading numerical code."""
import argparse,builtins,ctypes,hashlib,importlib.abc,importlib.util,json,os,pathlib,subprocess,sys,tempfile,types
from unittest.mock import patch
parser=argparse.ArgumentParser();parser.add_argument('--core',required=True);parser.add_argument('--unit',required=True);parser.add_argument('--expected',choices=('legacy-failure','corrected-success'),required=True);args=parser.parse_args()
unit=pathlib.Path(args.unit);plan=json.loads((unit/'measurement_plan.json').read_text());stage=unit/'code';sys.path.insert(0,str(stage));blocked=[]
class NoNumerical(importlib.abc.MetaPathFinder):
 def find_spec(self,name,path=None,target=None):
  if name.split('.')[0] in {'jax','jaxlib','numpy','torch','flax','optax','jaxtari','cupy','scipy','exploration_platform'}:blocked.append(name);raise RuntimeError('real numerical import forbidden')
sys.meta_path.insert(0,NoNumerical())
ctypes.CDLL=lambda *a,**k:(_ for _ in ()).throw(RuntimeError('real shared-library load forbidden'))
real_popen=subprocess.Popen
class GitOnly(real_popen):
 def __init__(self,command,*a,**k):
  assert isinstance(command,(list,tuple)) and len(command)>3 and list(command[:3])==['git','-C',plan['source_root']] and command[3] in {'rev-parse','symbolic-ref','status','ls-tree','cat-file'},'only exact read-only original-source commands are allowed'
  super().__init__(command,*a,**k)
subprocess.Popen=GitOnly
sys.addaudithook(lambda event,a:(_ for _ in ()).throw(RuntimeError('network forbidden')) if event in {'socket.connect','socket.bind','socket.getaddrinfo'} else None)
assert 'ideas' not in sys.modules and 'controller' not in sys.modules
spec=importlib.util.spec_from_file_location('actual_selected_import_order',stage/'measure_selected.py');producer=importlib.util.module_from_spec(spec);spec.loader.exec_module(producer)
spec=importlib.util.spec_from_file_location('actual_direct_import_core',args.core);core=importlib.util.module_from_spec(spec);spec.loader.exec_module(core)
claim=core.current_admission_claim(plan);core.check_live_admission(claim)
preloaded=getattr(sys.modules.get('ideas'),'__file__',None)
# Only actual numerical imports are replaced; the unchanged producer controls every path insertion/import/check.
class AfterSourceGate(RuntimeError):pass
fake_jax=types.SimpleNamespace(config=types.SimpleNamespace(update=lambda *a,**k:(_ for _ in ()).throw(AfterSourceGate('strict imported-source gate passed'))))
real_import=builtins.__import__
with tempfile.TemporaryDirectory() as temporary:
 package=pathlib.Path(temporary)/'jaxtari';package.mkdir();(package/'__init__.py').write_text('# synthetic package gate fixture\n')
 fake_jaxtari=types.ModuleType('jaxtari');fake_jaxtari.__file__=str(package/'__init__.py')
 synthetic_imports=[]
 def synthetic(name,globals=None,locals=None,fromlist=(),level=0):
  if name=='jax':synthetic_imports.append(name);return fake_jax
  if name=='jaxtari':synthetic_imports.append(name);sys.modules['jaxtari']=fake_jaxtari;return fake_jaxtari
  if name in {'exploration_platform.atari_pixels.config','exploration_platform.atari_pixels.environment','exploration_platform.atari_pixels.training'}:
   synthetic_imports.append(name);return types.SimpleNamespace(PixelPPOConfig=object,AtariPixelEnvironment=object,PixelPPORND=object)
  return real_import(name,globals,locals,fromlist,level)
 outcome=None
 with patch.object(builtins,'__import__',side_effect=synthetic),patch.object(producer.metadata,'distribution',return_value=types.SimpleNamespace(locate_file=lambda path:pathlib.Path(temporary)/path)):
  try:producer._compute(unit,plan,{},{} )
  except AfterSourceGate:outcome='corrected-success'
  except ValueError as error:
   assert str(error)=='the imported numerical helper differs from its recorded source: ideas',str(error)
   outcome='legacy-failure'
 assert outcome==args.expected and not blocked
 numerical_scripts=pathlib.Path(plan['source_root'])/'10_jax_exploration_platform/scripts/atari_pixel_campaign'
 paired=sys.modules['paired_benchmark'];ideas=sys.modules['ideas']
 assert pathlib.Path(paired.__file__).resolve()==numerical_scripts/'paired_benchmark.py'
 if args.expected=='legacy-failure':assert pathlib.Path(ideas.__file__).resolve()==stage/'ideas.py' and preloaded==str(stage/'ideas.py')
 else:assert pathlib.Path(ideas.__file__).resolve()==numerical_scripts/'ideas.py' and preloaded is None and 'controller' not in sys.modules
 print(json.dumps({'kind':'cold_actual_direct_metadata_and_selected_import_order','outcome':outcome,'core_path':args.core,'core_sha256':hashlib.sha256(pathlib.Path(args.core).read_bytes()).hexdigest(),'producer_path':str(stage/'measure_selected.py'),'producer_sha256':hashlib.sha256((stage/'measure_selected.py').read_bytes()).hexdigest(),'numerical_root':plan['source_root'],'metadata_ideas_path':preloaded,'actual_paired_path':paired.__file__,'actual_ideas_path':ideas.__file__,'synthetic_numerical_imports':synthetic_imports,'real_numerical_import_attempts':blocked,'actual_gpu_actions':0,'live_scheduler_queries':0,'scientific_credit':0,'admission_granted':False}),flush=True)
