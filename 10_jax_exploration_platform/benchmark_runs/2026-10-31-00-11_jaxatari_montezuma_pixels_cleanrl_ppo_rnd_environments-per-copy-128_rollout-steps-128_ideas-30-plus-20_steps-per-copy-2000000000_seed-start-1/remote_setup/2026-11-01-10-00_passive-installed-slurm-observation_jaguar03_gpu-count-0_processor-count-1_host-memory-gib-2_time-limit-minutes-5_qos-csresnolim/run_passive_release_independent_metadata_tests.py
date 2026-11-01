import hashlib,importlib.util,io,json,os,socket,subprocess,sys,unittest
from pathlib import Path
P=Path(__file__).absolute().parent;NUM={'jax','jaxlib','numpy','torch','flax','optax','cupy','scipy'}
def guard(event,args):
 if event=='import' and args[0].split('.')[0] in NUM:raise RuntimeError('numerical import forbidden')
 if event in {'socket.connect','socket.bind','ctypes.dlopen'}:raise RuntimeError('network/shared-library operation forbidden')
 if event=='subprocess.Popen':
  command=args[1]
  if not isinstance(command,list) or command[:3]!=['git','-C','/p/rlprojects/RND_worktrees/merge-atari-cleanrl'] or command[3]!='cat-file':raise RuntimeError('only pinned local Git read allowed')
sys.addaudithook(guard);assert os.environ.get('SLURM_JOB_ID')=='6781231' and os.environ.get('SLURM_STEP_ID','').isdigit() and os.getuid()==2618919
files=[P/'release_passive_observer_6781903.py',P/'test_release_passive_observer_6781903.py',P/'release_implementation_review_6781903.json']
freeze_raw=(P/'release_source_freeze_6781903.json').read_bytes()
assert hashlib.sha256(freeze_raw).hexdigest()=='43724b43405ad2a1ecac8a9a26aa7a344f88e140133193c3aa5eef8307cf7fcd'
freeze=json.loads(freeze_raw)
for item in freeze['files'].values():
 assert hashlib.sha256(Path(item['path']).read_bytes()).hexdigest()==item['sha256']
files += [Path(item['path']) for item in freeze['files'].values() if Path(item['path']) not in files]
files.append(P/'release_source_freeze_6781903.json')
def hashes():return {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
before=hashes();log=io.StringIO();spec=importlib.util.spec_from_file_location('tests',files[1]);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);r=unittest.TextTestRunner(stream=log,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(m))

from unittest.mock import patch
import tempfile
checks=[]
with tempfile.TemporaryDirectory() as tmp,patch.object(m.m,'PACKET',Path(tmp)):
 path=Path(tmp)/'release_intent.json'
 path.write_bytes(b'earlier exclusive attempt')
 with patch.object(m.m.subprocess,'run') as process:
  try:m.m.release_once(m.intent())
  except ValueError:rejected=True
  else:rejected=False
 assert rejected and process.call_count==0 and path.read_bytes()==b'earlier exclusive attempt'
 checks.append('existing intent prevents every second query/release')
with tempfile.TemporaryDirectory() as tmp,patch.object(m.m,'PACKET',Path(tmp)):
 value=m.intent();value['published_release_commit']=None
 with patch.object(m.m.subprocess,'run') as process:
  try:m.m.release_once(value)
  except ValueError:rejected=True
  else:rejected=False
 assert rejected and process.call_count==0 and not list(Path(tmp).iterdir())
 checks.append('missing publication refuses even direct helper before intent')
original=m.m.subprocess.check_output
original_run=m.m.subprocess.run
fake_publication='f'*40
publication_blob_reads=[]
def output(command,*args,**kwargs):
 if command[-1].startswith(fake_publication+':'):
  publication_blob_reads.append(command)
  return b'synthetic differently published bytes'
 return original(command,*args,**kwargs)
ancestor_calls=[]
def fake_run(command,*args,**kwargs):
 if command[3:5]==['merge-base','--is-ancestor']:
  ancestor_calls.append(command)
  return subprocess.CompletedProcess(command,0,b'',b'')
 return original_run(command,*args,**kwargs)
with patch.object(m.m.socket,'gethostname',return_value='portal05'),patch.object(m.m.subprocess,'run',side_effect=fake_run),patch.object(m.m.subprocess,'check_output',side_effect=output):
 try:m.m.checked(fake_publication)
 except ValueError as error:
  rejected='not published' in str(error)
 else:rejected=False
assert rejected and len(publication_blob_reads)==1 and len(ancestor_calls)==1 and not (P/'release_intent.json').exists()
checks.append('different published blob refuses before intent or scheduler')
after=hashes();assert before==after

v={'kind':'passive_observer_6781903_release_independent_metadata_verification','actual_job_id':os.environ['SLURM_JOB_ID'],'actual_step_id':os.environ['SLURM_STEP_ID'],'pid':os.getpid(),'uid':os.getuid(),'hostname':socket.gethostname(),'cgroup':Path('/proc/self/cgroup').read_text(),'proc_stat':Path('/proc/self/stat').read_text(),'tests':r.testsRun,'extra_checks':checks,'passed':r.wasSuccessful() and r.testsRun==6 and len(checks)==3,'source_before':before,'source_after':after,'test_output':log.getvalue(),'live_scheduler_or_network_actions':0,'numerical_modules_present':sorted(NUM.intersection(sys.modules))}
raw=(json.dumps(v,indent=2)+'\n').encode();target=P/('release_independent_6781231_'+os.environ['SLURM_STEP_ID']+'.json')
with target.open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
print(json.dumps({'receipt':str(target),'sha256':hashlib.sha256(raw).hexdigest(),'passed':r.wasSuccessful(),'tests':r.testsRun,'extra_checks':checks,'output':log.getvalue()}));raise SystemExit(0 if r.wasSuccessful() else 1)
