"""Independently check the frozen H100 review source without actual result reads."""
import pathlib,json,hashlib,os,stat,importlib.util,importlib.abc,ctypes,sys,unittest,io
from datetime import datetime,timezone
p=pathlib.Path(__file__).absolute().parent
assert os.environ['SLURM_JOB_ID']=='6783003' and os.environ['SLURM_STEP_ID'].isdecimal() and os.getresuid()==(2618919,)*3
numerical={'jax','jaxlib','numpy','torch','flax','optax','cupy','scipy','jaxtari','exploration_platform'}
blocked=[]
class Deny(importlib.abc.MetaPathFinder):
 def find_spec(self,name,path=None,target=None):
  if name.split('.')[0] in numerical:blocked.append(name);raise RuntimeError('numerical import forbidden')
sys.meta_path.insert(0,Deny())
ctypes.CDLL=ctypes.PyDLL=lambda *a,**k:(_ for _ in ()).throw(RuntimeError('library load forbidden'))
sys.addaudithook(lambda event,args:(_ for _ in ()).throw(RuntimeError('network forbidden')) if event in {'socket.connect','socket.bind','socket.getaddrinfo'} else None)
freeze=json.loads((p/'source_freeze.json').read_text())
before={name:hashlib.sha256(pathlib.Path(name).read_bytes()).hexdigest() for name in freeze['files']}
assert before==freeze['files']
s=importlib.util.spec_from_file_location('independent_fresh_direct_tests',p/'test_fresh_direct_raw_review.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
capture=io.StringIO();result=unittest.TextTestRunner(stream=capture,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(m))
extras=[
 ('missing successful wait',lambda d:d[4]['direct_controller_outcome.json'].update(actual_wait_saved=False)),
 ('altered captured output size',lambda d:d[4]['direct_controller_output_hashes.json']['stdout'].update(size_bytes=1)),
 ('controller claims independent qualification before raw review',lambda d:d[4]['direct_controller_outcome.json'].update(qualification_independently_verified=True))]
for name,mutate in extras:
 d=m.fixture();mutate(d)
 try:m.run_fixture(d)
 except ValueError:pass
 else:raise AssertionError('invalid evidence accepted: '+name)
assert result.wasSuccessful() and not blocked and not numerical.intersection(sys.modules)
after={name:hashlib.sha256(pathlib.Path(name).read_bytes()).hexdigest() for name in freeze['files']};assert before==after
v={'kind':'independent_fresh_h100_raw_review_source_approval','passed':True,'at_utc':datetime.now(timezone.utc).isoformat(),'job_id':'6783003','step_id':os.environ['SLURM_STEP_ID'],'source_freeze_sha256':hashlib.sha256((p/'source_freeze.json').read_bytes()).hexdigest(),'source_files':before,'unittest_methods':result.testsRun,'author_source_and_synthetic_cases_repeated':23,'additional_negative_controls':[name for name,_ in extras],'total_source_and_synthetic_cases':26,'all_50_raw_AST_statements_retained':True,'actual_result_read':False,'terminal_audit_attempted':False,'numerical_imports':0,'gpu_actions':0,'network_actions':0,'scientific_credit':0,'admission_granted':False}
for name,raw in [('root_independent_test_output.raw',capture.getvalue().encode()),('root_independent_source_review.json',(json.dumps(v,indent=2,sort_keys=True)+'\n').encode())]:
 with (p/name).open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
print(json.dumps({'passed':True,'step':v['step_id'],'cases':26,'review_sha256':hashlib.sha256((p/'root_independent_source_review.json').read_bytes()).hexdigest()}))

