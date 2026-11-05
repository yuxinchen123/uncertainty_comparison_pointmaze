import ctypes,hashlib,importlib.util,json,os,sys,time,unittest
from pathlib import Path
D=Path(__file__).parent
expected={'produce_common_profile_prerequisite_fact.py':'2657be58b2cd0030119278bfdba7533135afc9a815d8572e53b02e40a1517bda','test_produce_common_profile_prerequisite_fact.py':'7a636cf56281e777563bcd14e5e39733fc57d41d88042c977a13b311d56cf315'}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
assert os.environ['SLURM_JOB_ID']=='6825983' and os.environ['SLURM_STEP_ID'].isdigit()
assert os.getresuid()==(2618919,)*3
assert '/job_6825983/step_'+os.environ['SLURM_STEP_ID']+'/' in Path('/proc/self/cgroup').read_text()
assert all(sha(D/name)==value for name,value in expected.items())
bootstrap={'stdlib_imports_before_guard':['ctypes','unittest','importlib.util'],'native_library_calls_requested_before_guard':0}
def guard(event,args):
 if event in ('socket.connect','socket.bind','socket.getaddrinfo','ctypes.dlopen','ctypes.dlsym','os.system'):raise RuntimeError('nonmetadata operation refused')
 if event=='subprocess.Popen':
  a=args[1]
  if not(isinstance(a,(list,tuple)) and len(a)>4 and a[0]=='git' and a[1]=='-C' and a[3] in ('show','cat-file')):raise RuntimeError('only source Git reads allowed')
 if event=='open' and isinstance(args[0],(str,bytes)):
  name=os.fsdecode(args[0])
  if name.startswith(('/dev/nvidia','/dev/dri','/proc/driver/nvidia')) or name=='/tmp/sl5nw_serval02_card1.lock':raise RuntimeError('GPU access refused')
sys.addaudithook(guard)
spec=importlib.util.spec_from_file_location('root_fact_source_tests',D/'test_produce_common_profile_prerequisite_fact.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
suite=unittest.defaultTestLoader.loadTestsFromModule(module);started=time.monotonic();out=unittest.TextTestRunner(verbosity=2).run(suite)
assert out.wasSuccessful()
assert all(sha(D/name)==value for name,value in expected.items())
record={'kind':'independent_common_fact_producer_source_review','approved':True,'passed':True,'independent_of_source_author':True,'reviewer':'/root','source_files':expected,'tests_run':out.testsRun,'seconds':time.monotonic()-started,'owned_job_id':os.environ['SLURM_JOB_ID'],'owned_step_id':os.environ['SLURM_STEP_ID'],'stdlib_bootstrap':bootstrap,'scope':'Only thin metadata producer source and tests; actual fact creation and its independent review remain separate. Exact immutable published helper checkout and current verified owned CPU allocation are required execution premises.','actual_fact_created':False,'actual_GPU_qualification_or_admission_granted':False,'checks':['all four distinct typed context declarations before reading','exact accepted historical metadata readers reused','accepted original snapshot reused without raw payload rehash','native process component never relabeled numerical qualification','outputs created exclusively and durably; audit precedes fact','no self-approval output','default no action','unchanged exact author bytes after independent tests']}
p=D/'root_independent_common_fact_producer_source_review.json'
with p.open('x') as f:json.dump(record,f,indent=2);f.write('\n')
print(json.dumps({'approved':True,'review':str(p),'sha256':sha(p),'tests':out.testsRun}))
