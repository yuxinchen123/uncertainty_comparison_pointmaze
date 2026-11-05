"""Replay the exact two saved-metadata mutations from rejected review381 as author regressions."""
import ast,copy,hashlib,json,os,sys,time,unittest
from pathlib import Path
from unittest.mock import patch
assert os.environ.get('SLURM_JOB_ID')=='6783003' and os.environ['SLURM_STEP_ID'].isdecimal()
w=Path(__file__).resolve().parents[2];m=Path(__file__).resolve().parent;s=w/'10_jax_exploration_platform/scripts/atari_pixel_campaign';S=s
frozen=m/'fdinfo_correction_source_freeze.json';freeze=json.loads(frozen.read_text())
for name,sha in freeze['files'].items():assert hashlib.sha256((w/name).read_bytes()).hexdigest()==sha
sys.path.insert(0,str(s));sys.path.insert(0,str(s/'tests'))
import native_process_evidence as evidence
import test_native_process_evidence as calls
review=m/'run_independent_review.py';text=review.read_text();node=next(node for node in ast.parse(text).body if isinstance(node,ast.ClassDef) and node.name=='IndependentSavedFdinfoBoundaries');source=ast.get_source_segment(text,node)
step=os.environ['SLURM_STEP_ID'];artifacts=m/('fdinfo_exact_381_author_actual_cpu_'+step);artifacts.mkdir();findings=[]
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(name,value):
 p=artifacts/(name+'.json');p.write_text(json.dumps(value,indent=2,sort_keys=True)+'\n');return {'path':str(p),'sha256':sha(p)}
started=time.time();exec(compile(source,str(review),'exec'),globals());suite=unittest.defaultTestLoader.loadTestsFromTestCase(IndependentSavedFdinfoBoundaries);result=unittest.TextTestRunner(verbosity=2).run(suite)
assert not any(name.split('.')[0] in {'ctypes','linux_pidfd_process_control','jax','jaxlib','numpy','torch','flax','optax','jaxtari'} for name in sys.modules)
for name,expected in freeze['files'].items():assert sha(w/name)==expected
value={'kind':'author_exact_rejected_381_saved_fdinfo_mutations','job_id':os.environ['SLURM_JOB_ID'],'step_id':step,'passed':result.wasSuccessful(),'tests_run':result.testsRun,'started_at_seconds':started,'finished_at_seconds':time.time(),'checks_bounded_seconds':30,'original_class_source_sha256':hashlib.sha256(source.encode()).hexdigest(),'original_review_program':{'path':str(review),'sha256':sha(review)},'corrected_source_freeze':{'path':str(frozen),'sha256':sha(frozen)},'findings':findings,'actual_native_process_operations':0,'independent_review_claimed':False,'scientific_credit':0,'admission_granted':False}
p=m/('fdinfo_exact_381_author_'+step+'.json');p.write_text(json.dumps(value,indent=2,sort_keys=True)+'\n');print(json.dumps({'receipt':str(p),'sha256':sha(p),'tests':result.testsRun,'passed':result.wasSuccessful()}));raise SystemExit(0 if result.wasSuccessful() else 1)
