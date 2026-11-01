"""Check the default metadata-only terminal-review preparation without reading target progress again."""
import contextlib,ctypes,datetime,hashlib,importlib.abc,io,json,os,pathlib,runpy,subprocess,sys
P=pathlib.Path(__file__).absolute().parent
assert os.environ.get('SLURM_JOB_ID')=='6781231' and os.environ.get('SLURM_STEP_ID','').isdecimal()
names=['raw_qualification_checks.py','raw_review_source_contract.json','review_terminal_raw_then_consumer.py']
images=lambda:{n:hashlib.sha256((P/n).read_bytes()).hexdigest() for n in names}
before=images();blocked=[]
class NoNumerical(importlib.abc.MetaPathFinder):
 def find_spec(self,name,path=None,target=None):
  if name.split('.')[0] in {'jax','jaxlib','numpy','torch','flax','optax','jaxtari','cupy','scipy'}:blocked.append(name);raise RuntimeError('numerical import forbidden')
sys.meta_path.insert(0,NoNumerical())
subprocess.Popen=lambda *a,**k:(_ for _ in ()).throw(RuntimeError('subprocess forbidden in default mode'))
ctypes.CDLL=lambda *a,**k:(_ for _ in ()).throw(RuntimeError('device library forbidden'))
stdout=io.StringIO();sys.argv=[str(P/'review_terminal_raw_then_consumer.py')]
with contextlib.redirect_stdout(stdout):runpy.run_path(sys.argv[0],run_name='__main__')
actual=json.loads(stdout.getvalue());assert actual['source_contract_passed'] is True and actual['scheduler_queries']==actual['actual_result_reads']==0 and actual['independent_review_written'] is False and actual['formal_consumer_called'] is False
assert images()==before and not blocked
value={'schema_version':1,'kind':'terminal_raw_review_preparation_source_and_default_verification','passed':True,'at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'job_id':'6781231','step_id':os.environ['SLURM_STEP_ID'],'pid':os.getpid(),'birth':pathlib.Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19],'cgroup':pathlib.Path('/proc/self/cgroup').read_text(),'actual_default_result':actual,'source_before':before,'source_after':images(),'exact_source_body_only_review_load_and_final_review_predicate_absent':True,'actual_result_reads':0,'scheduler_queries':0,'gpu_actions':0,'network_actions':0,'independent_review_written':False,'scientific_credit':0,'admission_granted':False}
out=P/'raw_then_consumer_preparation_source_verification.json'
with out.open('x') as stream:json.dump(value,stream,sort_keys=True,indent=2);stream.write('\n');stream.flush();os.fsync(stream.fileno())
print(json.dumps({'receipt':str(out),'sha256':hashlib.sha256(out.read_bytes()).hexdigest(),'step_id':os.environ['SLURM_STEP_ID'],'passed':True}))
