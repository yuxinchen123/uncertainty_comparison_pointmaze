"""Independently review only the exact published runtime-pin/closure update."""
import ctypes  # Own-process standard-library bootstrap; no later library operation is allowed.
import ast,hashlib,importlib.util,io,json,os,subprocess,sys,time,unittest
from pathlib import Path
D=Path(__file__).resolve().parent;W=D.parents[1]
RUNTIME='2aa6df92f7d9da20e2fbc1d15a20df16ba10ba1e';PUBLICATION='cfd5fb82e06cc57c08c56b05d4105390570066f7'
HELPER=Path('/p/rlprojects/RND_worktrees/2026-11-05-13-52_profile-qualification-helper_source-2aa6df92f7d9_h100-helpers-45_a4500-helpers-36_gpu-count-0')
PREFIX='10_jax_exploration_platform/scripts/atari_pixel_campaign/'
started=time.monotonic();checks=[];os.environ['GIT_OPTIONAL_LOCKS']='0'
def need(value,label):
    if not value:raise ValueError(label)
    if time.monotonic()-started>285:raise TimeoutError('independent metadata review exceeded285seconds')
    checks.append(label)
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def git(root,*args):return subprocess.check_output(['git','-C',str(root),*args],env={**os.environ,'GIT_OPTIONAL_LOCKS':'0'},stderr=subprocess.PIPE,timeout=30)
def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module
need(os.environ.get('SLURM_JOB_ID')=='6825983' and os.environ.get('SLURM_STEP_ID','').isdecimal(),'current ownedCPU numeric step')
freeze=D/'final_source_freeze.json';need(sha(freeze)=='8d4f0cfc0b6ed2e179d997aa9cacda557cd5a6b1fdcded4a0020bdd1231cf53a','exact held fourfile freeze')
f=json.loads(freeze.read_text())
for path,record in f['files'].items():need(sha(path)==record['sha256'] and Path(path).stat().st_size==record['bytes'],'exact frozen bytes before '+Path(path).name)
def audit(event,args):
    if event=='import' and args[0].split('.')[0] in {'jax','jaxlib','numpy','torch','flax','optax','jaxtari','cupy','scipy'}:raise ValueError('no numerical imports')
    if event in ('ctypes.dlopen','ctypes.dlsym','os.system','os.exec','os.fork') or event.startswith('socket.') and event!='socket.gethostname':raise ValueError('no native, network or nonmetadata operation')
    if event=='subprocess.Popen' and (not isinstance(args[1],(list,tuple)) or args[1][0]!='git'):raise ValueError('only local metadata Git children are allowed')
    if event=='open' and isinstance(args[0],(str,bytes)) and os.fsdecode(args[0]).startswith('/dev/') and os.fsdecode(args[0])!='/dev/null':raise ValueError('no device access')
sys.addaudithook(audit)
git(W,'merge-base','--is-ancestor',RUNTIME,PUBLICATION);need(True,'actual runtime contained in actual supplied published commit')
git(W,'merge-base','--is-ancestor',PUBLICATION,'refs/remotes/origin/merge-atari-cleanrl');need(True,'actual publication in recorded origin branch; no fetch')
need(git(HELPER,'rev-parse','HEAD').decode().strip()==RUNTIME,'actual clean helper HEAD exactly published runtime')
need(git(HELPER,'status','--porcelain','--untracked-files=all')==b'','actual dedicated helper remains entirely clean')
builders=W/'.review_metadata/profile_target_operational_preparation'
modules=[load(builders/name,'independent_op_pin_'+str(index)) for index,name in enumerate((
    'derive_one_copy_qualification_request.py','derive_two_copy_qualification_request.py','start_direct_profile_qualification_once.py'))]
need(all(module.RUNTIME==RUNTIME for module in modules),'all three actual source pins match published runtime')
q=modules[0].runtime(HELPER)
profile=q.adjacent({'helpers':{'files':{PREFIX+'profile_selected.py':sha(HELPER/PREFIX/'profile_selected.py')}}},'profile_selected')
receipts=[]
for configured in (False,True):
    receipt=profile.helper_receipt(configured=configured,direct=not configured,owned_processes=configured,
        direct_profile=not configured,configured_native=configured,configured_prerequisites=configured,profile_target=True)
    need(receipt['source_commit']==RUNTIME and receipt['source_root']==str(HELPER)
        and len(receipt['files'])==(36 if configured else 45) and PREFIX+'profile_target_tool_scope.py' in receipt['files'],
        'actual full source/root/newscope helper closure '+('AA36' if configured else 'H45'))
    receipts.append(receipt)
tests=load(D/'test_runtime_commit_and_closure.py','independent_op_pin_focused_tests')
output=io.StringIO();result=unittest.TextTestRunner(stream=output,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(tests.RuntimeSourceAndClosureTests))
need(result.wasSuccessful() and result.testsRun==5,'all five focused metadata/preservation methods pass')
need(git(W,'diff','--name-only','--','10_jax_exploration_platform')==b'','no runtime/helper/NUM change in operational worktree')
aa='.review_metadata/profile_target_operational_preparation/submit_profile_qualification_held_once.py'
need(git(W,'show','HEAD:'+aa)==(W/aa).read_bytes(),'AA held utility unchanged')
for path,record in f['files'].items():need(sha(path)==record['sha256'],'frozen bytes unchanged after '+Path(path).name)
log=D/'independent_runtime_pin_tests.stdout.raw'
with log.open('x') as stream:stream.write(output.getvalue());stream.flush();os.fsync(stream.fileno())
record={'schema_version':1,'kind':'independent_profile_qualification_runtime_pin_operational_source_review','passed':True,'approved':True,'independent_of_author':True,
    'source_freeze':{'path':str(freeze),'sha256':sha(freeze)},'files':f['files'],'actual_runtime_commit':RUNTIME,'actual_runtime_publication':PUBLICATION,
    'actual_clean_helper_root':str(HELPER),'actual_helper_receipts':receipts,'tests_run':result.testsRun,'check_count':len(checks),'checks':checks,
    'actual_job_id':os.environ['SLURM_JOB_ID'],'actual_step_id':os.environ['SLURM_STEP_ID'],'seconds':time.monotonic()-started,
    'test_log':{'path':str(log),'sha256':sha(log)},'actual_derivation_or_canonical_preparation_executed':False,
    'tests_scope':'New guard boundary only: explicitly synthetic upstream receipts stop before unchanged four-context gate; actual source receipts/clean checkout checked separately.',
    'science_configuration_source_seed_original_reference_backend_and_fact_statements_unchanged':True,'H_transport_remains_one_copy_only':True,
    'AA_held_utility_unchanged':True,'runtime_helper_NUM_source_changes':False,'GPU_or_native_process_or_remote_actions':False,
    'current_target_qualification_or_admission':False,'scientific_credit':0,'admission_granted':False}
path=D/'independent_source_review.json'
with path.open('x') as stream:json.dump(record,stream,indent=2,sort_keys=True);stream.write('\n');stream.flush();os.fsync(stream.fileno())
print(json.dumps({'passed':True,'review':{'path':str(path),'sha256':sha(path)},'tests':result.testsRun,'checks':len(checks),
    'step':record['actual_step_id'],'seconds':record['seconds']}))
