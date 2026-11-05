"""Synthetic saved-clock/source ordering control only; no process, GPU or native operation."""
import ast,copy,hashlib,json,os,pathlib,time,types
D=pathlib.Path('/p/rlprojects/RND_worktrees/2026-11-05-10-14_profile-target-process_native-provider_separate-role-contracts_prior-abi-baseline-only_admission-closed/.review_metadata/task_roster_scope_independent_review')
SRC=D/'preserved_initial_runtime_50c170/profile_target_tool_scope.py'
assert os.environ['SLURM_JOB_ID']=='6825983'
raw=SRC.read_bytes();assert hashlib.sha256(raw).hexdigest()=='036664a7d4ec1f76cd018a249ffa5094a1fcea21cbc3a734518233a17129c009'
tree=ast.parse(raw);scopefn=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='scope_class');klass=next(n for n in scopefn.body if isinstance(n,ast.ClassDef))
method=next(n for n in klass.body if isinstance(n,ast.FunctionDef)and n.name=='child_lists_with_task_retry')
class Outcome(Exception):
 def __init__(self,record):self.record=record
clock=types.SimpleNamespace(value=0.0)
def require(value,message):
 if not value:raise ValueError(message)
def sleep(seconds):clock.value+=seconds
namespace=dict(copy=copy,time=types.SimpleNamespace(monotonic=lambda:clock.value,sleep=sleep),MAX_TASK_ROSTER_ATTEMPTS=4,MAX_TASK_ROSTER_RETRY_SECONDS=.5,EnumeratedTaskDisappeared=Outcome,VerifiedTaskRosterChanged=type('Changed',(Exception,),{}),base=types.SimpleNamespace(MAX_EVENTS=100,parsed_identity=lambda x:None),require=require)
exec(compile(ast.Module(body=[method],type_ignores=[]),str(SRC),'exec'),namespace)
class Synthetic:
 def __init__(self):
  self.task_roster_collection_sequence=0;self.deadline=None;self.calls=0;self.task_roster_records=[];self.directory=pathlib.Path('/explicitly-synthetic-not-opened');self.failures=[];self.published=[]
  self.observer=types.SimpleNamespace(publish=lambda p,v:self.published.append(copy.deepcopy(v)),receipt=lambda p:{'path':str(p),'sha256':'0'*64})
 def strict_task_child_lists_once(self,pid):
  self.calls+=1
  if self.calls==1:raise Outcome({'failed_task':101,'partial_rows':[],'parent_birth':'1','observed_at_seconds':10.0})
  return {200},[],{'rows':[],'task_roster_before':['100'],'finished_at_seconds':10.01}
 def pin(self,child,parent):
  clock.value+=2.0
  return {'pid':child,'parent_pid':parent,'observed_at_seconds':12.01,'scope':'explicit synthetic identity-order premise'}
 def task_failure(self,record,error):self.failures.append(str(error))
 def event(self,*a,**kw):pass
fake=Synthetic();result=namespace['child_lists_with_task_retry'](fake,100);record=fake.published[0];fresh=record['fresh_complete_collection']
assert result==({200},[])and not fake.failures
assert clock.value>.5 and fresh['bounded_retry_elapsed_seconds']<=.5
assert fresh['identities']['200']['observed_at_seconds']>fresh['finished_at_seconds']
proof=dict(kind='independent_synthetic_pin_work_budget_and_chronology_gap',source_sha256=hashlib.sha256(raw).hexdigest(),actual_job_id=os.environ['SLURM_JOB_ID'],actual_step_id=os.environ['SLURM_STEP_ID'],reproduced_gap=True,runtime_returned_success=True,actual_synthetic_clock_elapsed=clock.value,recorded_bounded_retry_elapsed_seconds=fresh['bounded_retry_elapsed_seconds'],recorded_finished_at_seconds=fresh['finished_at_seconds'],later_identity_observed_at_seconds=fresh['identities']['200']['observed_at_seconds'],all_process_and_native_and_GPU_and_target_operations=False,synthetic_scope='An exact extracted held method executes fake collector/time/pin/metadata sinks. No actual process or directory is inspected, and this is no actual lifecycle or qualification proof.',source_approval=False,required_correction='Measure elapsed and finish only after actual fresh child pins; retain final same-live-parent proof and bind all saved identity times inside that completed interval.')
out=D/'independent_initial_pin_chronology_control.json';data=(json.dumps(proof,indent=2,sort_keys=True)+'\n').encode()
with out.open('xb')as f:f.write(data);f.flush();os.fsync(f.fileno())
print(json.dumps({'path':str(out),'sha256':hashlib.sha256(data).hexdigest(),'step':proof['actual_step_id'],'reproduced':True}))

