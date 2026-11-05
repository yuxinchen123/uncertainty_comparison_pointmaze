"""Pure affected source-method controls for post-pin timing, with explicitly synthetic clocks and metadata."""
import ast,copy,hashlib,json,os,pathlib,types
D=pathlib.Path('/p/rlprojects/RND_worktrees/2026-11-05-10-14_profile-target-process_native-provider_separate-role-contracts_prior-abi-baseline-only_admission-closed/.review_metadata/task_roster_scope_independent_review')
SRC=D/'preserved_runtime_2_for_affected_review/profile_target_tool_scope.py';raw=SRC.read_bytes()
assert os.environ['SLURM_JOB_ID']=='6825983'and hashlib.sha256(raw).hexdigest()=='37ff07a2ddf5fe24ddca8d20371c94d48970b11667a25ea32741a072929dd297'
scopefn=next(n for n in ast.parse(raw).body if isinstance(n,ast.FunctionDef)and n.name=='scope_class');klass=next(n for n in scopefn.body if isinstance(n,ast.ClassDef));method=next(n for n in klass.body if isinstance(n,ast.FunctionDef)and n.name=='child_lists_with_task_retry')
class Outcome(Exception):
 def __init__(self,record):self.record=record
def require(v,m):
 if not v:raise ValueError(m)
cases=[]
for pin_cost in [2.0,.02,.49,.4901]:
 clock=types.SimpleNamespace(value=0.0)
 def sleep(seconds):clock.value+=seconds
 def live_state(plan,handle,pid,observer):return {'observed_at_seconds':10+clock.value,'descriptor':handle,'pid':pid,'handle_exited':False,'scope':'explicit synthetic final live parent'}
 namespace=dict(copy=copy,time=types.SimpleNamespace(monotonic=lambda:clock.value,sleep=sleep,time=lambda:10+clock.value),MAX_TASK_ROSTER_ATTEMPTS=4,MAX_TASK_ROSTER_RETRY_SECONDS=.5,EnumeratedTaskDisappeared=Outcome,VerifiedTaskRosterChanged=type('Changed',(Exception,),{}),base=types.SimpleNamespace(MAX_EVENTS=100,parsed_identity=lambda x:None),require=require,live_handle_state=live_state)
 exec(compile(ast.Module(body=[method],type_ignores=[]),str(SRC),'exec'),namespace)
 class Synthetic:
  def __init__(self):
   self.task_roster_collection_sequence=0;self.deadline=None;self.calls=0;self.task_roster_records=[];self.directory=pathlib.Path('/explicit-synthetic-not-opened');self.failures=[];self.published=[];self.pid=100;self.producer_handle=42;self.plan={}
   self.observer=types.SimpleNamespace(publish=lambda p,v:self.published.append(copy.deepcopy(v)),receipt=lambda p:{'path':str(p),'sha256':'0'*64})
  def strict_task_child_lists_once(self,pid):
   self.calls+=1
   if self.calls==1:raise Outcome({'failed_task':101,'partial_rows':[],'parent_birth':'1','observed_at_seconds':10.0})
   return {200},[],{'rows':[],'task_roster_before':['100'],'finished_at_seconds':10.01}
  def pin(self,child,parent):
   clock.value+=pin_cost
   return {'pid':child,'parent_pid':parent,'observed_at_seconds':10+clock.value,'scope':'explicit synthetic child snapshot'}
  def task_failure(self,r,error):self.failures.append(str(error))
  def event(self,*a,**kw):pass
 fake=Synthetic();accepted=False;error=None
 try:result=namespace['child_lists_with_task_retry'](fake,100);accepted=True
 except ValueError as problem:error=str(problem)
 expected=pin_cost<=.49
 assert accepted==expected
 if accepted:
  fresh=fake.published[0]['fresh_complete_collection']
  assert fresh['bounded_retry_elapsed_seconds']==clock.value<=.5
  assert fresh['finished_at_seconds']<=fresh['identities']['200']['observed_at_seconds']<=fresh['retained_live_handle']['observed_at_seconds']<=fresh['identities_finished_at_seconds']
  assert result==({200},[])
 else:
  assert error=='fresh task collection and child pins exceeded the one actual retry budget'and fake.failures and not fake.published
 cases.append(dict(synthetic_pin_cost=pin_cost,elapsed_after_pin=clock.value,accepted=accepted,expected=expected,error=error))
record=dict(kind='independent_task_roster_corrected_pin_timing_source_controls',passed=True,actual_job_id=os.environ['SLURM_JOB_ID'],actual_step_id=os.environ['SLURM_STEP_ID'],source_sha256=hashlib.sha256(raw).hexdigest(),cases=cases,old174_gap_now_rejected=True,actual_process_native_GPU_or_target_operations=False,source_approval_pending_actual_fixture_and_saved_reader=True,scope='Only an exact extracted method with synthetic clock/collector/pin/observer sinks. This is not an actual kernel task event or native role proof.')
out=D/'independent_corrected_pin_timing_controls.json';data=(json.dumps(record,indent=2,sort_keys=True)+'\n').encode()
with out.open('xb')as f:f.write(data);f.flush();os.fsync(f.fileno())
print(json.dumps({'path':str(out),'sha256':hashlib.sha256(data).hexdigest(),'step':record['actual_step_id'],'cases':len(cases)}))

