"""Review one actual durably ledger-owned held monitor release; no scheduler actions."""
import datetime
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import sys
import time
P=Path(__file__).resolve().parent;R=P.parent.parent;U=R/'data'/P.name
START=time.monotonic();STEP=os.environ['SLURM_STEP_ID']
assert os.environ['SLURM_JOB_ID']=='6783003' and STEP.isdecimal() and os.getresuid()==(2618919,)*3
assert Path('/proc/self/cgroup').read_text().splitlines()==['0::/system.slice/slurmstepd.scope/job_6783003/step_'+STEP+'/user/task_0']
def sha(raw):return hashlib.sha256(raw).hexdigest()
def ref(path):return {'path':str(path),'sha256':sha(path.read_bytes())}
def doc(path):return json.loads(path.read_bytes())
def require(value,message):
    if not value:raise ValueError(message)
def audit(event,args):
    if event=='import' and args[0].split('.')[0] in {'jax','jaxlib','numpy','torch','flax','optax','scipy','cupy','jaxtari','exploration_platform'}:raise RuntimeError('numerical import refused')
    if event in {'subprocess.Popen','ctypes.dlopen','ctypes.dlsym','socket.connect','socket.bind','socket.getaddrinfo','os.system'}:raise RuntimeError('process/native/network operation refused')
    if event=='open' and isinstance(args[0],(str,bytes)) and os.fsdecode(args[0]).startswith(('/dev/','/tmp/sl5nw_serval02_card1.lock')):raise RuntimeError('device/standing lock refused')
sys.addaudithook(audit)
binding_path=P/'monitor_release_binding.json'
require(ref(binding_path)['sha256']=='7df8e08686384b4938910b6fd5f5bbf28d141dda906c2f86de9b7ca62be008c1','actual fixed release binding changed')
binding=doc(binding_path);plan=doc(P/'monitor_replacement_plan.json')
require(binding['kind']=='actual_owned_monitor_fixed_identifier_release_binding' and binding['job_id']=='6821437'
    and binding['command']==['scontrol','release','6821437'] and binding['actual_release_performed'] is False
    and binding['old_monitor_stopped_or_adopted'] is False,'actual fixed release identity changed')
require(binding['plan']==ref(P/'monitor_replacement_plan.json') and binding['submission']==ref(P/'monitor_submission_receipt.json'),'release binding references changed')
source=P/'launch_monitor_once.py'
require(ref(source)['sha256']=='37e62db50f6e23f157ba033f02d9d6dcaafa0b5347c163e159bb2c60344494dd','launcher changed')
spec=importlib.util.spec_from_file_location('independent_actual_monitor_ownership_only',source)
launcher=importlib.util.module_from_spec(spec);spec.loader.exec_module(launcher)
# Only this pure ownership reader is called, never scheduler/query_held/release.
submitted=launcher.owned_submission(plan)
require(submitted['job_id']=='6821437' and submitted['publication']=='365509c6f3d39ea65fde029efc660133f141c8ea','actual submitted identifier/publication changed')
actual=doc(P/'monitor_actual_submission_return.json')
require(actual['returncode']==0 and actual['automatic_retry_permitted'] is False
    and actual['command']==plan['submission_command'],'actual submission command/return changed')
require((P/'monitor_actual_submission_stdout.raw').read_text().strip()=='6821437'
    and (P/'monitor_actual_submission_stderr.raw').read_bytes()==b'','actual native returned identifier is not unique')
sync=doc(P/'monitor_submission_ledgers_fsynced.json')
require(sync['job_id']=='6821437' and sync['submission_row']==submitted['submission_row'],'actual fsync ledger row changed')
require(datetime.datetime.fromisoformat(sync['at_utc'])<datetime.datetime.fromisoformat(submitted['at_utc']),'actual receipt predates ID durability')
expected_paths=[R/'slurm/submitted_jobids.txt',R/'slurm/submitted_jobids_uva.txt',R/'slurm/submissions.tsv']
require([v['path'] for v in sync['ledgers']]==[str(p) for p in expected_paths],'exact three owned ledgers changed')
ledger_rows=[]
for path,saved in zip(expected_paths,sync['ledgers']):
    raw=path.read_bytes();expected='6821437' if path.name!='submissions.tsv' else '\t'.join(submitted['submission_row'])
    require(raw.decode().splitlines().count(expected)==1,'actual owned ledger lacks unique exact monitor member')
    require(ref(path)==saved,'ledger changed after its actual fsync receipt; needs separate append-aware review')
    ledger_rows.append({'ledger':ref(path),'exact_member':expected,'unique_count':1})
job=R/'slurm/jobs/6821437'
for name in ('experiment_background.md','manifest.yaml','config_resolved.yaml','command.txt','progress.json','monitor.sbatch'):
    require((job/name).read_bytes()==(P/name).read_bytes(),'actual per-job standard source differs:'+name)
assignment=doc(job/'assignment.json')
require(assignment['job_id']=='6821437' and assignment['kind']=='monitor' and assignment['unit_id']==P.name
    and assignment['source_commit']=='b431573a272ccec354cfbccb2b7528b4b846e7b9'
    and assignment['actual_new_runtime_independent_review_required'] is True and assignment['old_monitor_to_preserve']=='6783003','actual job assignment/source/handover changed')
query=doc(P/'monitor_release_preparation_query_return.json')
require(binding['held_query']==ref(P/'monitor_release_preparation_query_return.json')
    and query['command']==['scontrol','show','job','-o','6821437'] and query['returncode']==0
    and query['automatic_retry_permitted'] is False,'actual cached held query changed')
require((P/'monitor_release_preparation_query_stderr.raw').read_bytes()==b'','held query emitted stderr')
raw=(P/'monitor_release_preparation_query_stdout.raw').read_text()
require(len(raw.splitlines())==1,'cached exact held query has multiple rows')
fields=dict(re.findall(r'(?:^|\s)([A-Za-z][A-Za-z0-9_/:]*)=(\S*)',raw))
wanted={'JobId':'6821437','UserId':'sl5nw(2618919)','JobState':'PENDING','Reason':'JobHeldUser',
    'Requeue':'0','Restarts':'0','TimeLimit':'7-00:00:00','Partition':'nolim','Account':'sl5nw_base','QOS':'normal',
    'CPUs/Task':'1','NumTasks':'1','MinMemoryNode':'2G','TresPerNode':'gres/gpu:0','Command':str(P/'monitor.sbatch')}
require(all(fields.get(k)==v for k,v in wanted.items()) and fields.get('NumCPUs') in {'1','2'}
    and fields.get('NumNodes') in {'1','1-1'} and fields.get('Reservation','') in {'','(null)','None'},'actual cached held resource or ownership row differs')
pairs=[x.split('=',1) for x in fields['ReqTRES'].split(',')]
require(len({k for k,v in pairs})==len(pairs),'cached ReqTRES repeats a key')
tres=dict(pairs)
require(tres.get('cpu')=='1' and tres.get('mem')=='2G' and tres.get('node')=='1'
    and all(not k.startswith('gres/gpu') or v=='0' for k,v in tres.items()),'actual held request contains extra resources')
require(datetime.datetime.fromisoformat(submitted['at_utc'])<=datetime.datetime.fromisoformat(query['started_at_utc'])
    <=datetime.datetime.fromisoformat(query['finished_at_utc'])<=datetime.datetime.fromisoformat(binding['at_utc']),'actual submission/query/binding chronology differs')
for name in ('root_monitor_publication_no_action_return.json','root_monitor_prepare_release_return.json'):
    require(doc(P/name)['returncode']==0,'root actual publication/prepare command failed')
require(not (P/'monitor_release_intent.json').exists() and not (P/'monitor_release_receipt.json').exists(),'actual release already attempted')
review={'schema_version':1,'kind':'independent_actual_owned_monitor_fixed_release_review','passed':True,'approved':True,
    'reviewer_identity':'/root/implement_direct_h100_profiles/review_direct_profile_source','reviewed_at_seconds':time.time(),
    'binding':ref(binding_path),'job_id':'6821437','fixed_release_command':['scontrol','release','6821437'],
    'submission':ref(P/'monitor_submission_receipt.json'),'three_ledgers_fsynced':ref(P/'monitor_submission_ledgers_fsynced.json'),
    'actual_unique_members':ledger_rows,'native_submission_return':ref(P/'monitor_actual_submission_return.json'),
    'actual_held_query':ref(P/'monitor_release_preparation_query_return.json'),
    'actual_held_stdout':ref(P/'monitor_release_preparation_query_stdout.raw'),'actual_held_fields':fields,
    'held_cpu_shape_verified':{'requested_cpus':1,'requested_memory':'2G','nodes':1,'gpus':0,'days':7,'requeue':0,'restarts':0},
    'actual_independent_CPU_job':'6783003','actual_independent_CPU_step':STEP,'elapsed_seconds':time.monotonic()-START,
    'old_monitor_to_preserve':'6783003','old_monitor_stop_or_adoption_allowed':False,
    'actual_release_performed_by_reviewer':False,'fresh_owned_query_still_required_by_root_at_release':True,
    'actual_runtime_and_first_full_report_approval_still_required':True,
    'NUM_GPU_native_library_network_tool_actions':0,'scientific_credit':0,'admission_authority':False}
out=P/'monitor_release_independent_review.json'
with out.open('xb') as stream:stream.write((json.dumps(review,indent=2,sort_keys=True)+'\n').encode());stream.flush();os.fsync(stream.fileno())
fd=os.open(P,os.O_RDONLY|os.O_DIRECTORY)
try:os.fsync(fd)
finally:os.close(fd)
print(json.dumps({'review':ref(out),'CPU_step':'6783003.'+STEP,'elapsed_seconds':review['elapsed_seconds'],'passed':True},sort_keys=True))

