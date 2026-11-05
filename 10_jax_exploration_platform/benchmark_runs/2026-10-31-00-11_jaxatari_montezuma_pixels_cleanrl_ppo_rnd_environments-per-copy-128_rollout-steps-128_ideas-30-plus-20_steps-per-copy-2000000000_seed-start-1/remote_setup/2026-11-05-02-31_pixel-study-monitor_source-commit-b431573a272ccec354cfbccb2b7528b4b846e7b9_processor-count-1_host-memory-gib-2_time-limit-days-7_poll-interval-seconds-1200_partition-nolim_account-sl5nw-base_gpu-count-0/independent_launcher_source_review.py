"""Review the real ten-test metadata proof and immutable once-only launcher source."""
import ast
import datetime
import hashlib
import json
import os
from pathlib import Path
import sys
import time
P=Path(__file__).resolve().parent;R=P.parent.parent
START=time.monotonic();STEP=os.environ['SLURM_STEP_ID']
assert os.environ['SLURM_JOB_ID']=='6783003' and STEP.isdecimal() and os.getresuid()==(2618919,)*3
assert Path('/proc/self/cgroup').read_text().splitlines()==['0::/system.slice/slurmstepd.scope/job_6783003/step_'+STEP+'/user/task_0']
def sha(raw):return hashlib.sha256(raw).hexdigest()
def ref(path):return {'path':str(path),'sha256':sha(path.read_bytes())}
def document(path):return json.loads(path.read_bytes())
def require(value,message):
    if not value:raise ValueError(message)
def audit(event,args):
    if event=='import' and args[0].split('.')[0] in {'jax','jaxlib','numpy','torch','flax','optax','scipy','cupy','jaxtari','exploration_platform'}:raise RuntimeError('numerical import refused')
    if event in {'subprocess.Popen','ctypes.dlopen','ctypes.dlsym','socket.connect','socket.bind','socket.getaddrinfo','os.system'}:raise RuntimeError('process/native/network operations refused')
    if event=='open' and isinstance(args[0],(str,bytes)):
        path=os.fsdecode(args[0])
        if path.startswith(('/dev/','/tmp/sl5nw_serval02_card1.lock')) or '/direct_history/content/' in path:raise RuntimeError('device/standinglock/payload refused')
sys.addaudithook(audit)
fpath=P/'monitor_launcher_source_freeze.json'
require(ref(fpath)['sha256']=='e3c483b646245427d10cbd7362f27c816bc7cb9da7a71431666c5469143cb044','launcher freeze changed')
freeze=document(fpath);static=document(P/'static_source_freeze.json')
require(type(freeze['schema_version']) is int and freeze['schema_version']==1 and len(freeze['files'])==324
    and freeze['source_commit']=='b431573a272ccec354cfbccb2b7528b4b846e7b9','launcher freeze shape/source differs')
require(ref(P/'static_source_freeze.json')['sha256']=='63c4802ecd41a92de317ac1350e117bae9667bd0f473ccfb2d3d5318d7525875','prior prepared freeze changed')
for path,value in static['files'].items():
    require(freeze['files'].get(path)==value['sha256'],'launcher freeze omitted or changed a mandatory prepared source')
require(set(freeze['files'])==set(static['files'])|{str(P/'static_source_freeze.json'),str(P/'independent_prepared_packet_review.json')},'launcher freeze contains an unexpected additional authority')
for name in ('static_source_freeze.json','independent_prepared_packet_review.json','launch_monitor_once.py','test_launch_monitor_once.py','run_monitor_launcher_metadata_tests.py','monitor_launch_policy.json'):
    require(freeze['files'][str(P/name)]==ref(P/name)['sha256'],'mandatory launcher or proof byte changed')
tests_ref=ref(P/'root_monitor_launcher_tests.json')
require(tests_ref['sha256']=='9f3b840d28fe69616e63302953fd6208a20b2f9f1edbf85bf2276ae3ae35c5f8','actual root test receipt changed')
tests=document(P/'root_monitor_launcher_tests.json');returned=document(P/'root_monitor_launcher_tests_return.json')
require(tests['passed'] is True and tests['source_files_unchanged'] is True and tests['source_freeze']==ref(fpath)
    and type(tests['tests_run']) is int and tests['tests_run']==10 and tests['failures']==tests['errors']==0,'actual ten-test metadata proof is partial')
require(tests['job_id']=='6783003' and tests['step_id']=='468' and tests['uid']==2618919
    and tests['all_scheduler_replies_are_synthetic'] is True and tests['actual_scheduler_actions']==tests['tool_executions']==tests['gpu_operations']==0,'actual test actor or no-action scope changed')
require(returned['returncode']==0,'root test invocation did not return zero')
elapsed=(datetime.datetime.fromisoformat(tests['finished_at_utc'])-datetime.datetime.fromisoformat(tests['started_at_utc'])).total_seconds()
require(0<elapsed<300,'actual focused tests exceeded five minutes')
tree=ast.parse((P/'test_launch_monitor_once.py').read_bytes())
expected={n.name for c in tree.body if isinstance(c,ast.ClassDef) for n in c.body if isinstance(n,ast.FunctionDef) and n.name.startswith('test_')}
require(len(expected)==10 and len(set(tests['test_ids']))==10 and {x.rsplit('.',1)[1] for x in tests['test_ids']}==expected,'actual test IDs differ from the exact frozen test source')
for name in expected:require(name in tests['test_output'],'actual test output omits a required fixture')
launcher_raw=(P/'launch_monitor_once.py').read_bytes()
require(sha(launcher_raw)=='37e62db50f6e23f157ba033f02d9d6dcaafa0b5347c163e159bb2c60344494dd','launcher source changed after prepared review')
functions={n.name:n for n in ast.parse(launcher_raw).body if isinstance(n,ast.FunctionDef)}
def body(name):return ast.unparse(functions[name])
# The real source and tests are inspected together; these checks keep critical
# publication/ownership/persistence ordering explicit without replaying actions.
for name in ('submit_held_once','prepare_release','release_once'):
    text=body(name)
    require(text.index('check_publication(')<text.index('scheduler(') if name=='submit_held_once' else text.index('check_publication(')<text.index('owned_submission('),'publication check occurs after an action or owned-ID use')
submit=body('submit_held_once')
require(submit.index("publish(PACKET / 'monitor_submission_intent.json'")<submit.index('actual = scheduler(')
    <submit.index('ledgers = [append_member(')<submit.index("publish(PACKET / 'monitor_submission_ledgers_fsynced.json'")
    <submit.index("publish(PACKET / 'monitor_submission_receipt.json'"),'actual ID ledger ordering changed')
require(all(name in submit for name in ('slurm/submitted_jobids.txt','slurm/submitted_jobids_uva.txt','slurm/submissions.tsv')),'three exact ID ledgers not retained')
require('os.fsync(stream.fileno())' in body('append_member') and 'os.fsync(directory)' in body('append_member'),'ledger/file directory durability changed')
require(body('query_held').index('owned_submission(plan)')<body('query_held').index('result = scheduler('),'held query occurs before campaign ownership')
require("'JobHeldUser'" in body('query_held') and "'gres/gpu:0'" in body('query_held') and "'2G'" in body('query_held')
    and "'Restarts': '0'" in body('query_held') and "'Requeue': '0'" in body('query_held'),'held owner/resources/restarts checks changed')
release=body('release_once')
require(release.index("publish(PACKET / 'monitor_release_intent.json'")<release.index("result = scheduler(")
    <release.index("publish(PACKET / 'monitor_release_receipt.json'")<release.index("publish(PACKET / 'monitor_release_complete.json'"),'release intent/receipt/completion ordering changed')
require("'monitor_release_independent_review.json'" in body('check_publication') and 'published_bytes(commit, paths)' in body('check_publication'),'exact fixed-release review and Git publication gate missing')
require("'SBATCH_'" in body('scheduler') and "'automatic_retry_permitted': False" in body('scheduler'),'scheduler inherited-option/no-retry guard missing')
require("'6783003'" in body('identifier') and "value != '6783003'" in body('identifier'),'old-monitor identifier exclusion missing')
main=body('main')
require("else:\n        checked_packet()\n        check_publication(args.publication_commit)" in main
    and "'action_performed': False" in main,'default mode is not publication-only')
require(not any(isinstance(n,ast.Constant) and n.value=='scancel' for n in ast.walk(ast.parse(launcher_raw))),'launcher contains old-monitor stop action')
require(not (P/'monitor_submission_intent.json').exists() and not (P/'monitor_release_intent.json').exists(),'prepared launcher already attempted action during review')
prepared=document(P/'independent_prepared_packet_review.json')
require(prepared['passed'] is True and prepared['source_files_verified']==287 and prepared['frozen_artifacts_verified']==322
    and prepared['actual_new_monitor_runtime_verified'] is False and prepared['actual_new_first_report_verified'] is False,'prepared-review/runtime boundary changed')
value={'schema_version':1,'kind':'independent_persistent_cpu_monitor_launcher_review','passed':True,'approved':True,
    'reviewer_identity':'/root/implement_direct_h100_profiles/review_direct_profile_source','reviewed_at_seconds':time.time(),
    'source_freeze':ref(fpath),'source_commit':freeze['source_commit'],'launcher':ref(P/'launch_monitor_once.py'),
    'prepared_packet_review':ref(P/'independent_prepared_packet_review.json'),'actual_ten_metadata_tests':tests_ref,
    'actual_test_return':ref(P/'root_monitor_launcher_tests_return.json'),
    'actual_independent_CPU_job':'6783003','actual_independent_CPU_step':STEP,'elapsed_seconds':time.monotonic()-START,
    'frozen_mandatory_artifacts':324,'prepared_full_source_archive_checks_repeated':False,'ten_synthetic_tests_repeated':False,
    'held_submission_once_three_ledgers_fsynced_before_receipt':True,'query_requires_prior_exact_campaign_ownership':True,
    'fixed_release_independent_review_and_published_bytes_required':True,'default_publication_mode_has_no_action':True,
    'inherited_SBATCH_removed_no_auto_retry':True,'normal_failure_and_partial_ledger_attempts_retained':True,
    'old_monitor_to_preserve':'6783003','old_monitor_stop_or_adoption_allowed_by_this_review':False,
    'actual_monitor_job_submitted_or_released':False,'actual_new_runtime_verified':False,'actual_new_first_full_report_verified':False,
    'NUM_GPU_device_native_library_network_tool_actions':0,'scientific_credit':0,'admission_authority':False}
out=P/'monitor_launcher_independent_review.json'
with out.open('xb') as stream:stream.write((json.dumps(value,indent=2,sort_keys=True)+'\n').encode());stream.flush();os.fsync(stream.fileno())
print(json.dumps({'review':ref(out),'CPU_step':'6783003.'+STEP,'elapsed_seconds':value['elapsed_seconds'],'passed':True},sort_keys=True))

