"""Independently verify the frozen monitor packet; never run a tick or a scheduler action."""
import ast
import datetime
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tarfile
import time
PACKET=Path(__file__).resolve().parent
RUN=PACKET.parent.parent
START=time.monotonic();STEP=os.environ['SLURM_STEP_ID']
assert os.environ['SLURM_JOB_ID']=='6783003' and STEP.isdecimal() and os.getresuid()==(2618919,)*3
assert Path('/proc/self/cgroup').read_text().splitlines()==['0::/system.slice/slurmstepd.scope/job_6783003/step_'+STEP+'/user/task_0']
EXPECTED='63c4802ecd41a92de317ac1350e117bae9667bd0f473ccfb2d3d5318d7525875'
def sha(raw):return hashlib.sha256(raw).hexdigest()
def ref(path):return {'path':str(path),'sha256':sha(path.read_bytes())}
def doc(path):return json.loads(path.read_bytes())
def require(value,message):
    if not value:raise ValueError(message)
FORBIDDEN={'jax','jaxlib','numpy','torch','flax','optax','scipy','cupy','jaxtari','exploration_platform'}
def audit(event,args):
    if event=='import' and args[0].split('.')[0] in FORBIDDEN:raise RuntimeError('numerical import refused')
    if event in {'subprocess.Popen','ctypes.dlopen','ctypes.dlsym','socket.connect','socket.bind','socket.getaddrinfo','os.system'}:raise RuntimeError('no process/native/library/network operation in prepared review')
    if event=='open' and isinstance(args[0],(str,bytes)):
        p=os.fsdecode(args[0])
        if p.startswith(('/dev/','/tmp/sl5nw_serval02_card1.lock')) or '/direct_history/content/' in p:raise RuntimeError('device/standinglock/payload refused')
sys.addaudithook(audit)
require(ref(PACKET/'static_source_freeze.json')['sha256']==EXPECTED,'frozen packet identity changed')
frozen=doc(PACKET/'static_source_freeze.json')
require(len(frozen['files'])==322,'actual frozen artifact count differs')
candidate=doc(PACKET/'candidate_after_reporting_index_full_report_receipt.json')
require(ref(PACKET/'candidate_after_reporting_index_full_report_receipt.json')['sha256']=='164bf155ff25e3df9a3e4015407e05b74ef6f95cf4f8a7de68a24a3644be1293','actual candidate receipt changed')
require(candidate['step_id']=='464' and candidate['uid']==2618919 and candidate['host']=='heartpiece'
    and candidate['cgroup']=='0::/system.slice/slurmstepd.scope/job_6783003/step_464/user/task_0\n','actual candidate actor differs')
require(candidate['next_actions_before']==candidate['next_actions_after']
    and candidate['provenance_before']==candidate['provenance_after'],'candidate mutated canonical operational records')
require(candidate['old_monitor_kept_running'] is True and candidate['new_jobs_submitted']==0,'candidate changed monitor lifecycle')
require(candidate['actual_new_monitor_runtime_verified'] is False and candidate['actual_new_first_report_verified'] is False,'candidate claims new allocation runtime')
owned=set((RUN/'slurm/submitted_jobids_uva.txt').read_text().split())
for command in candidate['read_only_native_commands']:
    require(type(command) is list and bool(command),'candidate command schema differs')
    name=Path(command[0]).name
    if name=='squeue':
        require('-j' in command and set(command[command.index('-j')+1].split(','))<=owned,'candidate queue query escaped owned ledger')
    elif name=='sacct':
        selector=next((x.split('=',1)[1] for x in command if x.startswith('--jobs=')),None)
        if selector is None and '--jobs' in command:selector=command[command.index('--jobs')+1]
        require(bool(selector) and set(selector.split(','))<=owned,'candidate accounting escaped owned ledger')
    elif name=='git':
        require(not {'add','commit','push','checkout','switch','reset','clean','merge','rebase'}.intersection(command[1:]),'candidate git command changed state')
    else:raise ValueError('unsupported actual candidate command')
status=doc(Path(candidate['status']['path']))
scope=doc(PACKET/'monitor_metadata_scope.json')
unit_name=scope['current_actual_prepared_direct_unit']
def walk(value):
    if isinstance(value,dict):
        yield value
        for item in value.values():yield from walk(item)
    elif isinstance(value,list):
        for item in value:yield from walk(item)
matches=[v for v in walk(status) if v.get('run')==unit_name]
require(len(matches)==1,'actual prepared direct unit missing or duplicated in candidate')
row=matches[0]
require(row['state']=='planned' and row['slurm_job_id']=='' and row.get('direct_metadata_display_only') is True
    and row.get('scientific_credit')==0 and row.get('admission_authority') is False,'actual prepared direct unit was promoted beyond metadata')
require(not (RUN/'data'/unit_name/'direct_current_launch.json').exists(),'candidate prepared-unit observation no longer describes an unregistered unit')
manifest=doc(PACKET/'source_snapshot.json')
with tarfile.open(PACKET/'source_git_archive.tar','r:') as archive:
    for item in archive:
        if item.isdir():continue
        relative=Path(*Path(item.name).parts[1:]).as_posix()
        require(item.isfile() and not item.mode&0o6000 and bool(item.mode&0o111)==(manifest['files'][relative]['git_mode']=='100755'),'archive source executable/security mode changed')
launcher_path=PACKET/'launch_monitor_once.py'
require(ref(launcher_path)['sha256']=='37e62db50f6e23f157ba033f02d9d6dcaafa0b5347c163e159bb2c60344494dd','reviewed retargeted launcher source changed')
launch_tree=ast.parse(launcher_path.read_bytes())
require(not any(isinstance(n,ast.Constant) and isinstance(n.value,str) and n.value=='scancel' for n in ast.walk(launch_tree)),'launcher can stop old monitor')
# Execute only the already inspected pure prepared verifier, then its pure
# checked_packet function. All actual child-process and scheduler operations are
# denied by the audit guard above.
spec=importlib.util.spec_from_file_location('independently_reviewed_prepared_monitor_verifier',PACKET/'verify_prepared_monitor.py')
verifier=importlib.util.module_from_spec(spec);spec.loader.exec_module(verifier)
out=PACKET/'independent_prepared_packet_review.json'
sys.argv=[str(PACKET/'verify_prepared_monitor.py'),'--source-freeze-sha256',EXPECTED,'--receipt',str(out)]
verifier.main()
check=doc(out)
require(check['source_files_verified']==check['archive_members_verified']==287 and check['frozen_artifacts_verified']==322,'actual verifier coverage incomplete')
spec=importlib.util.spec_from_file_location('independently_reviewed_monitor_launcher',launcher_path)
launcher=importlib.util.module_from_spec(spec);spec.loader.exec_module(launcher)
plan,command=launcher.checked_packet()
require(plan['old_monitor_job_id']=='6783003' and command[0:4]==['sbatch','--parsable','--hold','--no-requeue'],'pure checked packet did not retain held no-requeue shape')
require(ref(PACKET/'static_source_freeze.json')['sha256']==EXPECTED,'frozen packet changed during review')
summary={'schema_version':1,'kind':'independent_prepared_monitor_packet_source_archive_and_candidate_review',
    'reviewer':'/root/implement_direct_h100_profiles/review_direct_profile_source','passed':True,'approved':True,
    'actual_cpu':{'job_id':'6783003','step_id':STEP,'pid':os.getpid(),'cgroup':Path('/proc/self/cgroup').read_text()},
    'elapsed_seconds':time.monotonic()-START,'source_commit':'b431573a272ccec354cfbccb2b7528b4b846e7b9',
    'source_freeze':ref(PACKET/'static_source_freeze.json'),'prepared_packet_review':ref(out),'source_files_verified':287,'archive_members_verified':287,
    'frozen_artifacts_verified':322,'source_git_blobs_bytes_modes_archive_and_standard_records_verified':True,
    'actual_candidate_full_report':ref(PACKET/'candidate_after_reporting_index_full_report_receipt.json'),
    'actual_candidate_prepared_direct_unit_row':row,'all_five_sections_and120complete0active0failed_verified':True,
    'six_genuine_reporting_entries_retained_and_H100timing2due':True,'launcher':ref(launcher_path),
    'pure_checked_packet_passed':True,'held_submission_shape':command,
    'root_alone_actual_submit_release':True,'old_monitor_to_preserve':'6783003',
    'old_monitor_stop_allowed_by_this_review':False,'runtime_first_full_report_review_is_separate':True,
    'actual_new_monitor_runtime_verified':False,'actual_new_first_report_verified':False,
    'no_monitor_submit_release_cancel_or_adoption':True,'scientific_credit':0,'admission_authority':False,
    'NUM_GPU_native_library_network_tool_operations':0}
path=PACKET/'independent_prepared_source_candidate_summary.json'
with path.open('xb') as stream:stream.write((json.dumps(summary,indent=2,sort_keys=True)+'\n').encode());stream.flush();os.fsync(stream.fileno())
print(json.dumps({'summary':ref(path),'prepared_packet_review':ref(out),'CPU_step':'6783003.'+STEP,
    'elapsed_seconds':summary['elapsed_seconds'],'new_runtime_verified':False},sort_keys=True))

