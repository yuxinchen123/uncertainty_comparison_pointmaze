"""Independently check one actual four-context metadata fact without scientific recomputation."""
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import socket
import stat
import subprocess
import sys
import time

P = Path(__file__).resolve().parent
R = P.parent.parent
I = Path('/p/rlprojects/RND_worktrees/merge-atari-cleanrl')
HELPER = Path('/p/rlprojects/RND_worktrees/2026-11-05-11-56_profile-qualification-helper_source-d1469fd063a4_h100-helpers-44_a4500-helpers-35_gpu-count-0')
PREFIX = '10_jax_exploration_platform/scripts/atari_pixel_campaign/'
COMMIT = 'd1469fd063a44167ec56760729eb0beea7ecf7d3'
PUBLICATION = 'fd66474d4cc9c79e5e15a3b6ded388c9d038a77b'
FACT = 'e75b78e9a9b3fd4a857c2d7c5148de15c8800ef14398c5b56b13f5c78db4d3c7'
started = time.monotonic()
queries = []
checks = 0


def require(value, message):
    global checks
    checks += 1
    if not value:raise ValueError(message)


def guard(event, arguments):
    if event == 'import':
        name = arguments[0].split('.')[0]
        require(name not in {'jax','jaxlib','numpy','torch','flax','optax','jaxtari','cupy','scipy','ctypes'},
            'the metadata audit cannot import a numerical or native-library loader')
    if event in ('ctypes.dlopen','ctypes.dlsym','os.system','os.exec','os.fork') or event.startswith('socket.') and event != 'socket.gethostname':
        raise ValueError('the metadata audit refuses native libraries, network and non-Git processes')
    if event == 'subprocess.Popen':
        executable, argv = arguments[:2]
        require(type(argv) is list and len(argv) >= 4 and argv[0] == 'git' and argv[1] == '-C'
            and Path(argv[2]).is_absolute() and argv[3] in {'show','cat-file','rev-parse','status','config','merge-base'},
            'only declared read-only Git source queries are supported')
        queries.append(argv)
    if event == 'open' and isinstance(arguments[0], (str,bytes)):
        name = os.fsdecode(arguments[0])
        if name.startswith('/dev/') and name != '/dev/null':raise ValueError('no device access in metadata review')
        try:
            size = os.stat(name).st_size
        except (FileNotFoundError,NotADirectoryError,PermissionError):return
        require(size <= 16*1024*1024, 'an old raw result, payload or oversized file cannot be opened')


sys.addaudithook(guard)
require(os.environ.get('SLURM_JOB_ID') == '6825983' and re.fullmatch('[0-9]+',os.environ.get('SLURM_STEP_ID','')) is not None,
    'execute on the current exact owned CPU allocation')
cgroup = Path('/proc/self/cgroup').read_text()
require('/job_6825983/step_'+os.environ['SLURM_STEP_ID']+'/' in cgroup,
    'the actual reviewer kernel cgroup must match the exact numeric step')
require((R/'slurm/submitted_jobids_uva.txt').read_text().splitlines().count('6825983') == 1,
    'the current CPU allocation must occur once in the campaign owned ledger')
environment={key:value for key,value in os.environ.items() if not key.startswith('GIT_')}
environment.update(GIT_OPTIONAL_LOCKS='0',GIT_TERMINAL_PROMPT='0')
def git(root,*args):
    result=subprocess.run(['git','-C',str(root),*args],capture_output=True,check=True,timeout=15,env=environment)
    require(result.stderr == b'', 'read-only Git failed')
    return result.stdout

source=HELPER/PREFIX/'configured_profile_prerequisites.py'
raw=source.read_bytes()
require(raw == git(HELPER,'show',COMMIT+':'+PREFIX+source.name), 'use the exact published unchanged metadata reader')
spec=importlib.util.spec_from_file_location('independent_actual_common_profile_fact_metadata',source)
reader=importlib.util.module_from_spec(spec);exec(compile(raw,str(source),'exec'),reader.__dict__)
files=reader.MetadataFiles()
ref=lambda path:files.reference(path)
read=lambda path:files.document(ref(path))
packet_freeze=read(P/'packet_source_freeze.json')
for item in packet_freeze['files']:
    actual=ref(item['path'])
    require(actual['sha256'] == item['sha256'] and Path(item['path']).stat().st_size == item['bytes'],
        'the actual published packet source changed')
    relative=str(Path(item['path']).relative_to(I))
    require(files.raw(actual) == git(I,'show',PUBLICATION+':'+relative), 'the packet source was not published before execution')
require(git(I,'config','--get-all','remote.origin.url') == b'git@github.com:yuxinchen123/uncertainty_comparison_pointmaze.git\n',
    'the configured shared repository origin changed')
require(git(I,'config','--get-all','remote.origin.fetch') == b'+refs/heads/*:refs/remotes/origin/*\n', 'origin fetch mapping changed')
git(I,'merge-base','--is-ancestor',PUBLICATION,'refs/remotes/origin/merge-atari-cleanrl')
intent=read(P/'root_actual_metadata_fact_execution_intent.json')
returned=read(P/'root_actual_metadata_fact_return.json')
request=read(P/'four_context_fact_request_actual_refs.json')
fact_ref=ref(P/'four_context_prerequisite_fact.json');fact=files.document(fact_ref)
audit_ref=ref(P/'four_context_fact_production_audit.json');audit=files.document(audit_ref)
require(fact_ref['sha256'] == FACT and audit['fact'] == fact_ref, 'the actual fact and production audit hash differ')
require(intent['publication_commit'] == PUBLICATION and intent['one_attempt'] is True and intent['root_only_action'] is True
    and intent['current_owned_CPU_monitor'] == '6825983' and returned['returncode'] == 0 and type(returned['returncode']) is int,
    'the actual one-shot published CPU production did not return zero')
argv=intent['argv']
expected=['/usr/bin/python3','-I','-S','-B',str(P/'produce_common_profile_prerequisite_fact.py'),
    '--helper-root',str(HELPER),'--request',str(P/'four_context_fact_request_actual_refs.json'),
    '--proof',fact_ref['path'],'--audit',audit_ref['path'],'--owned-job-id','6825983','--execute']
require(argv[argv.index('/usr/bin/python3'):] == expected and argv[:4] == ['srun','--jobid=6825983','--overlap','--exact']
    and '--time=00:05:00' in argv and '-c1' in argv and '--mem=512M' in argv,
    'actual producer argv or declared resources changed')
require(audit['owned_job_id'] == '6825983' and type(audit['actual_pid']) is int and audit['actual_pid'] > 0
    and audit['actual_uid'] == os.getuid() == 2618919 and audit['hostname'] == 'slurm5'
    and type(audit['owned_step_id']) is str and audit['owned_step_id'].isdecimal()
    and '/job_6825983/step_'+audit['owned_step_id']+'/' in audit['cgroup']
    and audit['producer_source'] == ref(P/'produce_common_profile_prerequisite_fact.py')
    and audit['reader_source'] == {'commit':COMMIT,'path':str(source),'sha256':hashlib.sha256(raw).hexdigest()}
    and audit['request'] == ref(P/'four_context_fact_request_actual_refs.json')
    and audit['independent_review_created'] is False and audit['admission_granted'] is False
    and audit['worker_admitted'] is False and audit['GPU_numerical_qualification'] is False
    and audit['current_physical_identity_verified'] is False and 0 <= audit['elapsed_seconds'] < 285,
    'the actual source/process/step/scope production audit differs')
require((P/'root_actual_metadata_fact_stderr.raw').read_bytes() == b'', 'the actual producer stderr is not empty')
stdout=json.loads((P/'root_actual_metadata_fact_stdout.raw').read_bytes())
require(stdout['fact'] == fact_ref and stdout['independent_review_created'] is False and stdout['admission_granted'] is False,
    'actual stdout differs from the actual produced fact')
require(reader.equal({key:fact[key] for key in reader.PROTOCOL},reader.PROTOCOL)
    and fact['campaign_run_dir'] == str(R) and fact['source_commit'] == reader.SOURCE
    and reader.equal(fact['contexts'],sorted(request['contexts'],key=lambda row:(row['hardware_class'],row['copies'])))
    and reader.equal(fact['initial_snapshot_inputs'],request['initial_snapshot_inputs'])
    and reader.equal(fact['native_process_qualification'],request['native_process_qualification'])
    and fact['admission_granted'] is False and fact['worker_admitted'] is False
    and type(fact['scientific_credit']) is int and fact['scientific_credit'] == 0,
    'the complete produced fact differs from its published exact request or zero-credit protocol')
require(len(fact['contexts']) == 4 and all(type(row['copies']) is int for row in fact['contexts'])
    and {(row['hardware_class'],row['copies']) for row in fact['contexts']} == reader.CONTEXTS,
    'all four exact accepted contexts are mandatory')
proposed=files.document(request['proposed_plan'])
contexts=[reader.reviewed_context(files,row,proposed) for row in fact['contexts']]
aa=next(row for row in contexts if row['hardware_class']=='jaguar03_rtx_a4500' and row['copies']==1)
require(aa['plan'] == request['proposed_plan'] and all(reader.equal(row['target_qualification_inputs'],fact['target_qualification_inputs'])
    for row in contexts if row['hardware_class']=='jaguar03_rtx_a4500'), 'both A4500 original qualification references must remain identical')
target=fact['target_qualification_inputs'];target_review=files.document(target['review'])
require(target['review']['sha256'] == reader.QUALIFICATION_REVIEW and target['result']['sha256'] == reader.QUALIFICATION_RESULT
    and target_review['approved'] is True and target_review['result'] == target['result']
    and target_review['actual_terminal_state'] == 'COMPLETED' and target_review['actual_exit_code'] == '0:0'
    and target_review['admission_granted'] is False, 'the exact original A4500 qualification approval changed')
history=next(row['historical_plan'] for row in contexts if row['hardware_class']=='h100' and row['copies']==1)
require(all(reader.equal(row['historical_plan']['direct_qualification_inputs'],history['direct_qualification_inputs'])
    for row in contexts if row['hardware_class']=='h100'), 'both H100 contexts changed original qualification references')
hq=history['direct_qualification_inputs'];hreview=files.document(hq['review'])
require(hq['review']['sha256']=='8c61ec70059475a479a2e0b395bdc6cd2d5cb2a0e062ae05ef51e066c4a4d01a'
    and hq['result']['sha256']=='76eef4e18c06fef8d86fa704d3f9f00b8ebd78b34ab50e5083fcb0d4ee9bb318'
    and hreview['approved'] is True and hreview['independent_of_producer'] is True
    and hreview['result']==hq['result'] and hreview['actual_returncode']==0
    and hreview['admission_granted'] is False, 'the exact original H100 qualification approval changed')
native=fact['native_process_qualification'];nr=files.document(native['independent_review'])
np=files.document(native['diagnostic_plan']);nresult=files.document(native['diagnostic_result'])
files.document(native['target_contract'])
require(native['independent_review']['sha256']==reader.NATIVE_REVIEW and nr['approved'] is nr['passed'] is True
    and nr['independent_of_producer'] is True and nr['plan']==native['diagnostic_plan'] and nr['result']==native['diagnostic_result']
    and nr['target_contract']==native['target_contract'] and nr['profile_admission_granted'] is False
    and nresult['completed'] is True and nresult['plan']==native['diagnostic_plan'] and nresult['target_contract']==native['target_contract']
    and np['python']==proposed['python'], 'the actual process component links changed or lack independent acceptance')
snapshot=reader.initial_snapshot(files,fact['initial_snapshot_inputs'],history)
require(reader.equal(snapshot,audit['initial_snapshot']) and snapshot['comparisons_previously_validated']==120
    and snapshot['comparisons_recomputed']==0 and snapshot['historical_payloads_rehashed'] is False,
    'the original immutable snapshot or its exact historical namespace changed')
for path,sha256 in audit['metadata_files'].items():files.raw({'path':path,'sha256':sha256})
require(all(audit['metadata_files'].get(path)==sha256 for path,sha256 in files.records.items()
    if path not in {str(P/name) for name in ('packet_source_freeze.json','root_actual_metadata_fact_execution_intent.json',
        'root_actual_metadata_fact_return.json','four_context_prerequisite_fact.json','four_context_fact_production_audit.json')}
    and path in audit['metadata_files']), 'an actual producer metadata reference changed')
require(time.monotonic()-started<285, 'the independent metadata review exceeded its bound')
review={'schema_version':1,'kind':'independent_reviewed_four_context_configured_profile_prerequisites',
    'proof':fact_ref,'approved':True,'passed':True,'independent_of_producer':True,
    'all_four_raw_reviews_verified':True,'exact_historical_consumer_namespaces_verified':True,
    'admission_granted':False,'worker_admitted':False,'unresolved_anomalies':[],
    'source_commit':reader.SOURCE,'runtime_helper_commit':COMMIT,'producer_publication':PUBLICATION,
    'producer_audit':audit_ref,'producer_execution_return':ref(P/'root_actual_metadata_fact_return.json'),
    'contexts':[{key:value for key,value in row.items() if key!='historical_plan'} for row in contexts],
    'initial_snapshot':snapshot,'native_process_qualification':native,'target_qualification_inputs':target,
    'H100_original_qualification_inputs':hq,'scientific_raw_results_recomputed':False,
    'historical_payloads_rehashed':False,'comparisons_recomputed':0,'new_GPU_qualification':False,
    'actual_job_id':'6825983','actual_step_id':os.environ['SLURM_STEP_ID'],'actual_pid':os.getpid(),
    'hostname':socket.gethostname(),'cgroup':cgroup,'checks_passed':checks,'elapsed_seconds':time.monotonic()-started,
    'read_only_Git_queries':queries,'metadata_files':files.records,
    'reviewer_source':ref(__file__),
    'prior_reviewer_guard_failure':ref(P/'independent_fact_output_review_1_return.json'),
    'prior_reviewer_failure_reason':'The historical read-only snapshot labels its own host with socket.gethostname; only that local observation event was added to the strict audit allowlist.',
    'bootstrap_scope':'Python standard-library modules were loaded before the audit hook; no numerical modules, device operations or native-library loader calls were permitted.'}
output=P/'independent_actual_four_context_fact_review.json'
with output.open('xb') as stream:
    stream.write(reader.encoded(review));stream.flush();os.fsync(stream.fileno())
fd=os.open(P,os.O_RDONLY|os.O_DIRECTORY);os.fsync(fd);os.close(fd)
print(json.dumps({'passed':True,'review':ref(output),'checks_passed':checks,'elapsed_seconds':time.monotonic()-started}))
