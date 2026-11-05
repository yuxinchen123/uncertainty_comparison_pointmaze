"""Review complete immutable snapshot coverage without reopening any captured payload."""
import datetime
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import socket
import stat
import sys
import time
import traceback
R=Path('/p/rlprojects/RND_worktrees/merge-atari-cleanrl/10_jax_exploration_platform/benchmark_runs/2026-10-31-00-11_jaxatari_montezuma_pixels_cleanrl_ppo_rnd_environments-per-copy-128_rollout-steps-128_ideas-30-plus-20_steps-per-copy-2000000000_seed-start-1')
N='2026-11-05-01-24_accepted-execution-evidence_cpu-snapshot_original-capture-paths-1938_maximum-check-seconds-285_cpu-step-seconds-300_processor-count-1_host-memory-mib-512_gpu-count-0'
D,P=R/'data'/N,R/'remote_setup'/N
START=time.monotonic();STEP=os.environ['SLURM_STEP_ID']
assert os.environ['SLURM_JOB_ID']=='6783003' and STEP.isdecimal() and os.getresuid()==(2618919,)*3
assert Path('/proc/self/cgroup').read_text().splitlines()==['0::/system.slice/slurmstepd.scope/job_6783003/step_'+STEP+'/user/task_0']
assert (R/'slurm/submitted_jobids_uva.txt').read_text().splitlines().count('6783003')==1
assert socket.gethostname().split('.')[0]=='heartpiece'
def sha(raw):return hashlib.sha256(raw).hexdigest()
def ref(path):return {'path':str(path),'sha256':sha(path.read_bytes())}
def document(path):return json.loads(path.read_bytes())
def check(value,message):
    if not value:raise ValueError(message)
FORBIDDEN={'jax','jaxlib','numpy','torch','flax','optax','scipy','cupy','jaxtari','exploration_platform'}
READS=[]
def audit(event,args):
    if event=='import' and args[0].split('.')[0] in FORBIDDEN:raise RuntimeError('numerical import refused')
    if event in {'subprocess.Popen','ctypes.dlopen','ctypes.dlsym','socket.connect','socket.bind','socket.getaddrinfo','os.system'}:
        raise RuntimeError('process/native/library/network work refused in saved proof review')
    if event=='open' and isinstance(args[0],(str,bytes)):
        path=Path(os.fsdecode(args[0]));name=str(path)
        if name.startswith(('/dev/','/tmp/sl5nw_serval02_card1.lock')) or '/direct_history/content/' in name:
            raise RuntimeError('device, standinglock or historical payload read refused')
        if name.startswith('/proc/') and not name.startswith('/proc/self/'):raise RuntimeError('historical live PID query refused')
        if path.is_file() and path.suffix not in {'.json','.py','.pyc','.raw','.txt','.yaml','.md'}:
            raise RuntimeError('binary/nonmetadata read refused: '+name)
        if path.suffix=='.raw' and path.name not in {'direct_controller_stdout.raw','direct_controller_stderr.raw',
            'snapshot_stdout.raw','snapshot_stderr.raw','independent_445_accounting_UTC_stdout.raw','independent_445_accounting_UTC_stderr.raw'}:
            raise RuntimeError('other raw payload open refused: '+name)
        if path.is_file() and path.stat().st_size>16*1024*1024:raise RuntimeError('large payload read refused')
        READS.append(name)
sys.addaudithook(audit)
try:
    raw=(D/'snapshot.json').read_bytes()
    check(len(raw)==4087982 and sha(raw)=='88791d3f7e602523c547650d41bb2147d101a3793db0d969d34872ded8388a0e','actual proof byte identity changed')
    proof=document(D/'snapshot.json');inputs=document(P/'inputs.json')
    check(ref(P/'inputs.json')['sha256']=='9a69ba96aaeb5560fced8e6299674cf9ffa548d1466a062e7bbd8bb1e45c5c0f','input declarations changed')
    check(proof['inputs']==inputs,'proof inputs differ from immutable invocation bytes')
    attempt=document(D/'snapshot_attempt.json');returned=document(D/'snapshot_return.json')
    check(type(returned['returncode']) is int and returned['returncode']==0 and returned['local_wait_timed_out'] is False and returned['no_automatic_retry'] is True,'actual invocation did not complete normally')
    check(attempt['published_commit']=='11faa41c6722f3961fa3b7bed0f93b85e51103d3' and attempt['source_commit']=='f7e8f5c8f0a2d1847fff4f93726fce62205c23aa','actual invocation source/publication changed')
    check(attempt['stdin']==ref(P/'inputs.json') and attempt['single_use'] is True and attempt['maximum_source_seconds']==285 and attempt['maximum_step_seconds']==300,'actual attempt input or budget changed')
    check(attempt['argv'][0]=='srun' and '--mem=512M' in attempt['argv'] and '--time=00:05:00' in attempt['argv'],'actual resources differ')
    check(proof['producer_source_sha256']=='94dc00524bfc5d02ebc2f7b8f2b88444516938fe583967ea16e414a1e2e4e54a','producer source changed')
    producer=proof['producer'];producer_path=Path(producer['argv'][4])
    check(ref(producer_path)['sha256']==proof['producer_source_sha256'],'actual source file changed')
    spec=importlib.util.spec_from_file_location('independent_actual_snapshot_source',producer_path)
    S=importlib.util.module_from_spec(spec);spec.loader.exec_module(S)
    check(type(proof['schema_version']) is int and proof['schema_version']==1 and proof['kind']==S.PROOF_KIND
        and proof['complete'] is True and S.exact(proof['protocol'],S.PROTOCOL),'actual complete proof kind/protocol changed')
    S.zero_credit(proof)
    check(all(type(proof[k]) is int and proof[k]==0 for k in ('throughput_credit','packing_point_credit')),'CPU proof grants timing/packing credit')
    check(proof['previous_gpu_or_raw_gate_cases_reexecuted'] is False and proof['live_training_inputs_validated_by_this_snapshot'] is False
        and proof['cross_host_descriptor_identity_asserted'] is False,'snapshot boundary changed')
    actual=S.validate_complete_file_proof(proof)
    check(actual['distinct_paths']==actual['distinct_physical_files']==2128 and actual['bytes_hashed']==5673876120,'physical proof totals changed')
    check(actual['maximum_seconds']==285 and 0<actual['elapsed_seconds']<180,'actual source proof exceeds declared bound')
    check(producer['hostname']=='heartpiece' and producer['job_id']=='6783003' and producer['step_id']=='445'
        and producer['pid']==1250347 and producer['process_start_time']=='82672102' and producer['uids']==[2618919]*4,'actual producer identity changed')
    check(producer['argv']==attempt['argv'][attempt['argv'].index('/usr/bin/python3'):],'actual source argv differs from recorded command')
    check(producer['argv'][0:4]==['/usr/bin/python3','-I','-S','-B'] and producer['argv'][-2:]==['--maximum-seconds','285'],'actual source flags changed')
    started=datetime.datetime.fromisoformat(attempt['actual_attempted_at_utc']).timestamp()
    ended=datetime.datetime.fromisoformat(returned['at_utc']).timestamp()
    check(started<=proof['created_at_seconds']<=ended and 0<=actual['elapsed_seconds']<=returned['elapsed_seconds']<180,'actual clock/proof chronology differs')
    account_return=document(P/'independent_445_accounting_UTC_return.json')
    check(account_return['returncode']==0 and account_return['environment_override']=={'TZ':'UTC'},'owned accounting command failed')
    check(account_return['argv'][account_return['argv'].index('-j')+1]=='6783003.445','accounting queried another step')
    rows=[line.split('|') for line in (P/'independent_445_accounting_UTC_stdout.raw').read_text().splitlines() if line.strip()]
    check(len(rows)==1,'owned exact step accounting is ambiguous')
    row=rows[0]
    check(row[:6]==['6783003.445','flock','COMPLETED','0:0','heartpiece','1'] and row[7]=='cpu=1,mem=512M,node=1' and row[10]=='105','exact Slurm state/resource/wait changed')
    account_start=datetime.datetime.fromisoformat(row[8]).replace(tzinfo=datetime.timezone.utc).timestamp()
    account_end=datetime.datetime.fromisoformat(row[9]).replace(tzinfo=datetime.timezone.utc).timestamp()
    check(abs(account_start-started)<2 and abs(account_end-ended)<2,'explicit UTC accounting and actual invocation disagree')
    roster=document(P/'packet_roster.json');prelaunch=document(P/'root_prelaunch_review.json')
    check(prelaunch['passed'] is True and prelaunch['packet_roster']==ref(P/'packet_roster.json') and prelaunch['real_inputs']==ref(P/'inputs.json'),'published prelaunch review lost exact packet inputs')
    for name,value in roster['files'].items():
        path=P/name;check(path.stat().st_size==value['bytes'] and ref(path)['sha256']==value['sha256'],'prepared packet bytes changed:'+name)
    check(roster['source_review']['sha256']=='a770e41ee224522eeeff03f41a6cd1f66c76efe662b4a9958ad6bc69f70ccd62','actual source independent approval changed')
    # Rebuild all metadata declarations using the unchanged exact producer, without
    # calling hash_complete or any original GPU/raw gate. The audit denies payload opens.
    with S.ValidationFiles(120,metadata_only=True) as files:
        files.file(producer_path,proof['producer_source_sha256'],keep_bytes=True)
        inventory=S.AcceptedInventory(files)
        inventory.add({'path':str(producer_path),'sha256':proof['producer_source_sha256']},metadata=False,captured=True,physical_role='producer_python_source')
        qualification=S.accepted_role(files,inventory,inputs['qualification_review'],'qualification')
        qr=files.json(inputs['qualification_review'])
        verification=files.json(inputs['qualification_verification'])
        check(inputs['qualification_verification']['sha256']=='202b3f20943fc16d55bdb3f72c089c111e0886dd65ad2ec4ef8cb63c37a5a716','genuine qualification verification changed')
        check(verification['passed'] is True and verification['raw_checks_before_review'] is True
            and verification['unchanged_full_consumer_passed'] is True and verification['independent_review']==inputs['qualification_review']
            and verification['result']==qualification['result'] and verification['source_contract']==qr['source_contract'],'genuine qualification verification is partial')
        inventory.add(inputs['qualification_verification']);inventory.references(verification)
        correctness=[S.accepted_role(files,inventory,v,'correctness',qr,qualification['qualified_asset_receipt']) for v in inputs['correctness_reviews']]
        expected={'qualification':qualification,'correctness':{str(v['configuration']['learner']['n_copies']):v for v in correctness}}
        check(S.exact(proof['accepted'],expected),'proof accepted Q/C1/C2 fields differ from actual immutable metadata')
        check([v['copy_seeds'] for v in correctness]==[[1],[1,2]],'actual independent-copy seed identities changed')
        inventory.process_metadata();inventory.resolve_blobs()
        declared={v['path']:v['sha256'] for v in proof['declared_files']}
        check(declared==inventory.declared,'complete proof declared coverage differs from independently rebuilt input graph')
        check(S.exact(proof['historical_manifests'],inventory.manifests)
            and S.exact(proof['resolved_logical_references'],inventory.resolved_logical_references)
            and S.exact(proof['inline_historical_records'],inventory.inline_historical_records),'captured path/hash mappings or inline historical records changed')
        check(proof['logical_mapping_count']==len(inventory.logical_mappings)==2097,'logical path mapping total changed')
        check(S.exact(proof['frozen_process_ledger_scopes'],inventory.process_ledger_scopes)
            and S.exact(proof['historical_source_annotations'],inventory.historical_source_annotations),'frozen whole-ledger/current-owner/annotation scope differs')
        check(len(inventory.process_ledger_scopes)==3 and len(inventory.historical_source_annotations)==9,'exact Q/C1/C2 ledger and annotation coverage changed')
        annotations={(v['source_annotation']['path'],v['source_annotation']['sha256']) for v in inventory.historical_source_annotations}
        check(len(annotations)==2 and all(v['row_index'] in (9,10,11) and v['entire_ledger_bytes_hashed'] is True
            and v['treated_as_current_execution_dependency'] is False and v['executed_or_qualified_by_snapshot'] is False for v in inventory.historical_source_annotations),'unrelated diagnostic annotations mislabelled')
        check(all(v['whole_frozen_ledger_bytes_hashed'] is True and v['source_annotations_are_current_execution_dependencies'] is False for v in inventory.process_ledger_scopes.values()),'whole-ledger bytes or current owner checks omitted')
        histories=document(P/'capture_inventory.json')
        check(histories['declared_capture_path_total']==1938 and histories['all_original_1938_capture_paths_required'] is True,'prepared original capture scope changed')
        captures={}
        for history in histories['histories']:
            manifest_ref=history['evidence_manifest'] if 'evidence_manifest' in history else history.get('manifest')
            for value in history['captured_files']:
                check(value['path'] not in captures or captures[value['path']]==value['sha256'],'prepared capture path hashes conflict')
                captures[value['path']]=value['sha256']
        manifest_captures={}
        for accepted in (qualification,*correctness):
            manifest_ref=accepted['historical_manifest'];manifest=files.json(manifest_ref)
            check(len(manifest['files'])==693 and len(manifest['git_roots'])==8,'actual original history mappings or source roots missing')
            for value in manifest['files']:
                path=str(Path(manifest_ref['path']).parent/value['relative_path'])
                check(path not in manifest_captures or manifest_captures[path]==value['sha256'],'actual original capture hashes conflict')
                manifest_captures[path]=value['sha256']
        check(len(captures)==1938 and captures==manifest_captures and all(declared.get(k)==v for k,v in captures.items()),'original1938paths omitted or changed')
        aliases={path:v for v in actual['physical_files'] for path in v['aliases']}
        controller=[]
        for (path,digest_value),value in inventory.shared_metadata.items():
            if value['role'] in ('accepted_controller_stdout','accepted_controller_stderr'):
                authority=value['declaration']['authority'];field=value['declaration']['reference_field'][0]
                output=files.json(authority)[field];pinned=aliases[path]
                check(type(output['size_bytes']) is int and output['size_bytes']==pinned['bytes']==pinned['identity']['st_size']
                    and output['sha256']==digest_value==pinned['sha256'],'actual controller declared size/hash differs from complete pinned proof')
                controller.append({'path':path,'sha256':digest_value,'size_bytes':output['size_bytes'],'authority':authority,'field':field})
        check(len(controller)==6,'six genuine controller stream declarations missing')
        # Stat only, on the same CPU host. Never read payload bytes or transfer these
        # CPU device/inode assertions to a GPU host.
        for path,pinned in aliases.items():
            check(S.identity(Path(path).lstat())==pinned['identity'],'same-CPU actual path/identity changed after snapshot:'+path)
        small_bytes=files.bytes_hashed
        for value in files.physical.values():files.check_entry(value)
    check(ref(D/'snapshot.json')['sha256']=='88791d3f7e602523c547650d41bb2147d101a3793db0d969d34872ded8388a0e','proof changed during review')
    now=time.time()
    review={'schema_version':1,'kind':S.REVIEW_KIND,'approved':True,'passed':True,'independent_of_producer':True,
        'snapshot':{'path':str(D/'snapshot.json'),'sha256':'88791d3f7e602523c547650d41bb2147d101a3793db0d969d34872ded8388a0e'},
        'producer_source_sha256':proof['producer_source_sha256'],'complete_coverage_verified':True,'unresolved_anomalies':[],
        'reviewer_identity':'/root/implement_direct_h100_profiles/review_direct_profile_source',
        'reviewed_at_seconds':now,'admission_granted':False,'worker_admitted':False,'scientific_credit':0,
        'case_credit':0,'idea_credit':0,'throughput_credit':0,'packing_point_credit':0,'site_qualification_credit':0,'final_training_credit':0,
        'review_execution':{'owned_job':'6783003','owned_step':STEP,'hostname':socket.gethostname().split('.')[0],
            'pid':os.getpid(),'elapsed_seconds':time.monotonic()-START},
        'actual_producer':producer,'actual_source_complete_proof_elapsed_seconds':actual['elapsed_seconds'],
        'actual_normal_return':ref(D/'snapshot_return.json'),'actual_attempt':ref(D/'snapshot_attempt.json'),
        'actual_UTC_accounting':{'stdout':ref(P/'independent_445_accounting_UTC_stdout.raw'),'command':ref(P/'independent_445_accounting_UTC_return.json')},
        'distinct_physical_files':2128,'distinct_paths':2128,'actual_producer_bytes_hashed_once':5673876120,
        'producer_payload_hashes_recomputed_by_reviewer':False,'same_CPU_post_snapshot_stat_identity_checks':2128,
        'CPU_device_inode_identity_asserted_on_GPU_host':False,'original_captured_paths_verified':1938,
        'original_history_manifests':3,'original_manifest_logical_mappings':2079,'all_logical_mappings':2097,
        'accepted_reviews':proof['inputs'],'genuine_Q_C1_C2_configuration_assets_source_seeds_rebound':True,
        'rebuilt_complete_declaration_graph_equals_proof':True,'whole_frozen_ledgers_rehashed_and_exact_owner_scope_rechecked':3,
        'nonselected_diagnostic_annotation_instances':9,'distinct_historical_annotation_refs':2,
        'historical_annotations_are_execution_or_qualification_dependencies':False,
        'genuine_controller_stream_size_hash_joins':controller,'only_small_metadata_bytes_hashed_by_reviewer':small_bytes,
        'live_numerical_or_GPU_gates_reexecuted':False,'NUM_imports':sorted(FORBIDDEN.intersection(sys.modules)),
        'GPU_native_library_tool_remote_operations':0,'worker_or_formal_profile_admission':False,
        'scope':'Historical complete byte-coverage fact for exact accepted original Q/C1/C2 evidence only. Current live source/assets/backend/CUDA/admission gates remain separate.'}
    check(now>=proof['created_at_seconds'],'review predates producer creation')
    out=P/'independent_snapshot_review.json'
    with out.open('xb') as stream:stream.write((json.dumps(review,indent=2,sort_keys=True)+'\n').encode());stream.flush();os.fsync(stream.fileno())
    print(json.dumps({'review':ref(out),'passed':True,'CPU_step':'6783003.'+STEP,'elapsed_seconds':review['review_execution']['elapsed_seconds'],
        'small_metadata_bytes_hashed':small_bytes,'complete_declared_paths':2128,'original_capture_paths':1938,'payload_bytes_rehashed':0},sort_keys=True))
except BaseException as error:
    failure={'kind':'independent_actual_snapshot_review_failure','step_id':STEP,'error':repr(error),'traceback':traceback.format_exc(),
        'elapsed_seconds':time.monotonic()-START,'source_or_proof_modified':False,'payload_rehash_requested':False}
    path=P/('independent_snapshot_review_failure_'+STEP+'.json')
    with path.open('xb') as stream:stream.write((json.dumps(failure,indent=2,sort_keys=True)+'\n').encode());stream.flush();os.fsync(stream.fileno())
    print(json.dumps(failure),file=sys.stderr)
    raise

