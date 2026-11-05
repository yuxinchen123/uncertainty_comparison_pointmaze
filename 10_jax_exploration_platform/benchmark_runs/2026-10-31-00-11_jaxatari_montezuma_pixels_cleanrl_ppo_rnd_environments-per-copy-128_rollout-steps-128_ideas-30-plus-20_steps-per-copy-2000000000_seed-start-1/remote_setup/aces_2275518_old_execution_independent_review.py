"""Offline audit of the historical ACES execution, without reopening driver payloads."""
import ast
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
import time
import traceback

R = Path(__file__).resolve().parents[1]
I = R.parents[2]
PLAN_PATH = R / 'remote_setup/aces_2275518_exact_six_file_fetch_plan_2026-11-05.json'
JOB, HELPER = '2275518', 'bad2ee3736a1451094d5d1644b36b8cfc2d91c38'
NUM = 'd28e6735e79788f618acb0ebfde8609db1f776af'
DRIVER_SHA = 'dcdf334aeb177c95b3385b797c9b871d03a287895713c6d107f79096b0985aa9'
checks, hashes, binary_stat_checks = [], {}, []
started = time.monotonic()


def check(value, label):
    if not value:raise AssertionError(label)
    checks.append(label)


def sha(path):
    path = Path(path)
    check(path.is_absolute() and path.resolve(strict=True) == path and path.is_file()
          and not path.is_symlink() and path.stat().st_size <= 2 * 1024 * 1024,
          'ordinary small metadata/source only: ' + str(path))
    value = hashlib.sha256(path.read_bytes()).hexdigest();hashes[str(path)] = value
    return value


def read(path):
    sha(path)
    def unique(pairs):
        value = {}
        for key, item in pairs:
            if key in value:raise AssertionError('duplicate JSON key: ' + str(path))
            value[key] = item
        return value
    return json.loads(Path(path).read_bytes(), object_pairs_hook=unique,
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))


def saved(path, value):
    raw = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n').encode()
    with Path(path).open('xb') as stream:stream.write(raw);stream.flush();os.fsync(stream.fileno())
    return {'path': str(path), 'sha256': hashlib.sha256(raw).hexdigest()}


def git_blob(commit, relative):
    check(time.monotonic() - started < 220, 'metadata audit remains inside its bounded work window')
    result = subprocess.run(['git', '-C', str(I), 'cat-file', 'blob', commit + ':' + relative],
                            capture_output=True, timeout=10, check=True)
    return result.stdout


def main():
    check(os.environ.get('SLURM_JOB_ID') == '6783003' and os.environ.get('SLURM_STEP_ID', '').isdigit()
          and os.environ.get('SLURM_CPUS_PER_TASK') == '1' and not os.environ.get('SLURM_STEP_GPUS')
          and not os.environ.get('CUDA_VISIBLE_DEVICES'), 'exact owned CPU-only audit allocation')
    check(sha(PLAN_PATH) == '25112e056985f8ca2e4488b9683ca3d7a45edcd9c6bda6689e9dcd725968bda2', 'exact approved six-file fetch plan')
    plan = read(PLAN_PATH)
    D, P = Path(plan['destination_directory_to_create_only_at_root_fetch']), Path(plan['local_packet'])
    U = Path(plan['reported_site_result']['path']).parent
    check(not (D / 'independent_old_execution_review.json').exists(), 'retain prior review attempts')
    check(plan['job_id'] == JOB and plan['helper_commit'] == HELPER and plan['numerical_commit'] == NUM,
          'exact historical job/helper/numerical source')
    check(plan['native_accounting']['stdout'] == JOB + '|u.sl359075|COMPLETED|0:0|ac045\n'
          and plan['native_accounting']['returncode'] == 0, 'cached native terminal accounting, no remote query')
    snap = read(plan['actual_snapshot']['path'])
    check(sha(plan['actual_snapshot']['path']) == plan['actual_snapshot']['sha256']
          and snap['job_id'] == JOB and snap['sacct'] == plan['native_accounting'], 'fresh snapshot terminal/accounting join')
    small = plan['already_local_small_mirror']['files']
    check(len(small) == 22, 'exact 22 small mirrored records')
    for ref in small:
        path = Path(ref['path']);relative = str(path.relative_to(R))
        check(sha(path) == ref['sha256'] and path.stat().st_size == ref['bytes']
              and snap['mirrored_files'][relative]['sha256'] == ref['sha256'], 'actual mirrored metadata matches snapshot: ' + path.name)
    sub, release = read(P / 'submission_receipt.json'), read(P / 'release_receipt.json')
    check(sha(P / 'submission_receipt.json') == plan['submission_receipt']['sha256']
          and sha(P / 'release_receipt.json') == plan['release_receipt']['sha256']
          and sub['job_id'] == release['job_id'] == JOB and sub['stdout'] == JOB + '\n'
          and sub['automatic_retry_permitted'] is False and release['automatic_retry_permitted'] is False
          and release['returncode'] == 0 and release['submission_receipt_sha256'] == plan['submission_receipt']['sha256'],
          'exact single held submission and reviewed release retained')
    local_ledger = Path(plan['matching_local_owned_ledger']['path'])
    check(local_ledger.read_text().splitlines().count(JOB) == 1 and sub['ledger_path'] == plan['native_ledger_path']
          and sub['submission_row'][0] == JOB and sub['submission_row'][2] == HELPER,
          'separate native/local ACES ledger receipt bindings')
    utility_path = R / 'remote_setup/fetch_aces_2275518_exact_artifacts.py'
    check(sha(utility_path) == '1220f7798185d9d627ba1fcd2a586df7ba87b8c9699afbe64ea909a364aa332f', 'exact independently approved fetch utility source')
    utility = ast.parse(utility_path.read_text())
    embedded = {node.targets[0].id: ast.literal_eval(node.value) for node in utility.body
                if isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name) and node.targets[0].id in {'HASH', 'REMOTE'}}
    invocation = read(R / 'remote_setup/aces_2275518_root_artifact_fetch_intent.json')
    returned = read(R / 'remote_setup/aces_2275518_root_artifact_fetch_return.json')
    publication = invocation['publication']
    check(publication == '978fae7edea6fbf395fb462502bdbc7a1669c232' and invocation['automatic_retry'] is False
          and returned['returncode'] == 0, 'one successful root fetch invocation and source publication')
    for path in (utility_path, PLAN_PATH):
        check(hashlib.sha256(git_blob(publication, str(path.relative_to(I)))).hexdigest() == sha(path),
              'approved utility/plan bytes exist in exact execution publication')
    review = read(R / 'remote_setup/fetch_aces_2275518_independent_corrected_static_review_2026-11-05.json')
    check(review['approved'] is True and review['passed'] is True
          and review['source']['sha256'] == sha(utility_path), 'independent fetch source approval retained')
    fetched, hash_steps = [], []
    for index, item in enumerate(plan['files']):
        rec = read(D / (str(index) + '_fetch_receipt.json'))
        start = read(D / (str(index) + '_hash_process_start.json'))
        proof = json.loads(rec['hash_stdout'])
        path = Path(item['local_destination'])
        check(rec['role'] == item['role'] and rec['local_path'] == str(path) and rec['verified'] is True
              and rec['transport_returncode'] == rec['hash_returncode'] == 0 and rec['hash_stderr'] == ''
              and rec['no_job_action'] is True and 0 <= rec['elapsed_seconds'] <= 300,
              'actual completed transport/disk hash: ' + item['role'])
        check(rec['hash_command'][-4] == embedded['HASH'] and rec['hash_command'][-3:] == [str(path), item['expected_sha256'], str(D / (str(index) + '_hash_process_start.json'))]
              and rec['hash_command'][:5] == ['srun', '--jobid=6783003', '--overlap', '--exact', '--immediate=5']
              and rec['hash_lock_nonblocking'] is True and rec['hash_step_limit_seconds'] <= 240,
              'exact approved bounded hash program and matching payload/start record')
        check(proof['sha256'] == item['expected_sha256'] and proof['bytes'] == rec['remote_identity']['bytes']
              and proof['job_id'] == start['job_id'] == '6783003' and proof['step_id'] == start['step_id']
              and start['uid'] == 2618919 and start['hash_process_started'] is True and start['path'] == str(path)
              and ('/job_6783003/step_' + start['step_id'] + '/') in start['cgroup']
              and proof['image_loaded_or_executed'] is False, 'actual owned hash step/PID/UID/cgroup evidence')
        current = path.stat();identity = proof['identity']
        check(path.is_file() and not path.is_symlink() and path.resolve(strict=True) == path
              and all(getattr(current, key) == identity[key] for key in ('st_dev', 'st_ino', 'st_size', 'st_mtime_ns', 'st_uid'))
              and stat.S_IMODE(current.st_mode) == 0o440 and current.st_ctime_ns >= identity['st_ctime_ns'],
              'payload identity retained after approved readonly chmod transition')
        check(rec['remote_identity']['source_identity_unchanged'] is True and rec['remote_identity']['image_executed'] is False
              and rec['remote_identity']['path'] == item['remote_path'], 'remote exact ordinary source path/identity')
        if index < 4:
            check(sha(path) == item['expected_sha256'], 'small raw log bytes independently checked')
        else:
            check(item['expected_size_bytes'] == proof['bytes'] == 96284520 and proof['sha256'] == DRIVER_SHA,
                  'prior successful driver byte hash and exact size, without a repeated read')
            binary_stat_checks.append({'path': str(path), 'sha256_from_completed_owned_hash': proof['sha256'],
                'bytes': current.st_size, 'current_inode': current.st_ino, 'current_mtime_ns': current.st_mtime_ns,
                'current_mode': '0o440', 'binary_reopened_or_rehashed_by_review': False})
        hash_steps.append('6783003.' + start['step_id']);fetched.append({'role': item['role'], 'step_id': start['step_id'],
            'elapsed_seconds': rec['elapsed_seconds'], 'sha256': proof['sha256'], 'bytes': proof['bytes']})
    check(len(fetched) == 6 and len(set(hash_steps)) == 6, 'all six fetch/hash records are distinct and complete')
    summary = read(D / 'fetch_summary.json')
    check(summary['all_six_verified'] is True and summary['verified_files'] == summary['requested_files'] == 6
          and summary['worker_admission'] is False and summary['scientific_credit'] == 0, 'six-file summary is honest zero credit')
    # Query only exact completed steps already recorded in this campaign's owned monitor allocation.
    check((R / 'slurm/submitted_jobids_uva.txt').read_text().splitlines().count('6783003') == 1,
          'CPU accounting parent is uniquely campaign-owned before exact step query')
    command = ['sacct', '-n', '-P', '-j', ','.join(hash_steps),
               '--format=JobIDRaw,State,ExitCode,ElapsedRaw,AllocTRES,NodeList']
    query = subprocess.run(command, capture_output=True, timeout=30)
    for name, raw in [('independent_hash_accounting_stdout.raw', query.stdout), ('independent_hash_accounting_stderr.raw', query.stderr)]:
        with (D / name).open('xb') as stream:stream.write(raw);stream.flush();os.fsync(stream.fileno())
    saved(D / 'independent_hash_accounting_return.json', {'command': command, 'returncode': query.returncode,
        'queried_only_recorded_owned_steps': True, 'automatic_retry_permitted': False})
    check(query.returncode == 0 and query.stderr == b'', 'exact owned CPU accounting query completed')
    rows = [row.split('|') for row in query.stdout.decode().splitlines() if row]
    check({row[0] for row in rows} == set(hash_steps) and len(rows) == 6
          and all(row[1:3] == ['COMPLETED', '0:0'] for row in rows), 'all six actual disk hash steps completed normally')
    site_plan, result = read(P / 'site_validation_plan.json'), read(U / 'result.json')
    check(site_plan['helpers']['source_commit'] == HELPER and site_plan['source_commit'] == NUM
          and result['passed'] is True and result['completed_at_utc'] == '2026-11-02T11:11:16.278032+00:00',
          'old source-specific completed site declaration')
    for commit, entries in [(HELPER, site_plan['helpers']['files']), (NUM, site_plan['numerical_source']['files'])]:
        for relative, expected in entries.items():
            check(hashlib.sha256(git_blob(commit, relative)).hexdigest() == expected, 'declared historical Git source blob: ' + relative)
    native_root = plan['native_data_unit']
    def check_native_ref(value, local):
        check(value['path'] == native_root + '/' + local.name and value['sha256'] == sha(local), 'native/local raw reference join: ' + local.name)
    check(result['plan']['sha256'] == sha(P / 'site_validation_plan.json') == sub['plan_sha256'], 'actual result/plan/submission source join')
    parent = read(U / 'parent_owner.json')
    stages = {}
    for role in site_plan['child_order']:
        owner, proc, terminal = read(U / (role + '_owner.json')), read(U / (role + '_process.json')), read(U / (role + '_terminal.json'))
        check(terminal['kind'] == role and terminal['job_id'] == JOB and terminal['step_id'] == '0'
              and type(terminal['exit_code']) is int and terminal['exit_code'] == 0
              and terminal['pid'] == owner['pid'] == proc['pid']
              and terminal['process_start_time'] == owner['process_start_time'] == proc['process_start_time']
              and terminal['parent_pid'] == proc['parent_pid'] == parent['pid']
              and terminal['parent_start_time'] == proc['parent_start_time'] == parent['process_start_time'],
              'exact old child/owner/process/terminal/parent identity: ' + role)
        check(terminal['command'] == [site_plan['python'], '-I', site_plan['helpers']['source_root'] +
            '/10_jax_exploration_platform/scripts/atari_pixel_campaign/aces_slurm_site_validation.py',
            'child', '--directory', plan['native_packet'], '--kind', role], 'exact declared child source command: ' + role)
        for field in ('owner', 'process', 'terminal'):
            check_native_ref(result['children'][role][field], U / (role + '_' + field + '.json'))
        check(proc['source_commit'] == NUM and proc['configuration'] == site_plan['configuration']
              and proc['runtime_environment'] == site_plan['runtime_environment'], 'actual child original source/configuration/runtime: ' + role)
        log = next(item for item in plan['files'] if item['role'] == role + '_raw_child_log')
        check(terminal['log']['path'] == log['remote_path'] and terminal['log']['sha256'] == log['expected_sha256'], 'actual source child log binding: ' + role)
        stages[role] = {'pid': terminal['pid'], 'birth': terminal['process_start_time'], 'exit_code': 0}
    log_text = {role: Path(plan['files'][index]['local_destination']).read_text() for index, role in enumerate(site_plan['child_order'])}
    check(log_text['identity'] == 'null\n' and '48 passed in 214.05s' in log_text['learner'], 'identity and original learner raw logs')
    renderer = [json.loads(line) for line in log_text['renderer'].splitlines() if line.startswith('{')]
    check(renderer[-1] == {'rooms': 33, 'cases': 132, 'device': 'cuda:0', 'checks': {'cache': 'passed', 'palette': 'passed', 'fused': 'passed'}, 'synthetic_pixel_cases': 11}
          and all(row['cases'] == 132 and row['status'] == 'passed' for row in renderer[:-1]), 'renderer actual raw room/case/check coverage')
    check(log_text['probe'].endswith('null\n'), 'original probe raw return retained')
    identity, probe = read(U / 'identity_result.json'), read(U / 'probe_result.json')
    for role, record in [('identity', identity), ('probe', probe)]:
        check(record['completed'] is True and record['pid'] == stages[role]['pid']
              and record['source_commit'] == NUM and record['configuration'] == site_plan['configuration'], 'old actual result source/process/configuration: ' + role)
        device = record['identity']
        check(device['pid'] == record['pid'] and device['cuda_device_count'] == 1 and device['jax_local_hardware_id'] == 0
              and set(device['api_return_codes']) == {'cuDeviceGetCount', 'cuDeviceGet', 'cuDeviceGetUuid_v2', 'cuDeviceGetPCIBusId'}
              and all(type(code) is int and code == 0 for code in device['api_return_codes'].values()),
              'historical singleton device API evidence, explicitly not a current-context proof')
        capture = record['driver_capture']
        check(capture['pid'] == record['pid'] and capture['bytes'] == 96284520
              and capture['captured']['sha256'] == capture['original']['sha256'] == DRIVER_SHA, 'actual same-process captured driver metadata: ' + role)
        check(record['allocation']['physical_gres_mapping_verified'] is False
              and record['allocation']['kernel_device_enforcement_verified'] is False
              and record['allocation']['logical_allocation_agreement_verified'] is True, 'configured-isolation trust is not physical/kernel attestation')
    check(identity['identity']['gpu_uuid'] == probe['identity']['gpu_uuid']
          and identity['identity']['pci_address'] == probe['identity']['pci_address'], 'old singleton UUID/PCI agreement across source children')
    check(probe['iteration'] == 1 and probe['finite_state'] == {'passed': True, 'floating_leaves_checked': 112, 'scope': 'complete updated PixelTrainState'}
          and probe['dtypes'] == {'parameters': ['float32'], 'target': ['float32'], 'optimizer_first': ['float32'], 'optimizer_second': ['float32'],
              'observation_statistics': ['float64'], 'reward_statistics': ['float64']}, 'one full original update, finite leaves and original precision')
    libraries = probe['loaded_libraries']
    check(len(libraries['verified_package_libraries']) == 8
          and not any('xla_cuda_plugin' in name for name in libraries['verified_package_libraries']), 'eight package images omit mapped JAX plugin proof')
    for field in ('workers_permitted', 'campaign_claims_permitted'):
        check(result[field] is False, 'historical result grants no work or campaign admission')
    for field in ('scientific_hypothesis_credit', 'comparison_credit', 'idea_credit', 'admission_credit'):
        check(type(result[field]) is int and result[field] == 0, 'old site exact zero credit: ' + field)
    return {'kind': 'independent_historical_aces_2275518_execution_raw_review', 'passed': True,
        'declared_old_profile_execution_passed': True, 'site_job_id': JOB, 'site_node': 'ac045',
        'helper_commit': HELPER, 'numerical_commit': NUM, 'mirrored_metadata_records_verified': 22,
        'fetch_hash_receipts_verified': 6, 'actual_owned_hash_steps': hash_steps, 'hash_step_accounting': rows,
        'driver_binaries_reopened_or_rehashed_by_this_review': False, 'binary_stat_checks': binary_stat_checks,
        'native_ledger_scope': 'native ledger path/submission/adoption and exact local member plus successful cached native snapshot; no remote ledger reread',
        'learner_tests': {'passed': 48, 'seconds_from_actual_log': 214.05},
        'renderer': renderer[-1], 'probe_complete_updates': 1, 'finite_state_leaves_checked': 112,
        'known_warnings': 'game-side int64 truncation warning retained in renderer/probe logs; actual learner statistics recorded float64',
        'device_identity_scope': 'same-process singleton cuDeviceGet* UUID/PCI evidence only',
        'current_CUDA_context_proof': False, 'cuCtxGetCurrent_or_cuCtxGetDevice_recorded': False,
        'mapped_JAX_plugin_package_begin_and_completion_proof': False,
        'full_portable_backend_image_coverage': False,
        'missing_portable_images': list(libraries['verified_package_libraries']),
        'physical_gres_mapping_verified': False, 'kernel_device_enforcement_verified': False,
        'configured_isolation_trust_explicit': True,
        'current_selected_or_adaptive_source_qualified': False, 'worker_admission': False,
        'profile_admission': False, 'new_source_qualification': False,
        'scientific_credit': 0, 'comparison_credit': 0, 'case_credit': 0, 'idea_credit': 0,
        'profile_credit': 0, 'packing_credit': 0, 'final_training_credit': 0,
        'no_historical_backfill_permitted': True, 'prior_2274632_failure_not_relabelled': True,
        'scope': 'Actual logs and old declared source/metadata execution only; not current-context, complete portable backend, selected/adaptive, profile or worker qualification.'}


D = None
try:
    # Read only the destination string before the bounded owned-compute audit starts.
    D = Path(json.loads(PLAN_PATH.read_bytes())['destination_directory_to_create_only_at_root_fetch'])
    payload = main()
except BaseException as error:
    payload = {'kind': 'independent_historical_aces_2275518_execution_raw_review', 'passed': False,
               'error': repr(error), 'traceback': traceback.format_exc(), 'worker_admission': False, 'scientific_credit': 0}
payload.update({'schema_version': 1, 'review_job_id': os.environ.get('SLURM_JOB_ID'),
    'review_step_id': os.environ.get('SLURM_STEP_ID'), 'review_pid': os.getpid(),
    'elapsed_seconds': time.monotonic() - started, 'requested_memory_mib': 512,
    'hard_slurm_time_limit_seconds': 300, 'checks': checks, 'small_raw_sha256': hashes,
    'review_program': {'path': str(Path(__file__).resolve()), 'sha256': sha(Path(__file__).resolve())}})
output = D / 'independent_old_execution_review.json'
ref = saved(output, payload)
print(json.dumps({'passed': payload['passed'], 'receipt': ref, 'review_step': payload['review_step_id'],
                  'checks': len(checks), 'error': payload.get('error')}))
raise SystemExit(0 if payload['passed'] else 1)
