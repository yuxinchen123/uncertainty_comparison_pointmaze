"""Independent offline review of one completed, zero-credit process diagnostic."""
import ast
import copy
import datetime as dt
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import signal
import stat
import subprocess
import sys
import traceback

R = Path('/p/rlprojects/RND_worktrees/merge-atari-cleanrl/10_jax_exploration_platform/benchmark_runs/2026-10-31-00-11_jaxatari_montezuma_pixels_cleanrl_ppo_rnd_environments-per-copy-128_rollout-steps-128_ideas-30-plus-20_steps-per-copy-2000000000_seed-start-1')
B = '2026-11-04-20-09_nsight-process-observation_jaguar03_gpu-count-1_processor-count-1_host-memory-gib-2_time-limit-minutes-5_python-usr-bin-python3_reservation-sl5nw_168_qos-csresnolim_tool-version-2024.7.1.84_environment-transitions-0'
P, U = R / 'remote_setup' / B, R / 'data' / B
I = R.parents[2]
OUT = P / 'actual_raw_independent_review_v2.json'
COMMIT = '7aff76d9d163d01de805364de917cc558e783535'
UID, JOB = 2618919, '6820179'
checks, raw_files, controls = [], {}, []


def check(value, name):
    if not value:
        raise AssertionError(name)
    checks.append(name)


def sha(path):
    path = Path(path)
    check(path.is_absolute() and path.resolve(strict=True) == path and not path.is_symlink()
          and path.is_file(), 'exact regular file: ' + str(path))
    value = hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            value.update(block)
    raw_files[str(path)] = {'sha256': value.hexdigest(), 'bytes': path.stat().st_size,
                            'mode': oct(stat.S_IMODE(path.stat().st_mode))}
    return value.hexdigest()


def ref(path):
    return {'path': str(path), 'sha256': sha(path)}


def read(path):
    sha(path)
    def unique(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise AssertionError('duplicate JSON key: ' + str(path))
            value[key] = item
        return value
    return json.loads(Path(path).read_bytes(), object_pairs_hook=unique,
                      parse_constant=lambda item: (_ for _ in ()).throw(ValueError(item)))


def reference(value, expected_path):
    check(value == ref(expected_path), 'exact reference: ' + str(expected_path))


def timestamp(value):
    return dt.datetime.fromisoformat(value).timestamp()


def fail_native(*args, **kwargs):
    raise AssertionError('review attempted a process/device/native operation')


def git(*args, cwd=I):
    allowed = {'cat-file', 'rev-parse'}
    check(args and args[0] in allowed, 'read-only Git command is allowlisted')
    result = subprocess.run(['git', '-C', str(cwd), *args], capture_output=True, check=True)
    return result.stdout


def stable(row):
    return {key: row[key] for key in ('pid', 'parent_pid', 'process_group', 'session',
                                    'process_start_time', 'uids', 'cgroup', 'command', 'executable')}


def snapshot(row, cgroup):
    # Independent reconstruction from retained kernel-file bytes; no historical PID access.
    def parse(raw):
        left, fields = raw.rsplit(') ', 1)
        values = fields.split()
        check(values[0] not in {'Z', 'X', 'x'}, 'saved snapshot was live')
        return {'pid': int(left.split(' (', 1)[0]), 'parent_pid': int(values[1]),
                'process_group': int(values[2]), 'session': int(values[3]),
                'process_start_time': values[19]}
    first, last = parse(row['stat_before']), parse(row['stat_after'])
    check(first == last and all(type(row[k]) is type(v) and row[k] == v for k, v in first.items()),
          'independent raw stat reconstruction: ' + str(row['pid']))
    for name in ('status', 'status_after'):
        lines = [line.split()[1:] for line in row[name].splitlines() if line.startswith('Uid:')]
        check(len(lines) == 1 and [int(item) for item in lines[0]] == [UID] * 4
              and row['uids'] == [UID] * 4 and all(type(v) is int for v in row['uids']),
              'independent raw four-UID reconstruction: ' + name + ':' + str(row['pid']))
    raw = row['cmdline'].encode()
    check(raw.endswith(b'\0') and hashlib.sha256(raw).hexdigest() == row['cmdline_sha256']
          and row['command'] == [item.decode() for item in raw[:-1].split(b'\0')],
          'independent raw argv reconstruction: ' + str(row['pid']))
    check(row['cgroup'] == cgroup and '/job_6820179/step_0/' in cgroup,
          'exact actual owned numeric cgroup: ' + str(row['pid']))


def main():
    check(os.environ.get('SLURM_JOB_ID') == '6783003'
          and re.fullmatch('[0-9]+', os.environ.get('SLURM_STEP_ID', ''))
          and os.environ.get('SLURM_CPUS_PER_TASK') == '1'
          and os.environ.get('SLURM_MEM_PER_NODE') in {'512', '2048'}, 'bounded owned CPU context before review')
    check(not os.environ.get('SLURM_STEP_GPUS') and not os.environ.get('CUDA_VISIBLE_DEVICES'),
          'CPU review has no GPU assignment')
    check(not OUT.exists(), 'preserve previous actual review')
    os.pidfd_open = fail_native
    signal.pidfd_send_signal = fail_native
    plan = read(U / 'measurement_plan.json')
    result = read(U / 'result.json')
    check(plan['kind'] == result['kind'] == 'nsight_process_identity_diagnostic'
          and plan['job_id'] is None and plan['gpu_uuid'] is None,
          'diagnostic kind and immutable unresolved preparation')
    check(plan['helpers']['source_commit'] == COMMIT and plan['python'] == '/usr/bin/python3',
          'exact reviewed helper and separate system interpreter')
    check(result['completed'] is True and result['nsight_returncode'] == 0
          and type(result['nsight_returncode']) is int and not (U / 'failure.json').exists(),
          'producer completed normally without failure record')
    for field in ('scientific_credit', 'profile_credit', 'case_credit', 'idea_credit',
                  'environment_credit', 'learner_credit'):
        check(type(plan[field]) is int and plan[field] == 0 and type(result[field]) is int
              and result[field] == 0, 'zero exact diagnostic credit: ' + field)
    for field in ('whole_tool_library_or_device_activity_quantified', 'admission_granted',
                  'worker_admitted', 'physical_gres_mapping_verified'):
        check(plan[field] is False and result[field] is False, 'explicit diagnostic limitation: ' + field)
    check(plan['work'] == {'actual_nsight_profile_command_count': 1, 'cuda_profiler_api_calls_by_payload': 0,
                          'learner_updates': 0, 'numpy_or_jax_or_torch_imports': 0, 'warmup_rollouts': 0}
          and result['numerical_modules'] == [] and result['cuda_profiler_api_called_by_payload'] is False
          and result['same_numerical_profile_not_executed'] is True, 'zero scientific payload work boundary')
    initial = read(P / 'immutable_artifacts.json')['files']
    freeze = read(P / 'launcher_source_freeze.json')['files']
    check(len(initial) == 17 and len(freeze) == 22, 'exact initial and expanded freeze cardinalities')
    for path, expected in initial.items():
        check(sha(path) == expected['sha256'] and raw_files[path]['bytes'] == expected['bytes']
              and raw_files[path]['mode'] == expected['mode'], 'initial bytes/size/mode unchanged: ' + path)
    for path, expected in freeze.items():
        check(sha(path) == expected, 'expanded frozen source unchanged: ' + path)
    before = {path: sha(path) for path in freeze}
    source = Path(plan['helpers']['source_root'])
    check(git('rev-parse', 'HEAD', cwd=source).decode().strip() == COMMIT, 'private helper exact commit')
    for path, expected in plan['helpers']['files'].items():
        check(sha(source / path) == sha(U / 'code' / Path(path).name) == expected
              == hashlib.sha256(git('cat-file', 'blob', COMMIT + ':' + path)).hexdigest(),
              'reviewed Git blob/source/staged source join: ' + path)
    check(result['source'] == plan['helpers'], 'actual result exact reviewed source closure')
    reference(result['plan'], U / 'measurement_plan.json')
    sub, rel = read(P / 'submission_receipt.json'), read(P / 'release_receipt.json')
    owned = read(U / 'owned_launch_inputs.json')
    check(sha(P / 'release_receipt.json') == '97e4574d0e5d4153a72a88393da0a1db2adef4f5a172b75ab6d18f1c1dda87c6'
          and sha(U / 'owned_launch_inputs.json') == '99f1b467d09ec3189e71ae4b16eb3dbc90e3c6cfd3839dab68159f97c2c9ee28',
          'exact actual release and final inputs hashes supplied independently')
    row = [JOB, plan['kind'], COMMIT, str(P / 'job.sbatch')]
    for name in ('submitted_jobids.txt', 'submitted_jobids_uva.txt'):
        check((R / 'slurm' / name).read_text().splitlines().count(JOB) == 1, 'unique owned ID: ' + name)
    ledger = R / 'slurm/submissions.tsv'
    check([line.split('\t') for line in ledger.read_text().splitlines() if line.split('\t')[0] == JOB] == [row],
          'single exact owned submission row')
    for name, value, kind in [('submission', sub, 'owned_nsight_process_diagnostic_submission'),
                              ('release', rel, 'owned_nsight_process_diagnostic_release')]:
        reference(owned[name], P / (name + '_receipt.json'))
        check(value['job_id'] == JOB and value['kind'] == kind and value['returncode'] == 0
              and value['plan_sha256'] == sha(U / 'measurement_plan.json') and value['data_unit'] == str(U),
              'actual successful bound ' + name)
    reference(owned['plan'], U / 'measurement_plan.json')
    check(owned['job_id'] == JOB and owned['submission_row'] == sub['submission_row'] == row,
          'final owned release input exact ledger row')
    fsync_record = read(P / 'submission_ledgers_fsynced.json')
    check(fsync_record['job_id'] == JOB and fsync_record['submission_row'] == row
          and timestamp(fsync_record['at_utc']) < timestamp(sub['at_utc']) < timestamp(rel['at_utc']),
          'durable ID/row record before submission receipt and actual release')
    release_done = read(P / 'release_completed.json')
    reference(release_done['release'], P / 'release_receipt.json')
    reference(release_done['owned_launch_inputs'], U / 'owned_launch_inputs.json')
    check(release_done['released_once'] is True and release_done['job_id'] == JOB, 'one completed actual release')
    raw_release = read(P / 'actual_release_return.json')
    check(raw_release['command'] == ['scontrol', 'release', JOB] and raw_release['returncode'] == 0
          and raw_release['automatic_retry_permitted'] is False
          and timestamp(raw_release['finished_at_utc']) < timestamp(rel['at_utc']), 'cached exact one-release command')
    release_review = read(P / 'release_independent_review.json')
    check(release_review['passed'] is True, 'genuine fixed-release independent review retained')
    # Read the exact reviewed commit blobs, without running packet or tool code.
    publication = rel['publication']
    check(publication == 'dcc4614a6cc96a7710eeb1ee48220fed4d58c9c4', 'exact release publication')
    for path in (P / 'release_binding.json', P / 'release_independent_review.json', P / 'launch_diagnostic_once.py'):
        relative = str(path.relative_to(I))
        check(hashlib.sha256(git('cat-file', 'blob', publication + ':' + relative)).hexdigest() == sha(path),
              'published release bytes: ' + relative)
    capabilities = read(U / 'process_control_capabilities.json')
    check(capabilities['python'] == '/usr/bin/python3'
          and capabilities['sys_version'] == '3.10.12 (main, May 27 2025, 17:12:29) [GCC 11.4.0]'
          and capabilities['required_apis_callable'] == {'os.pidfd_open': True, 'signal.pidfd_send_signal': True}
          and capabilities['kernel_support_tested_by_this_check'] is False,
          'actual target system Python and API-presence evidence')
    reference(result['process_control_capabilities'], U / 'process_control_capabilities.json')
    tool = plan['profile_tool']
    check(result['tool'] == tool and sha(tool['path']) == tool['sha256'], 'actual installed tool bytes remain exact')
    version = read(U / 'tool_version_return.json')
    check(version == {'command': [tool['path'], '--version'], 'returncode': 0, 'stderr': '',
                      'stdout': tool['version'] + '\n'}, 'actual recorded tool version and exact wait')
    tree = ast.parse((U / 'code/diagnose_nsight_process_tree.py').read_text())
    functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
    run_text = ast.get_source_segment((U / 'code/diagnose_nsight_process_tree.py').read_text(), functions['run'])
    check(run_text.index('require_process_control_apis()') < run_text.index('actual_version = subprocess.run')
          and run_text.index('identity.receipt(tool["path"])') < run_text.index('actual_version = subprocess.run')
          and run_text.count('identity.receipt(tool["path"])') == 2
          and run_text.rindex('identity.receipt(tool["path"])') < run_text.index('child = subprocess.Popen'),
          'exact published source requires API preflight and tool hash before/after version before profile')
    # Import only the exact hash-verified stdlib evidence reader. Its live operations are replaced by refusals.
    helper_path = U / 'code/nsight_process_identity.py'
    spec = importlib.util.spec_from_file_location('independent_nsight_identity_saved_only', helper_path)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    helper.ProcReader = fail_native
    helper.OwnedProcesses = fail_native
    for name in ('open_process_handle', 'handle_exited', 'signal_handle', 'close_process_handle',
                 'parent_handle', 'validate_live_observation'):
        setattr(helper, name, fail_native)
    intent, claim = read(U / 'profile_process_intent.json'), read(U / 'capture_identity.json')
    observation = read(U / 'capture_process_observation.json')
    launch = read(U / 'capture_actual_launch.json')
    root = read(U / 'profiler_process.json')
    helper.validate_saved_intent(U, plan, intent)
    check(intent['job_id'] == JOB and intent['step_id'] == '0', 'saved challenge exact actual job and numeric step')
    rebuilt = helper.validate_observation(observation, claim, intent['capture_command'], intent['parent'])
    check(rebuilt['path'] == [2433894, 2433895, 2433901] and observation['root_pid'] == 2433871
          and rebuilt['capture']['pid'] == result['capture_pid'] == 2433901
          and claim['process_start_time'] == result['capture_start_time'] == '81355554',
          'pure exact staged reader reconstructs actual positive ancestry and capture birth')
    expected_commands = helper.commands(U, plan)
    check(intent['waiter_command'] == expected_commands[0] and intent['capture_command'] == expected_commands[1]
          and result['command'] == read(U / 'executed_command.json') == helper.nsight_command(U, plan),
          'exact actual source/tool/flag/role command routing')
    reference(claim['intent'], U / 'profile_process_intent.json')
    check(claim['source'] == intent['source'] and claim['plan'] == intent['plan']
          and claim['challenge'] == intent['challenge'] == launch['challenge'] and claim['hostname'] == 'jaguar03',
          'actual child claim exact plan/source/challenge/host')
    cgroup = intent['parent']['cgroup']
    snapshots = [intent['parent'], observation['parent'], observation['root_initial'], claim['process'],
                 launch['waiter'], launch['child'], *observation['nodes'].values()]
    for item in snapshots:
        snapshot(item, cgroup)
    check(intent['parent']['command'] == ['/usr/bin/python3', '-I', '-S', '-B',
          str(U / 'code/diagnose_nsight_process_tree.py'), '--prepared-unit', str(U)], 'exact actual diagnostic supervisor source argv')
    check(stable(claim['process']) == stable(launch['child']) == stable(rebuilt['capture'])
          and stable(launch['waiter']) == stable(observation['nodes'][str(launch['waiter']['pid'])])
          and launch['child']['parent_pid'] == launch['waiter']['pid']
          and launch['waiter']['command'] == intent['waiter_command']
          and launch['command'] == intent['capture_command'], 'actual own Python Popen parent/child stable joins')
    reference(launch['claim'], U / 'capture_identity.json')
    check(root['pid'] == observation['root_pid'] and root['process_start_time'] == observation['root_start_time']
          and root['parent_pid'] == intent['parent']['pid']
          and root['parent_start_time'] == intent['parent']['process_start_time']
          and root['command'] == intent['nsight_command'], 'actual original Nsight Popen stable identity')
    helper.validate_root_wait(U, intent, root)
    waited = read(U / 'capture_actual_wait.json')
    reference(waited['launch'], U / 'capture_actual_launch.json')
    check(type(waited['waited_pid']) is int and waited['waited_pid'] == claim['pid']
          and type(waited['returncode']) is int and waited['returncode'] == 0 and waited['exited_normally'] is True
          and waited['interventions'] == [] and waited['command'] == intent['capture_command']
          and waited['challenge'] == intent['challenge'], 'exact actual child wait is normal zero with no intervention')
    for field, filename in [('observation', 'capture_process_observation.json'),
                            ('actual_capture_wait', 'capture_actual_wait.json'),
                            ('actual_nsight_wait', 'profiler_actual_wait.json')]:
        reference(result[field], U / filename)
    finish = read(U / 'probe_finish.json')
    check(finish == {'challenge': intent['challenge'], 'claim': ref(U / 'capture_identity.json')},
          'exact challenge and claim acknowledgment releases only actual probe')
    probe = read(U / 'probe_result.json')
    check(probe['completed'] is True and probe['pid'] == claim['pid']
          and probe['process_start_time'] == claim['process_start_time'] and probe['numerical_modules'] == []
          and probe['cuda_profiler_api_called'] is False, 'actual completed standard-library probe evidence')
    for field in ('scientific_credit', 'profile_credit', 'case_credit', 'idea_credit', 'environment_credit', 'learner_credit'):
        check(type(probe[field]) is int and probe[field] == 0, 'probe zero credit: ' + field)
    chronology = ['process_control_capabilities.json', 'tool_version_return.json', 'profile_process_intent.json',
                  'profiler_process.json', 'capture_identity.json', 'capture_actual_launch.json',
                  'capture_process_observation.json', 'probe_finish.json', 'probe_result.json',
                  'capture_actual_wait.json', 'profiler_actual_wait.json', 'result.json']
    times = [int((U / name).stat().st_mtime_ns) for name in chronology]
    check(times == sorted(times), 'retained filesystem publication chronology from capability to result')
    check(timestamp(rel['at_utc']) <= timestamp(release_done['at_utc'])
          < intent['parent']['observed_at_seconds'] <= observation['root_initial']['observed_at_seconds']
          <= min(row['observed_at_seconds'] for row in observation['nodes'].values()),
          'actual release precedes saved source supervisor and pinned process observation')
    group_equal = rebuilt['capture']['process_group'] == rebuilt['root']['pid']
    check(group_equal is False and result['old_process_group_equality_observed'] is group_equal,
          'observed actual process-group difference is independent of ownership')
    # Focused negative controls are in-memory saved evidence only. They cannot query any PID or device.
    for label, mutate in [
        ('foreign_uid', lambda value: value['nodes'][str(claim['pid'])].__setitem__('uids', [UID + 1] * 4)),
        ('missing_ancestry_edge', lambda value: value['edges'].pop()),
        ('changed_capture_birth', lambda value: value['nodes'][str(claim['pid'])].__setitem__('process_start_time', '81355555')),
        ('false_pinned_handle_claim', lambda value: value.__setitem__('pidfds_opened_before_identity_reads', False)),
    ]:
        changed = copy.deepcopy(observation)
        mutate(changed)
        try:
            helper.validate_observation(changed, claim, intent['capture_command'], intent['parent'])
        except (ValueError, KeyError, AssertionError) as error:
            controls.append({'name': label, 'rejected': True, 'error': str(error)})
        else:
            raise AssertionError('saved negative control accepted: ' + label)
    check(len(controls) == 4 and all(row['rejected'] for row in controls), 'four focused saved-evidence mutation controls reject')
    query = read(P / 'root_actual_terminal_accounting_return.json')
    check(query['command'] == ['sacct', '-n', '-P', '-j', JOB,
          '--format=JobIDRaw,State,ExitCode,ElapsedRaw,NodeList,Submit,Start,End,ReqTRES,AllocTRES']
          and query['returncode'] == 0 and query['automatic_retry_permitted'] is False,
          'cached accounting queried only exact ledger-owned diagnostic ID')
    rows = [line.split('|') for line in (P / 'root_actual_terminal_accounting_stdout.raw').read_text().splitlines() if line]
    sha(P / 'root_actual_terminal_accounting_stdout.raw')
    check([row[0] for row in rows] == [JOB, JOB + '.batch', JOB + '.0']
          and all(row[1:5] == ['COMPLETED', '0:0', '2', 'jaguar03'] for row in rows),
          'actual parent/batch/numeric step terminal zero exits and exact runtime')
    for row in rows:
        actual = dict(item.split('=', 1) for item in row[9].split(','))
        check(actual['mem'] == '2G' and actual['gres/gpu'] == '1' and actual['node'] == '1'
              and actual['cpu'] == ('1' if row[0].endswith('.0') else '2'), 'actual allocated terminal TRES: ' + row[0])
    summary = read(P / 'root_actual_terminal_summary.json')
    check(summary['rows'] == rows and summary['job_id'] == JOB, 'root terminal summary exactly matches cached raw rows')
    reference(summary['source'], P / 'root_actual_terminal_accounting_stdout.raw')
    reference(summary['query'], P / 'root_actual_terminal_accounting_return.json')
    reference(summary['producer_result'], U / 'result.json')
    check(timestamp(query['started_at_utc']) > timestamp(rel['at_utc']), 'cached terminal query occurred after actual release')
    check(all(sha(path) == expected for path, expected in before.items()), 'all frozen source/packet bytes preserved after review')
    return {'kind': 'independent_actual_nsight_process_diagnostic_raw_review', 'passed': True,
        'actual_job_id': JOB, 'actual_numeric_step': JOB + '.0', 'exact_source_commit': COMMIT,
        'helper_staged_only_saved_validation': True, 'historical_pid_device_tool_library_operations_by_review': 0,
        'source_files_verified': len(plan['helpers']['files']), 'initial_artifacts_and_modes_verified': len(initial),
        'expanded_frozen_files_verified': len(freeze), 'terminal_rows': rows,
        'positive_actual_ancestry': [intent['parent']['pid'], observation['root_pid'], *rebuilt['path']],
        'capture_pid': claim['pid'], 'capture_birth': claim['process_start_time'],
        'process_group_difference_observed': True, 'exact_old_failure_cause_established': False,
        'whole_tree_completeness_established': False,
        'pinned_handles_evidence': 'exact source and saved positive-ancestry flags; no historical descriptor is reopened',
        'target_python': capabilities, 'actual_tool': {**tool, 'bytes': raw_files[tool['path']]['bytes']},
        'tool_hash_scope': 'published code hashes before and after actual --version; exact current bytes verified offline; no separate post-profile hash artifact',
        'publication_chronology': dict(zip(chronology, times)), 'negative_controls': controls,
        'scientific_credit': 0, 'profile_credit': 0, 'case_credit': 0, 'idea_credit': 0,
        'timing_credit': 0, 'packing_credit': 0, 'final_credit': 0, 'worker_credit': 0,
        'whole_tool_library_or_device_activity_quantified': False, 'physical_gres_mapping_verified': False,
        'kernel_device_enforcement_verified': False, 'qualification_granted': False,
        'admission_granted': False, 'worker_admitted': False,
        'ordinary_profile_placement_change_required': True,
        'scope': 'One completed system-Python zero-environment Nsight process observation on jaguar03; no direct H100/current-CUDA/standing-descriptor/numerical-profile qualification.'}


started = dt.datetime.now(dt.timezone.utc).isoformat()
try:
    payload = main()
except BaseException as error:
    payload = {'kind': 'independent_actual_nsight_process_diagnostic_raw_review', 'passed': False,
               'error': repr(error), 'traceback': traceback.format_exc(), 'scientific_credit': 0,
               'admission_granted': False}
payload.update({'schema_version': 1, 'review_started_at_utc': started,
                'review_finished_at_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
                'review_launch_resources': {'requested_cpus': 1, 'requested_memory_mib': 512, 'gpu_count': 0},
                'inherited_allocation_environment': {name: os.environ.get(name) for name in ('SLURM_JOB_ID', 'SLURM_STEP_ID', 'SLURM_CPUS_PER_TASK', 'SLURM_MEM_PER_NODE', 'SLURM_STEP_GPUS', 'CUDA_VISIBLE_DEVICES')},
                'review_cpu_job_id': os.environ.get('SLURM_JOB_ID'),
                'review_cpu_step_id': os.environ.get('SLURM_STEP_ID'), 'review_pid': os.getpid(),
                'review_program': ref(Path(__file__).resolve()), 'checks_passed': len(checks),
                'checks': checks, 'raw_files': raw_files})
raw = (json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()
with OUT.open('xb') as stream:
    stream.write(raw); stream.flush(); os.fsync(stream.fileno())
print(json.dumps({'passed': payload['passed'], 'checks_passed': len(checks), 'receipt': str(OUT),
                  'receipt_sha256': hashlib.sha256(raw).hexdigest(), 'error': payload.get('error'),
                  'review_step': os.environ.get('SLURM_STEP_ID')}, sort_keys=True))
raise SystemExit(0 if payload['passed'] else 1)
