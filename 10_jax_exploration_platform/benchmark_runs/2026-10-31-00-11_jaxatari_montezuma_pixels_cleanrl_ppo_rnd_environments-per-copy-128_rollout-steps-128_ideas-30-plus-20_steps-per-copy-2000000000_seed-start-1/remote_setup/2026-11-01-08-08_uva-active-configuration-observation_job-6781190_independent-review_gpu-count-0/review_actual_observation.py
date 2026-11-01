"""Check saved owned CPU metadata; query only the exact recorded completed ID."""
import base64
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import runpy
import socket
import stat
import subprocess
import sys

I = Path('/p/rlprojects/RND_worktrees/merge-atari-cleanrl')
R = I / '10_jax_exploration_platform/benchmark_runs/2026-10-31-00-11_jaxatari_montezuma_pixels_cleanrl_ppo_rnd_environments-per-copy-128_rollout-steps-128_ideas-30-plus-20_steps-per-copy-2000000000_seed-start-1'
P = R / 'remote_setup/2026-11-01-07-42_uva-active-configuration-observation_jaguar03_gpu-count-0_processor-count-1_host-memory-gib-2_time-limit-minutes-5_qos-csresnolim'
D = R / 'data' / P.name
OUT = Path(__file__).absolute().parent
JOB = '6781190'
SOURCE = '3df6419cb39ac20766af902c49f3b65a9e219d8a'
RELEASE_PUBLICATION = '6e2a67af353d4e3a2388dca574a98ed67ebc5255'
PREFIX = '10_jax_exploration_platform/scripts/atari_pixel_campaign/'
NUMERICAL = {'jax', 'jaxlib', 'numpy', 'torch', 'flax', 'optax', 'cupy', 'scipy'}
inputs = {}
checks = []
commands = []

def check(ok, label):
    if not ok:
        raise AssertionError(label)
    checks.append(label)

def digest(raw):
    return hashlib.sha256(raw).hexdigest()

def read(path):
    path = Path(path)
    check(path.is_absolute() and path.resolve(strict=True) == path, 'ordinary exact path: ' + str(path))
    value = path.lstat()
    check(stat.S_ISREG(value.st_mode) and not stat.S_ISLNK(value.st_mode), 'ordinary regular artifact: ' + str(path))
    raw = path.read_bytes()
    inputs[str(path)] = {'sha256': digest(raw), 'bytes': len(raw)}
    return raw

def record(path):
    return json.loads(read(path))

def write(name, value):
    raw = (json.dumps(value, indent=2, allow_nan=False) + '\n').encode()
    with (OUT / name).open('xb') as output:
        output.write(raw)
        output.flush()
        os.fsync(output.fileno())
    return {'path': str(OUT / name), 'sha256': digest(raw), 'bytes': len(raw)}

def audit(event, args):
    if event == 'import' and args[0].split('.')[0] in NUMERICAL:
        raise RuntimeError('numerical import forbidden: ' + args[0])
    if event in {'socket.connect', 'socket.bind', 'ctypes.dlopen'}:
        raise RuntimeError('network or shared-library loading forbidden')
    if event == 'subprocess.Popen':
        argv = args[1]
        permitted = (isinstance(argv, list) and len(argv) >= 2 and argv[0] == 'git'
            and (argv[1:3] == ['-C', str(I)] or argv[1:3] == ['-C', str(SOURCE_ROOT)]))
        permitted = permitted or argv == ['squeue', '-h', '-j', JOB, '-o', '%i|%u|%T|%R']
        permitted = permitted or argv == ['sacct', '-n', '-P', '-j', JOB, '--format=JobIDRaw,User,State,ExitCode,NodeList,AllocTRES,ReqTRES,Start,End,Elapsed,Restarts']
        if not permitted:
            raise RuntimeError('unapproved command: ' + repr(argv))

def git_blob(commit, relative):
    command = ['git', '-C', str(I), 'show', commit + ':' + relative]
    raw = subprocess.check_output(command)
    commands.append({'command': command, 'returncode': 0, 'stdout_sha256': digest(raw), 'stdout_bytes': len(raw)})
    return raw

def query(command):
    value = subprocess.run(command, capture_output=True, text=True, timeout=30, check=False)
    row = {'command': command, 'returncode': value.returncode, 'stdout': value.stdout, 'stderr': value.stderr,
           'queried_at_utc': datetime.now(timezone.utc).isoformat()}
    commands.append(row)
    check(value.returncode == 0 and value.stderr == '', 'exact owned scheduler command passed: ' + command[0])
    return row

plan = record(D / 'observation_plan.json')
SOURCE_ROOT = Path(plan['source']['source_root'])
sys.addaudithook(audit)
review = {'schema_version': 1, 'kind': 'uva_active_configuration_observation_independent_actual_review',
          'job_id': JOB, 'packet': str(P), 'unit': str(D), 'started_at_utc': datetime.now(timezone.utc).isoformat(),
          'review_process': {'job_id': os.environ.get('SLURM_JOB_ID'), 'step_id': os.environ.get('SLURM_STEP_ID'),
              'pid': os.getpid(), 'uid': os.getuid(), 'hostname': socket.gethostname(),
              'cgroup': Path('/proc/self/cgroup').read_text(), 'proc_stat': Path('/proc/self/stat').read_text()},
          'new_ssh_attempts': 0, 'job_actions': 0, 'numerical_computation': False,
          'physical_assignment_verified': False, 'kernel_device_enforcement_verified': False,
          'worker_admission': False, 'scientific_credit': 0, 'case_credit': 0, 'idea_credit': 0}
try:
    check(os.environ.get('SLURM_JOB_ID') == '6781087' and re.fullmatch(r'[0-9]+', os.environ.get('SLURM_STEP_ID', '')) is not None,
          'review runs only in current owned CPU monitor numeric step')
    check(os.getuid() == 2618919 and not NUMERICAL.intersection(sys.modules), 'review owner and numerical module guard')
    source_path = SOURCE_ROOT / PREFIX / 'uva_active_configuration_observation.py'
    committed_source = git_blob(SOURCE, PREFIX + source_path.name)
    check(read(source_path) == committed_source == read(D / 'code' / source_path.name), 'published detached and staged source bytes match exact source commit')
    check(digest(committed_source) == '8cc5a3544e426e3bd3e78707f6c77dc249a1d19165a593e157d2097daa07a648', 'reviewed source SHA256')
    original = record(P / 'source_manifest.json')
    for path, expected in original.items():
        raw = read(path)
        check(inputs[path] == expected, 'original preparation input unchanged: ' + path)
        check(git_blob('b19a56b533b6d90820c7ae4578ad2f4016a43cc0', str(Path(path).relative_to(I))) == raw,
              'original preparation bytes match published packet: ' + path)
    checksum_rows = read(P / 'source_checksums.sha256').decode().splitlines()
    check(len(checksum_rows) == len(original), 'checksum roster size matches source manifest')
    for line in checksum_rows:
        expected_sha, path = line.split('  ', 1)
        check(original[path]['sha256'] == expected_sha, 'original checksum roster binding: ' + path)
    module = runpy.run_path(str(source_path), run_name='reviewed_metadata_observer')
    check(module['verify_static'](D) == plan, 'pinned source independently replays unchanged static preparation')
    observation = record(D / 'observation.json')
    module['validate_allocation'](observation)
    check(observation['allocation_binding_passed'] is True and observation['completed'] is True,
          'pinned validator independently replays actual owned CPU allocation')
    check(observation['source'] == plan['source'] and observation['job_id'] == JOB and observation['step_id'] == '0',
          'actual source and numeric job/step are exact')
    for value in (plan, observation, record(D / 'progress.json'), record(D / 'hardware.json')):
        check(all(type(value[k]) is type(v) and value[k] == v for k, v in module['ZERO_CREDIT'].items()), 'strict zero-credit and false-admission fields')
    check(observation['process']['cpu_affinity'] == [56], 'actual CPU affinity retained')
    check(record(D / 'hardware.json')['allocated_gpu_count'] == 0, 'actual hardware record allocated zero GPUs')
    binding = record(D / 'held_identifier_binding.json')
    frozen_ledger = read(D / 'held_owned_job_ledger_snapshot.txt')
    execution_ledger = read(D / 'owned_job_ledger_snapshot.txt')
    canonical_ledger = read(R / 'slurm/submitted_jobids_uva.txt')
    check(binding == observation['held_identifier_binding'] and binding['owned_ledger_sha256'] == digest(frozen_ledger), 'held identifier binds frozen owned ledger bytes')
    check(observation['owned_ledger']['sha256'] == digest(execution_ledger), 'actual execution ledger hash retained')
    check(all(raw.decode().splitlines().count(JOB) == 1 for raw in (frozen_ledger, execution_ledger, canonical_ledger)), 'actual owned ID appears exactly once in all three ledgers')
    submission = record(P / 'submission_receipt.json')
    returned = record(P / 'submission_return.json')
    check(submission['job_id'] == JOB and submission['source_commit'] == SOURCE and submission['command'] == returned['command'] == record(P / 'manifest.yaml')['held_submission_command'], 'exact held submission command and source')
    check(returned['stdout'] == JOB + '\n' and returned['returncode'] == 0 and returned['stderr'] == '', 'raw submission returned only actual held ID')
    for key, name in [('plan_sha256', D / 'observation_plan.json'), ('manifest_sha256', P / 'manifest.yaml'), ('job_script_sha256', P / 'job.sbatch'), ('submission_intent_sha256', P / 'submission_intent.json'), ('submission_return_sha256', P / 'submission_return.json')]:
        check(submission[key] == digest(read(name)), 'submission receipt hash: ' + key)
    row = '\t'.join(submission['submission_row'])
    check(read(R / 'slurm/submissions.tsv').decode().splitlines().count(row) == 1, 'exact owned TSV row occurs once')
    held = record(P / 'held_job_observation.json')
    check(held['command'] == ['scontrol', 'show', 'job', JOB, '-o'] and held['returncode'] == 0 and held['stderr'] == '', 'raw own held query command')
    release = record(P / 'release_receipt.json')
    intent = record(P / 'release_intent.json')
    release_review = record(P / 'release_review.json')
    for k,v in release_review['exact_held_fields'].items():
        check(module['scheduler_field'](held['stdout'], k) == v, 'raw own held field agrees with release review: ' + k)
    check(release['command'] == intent['command'] == ['scontrol', 'release', JOB] and release['returncode'] == 0 and release['stdout'] == release['stderr'] == '', 'single preserved exact release command and successful return')
    check(release['published_review_commit'] == intent['published_review_commit'] == RELEASE_PUBLICATION, 'exact release publication')
    check(release['release_review_sha256'] == intent['review_sha256'] == digest(read(P / 'release_review.json')), 'release binds exact reviewed record')
    check(release_review['submission_receipt_sha256'] == digest(read(P / 'submission_receipt.json')) and release_review['immutable_binding_sha256'] == digest(read(D / 'held_identifier_binding.json')), 'release review binds actual submission and held identifier')
    for path in (P/'release_review.json', P/'submission_receipt.json', D/'held_identifier_binding.json', D/'observation_plan.json'):
        check(git_blob(RELEASE_PUBLICATION, str(path.relative_to(I))) == read(path), 'release prerequisite bytes were published: ' + str(path))
    dates = [intent['at_utc'], release['at_utc'], observation['started_utc'], observation['finished_utc']]
    check(dates == sorted(dates), 'UTC intent, release and observation chronology')
    out = read(P / ('slurm/job_' + JOB + '.out'))
    check(json.loads(out) == observation, 'job stdout equals complete saved observation')
    check(read(P / ('slurm/job_' + JOB + '.err')) == b'', 'job stderr empty')
    running = observation['running_configuration']
    check(running['command'] == ['scontrol','show','config'] and running['returncode'] == 0 and running['stderr_byte_count'] == 0 and running['stderr_sha256'] == digest(b''), 'filtered actual running configuration query')
    check(running['filtered_sha256'] == digest(running['filtered_bytes'].encode()), 'filtered running configuration bytes hash')
    settings = dict(re.findall(r'^([A-Za-z_]+) = (.*)$', running['filtered_bytes'], re.M))
    check(settings == {'GresTypes':'gpu','ProctrackType':'proctrack/cgroup','SlurmdSpoolDir':'/var/spool/slurmd','SLURM_CONF':'/etc/slurm/slurm.conf','TaskPlugin':'task/cgroup,task/affinity'}, 'exact five running settings')
    check(observation['cache_discovery'] == {'raw_slurmd_spool_values':['/var/spool/slurmd'],'cache_path_derived':True,'observed_slurmd_spool_dir':'/var/spool/slurmd'}, 'only actual unique spool directory derived cache path')
    files = observation['configuration_files']
    check([(x['kind'],x['path']) for x in files] == [(kind, str(Path(directory)/(kind+'.conf'))) for directory in ('/etc/slurm','/var/spool/slurmd/conf-cache') for kind in ('slurm','gres','cgroup')], 'exact six allowlisted candidate observations')
    for entry in files[:3]:
        check(entry['owner_uid'] == 64030 and entry['mode'] == 0o644 and entry['regular_file'] is True and entry['exact_path'] is True and entry['symlink'] is False and entry['read_status'] == 'read', 'observed ordinary config metadata: '+entry['path'])
        trace=entry['read_trace']
        before=trace['descriptor_before']
        check(before == trace['reader_before'] == trace['descriptor_after'], 'pinned inode unchanged through complete read: '+entry['path'])
        check(trace['complete_file'] is True and trace['bytes_read'] == before['byte_count'] == entry['byte_count'] and trace['inode_changed_during_read'] is False and trace['path_replaced_during_read'] is False, 'complete bounded ordinary read: '+entry['path'])
        for field in ('path','owner_uid','mode','inode','device','regular_file'):
            check(entry[field] == before[field] == trace['path_after'][field], 'descriptor/path identity '+field+': '+entry['path'])
        for pathrow, pinned in zip(entry['directory_chain'],trace['pinned_directory_chain']):
            check(pathrow['symlink'] is False and pathrow['exact_path'] is True and pathrow['directory'] is True, 'ordinary configuration ancestor: '+pathrow['path'])
            check(all(pathrow[k] == pinned[k] for k in ('path','owner_uid','mode','inode','device','directory','regular_file')), 'path and pinned ancestor identity: '+pathrow['path'])
        check(len(entry['directory_chain']) == len(trace['pinned_directory_chain']) == 3, 'complete pinned config directory chain')
        if entry['kind'] != 'slurm':
            raw=base64.b64decode(entry['raw_base64'],validate=True)
            check(raw == entry['raw'].encode() and digest(raw) == entry['sha256'] and len(raw) == entry['byte_count'], 'full saved configuration raw bytes/hash: '+entry['path'])
        else:
            check(entry['filtered_sha256'] == digest(entry['filtered_bytes'].encode()) and entry['include_directive_present'] is True, 'filtered slurm.conf bytes/hash and explicit include limitation')
    for entry in files[3:]:
        check(entry['status'] == 'unavailable' and entry['error_type'] == 'FileNotFoundError' and entry['read_status'] == 'not_read_ordinary_path_unavailable' and entry['directory_chain'][-1]['path'] == '/var/spool/slurmd/conf-cache' and entry['directory_chain'][-1]['status'] == 'unavailable', 'missing cache retained explicitly: '+entry['path'])
    gres = files[1]['raw']
    check('NodeName=jaguar[01-03,05-06] AutoDetect=nvml\n' in gres and 'File=' not in gres and 'Name=gpu' not in gres, 'actual local GRES is AutoDetect-only, without physical File mapping')
    # Only an owned exact-ID active queue/accounting read occurs after all saved ownership proofs pass.
    active=query(['squeue','-h','-j',JOB,'-o','%i|%u|%T|%R'])
    accounting=query(['sacct','-n','-P','-j',JOB,'--format=JobIDRaw,User,State,ExitCode,NodeList,AllocTRES,ReqTRES,Start,End,Elapsed,Restarts'])
    check(active['stdout'] == '', 'completed observer has no active owned queue row')
    fields=['JobIDRaw','User','State','ExitCode','NodeList','AllocTRES','ReqTRES','Start','End','Elapsed','Restarts']
    rows=[]
    for line in accounting['stdout'].splitlines():
        values=line.split('|')
        if len(values)==len(fields)+1 and values[-1]=='': values.pop()
        check(len(values)==len(fields), 'exact terminal accounting field count')
        rows.append(dict(zip(fields,values)))
    check({x['JobIDRaw'] for x in rows} == {JOB,JOB+'.batch',JOB+'.extern',JOB+'.0'} and len(rows)==4, 'only exact owned allocation and actual child steps returned')
    check(all(x['State']=='COMPLETED' and x['ExitCode']=='0:0' and x['NodeList']=='jaguar03' for x in rows), 'allocation and every actual step COMPLETED 0:0 on jaguar03')
    parent=next(x for x in rows if x['JobIDRaw']==JOB)
    check(parent['User']=='sl5nw' and parent['Restarts']=='0', 'terminal accounting owner and no restart')
    check(all(not re.search(r'(?:^|,)gres/gpu',x['AllocTRES']) for x in rows), 'terminal accounting contains no allocated GPU TRES')
    review.update(passed=True, observation_completed=True, accounting_rows=rows,
        configuration_summary=[{k:x.get(k) for k in ('path','owner_uid','mode','read_status','sha256','filtered_sha256','status','error_type')} for x in files],
        limitations=['The metadata observer records configuration; completion grants no physical GPU mapping or worker admission.',
            'The /etc/slurm directory and three files are owned by UID 64030, not root. Their ownership is observed and is not accepted by the current root-owned explicit-File profile.',
            'slurm.conf retained only allowlisted filtered bytes, its full-byte hash and an Include-presence flag. Its full original bytes and included files are not saved, so this review cannot independently recompute that full-byte hash or prove complete configuration resolution.',
            'scontrol show config retained filtered bytes plus a full-output hash, not the unfiltered full controller output.',
            'No NVML, CUDA, GPU library, GPU process, device access or merged Slurm GRES mapping was observed.',
            'The recorded release artifacts show one release attempt; this review does not query controller audit history for other sessions.'],
        mapping_profile_change=False)
except BaseException as error:
    review.update(passed=False,error_type=type(error).__name__,error=str(error))
    raise
finally:
    review.update(finished_at_utc=datetime.now(timezone.utc).isoformat(),checks_passed=len(checks),checks=checks,
        input_artifacts=inputs,commands=commands,numerical_modules_present=sorted(NUMERICAL.intersection(sys.modules)))
    receipt=write('independent_actual_observation_review.json',review)
    print(json.dumps({'receipt':receipt,'passed':review.get('passed'),'checks_passed':len(checks),'review_step':review['review_process']}))
