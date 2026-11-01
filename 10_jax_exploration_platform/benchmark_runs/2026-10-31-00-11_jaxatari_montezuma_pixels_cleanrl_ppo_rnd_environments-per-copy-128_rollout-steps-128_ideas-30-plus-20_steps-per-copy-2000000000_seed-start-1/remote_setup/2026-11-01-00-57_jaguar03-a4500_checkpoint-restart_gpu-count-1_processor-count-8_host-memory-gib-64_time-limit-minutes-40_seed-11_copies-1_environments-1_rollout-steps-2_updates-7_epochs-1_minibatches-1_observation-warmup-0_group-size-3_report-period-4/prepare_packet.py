"""Prepare one immutable restart component packet on the exact owned CPU allocation."""
import ast
from dataclasses import asdict
from datetime import datetime, timezone
import fcntl
import getpass
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shlex
import shutil
import stat
import subprocess
import sys
import tempfile
from zoneinfo import ZoneInfo

ROOT = Path('/p/rlprojects/RND_worktrees/merge-atari-cleanrl')
RUN = ROOT / '10_jax_exploration_platform/benchmark_runs/2026-10-31-00-11_jaxatari_montezuma_pixels_cleanrl_ppo_rnd_environments-per-copy-128_rollout-steps-128_ideas-30-plus-20_steps-per-copy-2000000000_seed-start-1'
COMMIT = '8a96999408d4e214b212074d932f0eeabcbb5eea'
PREFIX = '10_jax_exploration_platform/scripts/atari_pixel_campaign/'
PYTHON = Path('/p/rlprojects/RND/.venvs/atari_pixels_jax/bin/python')
REGISTRY = Path('/p/rlprojects/RND/.venvs/ENVS.md')
BOUNDARY = {'scientific_hypothesis_credit': 0, 'initial_case_credit': 0, 'adaptive_idea_credit': 0, 'final_training_credit': 0, 'worker_admission': False, 'source_admission_changed': False}


def sha(path):
    """Hash exact immutable artifact bytes without importing a runtime."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sync_directory(path):
    """Persist metadata directory entries before recording prepared state."""
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write(path, value, executable=False):
    """Create a complete preparation artifact without replacing an existing file."""
    contents = value if isinstance(value, bytes) else value.encode()
    with Path(path).open('xb') as stream:
        stream.write(contents)
        stream.flush()
        os.fsync(stream.fileno())
    Path(path).chmod(0o755 if executable else 0o644)


def document(path, value):
    """Write JSON as valid YAML when a standard launch record uses a YAML suffix."""
    write(path, json.dumps(value, indent=2, allow_nan=False) + '\n')


def command(argv):
    """Capture one explicit bounded read-only metadata command."""
    result = subprocess.run(argv, capture_output=True, text=True, check=True, timeout=30, env={**os.environ, 'TZ': 'UTC'})
    return {'argv': argv, 'returncode': result.returncode, 'stdout': result.stdout, 'stderr': result.stderr,
            'observed_at_utc': datetime.now(timezone.utc).isoformat()}


def main():
    """Prepare committed source, launch identity and exact held submission arguments without submitting."""
    sys.dont_write_bytecode = True
    ledger = RUN / 'slurm/submitted_jobids_uva.txt'
    owned = ledger.read_text().splitlines()
    assert os.environ.get('SLURM_JOB_ID') == '6780433' and owned.count('6780433') == 1
    assert '/job_6780433/' in Path('/proc/self/cgroup').read_text()
    assert getpass.getuser() == 'sl5nw' and os.getuid() == 2618919
    assert '6777946' in owned
    assert str(PYTHON.parent.parent) + '` is canonical for the new pixel Montezuma' in REGISTRY.read_text()
    assert PYTHON.is_file() and os.access(PYTHON, os.X_OK)
    assert not subprocess.check_output(['git', '-C', str(ROOT), 'diff', COMMIT, '--', '10_jax_exploration_platform/src', '10_jax_exploration_platform/scripts'])

    # Read current node/reservation/partition facts, exact historical owned shape, and literal caps.
    observations = [command(['scontrol', 'show', 'node', 'jaguar03']),
        command(['scontrol', 'show', 'partition', 'gpu', '-o']),
        command(['sacct', '-X', '-j', '6777946', '--format=JobIDRaw,User,Account,Partition,QOS,ReqCPUS,ReqMem,ReqTRES,Timelimit', '-P', '-n']),
        command(['sacctmgr', '-nP', 'show', 'qos', 'cspartgpu', 'format=Name,MaxTRESPU'])]
    fields = dict(token.split('=', 1) for token in observations[0]['stdout'].split() if '=' in token)
    reservation = fields['ReservationName']
    assert reservation not in ('', '(null)')
    observations.append(command(['scontrol', 'show', 'reservation', reservation, '-o']))
    res = dict(token.split('=', 1) for token in observations[-1]['stdout'].split() if '=' in token)
    assert 'sl5nw' in res['Users'].split(',') and res['State'] == 'ACTIVE'
    assert datetime.fromisoformat(res['EndTime']).replace(tzinfo=timezone.utc).timestamp() > datetime.now(timezone.utc).timestamp() + 2400
    assert observations[2]['stdout'].split('|')[:5] == ['6777946', 'sl5nw', 'sl5nw_base', 'gpu', 'csresnolim']
    assert 'cpu=400' in observations[3]['stdout'] and 'gres/gpu=40' in observations[3]['stdout']

    stamp = datetime.now(ZoneInfo('America/Los_Angeles')).strftime('%Y-%m-%d-%H-%M')
    slug = stamp + '_jaguar03-a4500_checkpoint-restart_gpu-count-1_processor-count-8_host-memory-gib-64_time-limit-minutes-40_seed-11_copies-1_environments-1_rollout-steps-2_updates-7_epochs-1_minibatches-1_observation-warmup-0_group-size-3_report-period-4'
    assert len(slug.encode()) <= 255
    packet = RUN / 'remote_setup' / slug
    packet.mkdir()
    background = '## Purpose\n\nVerify complete checkpoint restart, original single-update fallback and interrupted checkpoint publication on one allocated RTX A4500. This is a bounded component validation with zero study or final-training credit.\n\n## Key hyperparameters\n\nThe two frozen fixtures use seed 11, one policy copy, one pixel environment, two rollout steps, seven updates, one training epoch, one minibatch and zero observation warmup iterations. The grouped fixture resumes with both three-then-one and one-then-three update orders. The scheduled-report fixture reports every four updates and recovers interrupted publication at update three. Networks use float32 and statistics use float64, with highest matrix precision and deterministic operations. The allocation requests one typed A4500 on jaguar03, eight processors, 64 GiB host memory and a 40-minute limit.\n\n## Code and config changes\n\nThe exact two fixtures are loaded from committed helper source ' + COMMIT + '. All ten pixel numerical files are checked byte for byte against D28. The source archive includes only committed src and scripts files. The packet prepares an owned held submission; it submits, releases and admits no worker.\n\n## Git state\n\nHelper commit: `' + COMMIT + '`. Committed source bytes match the working source paths. Other campaign records continue to change during monitoring. Source and packet hashes are recorded separately.\n'
    write(packet / 'experiment_background.md', background)
    document(packet / 'preparation_observations.json', {'job_id': '6780433', 'step_id': os.environ['SLURM_STEP_ID'], 'observations': observations,
        'uid': os.getuid(), 'ledger': {'path': str(ledger), 'sha256': sha(ledger)}, 'reservation': reservation,
        'strict_caps': {'partition': 'gpu', 'cpus': 400, 'owner_processor_headroom': 16, 'host_memory_gib': 4096, 'gpus': 40,
            'requested_cpus': 8, 'requested_host_memory_gib': 64, 'requested_gpus': 1,
            'whole_uid_usage_inferred': False, 'scheduler_caps_remain_enforced': True, 'revalidate_before_submission_and_release': True}})

    # Freeze only committed source/scripts files, preserving their Git executable status.
    scopes = ['10_jax_exploration_platform/src', '10_jax_exploration_platform/scripts']
    records, blobs = {}, {}
    for line in subprocess.check_output(['git', '-C', str(ROOT), 'ls-tree', '-r', COMMIT, '--', *scopes], text=True).splitlines():
        identity, name = line.split('\t'); mode, kind, blob = identity.split()
        assert mode in {'100644', '100755'} and kind == 'blob' and '..' not in Path(name).parts
        contents = subprocess.check_output(['git', '-C', str(ROOT), 'show', COMMIT + ':' + name])
        records[name] = {'git_mode': mode, 'git_blob': blob, 'sha256': hashlib.sha256(contents).hexdigest(), 'bytes': len(contents)}
        blobs[name] = contents
    source = RUN / 'data/code_snapshots' / COMMIT
    source.parent.mkdir(exist_ok=True)
    with (source.parent / (COMMIT + '.lock')).open('a+') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        if not source.exists():
            with tempfile.TemporaryDirectory(dir=source.parent, prefix='building_checkpoint_restart_') as temporary:
                staging = Path(temporary) / 'source'; staging.mkdir()
                for name, contents in blobs.items():
                    path = staging / name; path.parent.mkdir(parents=True, exist_ok=True)
                    write(path, contents, records[name]['git_mode'] == '100755')
                for directory in sorted([p for p in staging.rglob('*') if p.is_dir()], key=lambda p: len(p.parts), reverse=True):
                    sync_directory(directory)
                sync_directory(staging)
                staging.rename(source)
                sync_directory(source.parent)
        actual = {p.relative_to(source).as_posix() for p in source.rglob('*') if p.is_file()}
        assert actual == set(records)
        for name, record in records.items():
            path = source / name
            assert not path.is_symlink() and path.read_bytes() == blobs[name]
            assert stat.S_IMODE(path.stat().st_mode) & 0o111 == (0o111 if record['git_mode'] == '100755' else 0)
    numerical_commit = subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'd28e673'], text=True).strip()
    pixel_names = subprocess.check_output(['git', '-C', str(ROOT), 'ls-tree', '-r', '--name-only', numerical_commit, '10_jax_exploration_platform/src/exploration_platform/atari_pixels'], text=True).splitlines()
    numerical = {}
    for name in pixel_names:
        original = subprocess.check_output(['git', '-C', str(ROOT), 'show', numerical_commit + ':' + name])
        assert original == blobs[name]
        numerical[name] = hashlib.sha256(original).hexdigest()
    document(packet / 'source_snapshot.json', {'schema_version': 1, 'kind': 'committed_source_snapshot', 'source_commit': COMMIT,
        'source_root': str(source), 'scopes': scopes, 'files': records, 'pixel_reference_commit': numerical_commit, 'pixel_reference_sha256': numerical})

    # Resolve ordinary defaults without importing a numerical package or constructing an environment.
    specification = importlib.util.spec_from_file_location('packet_pixel_configuration', source / '10_jax_exploration_platform/src/exploration_platform/atari_pixels/config.py')
    module = importlib.util.module_from_spec(specification);sys.modules[specification.name] = module;specification.loader.exec_module(module)
    learner = asdict(module.PixelPPOConfig(n_copies=1, n_envs=1, num_steps=2, total_timesteps=14, update_epochs=1,
        num_minibatches=1, observation_warmup_iterations=0, donate_state=True))
    tree = ast.parse((source / '10_jax_exploration_platform/src/exploration_platform/atari_pixels/environment.py').read_text())
    constructor = next(item for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'AtariPixelEnvironment'
                       for item in node.body if isinstance(item, ast.FunctionDef) and item.name == '__init__')
    environment = {arg.arg: ast.literal_eval(value) for arg, value in zip(constructor.args.args[-len(constructor.args.defaults):], constructor.args.defaults)}
    environment.update(num_envs=1, cache_room_backgrounds=True)
    tree = ast.parse((source / (PREFIX + 'paired_benchmark.py')).read_text())
    execution = next(ast.literal_eval(node.value) for node in tree.body if isinstance(node, ast.Assign)
                     and any(isinstance(target, ast.Name) and target.id == 'DEFAULT_EXECUTION' for target in node.targets))
    configuration = {'schema_version': 1, 'purpose': 'checkpoint_restart_component_validation', 'source_commit': COMMIT,
        'pixel_numerical_commit': numerical_commit, 'copy_seeds': [11], 'learner': learner, 'environment': environment,
        'fixtures': [{'test': 'test_group_checkpoint_resume_and_unpadded_tail_match_original_updates',
            'execution': {**execution, 'iterations_per_call': 3}, 'checkpoint_update': 3, 'remaining_call_orders': [[3, 1], [1, 3]]},
            {'test': 'test_scheduled_pending_report_resume_matches_uninterrupted_executor',
            'execution': {**execution, 'report_readout': 'scheduled', 'report_every_iterations': 4},
            'previous_checkpoint_update': 2, 'interrupted_checkpoint_update': 3, 'final_update': 7}],
        'runtime': {'JAX_ENABLE_X64': 'true', 'JAX_DEFAULT_MATMUL_PRECISION': 'highest', 'XLA_FLAGS': '--xla_gpu_exclude_nondeterministic_ops=true',
            'JAX_ENABLE_COMPILATION_CACHE': 'false', 'JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES': '', 'JAX_COMPILATION_CACHE_DIR': None,
            'JAXTARI_SKIP_SPRITE_UPDATE': '1', 'XDG_DATA_HOME': '/p/rlprojects/RND/reference_repos/jaxtari_pixel_study_assets',
            'XLA_PYTHON_CLIENT_PREALLOCATE': 'false', 'XLA_PYTHON_CLIENT_MEM_FRACTION': '0.85', 'OMP_NUM_THREADS': '8',
            'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1', 'PYTHONDONTWRITEBYTECODE': '1'},
        'precision': {'networks': 'float32', 'statistics': 'float64'}, 'python': str(PYTHON), **BOUNDARY}
    document(packet / 'config_resolved.yaml', configuration)
    document(packet / 'environment_registry.json', {'path': str(REGISTRY), 'sha256': sha(REGISTRY), 'python': str(PYTHON),
        'resolved_python': str(PYTHON.resolve()), 'python_binary_sha256': sha(PYTHON), 'canonical_pixel_environment': True})
    code = 'import importlib.metadata,json,sys;print(json.dumps({"python":sys.version,"prefix":sys.prefix,"packages":{name:importlib.metadata.version(name) for name in ["jax","jaxlib","jaxtari","flax","optax","numpy","pytest"]},"numerical_modules_imported":[name for name in sys.modules if name.split(".")[0] in {"jax","jaxlib","numpy","flax","optax"}]}))'
    runtime = command([str(PYTHON), '-I', '-c', code]); assert not json.loads(runtime['stdout'])['numerical_modules_imported']
    document(packet / 'runtime_metadata.json', runtime)

    expected_guard = {'owner': 'sl5nw', 'uid': 2618919, 'account': 'sl5nw_base', 'partition': 'gpu', 'qos': 'csresnolim',
        'reservation': reservation, 'node': 'jaguar03', 'cpus': 8, 'memory': '64G', 'time_limit': '00:40:00',
        'gres': 'gpu:nvidia_rtx_a4500:1', 'campaign_run_dir': str(RUN), 'unit_id': slug, 'source_commit': COMMIT,
        'submission_kind': 'checkpoint-restart-component-validation', 'job_script': str(packet / 'job.sbatch')}
    guard = '''"""Verify only this component job's owned allocation before numerical imports."""
import getpass,json,os,re,socket,subprocess
from datetime import datetime,timezone
from pathlib import Path
EXPECTED = REPLACE_EXPECTED

def validate(fields, job, hostname, cgroup):
    """Require the exact owned account, resource shape, host and current job cgroup."""
    required = {'JobId':job,'UserId':EXPECTED['owner']+'('+str(EXPECTED['uid'])+')','Account':EXPECTED['account'],
        'Partition':EXPECTED['partition'],'QOS':EXPECTED['qos'],'Reservation':EXPECTED['reservation'],'JobState':'RUNNING',
        'NodeList':EXPECTED['node'],'NumNodes':'1','NumCPUs':str(EXPECTED['cpus']),'TimeLimit':EXPECTED['time_limit'],'MinMemoryNode':EXPECTED['memory']}
    if any(fields.get(key)!=value for key,value in required.items()):
        raise ValueError('component allocation differs from its exact owned requested shape')
    if hostname.split('.')[0]!=EXPECTED['node'] or '/job_'+job+'/' not in cgroup:
        raise ValueError('component process is outside its exact allocated compute host/job cgroup')
    if fields.get('TresPerNode') not in {'gres:gpu:nvidia_rtx_a4500:1','gres/gpu:nvidia_rtx_a4500:1'}:
        raise ValueError('component allocation does not request exactly one typed A4500')
    allocated=dict(item.split('=',1) for item in fields.get('AllocTRES','').split(',') if '=' in item)
    if allocated.get('gres/gpu')!='1' or allocated.get('cpu')!='8' or allocated.get('mem')!='64G':
        raise ValueError('component allocated resources differ from the one-card/eight-processor/64GiB shape')

def main():
    """Check literal campaign ledger ownership before querying only this process's exact job."""
    job=os.environ.get('SLURM_JOB_ID','');run=Path(EXPECTED['campaign_run_dir'])
    if not job.isdecimal() or int(job)<1 or os.getuid()!=EXPECTED['uid'] or getpass.getuser()!=EXPECTED['owner']:
        raise ValueError('component requires the exact owned Slurm compute user and job')
    ledger=run/'slurm/submitted_jobids_uva.txt';rows=run/'slurm/submissions.tsv'
    if ledger.is_symlink() or rows.is_symlink() or ledger.read_text().splitlines().count(job)!=1:
        raise ValueError('component job is absent or duplicated in the exact UVA ledger')
    matching=[line.split('\\t') for line in rows.read_text().splitlines() if line.split('\\t')[0]==job]
    if matching!=[[job,EXPECTED['submission_kind'],EXPECTED['source_commit'],EXPECTED['job_script']]]:
        raise ValueError('component job lacks its exact owned source/script submission row')
    command=['scontrol','show','job',job,'-o']
    raw=subprocess.run(command,capture_output=True,text=True,check=True,timeout=20,env={**os.environ,'TZ':'UTC'})
    if len(raw.stdout.splitlines())!=1:
        raise ValueError('component allocation query must return exactly one current job row')
    fields=dict(token.split('=',1) for token in raw.stdout.split() if '=' in token)
    cgroup=Path('/proc/self/cgroup').read_text();validate(fields,job,socket.gethostname(),cgroup)
    directory=run/'data'/EXPECTED['unit_id'];directory.mkdir(parents=True,exist_ok=True)
    record={'kind':'owned_component_allocation','expected':EXPECTED,'job_id':job,'pid':os.getpid(),'uid':os.getuid(),
        'hostname':socket.gethostname(),'cgroup':cgroup,'command':command,'returncode':raw.returncode,'stdout':raw.stdout,
        'stderr':raw.stderr,'observed_at_utc':datetime.now(timezone.utc).isoformat(),'numerical_imports':False}
    path=directory/'allocation_guard.json'
    with path.open('x') as stream:
        json.dump(record,stream,indent=2);stream.write('\\n');stream.flush();os.fsync(stream.fileno())
if __name__=='__main__':main()
'''.replace('REPLACE_EXPECTED', repr(expected_guard))
    write(packet / 'allocation_guard.py', guard)
    source_hash, config_hash, registry_hash = sha(packet / 'source_snapshot.json'), sha(packet / 'config_resolved.yaml'), sha(packet / 'environment_registry.json')
    verifier = '''"""Verify the complete fixed source and prepared component configuration without numerical imports."""
import hashlib,json,os,stat,sys,time
from pathlib import Path
PACKET=Path(REPLACE_PACKET)
EXPECTED=REPLACE_HASHES

def main():
    """Wait at most120seconds for fixed preparation files, then verify their exact bytes and source tree."""
    deadline=time.monotonic()+120
    while any(not (PACKET/name).is_file() for name in EXPECTED):
        if time.monotonic()>=deadline:raise RuntimeError('component preparation files are not visible on this compute host')
        time.sleep(1)
    for name,checksum in EXPECTED.items():
        path=PACKET/name
        if path.is_symlink() or hashlib.sha256(path.read_bytes()).hexdigest()!=checksum:raise ValueError('prepared input changed: '+name)
    manifest=json.loads((PACKET/'source_snapshot.json').read_text());source=Path(manifest['source_root'])
    actual={p.relative_to(source).as_posix() for p in source.rglob('*') if p.is_file()}
    if actual!=set(manifest['files']):raise ValueError('component snapshot has missing or additional source files')
    for name,record in manifest['files'].items():
        path=source/name;contents=path.read_bytes()
        blob=hashlib.sha1(b'blob '+str(len(contents)).encode()+b'\\0'+contents).hexdigest()
        if path.is_symlink() or path.resolve()!=path.absolute() or hashlib.sha256(contents).hexdigest()!=record['sha256'] or blob!=record['git_blob']:
            raise ValueError('component snapshot differs from its committed Git source: '+name)
        if stat.S_IMODE(path.stat().st_mode)&0o111!=(0o111 if record['git_mode']=='100755' else 0):raise ValueError('component source executable mode changed')
    for name,checksum in manifest['pixel_reference_sha256'].items():
        if hashlib.sha256((source/name).read_bytes()).hexdigest()!=checksum:raise ValueError('pixel numerical source differs from D28')
    registry=json.loads((PACKET/'environment_registry.json').read_text())
    if hashlib.sha256(Path(registry['path']).read_bytes()).hexdigest()!=registry['sha256'] or not os.access(registry['python'],os.X_OK):
        raise ValueError('canonical pixel environment registry or executable changed')
    if hashlib.sha256(Path(registry['python']).read_bytes()).hexdigest()!=registry['python_binary_sha256']:raise ValueError('canonical pixel Python binary changed')
    if any(name.split('.')[0] in {'jax','jaxlib','numpy','flax','optax','torch'} for name in sys.modules):raise RuntimeError('source verification imported a numerical runtime')
    print(json.dumps({'source_commit':manifest['source_commit'],'files_verified':len(actual),'pixel_numerical_files_verified':len(manifest['pixel_reference_sha256']),'numerical_imports':False}))
if __name__=='__main__':main()
'''.replace('REPLACE_PACKET', repr(str(packet))).replace('REPLACE_HASHES', repr({'source_snapshot.json': source_hash, 'config_resolved.yaml': config_hash, 'environment_registry.json': registry_hash, 'allocation_guard.py': sha(packet / 'allocation_guard.py')}))
    write(packet / 'verify_source.py', verifier)

    # Absolute paths keep the submitted script independent of Slurm's spool directory.
    script = ['#!/bin/bash', 'set -euo pipefail', '# One owned bounded A4500 component validation; no study or final-training credit.',
        'export PYTHONNOUSERSITE=1', 'export PYTHONDONTWRITEBYTECODE=1', 'export PYTHONUNBUFFERED=1',
        'unset JAX_COMPILATION_CACHE_DIR JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES', 'export JAX_ENABLE_COMPILATION_CACHE=false',
        'export XLA_FLAGS=', 'export PYTEST_ADDOPTS="-p no:cacheprovider"',
        'printf '+shlex.quote(sha(packet / 'verify_source.py')+'  '+str(packet / 'verify_source.py')+'\n')+' | sha256sum --check --status',
        '/usr/bin/python3 '+shlex.quote(str(packet / 'verify_source.py')),
        '/usr/bin/python3 '+shlex.quote(str(packet / 'allocation_guard.py')),
        'source '+shlex.quote(str(source / (PREFIX + 'worker_environment.sh'))),
        'export PYTHONPATH='+shlex.quote(str(source / '10_jax_exploration_platform/src')),
        'cd '+shlex.quote(str(source)),
        'gpu_uuid=$('+shlex.join(['/usr/bin/python3',str(source / (PREFIX+'launch_queue.py')),'--context','jaguar03_rtx_a4500','--source-root',str(source),'--run-dir',str(RUN),'--resolve-slurm-gpu'])+')',
        'export CUDA_VISIBLE_DEVICES="$gpu_uuid"', 'export RND_RUN_GPU_COMPONENT_TESTS=1', 'export RND_COMPONENT_SOURCE_COMMIT='+COMMIT]
    fixture_command=[str(PYTHON),'-m','pytest','-q',str(source/(PREFIX+'tests/gpu_train_optimized_restart.py'))]
    wrapper=[str(PYTHON),str(source/(PREFIX+'run_checks.py')),'--run-dir',str(RUN),'--unit-id',slug,'--source-commit',COMMIT,'--',*fixture_command]
    script.append('exec '+shlex.join(wrapper));write(packet/'job.sbatch','\n'.join(script)+'\n',executable=True)
    planned=['sbatch','--hold','--no-requeue','--parsable','--nodes=1','--ntasks=1','--job-name=pixel_checkpoint_restart',
        '--account=sl5nw_base','--partition=gpu','--qos=csresnolim','--reservation='+reservation,'--nodelist=jaguar03',
        '--gres=gpu:nvidia_rtx_a4500:1','--cpus-per-task=8','--mem=64G','--time=00:40:00',
        '--output='+str(RUN/'slurm/jobs'/('checkpoint-restart-validation_source-'+COMMIT+'_job-%j.log')),str(packet/'job.sbatch')]
    write(packet/'command.txt',shlex.join(planned)+'\n')
    document(packet/'validation_plan.json',{'schema_version':1,'kind':'checkpoint_restart_component_validation','prepared_at_utc':datetime.now(timezone.utc).isoformat(),
        'packet_directory':str(packet),'campaign_run_dir':str(RUN),'data_unit':slug,'source_commit':COMMIT,'source_root':str(source),
        'source_snapshot_sha256':source_hash,'config_sha256':config_hash,'job_script_sha256':sha(packet/'job.sbatch'),
        'owner':{'name':'sl5nw','uid':2618919},'resources':expected_guard,'planned_held_command':planned,
        'required_submission_row':['<exact_returned_job_id>',expected_guard['submission_kind'],COMMIT,str(packet/'job.sbatch')],
        'required_ledger':str(ledger),'submission_and_release_permitted_by_this_preparation':False,'fixture_command':fixture_command,**BOUNDARY})
    document(packet/'manifest.yaml',{'schema_version':1,'run_id':slug,'description':'Bounded full-state checkpoint and original-update fallback component validation.',
        'git':{'commit':COMMIT,'dirty':False},'source_root':str(source),'config_resolved_sha256':config_hash,
        'compile_signature':{'copies':1,'environments_per_copy':1,'rollout_steps':2,'epochs':1,'minibatches':1,'network_dtype':'float32','statistics_dtype':'float64'},
        'seeding':{'base_seed':11,'mode':'paired'},'artifacts':{'data':str(RUN/'data'/slug),'plan':'validation_plan.json','source_snapshot':'source_snapshot.json'},**BOUNDARY})
    shutil.copyfile(__file__, packet/'prepare_packet.py')
    # Execute only stdlib source visibility, guard-shape tests and shell syntax on this owned CPU step.
    visibility=command(['/usr/bin/python3',str(packet/'verify_source.py')]);shell=command(['bash','-n',str(packet/'job.sbatch')])
    namespace={'__name__':'component_guard_review'};exec(compile(guard,str(packet/'allocation_guard.py'),'exec'),namespace)
    valid={'JobId':'123','UserId':'sl5nw(2618919)','Account':'sl5nw_base','Partition':'gpu','QOS':'csresnolim','Reservation':reservation,
        'JobState':'RUNNING','NodeList':'jaguar03','NumNodes':'1','NumCPUs':'8','TimeLimit':'00:40:00','MinMemoryNode':'64G',
        'TresPerNode':'gres:gpu:nvidia_rtx_a4500:1','AllocTRES':'cpu=8,mem=64G,node=1,gres/gpu=1'}
    namespace['validate'](valid,'123','jaguar03','0::/system.slice/slurmstepd.scope/job_123/step_batch/user/task_0\n')
    negatives=[]
    for field,value in [('UserId','foreign(1)'),('JobId','456'),('NumCPUs','16'),('MinMemoryNode','32G'),('JobState','PENDING'),('TresPerNode','gres:gpu:h100:1')]:
        try:namespace['validate']({**valid,field:value},'123','jaguar03','0::/job_123/step_batch/\n')
        except ValueError:negatives.append(field)
        else:raise AssertionError('guard accepted invalid '+field)
    assert not any(name.split('.')[0] in {'jax','jaxlib','numpy','flax','optax','torch'} for name in sys.modules)
    artifact_hashes={p.name:sha(p) for p in packet.iterdir() if p.is_file()}
    document(packet/'packet_manifest.json',{'schema_version':1,'kind':'immutable_component_preparation_manifest','source_commit':COMMIT,'files':artifact_hashes,**BOUNDARY})
    document(packet/'preparation_receipt.json',{'schema_version':1,'kind':'checkpoint_restart_packet_preparation','job_id':'6780433','step_id':os.environ['SLURM_STEP_ID'],
        'packet_directory':str(packet),'packet_manifest_sha256':sha(packet/'packet_manifest.json'),'source_visibility':visibility,'shell_syntax':shell,
        'allocation_guard_golden_passed':True,'allocation_guard_negative_fields':negatives,'source_files_verified':len(records),'pixel_numerical_files_verified':len(numerical),
        'numeric_code_executed':False,'submission_occurred':False,'release_occurred':False,'git_commit_occurred':False,**BOUNDARY})
    sync_directory(packet)
    print(json.dumps({'packet':str(packet),'basename_bytes':len(slug.encode()),'packet_manifest_sha256':sha(packet/'packet_manifest.json'),
        'preparation_receipt_sha256':sha(packet/'preparation_receipt.json'),'job_script_sha256':sha(packet/'job.sbatch'),'source_files':len(records),'planned_command':planned},indent=2))


if __name__=='__main__':
    main()
