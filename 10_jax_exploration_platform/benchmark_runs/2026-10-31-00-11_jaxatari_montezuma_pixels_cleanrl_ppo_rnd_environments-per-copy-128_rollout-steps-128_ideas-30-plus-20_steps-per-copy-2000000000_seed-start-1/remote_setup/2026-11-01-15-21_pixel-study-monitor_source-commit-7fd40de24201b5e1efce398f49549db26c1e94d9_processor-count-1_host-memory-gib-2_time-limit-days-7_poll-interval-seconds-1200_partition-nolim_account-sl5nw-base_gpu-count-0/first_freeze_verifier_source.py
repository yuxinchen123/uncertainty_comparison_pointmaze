"""Freeze the new prepared monitor after metadata-only source and candidate checks."""
import ast
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
import tarfile
import tempfile

W = Path(__file__).resolve().parents[1]
inputs = json.loads((W / '.review_metadata/monitor_7fd_preparation_inputs.json').read_text())
P, CACHE = Path(inputs['packet']), Path(inputs['cache'])
R = P.parent.parent
UNIT = R / 'data' / P.name
assert os.environ['SLURM_JOB_ID'] == '6782816' and os.environ['SLURM_STEP_ID'].isdigit()


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def read(path):
    return json.loads(path.read_bytes())


def record(path):
    status = path.lstat()
    assert stat.S_ISREG(status.st_mode) and not path.is_symlink(), str(path)
    raw = path.read_bytes()
    return {'path': str(path), 'sha256': sha(raw), 'bytes': len(raw), 'mode': format(stat.S_IMODE(status.st_mode), '04o')}


def save(name, value):
    raw = (json.dumps(value, indent=2) + '\n').encode()
    with (P / name).open('xb') as stream:
        stream.write(raw); stream.flush(); os.fsync(stream.fileno())
    descriptor = os.open(P, os.O_RDONLY | os.O_DIRECTORY)
    os.fsync(descriptor); os.close(descriptor)
    return record(P / name)


def git(*arguments):
    command = ['git', '--no-optional-locks', '-C', str(CACHE), *arguments]
    result = subprocess.run(command, capture_output=True, text=True, timeout=30,
                            env={**os.environ, 'GIT_OPTIONAL_LOCKS': '0'})
    assert result.returncode == 0, result.stderr
    return result.stdout


assert git('rev-parse', 'HEAD').strip() == inputs['commit']
assert not git('status', '--porcelain=v1', '--untracked-files=all')
manifest = read(P / 'source_snapshot.json')
SOURCE = Path(manifest['source_root'])
assert manifest['source_commit'] == inputs['commit'] and len(manifest['files']) == 263
assert all(not path.is_symlink() for path in SOURCE.rglob('*'))
assert {path.relative_to(SOURCE).as_posix() for path in SOURCE.rglob('*') if path.is_file()} == set(manifest['files'])
tree = {}
for line in git('ls-tree', '-r', inputs['commit'], '--', '10_jax_exploration_platform/src', '10_jax_exploration_platform/scripts').splitlines():
    metadata, name = line.split('\t'); mode, kind, blob = metadata.split()
    assert kind == 'blob' and mode in {'100644', '100755'}
    tree[name] = (mode, blob)
assert set(tree) == set(manifest['files'])
with tarfile.open(P / 'source_git_archive.tar', 'r') as archive:
    members = [member for member in archive.getmembers() if not member.isdir()]
    assert len(members) == len(tree) and {member.name for member in members} == set(tree)
    for member in members:
        assert member.isfile() and not member.issym() and not member.islnk()
        name = member.name; raw = archive.extractfile(member).read(); expected = manifest['files'][name]
        mode, blob = tree[name]
        assert sha(raw) == expected['sha256'] and len(raw) == expected['bytes']
        assert hashlib.sha1(b'blob ' + str(len(raw)).encode() + b'\0' + raw).hexdigest() == blob == expected['git_blob']
        assert mode == expected['git_mode']
        assert (member.mode & 0o111) == (0o111 if mode == '100755' else 0)
        for root in (SOURCE, CACHE):
            path = root / name
            assert record(path)['sha256'] == expected['sha256']
            assert (path.stat().st_mode & 0o111) == (0o111 if mode == '100755' else 0)

# Execute only the candidate's write observer with ordinary temporary files. It
# must forward the original write and retain this process's bytes after a later write.
candidate = ast.parse((P / 'candidate_full_tick.py').read_text())
main = next(node for node in candidate.body if isinstance(node, ast.FunctionDef) and node.name == 'main')
observer = next(node for node in main.body if isinstance(node, ast.FunctionDef) and node.name == 'record_write')
with tempfile.TemporaryDirectory(prefix='pixel_monitor_write_observer_') as temporary:
    directory = Path(temporary)
    report, status_path = directory / 'latest_report.md', directory / 'latest_status.json'
    namespace = {'original_write_text': Path.write_text, 'own_writes': {}, 'report_paths': {report, status_path}}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[observer], type_ignores=[])), '<exact-candidate-write-observer>', 'exec'), namespace)
    original_report, original_status = '# actual delegated candidate bytes\n', '{"synthetic":true}\n'
    assert namespace['record_write'](report, original_report) == len(original_report)
    assert namespace['record_write'](status_path, original_status) == len(original_status)
    assert report.read_text() == original_report and status_path.read_text() == original_status
    report.write_text('later separate writer\n'); status_path.write_text('{}\n')
    assert namespace['own_writes'] == {'latest_report.md': [original_report.encode()], 'latest_status.json': [original_status.encode()]}
observer_ref = save('candidate_write_observer_verification.json', {
    'schema_version': 1, 'kind': 'candidate_write_observer_metadata_verification', 'passed': True,
    'actual_job_id': os.environ['SLURM_JOB_ID'], 'actual_step_id': os.environ['SLURM_STEP_ID'],
    'exact_production_observer_ast': True, 'real_writes_delegated_unchanged': True,
    'later_global_writes_cannot_replace_saved_candidate_bytes': True,
    'synthetic_ordinary_files_only': True, 'monitor_tick_executed': False, 'job_actions': 0})

plan = read(P / 'monitor_replacement_plan.json')
assert plan['helper_commit'] == inputs['commit'] and plan['old_monitor_job_id'] == '6782816'
assert plan['job_id'] is None and plan['gpu_uuid'] is None and plan['reservation'] is None
assert plan['job_shape'] == {'account':'sl5nw_base','partition':'nolim','qos':'normal','nodes':1,'ntasks':1,
    'cpus_per_task':1,'host_memory_gib':2,'gpus_per_node':0,'time':'7-00:00:00','interval_seconds':1200,'requeue':False}
assert '--hold' in plan['submission_command'] and '--no-requeue' in plan['submission_command']
assert all(len(path.name.encode()) <= 255 for folder in (P, UNIT) for path in folder.rglob('*'))
assert len(P.name.encode()) <= 255
assert not any((P / name).exists() for name in ('submission_intent.json','submission_receipt.json','release_intent.json','release_receipt.json','candidate_full_tick_verification.json','candidate_full_tick_failure.json'))
metadata = read(P / 'metadata_verification.json')
assert metadata['job_id'] == '6782816' and metadata['step_id'] == '17'
assert metadata['protected_before'] == metadata['protected_after'] and metadata['protected_paths'] == 187
assert metadata['source_hashes_before'] == metadata['source_hashes_after']
assert not git('status', '--porcelain=v1', '--untracked-files=all')
assert not any(name.split('.')[0] in {'jax','jaxlib','numpy','torch','flax','optax','jaxtari','pynvml'} for name in __import__('sys').modules)
files = sorted([path for folder in (P, UNIT) for path in folder.rglob('*') if path.is_file()], key=str)
roster_ref = save('static_publication_roster.json', {
    'schema_version':1,'kind':'complete_prepared_monitor_static_publication_roster','source_commit':inputs['commit'],
    'packet':str(P),'data_unit':str(UNIT),'file_count':len(files),'files':[record(path) for path in files],
    'source_files':len(tree),'full_candidate_tick_executed':False,'new_jobs_submitted':0,
    'excludes_self_and_final_freeze_receipt':True})
freeze = {'schema_version':1,'kind':'persistent_monitor_static_preparation_freeze','passed':True,
    'at_utc':datetime.now(timezone.utc).isoformat(),'actual_job_id':os.environ['SLURM_JOB_ID'],
    'actual_step_id':os.environ['SLURM_STEP_ID'],'source_commit':inputs['commit'],'source_root':str(CACHE),
    'packet':str(P),'data_unit':str(UNIT),'source_files':len(tree),'source_cache_clean_detached':True,
    'all_git_archive_snapshot_cache_hashes_and_executable_bits_match':True,
    'source_manifest':record(P/'source_snapshot.json'),'plan':record(P/'monitor_replacement_plan.json'),
    'script':record(P/'monitor.sbatch'),'manifest':record(P/'manifest.yaml'),'candidate':record(P/'candidate_full_tick.py'),
    'metadata_verification':record(P/'metadata_verification.json'),'candidate_write_observer':observer_ref,
    'framing_verification':record(P/'candidate_framing_return.json'),'publication_roster':roster_ref,
    'preparation_step':'6782816.14','full_metadata_step':'6782816.17',
    'cold_source_imports_and_direct_uva_memory_learning_helper_union_passed':True,
    'unsupported_direct_consumers_refused':True,'protected_paths_unchanged':187,
    'old_monitor_preserved':'6782816','full_candidate_tick_executed':False,
    'candidate_waits_for_root_fresh_aces_mirrors':True,'new_monitor_job_id':None,
    'new_jobs_submitted':0,'jobs_released_or_cancelled':0,'numerical_imports':False,
    'worker_admission':False,'scientific_credit':0,
    'review_instructions':['Check the complete publication roster and its modes/hashes, then the263-file source manifest.',
        'Read candidate_full_tick.py, verify_candidate_framing.py and handover_plan.json.',
        'The full tick remains unrun until root coordinates fresh ACES snapshots.',
        'Source publication and actual held submission/release/runtime/full-report review remain separate.']}
saved = save('static_source_freeze.json', freeze)
print(json.dumps({'passed':True,'packet':str(P),'freeze':saved,'publication_roster':roster_ref,
    'source_files':len(tree),'static_files_before_roster':len(files),'actual_step':os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID']}), flush=True)
