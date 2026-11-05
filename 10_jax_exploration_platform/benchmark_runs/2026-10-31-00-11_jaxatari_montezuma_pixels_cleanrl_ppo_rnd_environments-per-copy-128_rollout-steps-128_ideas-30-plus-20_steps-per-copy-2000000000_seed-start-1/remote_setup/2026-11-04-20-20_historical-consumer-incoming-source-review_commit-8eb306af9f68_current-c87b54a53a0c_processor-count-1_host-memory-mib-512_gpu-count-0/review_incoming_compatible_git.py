"""Check one incoming tree with private Git indexes and source-only metadata imports."""
import ast
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import tempfile
import time

OUTPUT = Path(__file__).resolve().parent
INTEGRATION = OUTPUT.parents[4]
RUN = OUTPUT.parents[1]
WORKTREE = Path('/p/rlprojects/RND_worktrees/2026-11-04-19-22_direct-common-historical-consumer_linked-detached_original-helpers-26_common-helpers-28_read-only-owned-compute')
BASE = 'cc9972dd6a8bfba39ba439883cbc821b93c55e3d'
CURRENT = 'c87b54a53a0c4017b8a70236a78333005a71542d'
INCOMING = '8eb306af9f68aed5ab5cc0d01f75b35a1f234078'
PREFIX = '10_jax_exploration_platform/scripts/atari_pixel_campaign/'
SHARED = PREFIX + 'direct_measurement.py'
COMMANDS = []


def digest(raw):
    """Identify exact saved bytes."""
    return hashlib.sha256(raw).hexdigest()


def reference(path):
    """Keep the exact location and digest of one review input."""
    return {'path':str(path),'sha256':digest(path.read_bytes())}


def publish(name, value):
    """Preserve every attempt without replacing an earlier record."""
    with (OUTPUT/name).open('xb') as stream:
        stream.write((json.dumps(value,indent=2,sort_keys=True)+'\n').encode())


def git(arguments, *, env=None, input=None, require=True):
    """Use local read-only Git operations or the explicitly isolated index dry runs."""
    command = ['git','-C',str(INTEGRATION),*arguments]
    environment = dict(os.environ,GIT_OPTIONAL_LOCKS='0')
    if env is not None:
        environment.update(env)
    result = subprocess.run(command,input=input,env=environment,capture_output=True)
    COMMANDS.append({'command':command,'returncode':result.returncode,
        'stdout_sha256':digest(result.stdout),'stderr':result.stderr.decode(errors='replace')})
    if require:
        assert result.returncode == 0, COMMANDS[-1]
    return result


def files(tree, paths):
    """Read exact tree entries and their blobs without checking them out."""
    raw = git(['ls-tree','-r','-z',tree,'--',*paths]).stdout
    entries = {}
    for row in raw.split(b'\0'):
        if not row:
            continue
        fields,name = row.split(b'\t',1)
        mode,kind,object_id = fields.decode().split()
        assert kind == 'blob'
        entries[name.decode()] = {'mode':mode,'object_id':object_id}
    requested = list(entries)
    raw = git(['cat-file','--batch'],input=''.join(entries[name]['object_id']+'\n' for name in requested).encode()).stdout
    position = 0
    for name in requested:
        end = raw.index(b'\n',position)
        object_id,kind,length = raw[position:end].decode().split()
        assert object_id == entries[name]['object_id'] and kind == 'blob'
        position = end+1
        length = int(length)
        entries[name]['raw'] = raw[position:position+length]
        position += length
        assert raw[position:position+1] == b'\n'
        position += 1
    assert position == len(raw)
    return entries


def syntax(raw):
    """Compare source syntax without importing any numerical module."""
    return {node.name:ast.dump(node,include_attributes=False) for node in ast.parse(raw).body
        if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef))}


def protected_snapshot(canonical):
    """Hash every canonical protected source or admission record in bounded pieces."""
    result = {}
    for name,expected in canonical['files'].items():
        path = Path(name)
        accumulator = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda:stream.read(1024*1024),b''):
                accumulator.update(block)
        value = {'path':name,'sha256':accumulator.hexdigest(),'size_bytes':path.stat().st_size,
            'mode':stat.S_IMODE(path.stat().st_mode)}
        assert all(value[key] == expected[key] for key in ('sha256','size_bytes','mode')),name
        result[name] = value
    return result


def main():
    """Approve only the pinned predicted tree and preserve the real index and all private sources."""
    assert os.environ['SLURM_JOB_ID'] == '6783003' and os.environ['SLURM_STEP_ID'].isdigit()
    assert git(['rev-parse','HEAD']).stdout.decode().strip() == CURRENT
    assert git(['rev-parse',INCOMING+'^{tree}']).stdout.decode().strip() == 'a7fc055afe3fb3743a3e7d2003e28a6bb3229e3b'
    roster_path = WORKTREE/'.review_metadata/direct_common_historical_consumer/reviewed_publication_roster.json'
    assert reference(roster_path)['sha256'] == '01f134c06b5911cbeccf8122a3ff7f380d50c605719147a9270d254af6d70638'
    roster = json.loads(roster_path.read_bytes())
    incoming_paths = git(['diff','--name-only','-z',BASE,INCOMING]).stdout.decode().strip('\0').split('\0')
    expected_paths = set(roster['publication_files']) | {str(roster_path.relative_to(WORKTREE))}
    assert set(incoming_paths) == expected_paths and len(expected_paths) == 196
    incoming = files(INCOMING,incoming_paths)
    for name,row in roster['publication_files'].items():
        entry = incoming[name]
        assert entry['mode'] == row['git_mode'] and len(entry['raw']) == row['bytes']
        assert digest(entry['raw']) == row['sha256'] == digest((WORKTREE/name).read_bytes()),name
    assert incoming[str(roster_path.relative_to(WORKTREE))]['raw'] == roster_path.read_bytes()
    assert reference(Path(roster['source_freeze']['path']))['sha256'] == '2c6d79b3c32b8e49d762400c92a5950218a02be0cbdcff7a46fc76a2d3cb33b9'
    assert reference(Path(roster['genuine_independent_approval']['path']))['sha256'] == '04ed5f4d569d6f31e2760c507325b46e8f431ce0e6b939f6a5ec824f9403c497'
    changed_current = set(git(['diff','--name-only','-z',BASE,CURRENT]).stdout.decode().strip('\0').split('\0'))
    overlap = changed_current.intersection(incoming_paths)
    assert overlap == {SHARED},overlap
    lists = {'dirty':git(['diff','--name-only','-z','HEAD']).stdout,
        'untracked':git(['ls-files','--others','--exclude-standard','-z']).stdout,
        'ignored':git(['ls-files','--others','--ignored','--exclude-standard','-z']).stdout}
    collisions = []
    counts = {}
    for kind,raw in lists.items():
        names = [name for name in raw.decode().split('\0') if name]
        counts[kind] = len(names)
        for name in names:
            for target in incoming_paths:
                if name == target or name.startswith(target+'/') or target.startswith(name+'/'):
                    collisions.append([kind,name,target])
    assert not collisions,collisions
    index_path = Path(git(['rev-parse','--git-path','index']).stdout.decode().strip())
    if not index_path.is_absolute():
        index_path = INTEGRATION/index_path
    actual_index_before = reference(index_path)
    canonical_path = RUN/'remote_setup/direct_nsight_incoming_independent_2026-11-04-20-01-56_protected_before_compatible_git.json'
    canonical = json.loads(canonical_path.read_bytes())
    protected_before = protected_snapshot(canonical)
    assert len(protected_before) == 662
    publish('protected_before_compatible_git.json',{'canonical':reference(canonical_path),'head':CURRENT,'files':protected_before})
    incoming_for_prediction = files(INCOMING,incoming_paths)
    current_shared = files(CURRENT,[SHARED])[SHARED]
    base_shared = files(BASE,[SHARED])[SHARED]
    with tempfile.TemporaryDirectory(prefix='historical_source_three_way_') as directory:
        temporary = Path(directory)
        for name,raw in [('current.py',current_shared['raw']),('base.py',base_shared['raw']),('incoming.py',incoming_for_prediction[SHARED]['raw'])]:
            (temporary/name).write_bytes(raw)
        merge_command = ['git','merge-file','-p',str(temporary/'current.py'),str(temporary/'base.py'),str(temporary/'incoming.py')]
        merge = subprocess.run(merge_command,capture_output=True)
        publish('predicted_merge_return_compatible_git.json',{'command':merge_command,'returncode':merge.returncode,
            'stdout_sha256':digest(merge.stdout),'stderr':merge.stderr.decode()})
        assert merge.returncode == 0
        shared_object = git(['hash-object','-w','--stdin'],input=merge.stdout).stdout.decode().strip()
        environment = dict(os.environ,GIT_INDEX_FILE=str(temporary/'prediction-index'))
        git(['read-tree',CURRENT],env=environment)
        rows = []
        for name,entry in incoming_for_prediction.items():
            object_id = shared_object if name == SHARED else entry['object_id']
            rows.append(entry['mode']+' '+object_id+'\t'+name+'\0')
        git(['update-index','-z','--index-info'],env=environment,input=''.join(rows).encode())
        predicted = git(['write-tree'],env=environment).stdout.decode().strip()
    assert len(predicted) == 40
    predicted_changes = set(git(['diff','--name-only','-z',CURRENT,predicted]).stdout.decode().strip('\0').split('\0'))
    assert predicted_changes == set(incoming_paths)
    with tempfile.TemporaryDirectory(prefix='historical_incoming_private_index_') as directory:
        temporary = Path(directory)
        copied_index = temporary/'copied-real-index'
        shutil.copyfile(index_path,copied_index)
        first_env = dict(os.environ,GIT_INDEX_FILE=str(copied_index))
        git(['read-tree','--dry-run','-m',CURRENT,predicted],env=first_env)
        fresh_env = dict(os.environ,GIT_INDEX_FILE=str(temporary/'fresh-index'))
        git(['read-tree',CURRENT],env=fresh_env)
        git(['read-tree','--dry-run','-m',CURRENT,predicted],env=fresh_env)
        git(['read-tree','-m',CURRENT,predicted],env=fresh_env)
        assert git(['write-tree'],env=fresh_env).stdout.decode().strip() == predicted
        predicted_files = files(predicted,[PREFIX])
        current_files = files(CURRENT,[PREFIX])
        for name,entry in current_files.items():
            if name not in incoming_paths:
                assert predicted_files[name]['object_id'] == entry['object_id'] and predicted_files[name]['mode'] == entry['mode'],name
        current_functions = syntax(current_files[SHARED]['raw'])
        merged_functions = syntax(predicted_files[SHARED]['raw'])
        incoming_functions = syntax(incoming[SHARED]['raw'])
        for name,raw in current_functions.items():
            if name != 'verify_qualification_inputs':
                assert merged_functions[name] == raw, name
        assert merged_functions['historical_support'] == incoming_functions['historical_support']
        assert merged_functions['verify_qualification_inputs'] == incoming_functions['verify_qualification_inputs']
        assert syntax(current_files[PREFIX+'profile_selected.py']['raw'])['_capture_core'] == syntax(predicted_files[PREFIX+'profile_selected.py']['raw'])['_capture_core']
        for name in ('_compute','run_prepared','save_measurement_result','producer_support','card_admission'):
            assert syntax(current_files[PREFIX+'measure_selected.py']['raw'])[name] == syntax(predicted_files[PREFIX+'measure_selected.py']['raw'])[name]
        exported = temporary/'merged_sources'
        for name,entry in predicted_files.items():
            if name.endswith('.py'):
                path = exported/name
                path.parent.mkdir(parents=True,exist_ok=True)
                path.write_bytes(entry['raw'])
        cold = """import sys,json,ctypes
from pathlib import Path
forbidden={'jax','jaxlib','numpy','torch','flax','optax','cupy','scipy','exploration_platform','jaxtari','nvidia','jax_cuda12_plugin','ml_dtypes'}
def audit(event,values):
    if event=='import' and values[0].split('.')[0] in forbidden:
        raise RuntimeError('numerical import forbidden in merged metadata source check')
    if event in {'ctypes.dlopen','ctypes.dlsym','socket.connect','socket.bind','socket.getaddrinfo','subprocess.Popen'}:
        raise RuntimeError('native, network or child operation forbidden in merged metadata source check')
sys.addaudithook(audit)
sys.path.insert(0,sys.argv[1])
import measure_selected as m
import profile_selected as p
common=m.required_helpers(direct=True,common=True,backend_image=True)
profile=p.required_helpers(direct=True,direct_profile=True)
assert len(common)==28 and len(profile)==32
assert all((Path(sys.argv[1])/name).is_file() for name in set(common)|set(profile))
assert not forbidden.intersection(sys.modules)
print(json.dumps({'common':list(common),'profile':list(profile),'numerical_imports':[],
    'stdlib_own_process_ctypes_bootstrap_before_audit':True}))
"""
        result = subprocess.run(['/usr/bin/python3','-I','-S','-B','-c',cold,str(exported/PREFIX)],capture_output=True)
        publish('merged_cold_helper_import_compatible_git.json',{'returncode':result.returncode,'stdout':result.stdout.decode(),'stderr':result.stderr.decode()})
        assert result.returncode == 0 and not result.stderr
        closure = json.loads(result.stdout)
    protected_after = protected_snapshot(canonical)
    assert protected_after == protected_before
    assert reference(index_path) == actual_index_before
    assert git(['rev-parse','HEAD']).stdout.decode().strip() == CURRENT
    publish('independent_incoming_review_compatible_git.json',{'schema_version':1,'kind':'independent_historical_consumer_incoming_tree_review',
        'passed':True,'reviewer':'/root/review_historical_consumer_bridge','actual_job':'6783003','actual_step':os.environ['SLURM_STEP_ID'],
        'incoming_commit':INCOMING,'reviewed_current_head':CURRENT,'predicted_tree':predicted,'incoming_paths':196,
        'source_overlap':[SHARED],'incoming_blobs_modes_and_reviewed_bytes_verified':True,'fresh_collision_counts':counts,
        'dirty_untracked_ignored_ancestor_collisions':collisions,'actual_private_index_dry_runs_passed':2,
        'private_index_result_tree_matches_prediction':True,'actual_index_unchanged':True,'protected_files_unchanged':662,
        'all_current_source_blobs_preserved_except_declared_shared_additions':True,'current_profile_functions_and_capture_core_preserved':True,
        'incoming_historical_support_and_qualification_dispatch_preserved':True,'merged_common_helper_count':len(closure['common']),
        'merged_direct_profile_helper_count':len(closure['profile']),'merged_cold_metadata_import_passed':True,
        'future_unadmitted_profile_gap':'direct_profile_prerequisites.py still requires a 27-helper accepted confirmation at lines222 and389. Newly prepared common confirmation uses28. Before profiling preparation/admission, separately review the exact historical 28-helper consumer and its staged dependencies; do not merely change the count or invent evidence.',
        'source_roster':reference(roster_path),'protected_before':reference(OUTPUT/'protected_before_compatible_git.json'),
        'commands':COMMANDS,'saved_scientific_evidence_recomputed':False,'production_guard_success_established':False,
        'scientific_credit':0,'admission_granted':False,'requires_fresh_head_index_and_collision_check_before_actual_merge':True,
        'finished_at_seconds':time.time()})
    print(json.dumps(reference(OUTPUT/'independent_incoming_review_compatible_git.json'),sort_keys=True))


if __name__ == '__main__':
    try:
        main()
    except BaseException as error:
        publish('failed_attempt_compatible_git.json',{'type':type(error).__name__,'message':str(error),'commands':COMMANDS})
        raise
