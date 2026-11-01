"""Verify the exact direct card-one core merge with ordinary reads and separate Git index/object directories."""

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
from zoneinfo import ZoneInfo


INTEGRATION = Path('/p/rlprojects/RND_worktrees/merge-atari-cleanrl')
ORIGINAL = Path('/p/rlprojects/RND')
AUTHOR = Path('/p/rlprojects/RND_worktrees/2026-11-01-11-40_serval02-card-1-current-device-binding_gpu-count-1_driver-590.48.01_memory-mib-95830')
RELATIVE_RUN = '10_jax_exploration_platform/benchmark_runs/2026-10-31-00-11_jaxatari_montezuma_pixels_cleanrl_ppo_rnd_environments-per-copy-128_rollout-steps-128_ideas-30-plus-20_steps-per-copy-2000000000_seed-start-1'
RUN = INTEGRATION / RELATIVE_RUN
BASE = '0c42861e4ac209138ead5253fb8d1612cd3d3e3c'
INCOMING = 'e41e3da92508cea1be28c5d45ea13db8d930891f'
NUMERICAL = 'd28e6735e79788f618acb0ebfde8609db1f776af'
TEMPORARY = Path(__file__).resolve().parent
ENVIRONMENT = dict(os.environ, GIT_OPTIONAL_LOCKS='0', GIT_TERMINAL_PROMPT='0')
COMMANDS = []


def write(name, raw):
    """Retain exact new review bytes without overwriting a previous observation."""
    raw = raw if isinstance(raw, bytes) else (json.dumps(raw, indent=2) + '\n').encode()
    with (TEMPORARY / name).open('xb') as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    return {'relative_path': name, 'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}


def git(arguments, *, environment=None, input_bytes=None, allowed=(0,), repository=INTEGRATION):
    """Read shared Git data while placing every requested index/object write in private temporary paths."""
    argv = ['git', '--no-optional-locks', '-C', str(repository), *arguments]
    result = subprocess.run(argv, env=environment or ENVIRONMENT, input=input_bytes,
                            capture_output=True, timeout=120)
    number = len(COMMANDS)
    COMMANDS.append({'argv': argv, 'returncode': result.returncode,
                     'at_utc': datetime.now(timezone.utc).isoformat(),
                     'stdout': write(f'command_{number:03d}_stdout.raw', result.stdout),
                     'stderr': write(f'command_{number:03d}_stderr.raw', result.stderr),
                     'temporary_git_environment': {name: value for name, value in (environment or {}).items()
                         if name in {'GIT_INDEX_FILE', 'GIT_OBJECT_DIRECTORY', 'GIT_ALTERNATE_OBJECT_DIRECTORIES'}}})
    if result.returncode not in allowed:
        write('command_failure.json', COMMANDS)
        raise RuntimeError('A bounded Git review command failed: ' + repr(argv))
    return result.stdout, result.returncode


def file_record(path):
    """Record ordinary bytes, executable mode and literal symlink targets without following the final link."""
    try:
        before = path.lstat()
    except FileNotFoundError:
        return {'kind': 'absent'}
    result = {'mode': oct(stat.S_IMODE(before.st_mode)), 'inode': before.st_ino, 'device': before.st_dev,
              'size': before.st_size, 'mtime_ns': before.st_mtime_ns, 'ctime_ns': before.st_ctime_ns}
    if stat.S_ISLNK(before.st_mode):
        raw = os.fsencode(os.readlink(path))
        return dict(result, kind='symlink', target=raw.hex(), sha256=hashlib.sha256(raw).hexdigest())
    if stat.S_ISDIR(before.st_mode):
        return dict(result, kind='directory')
    if not stat.S_ISREG(before.st_mode):
        return dict(result, kind='special_not_read')
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    digest = hashlib.sha256()
    with os.fdopen(descriptor, 'rb') as stream:
        opened = os.fstat(stream.fileno())
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
        after = os.fstat(stream.fileno())
    current = path.lstat()
    result.update(kind='file', sha256=digest.hexdigest(), stable=(before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns)
        == (opened.st_dev, opened.st_ino, opened.st_size, opened.st_mtime_ns, opened.st_ctime_ns)
        == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns)
        == (current.st_dev, current.st_ino, current.st_size, current.st_mtime_ns, current.st_ctime_ns))
    return result


def tree(commit, environment=None):
    """Read exact ordinary Git tree entries as path-to-mode/blob records."""
    value = {}
    for item in git(['ls-tree', '-rz', commit], environment=environment)[0].split(b'\0'):
        if item:
            header, name = item.split(b'\t', 1)
            mode, kind, blob = header.decode().split()
            value[os.fsdecode(name)] = {'mode': mode, 'kind': kind, 'blob': blob}
    return value


def dirty_snapshot(label, root=INTEGRATION):
    """Preserve current integration dirty/untracked files and both names in rename records."""
    raw = git(['status', '--porcelain=v1', '-z', '--untracked-files=all'], repository=root)[0]
    values, position, rows = raw.split(b'\0'), 0, []
    while position < len(values):
        entry = values[position]
        position += 1
        if not entry:
            continue
        row = {'status': entry[:2].decode(), 'path': os.fsdecode(entry[3:])}
        if b'R' in entry[:2] or b'C' in entry[:2]:
            row['original_path'] = os.fsdecode(values[position])
            position += 1
        rows.append(row)
    paths = {row['path'] for row in rows} | {row['original_path'] for row in rows if 'original_path' in row}
    result = {'at_utc': datetime.now(timezone.utc).isoformat(), 'status': rows,
              'files': {name: file_record(root / name) for name in sorted(paths)}}
    write(label + '.json', result)
    return result


def main():
    """Verify the reviewed bytes, collision set and actual three-way resulting tree without checkout/ref mutation."""
    assert os.environ['SLURM_JOB_ID'] == '6781231' and os.environ['SLURM_STEP_ID'].isdigit()
    print('owned review step ' + os.environ['SLURM_JOB_ID'] + '.' + os.environ['SLURM_STEP_ID'], flush=True)
    roster_path = AUTHOR / '.review_metadata/direct_serval02_core_publication_roster.json'
    roster_raw = roster_path.read_bytes()
    assert hashlib.sha256(roster_raw).hexdigest() == 'd1840c0dc79e26a3d4969627b818d34a1396221ccf995d9757f5f6328c26fac4'
    roster = json.loads(roster_raw)
    assert roster['base_commit'] == BASE and roster['payload_path_count'] == len(roster['files']) == 48
    assert roster['total_commit_path_count'] == 49
    payload = {row['path']: row['sha256'] for row in roster['files']}
    assert len(payload) == 48
    payload[roster['roster_path']] = hashlib.sha256(roster_raw).hexdigest()
    roster['files'] = payload
    prefix = '10_jax_exploration_platform/scripts/atari_pixel_campaign/'
    production = {prefix + 'serval02_card1_binding.py', prefix + 'tests/test_serval02_card1_binding.py'}
    assert len([name for name in payload if not name.startswith('.review_metadata/')]) == 2
    assert production == {name for name in payload if not name.startswith('.review_metadata/')}
    publication_path = AUTHOR / '.review_metadata/direct_serval02_core_author_publication_receipt.json'
    assert hashlib.sha256(publication_path.read_bytes()).hexdigest() == '28f5979958930114765a08b3e2041112ed17b45b14591e2f814d04bca73585b5'
    publication = json.loads(publication_path.read_text())
    assert publication['commit'] == INCOMING and publication['parent'] == BASE and publication['path_count'] == 49
    assert {row['path']: row['sha256'] for row in publication['files']} == payload
    independent_path = AUTHOR / roster['final_independent_review']['path']
    assert hashlib.sha256(independent_path.read_bytes()).hexdigest() == roster['final_independent_review']['sha256'] == '2cd444f812c080eb9e5d86f888b3305ed78b6494e075bfaa214f6e9cbc0fad99'
    freeze_path = AUTHOR / '.review_metadata/direct_serval02_core_sixth_source_freeze.json'
    assert hashlib.sha256(freeze_path.read_bytes()).hexdigest() == roster['source_freeze_sha256'] == 'af07c0e781890e438fb4235ba9a24e475041e17a1fa4147ae3f4f00a67a7a6a0'
    historical_scan = RUN / 'remote_setup/2026-11-01-07-50_original-checkout-preservation_target-cef3073939e0_dirty-files-pdfs-and-incoming-paths/actual_original_fast_forward_to_cef3073/after/scan_after.json'
    assert hashlib.sha256(historical_scan.read_bytes()).hexdigest() == '6a850aef02b80862ae2e9617455ba5d7479e3d71ecf910c55660b24f63ad48b6'
    historical = json.loads(historical_scan.read_text())
    numerical_root = Path('/p/rlprojects/RND_worktrees/atari-cleanrl-source-d28e673')
    assert git(['rev-parse', 'HEAD'], repository=numerical_root)[0].decode().strip() == NUMERICAL
    assert not git(['status', '--porcelain=v1', '-z', '--untracked-files=all'], repository=numerical_root)[0]
    assert git(['rev-list', '--parents', '-n', '1', INCOMING])[0].decode().split() == [INCOMING, BASE]
    head = git(['rev-parse', 'HEAD'])[0].decode().strip()
    assert head == '2813b6b5e39d247ff33893386dd77afe6f8ac758'
    branch = git(['symbolic-ref', '--short', 'HEAD'])[0].decode().strip()
    assert branch == 'merge-atari-cleanrl'
    assert git(['merge-base', '--is-ancestor', BASE, head], allowed=(0, 1))[1] == 0
    index = Path(git(['rev-parse', '--git-path', 'index'])[0].decode().strip())
    index_before, original_index_before = file_record(index), file_record(ORIGINAL / '.git/index')
    original_head_before = (ORIGINAL / '.git/HEAD').read_bytes()
    before = dirty_snapshot('integration_before')
    original_before = dirty_snapshot('original_before', ORIGINAL)
    original_head_commit = git(['rev-parse', 'HEAD'], repository=ORIGINAL)[0].decode().strip()
    original_staged = git(['diff', '--cached', '--binary', '--no-ext-diff'], repository=ORIGINAL)[0]
    pdf_paths = {os.fsdecode(name) for name in git(['ls-files', '-z', '--', '*.pdf'], repository=ORIGINAL)[0].split(b'\0') if name}
    pdf_paths |= {name for name in original_before['files'] if name.lower().endswith('.pdf')}
    for name in ('07_reconstruction/development_document/RND_development_document.pdf', '10_jax_exploration_platform/development_document/RND_JAX_exploration_development_document.pdf'):
        if (ORIGINAL / name).exists(): pdf_paths.add(name)
    pdf_paths |= set(historical['pdf_paths'])
    all_original_paths = set(historical['files']) | set(original_before['files']) | pdf_paths
    original_before['files'] = {name: file_record(ORIGINAL / name) for name in sorted(all_original_paths)}
    write('original_all_files_before.json', original_before)
    original_pdfs_before = {name: original_before['files'][name] for name in sorted(pdf_paths)}
    five = historical['historical_paths']
    assert len(five) == 5
    historical_five_before = {name: original_before['files'][name] for name in five}
    assert all(value.get('sha256') == historical['files'][name].get('sha256') for name, value in historical_five_before.items())
    submodules_before = {}
    for row in original_before['status']:
        path = ORIGINAL / row['path']
        if path.is_dir() and (path / '.git').exists():
            submodules_before[row['path']] = {'head': git(['rev-parse', 'HEAD'], repository=path)[0].decode().strip(),
                'dirty': dirty_snapshot('original_submodule_before_' + str(len(submodules_before)), path), 'index': file_record(Path(git(['rev-parse', '--git-path', 'index'], repository=path)[0].decode().strip()))}
            all_submodule = [os.fsdecode(name) for name in git(['ls-files', '-c', '-o', '--exclude-standard', '-z'], repository=path)[0].split(b'\0') if name]
            submodules_before[row['path']]['all_current_files'] = {name: file_record(path / name) for name in sorted(set(all_submodule))}
    admission_paths = [RUN / 'SOURCE_AND_GATES_READY.json', *[RUN / 'data/source_checks' / NUMERICAL / (context + '.json') for context in ('h100', 'jaguar03_rtx_a4500')]]
    admission_before = {str(path): file_record(path) for path in admission_paths}
    staged = git(['diff', '--cached', '--binary', '--no-ext-diff'])[0]
    base_tree, current_tree, incoming_tree = tree(BASE), tree(head), tree(INCOMING)
    changed = {name for name in base_tree.keys() | incoming_tree.keys() if base_tree.get(name) != incoming_tree.get(name)}
    assert changed == set(roster['files'])
    overlap = {name for name in changed if base_tree.get(name) != current_tree.get(name)}
    incoming_bytes, collisions, symlinks = {}, [], []
    tracked = set(current_tree)
    assert all(name not in current_tree for name in changed), 'all49 direct-core paths are expected to be new'
    for name, checksum in roster['files'].items():
        entry = incoming_tree[name]
        assert entry['kind'] == 'blob' and entry['mode'] in {'100644', '100755'}
        raw = git(['cat-file', 'blob', entry['blob']])[0]
        assert hashlib.sha256(raw).hexdigest() == checksum
        assert (AUTHOR / name).read_bytes() == raw
        incoming_bytes[name] = {'sha256': checksum, 'git': entry}
        path = INTEGRATION / name
        observation = file_record(path)
        if observation['kind'] != 'absent' and name not in tracked:
            collisions.append({'path': name, 'working': observation})
        if observation['kind'] == 'symlink':
            symlinks.append(name)
        for parent in path.parents:
            if parent == INTEGRATION.parent:
                break
            information = file_record(parent)
            if information['kind'] not in {'directory', 'absent'}:
                symlinks.append(str(parent))
    candidates = sorted({str(parent) for name in changed for parent in [Path(name), *Path(name).parents] if str(parent) != '.'})
    ignored = git(['check-ignore', '--no-index', '-z', '--stdin'], input_bytes=b'\0'.join(os.fsencode(name) for name in candidates) + b'\0', allowed=(0, 1))[0]
    ignored_paths = [os.fsdecode(name) for name in ignored.split(b'\0') if name]
    dirty_overlap = [{'incoming': name, 'dirty': dirty} for name in changed for dirty in before['files']
                     if name == dirty or name.startswith(dirty + '/') or dirty.startswith(name + '/')]

    # The shared object store is read-only; new tree objects and both copied indices stay in this private directory.
    common = Path(git(['rev-parse', '--git-common-dir'])[0].decode().strip())
    if not common.is_absolute():
        common = (INTEGRATION / common).resolve()
    objects = TEMPORARY / 'isolated_objects'
    objects.mkdir()
    alternate = dict(ENVIRONMENT, GIT_OBJECT_DIRECTORY=str(objects), GIT_ALTERNATE_OBJECT_DIRECTORIES=str(common / 'objects'))
    dry_index = TEMPORARY / 'checkout_dry_run.index'
    shutil.copyfile(index, dry_index)
    dry_environment = dict(alternate, GIT_INDEX_FILE=str(dry_index))
    dry_return = git(['read-tree', '--dry-run', '-m', '-u', BASE, head, INCOMING], environment=dry_environment, allowed=(0, 1, 128))[1]
    merged_index = TEMPORARY / 'merged_result.index'
    shutil.copyfile(index, merged_index)
    merged_environment = dict(alternate, GIT_INDEX_FILE=str(merged_index))
    merge_return = git(['read-tree', '-m', BASE, head, INCOMING], environment=merged_environment, allowed=(0, 1, 128))[1]
    unmerged = git(['ls-files', '-u', '-z'], environment=merged_environment)[0]
    result_tree = None
    result_difference = []
    merged_paths = {}
    if merge_return == 0 and not unmerged:
        result_tree = git(['write-tree'], environment=merged_environment)[0].decode().strip()
        merged_paths = tree(result_tree, merged_environment)
        result_difference = sorted(name for name in current_tree.keys() | merged_paths.keys() if current_tree.get(name) != merged_paths.get(name))
        assert set(result_difference) == changed
        assert all(merged_paths[name] == incoming_tree[name] for name in changed)
        assert all(merged_paths.get(name) == value for name, value in current_tree.items() if name not in changed)

    # The original pixel implementation and paired helpers must remain exact D28 bytes in every proposed tree.
    reference = tree(NUMERICAL)
    protected = [name for name in reference if name.startswith('10_jax_exploration_platform/src/exploration_platform/atari_pixels/')
        or name in {'10_jax_exploration_platform/scripts/atari_pixel_campaign/paired_benchmark.py',
                    '10_jax_exploration_platform/scripts/atari_pixel_campaign/ideas.py',
                    '10_jax_exploration_platform/scripts/atari_pixel_campaign/check_rendering.py'}]
    protected_files = {}
    for name in protected:
        assert current_tree.get(name) == incoming_tree.get(name) == merged_paths.get(name) == reference[name]
        raw = git(['cat-file', 'blob', reference[name]['blob']])[0]
        checksum = hashlib.sha256(raw).hexdigest()
        assert hashlib.sha256((INTEGRATION / name).read_bytes()).hexdigest() == checksum
        assert hashlib.sha256((ORIGINAL / name).read_bytes()).hexdigest() == checksum
        assert hashlib.sha256((numerical_root / name).read_bytes()).hexdigest() == checksum
        protected_files[name] = checksum
    remote = git(['ls-remote', '--heads', 'origin', 'refs/heads/' + roster['branch']])[0]
    assert remote == (INCOMING + '\trefs/heads/' + roster['branch'] + '\n').encode()
    after = dirty_snapshot('integration_after')
    after_head = git(['rev-parse', 'HEAD'])[0].decode().strip()
    assert head == '2813b6b5e39d247ff33893386dd77afe6f8ac758'
    index_after, original_index_after = file_record(index), file_record(ORIGINAL / '.git/index')
    original_after = dirty_snapshot('original_after', ORIGINAL)
    original_after['files'] = {name: file_record(ORIGINAL / name) for name in sorted(all_original_paths | set(original_after['files']))}
    write('original_all_files_after.json', original_after)
    historical_five_after = {name: original_after['files'][name] for name in five}
    original_head_after = git(['rev-parse', 'HEAD'], repository=ORIGINAL)[0].decode().strip()
    original_staged_after = git(['diff', '--cached', '--binary', '--no-ext-diff'], repository=ORIGINAL)[0]
    original_pdfs_after = {name: file_record(ORIGINAL / name) for name in sorted(pdf_paths)}
    submodules_after = {}
    for name in submodules_before:
        path = ORIGINAL / name
        submodules_after[name] = {'head': git(['rev-parse', 'HEAD'], repository=path)[0].decode().strip(),
            'dirty': dirty_snapshot('original_submodule_after_' + str(len(submodules_after)), path), 'index': file_record(Path(git(['rev-parse', '--git-path', 'index'], repository=path)[0].decode().strip()))}
        all_submodule = [os.fsdecode(member) for member in git(['ls-files', '-c', '-o', '--exclude-standard', '-z'], repository=path)[0].split(b'\0') if member]
        submodules_after[name]['all_current_files'] = {member: file_record(path / member) for member in sorted(set(all_submodule))}
    admission_after = {str(path): file_record(path) for path in admission_paths}
    assert git(['rev-parse', 'HEAD'], repository=numerical_root)[0].decode().strip() == NUMERICAL
    assert not git(['status', '--porcelain=v1', '-z', '--untracked-files=all'], repository=numerical_root)[0]
    concurrent = {name: {'before': before['files'].get(name), 'after': after['files'].get(name)}
                  for name in before['files'].keys() | after['files'].keys() if before['files'].get(name) != after['files'].get(name)}
    after_overlap = [{'incoming': name, 'dirty': dirty} for name in changed for dirty in after['files']
                     if name == dirty or name.startswith(dirty + '/') or dirty.startswith(name + '/')]
    eligible = (not staged and not overlap and not collisions and not symlinks and not dirty_overlap and not after_overlap
                and dry_return == merge_return == 0 and not unmerged and result_tree is not None
                and index_before == index_after and original_index_before == original_index_after
                and original_head_before == (ORIGINAL / '.git/HEAD').read_bytes() and head == after_head
                and original_head_commit == original_head_after and original_staged == original_staged_after
                and original_before['files'] == original_after['files'] and original_pdfs_before == original_pdfs_after
                and submodules_before.keys() == submodules_after.keys()
                and all(submodules_before[name]['head'] == submodules_after[name]['head'] and submodules_before[name]['dirty']['files'] == submodules_after[name]['dirty']['files'] for name in submodules_before)
                and admission_before == admission_after and historical_five_before == historical_five_after
                and all(submodules_before[name]['all_current_files'] == submodules_after[name]['all_current_files'] and submodules_before[name]['index'] == submodules_after[name]['index'] for name in submodules_before))
    record = {'schema_version': 1, 'kind': 'independent_direct_card_one_core_incoming_commit_tree_review',
        'at_utc': datetime.now(timezone.utc).isoformat(), 'approved_for_exact_integration_merge': eligible,
        'incoming_commit': INCOMING, 'base': BASE, 'integration_head_before': head, 'integration_head_after': after_head,
        'integration_branch': branch, 'author_roster_sha256': hashlib.sha256(roster_raw).hexdigest(),
        'actual_pushed_author_ref': remote.decode(), 'incoming_path_count': len(changed), 'incoming_bytes': incoming_bytes,
        'committed_path_overlaps': sorted(overlap), 'dirty_incoming_overlaps_before': dirty_overlap,
        'dirty_incoming_overlaps_after': after_overlap, 'ignored_paths_checked_only_for_incoming_and_parents': ignored_paths,
        'untracked_or_ignored_existing_path_collisions': collisions, 'symlink_or_non_directory_parent_collisions': symlinks,
        'staged_diff_empty': not staged, 'integration_index_before': index_before, 'integration_index_after': index_after,
        'integration_index_unchanged': index_before == index_after, 'original_index_unchanged': original_index_before == original_index_after,
        'original_HEAD_bytes_unchanged': original_head_before == (ORIGINAL / '.git/HEAD').read_bytes(),
        'dirty_paths_before': len(before['status']), 'dirty_paths_after': len(after['status']), 'concurrent_outside_incoming_changes': concurrent,
        'checkout_three_way_dry_run_returncode': dry_return, 'index_only_three_way_read_tree_returncode': merge_return,
        'unmerged_index_entries': unmerged.decode(), 'result_tree': result_tree, 'result_paths_changed_from_integration': result_difference,
        'all_other_integration_tree_entries_identical': True, 'all_incoming_result_entries_exact': True,
        'isolated_object_directory': str(objects), 'existing_object_store_read_only_alternate': str(common / 'objects'),
        'protected_D28_numerical_and_paired_files': protected_files, 'historical_five_before': historical_five_before, 'historical_five_after': historical_five_after,
        'complete_original_scanned_paths_before': len(original_before['files']), 'complete_original_scanned_paths_after': len(original_after['files']),
        'original_discovery_scope': 'Full previous6454path set plus fresh all-current dirty/untracked paths,646known/tracked/current PDFs including ignored main PDF; all current tracked/untracked RLeXplore files separately. All bytes read and rehashed before and after; no abbreviated scan.',
        'original_pdf_count': len(pdf_paths), 'owned_cpu_job': os.environ['SLURM_JOB_ID'], 'step_id': os.environ['SLURM_STEP_ID'],
        'author_publication_receipt_sha256': hashlib.sha256(publication_path.read_bytes()).hexdigest(), 'independent_core_review_sha256': hashlib.sha256(independent_path.read_bytes()).hexdigest(), 'source_admission_changed': False,
        'actual_merge_or_ref_update_performed': False, 'original_or_integration_worktree_writes': False,
        'numerical_library_device_cluster_ssh_or_job_actions': False, 'github_remote_ref_read_performed': True,
        'original_dirty_files_before': original_before['files'], 'original_dirty_files_after': original_after['files'],
        'original_head_commit_before': original_head_commit, 'original_head_commit_after': original_head_after,
        'original_staged_diff_unchanged': original_staged == original_staged_after,
        'original_tracked_current_dirty_and_known_PDFs_before': original_pdfs_before, 'original_tracked_current_dirty_and_known_PDFs_after': original_pdfs_after,
        'original_submodules_before': submodules_before, 'original_submodules_after': submodules_after,
        'source_admission_before': admission_before, 'source_admission_after': admission_after,
        'git_status_index_refresh': 'GIT_OPTIONAL_LOCKS=0 and --no-optional-locks; actual original and integration index bytes/stat checked unchanged', 'auxiliary_Slurm_steps': 1, 'existing_owned_cpu_step_only': True,
        'proposed_parent_commands': [['git', '-C', str(INTEGRATION), 'merge', '--no-ff', '--no-edit', INCOMING],
            ['git', '-C', str(INTEGRATION), 'rev-parse', 'HEAD^{tree}']],
        'expected_result_tree_after_parent_merge': result_tree}
    write('commands.json', COMMANDS)
    write('review.json', record)
    receipt = TEMPORARY / 'review.json'
    print(json.dumps({'approved': eligible, 'receipt': str(receipt), 'sha256': hashlib.sha256(receipt.read_bytes()).hexdigest(),
        'integration_head': head, 'result_tree': result_tree, 'incoming_paths': len(changed),
        'collisions': len(collisions), 'dirty_overlaps': len(dirty_overlap), 'dry_run': dry_return, 'read_tree': merge_return}))


if __name__ == '__main__':
    try:
        main()
    except BaseException as error:
        import traceback
        write('review_failure.json', {'owned_cpu_job': os.environ.get('SLURM_JOB_ID'), 'step_id': os.environ.get('SLURM_STEP_ID'),
            'error': str(error), 'traceback': traceback.format_exc(), 'actual_ref_or_index_update_performed': False})
        raise
