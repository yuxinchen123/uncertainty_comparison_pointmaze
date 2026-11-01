"""Verify the exact original qualification pytest-root correction merge with ordinary reads and separate Git index/object directories."""

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
AUTHOR = Path('/p/rlprojects/RND_worktrees/2026-11-01-13-35_original-qualification-pytest-root_source-commit-c851fa90f7b25d6ced54b7bac31457106ac91aa3')
RELATIVE_RUN = '10_jax_exploration_platform/benchmark_runs/2026-10-31-00-11_jaxatari_montezuma_pixels_cleanrl_ppo_rnd_environments-per-copy-128_rollout-steps-128_ideas-30-plus-20_steps-per-copy-2000000000_seed-start-1'
RUN = INTEGRATION / RELATIVE_RUN
BASE = 'c851fa90f7b25d6ced54b7bac31457106ac91aa3'
INCOMING = 'e5afbf6b51a3d1666b46e71c0631a45d47080cd8'
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
    assert os.environ.get('SLURM_JOB_ID') == '6781231' and os.environ.get('SLURM_STEP_ID', '').isdigit()
    roster_path = AUTHOR / '.review_metadata/qualification_pytest_root_publication_roster.json'
    roster_raw = roster_path.read_bytes()
    assert hashlib.sha256(roster_raw).hexdigest() == 'c39f1b0a19ff98ef59908525f352c1f03cfa46166178d16617d8319c0585382e'
    roster = json.loads(roster_raw)
    assert roster['base_commit'] == BASE and roster['total_commit_path_count'] == 13
    assert roster['owned_evidence_path_count'] == 8 and len(roster['source_test_fixture_paths']) == 4
    assert len(roster['files']) == roster['payload_path_count'] == 12
    roster['files'][roster['roster_path']] = hashlib.sha256(roster_raw).hexdigest()
    freeze_path = AUTHOR / '.review_metadata/qualification_pytest_root_source_freeze.json'
    assert hashlib.sha256(freeze_path.read_bytes()).hexdigest() == '64d9d4ad2474f26299ec807579f1ee128315b0c6f74167c5e411bca9cf79c4d8'
    freeze = json.loads(freeze_path.read_bytes())
    assert freeze['files'] == {name: roster['files'][name] for name in roster['source_test_fixture_paths']}
    assert all(name in roster['source_test_fixture_paths'] or name.startswith('.review_metadata/') for name in roster['files'])
    prior_audit = RUN / 'remote_setup/2026-11-01-13-43-12_failed-full-original-qualification-independent-review_job-6782640_gpu-count-0_processor-count-1/review.json'
    assert hashlib.sha256(prior_audit.read_bytes()).hexdigest() == '2ecb365a6277c58353ae4de9b083081ded6769ecb48d5358045d788b4ba8c3f6'
    prior = json.loads(prior_audit.read_bytes())
    assert prior['review_job_id'] == '6781231' and prior['review_step_id'] == '98'
    assert prior['target_job_id'] == '6782640' and prior['target_qualification_passed'] is False
    original_packet = prior['original55_before']
    packet_before = {name: file_record(Path(name)) for name in original_packet}
    assert len(packet_before) == 55 and all(value['sha256'] == original_packet[name] for name,value in packet_before.items())
    unit = Path(next(name for name in prior['raw_files'] if name.endswith('/original_tests.json'))).parent
    failed_paths = sorted(path for path in unit.rglob('*') if path.is_file())
    failed_before = {str(path): file_record(path) for path in failed_paths}
    write('actual_failed_unit_before.json', failed_before)
    write('original55_before.json', packet_before)
    write('prior_actual_failed_qualification_audit.json', prior_audit.read_bytes())
    write('reviewed_publication_roster.json', roster_raw)
    assert git(['rev-list', '--parents', '-n', '1', INCOMING])[0].decode().split() == [INCOMING, BASE]
    head = git(['rev-parse', 'HEAD'])[0].decode().strip()
    assert head == 'c851fa90f7b25d6ced54b7bac31457106ac91aa3'
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
    for name in ('07_reconstruction/development_document/RND_development_document.pdf', '10_jax_exploration_platform/development_document/jax_exploration_platform_development_document.pdf'):
        if (ORIGINAL / name).exists(): pdf_paths.add(name)
    original_pdfs_before = {name: file_record(ORIGINAL / name) for name in sorted(pdf_paths)}
    submodules_before = {}
    for row in original_before['status']:
        path = ORIGINAL / row['path']
        if path.is_dir() and (path / '.git').exists():
            submodules_before[row['path']] = {'head': git(['rev-parse', 'HEAD'], repository=path)[0].decode().strip(),
                'dirty': dirty_snapshot('original_submodule_before_' + str(len(submodules_before)), path)}
    admission_paths = [RUN / 'SOURCE_AND_GATES_READY.json', *[RUN / 'data/source_checks' / NUMERICAL / (context + '.json') for context in ('h100', 'jaguar03_rtx_a4500')]]
    admission_before = {str(path): file_record(path) for path in admission_paths}
    staged = git(['diff', '--cached', '--binary', '--no-ext-diff'])[0]
    base_tree, current_tree, incoming_tree = tree(BASE), tree(head), tree(INCOMING)
    changed = {name for name in base_tree.keys() | incoming_tree.keys() if base_tree.get(name) != incoming_tree.get(name)}
    assert changed == set(roster['files'])
    overlap = {name for name in changed if base_tree.get(name) != current_tree.get(name)}
    incoming_bytes, collisions, symlinks = {}, [], []
    tracked = set(current_tree)
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
                    '10_jax_exploration_platform/scripts/atari_pixel_campaign/ideas.py'}]
    protected_files = {}
    for name in protected:
        assert current_tree.get(name) == incoming_tree.get(name) == merged_paths.get(name) == reference[name]
        raw = git(['cat-file', 'blob', reference[name]['blob']])[0]
        checksum = hashlib.sha256(raw).hexdigest()
        assert hashlib.sha256((INTEGRATION / name).read_bytes()).hexdigest() == checksum
        assert hashlib.sha256((ORIGINAL / name).read_bytes()).hexdigest() == checksum
        protected_files[name] = checksum
    remote = git(['ls-remote', '--heads', 'origin', 'refs/heads/' + roster['branch']])[0]
    assert remote == (INCOMING + '\trefs/heads/' + roster['branch'] + '\n').encode()
    after = dirty_snapshot('integration_after')
    after_head = git(['rev-parse', 'HEAD'])[0].decode().strip()
    assert head == 'c851fa90f7b25d6ced54b7bac31457106ac91aa3'
    index_after, original_index_after = file_record(index), file_record(ORIGINAL / '.git/index')
    original_after = dirty_snapshot('original_after', ORIGINAL)
    original_head_after = git(['rev-parse', 'HEAD'], repository=ORIGINAL)[0].decode().strip()
    original_staged_after = git(['diff', '--cached', '--binary', '--no-ext-diff'], repository=ORIGINAL)[0]
    original_pdfs_after = {name: file_record(ORIGINAL / name) for name in sorted(pdf_paths)}
    submodules_after = {}
    for name in submodules_before:
        path = ORIGINAL / name
        submodules_after[name] = {'head': git(['rev-parse', 'HEAD'], repository=path)[0].decode().strip(),
            'dirty': dirty_snapshot('original_submodule_after_' + str(len(submodules_after)), path)}
    admission_after = {str(path): file_record(path) for path in admission_paths}
    concurrent = {name: {'before': before['files'].get(name), 'after': after['files'].get(name)}
                  for name in before['files'].keys() | after['files'].keys() if before['files'].get(name) != after['files'].get(name)}
    after_overlap = [{'incoming': name, 'dirty': dirty} for name in changed for dirty in after['files']
                     if name == dirty or name.startswith(dirty + '/') or dirty.startswith(name + '/')]
    packet_after = {name: file_record(Path(name)) for name in original_packet}
    failed_after = {str(path): file_record(path) for path in failed_paths}
    assert failed_paths == sorted(path for path in unit.rglob('*') if path.is_file())
    assert packet_before == packet_after and failed_before == failed_after
    write('actual_failed_unit_after.json', failed_after)
    write('original55_after.json', packet_after)
    eligible = (not staged and not overlap and not collisions and not symlinks and not dirty_overlap and not after_overlap
                and dry_return == merge_return == 0 and not unmerged and result_tree is not None
                and index_before == index_after and original_index_before == original_index_after
                and original_head_before == (ORIGINAL / '.git/HEAD').read_bytes() and head == after_head
                and original_head_commit == original_head_after and original_staged == original_staged_after
                and original_before['files'] == original_after['files'] and original_pdfs_before == original_pdfs_after
                and submodules_before.keys() == submodules_after.keys()
                and all(submodules_before[name]['head'] == submodules_after[name]['head'] and submodules_before[name]['dirty']['files'] == submodules_after[name]['dirty']['files'] for name in submodules_before)
                and admission_before == admission_after)
    record = {'schema_version': 1, 'kind': 'independent_original_qualification_pytest_root_incoming_commit_tree_review',
        'at_utc': datetime.now(timezone.utc).isoformat(), 'approved_for_exact_integration_merge': eligible, 'actual_review_job_id': os.environ['SLURM_JOB_ID'], 'actual_review_step_id': os.environ['SLURM_STEP_ID'], 'original55_artifacts_unchanged': packet_before == packet_after, 'actual_failed_unit_files_unchanged': failed_before == failed_after, 'actual_failed_unit_file_count': len(failed_before), 'prior_actual_failed_qualification_audit_sha256': hashlib.sha256(prior_audit.read_bytes()).hexdigest(),
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
        'protected_D28_numerical_and_paired_files': protected_files, 'source_admission_changed': False,
        'actual_merge_or_ref_update_performed': False, 'original_or_integration_worktree_writes': False,
        'numerical_library_device_cluster_ssh_or_job_actions': False, 'github_remote_ref_read_performed': True,
        'original_dirty_files_before': original_before['files'], 'original_dirty_files_after': original_after['files'],
        'original_head_commit_before': original_head_commit, 'original_head_commit_after': original_head_after,
        'original_staged_diff_unchanged': original_staged == original_staged_after,
        'original_tracked_current_dirty_and_known_PDFs_before': original_pdfs_before, 'original_tracked_current_dirty_and_known_PDFs_after': original_pdfs_after,
        'original_submodules_before': submodules_before, 'original_submodules_after': submodules_after,
        'source_admission_before': admission_before, 'source_admission_after': admission_after,
        'git_status_index_refresh': 'GIT_OPTIONAL_LOCKS=0 and --no-optional-locks; actual original and integration index bytes/stat checked unchanged', 'auxiliary_Slurm_steps': 1,
        'proposed_parent_commands': [['git', '-C', str(INTEGRATION), 'merge', '--no-ff', '--no-edit', INCOMING],
            ['git', '-C', str(INTEGRATION), 'rev-parse', 'HEAD^{tree}']],
        'expected_result_tree_after_parent_merge': result_tree}
    write('commands.json', COMMANDS)
    write('review.json', record)
    destination = RUN / 'remote_setup' / (datetime.now(ZoneInfo('America/Los_Angeles')).strftime('%Y-%m-%d-%H-%M')
        + '_original-qualification-pytest-root-incoming-commit-' + INCOMING[:8] + '_base-c851fa90_paths-13_three-way-tree-review')
    destination.mkdir()
    for path in sorted(TEMPORARY.iterdir()):
        if path.is_file():
            shutil.copyfile(path, destination / path.name)
            with (destination / path.name).open('rb') as copied:
                os.fsync(copied.fileno())
    directory_descriptor = os.open(destination, os.O_RDONLY | os.O_DIRECTORY)
    try: os.fsync(directory_descriptor)
    finally: os.close(directory_descriptor)
    receipt = destination / 'review.json'
    print(json.dumps({'approved': eligible, 'receipt': str(receipt), 'sha256': hashlib.sha256(receipt.read_bytes()).hexdigest(),
        'integration_head': head, 'result_tree': result_tree, 'incoming_paths': len(changed),
        'collisions': len(collisions), 'dirty_overlaps': len(dirty_overlap), 'dry_run': dry_return, 'read_tree': merge_return}))


if __name__ == '__main__':
    main()
