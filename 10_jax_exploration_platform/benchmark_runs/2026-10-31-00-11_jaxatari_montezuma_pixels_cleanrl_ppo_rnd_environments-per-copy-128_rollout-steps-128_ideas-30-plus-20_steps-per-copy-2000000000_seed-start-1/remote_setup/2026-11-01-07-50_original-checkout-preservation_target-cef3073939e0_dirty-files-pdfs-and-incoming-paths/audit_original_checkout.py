"""Read original checkout bytes and test the exact proposed update through a separate index."""

import base64
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import stat
import subprocess


ORIGINAL = Path('/p/rlprojects/RND')
INTEGRATION = Path('/p/rlprojects/RND_worktrees/merge-atari-cleanrl')
TARGET = 'cef3073939e09ab4b2d3864a0bce8c1bb6cd7020'
OUTPUT = Path(__file__).resolve().parent
RUN = OUTPUT.parent.parent
HISTORICAL = RUN / 'data/original_checkout_preservation/before_merge.json'
COMMANDS = []
ENVIRONMENT = dict(os.environ, GIT_OPTIONAL_LOCKS='0', GIT_TERMINAL_PROMPT='0', LC_ALL='C')


def save(name, value):
    """Publish only new audit files and fsync their bytes and directory."""
    raw = value if isinstance(value, bytes) else (json.dumps(value, indent=2, ensure_ascii=True) + '\n').encode()
    with (OUTPUT / name).open('xb') as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    descriptor = os.open(OUTPUT, os.O_RDONLY | os.O_DIRECTORY)
    os.fsync(descriptor)
    os.close(descriptor)
    return {'path': str(OUTPUT / name), 'sha256': hashlib.sha256(raw).hexdigest(), 'bytes': len(raw)}


def command(root, arguments, *, input_bytes=None, environment=None, allowed=(0,)):
    """Capture exact read-only command bytes, with optional locks disabled for every Git read."""
    argv = ['git', '--no-optional-locks', *arguments]
    number = len(COMMANDS)
    result = subprocess.run(argv, cwd=root, env=environment or ENVIRONMENT, input=input_bytes,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=120)
    record = {'argv': argv, 'cwd': str(root), 'returncode': result.returncode,
              'observed_at_utc': datetime.now(timezone.utc).isoformat(),
              'stdout': save(f'command_{number:03d}_stdout.raw', result.stdout),
              'stderr': save(f'command_{number:03d}_stderr.raw', result.stderr),
              'GIT_OPTIONAL_LOCKS': '0'}
    if input_bytes is not None:
        record['stdin'] = save(f'command_{number:03d}_stdin.raw', input_bytes)
    COMMANDS.append(record)
    if result.returncode not in allowed:
        save('failed_commands.json', COMMANDS)
        raise RuntimeError(f'Read-only audit command {number} returned {result.returncode}: {argv}')
    return result.stdout, result.returncode


def names(raw):
    """Decode NUL-separated Git names without shell parsing or pathname normalization."""
    return [os.fsdecode(value) for value in raw.split(b'\0') if value]


def metadata(value):
    """Retain file type, access mode, physical identity and change times without atime."""
    return {'mode': stat.S_IMODE(value.st_mode), 'mode_octal': oct(stat.S_IMODE(value.st_mode)),
            'type_mode': value.st_mode, 'uid': value.st_uid, 'gid': value.st_gid,
            'device': value.st_dev, 'inode': value.st_ino, 'size': value.st_size,
            'mtime_ns': value.st_mtime_ns, 'ctime_ns': value.st_ctime_ns}


def path_record(path):
    """Hash regular file contents or literal symlink bytes without following a final symlink."""
    # Example: a symlink to ../file remains its own target bytes; it is never hashed as ../file.
    try:
        before = path.lstat()
    except FileNotFoundError:
        return {'kind': 'absent'}
    result = metadata(before)
    if stat.S_ISLNK(before.st_mode):
        target = os.readlink(os.fsencode(path))
        result.update(kind='symlink', target_bytes_base64=base64.b64encode(target).decode(),
                      target_display=os.fsdecode(target), sha256=hashlib.sha256(target).hexdigest(),
                      stable_during_read=metadata(path.lstat()) == metadata(before))
        return result
    if stat.S_ISDIR(before.st_mode):
        return dict(result, kind='directory')
    if not stat.S_ISREG(before.st_mode):
        return dict(result, kind='special_file_not_opened')

    # The descriptor identifies the actual file read; an atomic replacement is reported separately.
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    digest = hashlib.sha256()
    with os.fdopen(descriptor, 'rb') as stream:
        opened = os.fstat(stream.fileno())
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
        finished = os.fstat(stream.fileno())
    try:
        after = path.lstat()
    except FileNotFoundError:
        after = None
    result.update(kind='file', sha256=digest.hexdigest(), opened=metadata(opened), finished=metadata(finished),
                  path_after=None if after is None else metadata(after),
                  stable_during_read=metadata(before) == metadata(opened) == metadata(finished)
                  and after is not None and metadata(after) == metadata(finished))
    return result


def status(root):
    """Preserve exact porcelain records and both paths of each rename or copy."""
    raw, _ = command(root, ['status', '--porcelain=v1', '-z', '--untracked-files=all', '--ignore-submodules=none'])
    entries, index, result = raw.split(b'\0'), 0, []
    while index < len(entries):
        entry = entries[index]
        index += 1
        if not entry:
            continue
        record = {'status': entry[:2].decode('ascii'), 'path': os.fsdecode(entry[3:])}
        if b'R' in entry[:2] or b'C' in entry[:2]:
            record['original_path'] = os.fsdecode(entries[index])
            index += 1
        result.append(record)
    return result


def git_state(root):
    """Observe local heads, staged bytes and index bytes without refreshing or writing them."""
    head = command(root, ['rev-parse', 'HEAD'])[0].decode().strip()
    branch = command(root, ['symbolic-ref', '--short', 'HEAD'])[0].decode().strip()
    index_name = Path(command(root, ['rev-parse', '--git-path', 'index'])[0].decode().strip())
    index_path = index_name if index_name.is_absolute() else root / index_name
    return {'head': head, 'branch': branch, 'index_path': str(index_path), 'index': path_record(index_path),
            'index_entries_sha256': hashlib.sha256(command(root, ['ls-files', '--stage', '-z'])[0]).hexdigest(),
            'staged_diff_sha256': hashlib.sha256(command(root, ['diff', '--cached', '--binary', '--no-ext-diff'])[0]).hexdigest()}


def snapshot(label, historical):
    """Hash all dirty and untracked paths, submodule current files and all checkout PDFs twice."""
    # The dirty submodule is preserved through ordinary read-only metadata; no update or recursive Git is run.
    state = git_state(ORIGINAL)
    rows = status(ORIGINAL)
    paths = set(historical)
    paths.update(row['path'] for row in rows)
    paths.update(row['original_path'] for row in rows if 'original_path' in row)
    submodule = ORIGINAL / 'RLeXplore'
    nested_status = status(submodule)
    nested_names = names(command(submodule, ['ls-files', '--cached', '--others', '--exclude-standard', '-z'])[0])
    paths.update('RLeXplore/' + name for name in nested_names)
    nested_state = git_state(submodule)

    # PDF discovery includes ignored PDFs, but does not follow directory symlinks or enter Git metadata.
    pdfs = []
    for directory, children, files in os.walk(ORIGINAL, followlinks=False):
        children[:] = sorted(name for name in children if name != '.git')
        pdfs.extend(str((Path(directory) / name).relative_to(ORIGINAL))
                    for name in files if name.lower().endswith('.pdf'))
    paths.update(pdfs)
    files = {name: path_record(ORIGINAL / name) for name in sorted(paths)}
    value = {'label': label, 'observed_at_utc': datetime.now(timezone.utc).isoformat(), 'git': state,
             'status': rows, 'dirty_and_untracked_count': len(rows), 'files': files,
             'pdf_paths': sorted(pdfs), 'historical_paths': sorted(historical),
             'submodule': {'git': nested_state, 'status': nested_status, 'current_files': nested_names},
             'all_reads_stable': all(row.get('stable_during_read', True) for row in files.values())}
    save(label + '.json', value)
    return value


def collisions(before):
    """Check only exact incoming paths and parents for ignored, untracked and file-directory collisions."""
    old = before['git']['head']
    raw, _ = command(ORIGINAL, ['diff', '--name-status', '--no-renames', '-z', old, TARGET])
    pieces = raw.split(b'\0')
    incoming = [{'change': pieces[index].decode(), 'path': os.fsdecode(pieces[index + 1])}
                for index in range(0, len(pieces) - 1, 2)]
    tracked = set(names(command(ORIGINAL, ['ls-files', '--cached', '-z'])[0]))
    candidates = set()
    for row in incoming:
        path = Path(row['path'])
        candidates.add(str(path))
        candidates.update(str(parent) for parent in path.parents if str(parent) != '.')
    ignore_raw, _ = command(ORIGINAL, ['check-ignore', '--no-index', '--stdin', '-z'],
                            input_bytes=b'\0'.join(os.fsencode(name) for name in sorted(candidates)) + b'\0', allowed=(0, 1))
    ignored = set(names(ignore_raw))
    observed, found, overlaps = {}, [], []
    dirty = {row['path'] for row in before['status']}
    for row in incoming:
        name = row['path']
        actual = path_record(ORIGINAL / name)
        observed[name] = {'incoming_change': row['change'], 'tracked_in_original_index': name in tracked,
                          'ignored_by_current_rules': name in ignored, 'working_path': actual}
        if actual['kind'] != 'absent' and name not in tracked:
            found.append({'path': name, 'reason': 'incoming path exists outside the original index',
                          'ignored': name in ignored, 'working_path': actual})
        for parent in Path(name).parents:
            if str(parent) == '.':
                continue
            parent_record = path_record(ORIGINAL / parent)
            if parent_record['kind'] not in ('directory', 'absent'):
                found.append({'path': str(parent), 'incoming_path': name, 'reason': 'incoming parent is not an ordinary directory',
                              'ignored': str(parent) in ignored, 'working_path': parent_record})
        for path in dirty:
            if path == name or path.startswith(name + '/') or name.startswith(path + '/'):
                overlaps.append({'dirty_path': path, 'incoming_path': name})
    value = {'old': old, 'target': TARGET, 'incoming': incoming, 'path_observations': observed,
             'ignored_checks_exact_paths_and_parents': sorted(candidates), 'ignored_matching_paths': sorted(ignored),
             'untracked_or_ignored_collisions': found, 'dirty_incoming_overlaps': overlaps,
             'whole_ignored_environment_or_dataset_hashing_performed': False}
    save('incoming_paths_and_collisions.json', value)
    return value


def main():
    """Save two preservation scans, remote heads and the exact alternate-index dry run; never update the checkout."""
    assert os.environ.get('SLURM_JOB_ID') == '6781087'
    historical = json.loads(HISTORICAL.read_bytes())
    first = snapshot('scan_before', historical)
    local_heads_before = command(ORIGINAL, ['for-each-ref', '--format=%(refname)%09%(objectname)%09%(upstream)', 'refs/heads', 'refs/remotes/origin'])[0]
    remote_before = command(ORIGINAL, ['ls-remote', '--heads', 'origin'])[0]
    integration_before = git_state(INTEGRATION)
    ancestry = command(ORIGINAL, ['merge-base', '--is-ancestor', first['git']['head'], TARGET], allowed=(0, 1))[1]
    diverged_original = command(ORIGINAL, ['log', '--format=%H%x09%s', TARGET + '..' + first['git']['head']])[0]
    incoming = collisions(first)

    # The real worktree is examined with --dry-run; only a separate byte-for-byte index copy is supplied.
    source_index = Path(first['git']['index_path'])
    temporary_index = OUTPUT / 'alternate_index_for_read_tree'
    shutil.copyfile(source_index, temporary_index)
    alternate_before = path_record(temporary_index)
    dry_environment = dict(ENVIRONMENT, GIT_INDEX_FILE=str(temporary_index))
    _, dry_return = command(ORIGINAL, ['read-tree', '--dry-run', '-m', '-u', first['git']['head'], TARGET],
                            environment=dry_environment, allowed=(0, 1, 128))
    alternate_after = path_record(temporary_index)
    second = snapshot('scan_after', historical)
    local_heads_after = command(ORIGINAL, ['for-each-ref', '--format=%(refname)%09%(objectname)%09%(upstream)', 'refs/heads', 'refs/remotes/origin'])[0]
    remote_after = command(ORIGINAL, ['ls-remote', '--heads', 'origin'])[0]
    integration_after = git_state(INTEGRATION)

    # Compare content, mode and physical identity; real concurrent updates are retained rather than restored.
    changed = {name: {'before': first['files'].get(name), 'after': second['files'].get(name)}
               for name in sorted(first['files'].keys() | second['files'].keys())
               if first['files'].get(name) != second['files'].get(name)}
    historical_result = {name: {'current_sha256': second['files'][name].get('sha256'),
                                'historical_sha256': historical[name],
                                'matches': second['files'][name].get('sha256') == historical[name]}
                         for name in historical}
    eligible = (ancestry == 0 and not diverged_original and dry_return == 0 and not changed
                and first['git'] == second['git'] and first['status'] == second['status']
                and first['submodule'] == second['submodule'] and local_heads_before == local_heads_after
                and remote_before == remote_after and not incoming['untracked_or_ignored_collisions']
                and not incoming['dirty_incoming_overlaps'] and first['all_reads_stable'] and second['all_reads_stable'])
    value = {'schema_version': 1, 'kind': 'read_only_original_checkout_update_audit',
             'at_utc': datetime.now(timezone.utc).isoformat(), 'original': str(ORIGINAL), 'target': TARGET,
             'proposed_fast_forward_eligible_at_observed_snapshot': eligible,
             'original_head': second['git']['head'], 'original_branch': second['git']['branch'],
             'original_is_ancestor_of_target': ancestry == 0, 'original_commits_not_in_target': diverged_original.decode(),
             'local_head_rows_before': local_heads_before.decode(), 'local_head_rows_after': local_heads_after.decode(),
             'actual_remote_head_rows_before': remote_before.decode(), 'actual_remote_head_rows_after': remote_after.decode(),
             'integration_before': integration_before, 'integration_after': integration_after,
             'dirty_and_untracked_paths': second['dirty_and_untracked_count'], 'hashed_paths': len(second['files']),
             'pdf_count': len(second['pdf_paths']), 'changed_between_scans': changed,
             'historical_five': historical_result, 'all_file_reads_stable': first['all_reads_stable'] and second['all_reads_stable'],
             'original_status_identical': first['status'] == second['status'],
             'original_index_bytes_identical': first['git']['index'] == second['git']['index'],
             'original_staged_entries_identical': first['git']['index_entries_sha256'] == second['git']['index_entries_sha256'],
             'submodule_metadata_and_current_files_identical': first['submodule'] == second['submodule'],
             'incoming_paths': len(incoming['incoming']), 'incoming_collisions': len(incoming['untracked_or_ignored_collisions']),
             'incoming_dirty_overlaps': len(incoming['dirty_incoming_overlaps']),
             'read_tree': {'returncode': dry_return, 'alternate_index': str(temporary_index),
                           'before': alternate_before, 'after': alternate_after, 'dry_run': True,
                           'original_index_used_for_writes': False},
             'initial_status_stat_cache_note': 'One earlier git status omitted GIT_OPTIONAL_LOCKS=0 and might have refreshed index stat-cache metadata. No staged entry, branch, ref or worktree update was requested. This audit records exact current index bytes before and after, and every audit Git command disables optional locks.',
             'mutating_git_commands_executed': [], 'original_checkout_writes_requested': False,
             'ignored_collision_scope': 'Only exact incoming paths and their parents; ignored environments and datasets were not hashed, except explicitly requested PDF files.',
             'process': {'job_id': os.environ['SLURM_JOB_ID'], 'step_id': os.environ.get('SLURM_STEP_ID'),
                         'hostname': socket.gethostname(), 'pid': os.getpid(),
                         'birth': Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()[19],
                         'cgroup': Path('/proc/self/cgroup').read_text()}}
    save('commands.json', COMMANDS)
    receipt = save('audit.json', value)
    print(json.dumps({'receipt': receipt, 'eligible': eligible, 'dirty_paths': value['dirty_and_untracked_paths'],
                      'hashed_paths': value['hashed_paths'], 'pdfs': value['pdf_count'], 'incoming': value['incoming_paths'],
                      'collisions': value['incoming_collisions'], 'changed_between_scans': len(changed), 'read_tree_returncode': dry_return}))


if __name__ == '__main__':
    main()
