"""Finish the read-only audit from its first scan without a second ignored-tree PDF discovery."""

from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import socket
import time


BASE = Path(__file__).resolve().parent
SPECIFICATION = importlib.util.spec_from_file_location('read_only_original_audit', BASE / 'audit_original_checkout.py')
AUDIT = importlib.util.module_from_spec(SPECIFICATION)
SPECIFICATION.loader.exec_module(AUDIT)
AUDIT.OUTPUT = BASE / 'continuation_after_saved_first_scan'


def second_snapshot(first, historical):
    """Rehash the recorded first files plus fresh current paths and known development-document PDFs."""
    # Before: the first file roster; after: that same roster plus any fresh dirty/untracked file.
    state = AUDIT.git_state(AUDIT.ORIGINAL)
    rows = AUDIT.status(AUDIT.ORIGINAL)
    paths = set(first['files']) | set(historical)
    paths.update(row['path'] for row in rows)
    paths.update(row['original_path'] for row in rows if 'original_path' in row)
    submodule = AUDIT.ORIGINAL / 'RLeXplore'
    nested_status = AUDIT.status(submodule)
    nested_names = AUDIT.names(AUDIT.command(submodule, ['ls-files', '--cached', '--others', '--exclude-standard', '-z'])[0])
    paths.update('RLeXplore/' + name for name in nested_names)
    nested_state = AUDIT.git_state(submodule)

    # Git supplies the tracked and current non-ignored PDF roster without crawling ignored packages or datasets.
    visible = AUDIT.names(AUDIT.command(AUDIT.ORIGINAL, ['ls-files', '--cached', '--others', '--exclude-standard', '-z'])[0])
    paths.update(name for name in visible if name.lower().endswith('.pdf'))
    known = {
        '07_reconstruction/development_document/RND_development_document.pdf',
        '10_jax_exploration_platform/development_document/RND_JAX_exploration_development_document.pdf',
    }
    paths.update(known)
    files = {name: AUDIT.path_record(AUDIT.ORIGINAL / name) for name in sorted(paths)}
    value = {'label': 'scan_after', 'observed_at_utc': datetime.now(timezone.utc).isoformat(), 'git': state,
             'status': rows, 'dirty_and_untracked_count': len(rows), 'files': files,
             'pdf_paths': sorted(name for name in paths if name.lower().endswith('.pdf')),
             'historical_paths': sorted(historical),
             'submodule': {'git': nested_state, 'status': nested_status, 'current_files': nested_names},
             'all_reads_stable': all(row.get('stable_during_read', True) for row in files.values()),
             'discovery_scope': 'First recorded file set plus fresh full dirty/untracked status, current RLeXplore files, tracked/non-ignored PDFs and two named development-document PDFs. No second ignored-tree crawl.'}
    AUDIT.save('scan_after.json', value)
    return value


def main():
    """Preserve original bytes and exact target identities while recording a separate-index dry run."""
    assert os.environ.get('SLURM_JOB_ID') == '6781087'
    start = time.monotonic()
    AUDIT.OUTPUT.mkdir()
    raw_before = (BASE / 'scan_before.json').read_bytes()
    first = json.loads(raw_before)
    stop = json.loads((BASE / 'own_scan_stop_after_first_record.json').read_bytes())
    assert stop['first_scan_sha256'] == hashlib.sha256(raw_before).hexdigest()
    AUDIT.save('scan_before.json', raw_before)
    historical = json.loads(AUDIT.HISTORICAL.read_bytes())
    local_before = AUDIT.command(AUDIT.ORIGINAL, ['for-each-ref', '--format=%(refname)%09%(objectname)%09%(upstream)', 'refs/heads', 'refs/remotes/origin'])[0]
    remote_before = AUDIT.command(AUDIT.ORIGINAL, ['ls-remote', '--heads', 'origin'])[0]
    integration_before = AUDIT.git_state(AUDIT.INTEGRATION)
    ancestry = AUDIT.command(AUDIT.ORIGINAL, ['merge-base', '--is-ancestor', first['git']['head'], AUDIT.TARGET], allowed=(0, 1))[1]
    exclusive = AUDIT.command(AUDIT.ORIGINAL, ['log', '--format=%H%x09%s', AUDIT.TARGET + '..' + first['git']['head']])[0]
    incoming = AUDIT.collisions(first)

    # --dry-run prevents checkout writes; GIT_INDEX_FILE points to our new copy, never the original index.
    alternate = AUDIT.OUTPUT / 'alternate_index_for_read_tree'
    shutil.copyfile(first['git']['index_path'], alternate)
    alternate_before = AUDIT.path_record(alternate)
    environment = dict(AUDIT.ENVIRONMENT, GIT_INDEX_FILE=str(alternate))
    _, dry_return = AUDIT.command(AUDIT.ORIGINAL, ['read-tree', '--dry-run', '-m', '-u', first['git']['head'], AUDIT.TARGET],
                                  environment=environment, allowed=(0, 1, 128))
    alternate_after = AUDIT.path_record(alternate)
    second = second_snapshot(first, historical)
    local_after = AUDIT.command(AUDIT.ORIGINAL, ['for-each-ref', '--format=%(refname)%09%(objectname)%09%(upstream)', 'refs/heads', 'refs/remotes/origin'])[0]
    remote_after = AUDIT.command(AUDIT.ORIGINAL, ['ls-remote', '--heads', 'origin'])[0]
    integration_after = AUDIT.git_state(AUDIT.INTEGRATION)

    # A known absent PDF added to the explicit roster is coverage, not a change to original bytes.
    common = first['files'].keys() & second['files'].keys()
    changed = {name: {'before': first['files'][name], 'after': second['files'][name]}
               for name in sorted(common) if first['files'][name] != second['files'][name]}
    additions = {name: second['files'][name] for name in sorted(second['files'].keys() - first['files'].keys())}
    original_ref = 'refs/heads/' + first['git']['branch']
    original_ref_line = lambda value: next(line for line in value.decode().splitlines() if line.startswith(original_ref + '\t'))
    eligible = (ancestry == 0 and not exclusive and dry_return == 0 and not changed
                and all(record['kind'] == 'absent' for record in additions.values())
                and first['git'] == second['git'] and first['status'] == second['status']
                and first['submodule'] == second['submodule']
                and original_ref_line(local_before) == original_ref_line(local_after)
                and remote_before == remote_after and not incoming['untracked_or_ignored_collisions']
                and not incoming['dirty_incoming_overlaps'] and first['all_reads_stable'] and second['all_reads_stable'])
    historical_result = {name: {'current_sha256': second['files'][name].get('sha256'), 'historical_sha256': historical[name],
                                'matches': second['files'][name].get('sha256') == historical[name]}
                         for name in historical}
    index_mtime = datetime.fromtimestamp(second['git']['index']['mtime_ns'] / 1e9, timezone.utc).isoformat()
    value = {'schema_version': 1, 'kind': 'read_only_original_checkout_update_audit', 'at_utc': datetime.now(timezone.utc).isoformat(),
             'original': str(AUDIT.ORIGINAL), 'target': AUDIT.TARGET, 'original_head': second['git']['head'],
             'original_branch': second['git']['branch'], 'proposed_fast_forward_eligible_at_observed_snapshot': eligible,
             'original_is_ancestor_of_target': ancestry == 0, 'original_commits_not_in_target': exclusive.decode(),
             'local_head_rows_before': local_before.decode(), 'local_head_rows_after': local_after.decode(),
             'actual_remote_head_rows_before': remote_before.decode(), 'actual_remote_head_rows_after': remote_after.decode(),
             'integration_before': integration_before, 'integration_after': integration_after,
             'dirty_and_untracked_paths': second['dirty_and_untracked_count'], 'hashed_paths': len(second['files']),
             'pdf_count': len(second['pdf_paths']), 'first_full_discovery_pdf_count': len(first['pdf_paths']),
             'second_scan_scope': second['discovery_scope'], 'changed_between_scans': changed, 'additional_observations': additions,
             'historical_five': historical_result, 'all_file_reads_stable': first['all_reads_stable'] and second['all_reads_stable'],
             'original_status_identical': first['status'] == second['status'],
             'original_index_bytes_identical': first['git']['index'] == second['git']['index'],
             'original_staged_entries_identical': first['git']['index_entries_sha256'] == second['git']['index_entries_sha256'],
             'submodule_metadata_and_current_files_identical': first['submodule'] == second['submodule'],
             'incoming_paths': len(incoming['incoming']), 'incoming_collisions': len(incoming['untracked_or_ignored_collisions']),
             'incoming_dirty_overlaps': len(incoming['dirty_incoming_overlaps']),
             'read_tree': {'returncode': dry_return, 'alternate_index': str(alternate), 'before': alternate_before,
                           'after': alternate_after, 'dry_run': True, 'original_index_used_for_writes': False},
             'initial_status_stat_cache_note': 'One earlier git status omitted GIT_OPTIONAL_LOCKS=0. The observed index mtime and ctime remain 10:23:03 UTC, predating that read; no index refresh was observed. All audit Git commands disabled optional locks.',
             'original_index_mtime_utc': index_mtime, 'mutating_git_commands_executed': [], 'original_checkout_writes_requested': False,
             'ignored_collision_scope': 'Only exact incoming paths and parents. No ignored environment or dataset content hashes, except PDFs from the retained first discovery.',
             'first_scan_stop_record': {'path': str(BASE / 'own_scan_stop_after_first_record.json'), 'sha256': hashlib.sha256((BASE / 'own_scan_stop_after_first_record.json').read_bytes()).hexdigest()},
             'proposed_commands': {'path': str(BASE / 'proposed_commands.json'), 'sha256': hashlib.sha256((BASE / 'proposed_commands.json').read_bytes()).hexdigest()},
             'elapsed_seconds_this_continuation': time.monotonic() - start,
             'process': {'job_id': os.environ['SLURM_JOB_ID'], 'step_id': os.environ.get('SLURM_STEP_ID'),
                         'hostname': socket.gethostname(), 'pid': os.getpid(), 'birth': Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()[19],
                         'cgroup': Path('/proc/self/cgroup').read_text()}}
    AUDIT.save('commands.json', AUDIT.COMMANDS)
    receipt = AUDIT.save('audit.json', value)
    print(json.dumps({'receipt': receipt, 'eligible': eligible, 'dirty_paths': value['dirty_and_untracked_paths'],
                      'hashed_paths': value['hashed_paths'], 'pdfs': value['pdf_count'], 'incoming': value['incoming_paths'],
                      'collisions': value['incoming_collisions'], 'changed_between_scans': len(changed),
                      'read_tree_returncode': dry_return, 'actual_step_id': value['process']['step_id']}))


if __name__ == '__main__':
    main()
