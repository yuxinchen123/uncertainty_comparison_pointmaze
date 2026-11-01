"""Execute the reviewed exact original-branch fast-forward once after fresh preservation checks."""

from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import time


BASE = Path(__file__).resolve().parent
OLD = '0f694d32e6c904e57cf601c27f9530cba991697d'
TARGET = 'cef3073939e09ab4b2d3864a0bce8c1bb6cd7020'
BRANCH = 'Use-RLexplore-RND'
EXECUTION = BASE / 'actual_original_fast_forward_to_cef3073'
SPECIFICATION = importlib.util.spec_from_file_location('saved_read_only_audit', BASE / 'resume_saved_audit.py')
REVIEW = importlib.util.module_from_spec(SPECIFICATION)
SPECIFICATION.loader.exec_module(REVIEW)
AUDIT = REVIEW.AUDIT
COMMANDS = AUDIT.COMMANDS


def require(condition, message):
    """Stop without retry or restoration when an exact approved precondition does not hold."""
    if not condition:
        raise RuntimeError(message)


def recorded_run(argv, name):
    """Fsync the exact one-shot command intent before its process and raw returns before interpretation."""
    AUDIT.OUTPUT = EXECUTION
    intent = {'argv': argv, 'cwd': str(AUDIT.ORIGINAL), 'at_utc': datetime.now(timezone.utc).isoformat(),
              'attempt': 1, 'GIT_OPTIONAL_LOCKS': '0', 'retry_permitted': False}
    AUDIT.save(name + '_intent.json', intent)
    result = subprocess.run(argv, cwd=AUDIT.ORIGINAL, env=AUDIT.ENVIRONMENT,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=120)
    row = dict(intent, returncode=result.returncode,
               stdout=AUDIT.save(name + '_stdout.raw', result.stdout),
               stderr=AUDIT.save(name + '_stderr.raw', result.stderr),
               completed_at_utc=datetime.now(timezone.utc).isoformat())
    AUDIT.save(name + '_return.json', row)
    COMMANDS.append(row)
    require(result.returncode == 0, f'{name} refused or failed; exact raw returns are preserved and no retry will run.')
    return row


def original_head_and_remote(expected):
    """Require this exact local branch and exact remote branch head without fetching or changing any ref."""
    require(AUDIT.command(AUDIT.ORIGINAL, ['rev-parse', 'HEAD'])[0] == (expected + '\n').encode(), 'Original HEAD changed.')
    require(AUDIT.command(AUDIT.ORIGINAL, ['symbolic-ref', '--short', 'HEAD'])[0] == (BRANCH + '\n').encode(), 'Original branch changed.')
    raw = AUDIT.command(AUDIT.ORIGINAL, ['ls-remote', '--heads', 'origin', 'refs/heads/' + BRANCH])[0]
    require(raw == (expected + '\trefs/heads/' + BRANCH + '\n').encode(), 'Actual original remote head changed.')
    return raw.decode()


def same_preserved_files(before, after):
    """Compare every recorded working byte, file type/mode, link target and current submodule record."""
    # Before and after contain the same original user files; incoming committed files never enter this set.
    return {'files_identical': before['files'] == after['files'],
            'status_identical': before['status'] == after['status'],
            'submodule_identical': before['submodule'] == after['submodule'],
            'user_staged_diff_identical': before['git']['staged_diff_sha256'] == after['git']['staged_diff_sha256'],
            'all_reads_stable': before['all_reads_stable'] and after['all_reads_stable'],
            'changed_files': {name: {'before': before['files'].get(name), 'after': after['files'].get(name)}
                              for name in sorted(before['files'].keys() | after['files'].keys())
                              if before['files'].get(name) != after['files'].get(name)}}


def main():
    """Recheck, update once, prove preservation, then push only the exact original branch without force."""
    require(os.environ.get('SLURM_JOB_ID') == '6781087', 'Use only the current owned CPU allocation.')
    started = time.monotonic()
    EXECUTION.mkdir()
    AUDIT.OUTPUT = EXECUTION

    # Bind to the exact root-reviewed proof and source bytes before recording the authorized operation.
    reviewed_raw = (BASE / 'continuation_after_saved_first_scan/audit.json').read_bytes()
    require(hashlib.sha256(reviewed_raw).hexdigest() == '74f195148288ff834d1c4f2d645a76f792541a62b98cf089288fd826d3d3055a', 'Reviewed audit bytes changed.')
    reviewed = json.loads(reviewed_raw)
    require(reviewed['proposed_fast_forward_eligible_at_observed_snapshot'] is True and reviewed['target'] == TARGET, 'Reviewed update is not eligible.')
    manifest = json.loads((BASE / 'audit_artifact_manifest.json').read_bytes())
    for name in ('audit_original_checkout.py', 'resume_saved_audit.py'):
        row = next(row for row in manifest['artifact_files'] if row['path'] == name)
        require(hashlib.sha256((BASE / name).read_bytes()).hexdigest() == row['sha256'], 'Reviewed audit helper bytes changed: ' + name)
    intent = {'schema_version': 1, 'kind': 'authorized_exact_original_branch_fast_forward_intent',
              'at_utc': datetime.now(timezone.utc).isoformat(), 'expected_old_head': OLD, 'target': TARGET,
              'branch': BRANCH, 'original': str(AUDIT.ORIGINAL), 'reviewed_audit_sha256': hashlib.sha256(reviewed_raw).hexdigest(),
              'driver_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'allowed_mutation_argv': [['git', '-c', 'submodule.recurse=false', '-c', 'core.hooksPath=/dev/null',
                                         'merge', '--ff-only', '--no-edit', TARGET],
                                        ['git', 'push', 'origin', TARGET + ':refs/heads/' + BRANCH]],
              'retries_permitted': False, 'force_permitted': False, 'stash_or_reset_permitted': False,
              'submodule_update_permitted': False, 'other_branch_updates_permitted': False,
              'job_id': os.environ['SLURM_JOB_ID'], 'step_id': os.environ.get('SLURM_STEP_ID'),
              'hostname': socket.gethostname(), 'pid': os.getpid(),
              'birth': Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()[19],
              'cgroup': Path('/proc/self/cgroup').read_text()}
    AUDIT.save('execution_intent.json', intent)
    historical = json.loads(AUDIT.HISTORICAL.read_bytes())
    approved = json.loads((BASE / 'continuation_after_saved_first_scan/scan_after.json').read_bytes())
    original_head_and_remote(OLD)
    before_refs = AUDIT.command(AUDIT.ORIGINAL, ['for-each-ref', '--format=%(refname)%09%(objectname)', 'refs/heads'])[0]

    # Reuse the exact recorded PDF roster, fresh dirty/untracked status and current RLeXplore files only.
    before_dir = EXECUTION / 'before'
    before_dir.mkdir()
    AUDIT.OUTPUT = before_dir
    before = REVIEW.second_snapshot(approved, historical)
    pre_comparison = same_preserved_files(approved, before)
    require(all(pre_comparison[key] for key in ('files_identical', 'status_identical', 'submodule_identical', 'user_staged_diff_identical', 'all_reads_stable')), 'A preservation check changed before mutation.')
    require(before['git'] == approved['git'] and before['git']['head'] == OLD and before['git']['branch'] == BRANCH, 'Original head/index/staged state changed before mutation.')
    incoming = AUDIT.collisions(before)
    require(not incoming['untracked_or_ignored_collisions'] and not incoming['dirty_incoming_overlaps'], 'An incoming collision or dirty overlap appeared.')
    require(len(incoming['incoming']) == 679, 'The exact incoming path count changed.')
    require(AUDIT.command(AUDIT.ORIGINAL, ['merge-base', '--is-ancestor', OLD, TARGET], allowed=(0, 1))[1] == 0, 'Exact target no longer fast-forwards the original.')
    alternate = before_dir / 'alternate_index_for_read_tree'
    shutil.copyfile(before['git']['index_path'], alternate)
    environment = dict(AUDIT.ENVIRONMENT, GIT_INDEX_FILE=str(alternate))
    require(AUDIT.command(AUDIT.ORIGINAL, ['read-tree', '--dry-run', '-m', '-u', OLD, TARGET], environment=environment, allowed=(0, 1, 128))[1] == 0, 'Fresh alternate-index read-tree refused.')
    require(AUDIT.path_record(Path(before['git']['index_path'])) == before['git']['index'], 'Original index changed during preflight.')
    original_head_and_remote(OLD)
    AUDIT.OUTPUT = EXECUTION
    AUDIT.save('pre_mutation_commands.json', COMMANDS)
    AUDIT.save('pre_mutation_verified.json', {'passed': True, 'at_utc': datetime.now(timezone.utc).isoformat(),
                                           'preservation': pre_comparison, 'incoming_count': 679,
                                           'collisions': 0, 'dirty_overlaps': 0, 'alternate_index_dry_run_passed': True})

    # This is the single authorized local mutation; failure records remain in place and prohibit retry.
    merge = recorded_run(intent['allowed_mutation_argv'][0], 'single_fast_forward')
    require(AUDIT.command(AUDIT.ORIGINAL, ['rev-parse', 'HEAD'])[0] == (TARGET + '\n').encode(), 'Fast-forward returned without the exact target HEAD.')
    require(AUDIT.command(AUDIT.ORIGINAL, ['symbolic-ref', '--short', 'HEAD'])[0] == (BRANCH + '\n').encode(), 'Original branch changed after fast-forward.')
    after_dir = EXECUTION / 'after'
    after_dir.mkdir()
    AUDIT.OUTPUT = after_dir
    after = REVIEW.second_snapshot(before, historical)
    comparison = same_preserved_files(before, after)
    AUDIT.save('preservation_comparison.json', comparison)
    require(all(comparison[key] for key in ('files_identical', 'status_identical', 'submodule_identical', 'user_staged_diff_identical', 'all_reads_stable')), 'A post-update working-file change was observed; retain it and stop before push.')
    require(after['git']['head'] == TARGET and after['git']['branch'] == BRANCH, 'Post-scan original identity changed.')
    after_refs = AUDIT.command(AUDIT.ORIGINAL, ['for-each-ref', '--format=%(refname)%09%(objectname)', 'refs/heads'])[0]
    before_map = dict(line.split('\t') for line in before_refs.decode().splitlines())
    after_map = dict(line.split('\t') for line in after_refs.decode().splitlines())
    other_ref_changes = {name: {'before': before_map.get(name), 'after': after_map.get(name)}
                         for name in before_map.keys() | after_map.keys()
                         if name != 'refs/heads/' + BRANCH and before_map.get(name) != after_map.get(name)}
    require(not other_ref_changes, 'Another local branch changed during the operation; retain its state and stop before push.')

    # A fresh remote read must still equal the old head before the one ordinary non-forced push.
    AUDIT.OUTPUT = EXECUTION
    require(AUDIT.command(AUDIT.ORIGINAL, ['ls-remote', '--heads', 'origin', 'refs/heads/' + BRANCH])[0] == (OLD + '\trefs/heads/' + BRANCH + '\n').encode(), 'Remote advanced before push; no push will run.')
    AUDIT.save('post_preservation_verified_before_push.json', {'passed': True, 'at_utc': datetime.now(timezone.utc).isoformat(),
                                                            'preservation': comparison, 'other_local_ref_changes': other_ref_changes})
    push = recorded_run(intent['allowed_mutation_argv'][1], 'single_nonforced_push')
    remote = original_head_and_remote(TARGET)
    result = {'schema_version': 1, 'kind': 'exact_original_branch_fast_forward_completion', 'passed': True,
              'completed_at_utc': datetime.now(timezone.utc).isoformat(), 'old_head': OLD, 'target': TARGET,
              'branch': BRANCH, 'actual_remote_row': remote, 'dirty_untracked_entries': len(after['status']),
              'preserved_paths': len(after['files']), 'preserved_pdfs': len(after['pdf_paths']),
              'historical_five_match': all(after['files'][name].get('sha256') == value for name, value in historical.items()),
              'preservation': comparison, 'original_index_before_sha256': before['git']['index']['sha256'],
              'original_index_after_sha256': after['git']['index']['sha256'],
              'index_change_expected_for_incoming_committed_files': True, 'incoming_paths': 679,
              'other_local_branch_updates': other_ref_changes, 'local_fast_forward_attempts': 1, 'ordinary_push_attempts': 1,
              'force_or_reset_or_stash_or_submodule_update': False, 'elapsed_seconds': time.monotonic() - started,
              'merge_return': merge, 'push_return': push, 'process': intent}
    AUDIT.save('commands.json', COMMANDS)
    receipt = AUDIT.save('completion.json', result)
    print(json.dumps({'passed': True, 'receipt': receipt, 'target': TARGET, 'actual_step': intent['step_id'],
                      'preserved_paths': len(after['files']), 'preserved_pdfs': len(after['pdf_paths'])}))


if __name__ == '__main__':
    try:
        main()
    except BaseException as error:
        # Preserve the failure without changing or restoring original bytes, then re-raise it.
        if EXECUTION.exists() and not (EXECUTION / 'failure.json').exists():
            AUDIT.OUTPUT = EXECUTION
            AUDIT.save('failure.json', {'kind': 'exact_original_update_stopped', 'at_utc': datetime.now(timezone.utc).isoformat(),
                                        'error_type': type(error).__name__, 'message': str(error),
                                        'retry_performed': False, 'restoration_performed': False})
            AUDIT.save('commands_at_failure.json', COMMANDS)
        raise
