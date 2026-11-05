"""Review saved held-job release inputs without contacting the scheduler."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess

INTEGRATION = Path('/p/rlprojects/RND_worktrees/merge-atari-cleanrl')
RUN = INTEGRATION / '10_jax_exploration_platform/benchmark_runs/2026-10-31-00-11_jaxatari_montezuma_pixels_cleanrl_ppo_rnd_environments-per-copy-128_rollout-steps-128_ideas-30-plus-20_steps-per-copy-2000000000_seed-start-1'
NAME = '2026-11-04-19-37_nsight-process-observation_jaguar03_gpu-count-1_processor-count-1_host-memory-gib-2_time-limit-minutes-5_reservation-sl5nw_168_qos-csresnolim_tool-version-2024.7.1.84_environment-transitions-0'
PACKET = RUN / 'remote_setup' / NAME
UNIT = RUN / 'data' / NAME
PUBLICATION = '885f3736e6f3b2e49068e69875fb10cdc4e7f2b9'
HELPER = '18d727bda2e8c0ab48dc047a4dd89104de2b6cba'
PLAN_SHA = '40e8ee438a13108c51a07de252b9c5389f824a5eb8c0159cb232355ddc3d0f26'
BINDING_SHA = '82de02470dcb43fdead481ec5d0050aedc73d45f799c4998896126293019c3de'


def digest(raw):
    """Identify exact saved bytes."""
    return hashlib.sha256(raw).hexdigest()


def reference(path):
    """Bind an ordinary saved input without changing it."""
    assert path.resolve(strict=True) == path and path.is_file() and not path.is_symlink()
    return {'path': str(path), 'sha256': digest(path.read_bytes())}


def document(path):
    """Read strict metadata and refuse duplicate keys or nonfinite constants."""
    def pairs(items):
        result = {}
        for key, value in items:
            assert key not in result
            result[key] = value
        return result
    return json.loads(path.read_bytes(), object_pairs_hook=pairs,
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))


def main():
    """Approve only the declared fixed release after independently checking each saved input."""
    assert os.environ['SLURM_JOB_ID'] == '6783003' and os.environ['SLURM_STEP_ID'].isdigit()
    assert os.getuid() == os.geteuid() == 2618919
    assert Path('/proc/self/cgroup').read_text().splitlines() == [
        '0::/system.slice/slurmstepd.scope/job_6783003/step_' + os.environ['SLURM_STEP_ID'] + '/user/task_0']
    binding = document(PACKET / 'release_binding.json')
    assert reference(PACKET / 'release_binding.json')['sha256'] == BINDING_SHA
    assert binding['kind'] == 'actual_owned_nsight_diagnostic_fixed_identifier_release_binding'
    assert binding['job_id'] == '6820154' and binding['command'] == ['scontrol', 'release', '6820154']
    assert binding['actual_release_performed'] is False
    assert binding['plan'] == reference(UNIT / 'measurement_plan.json')
    assert binding['plan']['sha256'] == PLAN_SHA
    assert binding['submission'] == reference(PACKET / 'submission_receipt.json')
    assert binding['held_query'] == reference(PACKET / 'release_preparation_query_return.json')
    submission = document(PACKET / 'submission_receipt.json')
    assert submission['job_id'] == '6820154' and submission['publication'] == PUBLICATION
    assert submission['submitted_held'] is True and type(submission['returncode']) is int and submission['returncode'] == 0
    assert submission['plan_sha256'] == PLAN_SHA and submission['data_unit'] == str(UNIT)
    assert (PACKET / 'actual_submission_stdout.raw').read_bytes() == b'6820154\n'
    assert (PACKET / 'actual_submission_stderr.raw').read_bytes() == b''
    actual_submission = document(PACKET / 'actual_submission_return.json')
    command = document(PACKET / 'held_submission_command.json')
    assert actual_submission['command'] == command and actual_submission['returncode'] == 0
    assert command == ['sbatch', '--hold', '--parsable', '--no-requeue', '--partition=gpu',
        '--account=sl5nw_base', '--qos=csresnolim', '--reservation=sl5nw_168', '--nodelist=jaguar03',
        '--nodes=1', '--ntasks=1', '--cpus-per-task=1', '--mem=2G', '--gres=gpu:nvidia_rtx_a4500:1',
        '--time=00:05:00', str(PACKET / 'job.sbatch')]
    ledgers = document(PACKET / 'submission_ledgers_fsynced.json')
    assert ledgers['job_id'] == '6820154' and len(ledgers['ledgers']) == 3
    assert ledgers['submission_row'] == submission['submission_row'] == [
        '6820154', 'nsight_process_identity_diagnostic', HELPER, str(PACKET / 'job.sbatch')]
    assert {row['path'] for row in ledgers['ledgers']} == {str(RUN / 'slurm' / name)
        for name in ['submitted_jobids.txt', 'submitted_jobids_uva.txt', 'submissions.tsv']}
    for row in ledgers['ledgers']:
        path = Path(row['path'])
        assert reference(path) == row
        lines = path.read_text().splitlines()
        if path.suffix == '.tsv':
            assert [line for line in lines if line.split('\t')[0] == '6820154'] == ['\t'.join(submission['submission_row'])]
        else:
            assert lines.count('6820154') == 1
    assert ledgers['at_utc'] < submission['at_utc'] < binding['at_utc']
    query = document(PACKET / 'release_preparation_query_return.json')
    assert query['command'] == ['scontrol', 'show', 'job', '-o', '6820154'] and query['returncode'] == 0
    assert (PACKET / 'release_preparation_query_stderr.raw').read_bytes() == b''
    query_raw = (PACKET / 'release_preparation_query_stdout.raw').read_text()
    fields = dict(re.findall(r'(?:^|\s)([A-Za-z][A-Za-z0-9:/]*)=([^\s]*)', query_raw))
    expected = {'JobId':'6820154', 'UserId':'sl5nw(2618919)', 'Account':'sl5nw_base', 'QOS':'csresnolim',
        'JobState':'PENDING','Reason':'JobHeldUser','Requeue':'0','Restarts':'0','RunTime':'00:00:00',
        'TimeLimit':'00:05:00','Partition':'gpu','ReqNodeList':'jaguar03','NumCPUs':'1','NumTasks':'1',
        'CPUs/Task':'1','MinMemoryNode':'2G','Reservation':'sl5nw_168','TresPerNode':'gres/gpu:nvidia_rtx_a4500:1',
        'AllocTRES':'(null)','Command':str(PACKET / 'job.sbatch')}
    assert all(fields.get(key) == value for key,value in expected.items()), fields
    assert fields['ReqTRES'] == 'cpu=1,mem=2G,node=1,billing=310,gres/gpu=1' and fields['NodeList'] == ''
    plan = document(UNIT / 'measurement_plan.json')
    assert plan['job_id'] is None and plan['gpu_uuid'] is None
    assert plan['helpers']['source_commit'] == HELPER and plan['packet'] == str(PACKET) and plan['unit_id'] == NAME
    assert plan['resources'] == {'account':'sl5nw_base','gres':'gpu:nvidia_rtx_a4500:1','host_memory_gib':2,
        'node':'jaguar03','nodes':1,'partition':'gpu','qos':'csresnolim','requested_cpus':1,'requeue':False,
        'reservation':'sl5nw_168','tasks':1,'wall_seconds':300}
    assert plan['admission_granted'] is plan['worker_admitted'] is plan['physical_gres_mapping_verified'] is False
    assert plan['work']['learner_updates'] == plan['work']['warmup_rollouts'] == 0
    for key in ['scientific_credit','case_credit','idea_credit','environment_credit','learner_credit','profile_credit']:
        assert type(plan[key]) is int and plan[key] == 0
    immutable = document(PACKET / 'immutable_artifacts.json')
    for path,row in immutable['files'].items():
        raw = Path(path).read_bytes()
        assert len(raw) == row['bytes'] and digest(raw) == row['sha256']
    frozen = document(PACKET / 'launcher_source_freeze.json')
    publication_files = []
    for path,wanted in frozen['files'].items():
        path = Path(path)
        assert digest(path.read_bytes()) == wanted
        relative = str(path.relative_to(INTEGRATION))
        raw = subprocess.check_output(['git','-C',str(INTEGRATION),'show',PUBLICATION + ':' + relative])
        assert digest(raw) == wanted
        publication_files.append(relative)
    for relative,wanted in plan['helpers']['files'].items():
        source = Path(plan['helpers']['source_root']) / relative
        assert digest(source.read_bytes()) == wanted == digest((UNIT / 'code' / Path(relative).name).read_bytes())
        raw = subprocess.check_output(['git','-C',str(plan['helpers']['source_root']),'show',HELPER + ':' + relative])
        assert digest(raw) == wanted
    old_packet = RUN / 'remote_setup/2026-11-04-18-33_nsight-process-observation_jaguar03_gpu-count-1_processor-count-1_host-memory-gib-2_time-limit-minutes-5_placement-unreserved_qos-normal_tool-version-2024.7.1.84_environment-transitions-0'
    retirement = document(old_packet / 'retirement_completion.json')
    assert retirement['job_id'] == '6820096' and retirement['cancelled_once'] is retirement['held_once'] is True
    assert retirement['elapsed_seconds'] == 0 and retirement['actual_terminal_state'] == 'CANCELLED by 2618919'
    assert retirement['new_reviewed_packet'] == str(PACKET)
    assert retirement['raw_accounting'] == reference(old_packet / 'retirement_settled_terminal_accounting_stdout.raw')
    assert (old_packet / 'retirement_settled_terminal_accounting_stdout.raw').read_bytes() == b'6820096|CANCELLED by 2618919|0|None assigned|0:0\n'
    assert not (PACKET / 'release_receipt.json').exists() and not (PACKET / 'release_intent.json').exists()
    receipt = {'schema_version':1, 'kind':'independent_actual_reserved_diagnostic_fixed_release_review',
        'passed':True,'binding':reference(PACKET / 'release_binding.json'), 'job_id':'6820154',
        'reviewer':'/root/review_historical_consumer_bridge', 'reviewed_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'actual_cpu_job':'6783003','actual_cpu_step':os.environ['SLURM_STEP_ID'],
        'source_freeze':reference(PACKET / 'launcher_source_freeze.json'), 'published_source':PUBLICATION,
        'published_file_count':len(publication_files), 'all_three_ledger_snapshots_match':True,
        'actual_held_query':reference(PACKET / 'release_preparation_query_stdout.raw'),
        'submission':reference(PACKET / 'submission_receipt.json'), 'plan':reference(UNIT / 'measurement_plan.json'),
        'old_unstarted_request_terminal':reference(old_packet / 'retirement_completion.json'),
        'scheduler_actions_performed':0,'gpu_or_tool_actions_performed':0,'scientific_credit':0,'admission_granted':False,
        'limitation':'Saved exact owned scheduler records reviewed; this review made no fresh scheduler query.'}
    output = PACKET / 'release_independent_review.json'
    with output.open('xb') as stream:
        stream.write((json.dumps(receipt,indent=2,sort_keys=True)+'\n').encode())
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps(reference(output),sort_keys=True))


if __name__ == '__main__':
    main()
