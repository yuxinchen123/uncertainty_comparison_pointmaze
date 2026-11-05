"""Check corrected subprocess decisions with synthetic events only."""
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / '10_jax_exploration_platform/scripts/atari_pixel_campaign'
sys.path.insert(0, str(SCRIPTS))
import direct_common_confirmation as guard
import direct_historical_consumer as historical


def main():
    """Retest the earlier failures and fixed environment without launching commands."""
    assert os.environ['SLURM_JOB_ID'] == '6783003' and os.environ['SLURM_STEP_ID'].isdigit()
    before = dict(os.environ)
    environment = guard.metadata_environment(before)
    guard.require_metadata_environment(environment)
    guard._ACTIVE_HISTORICAL_SOURCE_SHA256 = hashlib.sha256((SCRIPTS / historical.NAME).read_bytes()).hexdigest()
    expected = ['/usr/bin/python3', '-I', '-S', '-B', str(SCRIPTS / historical.NAME), '--read-only-stdin']
    results = []

    def decision(name, executable, command, working_directory=None, env=environment, allowed=False):
        """Require the expected allow/refuse outcome of one synthetic audit event."""
        try:
            guard.metadata_event_guard('subprocess.Popen', (executable, command, working_directory, env), {'/declared/source'})
        except (ValueError, RuntimeError) as error:
            actual = False
            detail = str(error)
        else:
            actual = True
            detail = None
        assert actual is allowed, (name, actual, detail)
        results.append({'name':name,'allowed':actual,'passed':True,'exception':detail})

    decision('exact Python form and fixed environment', expected[0], expected, allowed=True)
    decision('substituted Python executable', '/another/executable', expected)
    decision('tuple arguments', expected[0], tuple(expected))
    decision('Python working directory override', expected[0], expected, '/another/directory')
    decision('Python extra argument', expected[0], expected + ['another'])
    decision('Python changed script', expected[0], expected[:4] + ['/another/script', expected[-1]])
    decision('Python missing environment', expected[0], expected, env=None)
    for change in ({'PATH':'/another/bin'}, {'LD_PRELOAD':'/another/library'}, {'PYTHONPATH':'/another/python'},
            {'GIT_CONFIG_PARAMETERS':'bad'}, {'GIT_CONFIG_COUNT':'0'}, {'GIT_DIR':'/another/repository'},
            {'GIT_EXTERNAL_DIFF':'/another/executable'}, {'GIT_INDEX_FILE':'/another/index'},
            {'GIT_CONFIG_VALUE_0':'/another/fsmonitor'}):
        decision('changed environment ' + next(iter(change)), expected[0], expected, env={**environment,**change})
    good_git = ['git','-C','/declared/source','show','a' * 40 + ':source.py']
    decision('exact Git form and fixed environment','git',good_git,allowed=True)
    decision('substituted Git executable','/another/executable',good_git)
    decision('Git working directory override','git',good_git,'/another/directory')
    for arguments in (['diff','--ext-diff'], ['show','--output=/another/output','a' * 40 + ':source.py'],
            ['status','--porcelain','--untracked-files=all','--','--output=/another/output'],
            ['show','HEAD:source.py'], ['show','a' * 40 + ':../source.py'],
            ['ls-tree','-rz','a' * 40,'--',':(attr:filter)source.py']):
        decision('refused Git arguments ' + repr(arguments),'git',['git','-C','/declared/source',*arguments])
    for arguments in (['rev-parse','HEAD'], ['rev-parse','--show-toplevel'], ['symbolic-ref','--quiet','HEAD'],
            ['cat-file','blob','a' * 40], ['ls-tree','-rz','a' * 40,'--','plain_source/'],
            ['ls-tree','-r','--name-only','a' * 40,'--','plain_source/'],
            ['status','--porcelain','--untracked-files=all'], ['status','--porcelain','--','plain_source/']):
        decision('allowed Git arguments ' + repr(arguments),'git',['git','-C','/declared/source',*arguments],allowed=True)
    assert before == dict(os.environ)
    assert historical.WAIT_SECONDS == 180 and historical.PROTOCOL['digest_cache'] == 'disabled'
    print(json.dumps({'kind':'independent_corrected_subprocess_boundary_metadata_checks','passed':True,
        'actual_job':'6783003','actual_step':os.environ['SLURM_STEP_ID'],'checks':results,
        'parent_environment_unchanged':True,'fixed_child_environment':environment,'actual_commands_executed':0,
        'saved_scientific_evidence_recomputed':False,'scientific_credit':0,'admission_granted':False},indent=2,sort_keys=True))


if __name__ == '__main__':
    main()
