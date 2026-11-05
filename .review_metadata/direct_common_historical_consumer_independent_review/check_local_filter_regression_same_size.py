"""Use a temporary Git repository to test the exact allowed status environment."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / '10_jax_exploration_platform/scripts/atari_pixel_campaign'
sys.path.insert(0, str(SCRIPTS))
import direct_common_confirmation as guard


def main():
    """Create only temporary fixture files and record an actual benign filter invocation."""
    assert os.environ['SLURM_JOB_ID'] == '6783003' and os.environ['SLURM_STEP_ID'].isdigit()
    before = dict(os.environ)
    environment = guard.metadata_environment(before)
    commands = []

    def run(command):
        """Run one declared temporary-repository Git command without a working-directory override."""
        outcome = subprocess.run(command, env=environment, capture_output=True, check=True)
        commands.append({'command':command,'returncode':outcome.returncode,
            'stdout':outcome.stdout.decode(),'stderr':outcome.stderr.decode()})
        return outcome

    with tempfile.TemporaryDirectory(prefix='independent_git_local_filter_') as directory:
        root = Path(directory)
        run(['git','-C',str(root),'init','--quiet'])
        tracked = root / 'tracked.txt'
        tracked.write_bytes(b'before\n')
        run(['git','-C',str(root),'add','--','tracked.txt'])
        marker = root / 'marker_events.jsonl'
        program = root / 'benign_clean_filter.py'
        program.write_text('''"""Write a temporary marker and preserve filter input bytes unchanged."""
import json
import os
from pathlib import Path
import sys
raw = sys.stdin.buffer.read()
with Path(__file__).with_name('marker_events.jsonl').open('a') as stream:
    stream.write(json.dumps({'pid':os.getpid(),'argv':sys.argv,'input_hex':raw.hex()})+'\\n')
sys.stdout.buffer.write(raw)
''')
        (root / '.gitattributes').write_text('tracked.txt filter=independent_benign_filter\n')
        with (root / '.git/config').open('a') as stream:
            stream.write('\n[filter "independent_benign_filter"]\n\tclean = /usr/bin/python3 -I -S -B ' + str(program) + '\n')
        tracked.write_bytes(b'change\n')
        assert not marker.exists()
        command = ['git','-C',str(root),'status','--porcelain','--untracked-files=all']
        guard.metadata_event_guard('subprocess.Popen',('git',command,None,environment),{str(root)})
        result = run(command)
        events = [json.loads(line) for line in marker.read_text().splitlines()] if marker.exists() else []
        receipt = {'kind':'independent_actual_temporary_git_filter_regression',
            'actual_job':'6783003','actual_step':os.environ['SLURM_STEP_ID'],
            'exact_fixed_environment':environment,'exact_allowed_command':command,
            'audit_accepted_exact_command':True,'marker_existed_before_status':False,
            'marker_exists_after_status':marker.exists(),'marker_events':events,
            'fixture_commands':commands,'temporary_fixture_root':str(root),
            'filter_source':program.read_text(),'git_config':(root/'.git/config').read_text(),
            'attributes':(root/'.gitattributes').read_text(),'parent_environment_unchanged':before==dict(os.environ),
            'campaign_or_production_source_changed':False,'saved_scientific_evidence_recomputed':False,
            'numerical_imports':[],'scientific_credit':0,'admission_granted':False,
            'scope':'Three fixture Git commands and any clean-filter callbacks; all fixture writes stayed in its temporary directory.'}
        assert before == dict(os.environ)
        print(json.dumps(receipt,indent=2,sort_keys=True))


if __name__ == '__main__':
    main()
