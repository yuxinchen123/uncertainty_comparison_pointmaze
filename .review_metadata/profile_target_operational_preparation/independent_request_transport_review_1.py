"""Preserve and reproduce the local-only publication gap without a target action."""
import ast
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import textwrap
import time
import unittest
from unittest.mock import patch

D = Path(__file__).resolve().parent
FREEZE = D / 'request_and_direct_transport_source_freeze_1.json'
EXPECTED = 'fd264f64ddaec08cf5452e606a5769c9f44045ec1d610aaabaf4edead9aeae10'


def sha(path):
    """Hash only the bounded source and metadata involved in this review."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    """Retain actual independent review bytes without replacing earlier evidence."""
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write('\n');stream.flush();os.fsync(stream.fileno())


def main():
    """Run source-only checks and one explicitly synthetic private-Git publication case."""
    started = time.monotonic()
    assert sha(FREEZE) == EXPECTED
    freeze = json.loads(FREEZE.read_bytes())
    for value in freeze['files'].values():
        assert sha(value['path']) == value['sha256']
    archive = D / 'preserved_request_transport_local_publication_rejection'
    archive.mkdir(exist_ok=True)
    copied = []
    for path in [Path(value['path']) for value in freeze['files'].values()] + [FREEZE,
            D / 'request_and_direct_transport_author_handoff.json']:
        destination = archive / path.name
        assert not destination.exists() or destination.read_bytes() == path.read_bytes()
        if not destination.exists():shutil.copyfile(path, destination)
        copied.append({'original':str(path),'preserved':str(destination),'sha256':sha(destination)})
    if not (archive / 'manifest.json').exists():write(archive / 'manifest.json', {'files':copied,'source_freeze_sha256':EXPECTED})
    specification = importlib.util.spec_from_file_location('independent_operational_tests',
        D / 'test_qualification_request_and_direct_transport.py')
    tests = importlib.util.module_from_spec(specification);specification.loader.exec_module(tests)
    output = io.StringIO()
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(tests.OperationalChecks)
    result = unittest.TextTestRunner(stream=output, verbosity=2).run(suite)
    assert result.wasSuccessful(), output.getvalue()

    # Retain the author's explicit synthetic science/runtime premises, but use a
    # genuine private Git repository for the proposed publication and ancestry.
    tree = ast.parse(Path(tests.__file__).read_text())
    method = next(node for node in next(node for node in tree.body if isinstance(node,ast.ClassDef)).body
        if isinstance(node,ast.FunctionDef) and node.name == 'test_complete_synthetic_transport_checks_and_changed_plan_refuse')
    original = textwrap.dedent(ast.get_source_segment(Path(tests.__file__).read_text(),method))
    changes = {
        'root = D.parent.parent':'root = LOCAL_REPRO_ROOT',
        "publication='a'*40":"actual_git(root,'add','-A');actual_git(root,'commit','-qm','synthetic local publication');publication=actual_git(root,'rev-parse','HEAD').decode().strip()",
        "if argv==('merge-base','--is-ancestor',publication,'HEAD'):return b''":"if argv==('merge-base','--is-ancestor',publication,'HEAD'):LOCAL_REPRO_QUERIES.append(list(argv));return actual_git(root,*argv)",
        "return (root/relative).read_bytes()":"return actual_git(root,*argv)",
    }
    modified = original
    for before,after in changes.items():
        assert modified.count(before) == 1, before
        modified = modified.replace(before,after)
    queries = []
    environment = dict(os.environ, GIT_OPTIONAL_LOCKS='0',GIT_TERMINAL_PROMPT='0',
        GIT_AUTHOR_NAME='synthetic fixture',GIT_AUTHOR_EMAIL='fixture@example.invalid',
        GIT_COMMITTER_NAME='synthetic fixture',GIT_COMMITTER_EMAIL='fixture@example.invalid')
    def actual_git(root,*arguments):
        """Execute only private fixture Git metadata with no network or scheduler access."""
        return subprocess.run(['git','-C',str(root),*arguments],check=True,capture_output=True,timeout=15,env=environment).stdout
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        actual_git(root,'init','-q')
        namespace = dict(tests.__dict__,LOCAL_REPRO_ROOT=root,LOCAL_REPRO_QUERIES=queries,actual_git=actual_git)
        exec(compile(modified,'<independent private local publication fixture>','exec'),namespace)
        namespace[method.name](unittest.TestCase())
        assert actual_git(root,'remote') == b''
        assert queries and not any('refs/remotes' in str(row) for row in queries)
    with patch.object(tests.TRANSPORT,'verify',return_value=({},Path('/synthetic'),['synthetic'])), \
            patch.object(tests.TRANSPORT,'execute',side_effect=AssertionError('default cannot execute')):
        tests.TRANSPORT.main(['--configuration','/synthetic','--publication','a'*40])
    for value in freeze['files'].values():
        assert sha(value['path']) == value['sha256']
    record = {'schema_version':1,'kind':'independent_operational_request_transport_initial_review',
        'approved':False,'passed':False,'source_freeze':{'path':str(FREEZE),'sha256':EXPECTED},
        'preservation_manifest':{'path':str(archive/'manifest.json'),'sha256':sha(archive/'manifest.json')},
        'author_methods_independently_passed':result.testsRun,'private_git_without_origin_accepted':True,
        'actual_local_ancestry_queries':queries,
        'findings':[{'kind':'publication_tracking_missing','detail':'verify accepts a local-only publication with no origin or remote-tracking ref'},
            {'kind':'actual_A4500_packet_name_join_missing','detail':'request deriver does not require packet.name==profile_unit_id, although owned allocation inputs use that exact remote_setup path'}],
        'default_no_action_verified':True,'source_unchanged':True,'target_operations':0,
        'actual_job_id':os.environ['SLURM_JOB_ID'],'actual_step_id':os.environ['SLURM_STEP_ID'],
        'seconds':time.monotonic()-started,'admission_granted':False}
    write(D/'independent_request_transport_initial_review.json',record)
    print(json.dumps(record));print(output.getvalue())


if __name__=='__main__':main()
