"""Preserve the actual failed48-case record while testing an explicitly source-pinned future pytest invocation."""
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock,patch

SCRIPTS=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(SCRIPTS))
import configured_measurement as support
REPOSITORY=SCRIPTS.parents[2]
FIXTURE=Path(__file__).with_name('fixtures')/'qualification_pytest_root/original_tests_failed_6782640.json'
ACTUAL_FAILED_SHA256='5c44c6868a10bceffa0cd59b2b20d6c157c33bbbd881e054f8847cd7911ca9ab'
# These original test-source bytes are bound by the actual failed plan's qualification_sources receipt.
ORIGINAL_TEST_SOURCE_HASHES={'10_jax_exploration_platform/tests/atari_pixels/test_equations.py': '51f179041971e3e6ff1f6a3a5f58e9100dafe1d9975d9d5d5a6698828dda7dd5', '10_jax_exploration_platform/tests/atari_pixels/test_indexed_frames.py': 'e55375717fbd61ba1dc3eac925c835af7b1578816e2a89d7a7fb927b176119a1', '10_jax_exploration_platform/tests/atari_pixels/test_optimizations.py': '1dafd378e1c45955fb24945f8c2978eb7b30747930bd2050dcd5aa16c3fed962', '10_jax_exploration_platform/tests/atari_pixels/test_scalar_settings.py': '20b498b70216646105ef901273c200b2f403a7994bc2924735cc3937d7317a3e', '10_jax_exploration_platform/tests/atari_pixels/test_training.py': 'a32983ea48bf3510ad9dcfe518fac2a3d7827366ef565592a8c0ce51645df1b9', '10_jax_exploration_platform/tests/test_atari_pixel_environment.py': '5df05885a4301e1a22416852e754f037e3abf35f5bc366ef78cc8fbe188651df'}


class OriginalPytestRootTests(unittest.TestCase):
    """Keep arbitrary roots and incomplete case matrices rejected without numerical work or actual pytest execution."""

    def setUp(self):
        """Load immutable actual failed bytes and derive an explicitly synthetic future-root record."""
        self.assertEqual(hashlib.sha256(FIXTURE.read_bytes()).hexdigest(),ACTUAL_FAILED_SHA256)
        self.failed=json.loads(FIXTURE.read_text())
        self.root=REPOSITORY/'10_jax_exploration_platform'
        for relative,digest in ORIGINAL_TEST_SOURCE_HASHES.items():
            self.assertEqual(hashlib.sha256((REPOSITORY/relative).read_bytes()).hexdigest(),digest)
        self.plan={'source_root':str(REPOSITORY),'python':sys.executable,
            'qualification_sources':{'files':ORIGINAL_TEST_SOURCE_HASHES}}
        self.fixed=copy.deepcopy(self.failed)
        old_platform=Path(self.failed['command'][-1]).parents[1]
        def remap(node):
            """Before: outer-root-prefixed actual node; after: same exact case relative to the pinned platform root."""
            filename,separator,case=node.partition('::')
            if not filename:return node
            relative=(Path(self.failed['collection_rootdir'])/filename).relative_to(old_platform)
            return relative.as_posix()+separator+case
        self.fixed['command']=[sys.executable,'-m','pytest',*support.qualification_pytest_arguments(self.plan)]
        self.fixed['collection_rootdir']=str(self.root)
        self.fixed['collected_nodeids']=[remap(node) for node in self.failed['collected_nodeids']]
        for row in self.fixed['reports']:row['nodeid']=remap(row['nodeid'])
        for row in self.fixed['collection_reports']:row['nodeid']=remap(row['nodeid'])

    def validate(self,record=None,count=48):
        """Run the actual strict matrix/root/command reader on the declared synthetic record only."""
        return support.validate_test_collection(self.plan,record or self.fixed,str(count)+' passed in 0.1s\n')

    def test_real_failed_root_and_raw_bytes_remain_rejected(self):
        """Actual passed tests under the unrelated outer root confer no retrospective qualification credit."""
        self.assertEqual(self.failed['collection_rootdir'],'/p/rlprojects/RND_worktrees')
        self.assertEqual(len(self.failed['collected_nodeids']),48)
        self.assertEqual(len(self.failed['reports']),144)
        self.assertEqual(self.failed['returncode'],0)
        with self.assertRaisesRegex(ValueError,'unrelated root directory'):
            self.validate(self.failed)
        self.assertEqual(hashlib.sha256(FIXTURE.read_bytes()).hexdigest(),ACTUAL_FAILED_SHA256)

    def test_exact_root_with_same_full48_cases_and144_phases_passes_metadata(self):
        """The derived record retains every actual case and all three passed phases under the one future exact root."""
        self.assertEqual(sum(support.test_collection_contract(self.plan).values()),48)
        self.assertEqual(self.validate(),48)

    def test_outer_and_formerly_allowed_roots_are_not_added(self):
        """Neither parent checkout, tests subdirectory nor the actual historical outer directory is accepted."""
        for root in (self.root.parent,self.root/'tests',Path(self.failed['collection_rootdir'])):
            with self.subTest(root=str(root)):
                changed=copy.deepcopy(self.fixed);changed['collection_rootdir']=str(root)
                with self.assertRaisesRegex(ValueError,'unrelated root directory'):self.validate(changed)

    def test_missing_or_changed_explicit_root_argument_is_rejected(self):
        """A coherent stored root cannot replace the exact source-bound runner command."""
        for command in ([sys.executable,'-m','pytest','-q',*self.fixed['command'][-2:]],
                [sys.executable,'-m','pytest','--rootdir',str(self.root.parent),'-q',*self.fixed['command'][-2:]],
                [*self.fixed['command'],'-k','test_forward']):
            with self.subTest(command=command):
                changed=copy.deepcopy(self.fixed);changed['command']=command
                with self.assertRaisesRegex(ValueError,'pytest command'):self.validate(changed)

    def test_reduced48_case_matrix_and_missing_phase_stay_rejected(self):
        """Matching success text never grants a reduced case set or missing teardown phase full credit."""
        changed=copy.deepcopy(self.fixed);removed=changed['collected_nodeids'].pop()
        changed['reports']=[row for row in changed['reports'] if row['nodeid']!=removed]
        with self.assertRaisesRegex(ValueError,'matrix was truncated'):self.validate(changed,count=47)
        changed=copy.deepcopy(self.fixed);changed['reports'].pop()
        with self.assertRaisesRegex(ValueError,'passed setup, call and teardown'):self.validate(changed)

    def test_parent_path_and_substituted_case_stay_rejected(self):
        """The pinned root cannot make an escaped or substituted source case legitimate."""
        changed=copy.deepcopy(self.fixed);old=changed['collected_nodeids'][0]
        changed['collected_nodeids'][0]='../outside.py::test_substitution'
        for row in changed['reports']:
            if row['nodeid']==old:row['nodeid']=changed['collected_nodeids'][0]
        with self.assertRaisesRegex(ValueError,'unsupported original test node'):self.validate(changed)

    def test_actual_runner_passes_and_records_the_same_pinned_argv(self):
        """Exercise the real runner to its post-test boundary using synthetic hooks, with all learner/render work forbidden."""
        captured=[]
        def pytest_main(arguments,plugins):
            """Only replay metadata hooks; no test implementation or numerical module is executed."""
            captured.append(arguments)
            reporter=plugins[0]
            reporter.pytest_collection_finish(types.SimpleNamespace(items=[types.SimpleNamespace(nodeid=node)
                for node in self.fixed['collected_nodeids']],config=types.SimpleNamespace(rootpath=self.root)))
            for row in self.fixed['collection_reports']:
                reporter.pytest_collectreport(types.SimpleNamespace(**row))
            for row in self.fixed['reports']:
                reporter.pytest_runtest_logreport(types.SimpleNamespace(nodeid=row['nodeid'],when=row['phase'],
                    outcome=row['outcome'],duration=row['duration_seconds']))
            print('48 passed in 0.1s')
            return 0
        fake_pytest=types.ModuleType('pytest');fake_pytest.main=pytest_main
        fake_training=types.ModuleType('train_optimized')
        fake_training.atomic_json=lambda path,value:Path(path).write_text(json.dumps(value))
        class StopBeforeRenderer(RuntimeError):
            """End the metadata-only runner probe before its first renderer/model import."""
        numerical=Mock(side_effect=AssertionError('numerical work forbidden'))
        with tempfile.TemporaryDirectory() as temporary,patch.dict(sys.modules,{'pytest':fake_pytest,'train_optimized':fake_training}),\
                patch.object(support,'validate_original_qualification',return_value={}),\
                patch.object(support,'verify_qualification_sources'),\
                patch.object(support.importlib.util,'spec_from_file_location',side_effect=StopBeforeRenderer):
            directory=Path(temporary)
            with self.assertRaises(StopBeforeRenderer):
                support.run_qualification(directory,self.plan,None,None,numerical,numerical,numerical)
            stored=json.loads((directory/'original_tests.json').read_text())
            self.assertEqual(captured,[support.qualification_pytest_arguments(self.plan)])
            self.assertEqual(stored['command'],self.fixed['command'])
            self.assertEqual(stored['collection_rootdir'],str(self.root))
            self.assertEqual(support.validate_test_collection(self.plan,stored,(directory/'original_tests.stdout').read_text()),48)
        numerical.assert_not_called()


if __name__=='__main__':unittest.main()
