"""Check only the actual runtime literal and unchanged metadata boundaries; no unit is prepared."""
import hashlib
import importlib.util
from pathlib import Path
import subprocess
import unittest
from unittest.mock import patch

W=Path(__file__).resolve().parents[2]
OPERATIONS=W/'.review_metadata/profile_target_operational_preparation'
PARENT='ee4168eecb3d5d9aa6af4ebcdb4dde9c4750b536'
PREVIOUS='2aa6df92f7d9da20e2fbc1d15a20df16ba10ba1e'
RUNTIME='4bb3a61cb92fe48ab5ba68fb52f70b47736a1174'
NAMES=('derive_one_copy_qualification_request.py','derive_two_copy_qualification_request.py','start_direct_profile_qualification_once.py')
PREFIX='10_jax_exploration_platform/scripts/atari_pixel_campaign/'

def load(name):
    """Load a metadata entrypoint without invoking its default or action path."""
    path=OPERATIONS/name;spec=importlib.util.spec_from_file_location('literal_pin_'+name.replace('.','_'),path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


class FinalRuntimePinTests(unittest.TestCase):
    """Use real source receipts and refuse unknown typed pins before source access."""
    helper_root=None
    actual_receipts=[]

    def test_only_actual_runtime_literal_changes_in_all_three_entrypoints(self):
        """Every prior science, count, seed, publication, process and custody byte is preserved."""
        for name in NAMES:
            path=OPERATIONS/name;raw=path.read_bytes()
            previous=subprocess.check_output(['git','-C',str(W),'show',PARENT+':'+str(path.relative_to(W))])
            self.assertEqual(raw.count(RUNTIME.encode()),1)
            self.assertEqual(raw.replace(RUNTIME.encode(),PREVIOUS.encode()),previous)
            self.assertEqual(load(name).RUNTIME,RUNTIME)

    def test_unknown_and_wrongly_typed_runtime_pins_refuse_before_git(self):
        """None, Boolean, integer and floating-point values cannot become a source commit."""
        for name in NAMES[:2]:
            module=load(name)
            for value in (None,True,1,1.0):
                with self.subTest(name=name,value=value),patch.object(module,'RUNTIME',value),patch.object(module.subprocess,'check_output') as command:
                    with self.assertRaisesRegex(ValueError,'has not been pinned'):module.runtime(Path('/not/a/source/image'))
                    command.assert_not_called()

    def test_real_clean_helper_has_exact_current_source_and_unchanged_target_counts(self):
        """Read actual committed45/36 helper receipts, and refuse the old source before module load."""
        self.assertIsNotNone(self.helper_root)
        root=Path(self.helper_root)
        q=load(NAMES[0]).runtime(root)
        profile_path=root/PREFIX/'profile_selected.py'
        profile=q.adjacent({'helpers':{'files':{PREFIX+profile_path.name:hashlib.sha256(profile_path.read_bytes()).hexdigest()}}},'profile_selected')
        receipts=[]
        for configured in (False,True):
            receipt=profile.helper_receipt(configured=configured,direct=not configured,owned_processes=configured,
                direct_profile=not configured,configured_native=configured,configured_prerequisites=configured,profile_target=True)
            self.assertEqual(receipt['source_commit'],RUNTIME);self.assertEqual(receipt['source_root'],str(root.resolve()))
            self.assertEqual(len(receipt['files']),36 if configured else 45)
            self.assertIn(PREFIX+'profile_target_tool_scope.py',receipt['files'])
            receipts.append({'hardware_class':'jaguar03_rtx_a4500' if configured else 'h100','helpers':receipt})
        self.__class__.actual_receipts=receipts
        old=Path('/p/rlprojects/RND_worktrees/2026-11-05-13-52_profile-qualification-helper_source-2aa6df92f7d9_h100-helpers-45_a4500-helpers-36_gpu-count-0')
        for name in NAMES[:2]:
            with self.assertRaisesRegex(ValueError,'exact reviewed commit'):load(name).runtime(old)


if __name__=='__main__':unittest.main(verbosity=2)
