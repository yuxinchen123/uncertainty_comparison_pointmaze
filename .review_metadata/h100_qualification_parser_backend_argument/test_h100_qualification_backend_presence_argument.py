"""Test only the H100 operational argument correction with real unchanged parser bytes."""
import contextlib
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import unittest
from unittest.mock import patch

D=Path(__file__).resolve().parent
W=D.parent.parent
S='10_jax_exploration_platform/scripts/atari_pixel_campaign'
HELPER=Path('/p/rlprojects/RND_worktrees/2026-11-05-11-56_profile-qualification-helper_source-d1469fd063a4_h100-helpers-44_a4500-helpers-35_gpu-count-0')
R=Path('/p/rlprojects/RND_worktrees/merge-atari-cleanrl/10_jax_exploration_platform/benchmark_runs/2026-10-31-00-11_jaxatari_montezuma_pixels_cleanrl_ppo_rnd_environments-per-copy-128_rollout-steps-128_ideas-30-plus-20_steps-per-copy-2000000000_seed-start-1')
P=R/'remote_setup/2026-11-05-12-14_profile-source-qualification_h100-nvl_serval02-card-1_copies-1_environments-128_rollout-steps-128_epochs-4_minibatches-4_warmup-rollouts-50_reference-updates-1_seed-start-1_hours-3'


def load(path,name):
    specification=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(specification);specification.loader.exec_module(module)
    return module


class H100PresenceArgument(unittest.TestCase):
    """Read the failed request; do not prepare it, alter it or create a successor."""
    def test_only_declared_operational_lines_changed(self):
        source=(W/'.review_metadata/profile_target_operational_preparation/derive_one_copy_qualification_request.py').read_text()
        old=(D/'derive_one_copy_qualification_request_before_presence_argument.py').read_text()
        addition="            '--direct-backend-image-record', backend_ref['path'],\n"
        guard="    if not configured:\n        require(argv.count('--direct-backend-image-record') == 1\n            and argv[argv.index('--direct-backend-image-record') + 1] == request['backend_image_record']['path'],\n            'the parser presence argument must name the same fresh profile backend record')\n"
        self.assertEqual(source.count(addition),1)
        self.assertEqual(source.count(guard),1)
        self.assertEqual(source.replace(addition,'').replace(guard,''),old)
        self.assertEqual(hashlib.sha256(old.encode()).hexdigest(),'b377e523465a803a9a58657519d8b6e1aa8922c124486adf839e50448cdfda19')

    def test_actual_failed_command_parses_and_old_backend_reader_stays_strict(self):
        request_raw=(P/'preparation_request.json').read_bytes()
        request=json.loads(request_raw)
        profile=load(HELPER/S/'profile_selected.py','unchanged_actual_profile_parser')
        args=request['profile_prepare_arguments']
        self.assertNotIn('--direct-backend-image-record',args)
        before_error=io.StringIO()
        with patch.object(profile,'prepare',side_effect=AssertionError('no actual preparation')),contextlib.redirect_stderr(before_error),self.assertRaises(SystemExit) as stopped:
            profile.parse_arguments(args)
        self.assertEqual(stopped.exception.code,2)
        self.assertIn('direct profiling requires',before_error.getvalue())
        record=request['backend_image_record']
        self.assertEqual(record['sha256'],'19e752ef9b93999e8bec28a0706947935c31f4229a3b33556597de8f86575412')
        raw=Path(record['path']).read_bytes()
        self.assertEqual(hashlib.sha256(raw).hexdigest(),record['sha256'])
        corrected=[*args,'--direct-backend-image-record',record['path']]
        self.assertEqual(corrected.count('--direct-backend-image-record'),1)
        self.assertEqual(corrected[corrected.index('--direct-backend-image-record')+1],request['backend_image_record']['path'])
        with patch.object(profile,'prepare',side_effect=AssertionError('no actual preparation')):
            parsed=profile.parse_arguments(corrected)
        self.assertTrue(parsed.prepare_only)
        self.assertEqual(parsed.direct_backend_image_record,Path(record['path']))
        new=load(HELPER/S/'profile_qualification_backend.py','unchanged_new_profile_backend_reader')
        new.validate_expected(json.loads(raw),str(parsed.python))
        old=load(HELPER/S/'direct_backend_image.py','unchanged_old_direct_backend_reader')
        with self.assertRaisesRegex(ValueError,'schema two'):
            old.validate_expected(json.loads(raw),str(parsed.python))
        self.assertFalse((R/'data'/request['controller_unit_id']).exists())
        self.assertFalse((R/'data'/parsed.unit_id).exists())
        self.assertEqual((P/'preparation_request.json').read_bytes(),request_raw)
        self.assertEqual(json.loads((P/'root_actual_canonical_preparation_return.json').read_bytes())['returncode'],2)


if __name__=='__main__':unittest.main(verbosity=2)
