"""Exercise the real opt-in preparation and worker ordering without numerical work."""
import hashlib
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

SCRIPTS = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(SCRIPTS), str(SCRIPTS/'tests')]
import common_confirmation as common
import test_configured_measurement as configured_tests


class CommonPreparationTests(unittest.TestCase):
    """Reuse the complete existing metadata fixture and retain actual producer entry points."""

    def setUp(self):
        """Prepare only synthetic prior references while staging all real helper bytes."""
        self.context = configured_tests.ConfiguredPreparationTests()
        self.context.setUp(); self.addCleanup(self.context.doCleanups)
        self.fixture = self.context.fixture
        self.selected = configured_tests.selected
        self.fixture.selected['learner']['rollout_unroll'] = self.fixture.reference['learner']['rollout_unroll']
        self.fixture.selected['environment'].update(cache_room_backgrounds=True, render_final_two_only=True)
        self.fixture.files['selected'].write_text(json.dumps(self.fixture.selected))
        self.fixture.helpers['files'][common.PREFIX+common.NAME] = hashlib.sha256((SCRIPTS/common.NAME).read_bytes()).hexdigest()
        self.arguments = self.context.arguments()
        self.arguments.python = Path(sys.executable)
        for key in ('common_correctness_record','common_correctness_review'):
            path = self.fixture.run_dir/(key+'.json');path.write_text('{"synthetic_metadata_fixture":true}')
            setattr(self.arguments,key,path)

    def prepare(self):
        """Use actual preparation, replacing only the independently exercised prior-proof child."""
        with patch.object(common,'request',return_value={'synthetic_metadata_fixture':True}) as child:
            directory=self.context.prepare(self.arguments)
        child.assert_called_once()
        return directory

    def test_actual_prepare_freezes_real_refs_and_complete_opt_in_staging(self):
        """The prepared plan must bind both inputs before its immutable hash is written."""
        directory=self.prepare();raw=(directory/'measurement_plan.json').read_bytes();plan=json.loads(raw)
        self.assertEqual(plan['common_correctness_inputs'],{'result':common.reference(self.arguments.common_correctness_record),
            'review':common.reference(self.arguments.common_correctness_review)})
        self.assertEqual((plan['rounds'],plan['iterations'],plan['minimum_updates_per_round']),(7,10,10))
        self.assertEqual(json.loads((directory/'measurement_prepared.json').read_bytes()),{'plan_sha256':hashlib.sha256(raw).hexdigest()})
        self.assertEqual(len(list((directory/'code').iterdir())),len(self.selected.required_helpers(True,False,True)))
        self.assertEqual((directory/'code'/common.NAME).read_bytes(),(SCRIPTS/common.NAME).read_bytes())
        self.assertFalse((directory/'owned_allocation_inputs.json').exists())

    def test_actual_worker_gate_precedes_card_metadata_and_numerical_boundary(self):
        """A failed prerequisite cannot start device admission, and success preserves the selector."""
        directory=self.prepare();support=configured_tests.support;identity=support.core();order=[]
        identity.collect_metadata=lambda *args:order.append('metadata') or {'synthetic':True}
        gate=SimpleNamespace(before_compute=lambda *args:order.append('common'))
        def compute(*args):
            """Observe the existing numerical boundary without importing numerical packages."""
            order.append('compute');self.assertEqual(os.environ['CUDA_VISIBLE_DEVICES'],'0')
        with patch.object(self.selected,'__file__',str(directory/'code/measure_selected.py')),patch.object(self.selected,'verify_prepared'),\
                patch.object(self.selected,'common_confirmation_support',return_value=gate),\
                patch.object(self.selected,'card_admission',side_effect=lambda *args:order.append('card') or self.selected.nullcontext()),\
                patch.object(support,'core',return_value=identity),patch.object(support,'verify_qualification_inputs'),\
                patch.object(self.selected.training,'verify_assets',return_value={}),patch.object(self.selected.training,'jaxtari_python_receipt',return_value={}),\
                patch.object(self.selected.training,'installed_gpu_library_receipts',return_value={}),patch.object(self.selected,'installed_backend_receipt',return_value={}),\
                patch.object(self.selected,'_compute',side_effect=compute),patch.dict(os.environ,{'CUDA_VISIBLE_DEVICES':'0'},clear=True):
            self.selected.run_prepared(directory)
        self.assertEqual(order,['common','card','metadata','compute'])


if __name__=='__main__':
    unittest.main()
