"""Independent negative controls for the optional common confirmation prerequisite."""
import copy, hashlib, json, pathlib, tempfile, unittest
import common_confirmation as common
import test_common_confirmation as fixtures
class IndependentCommonControls(unittest.TestCase):
    """Exercise changed labels, raw joins, and ordinary exact artifact requirements."""
    def fixture(self):
        item=fixtures.CommonSavedReaderTests();item.setUp();self.addCleanup(item.doCleanups);return item
    def test_coherent_review_configuration_changes_are_rejected(self):
        item=self.fixture()
        for key in ('configuration','reference_configuration'):
            with self.subTest(key=key):
                old=copy.deepcopy(item.review[key])
                item.review[key]['learner']['n_copies']=2
                item.refresh()
                with self.assertRaises(ValueError):item.validate()
                item.review[key]=old
    def test_coherent_review_resolved_binding_change_is_rejected(self):
        item=self.fixture();item.review['resolved_binding']={'path':str(item.prior/'resolved_device_binding.json'),'sha256':'a'*64};item.refresh()
        with self.assertRaises(ValueError):item.validate()
    def test_coherent_raw_execution_identity_change_is_rejected(self):
        item=self.fixture();item.raw['execution_identity']={**item.raw['execution_identity'],'pid':13};item.refresh()
        with self.assertRaises(ValueError):item.validate()
    def test_coherent_raw_seed_and_gpu_changes_are_rejected(self):
        item=self.fixture()
        for key,value in [('copy_seeds',[2]),('gpu_uuid','other')]:
            old=item.raw[key];item.raw[key]=value;item.refresh()
            with self.subTest(key=key),self.assertRaises(ValueError):item.validate()
            item.raw[key]=old
    def test_coherent_failed_numeric_step_is_rejected(self):
        item=self.fixture()
        raw=pathlib.Path(item.raw['terminal_accounting']['path'])
        item.raw['terminal_accounting_rows']['900001.0'][3]='FAILED'
        raw.write_text(''.join('|'.join(row)+'\n' for row in item.raw['terminal_accounting_rows'].values()))
        item.raw['terminal_accounting']['sha256']=common.digest(raw.read_bytes());item.refresh()
        with self.assertRaises(ValueError):item.validate()
    def test_current_full_config_change_is_not_covered(self):
        item=self.fixture()
        item.plan['configuration']['environment']['render_final_two_only']=False
        with self.assertRaises(ValueError):item.validate()
    def test_review_alias_and_duplicate_metadata_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=pathlib.Path(tmp);target=root/'target.json';target.write_text('{}');alias=root/'alias.json';alias.symlink_to(target)
            with self.assertRaises(ValueError):common.reference(alias)
            target.write_text('{"a":1,"a":2}')
            with self.assertRaises(ValueError):common.reference(target)
    def test_final_and_timing_scope_cannot_be_added(self):
        item=fixtures.CommonConfirmationTests();item.setUp();self.addCleanup(item.doCleanups)
        for key,value in [('final_protocol',{}),('completed_study_manifest',{}),('rounds',11),('iterations',20),('minimum_updates_per_round',20)]:
            with self.subTest(key=key),self.assertRaises(ValueError):common.contract({**item.plan,key:value})
