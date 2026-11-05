"""Use private metadata files only; no actual campaign file or target process is changed."""
import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import stat
import tempfile
import unittest
from unittest.mock import patch

D=Path(__file__).resolve().parent
SPEC=importlib.util.spec_from_file_location('private_permission_utility',D/'metadata_permission_once.py')
M=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(M)


class PermissionTests(unittest.TestCase):
    """Test actual private chmod/fsync operations with explicitly synthetic launch proofs."""
    def setUp(self):
        self.temporary=tempfile.TemporaryDirectory();self.addCleanup(self.temporary.cleanup)
        self.run=Path(self.temporary.name)/'campaign';self.packet=self.run/'remote_setup/private_capture'
        self.controller=self.run/'data/private_controller';self.child=self.run/'data/private_capture'
        for path in (self.packet,self.controller,self.child,self.run/'queue'):path.mkdir(parents=True,exist_ok=True)
        for name in ('manifest.yaml','config_resolved.yaml','command.txt','experiment_background.md'):(self.packet/name).write_text('Synthetic private fixture\n')
        self.locks=[self.run/'queue/metadata.lock',self.run/'processes.lock']
        for path in self.locks:path.write_bytes(b'private lock fixture\n');path.chmod(0o660)
        parent={'kind':'profile_target_qualification_controller','protocol':M.PROTOCOL,
            'campaign_run_dir':str(self.run),'profile_unit':str(self.child),'synthetic_fixture':True}
        self.parent_ref=self.put(self.controller/'measurement_plan.json',parent)
        self.child_ref=self.put(self.child/'measurement_plan.json',{'kind':'full_original_profile_source_target_qualification','synthetic_fixture':True})
        self.command=['/private/not-executed/python','/private/not-executed/producer.py']
        self.put(self.controller/'prepared_child_binding.json',{'controller_plan':self.parent_ref,'child_plan':self.child_ref,'command':self.command})
        self.put(self.controller/'controller_launch_independent_review.json',{
            'kind':'independent_profile_target_qualification_controller_launch_review','approved':True,'passed':True,
            'independent_of_producer':True,'controller_plan':self.parent_ref,'child_plan':self.child_ref,'synthetic_fixture':True})
        self.publication=patch.object(M,'publication',return_value={'publication':'a'*40,'synthetic_source_approval_premise':True})
        self.publication.start();self.addCleanup(self.publication.stop)
        self.original=[M.capture(path)[2] for path in self.locks]

    def put(self,path,value):
        """Write one real private metadata file with an explicitly synthetic premise."""
        path.write_text(json.dumps(value,sort_keys=True)+'\n');return M.reference(path)

    def change(self,mode,evidence=None):
        """Call the actual permission implementation only on this private directory."""
        return M.change(self.run,self.packet,[self.controller],mode,M.CAPABILITY,'a'*40,None,evidence)

    def protected(self):
        """Verify both actual private files have only their permission bits changed."""
        for path,before in zip(self.locks,self.original):
            self.assertTrue(M.same_file(M.capture(path)[2],before,0o640))

    def evidence(self,row):
        """Store one synthetic restoration request; production evidence is separate and actual."""
        path=self.packet/'evidence.json';self.put(path,{'controllers':[{'controller_plan':self.parent_ref,**row}]});return path

    def test_preflight_refuses_0660_and_is_read_only_after_protection(self):
        argv=['--run-dir',str(self.run),'--packet',str(self.packet),'--controller-unit',str(self.controller)]
        with self.assertRaisesRegex(ValueError,'protect both'):M.main(argv)
        self.assertFalse((self.packet/'metadata_permission_protection.json').exists())
        self.change('protect');self.protected()
        with patch.object(M.os,'fchmod',side_effect=AssertionError('default changed permission')),contextlib.redirect_stdout(io.StringIO()):M.main(argv)

    def test_pending_unknown_and_timeout_never_restore(self):
        self.change('protect')
        for mode in ('pending','unknown','timeout'):
            evidence=self.evidence({'mode':mode})
            with self.assertRaisesRegex(ValueError,'pending or unknown'):self.change('restore',evidence)
            self.protected()
        self.assertFalse((self.packet/'metadata_permission_restoration_intent.json').exists())

    def test_abort_before_submission_restores_and_marks_units_unlaunchable(self):
        self.change('protect')
        evidence=self.evidence({'mode':'aborted_before_submission_or_launch','root_aborted_before_submission_or_launch':True})
        self.change('restore',evidence)
        for path,before in zip(self.locks,self.original):self.assertTrue(M.same_file(M.capture(path)[2],before,0o660))
        self.assertTrue((self.packet/'metadata_permission_no_submit_abort.json').exists())
        with self.assertRaisesRegex(ValueError,'aborted prepared unit'):M.inputs(self.run,self.packet,[self.controller])

    def test_ambiguous_submission_abort_refuses_and_retains_protection(self):
        self.change('protect');self.put(self.packet/'submission_intent.json',{'synthetic_ambiguous_attempt':True})
        evidence=self.evidence({'mode':'aborted_before_submission_or_launch','root_aborted_before_submission_or_launch':True})
        with self.assertRaisesRegex(ValueError,'ambiguous submission'):self.change('restore',evidence)
        self.protected();self.assertFalse((self.packet/'metadata_permission_restoration_intent.json').exists())

    def test_same_file_bytes_host_and_symlink_refusals(self):
        self.change('protect');evidence=self.evidence({'mode':'aborted_before_submission_or_launch','root_aborted_before_submission_or_launch':True})
        self.locks[0].write_bytes(b'changed bytes\n')
        with self.assertRaisesRegex(ValueError,'same protected files'):self.change('restore',evidence)
        self.locks[0].write_bytes(b'private lock fixture\n')
        with patch.object(M.socket,'gethostname',return_value='different-synthetic-host'):
            with self.assertRaisesRegex(ValueError,'same protected files'):self.change('restore',evidence)
        other=self.run/'unrelated';other.write_text('untouched');self.locks[0].unlink();self.locks[0].symlink_to(other)
        with self.assertRaises(ValueError):M.inputs(self.run,self.packet,[self.controller])
        self.assertEqual(other.read_text(),'untouched')

    def test_exact_child_release_restores_original_mode(self):
        self.change('protect')
        child={'pid':222,'parent_pid':111,'process_start_time':'333','uid':M.OWNER,'synthetic_fixture':True}
        owner={'pid':111,'uid':M.OWNER,'synthetic_fixture':True}
        spawn=self.put(self.controller/'controller_spawn.json',{'actual_popen_pid':222,'child':child,'controller':owner,'command':self.command})
        self.put(self.child/'profile_target_controller_release.json',{'kind':'actual_profile_target_controller_child_release',
            'parent_plan':self.parent_ref,'child_plan':self.child_ref,'controller_spawn':spawn,'child':child,'controller':owner})
        self.put(self.child/'profile_target_child_startup_observations.json',{'exact_command_ready':True,'failure':None,
            'actual_popen_pid':222,'expected_command':self.command,'observations':[{'process':child,'expected_raw_command':True,
                'predicates':{key:True for key in ('pid','birth','parent','uid','affinity','raw_hash','raw_decode')}}]})
        self.change('restore',self.evidence({'mode':'child_released'}))
        for path,before in zip(self.locks,self.original):self.assertTrue(M.same_file(M.capture(path)[2],before,0o660))

    def test_independent_no_child_terminal_route_and_no_repeat(self):
        self.change('protect')
        failure=self.put(self.controller/'controller_failure.json',{'actual_popen_pid':None,'actual_identity':None,'synthetic_fixture':True})
        raw=self.packet/'terminal.raw';raw.write_text('Synthetic terminal accounting\n')
        review=self.put(self.packet/'terminal_review.json',{'kind':'independent_no_child_qualification_controller_terminal_review',
            'approved':True,'passed':True,'independent_of_producer':True,'controller_plan':self.parent_ref,'child_plan':self.child_ref,
            'failure':failure,'controller_finished':True,'actual_child_started':False,'unresolved_anomalies':[],
            'terminal_state':'FAILED','raw_terminal_evidence':M.reference(raw),'synthetic_fixture':True})
        self.change('restore',self.evidence({'mode':'terminal_without_child','terminal_review':review}))
        with self.assertRaises(ValueError):self.change('restore',self.packet/'evidence.json')

    def test_inherited_0640_is_preserved_and_changed_controller_refuses(self):
        for path in self.locks:path.chmod(0o640)
        self.change('protect')
        evidence=self.evidence({'mode':'aborted_before_submission_or_launch','root_aborted_before_submission_or_launch':True})
        self.change('restore',evidence)
        self.assertEqual([stat.S_IMODE(path.stat().st_mode) for path in self.locks],[0o640,0o640])


if __name__=='__main__':unittest.main(verbosity=2)
