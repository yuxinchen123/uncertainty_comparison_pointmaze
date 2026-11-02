"""Display synthetic copies of the actual selected-unit ownership schema without live commands."""
import contextlib
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import monitor

HELPER='9fcc43a478b93d731df780805fd7c3c514cdf67f'
NUMERICAL='d28e6735e79788f618acb0ebfde8609db1f776af'


class SelectedUnitDisplayTests(unittest.TestCase):
    """Join only exact synthetic owned records; producer completion is never scientific approval."""

    def setUp(self):
        """Build a private prepared unit using the actual receipt/envelope field names."""
        self.temporary=tempfile.TemporaryDirectory();self.addCleanup(self.temporary.cleanup)
        self.root=Path(self.temporary.name);self.unit=self.root/'data/selected_confirmation'
        self.packet=self.root/'remote_setup'/self.unit.name
        self.unit.mkdir(parents=True);self.packet.mkdir(parents=True);(self.root/'slurm').mkdir()
        self.job='12345001'
        self.plan={'schema_version':1,'kind':'selected_configuration_measurement','mode':'confirmation',
            'source_commit':NUMERICAL,'helpers':{'source_commit':HELPER},'hardware_class':'jaguar03_rtx_a4500',
            'campaign_run_dir':str(self.root),'unit_id':self.unit.name,
            'configured_isolation':{'consumer':'selected_measurement','owned_inputs_path':str(self.unit/'owned_allocation_inputs.json')}}
        self.write(self.unit/'measurement_plan.json',self.plan)
        self.write(self.unit/'progress.json',{'kind':'prepared_intermediate_common_confirmation_progress',
            'state':'prepared_not_submitted','actual_job_id':None,'gpu_uuid':None,'submitted':False,'completed_rounds':0})
        (self.packet/'job.sbatch').write_text('#!/bin/bash\n# synthetic metadata fixture only\n')

    def write(self,path,value):
        """Write one temporary metadata object, never a live campaign artifact."""
        path.write_text(json.dumps(value,sort_keys=True,indent=2)+'\n')

    def ref(self,path):
        """Hash exact fixture bytes as the real producer's private receipts do."""
        return {'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}

    def bind(self):
        """Create a separate submission and private pre-release envelope with matching UVA ledgers."""
        self.submission_row=[self.job,self.plan['kind'],HELPER,str(self.packet/'job.sbatch')]
        self.submission={'schema_version':1,'kind':'owned_uva_configured_isolation_submission','cluster':'uva',
            'job_id':self.job,'source_commit':HELPER,'data_unit':str(self.unit),'packet':str(self.packet),
            'returncode':0,'plan_sha256':self.ref(self.unit/'measurement_plan.json')['sha256'],
            'script_sha256':self.ref(self.packet/'job.sbatch')['sha256'],'submission_row':self.submission_row,
            'held_command':['sbatch','--hold','--no-requeue','--gres=gpu:nvidia_rtx_a4500:1',str(self.packet/'job.sbatch')],
            'ledgers_and_submission_row_fsynced_before_this_receipt':True,'released':False}
        self.write(self.packet/'submission_receipt.json',self.submission)
        self.ids=self.unit/'configured_isolation_owned_job_ids.txt';self.ids.write_text(self.job+'\n')
        self.rows=self.unit/'configured_isolation_owned_submissions.tsv';self.rows.write_text('\t'.join(self.submission_row)+'\n')
        (self.root/'slurm/submitted_jobids_uva.txt').write_bytes(self.ids.read_bytes())
        (self.root/'slurm/submissions.tsv').write_bytes(self.rows.read_bytes())
        self.pre={'schema_version':1,'kind':'uva_configured_isolation_pre_release_inputs','job_id':self.job,
            'plan':self.ref(self.unit/'measurement_plan.json'),'submission':self.ref(self.packet/'submission_receipt.json'),
            'ledger':{'snapshot':self.ref(self.ids),'source_path':str(self.root/'slurm/submitted_jobids_uva.txt')},
            'submissions':{'snapshot':self.ref(self.rows),'source_path':str(self.root/'slurm/submissions.tsv')},
            'submission_row':self.submission_row}
        self.write(self.unit/'configured_isolation_pre_release_inputs.json',self.pre)

    def finish_release(self):
        """Add a successful synthetic final envelope without changing the original submission receipt."""
        self.returned={'schema_version':1,'kind':'owned_uva_configured_isolation_release','cluster':'uva',
            'job_id':self.job,'data_unit':str(self.unit),'returncode':0,'plan_sha256':self.pre['plan']['sha256'],
            'command':['scontrol','release',self.job],'submission_receipt_sha256':self.pre['submission']['sha256']}
        self.write(self.packet/'release_receipt.json',self.returned)
        self.final={**self.pre,'kind':'uva_configured_isolation_owned_inputs','release':self.ref(self.packet/'release_receipt.json')}
        self.write(self.unit/'owned_allocation_inputs.json',self.final)

    def jobs(self,state='PENDING',reason='(JobHeldUser)'):
        """Return an already queried scheduler row for this one synthetic owned identifier."""
        return [[self.job,state,'jaguar03' if state=='RUNNING' else '','gres/gpu:nvidia_rtx_a4500:1',reason]]

    def row(self,jobs=()):
        """Read only candidate metadata and ensure no subprocess is attempted."""
        with patch.object(monitor.subprocess,'run') as command:
            value=monitor.unit_rows(self.root,jobs=jobs)[0];command.assert_not_called()
        return value

    def unknown(self,jobs=()):
        """Require an explicit ownership metadata error rather than a running or scientific-failure label."""
        row=self.row(jobs);self.assertEqual(row['state'],'unknown')
        self.assertEqual(row['slurm_job_id'],'');self.assertIn('metadata error',row['progress'])
        self.assertNotIn('admission_granted',row);self.assertNotIn('case_completed',row)
        return row

    def test_actual_prepared_progress_without_receipt_stays_planned(self):
        """A null future job field never becomes execution evidence."""
        row=self.row();self.assertEqual(row['state'],'planned');self.assertIn('not submitted',row['progress'])

    def test_exact_owned_pending_user_hold_is_held_without_hardware(self):
        """The historical receipt's released=False does not itself decide scheduler state."""
        self.bind();row=self.row(self.jobs());self.assertEqual(row['state'],'held')
        self.assertEqual(row['slurm_job_id'],self.job);self.assertEqual(row['gpu'],['RTX A4500 requested'])
        self.assertEqual(row['pool'],'UVA gpu')

    def test_pending_priority_is_waiting(self):
        """A pending job without a user hold remains waiting rather than held or running."""
        self.bind();row=self.row(self.jobs(reason='(Priority)'))
        self.assertEqual(row['state'],'waiting');self.assertIn('Priority',row['progress'])

    def test_actual_running_progress_without_job_id_uses_owned_row(self):
        """Rewritten diagnostic progress still joins the immutable receipt, not a lost initial job field."""
        self.bind();self.finish_release()
        self.write(self.unit/'progress.json',{'record':'diagnostic','state':'running','pid':100,'round':3,
            'total_rounds':7,'variant':'candidate'})
        row=self.row(self.jobs('RUNNING','jaguar03'));self.assertEqual(row['state'],'running')
        self.assertEqual(row['slurm_job_id'],self.job);self.assertEqual(row['pool'],'jaguar03')
        self.assertEqual(row['progress'],'candidate 3/7 rounds')
        self.assertFalse(self.submission['released'])

    def test_missing_current_owned_row_cannot_prove_running(self):
        """Stale active progress alone cannot label an owned allocation as currently training."""
        self.bind();self.write(self.unit/'progress.json',{'state':'running','round':2,'variant':'reference','total_rounds':7})
        row=self.row();self.assertEqual(row['state'],'waiting');self.assertIn('no current scheduler observation',row['progress'])

    def test_configuring_and_completing_do_not_claim_training(self):
        """Only RUNNING, not an adjacent scheduler lifecycle state, reaches the training group."""
        self.bind()
        for state in ('CONFIGURING','COMPLETING'):
            with self.subTest(state=state):self.assertEqual(self.row(self.jobs(state,state))['state'],'waiting')

    def test_partial_submission_stays_unverified_without_job_id(self):
        """A receipt before its private envelope is present cannot establish a unit's allocation."""
        self.bind();(self.unit/'configured_isolation_pre_release_inputs.json').unlink()
        row=self.row(self.jobs('RUNNING'));self.assertEqual(row['state'],'planned');self.assertEqual(row['slurm_job_id'],'')

    def test_foreign_job_id_or_cluster_is_not_adopted(self):
        """A receipt outside this campaign's UVA ownership is an explicit metadata error."""
        self.bind();(self.root/'slurm/submitted_jobids_uva.txt').write_text('99999001\n')
        self.unknown(self.jobs())
        (self.root/'slurm/submitted_jobids_uva.txt').write_text(self.job+'\n')
        self.submission['cluster']='aces';self.write(self.packet/'submission_receipt.json',self.submission)
        self.pre['submission']=self.ref(self.packet/'submission_receipt.json');self.write(self.unit/'configured_isolation_pre_release_inputs.json',self.pre)
        self.unknown(self.jobs())

    def test_wrong_campaign_unit_or_helper_source_is_not_adopted(self):
        """The helper commit, not the D28 numerical commit, must match the submission row."""
        self.bind();self.submission['source_commit']=NUMERICAL
        self.write(self.packet/'submission_receipt.json',self.submission);self.unknown(self.jobs())
        self.submission['source_commit']=HELPER;self.write(self.packet/'submission_receipt.json',self.submission)
        self.pre['submission']=self.ref(self.packet/'submission_receipt.json')
        self.write(self.unit/'configured_isolation_pre_release_inputs.json',self.pre)
        self.plan['campaign_run_dir']=str(self.root/'other_campaign');self.write(self.unit/'measurement_plan.json',self.plan)
        self.unknown(self.jobs())

    def test_duplicate_private_ids_and_conflicting_live_rows_are_not_adopted(self):
        """Coherently rehashed snapshots still cannot hide duplicate IDs or conflicting TSV rows."""
        self.bind();self.ids.write_text(self.job+'\n'+self.job+'\n')
        self.pre['ledger']['snapshot']=self.ref(self.ids);self.write(self.unit/'configured_isolation_pre_release_inputs.json',self.pre)
        self.unknown(self.jobs())
        self.ids.write_text(self.job+'\n');self.pre['ledger']['snapshot']=self.ref(self.ids)
        self.write(self.unit/'configured_isolation_pre_release_inputs.json',self.pre)
        with (self.root/'slurm/submissions.tsv').open('a') as stream:stream.write(self.job+'\tforeign\tother\t/other/job.sbatch\n')
        self.unknown(self.jobs())

    def test_conflicting_current_scheduler_rows_are_not_adopted(self):
        """Two different observations for one ID cannot determine a selected unit's state."""
        self.bind();self.unknown(self.jobs()+self.jobs('RUNNING','jaguar03'))

    def test_final_inputs_and_release_must_preserve_the_exact_static_chain(self):
        """A static final envelope cannot substitute a different job, plan or Boolean schema."""
        self.bind();self.finish_release()
        for key,value in (('job_id','99999001'),('schema_version',True)):
            changed={**self.final,key:value};self.write(self.unit/'owned_allocation_inputs.json',changed)
            self.unknown(self.jobs('RUNNING'))
        self.write(self.unit/'owned_allocation_inputs.json',self.final)
        self.returned['returncode']=False;self.write(self.packet/'release_receipt.json',self.returned)
        self.final['release']=self.ref(self.packet/'release_receipt.json');self.write(self.unit/'owned_allocation_inputs.json',self.final)
        self.unknown(self.jobs('RUNNING'))

    def test_aliased_script_is_not_read_as_owned(self):
        """A script symlink cannot borrow the correct content hash from outside its packet."""
        self.bind();script=self.packet/'job.sbatch';outside=self.root/'outside_script';outside.write_bytes(script.read_bytes())
        script.unlink();script.symlink_to(outside);self.unknown(self.jobs())

    def test_wrong_hardware_job_id_cannot_supply_gpu_identity(self):
        """Hardware from a different allocation stays unknown instead of supplying a GPU label."""
        self.bind()
        for identifier in ('99999001',[],{'foreign':'value'}):
            self.write(self.unit/'hardware.json',{'slurm_job_id':identifier,'hostname':'other',
                'gpu_query':'GPU-synthetic, NVIDIA RTX A4500, 20470, driver'})
            row=self.row(self.jobs('RUNNING'));self.assertEqual(row['state'],'unknown');self.assertEqual(row['gpu'],[])

    def test_completed_producer_progress_is_not_promoted_to_scientific_approval(self):
        """Producer completion stays visible while actual terminal accounting and review remain separate."""
        self.bind();self.write(self.unit/'progress.json',{'state':'completed','rounds':7})
        row=self.row(self.jobs('RUNNING'));self.assertEqual(row['state'],'completed')
        self.assertEqual(row['progress'],'7 rounds complete');self.assertNotIn('case_completed',row)

    def test_direct_and_legacy_selected_paths_keep_their_existing_display(self):
        """Direct ownership and older fixed-UUID plans do not enter the new UVA-only join."""
        self.write(self.unit/'progress.json',{'state':'running','round':2,'total_rounds':7,'variant':'reference'})
        for direct in (False,True):
            plan={key:value for key,value in self.plan.items() if key!='configured_isolation'}
            if direct:plan['direct_binding']={}
            self.write(self.unit/'measurement_plan.json',plan)
            row=self.row();self.assertEqual(row['state'],'running');self.assertNotIn('selected_owned_job',row)

    def test_held_report_contains_one_named_unit_and_no_generic_duplicate(self):
        """A validated association replaces only that identifier's generic held row."""
        self.bind()
        with patch.object(monitor,'own_jobs',return_value=self.jobs()),contextlib.redirect_stdout(io.StringIO()):
            monitor.tick(self.root)
        report=(self.root/'20_mins_monitoring/latest_report.md').read_text()
        training=report.split('TRAINING\n',1)[1].split('WAITING\n',1)[0]
        held=report.split('HELD\n',1)[1].split('FINISHED\n',1)[0]
        self.assertNotIn(self.unit.name,training);self.assertEqual(held.count(self.unit.name),1)
        self.assertNotIn('held study job',held)

    def test_invalid_selected_metadata_does_not_stop_the_full_report(self):
        """One metadata error remains UNKNOWN while the generic exact owned job row stays visible."""
        self.bind();self.pre['job_id']='99999001';self.write(self.unit/'configured_isolation_pre_release_inputs.json',self.pre)
        with patch.object(monitor,'own_jobs',return_value=self.jobs()),contextlib.redirect_stdout(io.StringIO()):
            monitor.tick(self.root)
        report=(self.root/'20_mins_monitoring/latest_report.md').read_text()
        self.assertIn('UNKNOWN',report);self.assertIn('held study job',report)
        self.assertNotIn('selected_confirmation                  UVA gpu       unknown       RUNNING',report)


if __name__=='__main__':
    unittest.main()

