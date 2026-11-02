"""Test synthetic held returns and temporary ledgers only; never query or submit a real job."""
import copy
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

HERE=Path(__file__).absolute().parent
spec=importlib.util.spec_from_file_location('common_held_submit',HERE/'submit_common_confirmation_held.py');submit=importlib.util.module_from_spec(spec);spec.loader.exec_module(submit)
REAL_INPUTS=json.loads((HERE/'submission_inputs.json').read_bytes())


class HeldSubmissionTests(unittest.TestCase):
    def setUp(self):
        self.temporary=tempfile.TemporaryDirectory();self.addCleanup(self.temporary.cleanup);self.root=Path(self.temporary.name);self.campaign=self.root/'campaign';(self.campaign/'slurm/jobs').mkdir(parents=True)
        self.ledgers=[self.campaign/'slurm'/n for n in ['submitted_jobids.txt','submitted_jobids_uva.txt','submissions.tsv']]
        for p,text in zip(self.ledgers,['10\n','10\n','10\tmonitor\told\t/old/script\n']):p.write_text(text)
        self.utility=self.root/'utility';self.utility.mkdir();(self.utility/'submission_inputs.json').write_text('{}\n')
        self.selected_copy=1
        self.contexts={}
        for copies in [1,2]:
            real=REAL_INPUTS['packets'][str(copies)];actual=Path(real['unit']);name='synthetic-common-confirmation-copies-'+str(copies);packet=self.campaign/'remote_setup'/name;unit=self.campaign/'data'/name;packet.mkdir(parents=True);unit.mkdir(parents=True)
            plan=json.loads((actual/'measurement_plan.json').read_text());plan['campaign_run_dir']=str(self.campaign);plan['unit_id']=name;plan['configured_isolation']['owned_inputs_path']=str(unit/'owned_allocation_inputs.json')
            (unit/'measurement_plan.json').write_text(json.dumps(plan,indent=2)+'\n');(unit/'config_resolved.yaml').write_bytes((actual/'config_resolved.yaml').read_bytes());shutil.copytree(actual/'code',unit/'code')
            (packet/'job.sbatch').write_text('#!/bin/bash\n# synthetic only\n');(packet/'experiment_background.md').write_text('synthetic metadata only\n')
            command=['sbatch','--parsable','--hold','--no-requeue',str(packet/'job.sbatch')]
            item={'packet':str(packet),'unit':str(unit),'copies':copies,'plan':submit.receipt(unit/'measurement_plan.json'),'script':submit.receipt(packet/'job.sbatch'),'command':command}
            self.contexts[copies]=(item,plan)
        self.inputs={'campaign_run_dir':str(self.campaign),'repository':str(self.root)}
        self.addCleanup(patch.stopall);patch.object(submit,'UTILITY',self.utility).start()
        def validated(copies):
            item,plan=self.contexts[copies];submit.unattempted(Path(item['packet']),Path(item['unit']),self.campaign);return self.inputs,item,plan
        patch.object(submit,'validate_inputs',side_effect=validated).start();self.publication=patch.object(submit,'validate_publication',return_value='a'*40).start()
        patch.object(submit.socket,'gethostname',return_value='portal05').start();patch.object(submit.os,'getuid',return_value=2618919).start();patch.object(submit.os,'geteuid',return_value=2618919).start()

    def result(self,stdout=b'9910001\n',stderr=b'',returncode=0):return SimpleNamespace(stdout=stdout,stderr=stderr,returncode=returncode)
    def path(self,copies=None):
        copies=self.selected_copy if copies is None else copies
        return Path(self.contexts[copies][0]['packet'])
    def unit(self,copies=None):
        copies=self.selected_copy if copies is None else copies
        return Path(self.contexts[copies][0]['unit'])
    def invoke(self,copies=1):return submit.submit_once(copies,'a'*40)

    def test_one_exact_held_call_ledgers_and_private_snapshots_before_receipt(self):
        original=submit.publish;events=[]
        def recorded(path,value):
            if path==self.path()/'submission_receipt.json':
                self.assertTrue(all(p.read_text().splitlines()[-1]=='9910001' for p in self.ledgers[:2]));self.assertEqual(self.ledgers[2].read_text().splitlines()[-1].split('\t'),['9910001','selected_configuration_measurement',submit.SOURCE,str(self.path()/'job.sbatch')]);self.assertTrue((self.unit()/'configured_isolation_owned_job_ids.txt').is_file());self.assertTrue((self.unit()/'configured_isolation_owned_submissions.tsv').is_file());events.append('durable ownership before receipt')
            return original(path,value)
        with patch.object(submit.subprocess,'run',return_value=self.result()) as scheduler,patch.object(submit,'publish',side_effect=recorded):out=self.invoke()
        scheduler.assert_called_once_with(self.contexts[1][0]['command'],capture_output=True,timeout=90);self.assertEqual(events,['durable ownership before receipt']);self.assertEqual(out['job_id'],'9910001');self.assertTrue(out['held']);self.assertFalse(out['released']);self.assertTrue((self.path()/'held_binding.json').is_file());self.assertTrue((self.campaign/'slurm/jobs/9910001/job.sbatch').is_file());self.assertFalse((self.unit()/'owned_allocation_inputs.json').exists())
        with patch.object(submit.subprocess,'run') as scheduler:
            with self.assertRaises(ValueError):self.invoke()
            scheduler.assert_not_called()

    def test_default_no_scheduler_or_write(self):
        with patch.object(submit.subprocess,'run') as scheduler:submit.main(['--copies','1']);scheduler.assert_not_called()
        self.assertFalse((self.path()/'submission_intent.json').exists())
    def test_publication_failure_before_intent(self):
        self.publication.side_effect=ValueError('not published')
        with patch.object(submit.subprocess,'run') as scheduler:
            with self.assertRaises(ValueError):self.invoke()
            scheduler.assert_not_called()
        self.assertFalse((self.path()/'submission_intent.json').exists())
    def test_missing_publication_cli_refuses_before_attempt(self):
        with patch.object(submit.subprocess,'run') as scheduler:
            with self.assertRaises(ValueError):submit.main(['--copies','1','--submit-held'])
            scheduler.assert_not_called()
        self.assertFalse((self.path()/'submission_intent.json').exists())
    def test_wrong_host_or_uid_refuses(self):
        for host,uid in [('foreign',2618919),('portal05',999)]:
            with patch.object(submit.socket,'gethostname',return_value=host),patch.object(submit.os,'getuid',return_value=uid),patch.object(submit.subprocess,'run') as scheduler:
                with self.assertRaises(ValueError):self.invoke()
                scheduler.assert_not_called()
        self.assertFalse((self.path()/'submission_intent.json').exists())
    def test_existing_attempt_files_and_dangling_symlink_refuse(self):
        for name in submit.ATTEMPTS:
            p=self.path()/name;p.write_bytes(b'old')
            with patch.object(submit.subprocess,'run') as scheduler:
                with self.assertRaises(ValueError):self.invoke()
                scheduler.assert_not_called()
            self.assertEqual(p.read_bytes(),b'old');p.unlink()
        p=self.path()/'submission_intent.json';p.symlink_to(self.root/'missing')
        with patch.object(submit.subprocess,'run') as scheduler:
            with self.assertRaises(ValueError):self.invoke()
            scheduler.assert_not_called()
    def test_existing_private_owned_sidecar_refuses(self):
        p=self.unit()/'configured_isolation_pre_release_inputs.json';p.write_bytes(b'preserved')
        with patch.object(submit.subprocess,'run') as scheduler:
            with self.assertRaises(ValueError):self.invoke()
            scheduler.assert_not_called()
        self.assertEqual(p.read_bytes(),b'preserved')
    def test_existing_row_refuses_without_receipt(self):
        self.ledgers[2].write_text('44\tselected_configuration_measurement\told\t'+str(self.path()/'job.sbatch')+'\n')
        with patch.object(submit.subprocess,'run') as scheduler:
            with self.assertRaises(ValueError):self.invoke()
            scheduler.assert_not_called()
    def test_failed_return_keeps_raw_and_no_ledger_change(self):
        before=[p.read_bytes() for p in self.ledgers]
        with patch.object(submit.subprocess,'run',return_value=self.result(b'bad\xff\n',b'error\x00',1)) as scheduler:
            with self.assertRaises(ValueError):self.invoke()
        self.assertEqual(scheduler.call_count,1);self.assertEqual((self.path()/'submission_stdout.raw').read_bytes(),b'bad\xff\n');self.assertEqual((self.path()/'submission_stderr.raw').read_bytes(),b'error\x00');self.assertEqual(before,[p.read_bytes() for p in self.ledgers]);self.assertFalse((self.path()/'submission_receipt.json').exists())
    def test_timeout_never_retries(self):
        with patch.object(submit.subprocess,'run',side_effect=subprocess.TimeoutExpired(['sbatch'],90,output=b'partial')) as scheduler:
            with self.assertRaises(ValueError):self.invoke()
        self.assertEqual((self.path()/'submission_stdout.raw').read_bytes(),b'partial');self.assertEqual(scheduler.call_count,1)
        with patch.object(submit.subprocess,'run') as scheduler:
            with self.assertRaises(ValueError):self.invoke()
            scheduler.assert_not_called()
    def test_ambiguous_success_refuses(self):
        with patch.object(submit.subprocess,'run',return_value=self.result(b'9910001\n9910002\n')):
            with self.assertRaises(ValueError):self.invoke()
        self.assertEqual(self.ledgers[0].read_text(),'10\n')
    def test_duplicate_returned_id_refuses_without_partial_append(self):
        before=[p.read_bytes() for p in self.ledgers]
        with patch.object(submit.subprocess,'run',return_value=self.result(b'10\n')):
            with self.assertRaises(ValueError):self.invoke()
        self.assertEqual(before,[p.read_bytes() for p in self.ledgers])
    def test_partial_ledger_failure_blocks_retry_and_retains_actual_id(self):
        def partial(campaign,job,row):
            with self.ledgers[0].open('a') as f:f.write(job+'\n');f.flush();submit.os.fsync(f.fileno())
            raise OSError('synthetic partial ledger failure')
        with patch.object(submit.subprocess,'run',return_value=self.result()),patch.object(submit,'append_ledgers',side_effect=partial):
            with self.assertRaises(OSError):self.invoke()
        self.assertEqual(self.ledgers[0].read_text(),'10\n9910001\n');self.assertEqual(self.ledgers[1].read_text(),'10\n');self.assertFalse((self.path()/'submission_receipt.json').exists());self.assertEqual(json.loads((self.path()/'submission_failure.json').read_text())['actual_returned_job_id_if_validated'],'9910001')
        with patch.object(submit.subprocess,'run') as scheduler:
            with self.assertRaises(ValueError):self.invoke()
            scheduler.assert_not_called()
    def test_only_selected_copy_submitted(self):
        before={str(p):p.read_bytes() for p in self.path(1).rglob('*') if p.is_file()}
        with patch.object(submit.subprocess,'run',return_value=self.result(b'9910002\n')) as scheduler:out=self.invoke(2)
        self.assertEqual(out['copies'],2);scheduler.assert_called_once_with(self.contexts[2][0]['command'],capture_output=True,timeout=90);self.assertEqual(before,{str(p):p.read_bytes() for p in self.path(1).rglob('*') if p.is_file()});self.assertFalse((self.path(1)/'submission_intent.json').exists())
    def test_private_snapshot_failure_after_global_appends_blocks_retry(self):
        original=submit.publish_bytes
        def failed(path,raw):
            if path==self.unit()/'configured_isolation_owned_job_ids.txt':
                raise OSError('synthetic private snapshot failure after durable global appends')
            return original(path,raw)
        with patch.object(submit.subprocess,'run',return_value=self.result()),patch.object(submit,'publish_bytes',side_effect=failed):
            with self.assertRaises(OSError):self.invoke()
        self.assertTrue(all(p.read_text().splitlines()[-1]=='9910001' for p in self.ledgers[:2]))
        self.assertFalse((self.path()/'submission_receipt.json').exists())
        with patch.object(submit.subprocess,'run') as scheduler:
            with self.assertRaises(ValueError):self.invoke()
            scheduler.assert_not_called()

    def test_second_ledger_fsync_failure_preserves_partial_bytes_and_blocks_retry(self):
        original=submit.os.fsync;failed=[]
        def interrupted(fd):
            target=Path(submit.os.readlink('/proc/self/fd/'+str(fd)))
            if target==self.ledgers[1] and not failed:
                failed.append(str(target));raise OSError('synthetic second ledger fsync failure')
            return original(fd)
        with patch.object(submit.subprocess,'run',return_value=self.result()),patch.object(submit.os,'fsync',side_effect=interrupted):
            with self.assertRaises(OSError):self.invoke()
        self.assertEqual(failed,[str(self.ledgers[1])])
        self.assertEqual(self.ledgers[0].read_text(),'10\n9910001\n')
        self.assertEqual(self.ledgers[1].read_text(),'10\n9910001\n')
        self.assertEqual(self.ledgers[2].read_text(),'10\tmonitor\told\t/old/script\n')
        self.assertFalse((self.path()/'submission_receipt.json').exists())
        with patch.object(submit.subprocess,'run') as scheduler:
            with self.assertRaises(ValueError):self.invoke()
            scheduler.assert_not_called()

    def test_generated_actual_pre_release_schema_passes_unchanged_core_with_synthetic_release(self):
        copies=2;self.selected_copy=copies
        with patch.object(submit.subprocess,'run',return_value=self.result()):self.invoke(copies)
        pre=json.loads((self.unit()/'configured_isolation_pre_release_inputs.json').read_text());self.assertEqual(set(pre),{'schema_version','kind','job_id','plan','ledger','submissions','submission','submission_row'})
        release=self.unit()/'synthetic_release.json';release.write_text(json.dumps({'schema_version':1,'kind':'owned_uva_configured_isolation_release','job_id':'9910001','cluster':'uva','plan_sha256':pre['plan']['sha256'],'data_unit':str(self.unit()),'returncode':0,'command':['scontrol','release','9910001'],'submission_receipt_sha256':pre['submission']['sha256']})+'\n')
        final={**pre,'kind':'uva_configured_isolation_owned_inputs','release':submit.receipt(release)};(self.unit()/'owned_allocation_inputs.json').write_text(json.dumps(final)+'\n')
        spec=importlib.util.spec_from_file_location('synthetic_unchanged_owned_inputs',self.unit()/'code/uva_slurm_configured_isolation.py');core=importlib.util.module_from_spec(spec);spec.loader.exec_module(core)
        self.assertEqual(core.owned_inputs(self.contexts[copies][1]),final)

if __name__=='__main__':unittest.main()
