"""Synthetic release control checks; no SSH or scheduler command is executed."""
import base64
import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

SOURCE = Path(__file__).absolute().with_name('release_aces_full_site_2275518.py')
spec = importlib.util.spec_from_file_location('release_2275518_under_test', SOURCE)
m = importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def actual_shape_fixture():
    """Derive only the ID and packet path from an older real held request; no new allocation claim."""
    old = next((m.RUN/'remote_setup').glob('2026-11-01-02-22_aces-full-site*'))
    raw = (old/'release_owned_job_before.raw').read_text()
    old_receipt = json.loads((old/'submission_receipt.json').read_text())
    native = json.loads((m.WRAPPER/'aces_full_site_held_submission_plan.json').read_text())['native_packet']
    return raw.replace('2274632', m.JOB).replace(old_receipt['held_command'][-1], native+'/job.sbatch'), Path(native)


def fixture():
    raw,packet=actual_shape_fixture()
    value={'published_release_commit':'a'*40,'held_monitor_verification_sha256':'b'*64,
        'release_source_sha256':'c'*64,'packet':str(packet)}
    released={'schema_version':1,'kind':'owned_aces_full_site_qualification_release','cluster':'aces','job_id':m.JOB,
        'command':['scontrol','release',m.JOB],'returncode':0,'stdout':'','stderr':'','released_at_utc':'2026-11-01T16:35:00+00:00',
        'submission_receipt_sha256':m.SUBMISSION_SHA,'published_release_commit':value['published_release_commit'],
        'held_monitor_verification_sha256':value['held_monitor_verification_sha256'],'persistent_monitor_job_id':m.MONITOR_JOB,
        'persistent_monitor_source_commit':m.MONITOR_SOURCE,'owned_job_detail_before_release':raw,
        'release_source_sha256':value['release_source_sha256'],'automatic_retry_permitted':False,**m.BOUNDARY}
    blobs={'release_receipt.json':m.encoded(released),'release_intent.json':m.encoded(value),
        'release_owned_job_before.raw':raw.encode(),'release_owned_job_before_stderr.raw':b'',
        'release_preflight_return.json':m.encoded({'command':['scontrol','show','job','-o',m.JOB],'returncode':0,'automatic_retry_permitted':False}),
        'release_stdout.raw':b'','release_stderr.raw':b'',
        'release_return.json':m.encoded({'command':['scontrol','release',m.JOB],'returncode':0,'automatic_retry_permitted':False})}
    return value,blobs


def response(blobs):
    return m.encoded({'files':{k:base64.b64encode(v).decode() for k,v in blobs.items()}})


class ReleaseTests(unittest.TestCase):
    def test_shared_native_program_compiles_and_uses_same_parsers(self):
        ns={'__name__':'metadata_only_native_test'};exec(compile(m.remote_program(),'<native-release>','exec'),ns)
        raw,packet=actual_shape_fixture();m.validate_held_detail(raw,packet);ns['validate_held_detail'](raw,packet)

    def test_actual_old_shape_derived_fixture_accepts_only_exact_new_job_and_packet(self):
        raw,packet=actual_shape_fixture();m.validate_held_detail(raw,packet)
        for old,new in [('JobId=2275518','JobId=2274632'),('u.sl359075(56626)','u.other(56626)'),('Account=158590754315','Account=other'),('Partition=gpu','Partition=debug'),('QOS=normal','QOS=other'),('JobState=PENDING','JobState=RUNNING'),('Reason=JobHeldUser','Reason=Resources'),('CPUs/Task=8','CPUs/Task=4'),('MinMemoryNode=32G','MinMemoryNode=64G'),('TimeLimit=03:00:00','TimeLimit=03:01:00'),('NumCPUs=8','NumCPUs=16'),('NumTasks=1','NumTasks=2'),('gres/gpu:h100=1','gres/gpu:a100=1'),('TresPerNode=gres/gpu:h100:1','TresPerNode=gres/gpu:h100:2')]:
            with self.subTest(field=old),self.assertRaises(ValueError):m.validate_held_detail(raw.replace(old,new),packet)

    def test_duplicate_and_additional_gpu_resources_reject(self):
        raw,packet=actual_shape_fixture()
        for changed in [raw+' JobId='+m.JOB,raw.replace('cpu=8,mem=32G','cpu=8,cpu=8,mem=32G'),raw.replace('gres/gpu:h100=1','gres/gpu:h100=1,gres/gpu:a100=0')]:
            with self.assertRaises(ValueError):m.validate_held_detail(changed,packet)

    def test_native_typed_false_boundary_is_required_before_any_observation(self):
        ns={'__name__':'metadata_only_native_test'};exec(m.remote_program(),ns)
        with self.assertRaises(ValueError):ns['native_main']({'schema_version':True})
        with self.assertRaises(ValueError):ns['native_main']({'schema_version':1,'kind':'owned_aces_full_site_release_intent','automatic_retry_permitted':False,**{**m.BOUNDARY,'workers_permitted':0}})

    def test_complete_native_response_is_accepted(self):
        value,blobs=fixture();self.assertEqual(m.accept_response(response(blobs),value),blobs)

    def test_changed_identity_or_boolean_schema_never_accepts(self):
        value,blobs=fixture()
        for key,new in [('schema_version',True),('returncode',False),('job_id','2274632'),('persistent_monitor_job_id','6780635'),('published_release_commit','f'*40),('workers_permitted',True),('idea_credit',False)]:
            altered=copy.deepcopy(blobs);row=json.loads(altered['release_receipt.json']);row[key]=new;altered['release_receipt.json']=m.encoded(row)
            with self.subTest(key=key),self.assertRaises(ValueError):m.accept_response(response(altered),value)

    def test_changed_raw_query_and_response_roster_reject(self):
        value,blobs=fixture()
        for key in ['release_owned_job_before.raw','release_owned_job_before_stderr.raw','release_stdout.raw']:
            altered=dict(blobs);altered[key]=b'changed'
            with self.subTest(key=key),self.assertRaises(ValueError):m.accept_response(response(altered),value)
        altered=dict(blobs);altered.pop('release_return.json')
        with self.assertRaises(ValueError):m.accept_response(response(altered),value)

    def test_exact_query_and_release_commands_are_bound(self):
        value,blobs=fixture()
        for key in ['release_preflight_return.json','release_return.json']:
            altered=dict(blobs);row=json.loads(blobs[key]);row['command']=['scontrol','release','2274632'];altered[key]=m.encoded(row)
            with self.assertRaises(ValueError):m.accept_response(response(altered),value)

    def test_intent_saved_before_mocked_stream_and_native_receipt_last(self):
        value,blobs=fixture()
        with tempfile.TemporaryDirectory() as directory:
            packet=Path(directory);native={'python':'/selected/private/python'};writes=[]
            original=m.write
            def track(path,raw):writes.append(path.name);return original(path,raw)
            def fake(command,**kwargs):
                self.assertTrue((packet/'release_intent.json').exists());self.assertFalse((packet/'release_receipt.json').exists())
                self.assertIn('ControlMaster=no',command);self.assertIn('ProxyCommand=false',command);self.assertIn('BatchMode=yes',command);self.assertNotIn('-O',command)
                self.assertIn('/selected/private/python',command[-1]);self.assertEqual(json.loads(kwargs['input']),value)
                kwargs['stdout'].write(response(blobs));return subprocess.CompletedProcess(command,0)
            with patch.object(m,'write',side_effect=track),patch.object(m.subprocess,'run',side_effect=fake):
                self.assertTrue(m.execute_once(packet,native,value)['released'])
            self.assertEqual(writes[0],'release_intent.json');self.assertEqual(writes[-1],'release_receipt.json')
            self.assertEqual((packet/'remote_release_intent.json').read_bytes(),blobs['release_intent.json'])

    def test_transport_timeout_preserves_intent_and_failure_without_retry(self):
        value,blobs=fixture()
        with tempfile.TemporaryDirectory() as directory:
            packet=Path(directory)
            with patch.object(m.subprocess,'run',side_effect=subprocess.TimeoutExpired('synthetic',1)) as command:
                with self.assertRaises(subprocess.TimeoutExpired):m.execute_once(packet,{'python':'/selected/private/python'},value)
            self.assertEqual(command.call_count,1);self.assertTrue((packet/'release_intent.json').exists());self.assertFalse((packet/'release_receipt.json').exists())
            self.assertIsNone(json.loads((packet/'release_transport_result.json').read_text())['returncode'])

    def test_existing_intent_prevents_any_stream(self):
        value,blobs=fixture()
        with tempfile.TemporaryDirectory() as directory:
            packet=Path(directory);(packet/'release_intent.json').write_bytes(b'old')
            with patch.object(m.subprocess,'run') as command,self.assertRaises(ValueError):m.execute_once(packet,{'python':'/selected/private/python'},value)
            command.assert_not_called();self.assertEqual((packet/'release_intent.json').read_bytes(),b'old')

    def test_default_actual_metadata_validation_creates_no_intent(self):
        with patch.object(m.socket,'gethostname',return_value='portal05'):
            packet,native,value=m.checked_payload()
        self.assertEqual(value['job_id'],m.JOB);self.assertIsNone(value['published_release_commit'])
        self.assertEqual(value['held_monitor_verification_sha256'],m.digest((packet/'held_monitor_verification.json').read_bytes()))
        self.assertFalse((packet/'release_intent.json').exists());self.assertEqual(len(value['native_hashes']),62)

    def test_release_cli_requires_publication_before_local_validation(self):
        with patch.object(m.sys,'argv',['release','--release']),patch.object(m,'checked_payload') as checked,self.assertRaises(ValueError):m.main()
        checked.assert_not_called()


if __name__=='__main__':unittest.main(verbosity=2)
