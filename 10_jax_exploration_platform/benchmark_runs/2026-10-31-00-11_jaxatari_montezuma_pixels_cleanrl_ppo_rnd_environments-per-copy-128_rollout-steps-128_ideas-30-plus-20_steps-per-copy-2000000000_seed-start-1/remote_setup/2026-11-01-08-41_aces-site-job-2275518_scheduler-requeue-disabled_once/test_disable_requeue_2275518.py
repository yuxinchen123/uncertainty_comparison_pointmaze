"""Synthetic control tests; no actual scheduler, SSH or GPU operation."""
import base64,copy,importlib.util,json,subprocess,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
p=Path(__file__).absolute().with_name('disable_requeue_2275518.py');spec=importlib.util.spec_from_file_location('no_requeue',p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
def fixture():
 w=json.loads((m.WRAPPER/'aces_full_site_held_submission_plan.json').read_text());packet=Path(w['native_packet']);local=Path(w['local_packet'])
 before=json.loads((local/'release_receipt.json').read_text())['owned_job_detail_before_release']
 value={'publication_commit':'a'*40,'source_sha256':'b'*64,'packet':str(packet),'private_python':'/selected/private/python'}
 receipt={'schema_version':1,'kind':'owned_aces_requeue_disabled_receipt','cluster':'aces','job_id':m.JOB,'command':m.COMMAND,'update_performed':True,'requeue_before':'1','requeue_after':'0','restart_count_before':'0','restart_count_after':'0','preflight_owner_verified':True,'submission_receipt_sha256':m.SUBMISSION_SHA,'release_receipt_sha256':m.RELEASE_SHA,'publication_commit':value['publication_commit'],'source_sha256':value['source_sha256'],'automatic_retry_permitted':False,**m.BOUNDARY}
 blobs={'intent.json':m.encoded(value),'pre_job.stdout.raw':before.encode(),'pre_job.stderr.raw':b'','pre_job.return.json':m.encoded({'command':['scontrol','show','job','-o',m.JOB],'returncode':0,'automatic_retry_permitted':False}),'update.stdout.raw':b'','update.stderr.raw':b'','update.return.json':m.encoded({'command':m.COMMAND,'returncode':0,'automatic_retry_permitted':False}),'post_job.stdout.raw':before.replace('Requeue=1','Requeue=0').encode(),'post_job.stderr.raw':b'','post_job.return.json':m.encoded({'command':['scontrol','show','job','-o',m.JOB],'returncode':0,'automatic_retry_permitted':False}),'receipt.json':m.encoded(receipt)}
 return value,blobs
def response(blobs):return m.encoded({'files':{k:base64.b64encode(v).decode() for k,v in blobs.items()}})
class Tests(unittest.TestCase):
 def test_same_native_parser_and_actual_saved_release_shape(self):
  ns={'__name__':'metadata_test'};exec(compile(m.remote_program(),'<native>','exec'),ns);v,b=fixture()
  self.assertEqual(m.validate_job(b['pre_job.stdout.raw'].decode(),Path(v['packet'])),'1')
  ns['validate_job'](b['post_job.stdout.raw'].decode(),Path(v['packet']),True)
 def test_foreign_terminal_restarted_duplicate_or_wrong_resources_reject(self):
  v,b=fixture();raw=b['pre_job.stdout.raw'].decode()
  for old,new in [('JobId=2275518','JobId=2274632'),('u.sl359075(56626)','u.other(56626)'),('Account=158590754315','Account=other'),('Restarts=0','Restarts=1'),('JobState=PENDING','JobState=COMPLETED'),('QOS=normal','QOS=other'),('cpu=8,mem=32G','cpu=8,cpu=8,mem=32G'),('gres/gpu:h100=1','gres/gpu:a100=1')]:
   with self.subTest(field=old),self.assertRaises(ValueError):m.validate_job(raw.replace(old,new),Path(v['packet']))
  with self.assertRaises(ValueError):m.validate_job(raw,Path(v['packet']),True)
 def test_complete_and_already_disabled_response(self):
  v,b=fixture();self.assertEqual(m.accept_response(response(b),v),b)
  b['pre_job.stdout.raw']=b['post_job.stdout.raw'];r=json.loads(b['receipt.json']);r.update(update_performed=False,requeue_before='0');b['receipt.json']=m.encoded(r);b['update.return.json']=m.encoded({'command':m.COMMAND,'returncode':None,'performed':False,'reason':'already Requeue=0','automatic_retry_permitted':False})
  self.assertEqual(m.accept_response(response(b),v),b)
 def test_substituted_identity_command_or_claim_refused(self):
  v,b=fixture()
  for key,new in [('job_id','2274632'),('schema_version',True),('requeue_after','1'),('workers_permitted',True),('update_performed',False),('release_receipt_sha256','c'*64)]:
   altered=dict(b);r=json.loads(b['receipt.json']);r[key]=new;altered['receipt.json']=m.encoded(r)
   with self.subTest(key=key),self.assertRaises(ValueError):m.accept_response(response(altered),v)
  altered=dict(b);q=json.loads(b['update.return.json']);q['command']=['scontrol','requeue',m.JOB];altered['update.return.json']=m.encoded(q)
  with self.assertRaises(ValueError):m.accept_response(response(altered),v)
 def test_intent_first_and_exact_update_only_in_mocked_existing_master_stream(self):
  v,b=fixture()
  with tempfile.TemporaryDirectory() as tmp,patch.object(m,'ACTION',Path(tmp)):
   def fake(command,**kwargs):
    self.assertTrue((Path(tmp)/'local_intent.json').exists());self.assertIn('ControlMaster=no',command);self.assertIn('ProxyCommand=false',command);self.assertIn('BatchMode=yes',command);self.assertNotIn('-O',command)
    kwargs['stdout'].write(response(b));return subprocess.CompletedProcess(command,0)
   with patch.object(m.subprocess,'run',side_effect=fake) as process:self.assertEqual(m.execute_once(v)['requeue_after'],'0')
   self.assertEqual(process.call_count,1);self.assertTrue((Path(tmp)/'received/receipt.json').exists())
 def test_timeout_preserves_attempt_and_does_not_retry(self):
  v,b=fixture()
  with tempfile.TemporaryDirectory() as tmp,patch.object(m,'ACTION',Path(tmp)),patch.object(m.subprocess,'run',side_effect=subprocess.TimeoutExpired('synthetic',1)) as process:
   with self.assertRaises(subprocess.TimeoutExpired):m.execute_once(v)
   self.assertEqual(process.call_count,1);self.assertTrue((Path(tmp)/'local_intent.json').exists());self.assertIsNone(json.loads((Path(tmp)/'transport.return.json').read_text())['returncode'])
 def test_actual_default_metadata_has_no_action(self):
  with patch.object(m.socket,'gethostname',return_value='portal05'):v=m.local_payload()
  self.assertEqual(v['command'],['scontrol','update','JobId=2275518','Requeue=0']);self.assertIsNone(v['publication_commit']);self.assertFalse((m.ACTION/'local_intent.json').exists())
 def test_execute_requires_publication(self):
  with patch.object(m.sys,'argv',['action','--execute']),patch.object(m,'local_payload') as prepare,self.assertRaises(ValueError):m.main()
  prepare.assert_not_called()
if __name__=='__main__':unittest.main(verbosity=2)
