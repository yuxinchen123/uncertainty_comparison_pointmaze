"""Check only the exact CPU release control; all live action calls are mocked."""
import importlib.util,json,subprocess,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
P=Path(__file__).absolute().parent;spec=importlib.util.spec_from_file_location('cpu_release',P/'release_passive_observer_6781903.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
RAW=json.loads((P/'held_monitor_review.json').read_text())['query']['stdout']
def intent():return {'published_release_commit':'a'*40,'release_review_sha256':'b'*64,'held_monitor_review_sha256':'c'*64,'release_source_sha256':'d'*64}
class Tests(unittest.TestCase):
 def test_actual_held_record_and_bad_owner_request_restart_or_duplicate(self):
  m.validate_held(RAW)
  for old,new in [('JobId=6781903','JobId=6781231'),('sl5nw(2618919)','other(2618919)'),('Requeue=0','Requeue=1'),('Restarts=0','Restarts=1'),('JobState=PENDING','JobState=RUNNING'),('NumCPUs=1','NumCPUs=2'),('Reservation=sl5nw_168','Reservation=sl5nw_167'),('ReqNodeList=jaguar03','ReqNodeList=other'),('TimeLimit=00:05:00','TimeLimit=00:06:00')]:
   with self.subTest(field=old),self.assertRaises(ValueError):m.validate_held(RAW.replace(old,new))
  with self.assertRaises(ValueError):m.validate_held(RAW+' JobId=6781903')
  with self.assertRaises(ValueError):m.validate_held(RAW+' TresPerNode=gres/gpu:0')
  with self.assertRaises(ValueError):m.validate_held(RAW.replace('cpu=1,mem=2G,node=1','cpu=1,mem=2G,node=1,gres/gpu=0'))
 def test_query_failure_is_saved_before_parse_and_does_not_release(self):
  with tempfile.TemporaryDirectory() as tmp,patch.object(m,'PACKET',Path(tmp)):
   raw=RAW.replace(str(P/'job.sbatch'),str(Path(tmp)/'job.sbatch')).replace('Requeue=0','Requeue=1').encode()
   with patch.object(m.subprocess,'run',return_value=subprocess.CompletedProcess([],0,raw,b'')) as process:
    with self.assertRaises(ValueError):m.release_once(intent())
   self.assertEqual(process.call_count,1);self.assertEqual((Path(tmp)/'release_owned_job_before.raw').read_bytes(),raw);self.assertTrue((Path(tmp)/'release_intent.json').exists());self.assertFalse((Path(tmp)/'release_receipt.json').exists())
 def test_one_query_then_one_release_and_no_old_job_action(self):
  with tempfile.TemporaryDirectory() as tmp,patch.object(m,'PACKET',Path(tmp)):
   raw=RAW.replace(str(P/'job.sbatch'),str(Path(tmp)/'job.sbatch')).encode();seen=[]
   def fake(command,**kwargs):
    self.assertTrue((Path(tmp)/'release_intent.json').exists());seen.append(command)
    return subprocess.CompletedProcess(command,0,raw if len(seen)==1 else b'',b'')
   with patch.object(m.subprocess,'run',side_effect=fake):r=m.release_once(intent())
   self.assertEqual(seen,[['scontrol','show','job','-o','6781903'],['scontrol','release','6781903']]);self.assertFalse(r['admission_granted']);self.assertTrue((Path(tmp)/'release_return.json').exists());self.assertTrue((Path(tmp)/'release_receipt.json').exists())
 def test_timeout_retains_intent_and_raw_return_without_retry(self):
  with tempfile.TemporaryDirectory() as tmp,patch.object(m,'PACKET',Path(tmp)),patch.object(m.subprocess,'run',side_effect=subprocess.TimeoutExpired('synthetic',1,output=b'partial')) as process:
   with self.assertRaises(subprocess.TimeoutExpired):m.release_once(intent())
   self.assertEqual(process.call_count,1);self.assertEqual((Path(tmp)/'release_preflight_stdout.raw').read_bytes(),b'partial');self.assertIsNone(json.loads((Path(tmp)/'release_preflight_return.json').read_text())['returncode'])
 def test_actual_default_metadata_only_and_publication_required(self):
  with patch.object(m.socket,'gethostname',return_value='portal05'):value=m.checked()
  self.assertEqual(value['job_id'],'6781903');self.assertIsNone(value['published_release_commit']);self.assertFalse((P/'release_intent.json').exists())
  with patch.object(m.sys,'argv',['release','--release']),patch.object(m,'checked') as check,self.assertRaises(ValueError):m.main()
  check.assert_not_called()
 def test_changed_unit_binding_or_foreign_monitor_proof_is_refused_without_writing(self):
  original=m.contents
  unit=m.UNIT/'passive_held_identifier_binding.json'
  def changed_binding(path):
   raw=original(path)
   return raw+b' ' if path==unit else raw
  with patch.object(m.socket,'gethostname',return_value='portal05'),patch.object(m,'contents',side_effect=changed_binding),self.assertRaises(ValueError):m.checked()
  proof=P/'held_monitor_review.json'
  def changed_proof(path):
   raw=original(path)
   if path==proof:
    value=json.loads(raw);value['monitor_job_id']='6781087';return m.encoded(value)
   return raw
  with patch.object(m.socket,'gethostname',return_value='portal05'),patch.object(m,'contents',side_effect=changed_proof),self.assertRaises(ValueError):m.checked()
  self.assertFalse((P/'release_intent.json').exists())
if __name__=='__main__':unittest.main(verbosity=2)
