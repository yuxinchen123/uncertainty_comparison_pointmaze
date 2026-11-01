"""Check only the exact CPU release control; all live action calls are mocked."""
import importlib.util,json,subprocess,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
P=Path(__file__).absolute().parent;spec=importlib.util.spec_from_file_location('cpu_release',P/'release_monitor_6782816.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
RAW=(P/'held_scontrol_stdout.raw').read_text()
def intent():return {'published_release_commit':'a'*40,'release_review_sha256':'b'*64,'held_monitor_review_sha256':'c'*64,'release_source_sha256':'d'*64}
class Tests(unittest.TestCase):
 def test_actual_held_record_and_bad_owner_request_restart_or_duplicate(self):
  m.validate_held(RAW)
  m.validate_held(RAW.replace('NodeList= ', 'NodeList=(null) '))
  for old,new in [('JobId=6782816','JobId=6781087'),('sl5nw(2618919)','other(2618919)'),('Requeue=0','Requeue=1'),('Restarts=0','Restarts=1'),('JobState=PENDING','JobState=RUNNING'),('NumCPUs=1','NumCPUs=2'),('TresPerNode=gres/gpu:0','TresPerNode=gres/gpu:1'),('TimeLimit=7-00:00:00','TimeLimit=1-00:00:00'),('AllocTRES=(null)','AllocTRES=cpu=1'),('NodeList= ','NodeList=unexpected ')]:
   with self.subTest(field=old),self.assertRaises(ValueError):m.validate_held(RAW.replace(old,new))
  with self.assertRaises(ValueError):m.validate_held(RAW+' JobId=6782816')
 def test_query_failure_is_saved_before_parse_and_does_not_release(self):
  with tempfile.TemporaryDirectory() as tmp,patch.object(m,'PACKET',Path(tmp)):
   raw=RAW.replace(str(P/'monitor.sbatch'),str(Path(tmp)/'monitor.sbatch')).replace('Requeue=0','Requeue=1').encode()
   with patch.object(m.subprocess,'run',return_value=subprocess.CompletedProcess([],0,raw,b'')) as process:
    with self.assertRaises(ValueError):m.release_once(intent())
   self.assertEqual(process.call_count,1);self.assertEqual((Path(tmp)/'release_owned_job_before.raw').read_bytes(),raw);self.assertTrue((Path(tmp)/'release_intent.json').exists());self.assertFalse((Path(tmp)/'release_receipt.json').exists())
 def test_one_query_then_one_release_and_no_old_job_action(self):
  with tempfile.TemporaryDirectory() as tmp,patch.object(m,'PACKET',Path(tmp)):
   raw=RAW.replace(str(P/'monitor.sbatch'),str(Path(tmp)/'monitor.sbatch')).encode();seen=[]
   def fake(command,**kwargs):
    self.assertTrue((Path(tmp)/'release_intent.json').exists());seen.append(command)
    return subprocess.CompletedProcess(command,0,raw if len(seen)==1 else b'',b'')
   with patch.object(m.subprocess,'run',side_effect=fake):r=m.release_once(intent())
   self.assertEqual(seen,[['scontrol','show','job','-o','6782816'],['scontrol','release','6782816']]);self.assertFalse(r['old_monitor_stopped']);self.assertTrue((Path(tmp)/'release_return.json').exists());self.assertTrue((Path(tmp)/'release_receipt.json').exists())
 def test_timeout_retains_intent_and_raw_return_without_retry(self):
  with tempfile.TemporaryDirectory() as tmp,patch.object(m,'PACKET',Path(tmp)),patch.object(m.subprocess,'run',side_effect=subprocess.TimeoutExpired('synthetic',1,output=b'partial')) as process:
   with self.assertRaises(subprocess.TimeoutExpired):m.release_once(intent())
   self.assertEqual(process.call_count,1);self.assertEqual((Path(tmp)/'release_preflight_stdout.raw').read_bytes(),b'partial');self.assertIsNone(json.loads((Path(tmp)/'release_preflight_return.json').read_text())['returncode'])
 def test_stray_preflight_raw_records_refuse_default_without_action(self):
  original=Path.exists
  for name in ['release_preflight_stdout.raw','release_preflight_stderr.raw']:
   def exists(path):return path==P/name or original(path)
   with self.subTest(name=name),patch.object(Path,'exists',new=exists),patch.object(m.socket,'gethostname',return_value='portal05'):
    with self.assertRaisesRegex(ValueError,'prior release attempt'):m.checked()
 def test_actual_default_metadata_only_and_publication_required(self):
  with patch.object(m.socket,'gethostname',return_value='portal05'):value=m.checked()
  self.assertEqual(value['job_id'],'6782816');self.assertIsNone(value['published_release_commit']);self.assertFalse((P/'release_intent.json').exists())
  with patch.object(m.sys,'argv',['release','--release']),patch.object(m,'checked') as check,self.assertRaises(ValueError):m.main()
  check.assert_not_called()
if __name__=='__main__':unittest.main(verbosity=2)
