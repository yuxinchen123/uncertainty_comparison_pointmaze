"""Exercise real publication refusal and exact mocked objects without any action."""
import contextlib,copy,io,json,subprocess,unittest
from pathlib import Path
from unittest.mock import patch
import test_submit_selected_profiles_held as existing
S=existing.submit
HERE=Path(__file__).resolve().parent
class PublicationTests(unittest.TestCase):
 def setUp(self):
  self.inputs=json.loads((HERE/'submission_inputs.json').read_bytes())
 def test_exact_mocked_publication_and_changed_object_refusal(self):
  paths=[row['path'] for row in self.inputs['required_files']]+[str(HERE/name) for name in ('submission_inputs.json','submit_selected_profiles_held.py','test_submit_selected_profiles_held.py','author_review.json')]
  objects={str(Path(p).relative_to(self.inputs['repository'])):Path(p).read_bytes() for p in paths}
  calls=[]
  def object_bytes(command,*args,**kwargs):
   calls.append(command);return objects[command[-1].split(':',1)[1]]
  with patch.object(S.subprocess,'run') as ancestry,patch.object(S.subprocess,'check_output',side_effect=object_bytes):
   S.validate_publication(self.inputs,'a'*40)
  self.assertEqual(len(calls),len(paths));ancestry.assert_called_once()
  changed=next(iter(objects));objects[changed]+=b'\n'
  with patch.object(S.subprocess,'run'),patch.object(S.subprocess,'check_output',side_effect=object_bytes):
   with self.assertRaises(ValueError):S.validate_publication(self.inputs,'a'*40)
 def test_actual_helper_commit_refuses_absent_new_publication_objects(self):
  original=S.subprocess.check_output;observed=[]
  def capture(command,*args,**kwargs):
   try:return original(command,*args,stderr=subprocess.PIPE,**kwargs)
   except subprocess.CalledProcessError as error:
    observed.append({'command':command,'returncode':error.returncode,'stdout':(error.output or b'').decode(errors='strict'),
     'stderr':(error.stderr or b'').decode(errors='strict')});raise
  with patch.object(S.subprocess,'check_output',side_effect=capture):
   with self.assertRaises(subprocess.CalledProcessError):
    S.validate_publication(self.inputs,'fdb35f886ad01c6d4ed387dc55b172d0221a9853')
  self.assertEqual(len(observed),1);self.assertEqual(observed[0]['returncode'],128)
  self.assertIn("exists on disk, but not in",observed[0]['stderr'])
  output=HERE/('actual_old_publication_refusal_'+str(__import__('os').environ['SLURM_STEP_ID'])+'.json')
  with output.open('x') as stream:json.dump({'expected_refusal':True,'actual_read_only_git_return':observed[0]},stream,indent=2);stream.write('\n')
 def test_both_actual_defaults_do_not_create_an_attempt(self):
  for copies in (1,2):
   unit=Path(self.inputs['packets'][str(copies)]['unit']);packet=Path(self.inputs['packets'][str(copies)]['packet'])
   with contextlib.redirect_stdout(io.StringIO()):S.main(['--copies',str(copies)])
   self.assertFalse((packet/'submission_intent.json').exists())
   self.assertFalse((unit/'configured_isolation_pre_release_inputs.json').exists())

if __name__=='__main__':unittest.main()
