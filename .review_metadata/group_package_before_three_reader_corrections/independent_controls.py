"""Check additional group-package metadata cases without reading any actual backend image."""
import copy,json,pathlib,unittest
import test_group_backend_package as original
B=original.backend
OUTCOMES=[]

class IndependentControls(unittest.TestCase):
 """Preserve the frozen draft and use private synthetic package fixtures only."""
 def setUp(self):
  """Reuse only the private filesystem fixture, not any scientific approval."""
  self.f=original.GroupBackendPackageTests();self.f.setUp();self.addCleanup(self.f.doCleanups)
 def refusal(self,name,action,why):
  """Record whether the actual reader rejected a declared inconsistent synthetic proof."""
  error=None
  try:action()
  except (ValueError,KeyError,TypeError) as observed:error=repr(observed)
  OUTCOMES.append({'name':name,'expected':'reject','actual':'reject' if error else 'accept','error':error,'reason':why})
  self.assertIsNotNone(error,name+' was accepted')
 def test_coherent_float_numerical_pid_is_refused(self):
  """A numerical process identity must retain an exact integer PID, even after coherent rehashing."""
  f=self.f;f.begin();final,result=f.complete();result=copy.deepcopy(result)
  result['execution_identity']['pid']=float(result['execution_identity']['pid'])
  final['numerical_execution']=result['execution_identity'];f.write(f.worker/'group_backend_capture.json',final)
  self.refusal('floating numerical execution PID',lambda:B.validate_capture(f.worker,f.plan,f.metadata,f.binding,result['loaded_gpu_libraries'],result),'A floating value is not an exact discrete process identity.')
 def test_relabelled_python_with_another_environment_package_is_refused(self):
  """The reader must reconstruct the same target-environment relationship enforced before the original pin."""
  f=self.f;other=f.root/'other-environment/bin/python';other.parent.mkdir(parents=True);other.write_bytes(b'other synthetic Python; never executed')
  expected=copy.deepcopy(f.expected);expected['selected_python']=str(other);f.plan['python']=str(other)
  f.write(f.prepared,expected);f.write(f.unit/'backend_image_expected.json',expected)
  f.plan['inputs']['backend_image_record']=B.files.reference(f.prepared)
  f.plan['group_backend_image']['expected']=B.files.reference(f.unit/'backend_image_expected.json');f.save_plan_and_binding()
  self.refusal('different selected Python environment',lambda:B.validate_prepared(f.plan),'The selected Python is in another environment while site-packages and the actual image remain in the original one.')
 def test_qualification_counters_must_agree_with_declared_copy_count(self):
  """One-copy transition totals cannot prove a declared two-copy controlled qualification update."""
  f=self.f;f.plan['mode']='qualification';f.plan['configuration']['learner']['n_copies']=2;f.plan['copy_seeds']=[[1,2]];f.save_plan_and_binding()
  result=f.result()
  self.refusal('qualification copy count versus one-copy work totals',lambda:B.completed_work(f.worker,f.plan,result),'Either qualification must explicitly require one copy, or its full work counts must be derived from the declared count.')
 def test_duplicate_worker_and_direct_protocol_are_refused(self):
  """The package helper must keep its group-only worker directory and protocol boundary."""
  f=self.f;wrong=copy.deepcopy(f.plan);wrong['direct_binding']={}
  self.refusal('direct protocol in group plan',lambda:B.group_directory(wrong),'A direct protocol is not a group package proof.')
 def test_changed_owning_record_bytes_are_refused_after_capture(self):
  """A coherently unchanged label cannot hide modified actual synthetic package bytes."""
  f=self.f;f.begin();final,result=f.complete();path=f.worker/'group_backend/METADATA';path.write_bytes(path.read_bytes()+b'changed')
  self.refusal('changed captured owner metadata',lambda:B.validate_capture(f.worker,f.plan,f.metadata,f.binding,result['loaded_gpu_libraries'],result),'The saved reader must hash the actual shared captured owner bytes.')
 def test_wrong_worker_directory_refuses_without_binary_read(self):
  """The coordinator directory cannot be substituted for its reserved member."""
  f=self.f
  self.refusal('coordinator used as worker',lambda:B.worker_index(f.unit,f.plan),'The package evidence belongs to an exact reserved worker directory.')
