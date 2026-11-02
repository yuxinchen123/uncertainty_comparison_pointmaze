"""Independently review source and synthetic Nsight process controls; never run Nsight or a GPU."""
import ast,copy,ctypes,datetime,hashlib,importlib.abc,io,json,os,stat,subprocess,sys,unittest
from pathlib import Path
from unittest.mock import patch
W=Path("/p/rlprojects/RND_worktrees/2026-11-01-22-45_profile-nsight-owned-process-observation_actual-child-source-and-slurm-step-before-numerical-work");S=Path("/p/rlprojects/RND_worktrees/2026-11-01-22-45_profile-nsight-owned-process-observation_actual-child-source-and-slurm-step-before-numerical-work/10_jax_exploration_platform/scripts/atari_pixel_campaign");R=Path("/p/rlprojects/RND_worktrees/merge-atari-cleanrl/10_jax_exploration_platform/benchmark_runs/2026-10-31-00-11_jaxatari_montezuma_pixels_cleanrl_ppo_rnd_environments-per-copy-128_rollout-steps-128_ideas-30-plus-20_steps-per-copy-2000000000_seed-start-1")
assert os.environ.get("SLURM_JOB_ID")=="6783003" and os.environ.get("SLURM_STEP_ID","").isdigit()
assert os.getresuid()==(2618919,2618919,2618919)
sha=lambda path:hashlib.sha256(Path(path).read_bytes()).hexdigest()
freeze=W/".review_metadata/nsight_process_tool_precheck_source_freeze.json"
assert sha(freeze)=="4ae2b838be50d725ce827a2ad50edbd7e60618bb0d0a1473748ed676b9ad1bda"
frozen=json.loads(freeze.read_bytes());before={name:sha(W/name) for name in frozen["files"]}
assert all(before[name]==value["sha256"] and (W/name).stat().st_size==value["bytes"]
 and oct(stat.S_IMODE((W/name).stat().st_mode))==value["mode"] for name,value in frozen["files"].items())
review=json.loads((W/".review_metadata/nsight_process_fix_author_review.json").read_bytes())
failed=review["failed_profile_files_preserved"];assert len(failed)==194
assert all(sha(path)==value["sha256"] for path,value in failed.items())
numerical=json.loads((W/".review_metadata/numerical_source_protection_thirteen_files.json").read_bytes())["files"]
assert len(numerical)==13 and all(sha(path)==value["sha256"] for path,value in numerical.items())
base="10604537c43da99c7f9ad48757491196ec222504"
old=subprocess.check_output(["git","-C",str(W),"show",base+":10_jax_exploration_platform/scripts/atari_pixel_campaign/profile_selected.py"])
function=lambda raw,name:ast.dump(next(n for n in ast.parse(raw).body if isinstance(n,ast.FunctionDef) and n.name==name),include_attributes=False)
assert function(old,"_capture_core")==function((S/"profile_selected.py").read_bytes(),"_capture_core")
blocked=[];children=[];gits=[]
class NoNumerical(importlib.abc.MetaPathFinder):
 def find_spec(self,name,path=None,target=None):
  if name.split(".")[0] in {"jax","jaxlib","numpy","torch","flax","optax","cupy","scipy","jaxtari","exploration_platform","pynvml"}:
   blocked.append(name);raise RuntimeError("numerical import forbidden in independent process metadata review")
sys.meta_path.insert(0,NoNumerical())
ctypes.CDLL=lambda *a,**k:(_ for _ in ()).throw(RuntimeError("shared-library loading forbidden"))
actual_popen=subprocess.Popen
cold=next(n for n in ast.walk(ast.parse((S/"tests/test_profile_owned_processes.py").read_bytes()))
 if isinstance(n,ast.FunctionDef) and n.name=="test_cold_new_helper_union_imports_without_site_or_numerical_modules")
cold_code=ast.literal_eval(next(n.value for n in ast.walk(cold) if isinstance(n,ast.Assign)
 and any(isinstance(t,ast.Name) and t.id=="code" for t in n.targets)))
configured_tree=ast.parse((S/"tests/test_configured_measurement.py").read_bytes())
configured_probe=ast.literal_eval(next(n.value for n in configured_tree.body if isinstance(n,ast.Assign)
 and any(isinstance(t,ast.Name) and t.id=="ISOLATED_IMPORT_PROBE" for t in n.targets)))
def permitted_popen(argv,*args,**kwargs):
 command=list(argv)
 if command[0]=="git":
  assert command[1]=="-C" and Path(command[2]).is_absolute() and str(command[2]).startswith("/tmp/") and Path(command[2]).name=="source"
  assert not any(word in command for word in ("push","fetch","pull","clone","remote","ls-remote","submodule"))
  gits.append(command)
 elif command[:5]==[sys.executable,"-I","-S","-B","-c"]:
  assert len(command)==7 and command[5]==cold_code
  assert str(command[6]).startswith("/tmp/") and Path(command[6]).is_dir();children.append(command)
 else:
  assert len(command)==7 and command[:4]==[sys.executable,"-I","-B","-c"] and command[4]==configured_probe
  assert str(command[5]).startswith("/tmp/") and Path(command[5]).is_dir();children.append(command)
 return actual_popen(argv,*args,**kwargs)
subprocess.Popen=permitted_popen
def audit(event,args):
 if event in {"socket.connect","socket.bind","socket.getaddrinfo","os.system","os.kill","os.killpg","ctypes.dlopen","ctypes.dlsym"}:
  blocked.append(event);raise RuntimeError("network, signal or shared-library action forbidden")
 if event=="open" and isinstance(args[0],(str,bytes)):
  path=os.fsdecode(args[0])
  if path.startswith(("/dev/nvidia","/dev/dri")) or path=="/tmp/sl5nw_serval02_card1.lock":
   blocked.append(path);raise RuntimeError("GPU device or standing lock access forbidden")
sys.addaudithook(audit)
sys.path.insert(0,str(S/"tests"))
names=("test_nsight_process_identity","test_profile_owned_processes","test_nsight_process_diagnostic",
 "test_nsight_diagnostic_preparation","test_profile_selected","test_profile_common_confirmation_helpers","test_configured_measurement")
modules=[__import__(name) for name in names]
identity_tests,profile_tests=modules[0],modules[1]
M=identity_tests.M
class IndependentControls(unittest.TestCase):
 def test_fsync_failure_retains_partial_without_publishing_final(self):
  import tempfile
  with tempfile.TemporaryDirectory() as temporary:
   path=Path(temporary)/"record.json"
   with patch.object(M.os,"fsync",side_effect=OSError("synthetic fsync failure")),self.assertRaises(OSError):
    M.publish(path,{"observed":True})
   self.assertFalse(path.exists());self.assertTrue(path.with_name(".record.json.pending").exists())
 def test_existing_final_is_not_replaced_by_new_publication(self):
  import tempfile
  with tempfile.TemporaryDirectory() as temporary:
   path=Path(temporary)/"record.json";path.write_bytes(b"original")
   with self.assertRaises(ValueError):M.publish(path,{"different":True})
   self.assertEqual(path.read_bytes(),b"original")
 def test_saved_nsight_wait_refuses_boolean_zero(self):
  fixture=profile_tests.SavedProcessTests();fixture.setUp();self.addCleanup(fixture.doCleanups)
  directory,plan,result=fixture.saved()
  value=M.read_json(directory/"profiler_actual_wait.json");value["returncode"]=False
  (directory/"profiler_actual_wait.json").write_text(json.dumps(value))
  with self.assertRaises(ValueError):M.validate_saved(directory,plan,result)
 def test_saved_raw_uid_change_refuses_even_when_summary_uid_is_unchanged(self):
  fixture=identity_tests.ProcessObservationTests();fixture.setUp();self.addCleanup(fixture.doCleanups)
  owner=fixture.observe();claim={"pid":103,"process_start_time":"40"}
  owner=json.loads(json.dumps(owner));owner["nodes"]["103"]["status_after"]="Uid:\t999\t999\t999\t999\n"
  with self.assertRaises(ValueError):M.validate_observation(owner,claim,fixture.capture_command,fixture.parent)
 def test_payload_and_original_tool_wait_are_distinct_records(self):
  fixture=profile_tests.SavedProcessTests();fixture.setUp();self.addCleanup(fixture.doCleanups)
  directory,plan,result=fixture.saved()
  self.assertNotEqual(result["capture_actual_wait"],result["profiler_actual_wait"])
  (directory/"profiler_actual_wait.json").unlink()
  with self.assertRaises(FileNotFoundError):M.validate_saved(directory,plan,result)
 def test_ordinary_new_profile_admission_still_refuses(self):
  with self.assertRaises(ValueError):profile_tests.P.require_reviewed_profile_process_admission()
def flattened(suite):
 for test in suite:
  if isinstance(test,unittest.TestSuite):yield from flattened(test)
  else:yield test
executions=[]
for module in modules:executions.extend(flattened(unittest.defaultTestLoader.loadTestsFromModule(module)))
executions.extend(flattened(unittest.defaultTestLoader.loadTestsFromTestCase(IndependentControls)))
unique={test.id():test for test in executions}
output=io.StringIO()
with __import__("contextlib").redirect_stdout(output),__import__("contextlib").redirect_stderr(output):
 result=unittest.TextTestRunner(stream=output,verbosity=2).run(unittest.TestSuite(unique.values()))
after={name:sha(W/name) for name in before}
assert before==after and not blocked
assert all(sha(path)==value["sha256"] for path,value in failed.items())
assert all(sha(path)==value["sha256"] for path,value in numerical.items())
prefix=R/"remote_setup"/("nsight_process_source_independent_6783003_"+os.environ["SLURM_STEP_ID"])
def put(path,raw):
 with path.open("xb") as stream:stream.write(raw);stream.flush();os.fsync(stream.fileno())
 return {"path":str(path),"sha256":sha(path)}
raw=put(prefix.with_suffix(".raw"),output.getvalue().encode())
value={"schema_version":1,"kind":"independent_nsight_owned_process_source_and_diagnostic_review",
 "approved":result.wasSuccessful(),"passed":result.wasSuccessful(),"source_freeze":{"path":str(freeze),"sha256":sha(freeze)},
 "frozen_source_before":before,"frozen_source_after":after,"discovered_test_executions":len(executions),
 "unique_test_IDs_executed":result.testsRun,"duplicate_test_IDs_removed":len(executions)-len(unique),
 "additional_independent_controls":6,"failures":len(result.failures),"errors":len(result.errors),"raw":raw,
 "failure_details":[{"test":case.id(),"traceback":message} for case,message in result.failures],
 "error_details":[{"test":case.id(),"traceback":message} for case,message in result.errors],
 "capture_core_AST_equal_published_base":base,"protected_failed_profile_files":failed,
 "protected_numerical_files":numerical,"all_protected_bytes_unchanged":True,
 "full_identity_diagnostic_formatter_and_profile_hooks_source_read":True,
 "approval_scope":"source and diagnostic preparation only; actual diagnostic packet/submission and raw review separate",
 "ordinary_new_profile_admission_enabled":False,"actual_Nsight_GPU_scheduler_or_network_actions":0,
 "allowed_stdlib_cold_import_children":children,"allowed_private_Git_fixture_commands":gits,
 "scientific_credit":0,"admission_granted":False,"job_id":"6783003","step_id":os.environ["SLURM_STEP_ID"],
 "pid":os.getpid(),"at_utc":datetime.datetime.now(datetime.timezone.utc).isoformat()}
receipt=put(prefix.with_suffix(".json"),(json.dumps(value,indent=2,sort_keys=True)+"\n").encode())
print(json.dumps({"passed":value["passed"],"unique_tests":result.testsRun,"errors":len(result.errors),
 "failures":len(result.failures),"receipt":receipt,"step":os.environ["SLURM_STEP_ID"]}))
raise SystemExit(0 if value["passed"] else 1)

