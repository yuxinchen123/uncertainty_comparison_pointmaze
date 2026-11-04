"""Review actual H100 correctness controller binding without host, scheduler or GPU action."""
import contextlib,copy,ctypes,datetime,hashlib,importlib.abc,importlib.util,io,json,os,sys,unittest,resource
from pathlib import Path
from unittest.mock import patch
P=Path("/p/rlprojects/RND_worktrees/merge-atari-cleanrl/10_jax_exploration_platform/benchmark_runs/2026-10-31-00-11_jaxatari_montezuma_pixels_cleanrl_ppo_rnd_environments-per-copy-128_rollout-steps-128_ideas-30-plus-20_steps-per-copy-2000000000_seed-start-1/remote_setup/2026-11-01-23-06_common_cache-room-backgrounds-true_render-final-two-only-true_correctness_h100-nvl_serval02-card-1_copies-2_environments-128_rollout-steps-128_epochs-4_minibatches-4_warmup-rollouts-50_seed-start-1_hours-3");R=Path("/p/rlprojects/RND_worktrees/merge-atari-cleanrl/10_jax_exploration_platform/benchmark_runs/2026-10-31-00-11_jaxatari_montezuma_pixels_cleanrl_ppo_rnd_environments-per-copy-128_rollout-steps-128_ideas-30-plus-20_steps-per-copy-2000000000_seed-start-1")
resource.setrlimit(resource.RLIMIT_AS,(512*1024*1024,512*1024*1024))
assert os.environ.get("SLURM_JOB_ID")=="6783003" and os.environ.get("SLURM_STEP_ID","").isdigit()
assert ctypes.pythonapi._name is None
stdlib_bootstrap={"scope":"stdlib own-process Python API initialized before the audit hook","module":ctypes.__file__,"python_api_target":None}
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
source=P/"controller_source_freeze.json";packet=P/"packet_metadata_freeze.json"
assert sha(source)=="a08274bb8c505f8c8069fd3dad649333b2faae02a028907346686f529d875047"
assert sha(packet)=="81596cef752f9a16bc8495073fc0026804ebb452e79f91fb50e9747d4225c240"
source_doc=json.loads(source.read_bytes());packet_doc=json.loads(packet.read_bytes())
files={**source_doc["files"],**packet_doc["files"]}
before={name:sha(name) for name in files};assert before==files
blocked=[]
class NoNumerical(importlib.abc.MetaPathFinder):
 def find_spec(self,name,path=None,target=None):
  if name.split(".")[0] in {"jax","jaxlib","numpy","torch","flax","optax","cupy","scipy","jaxtari","exploration_platform","pynvml"}:
   blocked.append(name);raise RuntimeError("numerical import forbidden in actual binding review")
sys.meta_path.insert(0,NoNumerical())
def audit(event,args):
 if event in {"subprocess.Popen","socket.connect","socket.bind","socket.getaddrinfo","ctypes.dlopen","ctypes.dlsym","os.system","os.kill","os.killpg"}:
  blocked.append(event);raise RuntimeError("process, host or shared-library action forbidden")
 if event=="open" and isinstance(args[0],(str,bytes)) and (os.fsdecode(args[0]).startswith(("/dev/nvidia","/dev/dri"))
  or os.fsdecode(args[0])=="/tmp/sl5nw_serval02_card1.lock"):
  blocked.append(str(args[0]));raise RuntimeError("GPU or standing-lock access forbidden")
sys.addaudithook(audit)
spec=importlib.util.spec_from_file_location("bound_correctness_controller_tests",P/"test_direct_qualification_controller.py")
tests=importlib.util.module_from_spec(spec);spec.loader.exec_module(tests)
controller=tests.controller
plan,command,frozen=controller.checked_inputs()
binding=controller.document(P/"controller_bound_metadata.json")
progress_path=controller.UNIT/"progress.json";progress_before=sha(progress_path)
assert len(frozen["files"])==73 and len(plan["helpers"]["files"])==26
assert binding["plan"]["sha256"]=="03ce65d2eaf2182f029bfbbc526aaa64259101044000a846a5745ba66ba15406"
assert binding["serial_predecessor_exit"]["sha256"]=="8037c9cc3df38b976e730cfde55d20237c0e1e313b308b4ef7e6e7546b478863"
assert plan["copy_seeds"]==[1,2] and plan["configuration"]["learner"]["n_copies"]==2
assert plan["helpers"]["source_commit"]=="0a880984b4a03a2ff7335337cc18576628bc9fc1"
assert command==[plan["python"],str(controller.UNIT/"code/measure_selected.py"),"--prepared-unit",str(controller.UNIT)]

# Join the independently reviewed actual predecessor without recomputing its numerical evidence.
raw_packet=R/"remote_setup/2026-11-02-00-06_h100-common-correctness_raw-review_copies-1-2_source-0a880984_helpers-26_before-seven-paired-rounds-ten-updates_gpu-count-0"
prior_review_path=raw_packet/"copies_1_terminal_review/independent_actual_review.json"
assert sha(prior_review_path)=="22f4ec55120c987fe8d44b55e6427ab75e25db61785e29d7cb6bdcbc55df8818"
prior_review=json.loads(prior_review_path.read_bytes())
serial=controller.document(P/"serial_predecessor_exit_inputs.json")
assert prior_review["approved"] is True and prior_review["passed"] is True and prior_review["independent_of_producer"] is True
assert prior_review["copies"]==1 and prior_review["copy_seeds"]==[1]
assert prior_review["plan"]==serial["plan"] and prior_review["result"]==serial["result"] and prior_review["terminal_process"]==serial["terminal"]
assert prior_review["pid"]==1577481 and prior_review["process_start_time"]=="56774003"
raw_path=Path(prior_review["raw_audit"]["path"])
assert sha(raw_path)==prior_review["raw_audit"]["sha256"]=="ade13b9c720c15a506d5eb516be74b6cc55cd76e3cd4af7003bad2ca1b8939ec"
exit_path=Path(serial["exit_observation"]["path"])
assert sha(exit_path)==serial["exit_observation"]["sha256"]=="a37b911b5c5578334487de2fae5b6a07f2e14ec44e0b92640e34494a6fca2703"
prior_metadata_before={str(path):sha(path) for path in [prior_review_path,raw_path,exit_path]}

class IndependentBoundChecks(unittest.TestCase):
 def test_literal_default_outputs_the_bound_two_copy_command_without_action(self):
  with contextlib.redirect_stdout(io.StringIO()) as output:self.assertEqual(controller.main([]),0)
  actual=json.loads(output.getvalue());self.assertFalse(actual["action_performed"])
  self.assertEqual(actual["command"],command);self.assertEqual(actual["plan"],binding["plan"])
 def test_missing_genuine_root_review_refuses_publication_before_any_process(self):
  self.assertFalse((P/"controller_root_independent_review.json").exists())
  with self.assertRaises(FileNotFoundError):controller.check_publication("a"*40)
 def test_each_required_metadata_or_helper_path_cannot_be_omitted(self):
  for selected in (str(P/"preparation_request.json"),str(controller.UNIT/"code/direct_backend_image.py"),
   str(controller.UNIT/"direct_history/inputs.json"),str(P/"initial_planned_data_records/progress.json")):
   value=copy.deepcopy(frozen);del value["files"][selected]
   with self.assertRaises(ValueError):controller.validate_packet_freeze(plan,value)
 def test_unlisted_extra_path_refuses_before_read(self):
  value=copy.deepcopy(frozen);value["files"]["/dev/nvidia0"]="0"*64
  with self.assertRaises(ValueError):controller.validate_packet_freeze(plan,value)
suite=unittest.TestSuite([unittest.defaultTestLoader.loadTestsFromTestCase(tests.BackendInputTests),
 unittest.defaultTestLoader.loadTestsFromTestCase(IndependentBoundChecks)])
output=io.StringIO()
with contextlib.redirect_stdout(output),contextlib.redirect_stderr(output):
 result=unittest.TextTestRunner(stream=output,verbosity=2).run(suite)
after={name:sha(name) for name in files};assert after==before and not blocked
assert sha(progress_path)==progress_before
assert {name:sha(name) for name in prior_metadata_before}==prior_metadata_before
controller.check_unattempted(controller.UNIT)
def put(path,raw):
 with path.open("xb") as stream:stream.write(raw);stream.flush();os.fsync(stream.fileno())
 return {"path":str(path),"sha256":sha(path)}
prefix=R/"remote_setup"/("h100_correctness_copy2_bound_independent_6783003_"+os.environ["SLURM_STEP_ID"])
raw=put(prefix.with_suffix(".raw"),output.getvalue().encode())
value={"schema_version":1,"kind":"independent_actual_h100_correctness_copy_two_bound_controller_review",
 "approved":result.wasSuccessful(),"passed":result.wasSuccessful(),"plan":binding["plan"],
 "controller_freeze":{"path":str(source),"sha256":sha(source)},"packet_freeze":{"path":str(packet),"sha256":sha(packet)},
 "actual_bound_inputs":{"path":str(P/"controller_bound_metadata.json"),"sha256":sha(P/"controller_bound_metadata.json")},
 "source_62_control_review":{"path":str(R/"remote_setup/h100_correctness_controller_source_independent_6783003_179.json"),
  "sha256":"ac8d4f0e4ac4655a67e3a1c4cc9de6a366e314509743e14c1cf1eebb9cef0020"},
 "tests":result.testsRun,"errors":len(result.errors),"failures":len(result.failures),"raw":raw,
 "failure_details":[{"test":case.id(),"traceback":message} for case,message in result.failures],
 "error_details":[{"test":case.id(),"traceback":message} for case,message in result.errors],
 "frozen_files_before":before,"frozen_files_after":after,"frozen_files":len(files),
 "formal_progress_unchanged_sha256":progress_before,"actual_default_command":command,
 "actual_default_no_action_passed":True,"actual_prior_review":{"path":str(prior_review_path),"sha256":sha(prior_review_path)},
 "actual_prior_raw_proof":prior_review["raw_audit"],"actual_prior_exit_observation":serial["exit_observation"],
 "actual_prior_acceptance_metadata_unchanged":prior_metadata_before,"stdlib_bootstrap":stdlib_bootstrap,"root_review_required_before_publication_passed":True,
 "controller_logic_approved_for_exact_copy2":result.wasSuccessful(),
 "remaining_required_action":"root writes genuine root launch review, publishes exact bytes, and passes actual publication-aware default before one reviewed transport launch",
 "actual_copy2_gpu_launch_performed":False,"actual_host_GPU_scheduler_process_actions":0,
 "large_binary_or_asset_reads":0,"scientific_credit":0,"admission_granted":False,
 "job_id":"6783003","step_id":os.environ["SLURM_STEP_ID"],"pid":os.getpid(),
 "at_utc":datetime.datetime.now(datetime.timezone.utc).isoformat()}
receipt=put(prefix.with_suffix(".json"),(json.dumps(value,indent=2,sort_keys=True)+"\n").encode())
print(json.dumps({"passed":value["passed"],"tests":result.testsRun,"receipt":receipt,"step":os.environ["SLURM_STEP_ID"]}))
raise SystemExit(0 if value["passed"] else 1)

