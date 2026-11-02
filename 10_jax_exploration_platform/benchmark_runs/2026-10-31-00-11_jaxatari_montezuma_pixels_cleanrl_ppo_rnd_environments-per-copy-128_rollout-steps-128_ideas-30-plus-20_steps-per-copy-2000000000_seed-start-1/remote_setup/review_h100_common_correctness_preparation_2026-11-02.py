"""Review fresh H100 correctness preparation without staging history or running scientific gates."""
import ast,copy,ctypes,datetime,difflib,hashlib,importlib.abc,importlib.util,io,json,os,subprocess,sys,unittest
from pathlib import Path
R=Path("/p/rlprojects/RND_worktrees/merge-atari-cleanrl/10_jax_exploration_platform/benchmark_runs/2026-10-31-00-11_jaxatari_montezuma_pixels_cleanrl_ppo_rnd_environments-per-copy-128_rollout-steps-128_ideas-30-plus-20_steps-per-copy-2000000000_seed-start-1")
PACKETS=[Path("/p/rlprojects/RND_worktrees/merge-atari-cleanrl/10_jax_exploration_platform/benchmark_runs/2026-10-31-00-11_jaxatari_montezuma_pixels_cleanrl_ppo_rnd_environments-per-copy-128_rollout-steps-128_ideas-30-plus-20_steps-per-copy-2000000000_seed-start-1/remote_setup/2026-11-01-23-06_common_cache-room-backgrounds-true_render-final-two-only-true_correctness_h100-nvl_serval02-card-1_copies-1_environments-128_rollout-steps-128_epochs-4_minibatches-4_warmup-rollouts-50_seed-start-1_hours-3"),Path("/p/rlprojects/RND_worktrees/merge-atari-cleanrl/10_jax_exploration_platform/benchmark_runs/2026-10-31-00-11_jaxatari_montezuma_pixels_cleanrl_ppo_rnd_environments-per-copy-128_rollout-steps-128_ideas-30-plus-20_steps-per-copy-2000000000_seed-start-1/remote_setup/2026-11-01-23-06_common_cache-room-backgrounds-true_render-final-two-only-true_correctness_h100-nvl_serval02-card-1_copies-2_environments-128_rollout-steps-128_epochs-4_minibatches-4_warmup-rollouts-50_seed-start-1_hours-3")]
FREEZES=["56f8cfad4af0bc6b78398e5823d1b82376b3331ff5598f66a0bcf50965f2b03e",
 "657cbd190b0b21edd31bb077bb5195951e9af59e1242344bd35def2721a54851"]
SOFTWARE=["cfc4c6272c5bc9b1b04d4339887666d23b88dbe2312e1f5e99969f28e52be566",
 "049a81d3dba5a690750e3ed62c5a6da447fe4bd1533361bdef1032125862a217"]
assert os.environ.get("SLURM_JOB_ID")=="6783003" and os.environ.get("SLURM_STEP_ID","").isdigit()
assert os.getresuid()==(2618919,2618919,2618919)
sha=lambda path:hashlib.sha256(Path(path).read_bytes()).hexdigest()
frozen_files={};freeze_refs=[];software_refs=[]
for packet,wanted,software in zip(PACKETS,FREEZES,SOFTWARE):
 path=packet/"preparation_source_freeze.json";assert sha(path)==wanted
 freeze_refs.append({"path":str(path),"sha256":wanted});frozen_files.update(json.loads(path.read_bytes())["files"])
 path=packet/"software_verifier_source_freeze.json";assert sha(path)==software
 software_refs.append({"path":str(path),"sha256":software});frozen_files.update(json.loads(path.read_bytes())["files"])
assert all(sha(path)==wanted for path,wanted in frozen_files.items())
before={path:sha(path) for path in frozen_files}
requests=[json.loads((packet/"preparation_request.json").read_bytes()) for packet in PACKETS]
qualified=json.loads(Path(requests[0]["qualification_plan"]["path"]).read_bytes())
protected={}
for relative,wanted in qualified["helpers"]["files"].items():
 for path in (Path(qualified["helpers"]["source_root"])/relative,Path(requests[0]["qualification_plan"]["path"]).parent/"code"/Path(relative).name):
  assert sha(path)==wanted;protected[str(path)]=wanted
for key in ("core","paired_source"):
 for relative,wanted in qualified[key]["files"].items():
  wanted=wanted["sha256"] if isinstance(wanted,dict) else wanted
  path=Path(qualified[key]["source_root"])/relative;assert sha(path)==wanted;protected[str(path)]=wanted
admission=R/"SOURCE_AND_GATES_READY.json"
assert sha(admission)=="4a71ef2b245ae784933c55bb2b5159aa1bb7f8f626eb94b15390e2f9a304ee7c";protected[str(admission)]=sha(admission)
small_refs={}
for request in requests:
 for record in [request["qualification_plan"],request["qualification_consumer_verification"],
  request["qualification_terminal"],request["qualification_controller_outcome"],request["direct_backend_image_record"],
  *request["direct_qualification_inputs"].values()]:
  assert sha(record["path"])==record["sha256"];small_refs[record["path"]]=record["sha256"]
verification=json.loads(Path(requests[0]["qualification_consumer_verification"]["path"]).read_bytes())
assert verification["passed"] is True and verification["raw_checks_before_review"] is True
assert verification["unchanged_full_consumer_passed"] is True and verification["audit_step_id"]=="156"
assert verification["result"]==requests[0]["direct_qualification_inputs"]["result"]
assert verification["independent_review"]==requests[0]["direct_qualification_inputs"]["review"]
blocked=[]
class NoNumerical(importlib.abc.MetaPathFinder):
 def find_spec(self,name,path=None,target=None):
  if name.split(".")[0] in {"jax","jaxlib","numpy","torch","flax","optax","cupy","scipy","jaxtari","exploration_platform","pynvml"}:
   blocked.append(name);raise RuntimeError("numerical imports forbidden in preparation source review")
sys.meta_path.insert(0,NoNumerical())
def audit(event,args):
 if event in {"subprocess.Popen","socket.connect","socket.bind","socket.getaddrinfo","ctypes.dlopen","ctypes.dlsym","os.system","os.kill","os.killpg"}:
  blocked.append(event);raise RuntimeError("process, network or library action forbidden during source fixtures")
 if event=="open" and isinstance(args[0],(str,bytes)):
  path=os.fsdecode(args[0])
  if path.startswith(("/dev/nvidia","/dev/dri")) or path=="/tmp/sl5nw_serval02_card1.lock":
   blocked.append(path);raise RuntimeError("GPU device or standing lock forbidden")
sys.addaudithook(audit)
modules=[]
for index,packet in enumerate(PACKETS,1):
 spec=importlib.util.spec_from_file_location("reviewed_correctness_preparation_tests_"+str(index),packet/"test_prepare_common_correctness.py")
 module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);modules.append(module)
class AdditionalIndependentChecks(unittest.TestCase):
 def test_actual_accepted_original_plan_matches_both_requested_arms(self):
  for number,(module,request) in enumerate(zip(modules,requests),1):
   self.assertEqual(module.preparation.configuration_contract(request,qualified),number)
   self.assertIs(request["configuration"]["environment"]["cache_room_backgrounds"],True)
   self.assertIs(request["configuration"]["environment"]["render_final_two_only"],True)
   self.assertEqual(request["copy_seeds"],list(range(1,number+1)))
 def test_changed_qualified_python_is_refused(self):
  for module,request in zip(modules,requests):
   changed=copy.deepcopy(request);changed["python"]="/not/the/qualified/python"
   with self.assertRaises(ValueError):module.preparation.configuration_contract(changed,qualified)
 def test_changed_qualified_source_path_is_refused(self):
  for module,request in zip(modules,requests):
   changed=copy.deepcopy(request);changed["numerical_source_root"]="/not/the/qualified/source"
   with self.assertRaises(ValueError):module.preparation.configuration_contract(changed,qualified)
 def test_private_helper_path_cannot_be_substituted(self):
  for module,request in zip(modules,requests):
   changed=copy.deepcopy(request);changed["helper_source_root"]="/not/the/qualified/helpers"
   with self.assertRaises(ValueError):module.preparation.configuration_contract(changed,qualified)
 def test_actual_all_five_initial_records_and_archives_match(self):
  expected={"manifest.yaml","config_resolved.yaml","experiment_background.md","command.txt","progress.json"}
  for packet,request in zip(PACKETS,requests):
   self.assertEqual(set(request["initial_standard_data_records"]),expected)
   for name,record in request["initial_standard_data_records"].items():
    self.assertEqual(sha(record["path"]),record["sha256"])
    self.assertEqual(sha(Path(request["data_unit"])/name),record["sha256"])
   for path in (packet/"manifest.yaml",Path(request["data_unit"])/"manifest.yaml",
    packet/"initial_planned_data_records/manifest.yaml"):
    self.assertEqual(json.loads(path.read_bytes())["packet"],str(packet))
 def test_stage_is_prior_proven_source_plus_only_current_namespace_guard(self):
  old_packet=R/"remote_setup/2026-11-01-20-53_original-qualification_h100-nvl_serval02-card-1_driver-590.48.01_gpu-count-1_processor-count-8_policies-1_environments-128_rollout-steps-128_epochs-4_minibatches-4_warmup-rollouts-50_reference-updates-1_seed-1_memory-mib-95830"
  functions=lambda path:{node.name:node for node in ast.parse(path.read_bytes()).body if isinstance(node,ast.FunctionDef)}
  old=functions(old_packet/"stage_immutable_history.py")
  for packet in PACKETS:
   new=functions(packet/"stage_immutable_history.py")
   self.assertEqual(set(old),set(new))
   for name in old:
    if name!="stage":self.assertEqual(ast.dump(old[name]),ast.dump(new[name]),name)
   previous=[ast.dump(node) for node in old["stage"].body];current=[ast.dump(node) for node in new["stage"].body]
   additions=[node for node in new["stage"].body if ast.dump(node) not in previous]
   self.assertEqual(len(additions),1)
   self.assertIn("request['packet'] == str(P)",ast.unparse(additions[0]))
   self.assertEqual(previous,[node for node in current if node in previous])
 def test_serial_predecessors_and_unresolved_launch_are_explicit(self):
  self.assertEqual(requests[0]["serial_predecessor_unit"],str(Path(requests[0]["qualification_plan"]["path"]).parent))
  self.assertEqual(requests[1]["serial_predecessor_unit"],requests[0]["data_unit"])
  for packet,request in zip(PACKETS,requests):
   self.assertIs(request["serial_predecessor_actual_exit_required_before_launch"],True)
   self.assertEqual(request["continuous_wall_time_limit_seconds"],10800)
   self.assertEqual(request["expected_staged_helper_count"],26)
   self.assertFalse((Path(request["data_unit"])/"measurement_plan.json").exists())
   self.assertEqual(set(Path(request["data_unit"]).iterdir()),{Path(request["data_unit"])/name for name in request["initial_standard_data_records"]})
 def test_original_stale_packet_versions_are_preserved(self):
  for packet in PACKETS:
   archive=packet/"source_before_request_packet_path_correction"
   self.assertTrue(archive.is_dir())
   candidates=list(archive.rglob("preparation_request.json"))
   self.assertEqual(len(candidates),1)
   self.assertNotEqual(json.loads(candidates[0].read_bytes())["packet"],str(packet))
suite=unittest.TestSuite()
for module in modules:suite.addTests(unittest.defaultTestLoader.loadTestsFromModule(module))
suite.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(AdditionalIndependentChecks))
output=io.StringIO()
with __import__("contextlib").redirect_stdout(output),__import__("contextlib").redirect_stderr(output):
 result=unittest.TextTestRunner(stream=output,verbosity=2).run(suite)
after={path:sha(path) for path in frozen_files};assert after==before and not blocked
assert all(sha(path)==wanted for path,wanted in protected.items())
assert all(sha(path)==wanted for path,wanted in small_refs.items())
def put(path,raw):
 with path.open("xb") as stream:stream.write(raw);stream.flush();os.fsync(stream.fileno())
 return {"path":str(path),"sha256":sha(path)}
prefix=R/"remote_setup"/("h100_common_correctness_preparation_independent_6783003_"+os.environ["SLURM_STEP_ID"])
raw=put(prefix.with_suffix(".raw"),output.getvalue().encode())
value={"schema_version":1,"kind":"independent_direct_common_correctness_preparation_and_software_verifier_review",
 "approved":result.wasSuccessful(),"passed":result.wasSuccessful(),"preparation_source_freezes":freeze_refs,
 "software_verifier_source_freezes":software_refs,"frozen_source_before":before,"frozen_source_after":after,
 "tests":result.testsRun,"failures":len(result.failures),"errors":len(result.errors),"raw":raw,
 "failure_details":[{"test":case.id(),"traceback":message} for case,message in result.failures],
 "error_details":[{"test":case.id(),"traceback":message} for case,message in result.errors],
 "accepted_small_qualification_metadata_refs_rehashed":small_refs,"protected_source_and_admission":protected,
 "source_review_included_full_preparation_and_software_verifier":True,
 "current_packet_path_correction_preserved":True,
 "new_controller_or_actual_GPU_launch_approval":False,
 "actual_staging_or_preparation_executed":False,"actual_saved_qualification_consumer_recomputed":False,
 "actual_binary_images_or_assets_read":False,"scheduler_queries":0,"network_or_host_actions":0,"numerical_imports":0,
 "scientific_credit":0,"admission_granted":False,"job_id":"6783003","step_id":os.environ["SLURM_STEP_ID"],
 "pid":os.getpid(),"at_utc":datetime.datetime.now(datetime.timezone.utc).isoformat()}
receipt=put(prefix.with_suffix(".json"),(json.dumps(value,indent=2,sort_keys=True)+"\n").encode())
print(json.dumps({"passed":value["passed"],"tests":result.testsRun,"receipt":receipt,"step":os.environ["SLURM_STEP_ID"]}))
raise SystemExit(0 if value["passed"] else 1)

