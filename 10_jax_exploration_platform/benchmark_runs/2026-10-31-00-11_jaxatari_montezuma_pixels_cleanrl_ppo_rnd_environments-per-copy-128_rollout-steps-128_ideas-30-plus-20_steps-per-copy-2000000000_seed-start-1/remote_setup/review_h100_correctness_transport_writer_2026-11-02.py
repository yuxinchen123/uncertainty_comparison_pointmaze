"""Test only transport metadata formatting with private files and mocked publication checks."""
import contextlib,ctypes,hashlib,importlib.util,io,json,os,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
P=Path("/p/rlprojects/RND_worktrees/merge-atari-cleanrl/10_jax_exploration_platform/benchmark_runs/2026-10-31-00-11_jaxatari_montezuma_pixels_cleanrl_ppo_rnd_environments-per-copy-128_rollout-steps-128_ideas-30-plus-20_steps-per-copy-2000000000_seed-start-1/remote_setup/2026-11-01-23-06_common_cache-room-backgrounds-true_render-final-two-only-true_correctness_h100-nvl_serval02-card-1_copies-1_environments-128_rollout-steps-128_epochs-4_minibatches-4_warmup-rollouts-50_seed-start-1_hours-3");R=Path("/p/rlprojects/RND_worktrees/merge-atari-cleanrl/10_jax_exploration_platform/benchmark_runs/2026-10-31-00-11_jaxatari_montezuma_pixels_cleanrl_ppo_rnd_environments-per-copy-128_rollout-steps-128_ideas-30-plus-20_steps-per-copy-2000000000_seed-start-1")
assert os.environ.get("SLURM_JOB_ID")=="6783003" and os.environ.get("SLURM_STEP_ID","").isdigit()
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
freeze=P/"launch_transport_preparation_source_freeze.json"
assert sha(freeze)=="8796752cd292a08a22674868d66a728c00bd6253c8102343ae9450661f7f0c2c"
frozen=json.loads(freeze.read_bytes())["files"];before={path:sha(path) for path in frozen};assert before==frozen
assert sha(P/"root_controller_transport_template.py.txt")=="bec5993bec977c21e8fbc12f4e0119b2971da76c44e83227cd5fbf6051f46df2"
def audit(event,args):
 if event in {"subprocess.Popen","socket.connect","socket.bind","socket.getaddrinfo","ctypes.dlopen","ctypes.dlsym","os.kill","os.killpg","os.system"}:
  raise RuntimeError("no actual process, host, library or GPU action permitted")
 if event=="open" and isinstance(args[0],(str,bytes)) and (os.fsdecode(args[0]).startswith(("/dev/nvidia","/dev/dri"))
  or os.fsdecode(args[0])=="/tmp/sl5nw_serval02_card1.lock"):
  raise RuntimeError("no GPU device or standing lock permitted")
__import__("sys").addaudithook(audit)
spec=importlib.util.spec_from_file_location("reviewed_transport_metadata_writer",P/"prepare_root_launch_transport.py")
writer=importlib.util.module_from_spec(spec);spec.loader.exec_module(writer)
actual_plan=writer.controller.document(writer.controller.UNIT/"measurement_plan.json")
class TransportMetadataTests(unittest.TestCase):
 def setUp(self):
  self.temporary=tempfile.TemporaryDirectory();self.addCleanup(self.temporary.cleanup)
  self.root=Path(self.temporary.name);self.packet=self.root/"remote_setup/unit";self.packet.mkdir(parents=True)
  self.unit=self.root/"data/unit";self.unit.mkdir(parents=True)
  for name in ("controller_source_freeze.json","packet_metadata_freeze.json","root_controller_transport_template.py.txt"):
   (self.packet/name).write_bytes((P/name).read_bytes())
  (self.unit/"measurement_plan.json").write_bytes(writer.controller.encoded(actual_plan))
  self.publication="a"*40;self.review={"path":str(self.packet/"synthetic_root_review.json"),"sha256":"b"*64}
  self.command=[actual_plan["python"],str(self.unit/"code/measure_selected.py"),"--prepared-unit",str(self.unit)]
  self.context=contextlib.ExitStack();self.addCleanup(self.context.close)
  self.context.enter_context(patch.object(writer,"P",self.packet))
  self.context.enter_context(patch.object(writer.controller,"RUN",self.root))
  self.context.enter_context(patch.object(writer.controller,"UNIT",self.unit))
  self.context.enter_context(patch.object(writer.controller,"checked_inputs",return_value=(actual_plan,self.command,{})))
  self.publication_check=self.context.enter_context(patch.object(writer.controller,"check_publication",return_value=self.review))
 def invoke(self):
  with contextlib.redirect_stdout(io.StringIO()):return writer.main(["--publication-commit",self.publication])
 def test_only_the_publication_literal_changes_and_command_is_exact(self):
  self.assertEqual(self.invoke(),0);self.publication_check.assert_called_once_with(self.publication)
  old=(P/"root_controller_transport_template.py.txt").read_text()
  self.assertEqual((self.packet/"root_start_controller_once.py").read_text(),old.replace(writer.OLD_PUBLISHED_CONTROLLER,self.publication))
  value=writer.controller.document(self.packet/"actual_launch_command_preparation.json")
  self.assertEqual(value["command"],["/usr/bin/env","-i","PATH=/usr/local/bin:/usr/bin:/bin","LANG=C.UTF-8","TZ=UTC",
   "CUDA_VISIBLE_DEVICES="+writer.controller.GPU_UUID,"PYTHONDONTWRITEBYTECODE=1","PYTHONNOUSERSITE=1",
   actual_plan["python"],"-I","-B",str(self.packet/"direct_qualification_controller.py"),"--launch","--publication-commit",self.publication])
  self.assertFalse(value["transport_executed"]);self.assertFalse(value["automatic_retry_permitted"])
 def test_changed_template_refuses_before_command_or_wrapper(self):
  (self.packet/"root_controller_transport_template.py.txt").write_bytes(b"changed")
  with self.assertRaises(ValueError):self.invoke()
  self.assertFalse((self.packet/"actual_launch_command_preparation.json").exists())
 def test_every_existing_or_dangling_command_and_wrapper_is_refused(self):
  for name in ("actual_launch_command_preparation.json","root_start_controller_once.py"):
   for suffix in ("",".prepared"):
    path=self.packet/(name+suffix);path.symlink_to(self.root/"absent")
    with self.assertRaises(ValueError):self.invoke()
    path.unlink()
 def test_publication_rejection_precedes_all_transport_outputs(self):
  self.publication_check.side_effect=ValueError("synthetic unpublished input")
  with self.assertRaises(ValueError):self.invoke()
  self.assertFalse((self.packet/"actual_launch_command_preparation.json").exists())
  self.assertFalse((self.root/"slurm").exists())
output=io.StringIO()
with contextlib.redirect_stdout(output),contextlib.redirect_stderr(output):
 result=unittest.TextTestRunner(stream=output,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(TransportMetadataTests))
after={path:sha(path) for path in frozen};assert after==before
prefix=R/"remote_setup"/("h100_correctness_transport_writer_independent_6783003_"+os.environ["SLURM_STEP_ID"])
def put(path,raw):
 with path.open("xb") as stream:stream.write(raw);stream.flush();os.fsync(stream.fileno())
 return {"path":str(path),"sha256":sha(path)}
raw=put(prefix.with_suffix(".raw"),output.getvalue().encode())
value={"kind":"independent_h100_correctness_transport_metadata_writer_review","approved":result.wasSuccessful(),
 "passed":result.wasSuccessful(),"tests":result.testsRun,"errors":len(result.errors),"failures":len(result.failures),"raw":raw,
 "error_details":[{"test":case.id(),"traceback":message} for case,message in result.errors],
 "failure_details":[{"test":case.id(),"traceback":message} for case,message in result.failures],
 "source_freeze":{"path":str(freeze),"sha256":sha(freeze)},"frozen_before":before,"frozen_after":after,
 "publication_check_mocked_only_in_private_formatter_tests":True,"only_prior_publication_literal_replaced":True,
 "prior_transport_template_sha256":writer.TEMPLATE_SHA256,"actual_writer_or_transport_executed":False,
 "actual_host_GPU_or_scheduler_actions":0,"scientific_credit":0,"admission_granted":False,
 "job_id":"6783003","step_id":os.environ["SLURM_STEP_ID"]}
receipt=put(prefix.with_suffix(".json"),(json.dumps(value,sort_keys=True,indent=2)+"\n").encode())
print(json.dumps({"passed":value["passed"],"tests":result.testsRun,"receipt":receipt,"step":os.environ["SLURM_STEP_ID"]}))
raise SystemExit(0 if value["passed"] else 1)

