"""Run only submission-control metadata tests on the already owned monitor allocation."""
import ctypes,datetime,hashlib,importlib.abc,json,os,pathlib,subprocess,sys,unittest,contextlib,io
from unittest.mock import patch
P=pathlib.Path('/p/rlprojects/RND_worktrees/merge-atari-cleanrl/10_jax_exploration_platform/benchmark_runs/2026-10-31-00-11_jaxatari_montezuma_pixels_cleanrl_ppo_rnd_environments-per-copy-128_rollout-steps-128_ideas-30-plus-20_steps-per-copy-2000000000_seed-start-1/remote_setup/2026-11-01-15-21_pixel-study-monitor_source-commit-7fd40de24201b5e1efce398f49549db26c1e94d9_processor-count-1_host-memory-gib-2_time-limit-days-7_poll-interval-seconds-1200_partition-nolim_account-sl5nw-base_gpu-count-0')
assert os.environ.get("SLURM_JOB_ID")=="6782816" and os.environ.get("SLURM_STEP_ID","").isdecimal()
freeze=json.loads((P/"held_submission_source_freeze.json").read_bytes())
images=lambda:{n:hashlib.sha256(pathlib.Path(n).read_bytes()).hexdigest() for n in freeze["files"]}
before=images();assert before==freeze["files"];blocked=[]
protected_before={e["path"]:hashlib.sha256(pathlib.Path(e["path"]).read_bytes()).hexdigest()
 for e in json.loads((P/"held_submission_inputs.json").read_bytes())["required_files"]}
assert protected_before=={e["path"]:e["sha256"] for e in json.loads((P/"held_submission_inputs.json").read_bytes())["required_files"]}
def audit(event,args):
 if event.startswith("socket.") and event!="socket.gethostname":raise RuntimeError("network access forbidden")
 if event=="open" and (str(args[0]).startswith(("/dev/nvidia","/dev/dri")) or str(args[0])=="/tmp/sl5nw_serval02_card1.lock"):raise RuntimeError("GPU/standing lock access forbidden")
sys.addaudithook(audit)
class Block(importlib.abc.MetaPathFinder):
 def find_spec(self,name,path=None,target=None):
  if name.split(".")[0] in {"jax","jaxlib","numpy","torch","flax","optax","cupy","scipy"}:blocked.append(name);raise RuntimeError("numerical import forbidden")
sys.meta_path.insert(0,Block())
ctypes.CDLL=lambda *a,**k:(_ for _ in ()).throw(RuntimeError("shared libraries forbidden"))
subprocess.Popen=lambda *a,**k:(_ for _ in ()).throw(RuntimeError("all live processes forbidden"))
sys.path.insert(0,str(P))
import test_submit_monitor_held as tests
def complete_ledger_failure(self):
 original=tests.submit.append_ledgers
 def append_then_fail(campaign,identifier,row):
  original(campaign,identifier,row)
  self.assertTrue(all(path.read_text().splitlines().count('9910002')==1 for path in self.paths[:2]))
  self.assertEqual(sum(line.split('\t')[0]=='9910002' for line in self.paths[2].read_text().splitlines()),1)
  raise OSError('private complete ledger append followed by synthetic failure')
 with patch.object(tests.submit.subprocess,'run',return_value=self.result(b'9910002\n')) as scheduler,patch.object(tests.submit,'append_ledgers',side_effect=append_then_fail):
  with self.assertRaises(OSError):tests.submit.submit_once('a'*40)
  scheduler.assert_called_once()
 self.assertEqual((self.packet/'submission_stdout.raw').read_bytes(),b'9910002\n')
 self.assertTrue((self.packet/'submission_failure.json').is_file())
 self.assertFalse((self.packet/'submission_receipt.json').exists())
 self.assertFalse((self.packet/'held_binding.json').exists())
 before=[path.read_bytes() for path in self.paths]
 with patch.object(tests.submit.subprocess,'run') as scheduler:
  with self.assertRaises(ValueError):tests.submit.submit_once('a'*40)
  scheduler.assert_not_called()
 self.assertEqual(before,[path.read_bytes() for path in self.paths])
tests.HeldMonitorSubmissionTests.test_independent_full_ledger_append_failure_refuses_all_retries=complete_ledger_failure
suite=unittest.defaultTestLoader.loadTestsFromModule(tests)
result=unittest.TextTestRunner(verbosity=2).run(suite)
module=tests.submit
inputs,plan=module.validate_inputs()
repo=pathlib.Path(inputs["repository"]);show=[]
def git_blob(command):
 assert command[:3]==["git","-C",str(repo)] and command[3]=="show"
 relative=command[4].split(":",1)[1];show.append(relative);return (repo/relative).read_bytes()
with patch.object(module.subprocess,"run") as branch,patch.object(module.subprocess,"check_output",side_effect=git_blob):
 module.validate_publication(inputs,"a"*40)
 branch.assert_called_once_with(["git","-C",str(repo),"merge-base","--is-ancestor","a"*40,"origin/merge-atari-cleanrl"],check=True)
 assert len(show)==len(inputs["required_files"])+4
 default_output=io.StringIO()
 with contextlib.redirect_stdout(default_output):module.main(['--publication-commit','a'*40])
 default_value=json.loads(default_output.getvalue());assert default_value['validated'] is True and default_value['submitted'] is False
 assert not (P/'submission_intent.json').exists()
with patch.object(module.subprocess,"run"),patch.object(module.subprocess,"check_output",return_value=b"wrong object"):
 try:module.validate_publication(inputs,"a"*40)
 except ValueError:pass
 else:raise AssertionError("changed published blob accepted")
after=images();assert after==before and not blocked
protected_after={name:hashlib.sha256(pathlib.Path(name).read_bytes()).hexdigest() for name in protected_before}
assert protected_after==protected_before
raw=pathlib.Path("/proc/self/stat").read_text();step=os.environ["SLURM_STEP_ID"]
v={"schema_version":1,"kind":"monitor_held_submission_independent_metadata_verification","at_utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),"job_id":"6782816","step_id":step,"pid":os.getpid(),"birth":raw[raw.rfind(")")+2:].split()[19],"cgroup":pathlib.Path("/proc/self/cgroup").read_text(),"tests":result.testsRun,"errors":len(result.errors),"failures":len(result.failures),"passed":result.wasSuccessful(),"failure_details":[{"test":t.id(),"traceback":s} for t,s in result.failures],"error_details":[{"test":t.id(),"traceback":s} for t,s in result.errors],"actual_packet_local_input_check":True,"actual_default_with_mocked_publication_no_action":default_value,"synthetic_publication_full_object_check":True,"changed_publication_object_refused":True,"publication_object_checks":len(show),"actual_publication_claimed":False,"source_before":before,"source_after":after,"protected_314_before":protected_before,"protected_314_after":protected_after,"numerical_import_attempts":blocked,"live_commands":0,"new_jobs":0,"release_calls":0,"scientific_credit":0}
q=pathlib.Path('/p/rlprojects/RND_worktrees/merge-atari-cleanrl/10_jax_exploration_platform/benchmark_runs/2026-10-31-00-11_jaxatari_montezuma_pixels_cleanrl_ppo_rnd_environments-per-copy-128_rollout-steps-128_ideas-30-plus-20_steps-per-copy-2000000000_seed-start-1/remote_setup/2026-11-01-15-58-36_cpu-monitor-held-submission_independent-review_gpu-count-0_processor-count-1')/("independent_6782816_"+step+".json")
with q.open("x") as o:json.dump(v,o,sort_keys=True,indent=2);o.write("\n");o.flush();os.fsync(o.fileno())
print(json.dumps({"receipt":str(q),"sha256":hashlib.sha256(q.read_bytes()).hexdigest(),"tests":result.testsRun,"passed":result.wasSuccessful()}))
sys.exit(0 if result.wasSuccessful() else 1)

