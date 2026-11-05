"""Independently review only private reporting metadata and unchanged source."""
import ast
import copy
import datetime
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import re
import socket
import stat
import subprocess
import sys
import unittest

W = Path("/p/rlprojects/RND_worktrees/2026-11-04-15-04_monitor-post-initial-validation-reporting_common-correctness-confirmation-kernel-profiles-and-packing")
RUN = Path("/p/rlprojects/RND_worktrees/merge-atari-cleanrl/10_jax_exploration_platform/benchmark_runs/2026-10-31-00-11_jaxatari_montezuma_pixels_cleanrl_ppo_rnd_environments-per-copy-128_rollout-steps-128_ideas-30-plus-20_steps-per-copy-2000000000_seed-start-1")
OUT = Path(__file__).resolve().parent
FREEZE_SHA = "0e4f4a967b282c4e2f5acd541fdec74ec5175410b0d46831b0d3fe5f7d9b07bd"
freeze_path = W / ".review_metadata/monitor_post_initial_reporting/source_freeze_after_independent_review_corrections.json"
assert os.environ.get("SLURM_JOB_ID") == "6783003" and os.getuid() == 2618919
step = os.environ.get("SLURM_STEP_ID", "")
assert re.fullmatch(r"[0-9]+", step)
assert re.search(r"/job_6783003/step_" + step + r"(?:/|$)", Path("/proc/self/cgroup").read_text())
assert socket.gethostname().split(".")[0] == "heartpiece"
assert "6783003" in (RUN / "slurm/submitted_jobids_uva.txt").read_text().splitlines()
assert hashlib.sha256(freeze_path.read_bytes()).hexdigest() == FREEZE_SHA
freeze = json.loads(freeze_path.read_text())
before = {name: hashlib.sha256((W/name).read_bytes()).hexdigest() for name in freeze["files"]}
assert all(before[name] == row["sha256"] for name, row in freeze["files"].items())
old = subprocess.check_output(["git", "-C", str(W), "show", freeze["base_commit"] + ":10_jax_exploration_platform/scripts/atari_pixel_campaign/controller.py"], text=True)
new = (W / "10_jax_exploration_platform/scripts/atari_pixel_campaign/controller.py").read_text()
def functions(source):
    return {node.name:node for node in ast.parse(source).body if isinstance(node, ast.FunctionDef)}
old_f, new_f = functions(old), functions(new)
unchanged = [name for name in old_f if name != "tick" and ast.dump(old_f[name], include_attributes=False)==ast.dump(new_f[name], include_attributes=False)]
assert len(unchanged)==32 and set(unchanged)==set(old_f)-{"tick"}
old_tick, new_tick = old_f["tick"], new_f["tick"]
# The first new statement in tick is reporting_requirements inside the queue-lock body.
def reporting_prefix(node):
    for item in ast.walk(node):
        if isinstance(item, ast.With):
            body=item.body
            for index, statement in enumerate(body):
                if isinstance(statement, ast.Assign) and any(isinstance(t, ast.Name) and t.id=="reporting_requirements" for t in statement.targets):
                    return body[:index]
    raise AssertionError("reporting block not located")
new_prefix=reporting_prefix(new_tick)
old_with=next(item for item in ast.walk(old_tick) if isinstance(item, ast.With))
assert [ast.dump(x,include_attributes=False) for x in old_with.body[:len(new_prefix)]]==[ast.dump(x,include_attributes=False) for x in new_prefix]
numerical_unchanged={}
for relative in freeze["protected_numerical_paths_unchanged"]:
    raw=subprocess.check_output(["git","-C",str(W),"show",freeze["base_commit"]+":"+relative])
    current=(W/relative).read_bytes()
    assert raw==current
    numerical_unchanged[relative]=hashlib.sha256(current).hexdigest()
denied=[]
def audit(event,args):
    blocked=event in {"subprocess.Popen","os.system","os.posix_spawn","ctypes.dlopen","ctypes.dlsym","socket.connect","socket.bind"}
    if event=="import" and args and str(args[0]).split(".")[0] in {"jax","jaxlib","numpy","torch","cupy","flax","optax"}:
        blocked=True
    if event=="open" and args and isinstance(args[0],(str,bytes,os.PathLike)):
        path=os.fsdecode(args[0])
        blocked |= path.startswith(("/dev/nvidia","/dev/dri","/p/rlprojects/RND/"))
        blocked |= path.startswith(str(RUN)+"/") and not path.startswith(str(OUT)+"/")
    if blocked:
        denied.append({"event":event,"arguments":repr(args)[:400]})
        raise RuntimeError("outside independent metadata scope: "+event)
sys.addaudithook(audit)
scripts=W/"10_jax_exploration_platform/scripts/atari_pixel_campaign"
sys.path.insert(0,str(scripts/"tests"));sys.path.insert(0,str(scripts))
spec=importlib.util.spec_from_file_location("independent_monitor_reporting_fixtures",scripts/"tests/test_post_initial_phase_reporting.py")
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
controller=m.controller
class AdditionalControls(m.PostInitialReportingTests):
    def test_optional_actions_json_parse_failure_remains_visible_due(self):
        (self.run/"next_actions.json").write_text("{broken JSON\n")
        rows=self.pending()
        self.assertEqual(len(rows),4)
        self.assertTrue(all(row["state"]=="awaiting_reviewed_evidence" and row["errors"] for row in rows))
    def test_optional_actions_nonobject_remains_visible_due(self):
        self.write(self.run/"next_actions.json",[])
        rows=self.pending()
        self.assertTrue(all(row["state"]=="awaiting_reviewed_evidence" and row["errors"] for row in rows))
    def test_symlink_cycle_is_reported_due_without_propagating(self):
        first,second=self.run/"first.json",self.run/"second.json"
        first.symlink_to("second.json");second.symlink_to("first.json")
        self.index["configuration_record"]={"path":str(first),"sha256":"0"*64}
        self.save_index()
        rows=self.pending()
        self.assertTrue(all(row["state"]=="awaiting_reviewed_evidence" and row["errors"] for row in rows))
    def test_source_only_approval_cannot_fill_actual_correctness_coverage(self):
        entry=self.evidence("common_correctness","h100",1)
        path=Path(entry["review"]["path"]);review=controller.read_json(path)
        review["kind"]="independent_common_correctness_source_review"
        review["actual_scientific_validation_performed"]=False
        entry["review"]=self.write(path,review);self.save_index()
        row=self.pending()[0]
        self.assertIn({"hardware_class":"h100","copies":1},row["missing"])
        self.assertTrue(row["errors"])
suite=unittest.defaultTestLoader.loadTestsFromModule(m)
# Add only the four independent methods; inherited base cases are not discovered again.
extras=[name for name in AdditionalControls.__dict__ if name.startswith("test_")]
for name in extras:suite.addTest(AdditionalControls(name))
for name in ("test_thirty_completed_ideas_leave_twenty_unchosen","test_additional_cohort_can_have_another_source_and_copy_grid","test_initial_completion_waits_for_every_matched_copy_count"):
    suite.addTest(m.queue_fixtures.QueueProtocolTests(name))
stream=io.StringIO()
result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
assert all(hashlib.sha256((W/name).read_bytes()).hexdigest()==sha for name,sha in before.items())
review={"kind":"independent_post_initial_monitor_reporting_source_review","passed":result.wasSuccessful() and not denied,"approved":False,"source_freeze":{"path":str(freeze_path),"sha256":FREEZE_SHA},"job_id":"6783003","step_id":step,"host":socket.gethostname(),"uid":os.getuid(),"at_utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),"method_executions":result.testsRun,"failures":len(result.failures),"errors":len(result.errors),"independent_controls":extras,"denied_events":denied,"old_functions_unchanged":unchanged,"queue_and_final_validation_prefix_ast_unchanged":True,"protected_numerical_files_unchanged":numerical_unchanged,"frozen_files_unchanged":True,"actual_scientific_or_gpu_or_device_or_library_or_network_actions":0,"actual_campaign_or_live_monitor_modified":False,"stdout":stream.getvalue()}
(OUT/"private_review.json").write_text(json.dumps(review,indent=2)+"\n")
print(json.dumps({k:v for k,v in review.items() if k not in {"stdout","old_functions_unchanged","protected_numerical_files_unchanged","independent_controls","denied_events"}}))
print(stream.getvalue(),file=sys.stderr)
raise SystemExit(not review["passed"])
