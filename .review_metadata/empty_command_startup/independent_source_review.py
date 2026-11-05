"""Independently verify the strict bounded empty-only startup observer on owned CPU."""
import ast
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import socket
import stat
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

D=Path(__file__).resolve().parent
W=D.parents[1]
S=W/"10_jax_exploration_platform/scripts/atari_pixel_campaign"
START=time.monotonic()
FROZEN="9fa36c9cef144d1008d9c91af41a301b75d44da863dcb00999706383f16052d1"
CHECKS=[]


def require(value,text):
    assert value,text
    CHECKS.append(text)


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


require(os.environ.get("SLURM_JOB_ID")=="6821437" and os.environ.get("SLURM_STEP_ID","").isdigit() and os.getresuid()==(2618919,)*3,"current exact owned CPU allocation/step/owner")
require(sha(D/"source_freeze.json")==FROZEN,"exact final held empty-only source freeze")
freeze=json.loads((D/"source_freeze.json").read_bytes())
for relative,item in freeze["files"].items():
    path=Path(item["path"])
    require(sha(path)==item["sha256"] and path.stat().st_size==item["bytes"] and oct(stat.S_IMODE(path.stat().st_mode))==item["mode"],"held exact source/test bytes and modes: "+Path(relative).name)
evidence=json.loads(Path(freeze["actual_evidence"]["path"]).read_bytes())
require(sha(freeze["actual_evidence"]["path"])==freeze["actual_evidence"]["sha256"] and evidence["earlier_failed_attempt_cause_inferred"] is False and evidence["exact_transition_time_or_kernel_mechanism_established"] is False and evidence["transition_ready_within_thirty_seconds_established"] is False,"actual evidence scope does not claim earlier cause, kernel mechanism or30second readytime")
for key in ("initial_observation","later_same_child_observation","independent_actual_transition_audit"):
    item=evidence[key]
    require(sha(item["path"])==item["sha256"],"exact already reviewed actual evidence reference: "+key)
new_tests=load(S/"tests/test_empty_command_startup.py","independent_final_empty_startup_tests")
preservation_tests=load(S/"tests/test_initial_child_identity.py","independent_final_initial_preservation_tests")
suite=unittest.defaultTestLoader.loadTestsFromModule(new_tests)
require(suite.countTestCases()==10,"exact ten affected empty-only startup methods")
suite.addTest(preservation_tests.InitialChildIdentityTests("test_existing_controller_checks_and_functions_are_unchanged"))
result=unittest.TextTestRunner(stream=sys.stderr,verbosity=2).run(suite)
require(result.wasSuccessful() and result.testsRun==11,"ten affected methods plus one source-preservation method pass on final bytes")
core=load(S/"direct_common_timing_controller.py","independent_final_empty_startup_controller")
r=Path("/p/rlprojects/RND_worktrees/merge-atari-cleanrl/10_jax_exploration_platform/benchmark_runs/2026-10-31-00-11_jaxatari_montezuma_pixels_cleanrl_ppo_rnd_environments-per-copy-128_rollout-steps-128_ideas-30-plus-20_steps-per-copy-2000000000_seed-start-1")
old=next((r/"remote_setup").glob("2026-11-05-05-10_common-timing*"))
old_text=(old/"direct_qualification_controller.py").read_text();new_text=Path(core.__file__).read_text()
old_functions={n.name:n for n in ast.parse(old_text).body if isinstance(n,ast.FunctionDef)}
new_functions={n.name:n for n in ast.parse(new_text).body if isinstance(n,ast.FunctionDef)}
require({name for name in old_functions if ast.dump(old_functions[name],include_attributes=False)!=ast.dump(new_functions[name],include_attributes=False)}=={"supervise"} and set(new_functions)-set(old_functions)=={"wait_for_exact_child_startup"},"only existing supervise and one new strict startup helper changed from failed09a source")
class RemoveStartupObserver(ast.NodeTransformer):
    def visit_Assign(self,node):
        if isinstance(node.value,ast.Call) and (any(isinstance(t,ast.Name) and t.id=="startup_started" for t in node.targets) or isinstance(node.value.func,ast.Name) and node.value.func.id=="wait_for_exact_child_startup"):
            return None
        return node
retained=RemoveStartupObserver().visit(copy.deepcopy(new_functions["supervise"]))
require(ast.dump(retained,include_attributes=False)==ast.dump(old_functions["supervise"],include_attributes=False),"removing only startup clock and strict helper call leaves all original predicates, observation records, publication, registration and wait logic unchanged")
helper=ast.get_source_segment(new_text,new_functions["wait_for_exact_child_startup"])
require('expected_raw = b"\\0".join' in helper and 'require(raw == expected_raw' in helper and 'raw == b"" and current["command"] == []' in helper,"only genuinely empty raw command may wait; ready requires complete exact raw bytes")
require('maximum_seconds <= 30' in helper and 'deadline = started_monotonic + maximum_seconds' in helper and helper.count('deadline =')==1,"one strict maximum30second deadline, not restarted per observation")
# Independent ordinary-error retention checks use explicitly synthetic metadata only.
for failure_point in ("observer","poll"):
    with tempfile.TemporaryDirectory() as temporary:
        initial=core.read_process(os.getpid(),2618919);expected=initial["command"]
        empty={**initial,"command":[],"command_bytes_hex":"","command_sha256":hashlib.sha256(b"").hexdigest()}
        class NoActualChild:
            pid=os.getpid()
            def poll(self):
                if failure_point=="poll":raise OSError("explicit synthetic own-Popen poll error")
                return None
        def fail_observer(pid,uid):raise OSError("explicit synthetic own-process observer error")
        try:
            core.wait_for_exact_child_startup(NoActualChild(),empty,expected,initial["affinity"],os.getppid(),2618919,Path(temporary),started_monotonic=time.monotonic(),observer=fail_observer,sleeper=lambda unused:None)
        except OSError:pass
        else:raise AssertionError("ordinary observer/poll error was incorrectly accepted")
        saved=json.loads((Path(temporary)/"direct_controller_startup_observation_sequence.json").read_bytes())
        require(saved["exact_command_observed"] is False and saved["registration_performed_by_this_observer"] is False and saved["failure"]["type"]=="OSError" and saved["observations"][0]["raw_command_is_empty"] is True,"ordinary "+failure_point+" error preserves actual empty observation sequence and refuses")
actual_cases=[]
for initial_kind in ("synthetic_empty_then_actual_exact","synthetic_extra_trailing_empty_argument"):
    with tempfile.TemporaryDirectory(prefix="independent_empty_startup_") as temporary:
        run=Path(temporary);unit=run/"unit";unit.mkdir();(run/"processes.json").write_text("[]\n")
        host=socket.gethostname().split(".")[0]
        plan={"campaign_run_dir":str(run),"direct_binding":{"profile":{"hostname":host}},"gpu_uuid":"synthetic_CPU_fixture_no_device","kind":"synthetic_CPU_only_startup_fixture","unit_id":"unit","source_commit":"synthetic_no_scientific_source","source_root":str(run),"helpers":{"source_commit":"synthetic_no_helper_admission"}}
        (unit/"measurement_plan.json").write_text(json.dumps(plan))
        child_source=unit/"CPU_only_child.py"
        child_source.write_text("import pathlib,time\np=pathlib.Path(__file__).parent/'direct_current_launch.json'\nend=time.monotonic()+1\nwhile not p.exists() and time.monotonic()<end: time.sleep(0.005)\nraise SystemExit(0 if p.exists() else 2)\n")
        command=["/p/rlprojects/RND/.venvs/atari_pixels_jax/bin/python","-I","-S","-B",str(child_source)]
        reader=core.read_process;first_seen=[]
        def first_modified(pid,uid):
            value=reader(pid,uid)
            if pid!=os.getpid() and not first_seen:
                first_seen.append(pid)
                if initial_kind=="synthetic_empty_then_actual_exact":return {**value,"command":[],"command_bytes_hex":"","command_sha256":hashlib.sha256(b"").hexdigest()}
                raw=b"\0".join(v.encode() for v in command)+b"\0\0"
                return {**value,"command":command,"command_bytes_hex":raw.hex(),"command_sha256":hashlib.sha256(raw).hexdigest()}
            return value
        original_affinity=sorted(os.sched_getaffinity(0))
        with patch.object(core,"read_process",side_effect=first_modified):
            outcome=core.supervise(plan,command,unit,{"PATH":"/usr/bin:/bin","LANG":"C.UTF-8","TZ":"UTC"},expected_uid=2618919,cpu_count=1,registration_hostname=host,seconds=3)
        wait=json.loads((unit/"direct_process_wait_return.json").read_bytes())
        sequence=json.loads((unit/"direct_controller_startup_observation_sequence.json").read_bytes())
        initial=json.loads((unit/"direct_controller_initial_child_observation.json").read_bytes())
        require(sorted(os.sched_getaffinity(0))==original_affinity and wait["pid"]==wait["waited_pid"]==initial["actual_popen_pid"] and wait["process_start_time"]==initial["observed_child"]["process_start_time"] and wait["timeouts"]==[],"actual CPU fixture retains exact own child wait and restores affinity")
        require(not Path("/proc",str(wait["pid"])).exists(),"actual waited CPU fixture child no longer exists")
        if initial_kind=="synthetic_empty_then_actual_exact":
            require(outcome["passed"] is True and wait["returncode"]==0 and initial["all_predicates_match"] is False and sequence["observations"][0]["raw_command_is_empty"] is True and sequence["exact_command_observed"] is True and sequence["observations"][-1]["expected_raw_command_matches"] is True,"real controller permits labelled synthetic empty first sample only after actual expected child command with exact raw bytes")
            require((unit/"direct_current_launch.json").exists() and len(json.loads((run/"processes.json").read_text()))==1,"real CPU success registers only after strict exact startup observer")
        else:
            require(outcome["passed"] is False and wait["returncode"]==2 and sequence["exact_command_observed"] is False and sequence["observations"][0]["expected_raw_command_matches"] is False,"real controller refuses synthetic extra empty argument and waits exact child to failure")
            require(not (unit/"direct_controller_spawn.json").exists() and not (unit/"direct_current_launch.json").exists() and json.loads((run/"processes.json").read_text())==[],"raw-command mismatch never registers or publishes success")
        actual_cases.append({"explicitly_synthetic_initial_metadata_case":initial_kind,"actual_child_pid":wait["pid"],"actual_child_birth":wait["process_start_time"],"actual_normal_wait_returncode":wait["returncode"],"CPU_count":1,"scientific_admission":False,"NUM_or_GPU_work":False})
for item in freeze["files"].values():require(sha(item["path"])==item["sha256"],"held final source/test unchanged after focused CPU checks: "+Path(item["path"]).name)
require(sha(D/"source_freeze.json")==FROZEN,"held exact source freeze unchanged after independent checks")
require(not {"jax","jaxlib","numpy","torch","cupy","flax","optax","scipy","jaxtari","exploration_platform"}.intersection(sys.modules),"no numerical imports in reviewer")
report={"schema_version":1,"kind":"independent_strict_empty_only_bounded_startup_source_review","passed":True,"approved":True,"source_freeze":{"path":str(D/"source_freeze.json"),"sha256":FROZEN},"files":freeze["files"],"focused_methods_passed":11,"additional_ordinary_error_retention_cases":2,"independent_actual_controller_CPU_cases":actual_cases,"source_change_scope":"one bounded empty-only startup observer and its supervise call/clock; original nonempty argv/parent/birth/affinity/publication/wait checks retained","maximum_startup_wait_seconds":30,"nonempty_command_requires_complete_exact_raw_bytes":True,"all_UID_PID_birth_PPID_affinity_rechecked_each_observation":True,"earlier_failed_attempt_cause_inferred":False,"kernel_mechanism_or_actual_readytime_established":False,"actual_original_flags_CPU_or_GPU_launch_repeated":False,"isolated_actual_CPU_fixture_children":3,"scientific_credit":0,"GPU_or_worker_admission":False,"requires_before_actual_new_timing_attempt":["Publish independently reviewed source and standard fresh-unit records; preserve failed05-10.","Verify actual previous child wait and both exact exits before any successor.","Prepare and bind a fresh unit using exact unchanged helper69f/29 and numericalD28/original scientific configuration.","Review actual bound source/records and one-shot transport before root-only actual launch; original complete expected argv remains mandatory before registration."],"actual_owned_CPU_context":{"job_id":os.environ["SLURM_JOB_ID"],"step_id":os.environ["SLURM_STEP_ID"],"hostname":socket.gethostname(),"pid":os.getpid()},"checks":CHECKS,"elapsed_seconds":time.monotonic()-START}
out=D/"independent_source_review.json"
with out.open("x") as stream:json.dump(report,stream,indent=2,sort_keys=True);stream.write("\n");stream.flush();os.fsync(stream.fileno())
print(json.dumps({"passed":True,"checks":len(CHECKS),"step":os.environ["SLURM_STEP_ID"],"elapsed_seconds":report["elapsed_seconds"],"review":{"path":str(out),"sha256":sha(out)}}))
