"""Independently test held observation-only source with selected-Python CPU children."""
import ast
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

D = Path(__file__).resolve().parent
W = D.parents[1]
S = W / "10_jax_exploration_platform/scripts/atari_pixel_campaign"
FROZEN = "63c3186658242633d913b8302c1b0dae38adb8ee72faf311c64b693d324665cc"
START = time.monotonic()
CHECKS = []


def require(value, statement):
    assert value, statement
    CHECKS.append(statement)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


require(os.environ.get("SLURM_JOB_ID") == "6821437" and os.environ.get("SLURM_STEP_ID", "").isdigit(), "current exact owned CPU job and step")
require(os.getresuid() == (2618919,)*3, "actual independent reviewer UID values")
freeze = json.loads((D / "corrected_source_freeze.json").read_bytes())
require(sha(D / "corrected_source_freeze.json") == FROZEN, "exact held three-file author freeze")
for item in freeze["files"].values():
    path = Path(item["path"])
    require(sha(path) == item["sha256"] and path.stat().st_size == item["bytes"] and oct(stat.S_IMODE(path.stat().st_mode)) == item["mode"], "exact held file bytes/mode: " + path.name)
require(sha(freeze["selected_executable"]["path"]) == freeze["selected_executable"]["sha256"], "actual selected private executable image hash")
require(freeze["historical_failure_cause_established"] is False and freeze["full_original_launch_reproduced"] is False and freeze["operational_admission"] is False, "source freeze grants no retrospective cause or operational/scientific admission")
tests = load(S / "tests/test_initial_child_identity.py", "independent_held_initial_child_tests")
suite = unittest.defaultTestLoader.loadTestsFromModule(tests)
require(suite.countTestCases() == 4, "exact four focused affected author methods, no broad replay")
result = unittest.TextTestRunner(stream=sys.stderr, verbosity=2).run(suite)
require(result.wasSuccessful() and result.testsRun == 4, "all four final-byte focused methods passed")
control = load(S / "direct_common_timing_controller.py", "independent_actual_observation_controller")
controller_sha = sha(S / "direct_common_timing_controller.py")
old_run = Path("/p/rlprojects/RND_worktrees/merge-atari-cleanrl/10_jax_exploration_platform/benchmark_runs/2026-10-31-00-11_jaxatari_montezuma_pixels_cleanrl_ppo_rnd_environments-per-copy-128_rollout-steps-128_ideas-30-plus-20_steps-per-copy-2000000000_seed-start-1")
old_packet = next((old_run / "remote_setup").glob("2026-11-05-04-07_common-timing*"))
old_source = (old_packet / "direct_qualification_controller.py").read_text()
old_functions = {node.name: ast.dump(node, include_attributes=False) for node in ast.parse(old_source).body if isinstance(node, ast.FunctionDef)}
new_functions = {node.name: ast.dump(node, include_attributes=False) for node in ast.parse(Path(control.__file__).read_text()).body if isinstance(node, ast.FunctionDef)}
require({name for name in old_functions if old_functions[name] != new_functions[name]} == {"supervise"} and set(new_functions)-set(old_functions) == {"child_identity_observation"}, "only original supervise observations and one pure record helper changed")
actual_cases = []
for mismatch in (False, True):
    with tempfile.TemporaryDirectory(prefix="independent_initial_child_") as temporary:
        run = Path(temporary);unit = run / "unit";unit.mkdir()
        (run / "processes.json").write_text("[]\n")
        hostname = socket.gethostname().split(".")[0]
        plan = {"campaign_run_dir":str(run),"direct_binding":{"profile":{"hostname":hostname}},"gpu_uuid":"synthetic_CPU_fixture_no_device","kind":"synthetic_CPU_only_observation_fixture","unit_id":"unit","source_commit":"synthetic_no_scientific_source","source_root":str(run),"helpers":{"source_commit":"synthetic_no_helper_admission"}}
        (unit / "measurement_plan.json").write_text(json.dumps(plan))
        child_script = unit / "CPU_only_child.py"
        child_script.write_text("import pathlib,time\nend=time.monotonic()+0.3\np=pathlib.Path(__file__).parent/'direct_current_launch.json'\nwhile time.monotonic()<end and not p.exists(): time.sleep(0.005)\nraise SystemExit(0 if p.exists() else 2)\n")
        command = [str(tests.diagnostic.SELECTED_PYTHON),"-I","-S","-B",str(child_script)]
        original_reader = control.read_process
        first_seen = []
        def read_actual(pid, uid):
            actual = original_reader(pid, uid)
            if mismatch and pid != os.getpid() and not first_seen:
                first_seen.append(pid)
                return {**actual,"command":["explicit_independent_synthetic_initial_mismatch"]}
            return actual
        original_affinity = sorted(os.sched_getaffinity(0))
        with patch.object(control,"read_process",side_effect=read_actual):
            outcome = control.supervise(plan,command,unit,{"PATH":"/usr/bin:/bin","LANG":"C.UTF-8","TZ":"UTC"},expected_uid=2618919,cpu_count=1,registration_hostname=hostname,seconds=3)
        initial = json.loads((unit / "direct_controller_initial_child_observation.json").read_bytes())
        wait = json.loads((unit / "direct_process_wait_return.json").read_bytes())
        require(sorted(os.sched_getaffinity(0)) == original_affinity, "actual controller fixture restores original affinity")
        require(wait["pid"] == wait["waited_pid"] == initial["actual_popen_pid"] and wait["process_start_time"] == initial["observed_child"]["process_start_time"] and wait["timeouts"] == [], "actual own CPU child PID/birth and normal wait retained")
        require(not Path("/proc",str(wait["pid"])).exists(), "exact waited CPU fixture child is no longer present")
        if mismatch:
            require(outcome["passed"] is False and initial["predicate_results"]["command_matches"] is False and wait["returncode"] == 2, "real controller publishes labelled synthetic mismatch then refuses and waits")
            require(not (unit / "direct_controller_spawn.json").exists() and not (unit / "direct_current_launch.json").exists() and json.loads((run / "processes.json").read_text()) == [], "initial refusal publishes no spawn, launch or registration")
        else:
            pre = json.loads((unit / "direct_controller_pre_registration_child_observation.json").read_bytes())
            require(outcome["passed"] is True and initial["all_predicates_match"] and pre["all_predicates_match"] and wait["returncode"] == 0, "real controller publishes both matching observations and waits for actual zero exit")
            require((unit / "direct_controller_spawn.json").is_file() and (unit / "direct_current_launch.json").is_file() and len(json.loads((run / "processes.json").read_text())) == 1, "normal CPU-only fixture publishes one actual registration")
        actual_cases.append({"synthetic_initial_command_mismatch":mismatch,"observed_initial_predicates":initial["predicate_results"],"actual_pid":wait["pid"],"actual_birth":wait["process_start_time"],"normal_wait_status":wait["returncode"],"passed":outcome["passed"],"scientific_admission":False,"CPU_count":1,"explicit_numerical_or_GPU_operations":False})
for item in freeze["files"].values():
    require(sha(item["path"]) == item["sha256"], "held source unchanged after actual CPU fixtures: " + Path(item["path"]).name)
require(sha(D / "corrected_source_freeze.json") == FROZEN, "author freeze unchanged after independent tests")
report = {"schema_version":1,"kind":"independent_initial_child_observation_and_selected_interpreter_CPU_source_review","passed":True,"approved":True,"freeze":{"path":str(D/"corrected_source_freeze.json"),"sha256":FROZEN},"controller_sha256":controller_sha,"files":freeze["files"],"focused_author_methods_passed":4,"independent_actual_controller_cases":actual_cases,"selected_private_interpreter_one_CPU_fixture_children":4,"actual_target_eight_CPU_diagnostic_run":False,"target_run_admitted":False,"scientific_credit":0,"historical_failure_cause_established":False,"full_original_launch_reproduced":False,"diagnostic_startup_flags_differ_from_original":True,"process_native_startup_work":"unquantified","source_review_scope":"observation-only controller correction and isolated selected-interpreter CPU acknowledgment diagnostic; no numerical, direct GPU, profiling or training admission","actual_owned_CPU_context":{"job_id":os.environ["SLURM_JOB_ID"],"step_id":os.environ["SLURM_STEP_ID"],"hostname":socket.gethostname(),"pid":os.getpid(),"uids":list(os.getresuid()),"cgroup":Path("/proc/self/cgroup").read_text()},"required_before_root_target_diagnostic":["Publish exact source, this independent review and standard zero-credit diagnostic records.","Bind actual selected-image and source hashes, run once on the approved direct host with eight selected CPUs; never touch the standing GPU lock or change campaign metadata permissions.","Preserve every first/final identity, acknowledgment and actual wait including failure; do not infer the failed historical child values.","No production timing repair or numerical admission follows from this isolated-startup diagnostic alone."],"checks":CHECKS,"elapsed_seconds":time.monotonic()-START}
out=D/"independent_source_review.json"
with out.open("x") as stream:
    json.dump(report,stream,indent=2,sort_keys=True);stream.write("\n");stream.flush();os.fsync(stream.fileno())
print(json.dumps({"passed":True,"checks":len(CHECKS),"step":os.environ["SLURM_STEP_ID"],"elapsed_seconds":report["elapsed_seconds"],"review":{"path":str(out),"sha256":sha(out)}}))
