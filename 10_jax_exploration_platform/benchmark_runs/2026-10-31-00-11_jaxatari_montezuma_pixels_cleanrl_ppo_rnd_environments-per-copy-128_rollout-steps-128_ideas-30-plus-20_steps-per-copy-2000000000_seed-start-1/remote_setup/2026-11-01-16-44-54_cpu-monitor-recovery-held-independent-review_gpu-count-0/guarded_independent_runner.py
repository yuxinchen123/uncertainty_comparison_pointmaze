"""Only standard-library submission-control metadata checks on portal05."""
import ctypes
from datetime import datetime, timezone
import hashlib
import importlib.abc
import importlib.util
import io
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import unittest

P = Path('/p/rlprojects/RND_worktrees/merge-atari-cleanrl/10_jax_exploration_platform/benchmark_runs/2026-10-31-00-11_jaxatari_montezuma_pixels_cleanrl_ppo_rnd_environments-per-copy-128_rollout-steps-128_ideas-30-plus-20_steps-per-copy-2000000000_seed-start-1/remote_setup/2026-11-01-16-33_pixel-study-monitor-recovery_source-commit-79a01e851ff92ddba7d0e12eaf8164cab2028878_processor-count-1_host-memory-gib-2_time-limit-days-7_poll-interval-seconds-1200_partition-nolim_account-sl5nw-base_gpu-count-0')
assert socket.gethostname().split(".")[0] == "portal05" and os.getuid() == 2618919
assert not os.environ.get("SLURM_JOB_ID")
NUM = {"jax", "jaxlib", "numpy", "torch", "flax", "optax", "cupy", "scipy", "jaxtari"}
assert not NUM.intersection(sys.modules)
names = ["submit_monitor_held.py", "test_submit_monitor_held.py", "run_recovery_held_metadata_tests.py",
         "held_submission_inputs.json", "held_submission_author_review.json"]
def images():
    return {name: hashlib.sha256((P / name).read_bytes()).hexdigest() for name in names}
before = images()
frozen=json.loads((P/'held_submission_source_freeze.json').read_bytes())['files']
frozen_before={path:hashlib.sha256(Path(path).read_bytes()).hexdigest() for path in frozen}
assert frozen_before==frozen
static_before=json.loads((P/'static_source_freeze.json').read_bytes())['files']
assert static_before=={path:hashlib.sha256(Path(path).read_bytes()).hexdigest() for path in static_before}
inputs = json.loads((P / "held_submission_inputs.json").read_bytes())
protected_before = {entry["path"]: hashlib.sha256(Path(entry["path"]).read_bytes()).hexdigest()
                    for entry in inputs["required_files"]}
assert protected_before == {entry["path"]: entry["sha256"] for entry in inputs["required_files"]}
blocked = []
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in NUM:
            blocked.append(fullname)
            raise AssertionError("numerical imports forbidden")
sys.meta_path.insert(0, Block())
def audit(event, args):
    if event.startswith("socket.") and event != "socket.gethostname":
        raise AssertionError("network access forbidden")
    if event in {"ctypes.dlopen", "ctypes.dlsym"}:
        raise AssertionError("dynamic-library operation forbidden")
    if event == "open" and (str(args[0]).startswith(("/dev/nvidia", "/dev/dri"))
                            or str(args[0]) == "/tmp/sl5nw_serval02_card1.lock"):
        raise AssertionError("GPU or standing-lock operation forbidden")
    if event == "subprocess.Popen":
        raise AssertionError("all live commands forbidden during control tests")
sys.addaudithook(audit)
spec = importlib.util.spec_from_file_location("recovery_held_tests", P / "test_submit_monitor_held.py")
tests = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tests)
def append_failure(self):
    original=tests.submit.append_ledgers
    def append_then_fail(campaign,identifier,row):
        original(campaign,identifier,row)
        self.assertTrue(all(path.read_text().splitlines().count('9910002')==1 for path in self.paths[:2]))
        raise OSError('private complete ledger append then failure')
    with tests.patch.object(tests.submit.subprocess,'run',return_value=self.result(b'9910002\n')) as scheduler,tests.patch.object(tests.submit,'append_ledgers',side_effect=append_then_fail):
        with self.assertRaises(OSError):tests.submit.submit_once('a'*40)
        scheduler.assert_called_once()
    before=[path.read_bytes() for path in self.paths]
    with tests.patch.object(tests.submit.subprocess,'run') as scheduler:
        with self.assertRaises(ValueError):tests.submit.submit_once('a'*40)
        scheduler.assert_not_called()
    self.assertEqual(before,[path.read_bytes() for path in self.paths])
    self.assertFalse((self.packet/'submission_receipt.json').exists())
    self.assertFalse((self.packet/'held_binding.json').exists())
tests.HeldMonitorSubmissionTests.test_independent_complete_ledger_append_failure_never_retries=append_failure
output = io.StringIO()
result = unittest.TextTestRunner(stream=output, verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(tests))
actual_inputs, plan = tests.submit.validate_inputs()
seen=[];repository=Path(actual_inputs['repository'])
def object_bytes(command):
    assert command[:4]==['git','-C',str(repository),'show']
    relative=command[4].split(':',1)[1];seen.append(relative);return (repository/relative).read_bytes()
with tests.patch.object(tests.submit.subprocess,'run') as ancestor,tests.patch.object(tests.submit.subprocess,'check_output',side_effect=object_bytes):
    tests.submit.validate_publication(actual_inputs,'a'*40)
    ancestor.assert_called_once_with(['git','-C',str(repository),'merge-base','--is-ancestor','a'*40,'origin/merge-atari-cleanrl'],check=True)
assert len(seen)==len(actual_inputs['required_files'])+5
with tests.patch.object(tests.submit.subprocess,'run'),tests.patch.object(tests.submit.subprocess,'check_output',return_value=b'private changed object'):
    try:tests.submit.validate_publication(actual_inputs,'a'*40)
    except ValueError:pass
    else:raise AssertionError('changed publication accepted')
assert frozen_before=={path:hashlib.sha256(Path(path).read_bytes()).hexdigest() for path in frozen_before}
assert static_before=={path:hashlib.sha256(Path(path).read_bytes()).hexdigest() for path in static_before}
after = images()
protected_after = {name: hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in protected_before}
assert after == before and protected_after == protected_before and not blocked
value = {"schema_version": 1, "kind": "recovery_held_submission_independent_portal_metadata_verification",
    "at_utc": datetime.now(timezone.utc).isoformat(), "metadata_host": socket.gethostname(),
    "metadata_allocation": None, "pid": os.getpid(), "uid": os.getuid(),
    "proc_stat": Path("/proc/self/stat").read_text(), "tests": result.testsRun,
    "passed": result.wasSuccessful(), "output": output.getvalue(), "source_before": before, "source_after": after,
    "actual_required_inputs": len(actual_inputs["required_files"]),
    "protected_before": protected_before, "protected_after": protected_after,
    "actual_static_input_check": True, "actual_publication_check": False,"actual_root_approval_checked":True,"mock_publication_objects":len(seen),"changed_publication_object_refused":True,"all_held_frozen_files_before":frozen_before,"all_held_frozen_files_after":{path:hashlib.sha256(Path(path).read_bytes()).hexdigest() for path in frozen_before},"all_static297_unchanged":True,
    "real_root_publication_approval_still_required": True,
    "prelaunch_full_candidate_available": False, "actual_full_campaign_report_on_login": False,
    "numerical_import_attempts": blocked, "live_commands": 0, "new_jobs": 0, "release_calls": 0,
    "scientific_credit": 0, "worker_admitted": False}
raw = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
target = Path('/p/rlprojects/RND_worktrees/merge-atari-cleanrl/10_jax_exploration_platform/benchmark_runs/2026-10-31-00-11_jaxatari_montezuma_pixels_cleanrl_ppo_rnd_environments-per-copy-128_rollout-steps-128_ideas-30-plus-20_steps-per-copy-2000000000_seed-start-1/remote_setup/2026-11-01-16-44-54_cpu-monitor-recovery-held-independent-review_gpu-count-0') / "independent_controls.json"
with target.open("xb") as stream:
    stream.write(raw)
    stream.flush()
    os.fsync(stream.fileno())
descriptor = os.open(P, os.O_DIRECTORY)
os.fsync(descriptor)
os.close(descriptor)
print(json.dumps({"receipt": str(target), "sha256": hashlib.sha256(raw).hexdigest(),
                  "tests": result.testsRun, "passed": result.wasSuccessful(), "output": output.getvalue()}))
raise SystemExit(0 if result.wasSuccessful() else 1)

