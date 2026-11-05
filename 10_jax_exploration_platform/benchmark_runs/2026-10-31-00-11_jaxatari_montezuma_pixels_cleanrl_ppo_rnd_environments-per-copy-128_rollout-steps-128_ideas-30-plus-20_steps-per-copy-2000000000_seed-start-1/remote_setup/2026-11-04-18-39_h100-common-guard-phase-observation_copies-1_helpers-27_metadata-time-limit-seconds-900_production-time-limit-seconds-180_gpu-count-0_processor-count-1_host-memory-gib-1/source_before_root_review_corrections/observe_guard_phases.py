"""Observe named metadata validation phases without changing the published guard."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
import time

PACKET = Path(__file__).resolve(strict=True).parent


def require(condition, message):
    """Stop this diagnostic when an actual input or observation is invalid."""
    if not condition:
        raise ValueError(message)


def digest(raw):
    """Identify exact unchanged metadata or diagnostic output."""
    return hashlib.sha256(raw).hexdigest()


def save(name, value):
    """Save a new durable artifact without replacing a prior diagnostic attempt."""
    raw = value if isinstance(value, bytes) else (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    with (PACKET / name).open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    return {"path": str(PACKET / name), "sha256": digest(raw)}


def process(pid):
    """Record only the actual owned metadata parent or its own child."""
    root = Path("/proc") / str(pid)
    fields = (root / "stat").read_text().rsplit(")", 1)[1].split()
    uid_line = next(line for line in (root / "status").read_text().splitlines() if line.startswith("Uid:"))
    uids = list(map(int, uid_line.split()[1:]))
    require(uids == [2618919] * 4, "metadata process ownership changed")
    return {"pid": pid, "parent_pid": int(fields[1]), "birth_counter": fields[19], "all_uids": uids,
        "cgroup": (root / "cgroup").read_text(),
        "command": (root / "cmdline").read_bytes().rstrip(b"\0").decode().split("\0")}


class PhaseObserver:
    """Record only enter/return times for ten explicitly named validator functions."""

    def __init__(self, manifest):
        """Open one small trace before the unchanged guard installs its audit hook."""
        self.targets = {(filename, name) for filename, names in manifest["targets"].items() for name in names}
        self.names = {name for unused, name in self.targets}
        self.event_limit = manifest["event_limit"]
        self.byte_limit = manifest["trace_byte_limit"]
        self.stream = (PACKET / "validator_phase_times.jsonl").open("xb")
        self.started = time.monotonic()
        self.sequence = 0
        self.events = 0
        self.bytes = 0
        self.active = {}

    def __call__(self, frame, event, argument):
        """Leave every untargeted function and all function arguments/results unobserved."""
        if event not in {"call", "return"} or frame.f_code.co_name not in self.names:
            return
        key = (str(Path(frame.f_code.co_filename).absolute()), frame.f_code.co_name)
        if key not in self.targets:
            return
        now = time.monotonic()
        # A frame key links each actual entry and return without reading its values.
        if event == "call":
            self.sequence += 1
            self.active[id(frame)] = (self.sequence, now)
            identifier, entered = self.active[id(frame)]
        else:
            require(id(frame) in self.active, "a named metadata return has no observed entry")
            identifier, entered = self.active.pop(id(frame))
        row = {"event": event, "call_id": identifier, "source_file": key[0], "function": key[1],
            "pid": os.getpid(), "elapsed_seconds": now - self.started}
        if event == "return":
            row["duration_seconds"] = now - entered
        raw = (json.dumps(row, sort_keys=True, allow_nan=False) + "\n").encode()
        require(self.events < self.event_limit and self.bytes + len(raw) <= self.byte_limit,
            "the declared small metadata trace bound was reached")
        self.stream.write(raw)
        self.stream.flush()
        os.fsync(self.stream.fileno())
        self.events += 1
        self.bytes += len(raw)

    def close(self, guard_returned_normally):
        """Retain finite diagnostic coverage without granting production acceptance."""
        self.stream.flush()
        os.fsync(self.stream.fileno())
        self.stream.close()
        save("validator_phase_summary.json", {"kind": "named_metadata_validator_phase_summary",
            "actual_pid": os.getpid(), "events": self.events, "trace_bytes": self.bytes,
            "active_calls_at_exit": len(self.active), "guard_returned_normally": guard_returned_normally,
            "production_guard_qualified": False, "scientific_credit": 0})


def observed_child():
    """Run the unchanged published guard with only the declared Python call observer."""
    manifest = json.loads((PACKET / "manifest.yaml").read_bytes())
    source = Path(manifest["original_guard"])
    plan = json.loads((PACKET / "unregistered_guard_input_plan.json").read_bytes())
    require(digest(source.read_bytes()) == plan["helpers"]["files"][
        "10_jax_exploration_platform/scripts/atari_pixel_campaign/direct_common_confirmation.py"],
        "the exact published guard source changed")
    observer = PhaseObserver(manifest)
    normal = False
    sys.argv = [str(source), "--plan-stdin"]
    sys.setprofile(observer)
    try:
        runpy.run_path(str(source), run_name="__main__")
        normal = True
    finally:
        sys.setprofile(None)
        observer.close(normal)


def parent():
    """Bound one separate zero-credit observation on the exact owned CPU job."""
    require(os.environ.get("SLURM_JOB_ID") == "6783003" and os.environ.get("SLURM_STEP_ID", "").isdecimal()
        and os.getuid() == os.geteuid() == 2618919, "use only the actual owned CPU allocation")
    manifest = json.loads((PACKET / "manifest.yaml").read_bytes())
    raw = (PACKET / "unregistered_guard_input_plan.json").read_bytes()
    require(manifest["diagnostic_timeout_seconds"] == 900 and manifest["production_timeout_seconds_unchanged"] == 180
        and digest(raw) == manifest["input_plan"]["sha256"], "the separate diagnostic limit or unchanged input changed")
    plan = json.loads(raw)
    for relative, expected in plan["helpers"]["files"].items():
        require(digest((Path(plan["helpers"]["source_root"]) / relative).read_bytes()) == expected,
            "an unchanged published helper differs")
    command = ["/usr/bin/python3", "-I", "-S", "-B", str(Path(__file__).resolve(strict=True)), "--observed-guard-child"]
    before = process(os.getpid())
    save("diagnostic_intent.json", {"kind": "one_distinct_metadata_phase_observation_attempt",
        "at_utc": datetime.now(timezone.utc).isoformat(), "job_id": "6783003", "step_id": os.environ["SLURM_STEP_ID"],
        "parent": before, "command": command, "plan_sha256": digest(raw), "diagnostic_timeout_seconds": 900,
        "production_timeout_seconds": 180, "automatic_retry_permitted": False, "scientific_credit": 0})

    # Observe one own child and normal wait; never start a numerical or GPU producer.
    started = time.time()
    child = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    identity = process(child.pid)
    require(identity["parent_pid"] == os.getpid() and identity["command"] == command, "metadata child identity changed")
    save("diagnostic_child.json", identity)
    timeout = False
    try:
        stdout, stderr = child.communicate(input=raw, timeout=900)
    except subprocess.TimeoutExpired:
        timeout = True
        child.kill()
        stdout, stderr = child.communicate()
    ended = time.time()
    stdout_ref = save("diagnostic_stdout.raw", stdout)
    stderr_ref = save("diagnostic_stderr.raw", stderr)
    after = process(os.getpid())
    wait = {"kind": "actual_distinct_metadata_phase_observation_wait", "actual_child": identity,
        "parent_before": before, "parent_after": after, "started_at_seconds": started, "finished_at_seconds": ended,
        "returncode": child.returncode, "timeout": timeout, "exited_normally": not timeout and child.returncode == 0,
        "stdout": stdout_ref, "stderr": stderr_ref, "production_timeout_seconds_unchanged": 180}
    wait_ref = save("diagnostic_wait.json", wait)
    reported = None
    if child.returncode == 0 and not timeout:
        proof = json.loads(stdout)
        require(not stderr and proof["passed"] is True and proof["plan_sha256"] == digest(raw)
            and proof["parent_pid"] == os.getpid() and proof["validation_pid"] == child.pid,
            "normal diagnostic output changes the actual guard identity")
        reported = proof["passed"]
    save("diagnostic_observation.json", {"kind": "zero_credit_metadata_phase_observation",
        "at_utc": datetime.now(timezone.utc).isoformat(), "job_id": "6783003", "step_id": os.environ["SLURM_STEP_ID"],
        "wait": wait_ref, "guard_reported_passed": reported, "returncode": child.returncode, "timeout": timeout,
        "production_guard_qualified": False, "production_timeout_seconds_unchanged": 180,
        "formal_timing_preparation_permitted_by_this_diagnostic": False, "scientific_credit": 0,
        "timing_credit": 0, "idea_credit": 0, "case_credit": 0, "final_training_credit": 0, "admission_granted": False})
    print(json.dumps({"step_id": os.environ["SLURM_STEP_ID"], "returncode": child.returncode,
        "timeout": timeout, "guard_reported_passed": reported, "production_guard_qualified": False}))


if __name__ == "__main__":
    if sys.argv[1:] == ["--observed-guard-child"]:
        observed_child()
    else:
        require(not sys.argv[1:], "the diagnostic parent accepts no operational override")
        parent()
