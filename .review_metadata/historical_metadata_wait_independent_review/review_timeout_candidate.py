"""Independently verify only the prepared metadata-wait candidate on owned CPU."""
import ast
import copy
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]
DIRECTORY = Path(__file__).resolve().parent
INTEGRATION = Path("/p/rlprojects/RND_worktrees/merge-atari-cleanrl")
RUN = INTEGRATION / "10_jax_exploration_platform/benchmark_runs/2026-10-31-00-11_jaxatari_montezuma_pixels_cleanrl_ppo_rnd_environments-per-copy-128_rollout-steps-128_ideas-30-plus-20_steps-per-copy-2000000000_seed-start-1"
BASE = "7fef164909b5eb8b31636da7ae0c653282388203"
PREFIX = "10_jax_exploration_platform/scripts/atari_pixel_campaign/"
FREEZE = ROOT / ".review_metadata/historical_metadata_wait_candidate/frozen_candidate_for_independent_review.json"
PROTECTED = RUN / "remote_setup/direct_nsight_incoming_independent_2026-11-04-20-01-56_protected_before.json"


def digest(path):
    """Hash an existing ordinary file with bounded memory, without loading it as code."""
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(1024 * 1024):
            value.update(block)
    return value.hexdigest()


def main():
    """Check exact source changes, immutable prior waits and protected source bytes."""
    if os.environ.get("SLURM_JOB_ID") != "6783003" or not os.environ.get("SLURM_STEP_ID", "").isdigit():
        raise ValueError("use only the admitted owned CPU numeric step")
    if os.getuid() != os.geteuid() or os.getuid() != 2618919:
        raise ValueError("the exact campaign owner is required")
    expected_cgroup = "0::/system.slice/slurmstepd.scope/job_6783003/step_" + os.environ["SLURM_STEP_ID"] + "/user/task_0"
    assert Path("/proc/self/cgroup").read_text().splitlines() == [expected_cgroup]
    assert (RUN / "slurm/submitted_jobids_uva.txt").read_text().splitlines().count("6783003") == 1
    assert digest(FREEZE) == "4422b94b3a585c061071313108da06ffad9e4b789c0679fb48de029096f593d5"
    frozen = json.loads(FREEZE.read_text())
    before = {name: digest(name) for name in frozen["files"]}
    assert before == {name: item["sha256"] for name, item in frozen["files"].items()}
    assert len(before) == 10 and frozen["admission_granted"] is False and frozen["actual_runtime_upper_bound_known"] is False
    changes = []
    for name, old, new in (("direct_historical_consumer.py",180,900),("direct_common_confirmation.py",180,1800)):
        relative = PREFIX + name
        original = subprocess.check_output(["git", "-C", str(INTEGRATION), "show", BASE + ":" + relative])
        candidate = (ROOT / relative).read_bytes()
        tree, prior = ast.parse(candidate), ast.parse(original)
        if name == "direct_historical_consumer.py":
            values = [node for node in tree.body if isinstance(node, ast.Assign) and any(isinstance(target,ast.Name) and target.id=="WAIT_SECONDS" for target in node.targets)]
            assert len(values)==1 and isinstance(values[0].value,ast.Constant) and type(values[0].value.value) is int and values[0].value.value==new
            values[0].value.value=old
            restored=candidate.replace(b"WAIT_SECONDS = 900",b"WAIT_SECONDS = 180")
            location="module WAIT_SECONDS"
        else:
            request=next(node for node in tree.body if isinstance(node,ast.FunctionDef)and node.name=="request")
            calls=[node for node in ast.walk(request) if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute)and isinstance(node.func.value,ast.Name)and node.func.value.id=="subprocess"and node.func.attr=="run"]
            assert len(calls)==1
            values=[key for key in calls[0].keywords if key.arg=="timeout"]
            assert len(values)==1 and isinstance(values[0].value,ast.Constant)and type(values[0].value.value) is int and values[0].value.value==new
            values[0].value.value=old
            restored=candidate.replace(b"timeout=1800",b"timeout=180")
            location="request subprocess.run timeout"
        assert restored==original, "a source byte changed outside its one explicit metadata wait"
        assert ast.dump(tree,include_attributes=False)==ast.dump(prior,include_attributes=False)
        changes.append({"path":relative,"before_seconds":old,"candidate_seconds":new,"location":location,
            "source_sha256":hashlib.sha256(candidate).hexdigest(),"all_other_source_bytes_and_syntax_unchanged":True})
    tracked=subprocess.check_output(["git","-C",str(ROOT),"diff","--name-only",BASE,"--",PREFIX],text=True).splitlines()
    assert set(tracked)=={PREFIX+item for item in ("direct_historical_consumer.py","direct_common_confirmation.py")}
    actual_worktrees=subprocess.check_output(["git","-C",str(INTEGRATION),"worktree","list","--porcelain"],text=True)
    private=[]
    for block in actual_worktrees.strip().split("\n\n"):
        lines=block.splitlines()
        if "HEAD "+BASE not in lines:continue
        path=Path(next(line[9:]for line in lines if line.startswith("worktree ")))
        if path==ROOT:continue
        values={name:digest(path/(PREFIX+name))for name in ("direct_historical_consumer.py","direct_common_confirmation.py")}
        assert all(values[name]==hashlib.sha256(subprocess.check_output(["git","-C",str(INTEGRATION),"show",BASE+":"+PREFIX+name])).hexdigest() for name in values)
        private.append({"path":str(path),"source_files":values,"admitted_child_wait_seconds":180,"admitted_guard_wait_seconds":180})
    assert private,"the previous clean private helper source must remain present and unchanged"
    protected=json.loads(PROTECTED.read_text())["files"]
    for name,item in protected.items():
        path=Path(name)
        assert digest(path)==item["sha256"]and path.stat().st_size==item["size_bytes"]and path.stat().st_mode&0o777==item["mode"],name
    after={name:digest(name)for name in frozen["files"]}
    assert before==after
    record={"schema_version":1,"kind":"independent_unadmitted_metadata_wait_candidate_source_review","approved":True,"passed":True,
        "at_utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),"candidate_source_freeze":{"path":str(FREEZE),"sha256":digest(FREEZE)},
        "actual_owned_job":"6783003","actual_numeric_step":os.environ["SLURM_STEP_ID"],"actual_pid":os.getpid(),
        "exact_changes":changes,"frozen10files_unchanged":True,"existing_private_sources_unchanged":private,
        "protected_source_roster":{"path":str(PROTECTED),"sha256":digest(PROTECTED),"files_verified":len(protected)},
        "preserved_author_metadata_tests":27,"author_actual_step":"6783003.376","actual_genuine_guard_invoked_by_reviewer":False,
        "actual_runtime_upper_bound_known":False,"previous_observation":"The prior nested child exceeded180seconds, was killed and reaped; this is a lower bound only.",
        "diagnostic_outer_seconds":2100,"diagnostic_packet_approval_granted":False,"admission_granted":False,
        "required_before_adoption":["Fresh reviewed standard diagnostic packet and source publication before one actual trial.","Actual full successful raw read and independent runtime/source review.","Separate explicit recorded adoption; current admitted180second limits remain unchanged."],
        "actual_numerical_gpu_device_tool_operations":0,"scientific_credit":0}
    path=DIRECTORY/"independent_source_review.json"
    with path.open("x")as stream:json.dump(record,stream,indent=2,sort_keys=True);stream.write("\n");stream.flush();os.fsync(stream.fileno())
    print(json.dumps({"approved":True,"step":record["actual_numeric_step"],"protected_files":len(protected),"receipt_sha256":digest(path)}))


if __name__=="__main__":main()
