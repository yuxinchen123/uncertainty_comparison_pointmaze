"""Review only the exact corrected starter derivation and prospective one-shot transport."""
import ast
import hashlib
import json
import os
from pathlib import Path
import shlex
import socket
import stat
import subprocess
import time

P = Path(__file__).resolve().parent
R = P.parent.parent
I = R.parents[2]
U = R / "data" / P.name
PUB = "61b57745832bb640ff5c66019edf0a4aca34ced4"
START = time.monotonic()
CHECKS = []


def require(condition, statement):
    """Record each actual small-source/literal assertion."""
    assert condition, statement
    CHECKS.append(statement)


def ref(path):
    """Bind an ordinary source or metadata file without executing it."""
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), "ordinary frozen input: " + str(path))
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


# Verify actual owned CPU context and the held nine-file transport freeze.
require(os.environ.get("SLURM_JOB_ID") == "6821437" and os.environ.get("SLURM_STEP_ID", "").isdigit(), "current exact owned CPU review step")
require(os.getresuid() == (2618919, 2618919, 2618919), "actual reviewer UID values")
freeze_path = P / "root_transport_source_freeze.json"
require(ref(freeze_path)["sha256"] == "74a5db144cfbc06596941ded09329056dda3f16cdcaec0bb6aa1017582f98cbc", "actual held root transport freeze")
freeze = json.loads(freeze_path.read_text())
require(freeze["actual_inner_publication"] == PUB and freeze["actual_outer_publication"] is None and len(freeze["files"]) == 9, "real inner publication and no invented future outer hash")
for name, expected in freeze["files"].items():
    require(ref(name)["sha256"] == expected["sha256"] and oct(stat.S_IMODE(Path(name).stat().st_mode)) == expected["mode"], "exact frozen transport bytes/mode: " + Path(name).name)
derivation = json.loads((P / "root_transport_metadata_derivation.json").read_text())
old_ref = derivation["original_corrected_source"]
require(ref(old_ref["path"]) == old_ref and old_ref["sha256"] == "060ee49018531e797d9a7dd23bf8ab9f49978de48a713b3956f1923289980fdc", "actual corrected source reference, not the rejected template")
old_source = Path(old_ref["path"]).read_text()
new_source = (P / "root_start_controller_once.py").read_text()
require(old_source.count("d3208df3e8588b29544835fc6e6d242c4be44153") == 1 and new_source == old_source.replace("d3208df3e8588b29544835fc6e6d242c4be44153", PUB), "whole source equals corrected source after exactly one real publication literal substitution")
old_functions = {node.name: node for node in ast.parse(old_source).body if isinstance(node, ast.FunctionDef)}
new_functions = {node.name: node for node in ast.parse(new_source).body if isinstance(node, ast.FunctionDef)}
require(set(old_functions) == set(new_functions) and {name for name in old_functions if ast.dump(old_functions[name], include_attributes=False) != ast.dump(new_functions[name], include_attributes=False)} == {"main"}, "all process/barrier/custody/write function bodies unchanged; only main publication literal differs")

# Inspect exact clean metadata/producer argv, one direct route and genuine inner publication bytes.
command_record = json.loads((P / "actual_launch_command_preparation.json").read_text())
expected_command = ["/usr/bin/env", "-i", "PATH=/usr/local/bin:/usr/bin:/bin", "LANG=C.UTF-8", "TZ=UTC", "CUDA_VISIBLE_DEVICES=GPU-fed69954-5a71-dbb2-ee91-544e0eb6b8a3", "PYTHONDONTWRITEBYTECODE=1", "PYTHONNOUSERSITE=1", "/usr/bin/python3", "-I", "-S", "-B", str(P / "direct_qualification_controller.py"), "--launch", "--publication-commit", PUB]
require(command_record["command"] == expected_command and command_record["publication"] == PUB and command_record["automatic_retry_permitted"] is False, "exact clean one-shot controller argv and actual inner publication")
require(command_record["producer_command"] == ["/p/rlprojects/RND/.venvs/atari_pixels_jax/bin/python", str(U / "code/measure_selected.py"), "--prepared-unit", str(U)], "unchanged private numerical producer command")
require(command_record["controller_review"] == ref(P / "controller_root_independent_review.json") and command_record["plan"] == ref(U / "measurement_plan.json"), "transport joins actual fresh independent review and plan")
for value in [command_record["controller_review"], command_record["controller_source_freeze"], command_record["packet_freeze"], command_record["plan"]]:
    path = Path(value["path"])
    require(ref(path) == value, "unchanged actual inner publication input")
    committed = subprocess.check_output(["git", "-C", str(I), "cat-file", "blob", PUB + ":" + str(path.relative_to(I))], timeout=30)
    require(hashlib.sha256(committed).hexdigest() == value["sha256"], "actual inner publication contains exact reviewed bytes")
logs = R / "slurm/direct_jobs" / P.name
require(command_record["stdout"] == str(logs / "controller.stdout") and command_record["stderr"] == str(logs / "controller.stderr"), "new exact unit controller logs")
require(not Path(command_record["stdout"]).exists() and not Path(command_record["stderr"]).exists(), "no earlier controller log output or attempted launch")
proposal = json.loads((P / "proposed_root_once_ssh_command.json").read_text())
argv = proposal["one_shot_ssh_argv"]
require(argv[:8] == ["ssh", "-o", "BatchMode=yes", "-o", "ControlMaster=no", "-o", "ConnectTimeout=15", "serval02"] and len(argv) == 9 and "ProxyCommand" not in " ".join(argv), "one normal direct serval02 route without external-cluster proxy override")
expected_remote = expected_command[:12] + [str(P / "root_start_controller_once.py")]
require(shlex.split(argv[8]) == expected_remote, "outer root starter also uses the explicit fixed clean metadata environment")
require(proposal["executed"] is False and proposal["actual_outer_publication"] is None and proposal["automatic_retry_permitted"] is False, "prospective transport has no fake execution/publication or retry")
for name in ("manifest.yaml", "config_resolved.yaml", "experiment_background.md", "command.txt"):
    require(str(logs / name) in freeze["files"], "standard direct transport record: " + name)
require(shlex.split((logs / "command.txt").read_text()) == argv, "standard recorded command is exact one-shot argv")

# Inspect only same-owner metadata protection/restoration; do not change or acquire any lock.
procedure = json.loads((P / "root_owned_metadata_permission_procedure.json").read_text())
require(procedure["mode_now"] == 0o660 and procedure["protected_mode_before_ssh"] == 0o640 and procedure["restore_mode_after_actual_registration_or_verified_failed_startup_exit"] == 0o660, "exact planned owner metadata permission transition")
require(procedure["permission_changes_by_this_preparation"] == procedure["standing_gpu_lock_operations"] == procedure["remote_actions"] == 0, "no metadata protection, standing descriptor or remote action has been performed")
require({row["path"] for row in procedure["current_metadata_lock_snapshots"]} == {str(R / "queue/metadata.lock"), str(R / "processes.lock")}, "only the two exact owned metadata files can change permissions")
cpu_permission_snapshots = []
for row in procedure["current_metadata_lock_snapshots"]:
    path = Path(row["path"])
    info = path.lstat()
    require(stat.S_ISREG(info.st_mode) and info.st_uid == row["st_uid"] == 2618919 and info.st_nlink == row["st_nlink"] == 1 and info.st_size == row["size"] and stat.S_IMODE(info.st_mode) == row["current_mode"] == 0o660 and ref(path)["sha256"] == row["sha256"], "actual owned metadata path, owner, mode and bytes; host-local device/inode is not compared across hosts")
    cpu_permission_snapshots.append({"path": str(path), "hostname": socket.gethostname(), "st_dev": info.st_dev, "st_ino": info.st_ino, "recorded_preparation_identity": {"st_dev": row["st_dev"], "st_ino": row["st_ino"]}})
require(not any((P / name).exists() for name in ("root_remote_launch_intent.json", "root_actual_controller_spawn.json", "root_transport_popen_attempt.json", "root_transport_failure.json")) and not (U / "direct_controller_intent.json").exists(), "fresh root/metadata/numerical launch remains unattempted")
for name, expected in freeze["files"].items():
    require(ref(name)["sha256"] == expected["sha256"], "held transport input unchanged after review")
review = {"schema_version": 1, "kind": "independent_actual_inner_publication_root_transport_literal_review", "passed": True, "approved": True,
          "independent_of_author": True, "source_freeze": ref(freeze_path), "starter": ref(P / "root_start_controller_once.py"),
          "actual_inner_publication": PUB, "actual_outer_publication": None, "proposed_ssh_argv": argv, "outer_clean_environment_reviewed": True,
          "whole_source_changed_only_one_inner_publication_literal": True, "all_process_barrier_custody_functions_unchanged": True,
          "previous_actual_causal_cases_repeated": False, "actual_owned_cpu_job_id": "6821437", "actual_cpu_step_id": os.environ["SLURM_STEP_ID"],
          "actual_cpu_process": {"pid": os.getpid(), "hostname": socket.gethostname(), "uids": list(os.getresuid()), "birth": Path("/proc/self/stat").read_text().rsplit(")", 1)[1].split()[19], "cgroup": Path("/proc/self/cgroup").read_text()},
          "metadata_permissions_changed_by_reviewer": False, "actual_target_or_gpu_or_remote_action": False, "scientific_credit": 0,
          "required_before_once_ssh": ["Publish exact held transport, standard records and this review as a real outer publication.", "Root verifies exact published outer bytes and a successful publication-aware inner default check.", "Root freshly verifies/protects only the two owned metadata files and records actual changes before one SSH.", "Preserve actual Popen PID/birth and descriptor ordering; never retry an attempted packet.", "Restore only same owned metadata inode/device/bytes after actual registration or verified failed startup exit; root/controller never open GPU standing descriptor."],
          "cpu_permission_snapshots": cpu_permission_snapshots, "cross_host_device_inode_equality_claimed": False, "prior_reviewer_failure_preserved": str(P / "independent_reviewer_cross_host_device_comparison_failure"), "reviewer_correction": "The first reviewer attempt incorrectly compared filesystem device numbers across hosts. The source preparation records device 76; owned CPU slurm2 observes device 61, with identical path, owner, mode, size, link count, inode number and bytes. Root must verify device/inode within its own before/after observations.", "checks": CHECKS, "elapsed_seconds": time.monotonic() - START}
out = P / "root_transport_independent_literal_review.json"
with out.open("x") as handle:
    json.dump(review, handle, indent=2, sort_keys=True)
    handle.write("\n")
    handle.flush()
    os.fsync(handle.fileno())
print(json.dumps({"passed": True, "review": ref(out), "actual_cpu_step_id": os.environ["SLURM_STEP_ID"], "checks": len(CHECKS), "elapsed_seconds": review["elapsed_seconds"]}))
