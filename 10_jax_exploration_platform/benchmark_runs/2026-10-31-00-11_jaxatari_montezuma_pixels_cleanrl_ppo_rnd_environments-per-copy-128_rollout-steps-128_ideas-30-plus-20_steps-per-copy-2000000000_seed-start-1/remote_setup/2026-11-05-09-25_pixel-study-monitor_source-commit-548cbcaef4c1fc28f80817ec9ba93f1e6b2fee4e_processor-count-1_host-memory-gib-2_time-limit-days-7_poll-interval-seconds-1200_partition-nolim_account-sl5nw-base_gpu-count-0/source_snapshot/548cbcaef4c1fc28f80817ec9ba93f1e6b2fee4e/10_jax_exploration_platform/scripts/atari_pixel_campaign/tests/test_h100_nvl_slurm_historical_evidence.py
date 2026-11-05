"""Exercise historical H100 evidence with real temporary Git roots and no numerical imports."""

import ast
import copy
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch


SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS))
import h100_nvl_slurm_evidence as history
import h100_nvl_slurm_case as case_runner
import h100_nvl_slurm_queue as queue
import h100_nvl_slurm_site_validation as site
import post_comparison_validation as post
from portable_evidence import ExactPathResolver


class HistoricalEvidenceTests(unittest.TestCase):
    """Keep all observed hardware and process records explicitly synthetic and temporary."""

    @classmethod
    def setUpClass(cls):
        """Cache committed D28 source bytes once; never import or change that numerical source."""
        project = SCRIPTS.parents[2]
        scopes = ["10_jax_exploration_platform/src/exploration_platform/__init__.py",
            "10_jax_exploration_platform/src/exploration_platform/atari_pixels", history.PREFIX,
            "10_jax_exploration_platform/tests/atari_pixels", "10_jax_exploration_platform/tests/test_atari_pixel_environment.py"]
        names = subprocess.check_output(["git", "-C", str(project), "ls-tree", "-r", "--name-only", site.SOURCE_COMMIT, "--", *scopes], text=True).splitlines()
        cls.source_bytes = {name: subprocess.check_output(["git", "-C", str(project), "show", site.SOURCE_COMMIT + ":" + name])
            for name in names if name.endswith((".py", "worker_environment.sh"))}

    def write(self, path, value):
        """Write one synthetic raw record while retaining its intentional original serialization."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(value if isinstance(value, bytes) else (json.dumps(value, indent=2) + "\n").encode())
        return self.receipt(path)

    def receipt(self, path):
        """Describe exact temporary bytes without resolving an original logical path elsewhere."""
        return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}

    def repository(self, name, files):
        """Build an isolated clean detached Git source with actual committed byte identities."""
        root = self.base / name
        root.mkdir()
        for relative, contents in files.items():
            self.write(root / relative, contents)
        subprocess.run(["git", "init", "-q", str(root)], check=True)
        subprocess.run(["git", "-C", str(root), "add", "."], check=True)
        subprocess.run(["git", "-C", str(root), "-c", "user.name=Synthetic metadata test", "-c", "user.email=test@example.invalid",
            "-c", "core.hooksPath=/dev/null", "commit", "-qm", "Create synthetic metadata fixture"], check=True)
        commit = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
        subprocess.run(["git", "-C", str(root), "checkout", "-q", "--detach"], check=True)
        self.git_roots[str(root)] = commit
        return root, commit

    def git_receipt(self, root, commit, names):
        """Record every selected committed fixture file's original SHA256."""
        return {"source_root": str(root), "source_commit": commit,
            "files": {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in names}}

    def setUp(self):
        """Construct a complete synthetic passed site, policy, current allocation and completed case."""
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name).resolve()
        self.git_roots = {}
        self.source, self.commit = self.repository("numerical_source", self.source_bytes)
        helpers = {history.PREFIX + name: ("# synthetic pinned helper " + name + "\n").encode() for name in queue.HELPERS}
        self.helper_root, helper_commit = self.repository("metadata_source", helpers)
        self.helper_receipt = self.git_receipt(self.helper_root, helper_commit, helpers)
        self.site_helpers = self.git_receipt(self.helper_root, helper_commit, [history.PREFIX + name for name in site.HELPERS])
        self.cleanrl, cleanrl_commit = self.repository("original_cleanrl", {"cleanrl/ppo_rnd_envpool.py": b"# synthetic extracted definitions\n"})
        self.jaxtari, self.jaxtari_commit = self.repository("jaxtari_source", {"src/jaxtari/__init__.py": b"# synthetic installed package\n"})
        self.constants = types.SimpleNamespace(**{name: getattr(site, name) for name in dir(site) if not name.startswith("__")})
        self.constants.SOURCE_COMMIT = self.commit
        self.constants.CLEANRL_COMMIT = cleanrl_commit
        self.software_proof = {"scope": "synthetic previously validated formal packet"}
        self.original_load = history.load
        self.addCleanup(patch.stopall)
        patch.object(history, "JAXTARI_COMMIT", self.jaxtari_commit).start()
        # The existing formal-fourth validator has its own tests; this fixture replaces only that prior-study boundary.
        self.formal_validation = patch.object(post, "validate_cohort_case", return_value=self.software_proof).start()
        patch.object(history, "load", side_effect=lambda name: post if name == "post_comparison_validation" else
            self.constants if name == "h100_nvl_slurm_site_validation" else self.original_load(name)).start()
        self.run = self.base / "campaign"
        self.run.mkdir()
        self.numerical = self.git_receipt(self.source, self.commit, self.source_bytes)
        self.config = history.configuration(self.numerical, post.Evidence())
        self.python = "/synthetic/shared/bin/python"
        self.gpu = "GPU-synthetic-owned"
        self.make_software()
        self.make_admission()
        first = self.make_site(self.run / "remote_setup/passed_site", "111")
        current = self.make_site(self.run / "remote_setup/current_allocation", "333")
        self.policy_path = self.run / "reviewed_policy.json"
        self.policy = {"schema_version": 1, "kind": "h100_nvl_slurm_placement_policy", "scope": "exact_passed_site",
            "decision": "approved", "review_rationale": "Synthetic metadata fixture only", "reviewed_at_utc": "2026-10-31T00:00:00+00:00",
            "runtime_environment": site.RUNTIME, "source_commit": self.commit, "ordinary_attempt_limit": 3,
            "scientific_hypothesis_credit": 0, "case_time_budget_seconds": 3600, "shutdown_reserve_seconds": 120,
            "campaign_run_dir": str(self.run), "site_directory": str(first), "site_result": self.receipt(first / "result.json"),
            "approved_targets": [{"hostname": "serval06", "gpu_uuid": self.gpu}], "admission": self.admission,
            "worker_helpers": self.helper_receipt}
        self.write(self.policy_path, self.policy)
        self.make_case(current)

    def make_software(self):
        """Create actual small image/import bytes plus Git-backed package and asset receipts."""
        packages = {name: "synthetic-version" for name in site.PACKAGE_NAMES}
        packages["jaxtari"] = "0.2.0"
        init = (self.jaxtari / "src/jaxtari/__init__.py").read_bytes()
        blob = hashlib.sha1(b"blob " + str(len(init)).encode() + b"\0" + init).hexdigest()
        python_files = {"jaxtari/__init__.py": {"git_blob": blob, "sha256": hashlib.sha256(init).hexdigest()}}
        source = {"source_root": str(self.jaxtari), "source_commit": self.jaxtari_commit, "version": "0.2.0",
            "python_files": python_files, "tree_digest": post.canonical_digest(python_files)}
        libraries = {}
        for name, package in (("libcudnn.so.9", "nvidia-cudnn-cu12"), ("libcudart.so.12", "nvidia-cuda-runtime-cu12")):
            path = self.base / "installed_gpu_images" / name
            self.write(path, ("synthetic historical image " + name).encode())
            libraries[str(path)] = {"package": package, "version": packages[package], "sha256": self.receipt(path)["sha256"]}
        self.installed = {"packages": packages, "jaxtari_source": source, "gpu_libraries": libraries}
        manifest = {"schema_version": 1, "source_commit": self.jaxtari_commit, "files": {"game/pixels.dat": "a" * 64}}
        manifest_path = self.base / "asset_manifest.json"
        self.write(manifest_path, manifest)
        self.asset_record = {"root": str(self.base / "unread_asset_tree"), "manifest": str(manifest_path), "sha256": self.receipt(manifest_path)["sha256"]}
        self.asset_receipt = {"root": self.asset_record["root"], "manifest_sha256": self.asset_record["sha256"],
            "source_commit": self.jaxtari_commit, "file_count": 1, "tree_digest": post.canonical_digest(manifest["files"])}
        packet = self.run / "prior_formal_packet"
        raw = {"attempt": 4, "case_completed": True, "correctness_passed": True, "measurement_valid": True}
        self.write(packet / "result.json", raw)
        self.write(packet / "case.json", {"hypothesis": "prior synthetic mechanism", "hardware_class": "jaguar03_rtx_a4500",
            "configuration": {"n_copies": 2}, "source_commit": self.commit})
        self.write(packet / "installed_backend.json", self.installed)
        self.write(packet / "asset_receipt.json", self.asset_receipt)
        old = self.run / "old_h100_observation"
        self.write(old / "hardware.json", {"hostname": "serval02", "gpu_query": site.REFERENCE_GPU + ", " + site.MODEL + ", 95830 MiB, " + site.DRIVER,
            "runtime_environment": {**site.RUNTIME, "XLA_FLAGS": " " + site.RUNTIME["XLA_FLAGS"]}})
        self.api = {"platform": "gpu", "platform_version": "PJRT C API\ncuda 12090", "cudnnGetVersion": 92400, "cudaRuntimeGetVersion": 12090}
        self.write(old / "loaded_gpu_libraries.json", self.api)
        self.write(old / "diagnostic_plan.json", {"core": self.numerical})
        self.write(old / "dtypes.json", site.EXPECTED_DTYPES)
        self.reference = {"hardware": self.receipt(old / "hardware.json"), "loaded_libraries": self.receipt(old / "loaded_gpu_libraries.json"),
            "diagnostic_plan": self.receipt(old / "diagnostic_plan.json"), "dtype_record": self.receipt(old / "dtypes.json"),
            "model": site.MODEL, "memory_mib": site.MEMORY_MIB, "driver": site.DRIVER, "backend": self.api,
            "shared_installed_backend": self.receipt(packet / "installed_backend.json"), "shared_asset_receipt": self.receipt(packet / "asset_receipt.json"),
            "software": {"result": self.receipt(packet / "result.json"), "installed_backend": self.receipt(packet / "installed_backend.json"),
                "asset_receipt": self.receipt(packet / "asset_receipt.json"), "packages": packages, "runtime_provenance": self.software_proof}}
        self.loaded = {**self.api, "verified_package_libraries": libraries, "loaded_shared_libraries": [*libraries, "/usr/lib/libcuda.so." + site.DRIVER],
            "cudnn_library": next(path for path in libraries if "libcudnn.so" in path), "cuda_runtime_library": next(path for path in libraries if "libcudart.so" in path)}
        package = self.base / "installed_python/jaxtari"
        self.write(package / "__init__.py", init)
        self.import_record = {"schema_version": 1, "kind": "verified_installed_jaxtari_import", "distribution_name": "jaxtari",
            "distribution_version": "0.2.0", "package_root": str(package), "module_file": self.receipt(package / "__init__.py"),
            "source_commit": self.jaxtari_commit, "source_tree_digest": source["tree_digest"]}

    def make_admission(self):
        """Capture original commands under the real snapshot layout and create four mutable admission originals."""
        snapshot = self.run / "data/code_snapshots" / self.commit
        for relative, contents in self.source_bytes.items():
            self.write(snapshot / relative, contents)
        self.gates = {}
        checks = {}
        for context in ("h100", "jaguar03_rtx_a4500"):
            records = {}
            for kind in ("learner", "renderer"):
                command = [self.python, "-m", "pytest", "-q", str(snapshot / "10_jax_exploration_platform/tests/atari_pixels"),
                    str(snapshot / "10_jax_exploration_platform/tests/test_atari_pixel_environment.py")] if kind == "learner" else [self.python, str(snapshot / history.PREFIX / "check_rendering.py"), "--candidate", "all"]
                hardware = {"hostname": "serval02" if context == "h100" else "jaguar03", "gpu_query": "H100" if context == "h100" else "A4500", "runtime_environment": site.RUNTIME}
                record = {"passed": True, "exit_code": 0, "source_commit": self.commit, "command": command, "hardware": hardware}
                path = self.run / "raw_original_checks" / (context + "_" + kind + ".json")
                records[kind] = self.write(path, record)
                if context == "h100":
                    checks[kind] = {**records[kind], "command": command, "hardware": hardware}
            path = self.run / "source_gates" / (context + ".json")
            self.gates[context] = self.write(path, {"passed": True, "source_commit": self.commit, "hardware_class": context, "checks": records})
        config_path = self.run / "reference_configuration.json"
        self.write(config_path, self.config["learner"])
        ready = {"authorized": True, "source_root": str(self.source), "source_commit": self.commit,
            "contexts": ["h100", "jaguar03_rtx_a4500"], "matched_copy_counts": [1, 2], "gate_records": {name: row["path"] for name, row in self.gates.items()},
            "reference_configuration_path": str(config_path)}
        self.write(self.run / "SOURCE_AND_GATES_READY.json", ready)
        files = [self.receipt(self.run / "SOURCE_AND_GATES_READY.json"), self.receipt(config_path), *self.gates.values()]
        self.admission = {"source_root": str(self.source), "source_commit": self.commit, "files": files,
            "reference_configuration": self.receipt(config_path), "original_h100_checks": checks}

    def make_site(self, directory, job):
        """Build a complete synthetic three-child site packet with immutable admission copies."""
        clean = self.cleanrl / "cleanrl/ppo_rnd_envpool.py"
        contents = clean.read_bytes()
        cleanrl = {"source_root": str(self.cleanrl), "source_commit": self.constants.CLEANRL_COMMIT, "file": self.receipt(clean),
            "git_blob": hashlib.sha1(b"blob " + str(len(contents)).encode() + b"\0" + contents).hexdigest()}
        plan = {"schema_version": 2, "kind": "h100_nvl_slurm_site_validation", "source_commit": self.commit, "campaign_run_dir": str(self.run),
            "helpers": self.site_helpers, "numerical_source": self.numerical, "admission": self.admission, "configuration": self.config,
            "runtime_environment": site.RUNTIME, "job_shape": site.JOB_SHAPE, "reference_updates": 1, "submission_control": site.SUBMISSION_CONTROL,
            "gate_environment_policy": site.GATE_ENVIRONMENT_POLICY, "workers_permitted": False, "campaign_claims_permitted": False,
            "scientific_hypothesis_credit": 0, "cleanrl_reference": cleanrl, "reference": self.reference,
            "packages": self.installed["packages"], "fresh_packages_at_prepare": self.installed["packages"], "asset_record": self.asset_record,
            "jaxtari_source_root": str(self.jaxtari), "python": self.python}
        copied = []
        for index, original in enumerate(self.admission["files"]):
            path = directory / "admission_files" / ("record-" + str(index + 1) + "_" + Path(original["path"]).name)
            copied.append({"original": original, "captured": self.write(path, Path(original["path"]).read_bytes())})
        plan["admission_archive"] = {"schema_version": 1, "kind": "original_source_admission_bytes", "files": copied}
        self.write(directory / "site_validation_plan.json", plan)
        self.write(directory / "job.sbatch", b"# synthetic held script, never executed\n")
        self.write(directory / "manifest.yaml", {"plan": self.receipt(directory / "site_validation_plan.json"),
            "job_script": self.receipt(directory / "job.sbatch"), "helpers": self.site_helpers})
        environment = {"SLURM_JOB_ID": job, "SLURM_JOB_GPUS": "7", "SLURM_STEP_GPUS": "7", "CUDA_VISIBLE_DEVICES": "0"}
        csv = "0, " + self.gpu + ", " + site.MODEL + ", 95830, " + site.DRIVER + "\n"
        xml = "<nvidia_smi_log><gpu><uuid>" + self.gpu + "</uuid><product_name>" + site.MODEL + "</product_name><minor_number>7</minor_number></gpu></nvidia_smi_log>"
        proof = site.resolve_allocation(environment, "serval06", csv, xml)
        cpu = queue.load("infrastructure_retry").cpu_allocation({"SLURM_CPUS_PER_TASK": "8", "SLURM_CPU_BIND_TYPE": "threads"}, [self.gpu], range(8))
        assignment = {"allocation": proof, "runtime_environment": site.RUNTIME, "cpu_allocation": cpu, "cpu_affinity": list(range(8)),
            "parent_pid": 90, "parent_start_time": "99", "owned_job": {"job_id": job, "ledger_path": str(self.run / "slurm/submitted_jobids_uva.txt"),
                "ledger_prefix": [job], "cgroup": "0::/slurm/job_" + job + "/step_0/task_0\n"},
            "gpu_modes": {"gpu_uuid": self.gpu, "mig_mode": "Disabled", "memory_used_mib": 0, "compute_processes": [], "confidential_compute": "OFF"}}
        self.write(directory / "site_assignment.json", assignment)
        children = {}
        for index, kind in enumerate(("learner", "renderer", "probe")):
            owner = {"pid": 100 + index, "process_start_time": str(200 + index), "parent_pid": 90, "parent_start_time": "99", "kind": kind, "job_id": job}
            process = {**owner, "gpu_uuid": self.gpu, "runtime_environment": site.RUNTIME, "cpu_affinity": list(range(8)), "source_commit": self.commit,
                "configuration": self.config, "cleanrl_reference": cleanrl, "gate_environment_policy": site.GATE_ENVIRONMENT_POLICY,
                "installed_backend": self.installed, "asset_receipt": self.asset_receipt,
                "original_command": site.original_command(plan, kind) if kind != "probe" else None}
            self.write(directory / (kind + "_owner.json"), owner)
            self.write(directory / (kind + "_process.json"), process)
            log = self.write(directory / (kind + ".log"), b"synthetic successful full check\n")
            self.write(directory / (kind + "_terminal.json"), {**owner, "exit_code": 0, "log": log})
            children[kind] = {name: self.receipt(directory / (kind + suffix)) for name, suffix in (("owner", "_owner.json"), ("process", "_process.json"), ("terminal", "_terminal.json"))}
        paths = {"exploration_platform": str(self.source / "10_jax_exploration_platform/src/exploration_platform/__init__.py")}
        paths.update({"exploration_platform.atari_pixels." + name: str(self.source / "10_jax_exploration_platform/src/exploration_platform/atari_pixels" / (name + ".py")) for name in ("config", "environment", "training", "networks", "state")})
        probe = {"completed": True, "kind": "original_reference_site_probe", "iteration": 1, "dtypes": site.EXPECTED_DTYPES,
            "finite_state": {"passed": True, "floating_leaves_checked": 50, "scope": "complete updated PixelTrainState"},
            "pid": 102, "process_start_time": "202", "gpu_uuid": self.gpu, "source_commit": self.commit,
            "formal_measurement": False, "capacity_evidence": False, "scientific_hypothesis_credit": 0,
            "loaded_libraries": {**self.loaded, "pid": 102}, "numerical_module_paths": paths}
        self.write(directory / "probe_result.json", probe)
        self.write(directory / "result.json", {"schema_version": 1, "kind": "h100_nvl_slurm_site_evidence", "passed": True,
            "workers_permitted": False, "campaign_claims_permitted": False, "scientific_hypothesis_credit": 0,
            "plan": self.receipt(directory / "site_validation_plan.json"), "assignment": self.receipt(directory / "site_assignment.json"),
            "probe": self.receipt(directory / "probe_result.json"), "children": children})
        return directory

    def make_case(self, allocation):
        """Produce all unchanged staged/canonical result bytes, leases and successful parent terminal evidence."""
        self.case = {"unit_id": "synthetic_original_h100_case", "campaign_run_dir": str(self.run), "hardware_class": "h100",
            "source_root": str(self.source), "source_commit": self.commit, "maximum_attempts": 3, "hypothesis": "synthetic mechanism",
            "configuration": self.config["learner"], "candidate": {"learner": {}, "environment": {"cache_room_backgrounds": True}, "execution": {}},
            "reference_execution": {}, "rounds": 7, "iterations": 1, "seed_start": 1}
        self.case["specification_digest"] = post.canonical_digest(self.case)
        self.unit = self.run / "data" / self.case["unit_id"]
        self.attempt = self.unit / "attempts/attempt-0001"
        self.lease = {"execution_kind": history.EXECUTION_KIND, "attempt": 1, "token": "synthetic-token", "unit_id": self.case["unit_id"],
            "runner_pid": 303, "runner_start_time": "403", "worker_pid": 300, "worker_start_time": "400", "slurm_job_id": "333",
            "gpu_uuid": self.gpu, "cpu_affinity": list(range(8)), "placement_policy": self.receipt(self.policy_path),
            "allocation_validation": self.receipt(allocation / "result.json"), "state": "completed", "runner_exit_code": 0}
        self.write(self.attempt / "lease.json", self.lease)
        self.write(self.run / "queue/leases" / (self.case["unit_id"] + ".json"), self.lease)
        self.write(self.attempt / "slurm_released_lease.json", {**self.lease, "state": "running"})
        self.write(self.attempt / "case.json", self.case)
        self.write(self.attempt / "hardware.json", {"gpu_query": self.gpu + ", " + site.MODEL + ", 95830 MiB, " + site.DRIVER,
            "hostname": "serval06", "pid": 303, "process_start_time": "403", "gpu_uuid": self.gpu,
            "slurm_job_id": "333", "source_root": str(self.source), "source_commit": self.commit})
        owner = {"pid": 303, "process_start_time": "403", "parent_pid": 300, "parent_start_time": "400", "job_id": "333"}
        self.write(self.attempt / "slurm_child_owner.json", owner)
        log = self.write(self.attempt / "runner.log", b"synthetic original numerical output\n")
        terminal = {"owner": self.receipt(self.attempt / "slurm_child_owner.json"), "exit_code": 0, "log": log,
            "absent_files": [str(self.attempt / "failure.json")]}
        self.write(self.attempt / "slurm_child_terminal.json", terminal)
        environment, execution, _ = post.case_source_defaults(self.case, post.Evidence())
        reference = {"learner": self.case["configuration"], "environment": environment, "execution": execution}
        candidate = copy.deepcopy(reference)
        candidate["environment"]["cache_room_backgrounds"] = True
        report = {"leaves": 10}
        correctness = {"passed": True, "complete_iteration": report, "supplied_inputs": {name: report for name in ("preparation", "fixed_batch", "fixed_batch_update")}}
        row = {"iterations": 1, "copies": 1, "environments_per_copy": 128, "rollout_steps": 128, "training_transitions": 16384, "seconds": 1.0,
            "memory": {"gpu_uuid": self.gpu, "other_gpu_processes": [], "sampled_process_peak_bytes": 2, "allocator": {"peak_bytes_in_use": 1}}}
        self.raw_result = {"case_completed": True, "idea": self.case["hypothesis"], "specification_digest": self.case["specification_digest"],
            "attempt": 1, "unit_id": self.case["unit_id"], "source_commit": self.commit, "hardware_class": "h100", "copies": 1,
            "decision": "inconclusive", "correctness_attempted": True, "correctness_passed": True, "correctness": correctness,
            "measurement_valid": True, "contaminated": False, "reference": reference, "candidate": candidate, "implementation_distinct": True,
            "rounds": [{"round": index + 1, "order": ["reference", "candidate"] if index % 2 == 0 else ["candidate", "reference"], "reference": row, "candidate": row} for index in range(7)]}
        progress = {"state": "completed", "case_completed": True, "attempt": 1, "unit_id": self.case["unit_id"], "pid": 303, "decision": "inconclusive"}
        files = {}
        for name in history.TERMINAL_NAMES:
            files[name] = self.write(self.attempt / "slurm_terminal_staging" / (name + ".json"), self.raw_result if name.endswith("result") else progress)
        paths = {"attempt_result": self.attempt / "result.json", "unit_result": self.unit / "result.json",
            "attempt_progress": self.attempt / "progress.json", "unit_progress": self.unit / "progress.json"}
        for name, path in paths.items():
            self.write(path, Path(files[name]["path"]).read_bytes())
        module_paths = {name: str(self.source / history.PREFIX / (name + ".py")) for name in ("run_case", "controller", "ideas", "paired_benchmark")}
        module_paths["jaxtari"] = self.import_record["module_file"]["path"]
        module_paths["exploration_platform"] = str(self.source / "10_jax_exploration_platform/src/exploration_platform/__init__.py")
        module_paths.update({"exploration_platform.atari_pixels." + name: str(self.source / "10_jax_exploration_platform/src/exploration_platform/atari_pixels" / (name + ".py")) for name in ("config", "environment", "training", "networks", "state")})
        self.execution = {"schema_version": 1, "kind": history.EXECUTION_KIND, "released_lease": self.receipt(self.attempt / "slurm_released_lease.json"),
            "placement_policy": self.lease["placement_policy"], "allocation_validation": self.lease["allocation_validation"], "lease_token": self.lease["token"],
            "attempt": 1, "specification_digest": self.case["specification_digest"], "source_commit": self.commit, "pid": 303, "process_start_time": "403",
            "job_id": "333", "gpu_uuid": self.gpu, "runtime_environment": site.RUNTIME, "cpu_affinity": list(range(8)),
            "hardware": self.receipt(self.attempt / "hardware.json"), "installed_backend": self.installed, "asset_receipt": self.asset_receipt,
            "loaded_libraries": {**self.loaded, "pid": 303}, "module_paths": module_paths, "jaxtari_import": self.import_record, "terminal_files": files}
        self.write(self.attempt / "slurm_execution.json", self.execution)
        self.write_intent()

    def write_intent(self):
        """Bind the exact successful parent receipt and four raw hashes into the immutable publication intent."""
        self.write(self.attempt / "slurm_publication_intent.json", {"kind": "h100_nvl_slurm_terminal_publication", "token": self.lease["token"],
            "attempt": 1, "unit_id": self.case["unit_id"], "execution": self.receipt(self.attempt / "slurm_execution.json"),
            "child_terminal": self.receipt(self.attempt / "slurm_child_terminal.json"),
            "terminal_files": {name: row["sha256"] for name, row in self.execution["terminal_files"].items()}})

    def validate(self, evidence=None):
        """Run the complete new historical validator with no current allocation or installed-package lookup."""
        return history.validate_execution(self.attempt, self.case, self.lease, require_completed=True, evidence=evidence)

    def portable_package(self, receipts):
        """Copy exact read bytes and detached Git roots to an explicit portable manifest."""
        root = self.base / "portable"
        root.mkdir()
        files = []
        for index, row in enumerate(receipts):
            relative = "files/" + str(index)
            self.write(root / relative, Path(row["path"]).read_bytes())
            files.append({"original_path": row["path"], "relative_path": relative, "sha256": row["sha256"]})
        roots = []
        for index, (original, commit) in enumerate(self.git_roots.items()):
            target = root / "repositories" / str(index)
            target.parent.mkdir(exist_ok=True)
            subprocess.run(["git", "clone", "-q", "--no-hardlinks", original, str(target)], check=True)
            subprocess.run(["git", "-C", str(target), "checkout", "-q", "--detach", commit], check=True)
            roots.append({"original_root": original, "relative_path": str(target.relative_to(root)), "commit": commit})
        manifest = root / "manifest.json"
        self.write(manifest, {"schema_version": 1, "kind": "portable_raw_evidence", "files": files, "git_roots": roots})
        return manifest

    def test_complete_history_uses_captured_admission_and_never_reader_packages(self):
        """Later live permission changes and an unavailable local JAXtari install cannot change old proof."""
        before = self.case.copy()
        self.write(self.run / "SOURCE_AND_GATES_READY.json", {"authorized": False, "source_commit": "later-revision"})
        for row in self.admission["files"][1:]:
            Path(row["path"]).unlink()
        evidence = post.Evidence()
        with patch.object(importlib.metadata, "distribution", side_effect=AssertionError("reader packages are forbidden")):
            record, result = self.validate(evidence)
        self.assertEqual(result, self.raw_result)
        self.assertEqual(record["jaxtari_import"], self.import_record)
        self.assertEqual(self.case, before)
        self.assertTrue(set(row["path"] for row in self.admission["files"]).isdisjoint(evidence.receipts))

    def test_compute_import_receipt_uses_actual_distribution_module_and_bytes(self):
        """Capture the actual installed import and reject a substituted version before recording proof."""
        root = Path(self.import_record["package_root"])
        distribution = types.SimpleNamespace(version=self.import_record["distribution_version"], locate_file=lambda name: root)
        imported = types.SimpleNamespace(__file__=self.import_record["module_file"]["path"])
        with patch.object(case_runner.metadata, "distribution", return_value=distribution), patch.dict(sys.modules, {"jaxtari": imported}):
            self.assertEqual(case_runner.jaxtari_import_receipt(self.installed), self.import_record)
            distribution.version = "another-version"
            with self.assertRaisesRegex(ValueError, "actual JAXtari import"):
                case_runner.jaxtari_import_receipt(self.installed)

    def test_strict_relocation_works_with_all_original_roots_removed(self):
        """Logical identity is unchanged while every historical file and Git root is read from its exact mapping."""
        evidence = post.Evidence()
        self.validate(evidence)
        manifest = self.portable_package(evidence.records())
        for path in list(self.base.iterdir()):
            if path != manifest.parent:
                shutil.rmtree(path) if path.is_dir() else path.unlink()
        resolver = ExactPathResolver(manifest)
        relocated = post.Evidence(resolver)
        with patch.object(importlib.metadata, "distribution", side_effect=AssertionError("reader packages are forbidden")):
            record, result = self.validate(relocated)
        self.assertEqual(result, self.raw_result)
        self.assertEqual(record["placement_policy"], self.lease["placement_policy"])
        self.assertEqual(evidence.records(), relocated.records())

    def test_changed_source_image_or_import_bytes_reject(self):
        """A saved receipt cannot hide a changed used Git file, GPU image or imported Python initializer."""
        targets = [self.source / "10_jax_exploration_platform/src/exploration_platform/atari_pixels/config.py",
            Path(self.loaded["cudnn_library"]), Path(self.import_record["module_file"]["path"])]
        for target in targets:
            original = target.read_bytes()
            target.write_bytes(original + b"\n")
            with self.subTest(target=target.name), self.assertRaises((ValueError, KeyError)):
                self.validate()
            target.write_bytes(original)

    def test_rehashed_identity_library_and_import_receipt_changes_reject(self):
        """Recomputed transport hashes cannot legitimize another process or imported library/package."""
        for mutate in (lambda value: value.update(pid=999),
                lambda value: value["loaded_libraries"].update(cudnn_library=self.loaded["cuda_runtime_library"]),
                lambda value: value["jaxtari_import"].update(distribution_version="another-version"),
                lambda value: value["module_paths"].update(jaxtari="/other/package/__init__.py")):
            changed = copy.deepcopy(self.execution)
            mutate(changed)
            self.write(self.attempt / "slurm_execution.json", changed)
            self.write_intent()
            with self.assertRaises((ValueError, KeyError)):
                self.validate()
        self.write(self.attempt / "slurm_execution.json", self.execution)
        self.write_intent()

    def test_failed_child_missing_absence_or_failed_lease_reject(self):
        """A successful result cannot override a nonzero child, missing absence record or terminal failure."""
        path = self.attempt / "slurm_child_terminal.json"
        original = json.loads(path.read_bytes())
        for changed in ({**original, "exit_code": 1}, {key: value for key, value in original.items() if key != "absent_files"}):
            self.write(path, changed)
            self.write_intent()
            with self.assertRaises(ValueError):
                self.validate()
        self.write(path, original)
        self.write_intent()
        self.write(self.attempt / "lease.json", {**self.lease, "state": "failed"})
        with self.assertRaises(ValueError):
            self.validate()

    def test_mapped_failure_cannot_be_hidden_by_absence_record(self):
        """A portable failure entry contradicts the exact parent's captured absence even if hashes are valid."""
        evidence = post.Evidence()
        self.validate(evidence)
        self.write(self.attempt / "failure.json", {"error": "synthetic contradictory failure"})
        receipts = evidence.records() + [self.receipt(self.attempt / "failure.json")]
        manifest = self.portable_package(receipts)
        with self.assertRaisesRegex(ValueError, "mapped failure"):
            self.validate(post.Evidence(ExactPathResolver(manifest)))

    def test_gate_command_source_exit_dtype_and_finite_mutations_reject(self):
        """All original full gates and the real reference update remain required after transport rehashing."""
        directory = self.run / "remote_setup/current_allocation"
        result_path = directory / "result.json"
        original_result = json.loads(result_path.read_bytes())
        for section, mutate in (("learner_terminal", lambda value: value.update(exit_code=1)),
                ("learner_process", lambda value: value.update(original_command=[self.python, "-m", "pytest", "-k", "one"])),
                ("probe_result", lambda value: value["dtypes"].update(reward_statistics=["float32"])),
                ("probe_result", lambda value: value.pop("finite_state"))):
            path = directory / (section + ".json")
            original = path.read_bytes()
            value = json.loads(original)
            mutate(value)
            self.write(path, value)
            result = copy.deepcopy(original_result)
            if section == "probe_result":
                result["probe"] = self.receipt(path)
            else:
                kind, field = section.split("_")
                result["children"][kind][field] = self.receipt(path)
            self.write(result_path, result)
            with self.subTest(section=section), self.assertRaises(ValueError):
                history.site_evidence(directory, self.helper_receipt, post.Evidence())
            self.write(path, original)
            self.write(result_path, original_result)

    def test_canonical_publication_or_round_changes_reject(self):
        """A completed lease still needs every original canonical byte and the full paired windows."""
        path = self.attempt / "result.json"
        original = path.read_bytes()
        self.write(path, original + b"\n")
        with self.assertRaisesRegex(ValueError, "canonical publication"):
            self.validate()
        # A retained running-state caller object cannot skip an actually completed lease's checks.
        with self.assertRaisesRegex(ValueError, "canonical publication"):
            history.validate_execution(self.attempt, self.case, {**self.lease, "state": "running"})
        self.write(path, original)
        changed = copy.deepcopy(self.raw_result)
        changed["rounds"] = changed["rounds"][:-1]
        with self.assertRaisesRegex(ValueError, "paired windows"):
            history.scientific_result(changed, self.case, post.Evidence())

    def test_correctness_rejection_retains_original_configuration_checks(self):
        """An actual correctness rejection still identifies the exact original compared implementations."""
        rejected = {**copy.deepcopy(self.raw_result), "decision": "rejected", "failure_stage": "correctness",
            "correctness_passed": False, "error": "synthetic fixed-input mismatch", "measurement_valid": False}
        history.scientific_result(rejected, self.case, post.Evidence())
        rejected["candidate"]["learner"]["learning_rate"] *= 2
        with self.assertRaises(ValueError):
            history.scientific_result(rejected, self.case, post.Evidence())

    def test_rehashed_timing_rows_cannot_name_another_physical_gpu(self):
        """Even consistently replaced result hashes must retain the child's actual selected device."""
        changed = copy.deepcopy(self.raw_result)
        changed["rounds"][0]["candidate"]["memory"]["gpu_uuid"] = "GPU-another-synthetic-device"
        for name, canonical in (("attempt_result", self.attempt / "result.json"), ("unit_result", self.unit / "result.json")):
            staged = Path(self.execution["terminal_files"][name]["path"])
            self.execution["terminal_files"][name] = self.write(staged, changed)
            self.write(canonical, staged.read_bytes())
        self.write(self.attempt / "slurm_execution.json", self.execution)
        self.write_intent()
        with self.assertRaisesRegex(ValueError, "another physical GPU"):
            self.validate()


if __name__ == "__main__":
    unittest.main()
