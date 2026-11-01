"""Check ACES metadata ownership and hardware-only collection using standard-library fixtures."""

import copy
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


SCRIPTS = Path(__file__).resolve().parents[1]
SPECIFICATION = importlib.util.spec_from_file_location("aces_metadata_observation", SCRIPTS / "aces_allocation_observation.py")
observation = importlib.util.module_from_spec(SPECIFICATION)
SPECIFICATION.loader.exec_module(observation)


def write(path, value):
    """Write an explicit fixture with the production finite-byte serialization."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(observation.json_bytes(value))
    return path


class AcesAllocationObservationTests(unittest.TestCase):
    """Never confuse hardware inventory with allocated GPU identity or a qualification result."""

    def setUp(self):
        """Create clean detached helper files, old owned resource evidence and one held observation packet."""
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.run = self.root / "campaign"
        (self.run / "remote_setup").mkdir(parents=True)
        (self.run / "slurm").mkdir()
        self.ledger = self.run / "slurm/submitted_jobids_aces.txt"
        self.ledger.write_text("2271806\n")
        self.submissions = self.run / "slurm/submissions_aces.tsv"
        self.submissions.write_text("2271806\t" + observation.BASIS_UNIT + "\td28e6735e79788f618acb0ebfde8609db1f776af\n")
        self.basis = self.run / "data" / observation.BASIS_UNIT
        write(self.basis / "hardware.json", {"slurm_job_id": "2271806"})
        write(self.basis / "progress.json", {"completed": True, "rounds": 7})
        (self.run / "slurm/aces_reference_d28e673.sbatch").write_text(
            "#!/bin/bash\n#SBATCH --account=158590754315\n#SBATCH --cpus-per-task=8\n#SBATCH --mem=32G\n#SBATCH --gres=gpu:h100:1\n")
        self.helper = self.root / "detached helper"
        for name in observation.HELPERS:
            destination = self.helper / (observation.PREFIX + name)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(SCRIPTS / name, destination)
        subprocess.run(["git", "init", "-q", str(self.helper)], check=True)
        subprocess.run(["git", "-C", str(self.helper), "add", "."], check=True)
        subprocess.run(["git", "-C", str(self.helper), "-c", "core.hooksPath=/dev/null", "-c", "user.name=Metadata tests",
            "-c", "user.email=metadata@example.invalid", "commit", "-qm", "Create metadata helper fixture"], check=True)
        subprocess.run(["git", "-C", str(self.helper), "checkout", "-q", "--detach"], check=True)
        self.owner = patch.object(observation.pwd, "getpwuid", return_value=SimpleNamespace(pw_name=observation.USERNAME))
        self.owner.start()
        self.addCleanup(self.owner.stop)
        self.filename = self.helper / (observation.PREFIX + observation.HELPERS[0])
        self.directory = self.run / "remote_setup/2026-11-01-00-00_aces_allocation-observation_gpu-count-1_processors-8_memory-gib-32"
        with patch.object(observation, "__file__", str(self.filename)):
            self.prepared = observation.prepare(self.directory, self.run, self.helper, Path(sys.executable))
        self.plan = observation.read(self.directory / "allocation_observation_plan.json")
        self.unit = Path(self.plan["data_unit"]["path"])
        self.job = "5000001"
        self.ledger.write_text(self.ledger.read_text() + self.job + "\n")
        row = [self.job, self.directory.name, self.plan["helpers"]["source_commit"], str(self.run)]
        self.submissions.write_text(self.submissions.read_text() + "\t".join(row) + "\n")
        submission = {"schema_version": 1, "kind": "owned_aces_allocation_observation_submission", "cluster": "aces",
            "job_id": self.job, "plan_sha256": observation.file_record(self.directory / "allocation_observation_plan.json")["sha256"],
            "manifest_sha256": observation.file_record(self.directory / "manifest.yaml")["sha256"],
            "job_script_sha256": observation.file_record(self.directory / "job.sbatch")["sha256"],
            "held_command": self.prepared["submission_command"], "data_unit": self.plan["data_unit"],
            "ledger_path": str(self.ledger), "submission_row": row}
        write(self.directory / "submission_receipt.json", submission)
        self.release = {"schema_version": 1, "kind": "owned_aces_allocation_observation_release", "cluster": "aces",
            "job_id": self.job, "command": ["scontrol", "release", self.job], "returncode": 0,
            "submission_receipt_sha256": observation.file_record(self.directory / "submission_receipt.json")["sha256"]}
        write(self.directory / "release_receipt.json", self.release)
        self.environment = {"SLURM_JOB_ID": self.job, "SLURM_STEP_ID": "0", "SLURM_JOB_GPUS": "1",
            "SLURM_STEP_GPUS": "1", "SLURM_GPUS_ON_NODE": "1", "SLURM_CPUS_PER_TASK": "8", "SLURM_LOCALID": "0",
            "CUDA_VISIBLE_DEVICES": "0"}
        self.cgroup = "0::/slurm/uid_1/job_" + self.job + "/step_0/task_0\n"
        uid = str(os.getuid())
        self.job_text = (f"JobId={self.job} JobState=RUNNING UserId={observation.USERNAME}({uid}) Account={observation.ACCOUNT} "
            "Partition=gpu QOS=normal NodeList=ac045 NumNodes=1 NumCPUs=8 JOB_GRES=gpu:h100:1 TresPerNode=gres/gpu:h100:1\n"
            "AllocTRES=cpu=8,mem=32G,node=1,gres/gpu=1,gres/gpu:h100=1\n")
        self.step_text = (f"StepId={self.job}.0 State=RUNNING UserId={uid} NodeList=ac045 Nodes=1 CPUs=8 Tasks=1 "
                          "TRES=cpu=8,mem=32G,gres/gpu=1,gres/gpu:h100=1\n")

    def scheduler(self, command):
        """Return exact owned raw records without contacting a scheduler."""
        kind = command[2]
        return {"command": command, "returncode": 0,
            "stdout": self.job_text if kind == "job" else self.step_text if kind == "step" else "NodeName=ac045 Version=observed-version\n",
            "stderr": ""}

    def owned(self, environment=None, cgroup=None, hostname="ac045"):
        """Run the real ownership/resource checks with only scheduler and CPU-affinity queries replaced."""
        with patch.object(observation, "run_query", side_effect=self.scheduler), \
                patch.object(observation.os, "sched_getaffinity", return_value=set(range(8))):
            return observation.owned_allocation(self.plan, self.directory, environment or self.environment,
                self.cgroup if cgroup is None else cgroup, hostname, release_wait_seconds=0)

    def test_prepare_has_four_standard_records_exact_helpers_and_no_submission(self):
        """A prepare-only packet requests one short metadata task and grants no experiment admission."""
        self.assertFalse(self.prepared["submitted"])
        self.assertEqual(set(self.plan["helpers"]["files"]), {observation.PREFIX + name for name in observation.HELPERS})
        for name in observation.STANDARD_FILES:
            self.assertEqual((self.directory / name).read_bytes(), (self.unit / name).read_bytes())
        for argument in ("--hold", "--account=158590754315", "--gres=gpu:h100:1", "--mem=32G", "--time=00:05:00"):
            self.assertIn(argument, self.prepared["submission_command"])
        script = (self.directory / "job.sbatch").read_text()
        self.assertIn("srun --unbuffered --ntasks=1 --cpus-per-task=8 --cpu-bind=threads", script)
        self.assertIn(" -I ", script)
        self.assertFalse((self.run / "queue").exists())
        self.assertEqual({name: self.plan[name] for name in observation.BOUNDARY}, observation.BOUNDARY)
        self.assertEqual(self.plan["gpu_xml"]["status"], "not_collected")

    def test_prepare_preserves_existing_packets_and_resource_records(self):
        """A second preparation cannot replace a packet or require a non-ACES basis identifier."""
        with patch.object(observation, "__file__", str(self.filename)), self.assertRaisesRegex(ValueError, "both ACES"):
            observation.prepare(self.directory, self.run, self.helper, Path(sys.executable))
        self.ledger.write_text(self.job + "\n")
        (self.run / "slurm/submitted_jobids_uva.txt").write_text("2271806\n")
        with self.assertRaisesRegex(ValueError, "exact ACES ledger"):
            observation.resource_basis(self.run, "2271806")

    def test_both_helpers_require_their_committed_bytes(self):
        """The shared collector dependency cannot change while the new helper remains unchanged."""
        path = self.helper / (observation.PREFIX + observation.HELPERS[1])
        path.write_bytes(path.read_bytes() + b"\n# changed collector\n")
        with self.assertRaisesRegex(ValueError, "differs from committed bytes"):
            observation.helper_source(self.helper)

    def test_owned_numeric_step_does_not_equate_scheduler_gpu_and_cuda_selector(self):
        """Scheduler GPU one and CUDA selector zero remain distinct captured identifiers."""
        result = self.owned()
        self.assertEqual(result["job_id"], self.job)
        self.assertEqual(result["step_id"], "0")
        self.assertFalse(result["mapping_accepted"])
        self.assertNotIn("physical_gpu_uuid", result)
        self.assertEqual(result["allocation_environment"]["SLURM_JOB_GPUS"], "1")
        self.assertEqual(result["allocation_environment"]["CUDA_VISIBLE_DEVICES"], "0")

    def test_unfamiliar_scheduler_fields_keep_raw_owned_outputs_before_rejection(self):
        """A changed Slurm spelling cannot discard the raw own-ID job/step/environment/cgroup proof."""
        self.job_text = self.job_text.replace("JOB_GRES=", "UNFAMILIAR_GRES=")
        with self.assertRaisesRegex(ValueError, "JOB_GRES"):
            self.owned()
        record = observation.read(self.unit / "owned_allocation.json")
        self.assertEqual(record["scheduler"]["job"]["stdout"], self.job_text)
        self.assertEqual(record["scheduler"]["step"]["stdout"], self.step_text)
        self.assertEqual(record["allocation_environment"]["SLURM_STEP_ID"], "0")
        self.assertEqual(record["cgroup"], self.cgroup)
        self.assertGreater(record["pid"], 0)
        self.assertTrue(record["process_start_time"].isdigit())

    def test_foreign_ledger_batch_step_login_host_and_wrong_cgroup_are_rejected(self):
        """No hardware query is permitted outside the exact current owned numerical step."""
        cases = [(dict(self.environment, SLURM_JOB_ID="777"), None, "ac045"),
                 (dict(self.environment, SLURM_STEP_ID="batch"), None, "ac045"),
                 (self.environment, None, "alogin1"), (self.environment, "0::/job_5000001/step_9/task_0", "ac045")]
        for environment, cgroup, hostname in cases:
            with self.subTest(environment=environment, hostname=hostname), self.assertRaisesRegex(ValueError, "exact ACES owner"):
                self.owned(environment, cgroup, hostname)

    def test_submission_and_release_bytes_bind_this_exact_packet(self):
        """Another cluster, changed plan hash or failed release cannot authorize collection."""
        for name, key, value in (("submission_receipt.json", "cluster", "uva"),
                ("submission_receipt.json", "plan_sha256", "a" * 64),
                ("submission_receipt.json", "schema_version", True),
                ("release_receipt.json", "schema_version", True),
                ("release_receipt.json", "returncode", 1),
                ("release_receipt.json", "returncode", False),
                ("release_receipt.json", "submission_receipt_sha256", "b" * 64)):
            path = self.directory / name
            original = path.read_bytes()
            record = json.loads(original)
            record[key] = value
            write(path, record)
            with self.subTest(name=name, key=key), self.assertRaises(ValueError):
                self.owned()
            path.write_bytes(original)
        (self.directory / "release_receipt.json").unlink()
        with self.assertRaises(FileNotFoundError):
            self.owned()

    def test_changed_scheduler_owner_account_gpu_cpu_or_memory_is_rejected(self):
        """Original failed scheduler rows survive but cannot pass the owned-allocation resource check."""
        original = self.job_text
        for old, new in (("Account=158590754315", "Account=another"), ("QOS=normal", "QOS=other"),
                ("NumCPUs=8", "NumCPUs=16"), ("mem=32G", "mem=64G"),
                ("gres/gpu=1", "gres/gpu=2"), ("gres/gpu:h100=1", "gres/gpu:h100=2"),
                ("UserId=" + observation.USERNAME, "UserId=other")):
            self.job_text = original.replace(old, new)
            with self.subTest(field=old), self.assertRaises(ValueError):
                self.owned()
            (self.unit / "owned_allocation.json").unlink()
        self.job_text = original

    def test_actual_cpu_affinity_must_match_the_eight_cpu_task(self):
        """A declared eight-CPU environment cannot replace the actual process affinity."""
        with patch.object(observation.os, "sched_getaffinity", return_value={0, 1}), \
                self.assertRaisesRegex(ValueError, "actual eight-CPU affinity"):
            observation.owned_allocation(self.plan, self.directory, self.environment, self.cgroup, "ac045", release_wait_seconds=0)

    def collect(self, *, minor_supported=True, bad_csv=False):
        """Run the actual collection with explicit hardware rows and unavailable configuration observations."""
        generic = observation.generic_collector()
        owned = {"hostname": "ac045", "job_id": self.job, "step_id": "0", "pid": os.getpid(),
            "process_start_time": "123456", "cgroup": self.cgroup}
        write(self.unit / "owned_allocation.json", owned)
        commands = []
        def query(command):
            """Supply original-style hardware strings while refusing every process or selector query."""
            commands.append(command)
            self.assertNotIn("-i", command)
            self.assertNotIn("-q", command)
            self.assertNotIn("-x", command)
            self.assertFalse(any("process" in value for value in command))
            if "--help-query-gpu" in command:
                output = '"minor_number"\n' if minor_supported else '"uuid"\n'
            else:
                suffix = ", 0" if minor_supported else ""
                output = "0, GPU-first, NVIDIA H100 PCIe, 00000000:C1:00.0, 81559, 580.178.04" + suffix + "\n"
                suffix = ", 1" if minor_supported else ""
                output += "1, GPU-second, NVIDIA H100 PCIe, 00000000:C2:00.0, 81559, 580.178.04" + suffix + "\n"
                if bad_csv: output = "incomplete,row\n"
            return {"command": command, "returncode": 0, "stdout": output, "stderr": ""}
        with patch.object(observation, "generic_collector", return_value=generic), \
                patch.object(observation, "owned_allocation", return_value=owned), \
                patch.object(observation, "run_query", side_effect=query), \
                patch.object(generic, "host_inventory", return_value={"proc_gpu_inventory": {"status": "absent"}, "gpu_gres_mapping": {"status": "absent"}}), \
                patch.object(observation, "configured_observations", return_value={"node_cgroup_configuration": {"status": "permission denied"}, "observed_configuration_directories": []}), \
                patch.dict(os.environ, self.environment):
            result = observation.collect(self.directory, self.plan)
        return result, commands

    def test_collection_preserves_all_hardware_rows_without_selecting_a_card(self):
        """Two visible hardware rows do not change a one-GPU request or create mapping admission."""
        result, commands = self.collect()
        self.assertTrue(result["passed"])
        self.assertEqual(result["meaning_of_passed"], "metadata collection only")
        hardware = observation.read(self.unit / "hardware.json")
        self.assertEqual(hardware["requested_gpu_count"], 1)
        self.assertEqual(len(hardware["observed_hardware_rows"]), 2)
        self.assertFalse(hardware["gpu_uuid_selected"])
        self.assertEqual(hardware["gpu_query"], "")
        self.assertEqual([row["minor_number"] for row in hardware["observed_hardware_rows"]], ["0", "1"])
        record = observation.read(result["observations"]["path"])
        self.assertEqual(record["hardware_queries"]["gpu_xml"]["status"], "not_collected")
        self.assertEqual(record["device_isolation_observations"]["node_cgroup_configuration"]["status"], "permission denied")
        self.assertEqual({name: result[name] for name in observation.BOUNDARY}, observation.BOUNDARY)
        self.assertEqual(len(commands), 2)
        self.assertNotIn("jax", sys.modules)
        self.assertNotIn("numpy", sys.modules)

    def test_absent_minor_field_is_explicit_and_no_query_is_invented(self):
        """An unsupported installed query field stays absent beside its independent proc/sysfs observations."""
        result, commands = self.collect(minor_supported=False)
        record = observation.read(result["observations"]["path"])
        self.assertFalse(record["hardware_queries"]["minor_number_field"]["supported"])
        self.assertNotIn("minor_number", record["hardware_queries"]["gpu_csv"]["fields"])
        self.assertEqual(len(commands), 2)

    def test_partial_csv_is_preserved_and_never_publishes_success(self):
        """Failed collection keeps original hardware responses without a successful terminal record."""
        with self.assertRaisesRegex(ValueError, "field count"):
            self.collect(bad_csv=True)
        self.assertTrue((self.unit / "nvidia_hardware_queries.json").is_file())
        self.assertTrue((self.unit / "hardware_observations.json").is_file())
        self.assertFalse((self.unit / "result.json").exists())

    def test_capture_rejects_changed_standard_records_before_hardware(self):
        """The monitored data shard must retain its original launch-time records exactly."""
        path = self.unit / "command.txt"
        path.write_bytes(path.read_bytes() + b"\n")
        with patch.object(observation, "__file__", str(self.filename)), \
                patch.object(observation, "collect") as collect, self.assertRaisesRegex(ValueError, "data shard preparation differs"):
            observation.capture(self.directory)
        collect.assert_not_called()
        self.assertFalse(observation.read(self.unit / "failure.json")["passed"])

    def test_observed_slurm_configuration_path_is_explicit_and_only_selected_fields_survive(self):
        """An actual reported slurm.conf directory supplies a labeled candidate rather than an assumed /etc path."""
        generic = observation.generic_collector()
        config = self.root / "actual slurm configuration"
        config.mkdir()
        (config / "cgroup.conf").write_text("ConstrainDevices=yes\nCgroupPlugin=cgroup/v2\nUnrelatedSecret=never retain\n")
        response = {"command": ["scontrol", "show", "config"], "returncode": 0,
            "stdout": "TaskPlugin = task/cgroup\nProctrackType = proctrack/cgroup\nSLURM_CONF = " + str(config / "slurm.conf") + "\nUnrelatedSecret=never retain\n", "stderr": ""}
        with patch.object(observation, "run_query", return_value=response), patch.object(generic, "own_placement", return_value={}):
            record = observation.configured_observations(generic, {}, self.cgroup)
        self.assertEqual(record["node_cgroup_configuration"]["file"]["path"], str(config / "cgroup.conf"))
        self.assertEqual(record["node_cgroup_configuration"]["fields"]["ConstrainDevices"], "yes")
        self.assertIn("not independent proof", record["node_cgroup_configuration"]["path_basis"])
        self.assertNotIn("UnrelatedSecret", json.dumps(record))

    def test_capture_rejects_boolean_plan_and_manifest_versions(self):
        """Boolean true cannot substitute for the integer version in either immutable document."""
        for name in ("allocation_observation_plan.json", "manifest.yaml"):
            path = self.directory / name
            original = path.read_bytes()
            record = json.loads(original)
            record["schema_version"] = True
            write(path, record)
            if name == "manifest.yaml":
                (self.unit / name).write_bytes(path.read_bytes())
            with patch.object(observation, "__file__", str(self.filename)), \
                    patch.object(observation, "collect") as collect, self.subTest(document=name), self.assertRaises(ValueError):
                observation.capture(self.directory)
            collect.assert_not_called()
            path.write_bytes(original)
            if name == "manifest.yaml":
                (self.unit / name).write_bytes(original)
            (self.unit / "failure.json").unlink()

    def test_missing_or_ambiguous_configuration_paths_remain_explicit(self):
        """A missing path or conflicting configured directories cannot select one configuration silently."""
        generic = observation.generic_collector()
        response = {"command": ["scontrol", "show", "config"], "returncode": 0,
            "stdout": "TaskPlugin = task/cgroup\n", "stderr": ""}
        with patch.object(observation, "run_query", return_value=response), patch.object(generic, "own_placement", return_value={}):
            missing = observation.configured_observations(generic, {}, self.cgroup)
            response["stdout"] += "SLURM_CONF = /another/slurm.conf\n"
            ambiguous = observation.configured_observations(generic, {"SLURM_CONF": "/observed/slurm.conf"}, self.cgroup)
        self.assertEqual(missing["node_cgroup_configuration"]["status"], "not_located")
        self.assertEqual(ambiguous["node_cgroup_configuration"]["status"], "ambiguous")
        self.assertEqual(len(ambiguous["node_cgroup_configuration_candidates"]), 2)


if __name__ == "__main__":
    unittest.main()
