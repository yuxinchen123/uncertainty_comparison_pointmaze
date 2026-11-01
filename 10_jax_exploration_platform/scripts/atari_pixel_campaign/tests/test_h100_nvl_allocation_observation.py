"""Verify metadata collection boundaries without GPUs, numerical imports or scheduler access."""

import importlib.util
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import h100_nvl_allocation_observation as observation


def write(path, value):
    """Write one explicit synthetic receipt, preserving the same byte-hash contracts as production."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n")
    return observation.file_record(path)


def scheduler_text(identifier="123", count=1):
    """Represent the actual one-card typed Slurm fields, independently of visible device indices."""
    return (f"JobId={identifier} JobState=RUNNING NodeList=serval08 NumNodes=1 NumCPUs=8\n"
        f"AllocTRES=cpu=8,mem=64G,node=1,gres/gpu={count}\n"
        f"JOB_GRES=gpu:nvidia_h100_nvl:{count}\n"
        f" Nodes=serval08 CPU_IDs=0-7 Mem=65536 GRES=gpu:nvidia_h100_nvl:{count}(IDX:1)\n"
        f"TresPerNode=gres/gpu:nvidia_h100_nvl:{count}\n")


class AllocationObservationTests(unittest.TestCase):
    """Keep preparation unsubmitted and reject unowned allocations before hardware reads."""

    def setUp(self):
        """Create a detached helper Git fixture and an explicitly owned resource proof."""
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.run = self.root / "campaign"
        (self.run / "remote_setup").mkdir(parents=True)
        (self.run / "data").mkdir()
        ledger = self.run / "slurm/submitted_jobids_uva.txt"
        ledger.parent.mkdir()
        ledger.write_text("123\n456\n")
        self.helper_root = self.root / "detached helpers"
        helper_path = self.helper_root / observation.RELATIVE_PATH
        helper_path.parent.mkdir(parents=True)
        shutil.copyfile(observation.__file__, helper_path)
        for command in (["init", "-q"], ["config", "user.email", "metadata@example.invalid"],
                        ["config", "user.name", "Metadata tests"], ["add", observation.RELATIVE_PATH],
                        ["commit", "-qm", "Record the metadata helper"], ["checkout", "--detach", "-q"]):
            subprocess.run(["git", "-C", str(self.helper_root), *command], check=True, capture_output=True)
        basis = self.run / "remote_setup/previous failed site"
        plan = write(basis / "site_validation_plan.json", {"campaign_run_dir": str(self.run), "job_shape": {
            "account": "recorded_account", "partition": "gpu", "qos": "normal", "nodes": 1, "ntasks": 1,
            "gres": "gpu:nvidia_h100_nvl:1", "cpus_per_task": 8, "memory": "64G", "time": "03:00:00",
            "eligible_nodes": list(observation.HOSTS)}})
        script = basis / "job.sbatch"
        script.write_text("#!/bin/bash\nexit 1\n")
        self.resource_submission = basis / "submission_receipt.json"
        write(self.resource_submission, {"kind": "owned_h100_nvl_site_validation_submission", "cluster": "uva",
            "job_id": "456", "plan_sha256": plan["sha256"], "job_script_sha256": observation.digest(script),
            "held_command": ["sbatch", "--hold", "--constraint=h100_94gb&amd_epyc_9354", str(script)]})
        self.directory = self.run / "remote_setup/2026-10-31-17-00_h100-nvl-allocation-observation_gpu-1_cpus-8_memory-64g_seconds-300"
        with patch.object(observation, "__file__", str(self.helper_root / observation.RELATIVE_PATH)):
            self.prepared = observation.prepare(self.directory, self.run, self.helper_root,
                                                self.resource_submission, Path(sys.executable))
        self.plan = observation.read(self.directory / "allocation_observation_plan.json")
        self.unit = Path(self.plan["data_unit"]["path"])
        write(self.directory / "submission_receipt.json", {"kind": "owned_h100_nvl_allocation_observation_submission",
            "cluster": "uva", "job_id": "123", "plan_sha256": observation.digest(self.directory / "allocation_observation_plan.json"),
            "job_script_sha256": observation.digest(self.directory / "job.sbatch")})
        self.environment = {"SLURM_JOB_ID": "123", "SLURM_STEP_ID": "0", "SLURM_JOB_GPUS": "1", "SLURM_STEP_GPUS": "1",
            "SLURM_GPUS_ON_NODE": "1", "SLURM_CPUS_PER_TASK": "8", "CUDA_VISIBLE_DEVICES": "0"}

    def owned(self, environment=None, hostname="serval08", cgroup="0::/slurm/uid_1/job_123/step_0"):
        """Run only the metadata allocation guard against the fixture's exact job."""
        return observation.owned_allocation(self.plan, self.directory, environment or self.environment, cgroup, hostname)

    def test_prepare_binds_resource_source_and_monitored_data_shard(self):
        """The prepared packet requests five minutes and contains no submission or authorization."""
        manifest = observation.read(self.directory / "manifest.yaml")
        self.assertFalse(self.prepared["submitted"])
        self.assertEqual(self.plan["job_shape"]["account"], "recorded_account")
        for argument in ("--hold", "--gres=gpu:nvidia_h100_nvl:1", "--cpus-per-task=8", "--mem=64G", "--time=00:05:00"):
            self.assertIn(argument, manifest["submission_command"])
        self.assertEqual(self.plan["data_unit"], {"path": str(self.unit), "name": self.directory.name})
        for filename in ("experiment_background.md", "manifest.yaml", "config_resolved.yaml", "command.txt"):
            self.assertEqual((self.unit / filename).read_bytes(), (self.directory / filename).read_bytes())
        self.assertEqual(observation.read(self.unit / "progress.json")["state"], "waiting_for_allocation_observation")
        self.assertEqual((self.directory / "experiment_background.md").read_text().count("## "), 4)
        self.assertIn(" -I ", (self.directory / "job.sbatch").read_text())
        self.assertIn("export TZ=UTC", (self.directory / "job.sbatch").read_text())
        self.assertFalse((self.run / "queue").exists())
        self.assertFalse((self.run / "SOURCE_AND_GATES_READY.json").exists())
        for name, value in observation.BOUNDARY.items():
            self.assertEqual(self.plan[name], value)

    def test_prepare_does_not_replace_existing_packets(self):
        """A second prepare cannot overwrite original observations or a failed site's files."""
        before = observation.digest(self.directory / "manifest.yaml")
        with patch.object(observation, "__file__", str(self.helper_root / observation.RELATIVE_PATH)), \
                self.assertRaisesRegex(ValueError, "both be new"):
            observation.prepare(self.directory, self.run, self.helper_root, self.resource_submission, Path(sys.executable))
        self.assertEqual(observation.digest(self.directory / "manifest.yaml"), before)

    def test_prepare_must_execute_from_the_selected_committed_helper(self):
        """A mutable integration invocation cannot claim it ran from a different detached source."""
        new_directory = self.run / "remote_setup/another-allocation-observation"
        with self.assertRaisesRegex(ValueError, "prepare must execute the exact committed"):
            observation.prepare(new_directory, self.run, self.helper_root, self.resource_submission, Path(sys.executable))
        self.assertFalse(new_directory.exists())
        self.assertFalse((self.run / "data" / new_directory.name).exists())

    def test_changed_or_attached_helper_is_rejected(self):
        """Preparation cannot point the future job at mutable or uncommitted helper bytes."""
        helper_path = self.helper_root / observation.RELATIVE_PATH
        helper_path.write_text(helper_path.read_text() + "\n# changed helper\n")
        with self.assertRaisesRegex(ValueError, "committed bytes"):
            observation.helper_source(self.helper_root)
        subprocess.run(["git", "-C", str(self.helper_root), "checkout", "--", observation.RELATIVE_PATH], check=True)
        subprocess.run(["git", "-C", str(self.helper_root), "checkout", "-qb", "mutable helper".replace(" ", "-")], check=True)
        with self.assertRaisesRegex(ValueError, "detached"):
            observation.helper_source(self.helper_root)

    def test_foreign_resource_proof_and_changed_plan_fail_before_new_outputs(self):
        """An ID in another cluster's ledger or changed resource bytes cannot define this probe."""
        (self.run / "slurm/submitted_jobids_uva.txt").write_text("123\n")
        (self.run / "slurm/submitted_jobids_aces.txt").write_text("456\n")
        with self.assertRaisesRegex(ValueError, "campaign-owned UVA"):
            observation.resource_basis(self.run, self.resource_submission)
        (self.run / "slurm/submitted_jobids_uva.txt").write_text("123\n456\n")
        plan = self.resource_submission.parent / "site_validation_plan.json"
        plan.write_text(plan.read_text() + " ")
        with self.assertRaisesRegex(ValueError, "changed"):
            observation.resource_basis(self.run, self.resource_submission)

    def test_owned_global_index_and_visible_index_are_both_preserved(self):
        """Slurm index one and visible index zero are observations rather than an accepted mapping."""
        with patch.object(observation.subprocess, "run", return_value=subprocess.CompletedProcess([], 0, scheduler_text(), "")) as query:
            record = self.owned()
        self.assertEqual(record["visible_selector"], "0")
        self.assertIn("IDX:1", record["scheduler"]["stdout"])
        self.assertEqual(query.call_args.args[0], ["scontrol", "show", "job", "-d", "123"])
        self.assertNotIn("env", query.call_args.kwargs)

    def test_unowned_job_cgroup_and_ineligible_host_never_query_hardware(self):
        """Ownership and host checks run before even the owned scheduler query."""
        for environment, hostname, cgroup in (
                ({**self.environment, "SLURM_JOB_ID": "999"}, "serval08", "0::/job_999/step_0"),
                (self.environment, "portal05", "0::/job_123/step_0"),
                (self.environment, "serval08", "0::/job_124/step_0")):
            with self.subTest(host=hostname, job=environment["SLURM_JOB_ID"]), patch.object(observation.subprocess, "run") as query:
                with self.assertRaisesRegex(ValueError, "exact UVA job"):
                    self.owned(environment, hostname, cgroup)
                query.assert_not_called()

    def test_multiple_visible_or_allocated_cards_are_rejected(self):
        """Lists, ranges and extra allocated GPUs never reach an NVIDIA query or inventory read."""
        for changes in ({"CUDA_VISIBLE_DEVICES": "0,1"}, {"SLURM_JOB_GPUS": "0-1"}, {"SLURM_GPUS_ON_NODE": "2"}):
            with self.subTest(changes=changes), patch.object(observation.subprocess, "run") as query:
                with self.assertRaises(ValueError):
                    self.owned({**self.environment, **changes})
                query.assert_not_called()
        with patch.object(observation.subprocess, "run", return_value=subprocess.CompletedProcess([], 0, scheduler_text(count=2), "")) as query:
            with self.assertRaisesRegex(ValueError, "one typed H100 NVL"):
                self.owned()
            self.assertEqual(query.call_count, 1)

    def test_other_job_or_untyped_scheduler_record_is_rejected(self):
        """Exact top-level job identity and the allocated typed GRES row are both required."""
        for text in (scheduler_text("124"), scheduler_text().replace("gpu:nvidia_h100_nvl:1", "gpu:other:1"),
                     scheduler_text() + "JobId=999\n", scheduler_text().replace("(IDX:1)", "(IDX:0-1)")):
            with self.subTest(text=text), patch.object(observation.subprocess, "run", return_value=subprocess.CompletedProcess([], 0, text, "")):
                with self.assertRaises(ValueError):
                    self.owned()

    def test_inventory_keeps_only_nvidia_gpu_hardware_fields(self):
        """Full-host records may include both GPUs, while unrelated PCI devices and fields are omitted."""
        proc, pci, dev = (self.root / name for name in ("proc GPUs", "PCI devices", "device files"))
        for address, minor in (("0000:c1:00.0", "0"), ("0000:c2:00.0", "1")):
            information = proc / address / "information"
            information.parent.mkdir(parents=True)
            information.write_text(f"Model: NVIDIA H100 NVL\nGPU UUID: GPU-{minor}\nBus Location: {address}\nDevice Minor: {minor}\nUnrelated field: must not persist\n")
        for address, vendor, device_class in (("0000:c1:00.0", "0x10de", "0x030200"),
                ("0000:c2:00.0", "0x10de", "0x030000"), ("0000:c3:00.0", "0x10de", "0x060400"), ("0000:c4:00.0", "0x8086", "0x030000")):
            for key, value in {"vendor": vendor, "device": "0x2321", "class": device_class,
                               "subsystem_vendor": "0x10de", "subsystem_device": "0x0000"}.items():
                path = pci / address / key
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(value + "\n")
        dev.mkdir()
        (dev / "nvidia0").touch()
        (dev / "nvidia1").touch()
        (dev / "nvidia1_processes").write_text("must not read")
        gres = self.root / "gres.conf"
        gres.write_text("# comment\nNodeName=serval[06-09] AutoDetect=nvml\nName=gpu Type=nvidia_h100_nvl File=/dev/nvidia[0-1]\nName=mps Count=100\nName=license AutoDetect=nvml\n")
        original_stat = Path.stat
        def metadata_stat(path, *args, **kwargs):
            """Represent actual character-device numbers without creating special device files."""
            if path.parent == dev and path.name in {"nvidia0", "nvidia1"}:
                return SimpleNamespace(st_mode=stat.S_IFCHR | 0o600, st_rdev=os.makedev(195, int(path.name[-1])))
            return original_stat(path, *args, **kwargs)
        with patch.object(Path, "stat", metadata_stat):
            record = observation.host_inventory(proc, pci, dev, gres)
        self.assertEqual(len(record["proc_gpu_inventory"]["cards"]), 2)
        self.assertEqual(len(record["pci_gpu_inventory"]["cards"]), 2)
        self.assertEqual([card["minor"] for card in record["device_files"]], [0, 1])
        self.assertEqual([card["major"] for card in record["device_files"]], [195, 195])
        self.assertEqual(len(record["gpu_gres_mapping"]["lines"]), 2)
        for forbidden in ("must not", "Unrelated field", "Name=mps", "Name=license", "processes"):
            self.assertNotIn(forbidden, json.dumps(record))

    def test_missing_inventory_is_explicit_without_alternative_mapping(self):
        """Unavailable host files remain absent evidence rather than an inferred identity."""
        record = observation.host_inventory(*(self.root / name for name in ("absent proc", "absent PCI", "absent dev", "absent gres")))
        self.assertEqual(record["proc_gpu_inventory"]["status"], "absent")
        self.assertEqual(record["pci_gpu_inventory"]["status"], "absent")
        self.assertEqual(record["gpu_gres_mapping"]["status"], "absent")

    def test_capture_publishes_only_monitored_metadata_and_exact_visible_queries(self):
        """The complete mock capture never selects an alternative card or grants admission."""
        csv = "0, GPU-abc, NVIDIA H100 NVL, 00000000:C1:00.0, 95830, 580.159.04\n"
        outputs = [scheduler_text(), csv]
        def query(command, **kwargs):
            """Return the owned scheduler row, then only the explicitly selected visible GPU."""
            return subprocess.CompletedProcess(command, 0, outputs.pop(0), "")
        with patch.object(observation, "helper_source", return_value=self.plan["helpers"]), \
                patch.object(observation, "__file__", str(self.helper_root / observation.RELATIVE_PATH)), \
                patch.object(observation.Path, "read_text", autospec=True) as read_text, \
                patch.object(observation.socket, "gethostname", return_value="serval08"), \
                patch.dict(os.environ, {**self.environment, "PRIVATE_TOKEN": "must not read or record"}, clear=True), \
                patch.object(observation.subprocess, "run", side_effect=query) as queries, \
                patch.object(observation, "owned_allocation", wraps=observation.owned_allocation) as allocation, \
                patch.object(observation, "host_inventory", return_value={"hardware fields": "only"}):
            original_read = Path.open
            def read_file(path, *args, **kwargs):
                """Provide an owned cgroup while reading all prepared bytes normally."""
                if path == Path("/proc/self/cgroup"):
                    return "0::/slurm/uid_1/job_123/step_0\n"
                with original_read(path, "r") as stream:
                    return stream.read()
            read_text.side_effect = read_file
            result = observation.capture(self.directory)
        self.assertTrue(result["observations_completed"])
        self.assertFalse(result["mapping_accepted"])
        self.assertFalse(result["compatibility_verified"])
        self.assertEqual(result["scientific_hypothesis_credit"], 0)
        self.assertEqual(observation.read(self.unit / "progress.json")["state"], "completed")
        hardware = observation.read(self.unit / "hardware.json")
        self.assertEqual(hardware["slurm_job_id"], "123")
        self.assertIn("NVIDIA H100 NVL", hardware["gpu_query"])
        self.assertEqual(queries.call_count, 2)
        self.assertEqual(queries.call_args_list[1].args[0], ["nvidia-smi", "-i", "0",
            "--query-gpu=index,uuid,name,pci.bus_id,memory.total,driver_version", "--format=csv,noheader,nounits"])
        for call in queries.call_args_list:
            self.assertNotIn("env", call.kwargs)
            self.assertFalse(any(argument in {"-q", "-x", "--query-compute-apps"} or "process" in argument
                                 for argument in call.args[0]))
        self.assertEqual(set(allocation.call_args.args[2]), set(self.environment))
        observations = observation.read(self.unit / "hardware_observations.json")
        self.assertEqual(set(observations["allocation_environment"]), set(observation.ALLOCATION_FIELDS))
        self.assertEqual(set(observations["visible_gpu_queries"]), {"gpu_csv"})
        self.assertNotIn("PRIVATE_TOKEN", json.dumps(observations))
        import monitor
        row = monitor.unit_rows(self.run)[0]
        self.assertEqual(row["state"], "completed")
        self.assertIn("allocation-observation", row["run"])
        self.assertEqual(row["gpu"], ["NVIDIA H100 NVL"])
        self.assertFalse((self.run / "queue").exists())

    def test_guard_failure_is_monitored_without_any_hardware_query(self):
        """An extra GPU allocation writes a failed metadata shard and never reads host inventory."""
        with patch.object(observation, "helper_source", return_value=self.plan["helpers"]), \
                patch.object(observation, "__file__", str(self.helper_root / observation.RELATIVE_PATH)), \
                patch.object(observation, "owned_allocation", side_effect=ValueError("extra allocated GPU")), \
                patch.object(observation.subprocess, "run") as query, patch.object(observation, "host_inventory") as inventory:
            with self.assertRaisesRegex(ValueError, "extra allocated GPU"):
                observation.capture(self.directory)
            query.assert_not_called()
            inventory.assert_not_called()
        self.assertEqual(observation.read(self.unit / "progress.json")["state"], "failed")
        self.assertIn("extra allocated GPU", observation.read(self.unit / "failure.json")["error"])
        self.assertFalse((self.unit / "result.json").exists())

    def test_fresh_alias_import_uses_standard_library_only(self):
        """The standalone staged helper has no array, learner, controller or source-admission import."""
        code = ("import importlib.util,json,sys;from pathlib import Path;"
            "p=Path(sys.argv[1]);s=importlib.util.spec_from_file_location('isolated_observation',p);"
            "m=importlib.util.module_from_spec(s);s.loader.exec_module(m);"
            "bad=[n for n in sys.modules if n.split('.')[0] in {'jax','numpy','torch','jaxtari','controller','post_comparison_validation'}];"
            "print(json.dumps(bad));assert not bad")
        result = subprocess.run([sys.executable, "-I", "-c", code, str(self.helper_root / observation.RELATIVE_PATH)],
                                capture_output=True, text=True, check=True)
        self.assertEqual(json.loads(result.stdout), [])


if __name__ == "__main__":
    unittest.main()
