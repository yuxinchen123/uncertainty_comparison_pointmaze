"""Check the combined readers with synthetic ownership and allocation records.

These tests call the real readers. They never run numerical code, query Slurm,
or turn the preserved CPU observation into a GPU qualification.
"""

import copy
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "10_jax_exploration_platform/scripts/atari_pixel_campaign"
sys.path[:0] = [str(SCRIPTS), str(SCRIPTS / "tests")]
import device_memory_limit as capacity
import post_comparison_validation as validation
import portable_evidence
import test_device_memory_limit as memory_fixtures
import test_measure_selected as measurement_fixtures
import test_uva_cgroup_v2_ownership as ownership_fixtures
from final_phase_fixtures import FinalPhaseFixture


class CombinedExactEightTests(unittest.TestCase):
    """Combine exact-eight failure bytes, the explicit profile and final-phase timing checks."""

    @classmethod
    def setUpClass(cls):
        """Create synthetic records under a real detached combined-source Git fixture."""
        cls.memory = memory_fixtures.DeviceMemoryLimitTests("runTest")
        cls.memory.setUp()
        cls.addClassCleanup(cls.memory.doCleanups)
        cls.memory.build(fitting=4)
        cls.unit = cls.memory.unit
        cls.public_source = patch.object(validation, "JAXTARI_COMMIT", memory_fixtures.deployment.JAXTARI_COMMIT)
        cls.public_source.start()
        cls.addClassCleanup(cls.public_source.stop)
        cls.final = FinalPhaseFixture(source_root=cls.memory.fixture.source,
            asset_record=cls.memory.fixture.global_asset_record)
        cls.addClassCleanup(cls.final.close)
        cls.ownership = ownership_fixtures.ExplicitUvaOwnershipTests("runTest")
        cls.ownership.setUp()
        cls.addClassCleanup(cls.ownership.doCleanups)
        cls.convert_synthetic_capacity_to_explicit_uva_profile()

        cls.specification = copy.deepcopy(cls.final.specification)
        cls.specification["fixed_total_scaling"] = {"status": "not_applicable",
            "eligibility_kind": "instrumented_exact_eight_capacity_exclusion",
            "eligibility_record": str(cls.unit / "device_memory_limit.json")}
        confirmation = Path(cls.specification["contexts"]["jaguar03_rtx_a4500"]["confirmation_record"])
        for path in confirmation.parent.glob("*.json"):
            contents = json.loads(path.read_bytes())
            # These are deliberately derived identities, not observations of an actual allocation.
            contents = cls.replace_gpu(contents)
            if path.name == "result.json":
                contents["installed_backend"] = cls.memory.fixture.installed
            elif path.name == "installed_backend.json":
                contents = cls.memory.fixture.installed
            cls.write(path, contents)
        plan = json.loads((confirmation.parent / "measurement_plan.json").read_bytes())
        cls.final.prepare(confirmation.parent, plan)
        cls.selected = json.loads(confirmation.read_bytes())["candidate"]
        cls.originals = {path: path.read_bytes() for path in cls.unit.rglob("*") if path.is_file()}

    @staticmethod
    def write(path, value):
        """Write only temporary synthetic JSON records with finite literal values."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, allow_nan=False) + "\n")
        return path

    @classmethod
    def replace_gpu(cls, value):
        """Align the synthetic final confirmation with the independently verified failed-device fixture."""
        if isinstance(value, dict):
            return {key: cls.replace_gpu(item) for key, item in value.items()}
        if isinstance(value, list):
            return [cls.replace_gpu(item) for item in value]
        return value.replace("GPU-one", "GPU-target") if isinstance(value, str) else value

    @classmethod
    def synthetic_process_proof(cls, process):
        """Derive a new labelled structural proof; the original CPU fixture remains unchanged."""
        value = copy.deepcopy(cls.ownership.value)
        original_group = cls.ownership.process["cgroup"].strip().split("::", 1)[1]
        new_group, unused = capacity.v2_paths("100", "0")
        for stage in ("before", "after"):
            row = value[stage]
            row.update(pid=process["pid"], uid=2618919, effective_uid=2618919)
            raw = row["stat"]["raw"]
            command_end = raw.rfind(")")
            fields = raw[command_end + 2:].split()
            fields[1], fields[19] = str(process["parent_pid"]), process["process_start_time"]
            row["stat"]["raw"] = str(process["pid"]) + raw[raw.find(" "):command_end + 1] + " " + " ".join(fields) + "\n"
            row["stat"]["sha256"] = hashlib.sha256(row["stat"]["raw"].encode()).hexdigest()
            row["cgroup"]["raw"] = "0::" + new_group + "\n"
            row["cgroup"]["sha256"] = hashlib.sha256(row["cgroup"]["raw"].encode()).hexdigest()
        for collection in ("directories_before", "directories_after", "memberships"):
            for row in value[collection]:
                row["path"] = row["path"].replace(original_group, new_group)
                row["path"] = row["path"].replace("job_6780635", "job_100").replace("step_50", "step_0")
        value["memberships"][-1]["raw"] = str(process["pid"]) + "\n"
        for row in value["memberships"]:
            row["sha256"] = hashlib.sha256(row["raw"].encode()).hexdigest()
        return value

    @classmethod
    def convert_synthetic_capacity_to_explicit_uva_profile(cls):
        """Keep the complete native failure while deriving a separate synthetic UVA numeric-step identity."""
        plan = json.loads((cls.unit / "measurement_plan.json").read_bytes())
        plan["hardware_class"] = "jaguar03_rtx_a4500"
        plan["ownership_profile"] = copy.deepcopy(capacity.UVA_CGROUP_V2_OWNERSHIP_PROFILE)
        plan["device_memory_limit_protocol"]["cluster"] = "uva"
        ledger = cls.memory.fixture.run / "slurm/submitted_jobids_uva.txt"
        ledger.write_text("100\n")
        plan["device_memory_limit_ledger_source"] = capacity.file_receipt(ledger)
        cls.memory.fixture.write_prepared_plan(cls.unit, plan)
        hardware = json.loads((cls.unit / "hardware.json").read_bytes())
        hardware.update(hostname="jaguar03", gpu_query=hardware["gpu_query"].replace("NVIDIA H100 PCIe", "NVIDIA RTX A4500"))
        cls.write(cls.unit / "hardware.json", hardware)
        observation = json.loads((cls.unit / "capacity_observation.json").read_bytes())
        observation["protocol"] = plan["device_memory_limit_protocol"]
        group, unused = capacity.v2_paths("100", "0")
        for name in ("ownership_before", "ownership_after"):
            owned = observation[name]
            owned.update(cluster="uva", ownership_profile=copy.deepcopy(plan["ownership_profile"]),
                ownership_capability=capacity.UVA_CGROUP_V2_OWNERSHIP_CAPABILITY)
            owned["process"].update(hostname="jaguar03", uid=2618919, cgroup="0::" + group + "\n")
            owned["cgroup_v2_ownership"] = cls.synthetic_process_proof(owned["process"])
            owned["environment"]["SLURM_CONF"] = "/etc/slurm/slurm.conf"
            for query in ("job", "step", "node"):
                owned[query]["command"] = [part.replace("ac049", "jaguar03") for part in owned[query]["command"]]
                owned[query]["stdout"] = owned[query]["stdout"].replace("u.sl359075(56626)", "sl5nw(2618919)").replace("ac049", "jaguar03").replace("h100", "nvidia_rtx_a4500")
            mapping = owned["physical_mapping"]
            mapping.update(slurm_conf="/etc/slurm/slurm.conf", other_configuration_paths=[], profile=capacity.CONFIGURATION_PROFILES["uva"])
            mapping["directory_chain"] = [{"path": str(path), "owner_uid": 0, "mode": 0o755, "directory": True, "symlink": False}
                for path in reversed((Path("/etc/slurm"), *Path("/etc/slurm").parents))]
            for row in mapping["configuration"]:
                row["path"] = "/etc/slurm/" + row["kind"] + ".conf"
                row["filtered_bytes"] = row["filtered_bytes"].replace("Type=h100", "Type=nvidia_rtx_a4500")
                row["sha256"] = row["filtered_sha256"] = hashlib.sha256(row["filtered_bytes"].encode()).hexdigest()
        raw = "low 0\nhigh 0\nmax 0\noom 0\noom_kill 0\n"
        counters = [{"path": str(path), "raw": raw, "sha256": hashlib.sha256(raw.encode()).hexdigest(),
            "counters": {key: 0 for key in ("max", "oom", "oom_kill")}}
            for path in capacity.host_counter_paths("0::" + group + "\n")]
        observation["host_memory_before"] = counters
        observation["host_memory_after"] = copy.deepcopy(counters)
        parent = json.loads((cls.unit / "capacity_parent.json").read_bytes())
        parent["process"].update(hostname="jaguar03", uid=2618919, cgroup="0::" + group + "\n")
        parent["plan"] = capacity.file_receipt(cls.unit / "measurement_plan.json")
        terminal = json.loads((cls.unit / "capacity_child_exit.json").read_bytes())
        terminal["parent"] = parent["process"]
        def later(value):
            """Move only synthetic wall-clock timestamps after the separately reviewed fifty-idea proof."""
            if isinstance(value, dict):
                return {key: item + 200000. if "seconds" in key and type(item) in (int, float) else later(item)
                    for key, item in value.items()}
            return [later(item) for item in value] if isinstance(value, list) else value
        observation = later(observation)
        observation["native_map_original_name"] = "capacity_native_map_GPU_0_bfc.201000000000"
        for name, value in (("capacity_observation.json", observation), ("capacity_parent.json", later(parent)),
                ("capacity_child_exit.json", later(terminal))):
            cls.write(cls.unit / name, value)
        cls.memory.seal()

    def tearDown(self):
        """Restore every original synthetic failure byte between negative checks."""
        self.restore()

    def restore(self):
        """Keep each deliberate mutation independent from the next negative case."""
        for path in self.unit.rglob("*"):
            if path.is_file() and path not in self.originals:
                path.unlink()
        for path, raw in self.originals.items():
            path.write_bytes(raw)

    def test_actual_exact_eight_readers_accept_complete_synthetic_v2_evidence(self):
        """The real capacity and final fixed-total readers accept complete evidence without granting science credit."""
        common = {"source_commit": self.final.commit, "hardware_class": "jaguar03_rtx_a4500",
            "runtime_environment": self.final.runtime}
        excluded = capacity.validate_exclusion(self.unit / "device_memory_limit.json", common, validation.Evidence())
        self.assertEqual(excluded["copies"], 8)
        self.assertEqual(excluded["throughput_credit"], 0)
        result = validation.validate_fixed_total_scaling(self.specification, self.selected, validation.Evidence())
        self.assertEqual(result["status"], "not_applicable")
        self.assertEqual(result["exact_total_copies"], 8)
        self.assertFalse((self.unit / "result.json").exists())

    def reseal_plan(self, plan):
        """Keep all hashes coherent so a negative test must inspect the changed meaning."""
        self.memory.fixture.write_prepared_plan(self.unit, plan)
        parent = json.loads((self.unit / "capacity_parent.json").read_bytes())
        parent["plan"] = capacity.file_receipt(self.unit / "measurement_plan.json")
        self.write(self.unit / "capacity_parent.json", parent)
        self.memory.seal()

    def test_wrong_count_profile_and_source_cannot_establish_exact_eight(self):
        """A rehashed four-copy failure, changed profile or changed source remains ineligible."""
        for kind in ("four_copies", "changed_profile", "changed_source"):
            self.restore()
            plan = json.loads((self.unit / "measurement_plan.json").read_bytes())
            if kind == "four_copies":
                plan["configuration"]["learner"]["n_copies"] = 4
                plan["copy_seeds"] = plan["copy_seeds"][:4]
            elif kind == "changed_profile":
                plan["ownership_profile"]["kernel_device_enforcement_verified"] = True
                observation = json.loads((self.unit / "capacity_observation.json").read_bytes())
                for name in ("ownership_before", "ownership_after"):
                    observation[name]["ownership_profile"] = copy.deepcopy(plan["ownership_profile"])
                    observation[name]["cgroup_v2_ownership"]["ownership_profile"] = copy.deepcopy(plan["ownership_profile"])
                self.write(self.unit / "capacity_observation.json", observation)
            else:
                plan["source_commit"] = "0" * 40
            self.reseal_plan(plan)
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                validation.validate_fixed_total_scaling(self.specification, self.selected, validation.Evidence())

    def test_diagnostic_and_completed_attempts_cannot_be_capacity_evidence(self):
        """Operational diagnostic and completion records never acquire formal capacity credit."""
        for kind in ("device_memory_capture_component_regression", "instrumented_device_memory_attempt_completion"):
            self.restore()
            record = json.loads((self.unit / "device_memory_limit.json").read_bytes())
            record["kind"] = kind
            self.write(self.unit / "device_memory_limit.json", record)
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                validation.validate_fixed_total_scaling(self.specification, self.selected, validation.Evidence())
        self.restore()
        self.write(self.unit / "result.json", self.memory.original_result)
        with self.assertRaisesRegex(ValueError, "completed result"):
            validation.validate_fixed_total_scaling(self.specification, self.selected, validation.Evidence())

    def test_portable_exact_eight_requires_all_original_raw_bytes(self):
        """The actual reader still passes after original paths disappear, and refuses missing or changed copies."""
        evidence = validation.Evidence()
        expected = validation.validate_fixed_total_scaling(self.specification, self.selected, evidence)
        with tempfile.TemporaryDirectory() as temporary:
            staging = Path(temporary)
            manifest = {"schema_version": 1, "kind": "portable_raw_evidence", "files": [], "git_roots": []}
            roots = ((self.memory.fixture.source, "combined_source", self.memory.fixture.commit),
                     (self.memory.fixture.public, "synthetic_public_source", memory_fixtures.deployment.JAXTARI_COMMIT))
            for original, name, commit in roots:
                destination = staging / name
                subprocess.run(["git", "clone", "-q", "--no-hardlinks", str(original), str(destination)], check=True)
                subprocess.run(["git", "-C", str(destination), "checkout", "-q", "--detach", commit], check=True)
                manifest["git_roots"].append({"original_root": str(original), "relative_path": name, "commit": commit})
            for index, receipt in enumerate(evidence.records()):
                original = Path(receipt["path"])
                relative = Path("files") / str(index)
                destination = staging / relative
                destination.parent.mkdir(exist_ok=True)
                shutil.copyfile(original, destination)
                manifest["files"].append({"original_path": str(original), "relative_path": str(relative),
                    "sha256": hashlib.sha256(original.read_bytes()).hexdigest()})
            mapping = self.write(staging / "portable.json", manifest)
            moved = []
            original_roots = {self.memory.fixture.directory, self.memory.fixture.fixture.directory,
                              self.final.root, self.final.raw.directory}
            try:
                for original in original_roots:
                    hidden = original.with_name(original.name + "_temporarily_absent")
                    original.rename(hidden)
                    moved.append((original, hidden))
                actual = validation.validate_fixed_total_scaling(self.specification, self.selected,
                    validation.Evidence(portable_evidence.ExactPathResolver(mapping)))
                self.assertEqual(actual, expected)
                missing = copy.deepcopy(manifest)
                missing["files"] = [row for row in missing["files"]
                    if row["original_path"] != str(self.unit / "capacity_native_memory_map.pb")]
                self.write(mapping, missing)
                with self.assertRaises((ValueError, FileNotFoundError)):
                    validation.validate_fixed_total_scaling(self.specification, self.selected,
                        validation.Evidence(portable_evidence.ExactPathResolver(mapping)))
                self.write(mapping, manifest)
                row = next(row for row in manifest["files"]
                    if row["original_path"] == str(self.unit / "capacity_observation.json"))
                path = staging / row["relative_path"]
                path.write_bytes(path.read_bytes() + b"\n")
                with self.assertRaises(ValueError):
                    validation.validate_fixed_total_scaling(self.specification, self.selected,
                        validation.Evidence(portable_evidence.ExactPathResolver(mapping)))
            finally:
                for original, hidden in reversed(moved):
                    hidden.rename(original)

    def test_instrumented_preparation_keeps_resolved_default_rounds(self):
        """The merged optional CLI defaults still prepare seven-by-ten instrumented packing work."""
        case = measurement_fixtures.SelectedMeasurementTests("runTest")
        case.setUp()
        self.addCleanup(case.doCleanups)
        arguments = case.arguments()
        arguments.mode, arguments.reference_configuration, arguments.copies = "packing-point", None, 2
        arguments.rounds = arguments.minimum_updates_per_round = None
        arguments.record_device_memory_limit = True
        arguments.memory_limit_allocation_kind, arguments.memory_limit_cluster = "slurm", "uva"
        ledger = case.run_dir / "slurm/submitted_jobids_uva.txt"
        ledger.parent.mkdir()
        ledger.write_text("123\n")
        arguments.memory_limit_owned_job_ledger = ledger
        directory = case.prepare(arguments)
        plan = json.loads((directory / "measurement_plan.json").read_bytes())
        self.assertEqual((plan["rounds"], plan["minimum_updates_per_round"]), (7, 10))
        self.assertEqual(plan["device_memory_limit_protocol"]["cluster"], "uva")


if __name__ == "__main__":
    unittest.main()
