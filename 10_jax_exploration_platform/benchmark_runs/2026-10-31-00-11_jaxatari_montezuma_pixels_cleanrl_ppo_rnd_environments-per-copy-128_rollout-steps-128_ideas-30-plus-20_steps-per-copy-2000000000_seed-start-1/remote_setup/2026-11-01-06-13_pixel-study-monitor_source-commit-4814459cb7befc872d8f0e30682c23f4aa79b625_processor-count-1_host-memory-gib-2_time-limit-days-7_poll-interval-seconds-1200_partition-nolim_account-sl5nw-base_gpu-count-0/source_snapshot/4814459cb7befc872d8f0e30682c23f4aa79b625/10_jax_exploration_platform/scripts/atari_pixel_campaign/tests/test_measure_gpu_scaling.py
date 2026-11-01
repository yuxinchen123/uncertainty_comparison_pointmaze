"""Test synchronized scaling metadata and communication without numerical libraries."""

import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time
import types
import unittest
from unittest.mock import Mock, patch


SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS))
SPECIFICATION = importlib.util.spec_from_file_location("synchronized_gpu_scaling", SCRIPTS / "measure_gpu_scaling.py")
SCALING = importlib.util.module_from_spec(SPECIFICATION)
SPECIFICATION.loader.exec_module(SCALING)


class ScalingMeasurementTests(unittest.TestCase):
    """Exercise complete source/configuration and group-time evidence using only standard-library fixtures."""

    def setUp(self):
        """Create a full source-default optimized configuration and explicit launch inputs."""
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.source = SCRIPTS.parents[2]
        defaults = SCALING.diagnostic.default_configuration(self.source)
        self.configuration = {name: defaults[name] for name in ("learner", "environment")}
        self.configuration["learner"]["n_copies"] = 2
        self.configuration["execution"] = {"compilation_mode": "whole_iteration", "iterations_per_call": 5,
                                            "report_readout": "none", "report_every_iterations": 10}
        self.runtime = {"JAX_ENABLE_X64": "true", "JAX_DEFAULT_MATMUL_PRECISION": "highest",
                        "XLA_FLAGS": "--xla_gpu_exclude_nondeterministic_ops=true"}
        self.inputs = {}
        for name, document in (("configuration", self.configuration), ("runtime", self.runtime),
                ("assets", {"root": "/staged/assets", "manifest": "/staged/manifest.json", "sha256": "a" * 64})):
            path = self.directory / (name + ".json")
            path.write_text(json.dumps(document))
            self.inputs[name] = path

    def arguments(self, count):
        """Use the real metadata-only CLI with one explicitly recorded UUID per physical card."""
        args = ["--prepare-only", "--source-root", str(self.source), "--source-commit", "c" * 40,
                "--configuration", str(self.inputs["configuration"]), "--runtime-environment", str(self.inputs["runtime"]),
                "--asset-record", str(self.inputs["assets"]), "--run-dir", str(self.directory),
                "--unit-id", "selected_configuration_a4500_synchronized_gpus-" + str(count) + "_seed-start-1"]
        for index in range(count):
            args.extend(["--gpu-uuid", "GPU-" + str(index)])
        return SCALING.parse_arguments(args)

    def prepare(self, arguments):
        """Mock only committed Git receipts; preserve the real staging and study-configuration checks."""
        helpers = {"source_commit": "b" * 40, "files": {
            "10_jax_exploration_platform/scripts/atari_pixel_campaign/" + name:
                hashlib.sha256((SCRIPTS / name).read_bytes()).hexdigest() for name in SCALING.HELPERS}}
        with patch.object(SCALING.diagnostic, "core_identity", return_value={"source_commit": "c" * 40}), \
                patch.object(SCALING.training, "source_receipt", return_value={"source_commit": "c" * 40, "files": {}}), \
                patch.object(SCALING, "helper_receipt", return_value=helpers), \
                patch.object(SCALING.subprocess, "Popen") as process, patch.object(SCALING, "allocation_proof") as gpu:
            directory = SCALING.prepare(arguments)
            process.assert_not_called()
            gpu.assert_not_called()
        return directory

    def test_complete_eight_card_records_precede_any_compute(self):
        """One group-five configuration produces seven-by-ten measurements with distinct seed blocks."""
        directory = self.prepare(self.arguments(8))
        plan = json.loads((directory / "scaling_plan.json").read_text())
        self.assertEqual((plan["gpu_count"], plan["rounds"], plan["iterations"]), (8, 7, 10))
        self.assertEqual(plan["copy_seeds"], [list(range(index * 2 + 1, index * 2 + 3)) for index in range(8)])
        self.assertEqual(plan["configuration"], json.loads(json.dumps(self.configuration)))
        for name in ("scaling_prepared.json", "manifest.yaml", "config_resolved.yaml", "command.txt", "experiment_background.md"):
            self.assertTrue((directory / name).is_file())
        for name in SCALING.HELPERS:
            self.assertEqual((directory / "code" / name).read_bytes(), (SCRIPTS / name).read_bytes())
        self.assertNotIn("jax", sys.modules)
        self.assertNotIn("numpy", sys.modules)

    def test_control_and_eight_cards_preserve_first_policy_identities_and_work(self):
        """Only the number of physical cards changes between the matching prepared groups."""
        one = json.loads((self.prepare(self.arguments(1)) / "scaling_plan.json").read_text())
        eight = json.loads((self.prepare(self.arguments(8)) / "scaling_plan.json").read_text())
        for name in ("configuration", "runtime_environment", "source_commit", "iterations", "rounds"):
            self.assertEqual(one[name], eight[name])
        self.assertEqual(one["copy_seeds"][0], eight["copy_seeds"][0])
        self.assertEqual(one["gpu_uuids"][0], eight["gpu_uuids"][0])

    def test_invalid_rosters_seed_range_and_algorithm_changes_are_rejected(self):
        """A prepared measurement cannot change the study algorithm or reuse a physical card."""
        for count in (0, 2, 7):
            with self.subTest(count=count), self.assertRaises(ValueError):
                self.prepare(self.arguments(count))
        args = self.arguments(8)
        args.gpu_uuid[-1] = args.gpu_uuid[0]
        with self.assertRaises(ValueError):
            self.prepare(args)
        args = self.arguments(8)
        args.seed_start = 2 ** 32 - 15
        with self.assertRaisesRegex(ValueError, "seed blocks"):
            self.prepare(args)
        self.configuration["learner"]["learning_rate"] = .01
        self.inputs["configuration"].write_text(json.dumps(self.configuration))
        with self.assertRaisesRegex(ValueError, "CleanRL"):
            self.prepare(self.arguments(1))

    def allocation_queries(self, cards):
        """Represent remapped NVML indices and stable physical device minors through CSV/XML."""
        csv_rows, xml_rows = [], []
        for index, (minor, gpu) in enumerate(cards):
            csv_rows.append(f"{index}, {gpu}, NVIDIA RTX A4500")
            xml_rows.append(f"<gpu><uuid>{gpu}</uuid><minor_number>{minor}</minor_number>"
                            "<product_name>NVIDIA RTX A4500</product_name></gpu>")
        output = {("nvidia-smi", "--query-gpu=index,uuid,name", "--format=csv,noheader"): "\n".join(csv_rows) + "\n",
                  ("nvidia-smi", "-q", "-x"): "<nvidia_smi_log>" + "".join(xml_rows) + "</nvidia_smi_log>"}
        def query(command, **options):
            """Fail on any unsupported physical-number '-i' lookup rather than returning card zero."""
            return subprocess.CompletedProcess(command, 0, output[tuple(command)], "")
        return query

    def test_one_card_allocation_proves_minor_five_behind_visible_zero(self):
        """Physical GPU five can appear at NVML and CUDA ordinal zero under Slurm."""
        plan = {"gpu_count": 1, "gpu_uuids": ["GPU-fifth"]}
        environment = {"SLURM_JOB_ID": "1234", "SLURM_JOB_GPUS": "5", "SLURM_STEP_GPUS": "GPU-fifth",
                       "CUDA_VISIBLE_DEVICES": "0"}
        with patch.object(SCALING.subprocess, "run", side_effect=self.allocation_queries([(5, "GPU-fifth")])):
            proof = SCALING.allocation_proof(plan, environment, "jaguar03")
        self.assertEqual(proof["cards"][0]["physical_minor"], "5")
        self.assertEqual(environment["SLURM_JOB_GPUS"], "5")
        self.assertEqual(proof["cuda_visible_devices_original"], "0")

    def test_full_eight_card_allocation_cannot_hide_an_extra_or_conflicting_device(self):
        """Both job and step allocations must identify every planned physical card."""
        plan = {"gpu_count": 8, "gpu_uuids": ["GPU-" + str(index) for index in range(8)]}
        environment = {"SLURM_JOB_ID": "1234", "SLURM_JOB_GPUS": ",".join(map(str, range(8))),
                       "SLURM_STEP_GPUS": ",".join(plan["gpu_uuids"]), "CUDA_VISIBLE_DEVICES": ",".join(map(str, range(8)))}
        with patch.object(SCALING.subprocess, "run", side_effect=self.allocation_queries(list(enumerate(plan["gpu_uuids"])))):
            proof = SCALING.allocation_proof(plan, environment, "jaguar03")
            self.assertEqual(len(proof["cards"]), 8)
            for changes in ({"SLURM_STEP_GPUS": "0"}, {"CUDA_VISIBLE_DEVICES": "0"}, {"SLURM_JOB_ID": ""}):
                with self.subTest(changes=changes), self.assertRaises(ValueError):
                    SCALING.allocation_proof(plan, {**environment, **changes}, "jaguar03")
            with self.assertRaisesRegex(ValueError, "jaguar03"):
                SCALING.allocation_proof(plan, environment, "portal05")
        with patch.object(SCALING.subprocess, "run", side_effect=self.allocation_queries([(5, "GPU-5")])):
            with self.assertRaisesRegex(ValueError, "count"):
                SCALING.allocation_proof(plan, environment, "jaguar03")

    def test_child_retains_full_allocation_and_its_one_proved_uuid(self):
        """A numerical child cannot invent a single-card Slurm allocation from an eight-card parent."""
        plan = {"gpu_count": 8, "gpu_uuids": ["GPU-" + str(index) for index in range(8)]}
        (self.directory / "scaling_plan.json").write_text("{}")
        allocations = {"SLURM_JOB_GPUS": ",".join(map(str, range(8)))}
        assignment = {"plan_sha256": SCALING.training.file_sha256(self.directory / "scaling_plan.json"),
            "gpu_uuids": plan["gpu_uuids"], "hostname": "jaguar03", "parent_pid": 100,
            "parent_process_start_time": "200", "allocation": {"slurm_job_id": "1234", "slurm_allocations": allocations,
                "cards": [{"gpu_uuid": gpu} for gpu in plan["gpu_uuids"]]}}
        (self.directory / "assignment.json").write_text(json.dumps(assignment))
        def query(gpu, fields, **options):
            """Expose only the already proved assigned UUID to per-device queries."""
            self.assertEqual(gpu, "GPU-5")
            return [] if options.get("processes") else [["GPU-5", "NVIDIA RTX A4500", "20470", "driver", "0"]]
        environment = {**allocations, "SLURM_JOB_ID": "1234", "CUDA_VISIBLE_DEVICES": "GPU-5"}
        with patch.dict(os.environ, environment, clear=True), patch.object(SCALING, "process_birth", return_value="200"), \
                patch.object(SCALING.socket, "gethostname", return_value="jaguar03"), \
                patch.object(SCALING.selected, "selected_query", side_effect=query):
            self.assertEqual(SCALING.verify_assignment(self.directory, plan, 5)["gpu_uuid"], "GPU-5")
            os.environ["SLURM_JOB_GPUS"] = "5"
            with self.assertRaisesRegex(ValueError, "original full"):
                SCALING.verify_assignment(self.directory, plan, 5)

    def worker(self, index, seconds=1., start=10.1, end=11.1):
        """Build one complete per-card raw record with actual-style free/used device samples."""
        gpu, copies = "GPU-" + str(index), 2
        return {"gpu_uuid": gpu, "copy_seeds": [2 * index + 1, 2 * index + 2],
            "copies": copies, "iterations": 10, "environments_per_copy": 128, "rollout_steps": 128,
            "training_transitions": copies * 128 * 128 * 10, "seconds": seconds,
            "start_time_seconds": start, "end_time_seconds": end,
            "memory": {"gpu_uuid": gpu, "other_gpu_processes": [], "sampled_process_peak_bytes": 1024,
                "allocator": {"peak_bytes_in_use": 1024}, "device_memory_samples": [
                    {"gpu_uuid": gpu, "total_bytes": 1000, "used_bytes": 100, "free_bytes": 900} for _ in range(2)]}}

    def test_aggregate_uses_common_wall_time_and_matches_final_validator(self):
        """A two-second common interval cannot be replaced by a sum of one-second worker rates."""
        records = []
        for count in (1, 8):
            plan = {"gpu_count": count, "gpu_uuids": ["GPU-" + str(index) for index in range(count)],
                    "copy_seeds": [[2 * index + 1, 2 * index + 2] for index in range(count)],
                    "configuration": self.configuration, "iterations": 10}
            workers = [self.worker(index) for index in range(count)]
            row = SCALING.aggregate_round(10., 12., workers, plan, 1)
            self.assertEqual(row["transitions_per_second_total"], sum(worker["training_transitions"] for worker in workers) / 2)
            self.assertNotEqual(row["transitions_per_second_total"], sum(worker["training_transitions"] / worker["seconds"] for worker in workers))
            record = {"source_commit": "c" * 40, "hardware_class": "jaguar03_rtx_a4500", "runtime_environment": self.runtime,
                      "configuration": self.configuration, "gpu_count": count, "rounds": [copy.deepcopy(row) for _ in range(7)]}
            path = self.directory / (str(count) + "_gpu_result.json")
            path.write_text(json.dumps(record))
            records.append(path)
        result = SCALING.validation.validate_scaling({"single_gpu_record": str(records[0]), "eight_gpu_record": str(records[1]),
            "source_commit": "c" * 40, "runtime_environment": self.runtime}, SCALING.validation.Evidence())
        self.assertEqual(result["median_aggregate_speedup"], 8.)

    def test_nonoverlap_changed_work_and_low_actual_memory_headroom_fail(self):
        """Separate-rate results, padded work and insufficient device free memory cannot pass."""
        plan = {"gpu_count": 8, "gpu_uuids": ["GPU-" + str(index) for index in range(8)],
                "copy_seeds": [[2 * index + 1, 2 * index + 2] for index in range(8)],
                "configuration": self.configuration, "iterations": 10}
        for failure in ("nonoverlap", "work", "memory"):
            workers = [self.worker(index) for index in range(8)]
            if failure == "nonoverlap":
                workers[0].update(start_time_seconds=10., end_time_seconds=10.05, seconds=.05)
            elif failure == "work":
                workers[0]["iterations"] = 11
            else:
                workers[0]["memory"]["device_memory_samples"][0].update(used_bytes=900, free_bytes=100)
            with self.subTest(failure=failure), self.assertRaises(ValueError):
                SCALING.aggregate_round(10., 12., workers, plan, 1)

    def test_socket_barrier_collects_out_of_order_workers_and_bounds_waits(self):
        """All worker releases/completions share a bounded barrier rather than independent timers."""
        channels, children = [], []
        self.addCleanup(lambda: [channel.close() for channel in channels])
        for index in range(2):
            parent, child = socket.socketpair()
            channels.extend([parent, child])
            children.append({"index": index, "channel": parent,
                             "process": types.SimpleNamespace(poll=lambda: None, returncode=None)})
        SCALING.send_message(channels[3], {"type": "round_ready", "round": 1})
        SCALING.send_message(channels[1], {"type": "round_ready", "round": 1})
        values = SCALING.collect_messages(children, "round_ready", 1, 1)
        self.assertEqual(len(values), 2)
        with self.assertRaises(TimeoutError):
            SCALING.receive_message(channels[0], time.monotonic() + .01)
        SCALING.send_message(channels[1], {"type": "failure", "error": "recorded child failure"})
        with self.assertRaisesRegex(RuntimeError, "recorded child failure"):
            SCALING.collect_messages(children, "round_ready", 1, 2)

    def test_staged_entrypoint_is_required_before_source_or_device_work(self):
        """The mutable integration script cannot run an already prepared physical-GPU unit."""
        directory = self.prepare(self.arguments(1))
        with patch.object(SCALING, "allocation_proof") as gpu:
            with self.assertRaisesRegex(ValueError, "immutable staged"):
                SCALING.run_parent(directory)
            gpu.assert_not_called()

    def test_failure_cleanup_signals_only_children_with_matching_birth_and_parent(self):
        """A reused PID or another session's child cannot receive a termination signal."""
        children, fields = [], {}
        for index, (birth, parent) in enumerate((("123", os.getpid()), ("456", os.getpid()), ("123", os.getpid() + 1))):
            process = types.SimpleNamespace(pid=90000 + index, poll=lambda: None, terminate=Mock(), kill=Mock(),
                                            wait=Mock(return_value=0))
            children.append({"process": process, "process_start_time": "123"})
            fields[f"/proc/{process.pid}/stat"] = "worker) S " + str(parent) + " " + " ".join(["0"] * 17) + " " + birth
        def path(value):
            """Supply the kernel fields used by the actual ownership check."""
            return types.SimpleNamespace(read_text=lambda: fields[str(value)])
        with patch.object(SCALING, "Path", side_effect=path):
            SCALING.stop_owned_children(children)
        children[0]["process"].terminate.assert_called_once()
        for child in children[1:]:
            child["process"].terminate.assert_not_called()
        for child in children:
            child["process"].kill.assert_not_called()

    def test_interruption_raises_through_cleanup_and_restores_prior_handlers(self):
        """SIGTERM reaches the same owned-child cleanup path as a recorded child failure."""
        previous = {SCALING.signal.SIGTERM: object(), SCALING.signal.SIGINT: object()}
        handlers, cleanup = {}, Mock()
        with patch.object(SCALING.signal, "getsignal", side_effect=lambda number: previous[number]), \
                patch.object(SCALING.signal, "signal", side_effect=lambda number, handler: handlers.__setitem__(number, handler)):
            with self.assertRaises(InterruptedError):
                with SCALING.interruption_cleanup():
                    try:
                        handlers[SCALING.signal.SIGTERM](SCALING.signal.SIGTERM, None)
                    except BaseException:
                        cleanup()
                        raise
        cleanup.assert_called_once()
        self.assertEqual(handlers, previous)


if __name__ == "__main__":
    unittest.main()
