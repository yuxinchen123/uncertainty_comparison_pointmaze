"""Build complete synthetic final-phase records with real detached metadata-only Git sources."""

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
from unittest.mock import patch

import measure_selected as selected
import post_comparison_validation as validation
import short_learning


SCRIPTS = Path(__file__).resolve().parents[1]


class FinalPhaseFixture:
    """Keep synthetic numeric observations separate from actual source and byte-identity checks."""

    def __init__(self, source_root=None, asset_record=None):
        """Create full study evidence and all predeclared final requirements without importing arrays."""
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.source = self.root / "source" if source_root is None else Path(source_root)
        self.scripts = self.source / "10_jax_exploration_platform/scripts/atari_pixel_campaign"
        if source_root is None:
            self.scripts.mkdir(parents=True)
            for path in [*SCRIPTS.glob("*.py"), SCRIPTS / "worker_environment.sh"]:
                shutil.copyfile(path, self.scripts / path.name)
            pixel_source = SCRIPTS.parents[1] / "src/exploration_platform/atari_pixels"
            shutil.copytree(pixel_source, self.source / "10_jax_exploration_platform/src/exploration_platform/atari_pixels")
            self.git("init", "-q")
            self.git("add", ".")
            self.git("-c", "user.name=Metadata fixture", "-c", "user.email=metadata@example.invalid",
                     "-c", "core.hooksPath=/dev/null", "commit", "-qm", "Create complete final metadata fixture")
        self.commit = self.git("rev-parse", "HEAD").strip()
        self.git("checkout", "-q", "--detach", self.commit)
        spec = importlib.util.spec_from_file_location("complete_final_raw_fixtures", SCRIPTS / "tests/test_post_comparison_validation.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.raw = module.PostComparisonTests("runTest")
        with patch.object(module, "SCRIPTS", self.scripts):
            self.raw.setUp()
        registry_path = Path(self.raw.specification["study_registry"])
        registry = json.loads(registry_path.read_bytes())
        entries = registry["initial_ideas"] + [item for batch in registry["adaptive_batches"] for item in batch["hypotheses"]]
        for entry in entries:
            for measurements in entry["results"].values():
                for measurement in measurements:
                    measurement.update(implementation_distinct=True, measurement_valid=True, correctness_passed=True)
        self.write(registry_path, registry)
        self.specification = copy.deepcopy(self.raw.specification)
        self.specification["final_protocol"] = copy.deepcopy(validation.FINAL_PROTOCOL)
        self.runtime = self.raw.runtime
        self.run = self.root / "campaign"
        self.counter = 50000
        self.core = selected.diagnostic.core_identity(self.source, self.commit)
        self.helpers = self.receipt(selected.HELPERS)
        self.paired = self.receipt(("paired_benchmark.py", "ideas.py"))
        if asset_record is None:
            manifest = self.write(self.root / "asset_manifest.json", {"schema_version": 1,
                "source_commit": validation.JAXTARI_COMMIT, "files": {"player.npy": "a" * 64}})
            self.assets = {"root": str(self.root / "assets"), "manifest": str(manifest), "sha256": self.sha(manifest)}
        else:
            self.assets = asset_record
        asset_manifest = json.loads(Path(self.assets["manifest"]).read_bytes())
        self.asset_identity = {"manifest_sha256": self.assets["sha256"], "source_commit": asset_manifest["source_commit"],
            "file_count": len(asset_manifest["files"]), "tree_digest": validation.canonical_digest(asset_manifest["files"])}
        self.completed_study = self.study_proof()
        self.counter = 100000
        for context in validation.CONTEXTS:
            old = self.specification["contexts"][context]
            correctness = self.measurement(context, "correctness", self.raw.comparison(context))
            confirmation = self.measurement(context, "confirmation", self.raw.comparison(context), final=True)
            packing, repeats = [], []
            for index, (copies, seconds) in enumerate(((1, 1.), (2, 1.9))):
                packing.append(str(self.measurement(context, "packing-point", self.raw.packing(context, copies, seconds, 100 + index))))
                repeats.append(str(self.measurement(context, "packing-point", self.raw.packing(context, copies, seconds, 200 + index))))
            learning = self.learning(context, confirmation)
            self.specification["contexts"][context] = {**old, "correctness_record": str(correctness),
                "confirmation_record": str(confirmation), "packing_records": packing,
                "packing_confirmation_records": repeats, "learning_record": str(learning),
                "learning_analysis_record": str(learning.parent / "independent_analysis.json")}
        excluded = self.raw.packing("jaguar03_rtx_a4500", 8, 1., 400)
        for row in excluded["rounds"]:
            for sample in row["memory"]["device_memory_samples"]:
                sample.update(used_bytes=900 * 2 ** 20, free_bytes=100 * 2 ** 20)
        eligibility = self.measurement("jaguar03_rtx_a4500", "packing-point", excluded)
        self.specification["fixed_total_scaling"] = {"status": "not_applicable",
            "eligibility_kind": "completed_below_memory_headroom", "eligibility_record": str(eligibility)}
        self.fixed_groups = (self.fixed_total_group(1), self.fixed_total_group(8))

    def close(self):
        """Delete only this synthetic fixture and its original synthetic study records."""
        self.raw.doCleanups()
        self.temporary.cleanup()

    def git(self, *arguments):
        """Run local Git only for the synthetic detached source identity."""
        return subprocess.check_output(["git", "-C", str(self.source), *arguments], text=True)

    def sha(self, path):
        """Hash literal fixture bytes, including each separately prepared plan."""
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()

    def write(self, path, value):
        """Write a complete independently hashable synthetic document."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, allow_nan=False) + "\n")
        return path

    def receipt(self, names):
        """Bind actual copied helper bytes to the synthetic detached Git source."""
        prefix = "10_jax_exploration_platform/scripts/atari_pixel_campaign/"
        return {"source_root": str(self.source), "source_commit": self.commit,
                "files": {prefix + name: self.sha(self.scripts / name) for name in names}}

    def prepare(self, directory, plan):
        """Freeze a real prepared-plan hash and exact staged helper copies."""
        path = self.write(directory / "measurement_plan.json", plan)
        self.write(directory / "measurement_prepared.json", {"plan_sha256": self.sha(path)})
        for name in selected.HELPERS:
            destination = directory / "code" / name
            destination.parent.mkdir(exist_ok=True)
            shutil.copyfile(self.scripts / name, destination)

    def measurement(self, context, mode, record, final=False):
        """Produce full source/asset/process sidecars with synthetic actual-style numeric rows."""
        self.counter += 1
        directory = self.run / "data" / (context + "_" + mode + "_" + str(self.counter))
        record = copy.deepcopy(record)
        configuration = record["configuration"] if mode == "packing-point" else record["candidate"]
        copies = configuration["learner"]["n_copies"]
        hostname, model = ("serval02", "NVIDIA H100 NVL") if context == "h100" else ("jaguar03", "NVIDIA RTX A4500")
        hardware = {"hostname": hostname, "pid": self.counter, "process_start_time": str(self.counter),
            "gpu_uuid": "GPU-one", "gpu_query": f"GPU-one, {model}, 1000 MiB, synthetic-driver",
            "runtime_environment": {**self.runtime, "JAX_ENABLE_COMPILATION_CACHE": "false"}}
        record.update(hardware=hardware, source_commit=self.commit, hardware_class=context, copies=copies, helpers=self.helpers,
            installed_backend={"synthetic_actual_backend": context}, measurement_mode=mode,
            runtime_environment=self.runtime, copy_seeds=list(range(1, copies + 1)),
            execution_identity={"hostname": hostname, "pid": self.counter, "process_start_time": str(self.counter),
                "started_at_seconds": float(self.counter), "finished_at_seconds": float(self.counter + 1)})
        rounds, minimum = (11, 20) if final else (7, 10)
        if mode == "correctness":
            record["rounds"] = []
        elif final:
            record["rounds"] = [copy.deepcopy(record["rounds"][index % 7]) for index in range(11)]
            for index, pair in enumerate(record["rounds"]):
                pair["order"] = ["reference", "candidate"] if index % 2 == 0 else ["candidate", "reference"]
                for arm in ("reference", "candidate"):
                    pair[arm].update(iterations=20, training_transitions=copies * 128 * 128 * 20)
            record["synchronized_per_update_latency"] = {}
            for arm in ("reference", "candidate"):
                declared = short_learning.latency_protocol(record[arm])
                window = declared["complete_window_updates"]
                durations = [.1, .2, .3] * window
                warmup = [{"updates": window, "seconds": .1, "compiled_signatures": [1]}] * 2
                reports = []
                for index in range(3):
                    published = [{"iteration": index * window + count * declared["selected_report_every_iterations"],
                        "window_updates": declared["selected_report_every_iterations"], "metrics": {
                            "steps_per_policy": (index * window + count * declared["selected_report_every_iterations"]) * 16384}}
                        for count in range(1, declared["report_rows_per_window"] + 1)]
                    reports.append({"passed": True, "from_iteration": index * window, "to_iteration": (index + 1) * window,
                        "state_and_final_metrics": {"leaves": 100}, "selected_reports": {"leaves": 10 if published else 0},
                        "report_rows": len(published), "published_reports": published})
                record["synchronized_per_update_latency"][arm] = {"calls": len(durations), "updates_per_call": 1,
                    "protocol": declared, "seconds_per_synchronized_call": durations,
                    "median_seconds_per_synchronized_call": .2, "mean_seconds_per_synchronized_call": sum(durations) / len(durations),
                    "median_amortized_seconds_per_update": .2, "synchronization_boundary": "each_original_complete_update",
                    "selected_iterations_per_call": record[arm]["execution"]["iterations_per_call"],
                    "warmup": warmup, "selected_executor_warmup": warmup, "window_agreement": reports}
            record["synchronized_per_update_latency_memory"] = self.raw.row()["memory"]
            self.write(directory / "single_update_latency_memory.json", record["synchronized_per_update_latency_memory"])
        record["protocol"] = {"rounds": rounds, "requested_minimum_updates_per_round": minimum, "actual_updates_per_round": minimum}
        plan = {"schema_version": 1, "kind": "selected_configuration_measurement", "mode": mode,
            "campaign_run_dir": str(self.run), "unit_id": directory.name, "source_root": str(self.source),
            "source_commit": self.commit, "hardware_class": context, "runtime_environment": self.runtime,
            "core": self.core, "paired_source": self.paired, "helpers": self.helpers,
            "configuration": configuration, "reference_configuration": record.get("reference"),
            "gpu_uuid": "GPU-one", "asset_record": self.assets, "copy_seeds": record["copy_seeds"],
            "compilation_cache_policy": "disabled", "rounds": rounds, "minimum_updates_per_round": minimum, "iterations": minimum}
        if final:
            plan["final_confirmation_protocol"] = copy.deepcopy(validation.FINAL_PROTOCOL)
            plan["single_update_latency_protocol"] = {arm: short_learning.latency_protocol(record[arm]) for arm in ("reference", "candidate")}
        if final or mode == "learning-check":
            plan["completed_study"] = self.completed_study
            plan["completed_study_manifest"] = None
        self.prepare(directory, plan)
        self.write(directory / "hardware.json", hardware)
        self.write(directory / "installed_backend.json", record["installed_backend"])
        self.write(directory / "asset_receipt.json", {"root": self.assets["root"], **self.asset_identity})
        return self.write(directory / "result.json", record)

    def study_proof(self):
        """Freeze full synthetic fifty-idea coverage with literal owner snapshots and separate review."""
        directory = self.root / "completed_study"
        registry_path = Path(self.raw.specification["study_registry"])
        registry = json.loads(registry_path.read_bytes())
        evidence = validation.Evidence()
        validation.require_fifty_investigated(registry, evidence)
        accepted = {}
        self.accepted_combination_inputs = {}
        for context in validation.CONTEXTS:
            inputs = {name: str(self.measurement(context, mode, self.raw.comparison(context)))
                      for name, mode in (("correctness_record", "correctness"), ("confirmation_record", "confirmation"))}
            self.accepted_combination_inputs[context] = inputs
            common = {**inputs, "source_commit": self.commit, "hardware_class": context, "runtime_environment": self.runtime}
            checked = validation.validate_combination(common, evidence)
            accepted[context] = {"source_commit": self.commit, "configuration": checked["configuration"],
                "records": {name: {"path": path, "sha256": self.sha(path)} for name, path in inputs.items()}}
        snapshot = self.write(directory / "completed_fifty_registry.json", registry)
        script = directory / "synthetic_owned_monitor.sbatch"
        script.write_text('#!/bin/sh\n# Synthetic metadata fixture; never submitted.\n')
        job, row = "123", ["123", "synthetic_monitor", self.commit, str(script)]
        # The original registry parent, not the detached source, owns the synthetic study ID.
        script = registry_path.parent / "synthetic_owned_monitor.sbatch"
        script.write_text('#!/bin/sh\n# Synthetic metadata fixture; never submitted.\n')
        row[-1] = str(script)
        ids = directory / "completed_fifty_submitted_jobids_uva.txt"
        ids.write_text(job + '\n')
        rows = directory / "completed_fifty_submissions.tsv"
        rows.write_text('\t'.join(row) + '\n')
        records = evidence.records() + [{"path": str(snapshot), "sha256": self.sha(snapshot)}]
        last_finish = max(json.loads(Path(row["result_path"]).read_bytes())["execution_identity"]["finished_at_seconds"]
            for item in registry["adaptive_batches"][-1]["hypotheses"] for context in validation.CONTEXTS for row in item["results"][context])
        proof = {"schema_version": 1, "kind": "completed_fifty_idea_study_proof", "source_commit": self.commit,
            "runtime_environment": self.runtime, "registry": {"path": str(snapshot), "sha256": self.sha(snapshot),
                "original_path": str(registry_path)}, "helpers": self.receipt((*selected.HELPERS, "controller.py", "ideas.py")),
            "investigated_hypotheses": 50, "scientific_recomputation_on_owned_compute": True,
            "accepted_combinations": accepted, "last_adaptive_case_finished_at_seconds": last_finish,
            "evidence": records, "hypothesis_count_contribution": 0, "comparison_count_contribution": 0, "final_training_credit": 0,
            "owner": {"cluster": "uva", "job_id": job, "submission_row": row, "uid": 123,
                "snapshots": {"submitted_jobids_uva.txt": {"path": str(ids), "sha256": self.sha(ids)},
                    "submissions.tsv": {"path": str(rows), "sha256": self.sha(rows)}},
                "submission_script": {"path": str(script), "sha256": self.sha(script)}},
            "producer": {"hostname": "synthetic_owned_compute", "pid": 100, "process_start_time": "100",
                "slurm_job_id": job, "started_at_seconds": 60000., "finished_at_seconds": 60001.}}
        proof["verification_digest"] = validation.canonical_digest(proof)
        proof_path = self.write(directory / "completed_fifty_idea_proof.json", proof)
        binding = {"proof": {"path": str(proof_path), "sha256": self.sha(proof_path)}}
        review = self.write(directory / "independent_review.json", {"schema_version": 1,
            "kind": "independent_completed_fifty_idea_review", "proof": binding["proof"],
            "decision": "accepted_completed_fifty_idea_evidence", "reviewer_identity": "independent synthetic metadata reviewer",
            "independent_of_producer": True, "unresolved_anomalies": [], "reviewed_at_seconds": 60002.})
        binding["review"] = {"path": str(review), "sha256": self.sha(review)}
        return binding

    def learning(self, context, confirmation_path):
        """Build complete finite curves and an independently bound comparative-analysis receipt."""
        confirmation = json.loads(confirmation_path.read_bytes())
        record = {"reference": confirmation["reference"], "candidate": confirmation["candidate"]}
        path = self.measurement(context, "learning-check", record)
        return self.complete_learning(path)

    def complete_learning(self, path):
        """Complete either local or target synthetic learning evidence while retaining its actual-shape sidecars."""
        result = json.loads(path.read_bytes())
        result.pop("rounds", None)
        directory = path.parent
        context, copies, seeds = result["hardware_class"], result["candidate"]["learner"]["n_copies"], result["copy_seeds"]
        declared = short_learning.protocol([result["reference"], result["candidate"]])
        plan = json.loads((directory / "measurement_plan.json").read_bytes())
        plan["learning_protocol"] = declared
        plan["completed_study"] = self.completed_study
        plan["completed_study_manifest"] = None
        self.prepare(directory, plan)
        rows = []
        for index in range(1, declared["updates"] + 1):
            for arm in ("reference", "candidate"):
                metrics = {name: [float(index) / 100] * copies for name in short_learning.METRICS}
                metrics["steps_per_policy"] = index * 128 * 128
                metrics["episode_count"] = [1] * copies
                rows.append({"variant": arm, "update_in_check": index, "iteration": index,
                    "optimizer_step": index * 16, "copy_seeds": seeds, "floating_state_nonfinite_count": 0,
                    "observation_variance_mean": [1.] * copies,
                    "reward_variance": [1.] * copies, "metrics": metrics, "boundary_action_histogram": [[64, 64] + [0] * 16] * copies,
                    "observation_count": [float(index + 100)] * copies, "reward_count": [float(index + 100)] * copies,
                    "boundary_action_bins": list(range(18)), "boundary_action_unknown_count": [0] * copies,
                    "boundary_room_histogram": [[128] + [0] * 32] * copies, "boundary_room_bins": list(range(33)),
                    "boundary_room_unknown_count": [0] * copies, "boundary_inventory_sum": [[0] * 4] * copies,
                    "environment_key": [[[seed, lane] for lane in range(128)] for seed in seeds],
                    "frame_count": [[index * 512] * 128] * copies, "step_count": [[index * 128] * 128] * copies,
                    "reset_count": [[0] * 128] * copies})
        curve_path = directory / "learning_curves.jsonl"
        curve_path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
        precision = {"all_state_finite": True, "network_dtype": "float32", "statistics_dtype": "float64",
                     "network_leaves": 10, "statistics_leaves": 6, "full_state_leaves": 100}
        arm = {"completed_updates": declared["updates"], "optimizer_step": declared["updates"] * 16,
            "parameter_change_scope": "initial_versus_final_existing_host_snapshots",
            "agent_change_squared": [.01] * copies, "predictor_change_squared": [.02] * copies,
            "frozen_target_unchanged": True, "precision": {"initial": precision, "latest": precision},
            "replay": {"passed": True, "window_updates": declared["replay_window_updates"],
                "from_iteration": declared["updates"] - declared["replay_window_updates"], "to_iteration": declared["updates"],
                "scope": "same_implementation_same_full_state_inputs", "agreement": {"leaves": 100},
                "agreement_policy": short_learning.REPLAY_AGREEMENT},
            "observer_agreement": {"passed": True, "window_updates": declared["complete_window_updates"],
                "scope": "instrumented_window_versus_actual_selected_Executor_run_same_full_state",
                "automatic_tolerance_adjustment": False, "state_and_final_metrics": {"leaves": 100},
                "report_rows": 0, "selected_reports": {"leaves": 0}, "agreement_policy": short_learning.OBSERVER_AGREEMENT}}
        memory = self.raw.row(copies=copies, gpu=result["hardware"]["gpu_uuid"])["memory"]
        self.write(directory / "learning_memory.json", memory)
        result.update(completed=True, learning_protocol=declared, arms={"reference": arm, "candidate": arm},
            training_start_iteration={"reference": 0, "candidate": 0}, memory=memory,
            curves={"path": str(curve_path), "sha256": self.sha(curve_path)}, reports=None,
            learning_equivalence_established=False, zero_reward_only_evidence=False,
            independent_comparative_analysis_required=True,
            warmup={name: [{"seconds": .1, "cache_size": [1]}] * 2 for name in ("reference", "candidate")})
        self.write(path, result)
        findings = {name: "Synthetic finite noncollapsed dynamics reviewed independently."
            for name in ("losses", "policy_updates", "predictor_updates", "intrinsic_statistics", "action_distributions",
                         "episodes_and_exploration", "trajectory_differences")}
        self.write(directory / "independent_analysis.json", {"schema_version": 1,
            "kind": "independent_short_learning_analysis", "decision": "accepted_short_learning_checks",
            "result_record": {"path": str(path), "sha256": self.sha(path)}, "curves": result["curves"],
            "source_commit": self.commit, "hardware_class": context,
            "configuration_digest": validation.canonical_digest(result["candidate"]), "copy_seeds": seeds,
            "learning_equivalence_established": False, "zero_reward_only_evidence": False,
            "comparative_findings": findings, "reviewer": {"identity": "independent synthetic metadata reviewer",
                "independent_of_producer": True}, "unresolved_anomalies": []})
        return path

    def fixed_total_group(self, cards, total=8):
        """Create independently hashed one-or-eight worker groups with equal total policy identities."""
        self.counter += 100
        parent = self.counter
        copies = total // cards
        configuration = copy.deepcopy(self.raw.configuration)
        configuration["learner"]["n_copies"] = copies
        directory = self.run / "data" / ("fixed_total_policies-" + str(total) + "_gpu-count-" + str(cards))
        roster = ["GPU-one"] + ["GPU-" + str(index) for index in range(1, cards)]
        seeds = [list(range(1 + index * copies, 1 + (index + 1) * copies)) for index in range(cards)]
        protocol = {"kind": "fixed_total_copies", "total_copies": total,
            "global_copy_seeds": list(range(1, total + 1)), "gradient_averaging": False,
            "measurement_scope": "synchronized_gpu_group_wall_time"}
        helpers = self.receipt(("measure_gpu_scaling.py", *selected.HELPERS))
        plan = {"schema_version": 1, "kind": "synchronized_a4500_scaling_measurement",
            "campaign_run_dir": str(self.run), "unit_id": directory.name, "source_root": str(self.source),
            "source_commit": self.commit, "hardware_class": "jaguar03_rtx_a4500", "runtime_environment": self.runtime,
            "configuration": configuration, "core": self.core, "paired_source": self.paired, "helpers": helpers,
            "gpu_count": cards, "gpu_uuids": roster, "copy_seeds": seeds, "asset_record": self.assets,
            "rounds": 7, "minimum_updates_per_round": 10, "iterations": 10,
            "scaling_protocol": protocol, "compilation_cache_policy": "disabled",
            "completed_study": self.completed_study, "completed_study_manifest": None}
        plan_path = self.write(directory / "scaling_plan.json", plan)
        self.write(directory / "scaling_prepared.json", {"plan_sha256": self.sha(plan_path)})
        for relative in helpers["files"]:
            destination = directory / "code" / Path(relative).name
            destination.parent.mkdir(exist_ok=True)
            shutil.copyfile(self.source / relative, destination)
        assignment = {"hostname": "jaguar03", "parent_pid": parent, "parent_process_start_time": str(parent),
            "gpu_uuids": roster, "plan_sha256": self.sha(plan_path), "allocation": {"slurm_job_id": "9999"}}
        assignment_path = self.write(directory / "assignment.json", assignment)
        observations, worker_receipts = [], []
        for index, gpu in enumerate(roster):
            pid = parent + index + 1
            hardware = {"hostname": "jaguar03", "pid": pid, "process_start_time": str(pid),
                "gpu_uuid": gpu, "gpu_query": f"{gpu}, NVIDIA RTX A4500, 1000 MiB, synthetic-driver",
                "runtime_environment": {**self.runtime, "JAX_ENABLE_COMPILATION_CACHE": "false"}, "slurm_job_id": "9999"}
            rows = [{**self.raw.row(copies=copies, gpu=gpu), "gpu_uuid": gpu, "copy_seeds": seeds[index],
                "start_time_seconds": 200001. + round_index * 3, "end_time_seconds": 200002. + round_index * 3,
                "pid": pid, "process_start_time": str(pid)} for round_index in range(7)]
            worker_directory = directory / "workers" / ("physical_gpu_" + str(index + 1))
            loaded = {"pid": pid, "synthetic_actual_libraries": True}
            worker = {"completed": True, "gpu_uuid": gpu, "copy_seeds": seeds[index], "configuration": configuration,
                "source_commit": self.commit, "runtime_environment": self.runtime, "hardware": hardware,
                "rounds": rows, "loaded_gpu_libraries": loaded}
            worker_path = self.write(worker_directory / "result.json", worker)
            self.write(worker_directory / "hardware.json", hardware)
            self.write(worker_directory / "installed_backend.json", {"synthetic_actual_backend": "jaguar03_rtx_a4500"})
            self.write(worker_directory / "asset_receipt.json", {"root": self.assets["root"], **self.asset_identity})
            self.write(worker_directory / "loaded_gpu_libraries.json", loaded)
            self.write(worker_directory / "warmup.json", [{"seconds": .1, "cache_size": [1]}] * 2)
            observations.append(rows)
            worker_receipts.append({"type": "finished", "gpu_uuid": gpu, "worker_result_sha256": self.sha(worker_path)})
        seconds = 2. if cards == 1 else 1.
        rounds = [{"measurement_scope": "synchronized_gpu_group_wall_time", "start_time_seconds": 200001. + round_index * 3,
            "end_time_seconds": 200001. + round_index * 3 + seconds, "seconds": seconds, "training_transitions": total * 128 * 128 * 10,
            "transitions_per_second_total": total * 128 * 128 * 10 / seconds,
            "workers": [observations[index][round_index] for index in range(cards)]} for round_index in range(7)]
        result = {"completed": True, "source_commit": self.commit, "source_root": str(self.source),
            "hardware_class": "jaguar03_rtx_a4500", "runtime_environment": self.runtime, "configuration": configuration,
            "gpu_count": cards, "gpu_uuids": roster, "copy_seeds": seeds, "rounds": rounds, "helpers": helpers,
            "workers": worker_receipts, "assignment_sha256": self.sha(assignment_path), "scaling_protocol": protocol,
            "normalization_warmup_iterations": 50, "training_start_iteration": 0,
            "protocol": {"rounds": 7, "requested_minimum_updates_per_round": 10, "actual_updates_per_round": 10,
                "group_clock": "one shared host monotonic clock"},
            "execution_identity": {"hostname": "jaguar03", "pid": parent, "process_start_time": str(parent),
                "started_at_seconds": 200000., "finished_at_seconds": 200030.}}
        return self.write(directory / "result.json", result)
