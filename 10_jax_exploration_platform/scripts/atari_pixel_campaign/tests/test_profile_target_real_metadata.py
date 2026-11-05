"""Use accepted component metadata with real new source layouts, without preparing or running a target."""
import copy
import hashlib
import importlib.util
import json
import os
import types
import shutil
import subprocess
from pathlib import Path
import tempfile
import unittest

SCRIPTS = Path(__file__).resolve().parents[1]
PREFIX = "10_jax_exploration_platform/scripts/atari_pixel_campaign/"
RUN = Path("/p/rlprojects/RND_worktrees/merge-atari-cleanrl/10_jax_exploration_platform/benchmark_runs/2026-10-31-00-11_jaxatari_montezuma_pixels_cleanrl_ppo_rnd_environments-per-copy-128_rollout-steps-128_ideas-30-plus-20_steps-per-copy-2000000000_seed-start-1")


def load(name):
    """Load only this test's exact source; no numerical or native APIs are called."""
    spec = importlib.util.spec_from_file_location("real_profile_metadata_" + name, SCRIPTS / (name + ".py"))
    result = importlib.util.module_from_spec(spec);spec.loader.exec_module(result)
    return result


class ActualComponentMetadataInterfaceTests(unittest.TestCase):
    """Actual prior ABI facts are read only; synthetic unit paths grant no operational authority."""
    def test_both_real_hardware_layouts_share_one_exact_process_namespace(self):
        """Join accepted component schemas through the real qualifier/facade/configure API."""
        if os.environ.get("PROFILE_COLD_METADATA_FIXTURE") != "1":
            with tempfile.TemporaryDirectory() as temporary:
                code = Path(temporary) / "code"; (code / "tests").mkdir(parents=True)
                for source in SCRIPTS.glob("*.py"):
                    shutil.copyfile(source, code / source.name)
                script = code / "tests" / Path(__file__).name;shutil.copyfile(__file__, script)
                copied = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in code.rglob("*.py")}
                completed = subprocess.run(["/usr/bin/python3", "-I", "-S", "-B", str(script), "-v"],
                    env={"PATH": "/usr/bin:/bin", "PROFILE_COLD_METADATA_FIXTURE": "1"},
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=180)
                self.assertEqual(completed.returncode, 0, (completed.stdout + completed.stderr).decode())
                self.assertTrue(all(hashlib.sha256(Path(path).read_bytes()).hexdigest() == sha for path, sha in copied.items()))
            return
        q, profile = load("profile_target_qualification"), load("profile_selected")
        n = json.loads((RUN / "next_actions.json").read_text())
        component_units = {"jaguar03_rtx_a4500": RUN / "data/2026-11-05-06-39_native-process-control_jaguar03_python-3.12_gpu-count-1_processor-count-1_host-memory-gib-2_total-seconds-300_work-seconds-240_cleanup-seconds-60_environment-transitions-0_training-updates-0",
            "h100": RUN / "data/2026-11-05-01-01_process-diagnostic_serval02-card-1_uuid-fed69954-5a71-dbb2-ee91-544e0eb6b8a3_python-3.12_native-process-control-true_helpers-7_total-seconds-300_work-seconds-240_cleanup-seconds-60_environment-transitions-0_training-updates-0"}
        component_reviews = {"jaguar03_rtx_a4500": RUN / "remote_setup/2026-11-05-06-39_native-process-control_jaguar03_python-3.12_gpu-count-1_processor-count-1_host-memory-gib-2_total-seconds-300_work-seconds-240_cleanup-seconds-60_environment-transitions-0_training-updates-0/independent_actual_native_component_review.json",
            "h100": Path(n["native_direct_process_diagnostic_source"]["independent_actual_review"]["path"])}
        original = json.loads((Path(n["h100_backend_package_full_qualification_preparation"]["unit"]) / "measurement_plan.json").read_text())
        accepted_profile = json.loads((RUN / "data/2026-11-05-05-49_common-timing_cache-room-backgrounds-true_render-final-two-only-true_h100-nvl_serval02-card-1_copies-1_environments-128_rollout-steps-128_epochs-4_minibatches-4_warmup-rollouts-50_paired-rounds-7_updates-per-arm-10_seed-start-1/measurement_plan.json").read_text())
        helper_files = {PREFIX + p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in SCRIPTS.glob("*.py")}
        for hardware in ("jaguar03_rtx_a4500", "h100"):
            with self.subTest(hardware=hardware), tempfile.TemporaryDirectory() as temporary:
                run = Path(temporary);unit = run / "data/explicit-nonoperational-source-interface-fixture";unit.mkdir(parents=True)
                prior = json.loads((component_units[hardware] / "measurement_plan.json").read_text())
                plan = {"kind": q.KIND, "campaign_run_dir": str(run), "unit_id": unit.name,
                    "hardware_class": hardware, "gpu_uuid": None if hardware == "jaguar03_rtx_a4500" else "GPU-fed69954-5a71-dbb2-ee91-544e0eb6b8a3",
                    "python": prior["python"], "source_root": original["source_root"], "source_commit": q.SOURCE,
                    "core": original["core"], "paired_source": original["paired_source"],
                    "configuration": copy.deepcopy(accepted_profile["configuration"]), "runtime_environment": original["runtime_environment"],
                    "asset_record": original["asset_record"], "copy_seeds": [1], "profile_updates": 1,
                    "profile_tool": prior["profile_tool"], "helpers": {"files": helper_files}}
                if hardware == "jaguar03_rtx_a4500":
                    original_units = list((RUN / "data").glob("2026-11-01-13-50_original-qualification_explicit-pytest-root_jaguar03_*"))
                    self.assertEqual(len(original_units), 1)
                    actual_original = json.loads((original_units[0] / "measurement_plan.json").read_text())
                    plan["configured_isolation"] = copy.deepcopy(actual_original["configured_isolation"])
                    plan["configured_isolation"]["consumer"] = "component_profile"
                else:
                    plan["direct_binding"] = copy.deepcopy(original["direct_binding"])
                    plan["direct_binding"]["consumer"] = "component_profile"
                parent = {"kind": q.CONTROLLER_KIND, "protocol": q.PROTOCOL, "profile_unit": str(unit),
                    "campaign_run_dir": str(run), "source_commit": q.SOURCE, "target_fields": q.target_fields(plan),
                    "scientific_credit": 0, "admission_granted": False, "qualification_time_budget": dict(q.QUALIFICATION_BUDGET),
                    "controller_maximum_seconds": 10800}
                parent_path = run / "explicit_nonoperational_parent.json";parent_path.write_bytes(q.encoded(parent))
                plan["profile_target_qualification"] = {"schema_version": 1, "protocol": q.PROTOCOL, "parent": q.reference(parent_path)}
                process = profile.target_process_reader(selected=True)
                result = component_units[hardware] / "result.json"
                process.configure_plan(plan, result, component_reviews[hardware])
                self.assertIs(profile.target_process_reader(plan), process)
                self.assertIs(q.adjacent(plan, "profile_target_process"), process)
                self.assertIs(process.adjacent(plan, "profile_target_qualification"), process.adjacent(plan, "profile_target_qualification"))
                facade = q.adjacent(plan, "profile_target_direct_process")
                self.assertIs(facade.adjacent_helper(plan, "profile_target_process"), process)
                self.assertEqual(process.budget(plan), q.QUALIFICATION_BUDGET)
                # This uninitialized own-CPU object checks namespace ownership only; it never creates a provider.
                runtime = process.Runtime.__new__(process.Runtime)
                observer = types.SimpleNamespace(native_process_runtime=runtime)
                runtime.pid, runtime.completed, runtime.plan, runtime.observer = os.getpid(), False, plan, observer
                self.assertIs(q.adjacent(plan, "profile_target_process").runtime_observer(plan, observer), observer)
                self.assertIs(facade.adjacent_helper(plan, "profile_target_process").Runtime, process.Runtime)
                self.assertFalse(process.policy(plan)["prior_component_qualifies_current_adapter"])
                self.assertEqual(plan["maximum_active_seconds"], 10200)
                self.assertFalse(any((unit / name).exists() for name in ("measurement_plan.json", "result.json", "profile_target_qualification_result.json")))


if __name__ == "__main__":
    unittest.main()
