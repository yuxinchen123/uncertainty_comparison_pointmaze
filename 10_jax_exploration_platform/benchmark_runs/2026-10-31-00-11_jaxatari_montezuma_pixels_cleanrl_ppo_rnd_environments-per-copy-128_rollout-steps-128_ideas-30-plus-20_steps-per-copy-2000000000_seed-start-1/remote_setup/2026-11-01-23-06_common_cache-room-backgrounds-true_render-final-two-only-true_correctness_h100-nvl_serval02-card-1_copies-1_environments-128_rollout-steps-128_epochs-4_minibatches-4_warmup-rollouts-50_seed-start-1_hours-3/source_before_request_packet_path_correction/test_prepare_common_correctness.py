"""Metadata refusal controls; no actual history, qualification result, device or host read."""

import copy
import importlib.util
import json
from pathlib import Path
import unittest

P = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("fresh_common_correctness_preparation", P / "prepare_common_correctness.py")
preparation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(preparation)


class ConfigurationContractTests(unittest.TestCase):
    """Use only complete copied metadata, never execute preparation or numerical gates."""

    def setUp(self):
        self.request = preparation.read(P / "preparation_request.json")
        self.qualified = {"configuration": copy.deepcopy(self.request["reference_configuration"]),
            "source_root": self.request["numerical_source_root"], "source_commit": self.request["numerical_source_commit"],
            "python": self.request["python"], "runtime_environment": self.request["runtime_environment"],
            "asset_record": self.request["asset_record"], "helpers": {
                "source_root": self.request["helper_source_root"], "source_commit": self.request["helper_source_commit"]}}
        self.qualified["configuration"]["learner"]["n_copies"] = 1

    def refuse(self, mutate):
        mutate(self.request)
        with self.assertRaises((ValueError, KeyError)):
            preparation.configuration_contract(self.request, self.qualified)

    def test_exact_original_plus_two_flags(self):
        self.assertEqual(preparation.configuration_contract(self.request, self.qualified),
            self.request["configuration"]["learner"]["n_copies"])

    def test_changed_warmup(self):
        self.refuse(lambda x: x["configuration"]["learner"].update(observation_warmup_iterations=49))

    def test_changed_total_horizon(self):
        self.refuse(lambda x: x["configuration"]["learner"].update(total_timesteps=1_000_000))

    def test_changed_environment_count(self):
        self.refuse(lambda x: x["configuration"]["environment"].update(num_envs=64))

    def test_changed_reference(self):
        self.refuse(lambda x: x["reference_configuration"]["learner"].update(update_epochs=3))

    def test_missing_cache_change(self):
        self.refuse(lambda x: x["configuration"]["environment"].update(cache_room_backgrounds=False))

    def test_extra_environment_change(self):
        self.refuse(lambda x: x["configuration"]["environment"].update(fuse_pixel_preprocessing=True))

    def test_seed_substitution(self):
        self.refuse(lambda x: x.update(copy_seeds=[7] * len(x["copy_seeds"])))

    def test_boolean_copy_count(self):
        self.refuse(lambda x: x["configuration"]["learner"].update(n_copies=True))

    def test_other_copy_count(self):
        self.refuse(lambda x: x["configuration"]["learner"].update(n_copies=3))

    def test_guard_source_cannot_replace_qualified_source(self):
        self.refuse(lambda x: x.update(helper_source_commit="c" * 40))

    def test_other_physical_selector(self):
        self.refuse(lambda x: x.update(gpu_uuid="GPU-other"))

    def test_extra_helper(self):
        self.refuse(lambda x: x.update(expected_staged_helper_count=27))

    def test_confirmation_mode(self):
        self.refuse(lambda x: x.update(mode="confirmation"))

    def test_timing_requested(self):
        self.refuse(lambda x: x.update(timing_rounds_executed_by_correctness_mode=True))

    def test_wrong_round_protocol(self):
        self.refuse(lambda x: x.update(rounds=11))

    def test_changed_runtime(self):
        self.refuse(lambda x: x.update(runtime_environment={"JAX_ENABLE_X64": "false"}))

    def test_changed_original_source(self):
        self.refuse(lambda x: x.update(numerical_source_commit="f" * 40))


if __name__ == "__main__":
    unittest.main()
