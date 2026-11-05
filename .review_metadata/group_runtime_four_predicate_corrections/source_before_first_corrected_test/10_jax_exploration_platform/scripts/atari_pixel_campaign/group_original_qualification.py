"""Run the unchanged complete original gates and a count-specific full reference update.

The controlled reference counts scale by independent copy count. Test and renderer
fixture work remains explicitly unquantified; this producer gives no study credit.
"""
import contextlib
import importlib.util
import json
import os
from pathlib import Path
import sys

import uva_group_binding as group

configured = group.adjacent("configured_measurement")
selected = group.adjacent("measure_selected")
require = group.require
EXPECTED_DTYPES = configured.EXPECTED_DTYPES
QUALIFICATION = configured.QUALIFICATION
qualification_identity = configured.qualification_identity
qualification_pytest_arguments = configured.qualification_pytest_arguments
validate_test_collection = configured.validate_test_collection
verify_qualification_sources = configured.verify_qualification_sources


def validate_configuration(plan):
    """Require the complete original configuration with only its explicit independent copy count changed."""
    group.validate_plan(plan)
    require(plan["mode"] == "qualification", "only the separate zero-credit group qualification may run original gates")
    configuration = selected.validation.configuration(plan["configuration"])
    original = selected.diagnostic.default_configuration(Path(plan["source_root"]))
    copies = configuration["learner"]["n_copies"]
    require(type(copies) is int and copies > 0, "qualification copy count must be positive")
    original["learner"]["n_copies"] = copies
    # Before: one-copy original defaults; after: the identical complete defaults with the declared copy count.
    require(configuration["learner"] == json.loads(json.dumps(original["learner"]))
        and configuration["environment"] == json.loads(json.dumps(original["environment"]))
        and configuration == plan["reference_configuration"]
        and configuration["execution"] == {"compilation_mode": "whole_iteration", "iterations_per_call": 1,
            "report_readout": "none", "report_every_iterations": 10},
        "qualification must retain every original learner/environment option and one full reference update")
    return configuration


def run(directory, plan, paired, jax, Config, Trainer, Environment):
    """Execute every original gate followed by one full-size reference update, with no automatic admission."""
    import contextlib
    import pytest
    import train_optimized as training
    value = validate_configuration(plan)
    verify_qualification_sources(plan)
    directory = Path(directory)
    root = Path(plan["source_root"]) / "10_jax_exploration_platform"
    arguments = qualification_pytest_arguments(plan)
    gates, collected, collection_reports, deselected, collection_rootdir = [], [], [], [], []
    class Reports:
        """Retain each real collected test and its call/fixture outcome."""
        def pytest_runtest_logreport(self, report):
            """Record each executed test phase rather than treating pytest exit zero as complete evidence."""
            gates.append({"nodeid": report.nodeid, "phase": report.when, "outcome": report.outcome,
                          "duration_seconds": report.duration})
        def pytest_collection_finish(self, session):
            """Save the actual complete collected matrix and source root without selecting cases."""
            collected.extend(item.nodeid for item in session.items)
            collection_rootdir.append(str(session.config.rootpath))
        def pytest_collectreport(self, report):
            """Keep collection failures/skips explicit rather than losing them before test calls."""
            collection_reports.append({"nodeid": report.nodeid, "outcome": report.outcome})
        def pytest_deselected(self, items):
            """Refuse any external plugin or setting that removes original cases."""
            deselected.extend(item.nodeid for item in items)
    # All tests run inside the already physically bound process; no unbound numerical subprocess is created.
    with (directory / "original_tests.stdout").open("x") as stdout, (directory / "original_tests.stderr").open("x") as stderr:
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            returncode = int(pytest.main(arguments, plugins=[Reports()]))
    test_record = {"command": [plan["python"], "-m", "pytest", *arguments],
        "execution": "same already identity-bound process via pytest.main", "returncode": returncode, "reports": gates,
        "collected_nodeids": collected, "collection_rootdir": collection_rootdir[0] if len(collection_rootdir) == 1 else None,
        "collection_reports": collection_reports, "deselected_nodeids": deselected}
    training.atomic_json(directory / "original_tests.json", test_record)
    require(returncode == 0 and gates and all(row["outcome"] == "passed" for row in gates),
        "full original learner/environment gates failed or skipped")
    validate_test_collection(plan, test_record, (directory / "original_tests.stdout").read_text())
    renderer_path = root / "scripts/atari_pixel_campaign/check_rendering.py"
    specification = importlib.util.spec_from_file_location("qualification_original_renderer", renderer_path)
    renderer = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(renderer)
    rendering = renderer.check("all")
    training.atomic_json(directory / "original_renderer.json", rendering)
    require(rendering.get("checks") == {"cache": "passed", "palette": "passed", "fused": "passed"}
        and all(type(rendering.get(key)) is int and rendering[key] == plan["renderer_contract"][key]
            for key in ("rooms", "cases", "synthetic_pixel_cases")), "full original renderer checks did not finish every source-bound case")
    # This controlled full reference work is separate from all numerical work performed by the test suite.
    trainer = Trainer(Config(**value["learner"]), Environment(**value["environment"]))
    state = trainer.initialize(tuple(plan["copy_seeds"]))
    state = trainer.warmup(state)
    before_target = paired.host_copy(state.target)
    before_policy = paired.host_copy(state.parameters["agent"])
    state, metrics = trainer.train_iteration(state)
    jax.block_until_ready((state, metrics))
    import jax.numpy as jnp
    import numpy as np
    leaves = [leaf for leaf in jax.tree.leaves(state) if hasattr(leaf, "dtype") and leaf.dtype.kind in "fc"]
    require(leaves and bool(jax.device_get(jnp.all(jnp.stack([jnp.all(jnp.isfinite(leaf)) for leaf in leaves])))),
        "the full reference state contains nonfinite values")
    groups = {"parameters": state.parameters, "target": state.target, "optimizer_first": state.optimizer.first,
        "optimizer_second": state.optimizer.second, "observation_statistics": state.observation_statistics,
        "reward_statistics": state.reward_statistics}
    dtypes = {name: sorted({str(leaf.dtype) for leaf in jax.tree.leaves(tree)}) for name, tree in groups.items()}
    require(dtypes == EXPECTED_DTYPES
        and int(state.iteration) == 1 and int(state.optimizer.step) == 16,
        "full reference precision or update count changed")
    target_equal = all(np.array_equal(a, b) for a, b in zip(jax.tree.leaves(before_target),
        jax.tree.leaves(paired.host_copy(state.target))))
    policy_changed = any(not np.array_equal(a, b) for a, b in zip(jax.tree.leaves(before_policy),
        jax.tree.leaves(paired.host_copy(state.parameters["agent"]))))
    require(target_equal and policy_changed, "full reference update changed its frozen target or left the policy unchanged")
    return {"completed": True, "qualification_protocol": QUALIFICATION, "all_original_gates_passed": True,
        "asset_record": plan["asset_record"], "original_test_reports": gates,
        "original_collected_test_nodeids": collected, "renderer": rendering, "renderer_contract": plan["renderer_contract"],
        "dtypes": dtypes, "finite_state_leaves": len(leaves), "reference_iteration": int(state.iteration),
        "reference_optimizer_step": int(state.optimizer.step), "frozen_target_unchanged": target_equal,
        "policy_parameters_changed": policy_changed,
        "gate_artifacts": {name: qualification_identity(plan).receipt(directory / name) for name in (
            "original_tests.json", "original_tests.stdout", "original_tests.stderr", "original_renderer.json")},
        "executed_work": {"full_reference_warmup_transitions": 50 * 128 * 128 * value["learner"]["n_copies"],
            "full_reference_training_transitions": 128 * 128 * value["learner"]["n_copies"], "full_reference_updates": 1,
            "full_reference_optimizer_minibatch_updates": 16,
            "controlled_transition_scope": "agent transitions; reset NOOP/FIRE actions and raw game frames are not counted",
            "original_test_and_renderer_work": "complete commands/source/test outcomes/case counts retained; synthetic/partial fixture runtime work is unquantified",
            "whole_job_transition_total_known": False},
        "test_work_accounting_complete": False}

