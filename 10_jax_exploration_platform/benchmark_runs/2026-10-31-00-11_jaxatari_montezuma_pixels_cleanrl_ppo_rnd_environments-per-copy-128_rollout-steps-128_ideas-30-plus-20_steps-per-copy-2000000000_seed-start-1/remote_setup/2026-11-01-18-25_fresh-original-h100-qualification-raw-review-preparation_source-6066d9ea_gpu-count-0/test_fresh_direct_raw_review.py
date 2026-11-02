"""Check the fresh read-only review without actual qualification results."""
import ast
import copy
import importlib.util
import math
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest import mock

ROOT = Path(__file__).absolute().parent
spec = importlib.util.spec_from_file_location("fresh_direct_review", ROOT / "review_direct_raw_then_consumer.py")
review = importlib.util.module_from_spec(spec)
spec.loader.exec_module(review)


def fixture():
    """Build synthetic metadata for extra actual-join controls."""
    plan = {"copy_seeds": [1], "mode": "correctness", "rounds": 7,
        "minimum_updates_per_round": 10, "iterations": 10,
        "configuration": {"environment": {"num_envs": 128}},
        "reference_configuration": {"environment": {"num_envs": 128}}}
    process = {"hostname": "serval02", "pid": 2000, "process_start_time": "1234", "parent_pid": 1000}
    command = ["/usr/bin/env", "-i", "PATH=x", "LANG=x", "TZ=x", "CUDA_VISIBLE_DEVICES=x",
        "PYTHONDONTWRITEBYTECODE=1", "PYTHONNOUSERSITE=1", "/private/python", "-I", "-B", "/controller",
        "--launch", "--publication-commit", review.CONTROLLER_PUBLICATION]
    fields = ["S"] + ["0"] * 21
    fields[19] = "999"
    parent = {"stat": "1000 (python) " + " ".join(fields)}
    result = {"execution_identity": {**process, "started_at_seconds": 1.0, "finished_at_seconds": 2.0},
        "copy_seeds": [1], "measurement_mode": "correctness",
        "protocol": {"rounds": 7, "requested_minimum_updates_per_round": 10, "actual_updates_per_round": 10},
        "configuration": copy.deepcopy(plan["configuration"])}
    spawn = {"schema_version": 1, "kind": "actual_direct_qualification_popen_return",
        "observed_child": {**process, "affinity": list(range(8))}, "popen_pid": 2000,
        "controller": {"pid": 1000, "process_start_time": "999", "affinity": list(range(8)),
            "command": command[8:]}, "cpu_affinity_is_exclusive": False}
    records = {"direct_controller_spawn.json": spawn,
        "direct_current_launch.json": {"plan": {"path": str(review.UNIT / "measurement_plan.json"), "sha256": "test"},
            "process_row": {"pid": 2000, "start_time": "1234", "helper_commit": review.HELPER_COMMIT}},
        "direct_process_wait_return.json": {"waiter_before": parent, "waiter_after": copy.deepcopy(parent)},
        "actual_launch_command_preparation.json": {"command": command, "host": "serval02", "automatic_retry_permitted": False},
        "root_remote_launch_return.json": {"returncode": 0, "reported_controller_pid": 1000},
        "root_remote_launch_intent.json": {"published_commit": review.LAUNCH_COMMAND_PUBLICATION,
            "command_preparation_sha256": "test"},
        "direct_controller_outcome.json": {"schema_version": 1, "kind": "direct_qualification_controller_outcome",
            "passed": True, "actual_child_started": True, "actual_wait_saved": True, "failure": None,
            "qualification_independently_verified": False}}
    for key, name in (("stdout", "direct_controller_stdout.raw"), ("stderr", "direct_controller_stderr.raw")):
        records.setdefault("direct_controller_output_hashes.json", {
            "kind": "actual_direct_child_output_hashes_after_wait",
            "wait": {"path": str(review.UNIT / "direct_process_wait_return.json"), "sha256": "test"}})[key] = {
                "path": str(review.UNIT / name), "sha256": "test", "size_bytes": 0}
    direct = SimpleNamespace(core=lambda: SimpleNamespace(exact=lambda a, b: type(a) is type(b) and a == b),
        BOUNDARY={})
    return plan, result, {"after": {"identity": process}}, {"native": {"device_ordinal": 0}, "gpu_uuid": "GPU-test"}, records, direct


def run_fixture(data):
    """Run extra joins with only controlled synthetic reads."""
    plan, result, metadata, binding, records, direct = data
    with mock.patch.object(review, "doc", side_effect=lambda p: records[Path(p).name]), \
            mock.patch.object(review, "ref", side_effect=lambda p: {"path": str(p), "sha256": "test"}), \
            mock.patch.object(review, "digest", return_value="test"), \
            mock.patch.object(Path, "stat", return_value=SimpleNamespace(st_size=0)):
        return review.extra_actual_joins(direct, plan, result, metadata, binding)


class SourceContractTests(unittest.TestCase):
    """Check source-only success and changed pin or omitted gate refusal."""
    def test_source_only(self):
        """Current source passes without actual result reads."""
        actual_doc = review.doc
        reads = []
        def read(path):
            reads.append(Path(path).name)
            return actual_doc(path)
        with mock.patch.object(review, "doc", side_effect=read):
            review.verify_source_contract()
        self.assertNotIn("direct_qualification_result.json", reads)

    def test_changed_source_pin(self):
        """A changed staged direct consumer hash is refused."""
        with mock.patch.object(review, "SOURCE_SHA", "0" * 64):
            with self.assertRaises(ValueError):
                review.verify_source_contract()

    def test_changed_raw_pin(self):
        """A changed raw reader hash is refused."""
        with mock.patch.object(review, "RAW_SHA", "0" * 64):
            with self.assertRaises(ValueError):
                review.verify_source_contract()

    def test_changed_plan_pin(self):
        """A different prepared plan is refused."""
        with mock.patch.object(review, "PLAN_SHA", "0" * 64):
            with self.assertRaises(ValueError):
                review.verify_source_contract()

    def test_extra_omitted_statement(self):
        """Removing a third raw gate is refused before raw execution."""
        actual_doc = review.doc
        def read(path):
            value = actual_doc(path)
            if Path(path).name == "raw_review_source_contract.json":
                value["removed_statement_indices"] = [4, 46, 47]
            return value
        with mock.patch.object(review, "doc", side_effect=read):
            with self.assertRaises(ValueError):
                review.verify_source_contract()

    def test_terminal_gate_changed(self):
        """A changed final direct terminal call fails the AST contract."""
        actual_parse = ast.parse
        def parse(source, *args, **kwargs):
            tree = actual_parse(source, *args, **kwargs)
            for node in tree.body:
                if isinstance(node, ast.FunctionDef) and node.name == "verify_unreviewed_actual_raw_qualification":
                    node.body[-2] = ast.Pass()
            return tree
        with mock.patch.object(review.ast, "parse", side_effect=parse):
            with self.assertRaises(ValueError):
                review.verify_source_contract()


class ActualJoinTests(unittest.TestCase):
    """Check synthetic joins and current launch refusal predicates."""
    def test_complete_synthetic_chain(self):
        """The synthetic chain passes and labels fixture work unknown."""
        result = run_fixture(fixture())
        self.assertFalse(result["test_fixture_transition_total_known"])
        self.assertEqual(result["controlled_reference_updates"], 1)

    def test_invalid_synthetic_fields(self):
        """Changed identity, protocol or controller fields are refused."""
        mutations = [
            ("nonfinite time", lambda d: d[1]["execution_identity"].update(started_at_seconds=math.nan)),
            ("reversed time", lambda d: d[1]["execution_identity"].update(finished_at_seconds=0.5)),
            ("changed seed", lambda d: d[1].update(copy_seeds=[2])),
            ("changed protocol", lambda d: d[1]["protocol"].update(actual_updates_per_round=1)),
            ("changed configuration", lambda d: d[1].update(configuration={"environment": {"num_envs": 1}})),
            ("changed child", lambda d: d[4]["direct_controller_spawn.json"].update(popen_pid=2001)),
            ("changed helper", lambda d: d[4]["direct_current_launch.json"]["process_row"].update(helper_commit="0"*40)),
            ("changed parent birth", lambda d: d[4]["direct_controller_spawn.json"]["controller"].update(process_start_time="998")),
            ("changed CPU selector count", lambda d: d[4]["direct_controller_spawn.json"]["observed_child"].update(affinity=[0])),
            ("exclusive CPU claim", lambda d: d[4]["direct_controller_spawn.json"].update(cpu_affinity_is_exclusive=True)),
            ("changed controller argv", lambda d: d[4]["direct_controller_spawn.json"]["controller"].update(command=["other"])),
            ("automatic retry", lambda d: d[4]["actual_launch_command_preparation.json"].update(automatic_retry_permitted=True)),
            ("transport failure", lambda d: d[4]["root_remote_launch_return.json"].update(returncode=1)),
            ("wrong transport parent", lambda d: d[4]["root_remote_launch_return.json"].update(reported_controller_pid=1001)),
            ("wrong publication", lambda d: d[4]["root_remote_launch_intent.json"].update(published_commit="0"*40)),
            ("controller failed", lambda d: d[4]["direct_controller_outcome.json"].update(passed=False)),
        ]
        for name, mutate in mutations:
            with self.subTest(name=name):
                data = fixture()
                mutate(data)
                with self.assertRaises(ValueError):
                    run_fixture(data)


if __name__ == "__main__":
    unittest.main()
