"""Exercise metadata prerequisite checks without numerical or device operations."""
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS))
import common_confirmation as common
import measure_selected as measurement


class CommonConfirmationTests(unittest.TestCase):
    """Keep current common checks separate from historical and final protocols."""

    def setUp(self):
        """Make one minimal explicit intermediate configuration and immutable inputs."""
        self.temporary = tempfile.TemporaryDirectory(); self.addCleanup(self.temporary.cleanup)
        self.unit = Path(self.temporary.name)
        self.plan = {"kind": "selected_configuration_measurement", "mode": "confirmation",
            "hardware_class": "jaguar03_rtx_a4500", "gpu_uuid": None, "configured_isolation": {"profile": {}},
            "rounds": 7, "minimum_updates_per_round": 10, "iterations": 10, "copy_seeds": [1],
            "configuration": {"learner": {"n_copies": 1}, "environment": {"cache_room_backgrounds": True,
                "render_final_two_only": True}, "execution": {}},
            "reference_configuration": {"learner": {"n_copies": 1}, "environment": {"cache_room_backgrounds": False,
                "render_final_two_only": False}, "execution": {}},
            "helpers": {"source_commit": "a" * 40, "files": {common.PREFIX + common.NAME:
                hashlib.sha256(Path(common.__file__).read_bytes()).hexdigest()}},
            "common_correctness_inputs": {"result": {"path": str(self.unit / "prior.json"), "sha256": "b" * 64},
                "review": {"path": str(self.unit / "review.json"), "sha256": "c" * 64}}}
        (self.unit / "measurement_plan.json").write_bytes(common.encoded(self.plan))

    def proof(self, raw=None):
        """Return a synthetic successful child record with this actual test parent."""
        raw = common.encoded(self.plan) if raw is None else raw
        return {"schema_version": 1, "kind": "isolated_common_correctness_validation", "passed": True,
            "plan_sha256": common.digest(raw), "common_correctness_inputs": self.plan["common_correctness_inputs"],
            "parent_pid": os.getpid(), "validation_pid": 12, "finished_at_seconds": 1., "numerical_imports": [], "gpu_operations": 0}

    def completed(self, proof=None):
        """Stand in only for the separately tested metadata child process."""
        return subprocess.CompletedProcess([], 0, json.dumps(proof or self.proof()).encode(), b"")

    def test_intermediate_contract_accepts_both_explicit_copy_counts(self):
        """The two accepted copy configurations retain their exact seed lists."""
        common.contract(self.plan)
        self.plan["configuration"]["learner"]["n_copies"] = 2
        self.plan["reference_configuration"]["learner"]["n_copies"] = 2
        self.plan["copy_seeds"] = [1, 2]
        common.contract(self.plan)

    def test_final_direct_capacity_and_wrong_mode_are_rejected(self):
        """No opt-in field can authorize a different or later protocol."""
        for key in ("direct_binding", "target_qualification", "final_confirmation_protocol", "completed_study",
                    "device_memory_limit_protocol", "fixed_total_scaling", "scaling_protocol"):
            with self.subTest(key=key), self.assertRaises(ValueError):common.contract({**self.plan, key: {}})
        for mode in ("correctness", "packing-point", "learning-check"):
            with self.subTest(mode=mode), self.assertRaises(ValueError):common.contract({**self.plan, "mode": mode})

    def test_window_types_and_incomplete_prerequisites_are_rejected(self):
        """Boolean counters, a reduced window and one missing record cannot pass."""
        for key, wrong in (("rounds", True), ("rounds", 11), ("iterations", 1), ("minimum_updates_per_round", 20),
                ("common_correctness_inputs", {"result": {}})):
            with self.subTest(key=key), self.assertRaises(ValueError):common.contract({**self.plan, key: wrong})

    def test_equal_or_extra_candidate_change_and_wrong_seeds_are_rejected(self):
        """Only the independently checked two environment changes are permitted."""
        equal = {**self.plan, "configuration": self.plan["reference_configuration"]}
        with self.assertRaises(ValueError):common.contract(equal)
        changed = copy.deepcopy(self.plan); changed["configuration"]["learner"]["learning_rate"] = 1
        changed["reference_configuration"]["learner"]["learning_rate"] = 2
        with self.assertRaises(ValueError):common.contract(changed)
        with self.assertRaises(ValueError):common.contract({**self.plan, "copy_seeds": [True]})

    def test_staged_helper_bytes_and_reference_hashes_are_required(self):
        """A different helper or changed saved bytes do not substitute for the frozen records."""
        changed = copy.deepcopy(self.plan); changed["helpers"]["files"][common.PREFIX + common.NAME] = "0" * 64
        with self.assertRaises(ValueError):common.contract(changed)
        p = self.unit / "reference.json"; p.write_text('{"value":1}')
        record = common.reference(p); self.assertEqual(common.checked_reference(record), {"value": 1})
        p.write_text('{"value":2}')
        with self.assertRaises(ValueError):common.checked_reference(record)

    def test_child_request_leaves_parent_paths_and_public_modules_unchanged(self):
        """The parent dispatches only the declared no-site interpreter and changes no import paths."""
        before = list(sys.path); ideas = sys.modules.get("ideas")
        with patch.object(common.subprocess, "run", return_value=self.completed()) as run:
            common.request(self.plan)
        command = run.call_args.args[0]
        self.assertEqual(command, ["/usr/bin/python3", "-I", "-S", "-B", str(Path(common.__file__).absolute()), "--plan-stdin"])
        self.assertEqual(before, sys.path); self.assertIs(sys.modules.get("ideas"), ideas)

    def test_different_parent_raw_plan_and_boolean_gpu_claim_are_rejected(self):
        """A successful child response must describe this exact parent and plan."""
        for key, value in (("parent_pid", True), ("parent_pid", os.getpid() + 1), ("plan_sha256", "0" * 64), ("gpu_operations", False)):
            with self.subTest(key=key), patch.object(common.subprocess, "run", return_value=self.completed({**self.proof(), key: value})):
                with self.assertRaises(ValueError):common.request(self.plan)

    def test_success_captures_raw_child_before_publishing_proof_and_cannot_repeat(self):
        """Keep one durable actual validation attempt and its exact raw successful return."""
        with patch.object(common.subprocess, "run", return_value=self.completed()) as run:
            common.before_compute(self.unit, self.plan)
            binding = common.result_binding(self.unit, self.plan)
            self.assertEqual(binding["common_correctness_inputs"], self.plan["common_correctness_inputs"])
            self.assertTrue((self.unit / "common_correctness_validation_stdout.raw").exists())
            with self.assertRaises(ValueError):common.before_compute(self.unit, self.plan)
            self.assertEqual(run.call_count, 1)

    def test_failed_and_malformed_child_return_preserves_raw_without_success_proof(self):
        """Retain both nonzero and invalid successful output before refusing the worker."""
        for value in (subprocess.CompletedProcess([], 1, b"partial", b"failed"), subprocess.CompletedProcess([], 0, b"invalid JSON", b"")):
            with tempfile.TemporaryDirectory() as tmp:
                unit = Path(tmp); (unit / "measurement_plan.json").write_bytes(common.encoded(self.plan))
                with patch.object(common.subprocess, "run", return_value=value):
                    with self.assertRaises((ValueError, json.JSONDecodeError)):common.before_compute(unit, self.plan)
                self.assertEqual((unit / "common_correctness_validation_stdout.raw").read_bytes(), value.stdout)
                self.assertFalse((unit / common.PROOF_NAME).exists())
                with self.assertRaises(ValueError):common.before_compute(unit, self.plan)

    def test_timeout_preserves_partial_output_and_attempt(self):
        """A timed-out child never becomes a successful prerequisite or starts a repeat."""
        with patch.object(common.subprocess, "run", side_effect=subprocess.TimeoutExpired('synthetic', 1, output=b'partial', stderr=b'error')):
            with self.assertRaises(subprocess.TimeoutExpired):common.before_compute(self.unit, self.plan)
        self.assertEqual((self.unit / 'common_correctness_validation_stdout.raw').read_bytes(), b'partial')
        self.assertIsNone(common.read(self.unit / 'common_correctness_validation_return.json')[0]['returncode'])
        self.assertFalse((self.unit / common.PROOF_NAME).exists())

    def test_saved_proof_requires_same_actual_parent_and_precedes_numeric_work(self):
        """The saved result cannot claim another worker's or later validation."""
        with patch.object(common.subprocess, 'run', return_value=self.completed()):common.before_compute(self.unit, self.plan)
        result = {**common.result_binding(self.unit, self.plan), 'execution_identity': {'pid': os.getpid(), 'started_at_seconds': 2.}}
        with patch.object(common, 'request') as request:common.verify_saved(self.unit, self.plan, result)
        request.assert_called_once()
        for changed in ({'pid': os.getpid() + 1, 'started_at_seconds': 2.}, {'pid': os.getpid(), 'started_at_seconds': .5}):
            with self.subTest(changed=changed), self.assertRaises(ValueError):common.verify_saved(self.unit, self.plan, {**result, 'execution_identity': changed})

    def test_default_direct_and_new_helper_closures_stay_separate(self):
        """Add only the new configured dependency while retaining all existing helper declarations."""
        baseline = measurement.required_helpers(True)
        self.assertEqual(measurement.required_helpers(True, False, True), (*baseline, common.NAME))
        self.assertNotIn(common.NAME, measurement.required_helpers())
        self.assertNotIn(common.NAME, measurement.required_helpers(False, True))
        with self.assertRaises(ValueError):measurement.required_helpers(False, True, True)
        self.assertEqual(len(measurement.required_helpers(False, True, True, True)), 27)

    def test_cli_requires_both_actual_records_and_refuses_final_or_prepared_overrides(self):
        """Preparation options cannot silently change the frozen protocol."""
        for extra in (["--common-correctness-record", "/missing"],
                ["--prepared-unit", "/missing", "--common-correctness-record", "/missing", "--common-correctness-review", "/missing"],
                ["--mode", "confirmation", "--final-confirmation", "--configured-isolation-profile", "/missing", "--common-correctness-record", "/missing", "--common-correctness-review", "/missing"]):
            with self.subTest(extra=extra), self.assertRaises(SystemExit):measurement.parse_arguments(extra)


class CommonSavedReaderTests(unittest.TestCase):
    """Exercise semantic joins with the unrelated full saved consumer explicitly replaced."""

    def setUp(self):
        """Build synthetic raw, review and terminal files for one reader call."""
        CommonConfirmationTests.setUp(self)
        self.previous = copy.deepcopy(self.plan)
        self.previous.pop('common_correctness_inputs');self.previous['mode'] = 'correctness'
        for key,value in {'source_root':'/synthetic/source','source_commit':'d'*40,'runtime_environment':{'JAX_ENABLE_X64':'true'},
                'asset_record':{'synthetic':True},'target_qualification_inputs':{'synthetic':True}}.items():
            self.previous[key] = value;self.plan[key] = copy.deepcopy(value)
        self.prior = self.unit/'prior';self.prior.mkdir()
        self.binding = {'gpu_uuid':'GPU-00000000-0000-0000-0000-000000000001'}
        self.metadata = {'inputs':{'job_id':'900001'},'environment':{'SLURM_STEP_ID':'0'}}
        self.result = {'measurement_mode':'correctness','measurement_valid':False,'implementation_distinct':True,
            'execution_identity':{'hostname':'jaguar03','pid':12,'process_start_time':'1','started_at_seconds':1.,'finished_at_seconds':2.}}
        self.write(self.prior/'measurement_plan.json',self.previous)
        self.write(self.prior/'configured_isolation_metadata.json',self.metadata)
        self.write(self.prior/'resolved_device_binding.json',self.binding)
        self.write(self.prior/'result.json',self.result)
        self.rows = {key:[key,'synthetic','sl5nw' if key=='900001' else '', 'COMPLETED','0:0','jaguar03',
            '2026-11-02T01:00:00','2026-11-02T01:01:00','8','64G','',''] for key in ('900001','900001.batch','900001.0')}
        accounting = self.unit/'terminal.raw';accounting.write_text(''.join('|'.join(row)+'\n' for row in self.rows.values()))
        self.raw = {'kind':'independent_actual_common_a4500_correctness_raw_validation','passed':True,
            'result':common.reference(self.prior/'result.json'),'plan':common.reference(self.prior/'measurement_plan.json'),
            'job_id':'900001','numeric_step':'900001.0','terminal_state':'COMPLETED','exit_code':'0:0',
            'unchanged_staged_consumer_passed':True,'execution_identity':self.result['execution_identity'],
            'copy_seeds':[1],'gpu_uuid':self.binding['gpu_uuid'],'terminal_accounting_rows':self.rows,
            'terminal_accounting':{'path':str(accounting),'sha256':common.digest(accounting.read_bytes())}}
        self.review = {'schema_version':1,'kind':'independent_actual_common_configuration_correctness_review','approved':True,'passed':True,
            'result':self.raw['result'],'plan':self.raw['plan'],'job_id':'900001','actual_numeric_step':'900001.0',
            'actual_terminal_state':'COMPLETED','actual_exit_code':'0:0','admission_granted':False,'worker_admitted':False,
            'scientific_credit':0,'idea_credit':0,'case_credit':0,'throughput_credit':0,'final_training_credit':0,
            'source_commit':self.previous['source_commit'],'helper_commit':self.previous['helpers']['source_commit'],
            'copies':1,'copy_seeds':[1],'configuration':self.previous['configuration'],
            'reference_configuration':self.previous['reference_configuration'],'gpu_uuid':self.binding['gpu_uuid'],
            'resolved_binding':common.reference(self.prior/'resolved_device_binding.json')}
        exact=lambda first,second:json.dumps(first,sort_keys=True)==json.dumps(second,sort_keys=True)
        self.consumer = unittest.mock.Mock(return_value=(self.previous,self.binding))
        self.support = SimpleNamespace(core=lambda:SimpleNamespace(exact=exact),verify_saved_measurement=self.consumer)
        self.validation = SimpleNamespace(correctness=unittest.mock.Mock(),execution_identity=unittest.mock.Mock(),configuration=unittest.mock.Mock())
        self.refresh()

    def write(self,path,value):
        """Publish synthetic fixture bytes before the tested validation call."""
        path.write_bytes(common.encoded(value))

    def refresh(self):
        """Rehash a changed synthetic raw file and its independent review coherently."""
        self.write(self.unit/'raw.json',self.raw)
        self.review['raw_audit'] = common.reference(self.unit/'raw.json')
        self.write(self.unit/'review.json',self.review)
        self.plan['common_correctness_inputs'] = {'result':common.reference(self.prior/'result.json'),
            'review':common.reference(self.unit/'review.json')}

    def validate(self):
        """Call the production semantic reader with only its unrelated consumer replaced."""
        with patch.dict(sys.modules,{'configured_measurement':self.support,'post_comparison_validation':self.validation}):
            return common.validate_actual(self.plan,'f'*64)

    def test_actual_semantic_reader_positive_joins_all_bound_inputs(self):
        """A complete matching synthetic review and raw accounting pass."""
        result = self.validate();self.assertTrue(result['passed']);self.consumer.assert_called_once()

    def test_coherently_rehashed_review_labels_and_credit_types_are_rejected(self):
        """Changing approved labels cannot disagree with the saved correctness plan."""
        original = copy.deepcopy(self.review)
        for key,value in [('source_commit','0'*40),('copies',True),('copy_seeds',[2]),('gpu_uuid','other'),
                ('helper_commit','0'*40),('actual_numeric_step','900001.1'),('scientific_credit',False),('approved',1)]:
            self.review={**copy.deepcopy(original),key:value};self.refresh()
            with self.subTest(key=key),self.assertRaises(ValueError):self.validate()

    def test_raw_terminal_bytes_and_derived_rows_must_agree(self):
        """A successful review cannot replace failed or modified actual terminal evidence."""
        path=Path(self.raw['terminal_accounting']['path']);path.write_text(path.read_text().replace('COMPLETED','FAILED',1))
        self.raw['terminal_accounting']['sha256']=common.digest(path.read_bytes());self.refresh()
        with self.assertRaises(ValueError):self.validate()
        self.raw['terminal_accounting_rows']['900001'][3]='FAILED';self.refresh()
        with self.assertRaises(ValueError):self.validate()

    def test_outer_runtime_or_seed_change_is_not_covered_by_old_correctness(self):
        """A later plan must preserve the full independently checked configuration."""
        self.plan['runtime_environment']={'JAX_ENABLE_X64':'false'}
        with self.assertRaises(ValueError):self.validate()


if __name__ == '__main__':
    unittest.main()
