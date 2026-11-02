"""Exercise only synthetic metadata; every scheduler call is replaced before use."""

import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

PATH = Path(__file__).with_name("release_common_confirmation.py")
SPECIFICATION = importlib.util.spec_from_file_location("qualification_release_test_target", PATH)
release = importlib.util.module_from_spec(SPECIFICATION)
SPECIFICATION.loader.exec_module(release)


class QualificationReleaseTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(); self.addCleanup(self.temporary.cleanup)
        self.run = Path(self.temporary.name); self.packet = self.run / "remote_setup/synthetic_qualification"
        self.unit = self.run / "data/synthetic_qualification"
        self.packet.mkdir(parents=True); self.unit.mkdir(parents=True)
        for name, value in (("PACKET", self.packet), ("UNIT", self.unit), ("RUN", self.run), ("JOB", "6783295"), ("COMMAND", ["scontrol", "release", "6783295"]), ("REQUESTED_HOURS", 2)):
            control = patch.object(release, name, value); control.start(); self.addCleanup(control.stop)
        self.plan = {"kind": "selected_configuration_measurement", "synthetic": True}
        self.write(self.unit / "measurement_plan.json", self.plan)
        self.pre = {"schema_version": 1, "kind": "uva_configured_isolation_pre_release_inputs", "job_id": release.JOB,
            "plan": release.ref(self.unit / "measurement_plan.json"), "ledger": {}, "submissions": {},
            "submission": {}, "submission_row": [release.JOB, self.plan["kind"], release.SOURCE, str(self.packet / "job.sbatch")]}
        self.write(self.unit / "configured_isolation_pre_release_inputs.json", self.pre)
        control = patch.object(release, "PRE_RELEASE_SHA", release.sha(release.contents(self.unit / "configured_isolation_pre_release_inputs.json")))
        control.start(); self.addCleanup(control.stop)
        self.intent = {"publication_commit": "a" * 40, "job_id": release.JOB, "command": release.COMMAND, "source_sha256": "b" * 64,
            "held_monitor_review": {"path": "synthetic-only", "sha256": "c" * 64}}
        self.calls = []

    def write(self, path, value):
        path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(release.encoded(value))

    def held(self):
        fields = {"JobId": release.JOB, "UserId": "sl5nw(2618919)", "Account": "sl5nw_base", "Partition": "gpu",
            "QOS": "csresnolim", "JobState": "PENDING", "Reason": "JobHeldUser", "ReqNodeList": "jaguar03",
            "Reservation": "sl5nw_168", "Requeue": "0", "Restarts": "0", "NumCPUs": "8", "NumTasks": "1",
            "CPUs/Task": "8", "MinMemoryNode": "64G", "TimeLimit": str(release.REQUESTED_HOURS).zfill(2)+":00:00", "NumNodes": "1",
            "Command": str(self.packet / "job.sbatch"), "TresPerNode": "gres/gpu:nvidia_rtx_a4500:1",
            "ReqTRES": "cpu=8,mem=64G,node=1,billing=8,gres/gpu=1", "AllocTRES": "(null)", "NodeList": ""}
        return " ".join(key + "=" + value for key, value in fields.items())

    def fake_run(self, command, **arguments):
        self.assertTrue((self.packet / "release_intent.json").is_file())
        self.calls.append(command)
        if command == ["scontrol", "show", "job", "-o", release.JOB]:
            return SimpleNamespace(returncode=0, stdout=self.held().encode(), stderr=b"")
        self.assertEqual(command, ["scontrol", "release", release.JOB])
        self.assertTrue((self.packet / "release_preflight_return.json").is_file())
        self.assertFalse((self.packet / "release_receipt.json").exists())
        self.assertFalse((self.unit / "owned_allocation_inputs.json").exists())
        return SimpleNamespace(returncode=0, stdout=b"", stderr=b"")

    def fake_core(self, plan):
        self.assertEqual(plan, self.plan)
        def owned_inputs(observed):
            self.assertTrue((self.packet / "release_receipt.json").is_file())
            self.assertEqual(observed, self.plan)
            return release.document(self.unit / "owned_allocation_inputs.json")
        return SimpleNamespace(owned_inputs=owned_inputs)

    def test_exact_single_card_held_shape_and_changed_fields(self):
        raw = self.held(); release.validate_held(raw)
        for old, new in (("JobId=" + release.JOB, "JobId=999999"), ("Requeue=0", "Requeue=1"),
                ("Restarts=0", "Restarts=1"), ("NumCPUs=8", "NumCPUs=16"), ("Reservation=sl5nw_168", "Reservation=other"),
                ("JobHeldUser", "Priority"), ("gpu:nvidia_rtx_a4500:1", "gpu:nvidia_rtx_a4500:2"),
                ("cpu=8,", "cpu=8,cpu=8,"), ("gres/gpu=1", "gres/gpu=1,gres/gpu:other=1")):
            with self.subTest(old=old), self.assertRaises(ValueError): release.validate_held(raw.replace(old, new))
        release.validate_held(raw.replace("gres/gpu=1", "gres/gpu=1,gres/gpu:nvidia_rtx_a4500=1"))
        with self.assertRaises(ValueError):
            release.validate_held(raw.replace("gres/gpu=1", "gres/gpu=1,gres/gpu:nvidia_rtx_a4500=2"))

    def test_one_intent_raw_before_parse_real_receipt_before_atomic_inputs(self):
        original = release.validate_held
        def parsed(raw):
            self.assertEqual((self.packet / "release_preflight_stdout.raw").read_text(), raw)
            self.assertTrue((self.packet / "release_preflight_return.json").is_file())
            original(raw)
        with patch.object(release.subprocess, "run", side_effect=self.fake_run), patch.object(release, "core", side_effect=self.fake_core),\
                patch.object(release, "validate_held", side_effect=parsed):
            result = release.release_once(self.intent)
            self.assertTrue(result["actual_core_reader_passed"])
            self.assertEqual(len(self.calls), 2)
            final = release.document(self.unit / "owned_allocation_inputs.json")
            self.assertEqual({key: value for key, value in final.items() if key not in {"kind", "release"}},
                {key: value for key, value in self.pre.items() if key != "kind"})
            self.assertEqual(final["release"], release.ref(self.packet / "release_receipt.json"))
            self.assertEqual(release.document(final["release"]["path"])["kind"], "owned_uva_configured_isolation_release")
            with self.assertRaises(ValueError): release.release_once(self.intent)
            self.assertEqual(len(self.calls), 2)

    def test_bad_preflight_keeps_raw_and_never_releases(self):
        def bad(command, **arguments):
            self.calls.append(command)
            return SimpleNamespace(returncode=0, stdout=self.held().replace("JobHeldUser", "Priority").encode(), stderr=b"")
        with patch.object(release.subprocess, "run", side_effect=bad):
            with self.assertRaises(ValueError): release.release_once(self.intent)
        self.assertEqual(len(self.calls), 1)
        self.assertTrue((self.packet / "release_preflight_stdout.raw").exists())
        self.assertTrue((self.packet / "release_intent.json").exists())
        self.assertFalse((self.packet / "release_receipt.json").exists())
        self.assertFalse((self.unit / "owned_allocation_inputs.json").exists())

    def test_nonzero_release_retains_return_without_receipt_or_final_inputs(self):
        def failed(command, **arguments):
            if command[1] == "show": return self.fake_run(command, **arguments)
            self.calls.append(command)
            return SimpleNamespace(returncode=1, stdout=b"", stderr=b"synthetic scheduler failure")
        with patch.object(release.subprocess, "run", side_effect=failed):
            with self.assertRaises(ValueError): release.release_once(self.intent)
        self.assertEqual(release.document(self.packet / "release_return.json")["returncode"], 1)
        self.assertFalse((self.packet / "release_receipt.json").exists())
        self.assertFalse((self.unit / "owned_allocation_inputs.json").exists())

    def test_missing_publication_never_creates_intent_or_calls_a_command(self):
        with patch.object(release.subprocess, "run") as query:
            with self.assertRaises(ValueError): release.release_once({**self.intent, "publication_commit": None})
            query.assert_not_called()
        self.assertFalse((self.packet / "release_intent.json").exists())

    def test_timeout_keeps_partial_raw_and_prevents_a_second_attempt(self):
        with patch.object(release.subprocess, "run", side_effect=release.subprocess.TimeoutExpired(["synthetic"], 30, output=b"partial", stderr=b"error")):
            with self.assertRaises(release.subprocess.TimeoutExpired): release.release_once(self.intent)
        self.assertEqual((self.packet / "release_preflight_stdout.raw").read_bytes(), b"partial")
        self.assertIsNone(release.document(self.packet / "release_preflight_return.json")["returncode"])
        with self.assertRaises(ValueError): release.release_once(self.intent)

    def test_atomic_publication_never_overwrites_existing_bytes(self):
        target = self.unit / "owned_allocation_inputs.json"; target.write_bytes(b"existing")
        with self.assertRaises(ValueError): release.put(target, b"changed", atomic=True)
        self.assertEqual(target.read_bytes(), b"existing")

    def test_changed_pre_release_envelope_after_actual_release_refuses_final_inputs(self):
        def changed(command, **arguments):
            result = self.fake_run(command, **arguments)
            if command[1] == "release": self.write(self.unit / "configured_isolation_pre_release_inputs.json", {**self.pre, "job_id": "999999"})
            return result
        with patch.object(release.subprocess, "run", side_effect=changed):
            with self.assertRaises(ValueError): release.release_once(self.intent)
        self.assertTrue((self.packet / "release_receipt.json").exists())
        self.assertFalse((self.unit / "owned_allocation_inputs.json").exists())


    def test_default_main_does_not_release_or_call_scheduler(self):
        with patch.object(release.sys, "argv", [str(PATH), "--copies", "1"]), patch.object(release, "checked", return_value=self.intent) as checked, \
                patch.object(release, "release_once") as action, patch.object(release.subprocess, "run") as scheduler:
            release.main()
            checked.assert_called_once_with(1, None)
            action.assert_not_called()
            scheduler.assert_not_called()
        self.assertFalse((self.packet / "release_intent.json").exists())

    def test_explicit_release_without_publication_refuses_before_checks(self):
        with patch.object(release.sys, "argv", [str(PATH), "--copies", "1", "--release"]), patch.object(release, "checked") as checked, \
                patch.object(release.subprocess, "run") as scheduler:
            with self.assertRaises(ValueError): release.main()
            checked.assert_not_called()
            scheduler.assert_not_called()
        self.assertFalse((self.packet / "release_intent.json").exists())

    def test_atomic_pending_file_and_symlink_conflicts_are_preserved(self):
        target = self.unit / "owned_allocation_inputs.json"
        pending = target.with_name("." + target.name + ".pending")
        pending.write_bytes(b"previous partial publication")
        with self.assertRaises(FileExistsError): release.put(target, b"changed", atomic=True)
        self.assertEqual(pending.read_bytes(), b"previous partial publication")
        self.assertFalse(target.exists())
        pending.unlink()
        outside = self.run / "outside.json"; outside.write_bytes(b"other bytes")
        target.symlink_to(outside)
        with self.assertRaises(ValueError): release.put(target, b"changed", atomic=True)
        self.assertEqual(outside.read_bytes(), b"other bytes")
        self.assertTrue(target.is_symlink())

    def test_dangling_pending_link_refuses_preflight_before_any_scheduler_call(self):
        pending=self.unit/'.owned_allocation_inputs.json.pending'
        pending.symlink_to(self.run/'absent_previous_target')
        with patch.object(release.subprocess,'run') as scheduler:
            with self.assertRaises(ValueError):release.require_unattempted_release()
            with self.assertRaises(ValueError):release.release_once(self.intent)
            scheduler.assert_not_called()
        self.assertTrue(pending.is_symlink())
        self.assertFalse((self.packet/'release_intent.json').exists())

    def test_stale_foreign_and_boolean_full_monitor_proofs_refuse(self):
        from datetime import datetime, timedelta, timezone
        proof = {"schema_version": 1, "kind": "owned_uva_configured_isolation_actual_full_held_monitor_proof",
            "passed": True, "job_id": release.JOB, "state": "PENDING", "reason": "JobHeldUser",
            "publication_commit": release.INITIAL_PUBLICATION, "helper_source_commit": release.SOURCE,
            "at_utc": (datetime.now(timezone.utc) - timedelta(seconds=1201)).isoformat()}
        for changed in (proof, {**proof, "job_id": "6782640"}, {**proof, "schema_version": True},
                {**proof, "at_utc": (datetime.now(timezone.utc) + timedelta(seconds=30)).isoformat()}):
            with self.subTest(proof=changed), self.assertRaises(ValueError): release.validate_proof(changed)

    def test_query_and_full_monitor_observations_each_require_recent_utc(self):
        from datetime import datetime,timedelta,timezone
        now=datetime.now(timezone.utc)
        self.assertIsNotNone(release.utc_observation(now.isoformat(),'synthetic observation'))
        for value in ((now-timedelta(seconds=1201)).isoformat(),(now+timedelta(seconds=10)).isoformat(),
                now.replace(tzinfo=None).isoformat(),now.astimezone(timezone(timedelta(hours=1))).isoformat()):
            with self.subTest(value=value),self.assertRaises(ValueError):release.utc_observation(value,'synthetic observation')

    def test_foreign_selector_and_intent_refuse_before_scheduler(self):
        for copies in (0, 3, True, "1"):
            with self.subTest(copies=copies), self.assertRaises(ValueError): release.select(copies)
        for intent in ({**self.intent, "job_id": "6783296"},
                {**self.intent, "command": ["scontrol", "release", "6783296"]}):
            with patch.object(release.subprocess, "run") as query:
                with self.assertRaises(ValueError): release.release_once(intent)
                query.assert_not_called()
        self.assertFalse((self.packet / "release_intent.json").exists())

    def test_other_target_progress_and_launch_state_do_not_block_selected_target(self):
        other = self.run / "data/other_target"; other.mkdir(parents=True)
        own = self.unit / "progress.json"; self.write(own, {"state": "prepared_not_submitted"})
        foreign = other / "progress.json"; self.write(foreign, {"state": "prepared_not_submitted"})
        shared = self.packet / "shared_source.py"; shared.write_text("# exact shared source\n")
        row = lambda path: {"sha256": release.ref(path)["sha256"], "bytes": len(release.contents(path))}
        frozen = {"shared_files": {str(shared): row(shared)}}
        selected = {"files": {str(own): row(own)}}
        with patch.object(release, "SELECTED", selected):
            release.validate_frozen_files(frozen)
            self.write(foreign, {"state": "running", "actual_job_id": "synthetic_other"})
            self.write(other / "owned_allocation_inputs.json", {"actual_release": True})
            release.validate_frozen_files(frozen)
            self.write(own, {"state": "running"})
            with self.assertRaises(ValueError): release.validate_frozen_files(frozen)

    def test_shared_source_change_refuses_even_when_other_target_is_running(self):
        shared = self.packet / "shared_source.py"; shared.write_text("# initial shared bytes\n")
        frozen = {"shared_files": {str(shared): {"sha256": release.ref(shared)["sha256"], "bytes": len(release.contents(shared))}}}
        with patch.object(release, "SELECTED", {"files": {}}):
            release.validate_frozen_files(frozen); shared.write_text("# changed shared bytes\n")
            with self.assertRaises(ValueError): release.validate_frozen_files(frozen)

    def test_both_request_durations_and_unallocated_resources_are_exact(self):
        for hours in (2, 3):
            with patch.object(release, "REQUESTED_HOURS", hours):
                raw = self.held(); release.validate_held(raw)
                for old,new in (("TimeLimit="+str(hours).zfill(2)+":00:00", "TimeLimit=04:00:00"),
                        ("AllocTRES=(null)", "AllocTRES=cpu=8,mem=64G,node=1,gres/gpu=1"),
                        (" NodeList=", " NodeList=jaguar03")):
                    with self.subTest(hours=hours, field=old), self.assertRaises(ValueError):
                        release.validate_held(raw.replace(old,new))

    def test_both_actual_staged_readers_accept_only_synthetic_successful_release_chains(self):
        for copies in (1, 2):
            with self.subTest(copies=copies):
                original=Path(release.SCOPES[str(copies)]['unit'])
                plan=json.loads((original/'measurement_plan.json').read_bytes())
                core_path=original/'code/uva_slurm_configured_isolation.py'
                relative='10_jax_exploration_platform/scripts/atari_pixel_campaign/'+core_path.name
                self.assertEqual(release.ref(core_path)['sha256'],plan['helpers']['files'][relative])
                spec=importlib.util.spec_from_file_location('actual_staged_owned_reader_'+str(copies),core_path)
                actual=importlib.util.module_from_spec(spec);spec.loader.exec_module(actual)
                campaign=self.run/('synthetic_actual_core_'+str(copies));unit=campaign/'data/synthetic_confirmation'
                unit.mkdir(parents=True);(campaign/'slurm').mkdir()
                plan['campaign_run_dir']=str(campaign);plan['unit_id']=unit.name
                plan['configured_isolation']['owned_inputs_path']=str(unit/'owned_allocation_inputs.json')
                self.write(unit/'measurement_plan.json',plan);job='123456789'+str(copies)
                row=[job,plan['kind'],release.SOURCE,str(campaign/'remote_setup'/unit.name/'job.sbatch')]
                for name,text in (('submitted_jobids_uva.txt',job+'\n'),('submissions.tsv','\t'.join(row)+'\n')):
                    (campaign/'slurm'/name).write_text(text)
                ids=unit/'configured_isolation_owned_job_ids.txt';ids.write_text(job+'\n')
                rows=unit/'configured_isolation_owned_submissions.tsv';rows.write_text('\t'.join(row)+'\n')
                common={'schema_version':1,'job_id':job,'cluster':'uva','returncode':0,
                    'plan_sha256':release.ref(unit/'measurement_plan.json')['sha256'],'data_unit':str(unit)}
                submission=unit/'synthetic_submission.json';self.write(submission,{**common,
                    'kind':'owned_uva_configured_isolation_submission','held_command':['sbatch','--hold','--no-requeue',row[-1]],'submission_row':row})
                returned=unit/'synthetic_release.json';self.write(returned,{**common,
                    'kind':'owned_uva_configured_isolation_release','command':['scontrol','release',job],
                    'submission_receipt_sha256':release.ref(submission)['sha256']})
                final={'schema_version':1,'kind':'uva_configured_isolation_owned_inputs','job_id':job,
                    'plan':release.ref(unit/'measurement_plan.json'),'ledger':{'snapshot':release.ref(ids),
                    'source_path':str(campaign/'slurm/submitted_jobids_uva.txt')},
                    'submissions':{'snapshot':release.ref(rows),'source_path':str(campaign/'slurm/submissions.tsv')},
                    'submission':release.ref(submission),'release':release.ref(returned),'submission_row':row}
                release.put(unit/'owned_allocation_inputs.json',release.encoded(final),atomic=True)
                self.assertEqual(actual.owned_inputs(plan),final)
                failed=json.loads(returned.read_bytes());failed['returncode']=1;self.write(returned,failed)
                final['release']=release.ref(returned);self.write(unit/'owned_allocation_inputs.json',final)
                with self.assertRaises(ValueError):actual.owned_inputs(plan)

    def test_second_literal_job_has_separate_intent_and_final_receipt(self):
        job = "6783296"
        with patch.object(release, "JOB", job), patch.object(release, "COMMAND", ["scontrol", "release", job]):
            self.pre["job_id"] = job
            self.pre["submission_row"][0] = job
            self.write(self.unit / "configured_isolation_pre_release_inputs.json", self.pre)
            intent = {**self.intent, "job_id": job, "command": ["scontrol", "release", job]}
            with patch.object(release, "PRE_RELEASE_SHA", release.ref(self.unit / "configured_isolation_pre_release_inputs.json")["sha256"]), \
                    patch.object(release.subprocess, "run", side_effect=self.fake_run), \
                    patch.object(release, "core", side_effect=self.fake_core):
                result = release.release_once(intent)
            self.assertEqual(result["job_id"], job)
            self.assertEqual(release.document(self.packet / "release_receipt.json")["command"], ["scontrol", "release", job])



if __name__ == "__main__":
    unittest.main()
