"""Check that controllers recover checkpoint transactions only for their stopped current attempt."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import launch_deployed_training as deployed
import launch_final_training as final
from tests import test_launch_deployed_training as aces_fixtures
from tests import test_launch_final_training as uva_fixtures


class CheckpointContinuationTests(unittest.TestCase):
    """Keep terminal ownership, failure checks and raw evidence ahead of any checkpoint changes."""

    def uva(self):
        """Reuse complete immutable UVA admission fixtures without calling a real scheduler."""
        fixture = uva_fixtures.FinalLauncherTests("runTest")
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        unit = fixture.prepare(1)[0]
        fixture.submit(gpus=1)
        fixture.checkpoint(unit)
        directory = fixture.run / "data" / unit["unit_id"]
        progress = {"state": "running", "attempt": 1, "pid": 555}
        fixture.write(directory / "progress.json", progress)
        fixture.write(directory / "attempts/attempt-0001/progress.json", progress)
        fixture.write(directory / "checkpoints/latest.transaction.json", {"synthetic": "pending transaction"})
        path = fixture.queue / "status" / (unit["unit_id"] + ".json")
        final.atomic(path, {"state": "running", "unit_id": unit["unit_id"], "last_job_id": "9001", "last_job_cluster": "uva"})
        return fixture, unit, directory, path

    def test_uva_recovers_before_checkpoint_reads_and_keeps_validation_required(self):
        """A terminal exact current job recovers first, then still needs the ordinary strict checkpoint checks."""
        fixture, unit, directory, path = self.uva()
        events = []
        def recover(checkpoint, identity):
            """Represent the reviewed metadata recovery boundary while retaining the existing valid pair."""
            self.assertEqual(checkpoint, directory / "checkpoints/latest.pkl")
            self.assertEqual(identity, final.final_training_identity(fixture.run, unit))
            events.append("recovery")
            (directory / "checkpoints/latest.transaction.json").unlink()
            return {"recovered": True}
        original = final.checkpoint_record
        def read(*args, **kwargs):
            """Prove the ordinary reader observes an already recovered generation."""
            self.assertEqual(events, ["recovery"])
            self.assertFalse((directory / "checkpoints/latest.transaction.json").exists())
            events.append("validation")
            return original(*args, **kwargs)
        with mock.patch.object(final, "module", return_value=SimpleNamespace(recover_checkpoint=recover)), \
                mock.patch.object(final, "active_owned_jobs", return_value=({}, {"raw": "exact absent"})), \
                mock.patch.object(final, "checkpoint_record", side_effect=read):
            final.refresh(fixture.run, fixture.queue, {"9001": "TIMEOUT"})
        self.assertEqual(events, ["recovery", "validation"])
        self.assertEqual(final.read(path)["state"], "ready")
        self.assertEqual(final.read(path)["checkpoint_transaction_recovery"]["job_id"], "9001")

    def test_uva_active_or_requeued_job_never_invokes_recovery(self):
        """Current queue PENDING overrides a previously terminal observation before any write."""
        fixture, unit, directory, path = self.uva()
        with mock.patch.object(final, "module") as load, \
                mock.patch.object(final, "active_owned_jobs", return_value=({"9001": "PENDING"}, {})):
            final.refresh(fixture.run, fixture.queue, {"9001": "PREEMPTED"})
        load.assert_not_called()
        self.assertEqual(final.read(path)["state"], "held")
        self.assertTrue((directory / "checkpoints/latest.transaction.json").exists())

    def test_uva_old_terminal_status_cannot_recover_newer_job_or_pid(self):
        """A stale old job receipt cannot mutate the newer process's artifacts with the same identity."""
        for field, value in (("slurm_job_id", "9002"), ("pid", 556)):
            fixture, unit, directory, path = self.uva()
            hardware = final.read(directory / "hardware.json")
            hardware[field] = value
            fixture.write(directory / "hardware.json", hardware)
            before = (directory / "checkpoints/latest.pkl").read_bytes()
            with self.subTest(field=field), mock.patch.object(final, "module") as load, \
                    mock.patch.object(final, "active_owned_jobs", return_value=({}, {})):
                final.refresh(fixture.run, fixture.queue, {"9001": "TIMEOUT"})
            load.assert_not_called()
            self.assertEqual(final.read(path)["state"], "held")
            self.assertIn("another process or job", final.read(path)["reason"])
            self.assertEqual((directory / "checkpoints/latest.pkl").read_bytes(), before)

    def test_current_failure_or_changed_admitted_assets_holds_before_recovery(self):
        """Neither a current application failure nor changed asset identity grants recovery permission."""
        for change in ("failure", "assets", "progress"):
            fixture, unit, directory, path = self.uva()
            if change == "failure":
                fixture.write(directory / "attempts/attempt-0001/failure.json", {"attempt": 1})
            elif change == "assets":
                identity = final.read(directory / "training_identity.json")
                identity["assets"]["tree_digest"] = "changed"
                fixture.write(directory / "training_identity.json", identity)
            else:
                fixture.write(directory / "attempts/attempt-0001/progress.json", {"attempt": 2, "pid": 556})
            with self.subTest(change=change), mock.patch.object(final, "module") as load, \
                    mock.patch.object(final, "active_owned_jobs", return_value=({}, {})):
                final.refresh(fixture.run, fixture.queue, {"9001": "NODE_FAIL"})
            load.assert_not_called()
            self.assertEqual(final.read(path)["state"], "held")
            self.assertTrue((directory / "checkpoints/latest.transaction.json").exists())

    def test_no_transaction_leaves_legacy_readers_and_helpers_unchanged(self):
        """A valid old unit does not acquire a new runtime dependency just to read its existing checkpoint."""
        fixture, unit, directory, path = self.uva()
        (directory / "checkpoints/latest.transaction.json").unlink()
        with mock.patch.object(final, "module") as load, mock.patch.object(final, "active_owned_jobs") as query:
            final.refresh(fixture.run, fixture.queue, {"9001": "TIMEOUT"})
        load.assert_not_called()
        query.assert_not_called()
        self.assertEqual(final.read(path)["state"], "ready")

    def test_uva_unmarked_reserved_stage_holds_before_loading_helpers(self):
        """A valid old canonical pair cannot hide an unrecorded reserved next generation."""
        for state in ("running", "ready"):
            fixture, unit, directory, path = self.uva()
            final.atomic(path, {**final.read(path), "state": state})
            (directory / "checkpoints/latest.transaction.json").unlink()
            reserved = directory / "checkpoints/latest.next.pkl"
            reserved.write_bytes(b"unrecorded staged bytes")
            with self.subTest(state=state), mock.patch.object(final, "module") as load, \
                    mock.patch.object(final, "active_owned_jobs") as query:
                final.refresh(fixture.run, fixture.queue, {"9001": "TIMEOUT"})
            self.assertEqual(final.read(path)["state"], "held")
            self.assertIn("unmarked reserved", final.read(path)["reason"])
            self.assertEqual(reserved.read_bytes(), b"unrecorded staged bytes")
            load.assert_not_called()
            query.assert_not_called()

    def aces(self):
        """Build an exact owned ACES terminal attempt with its per-attempt placement record."""
        fixture = aces_fixtures.DeployedLauncherTests("runTest")
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        directory = fixture.run / "data" / fixture.unit["unit_id"]
        progress = {"attempt": 1, "pid": 555, "state": "running"}
        hardware = {"slurm_job_id": "500", "pid": 555, "hostname": "ac049"}
        for name, value in (("progress.json", progress), ("attempts/attempt-0001/progress.json", progress),
                            ("hardware.json", hardware), ("attempts/attempt-0001/placement_hardware.json", hardware),
                            ("training_identity.json", {"fixture": "strict verifier input"}),
                            ("checkpoints/latest.transaction.json", {"synthetic": "pending transaction"})):
            fixture.write(directory / name, value)
        fixture.write("final_training_queue/status/" + fixture.unit["unit_id"] + ".json", {
            "state": "running", "unit_id": fixture.unit["unit_id"], "last_job_cluster": "aces", "last_job_id": "500"})
        fixture.write("final_training_queue/jobs/aces_job-500.json", {"job_id": "500", "kind": "aces_final_h100_worker"})
        return fixture, directory

    def test_aces_identity_then_recovery_precede_terminal_snapshot_and_portable_validator(self):
        """New output evidence is created only after the exact terminal generation has recovered."""
        fixture, directory = self.aces()
        events = []
        def check_identity(unit, admitted, receipt, identity):
            """Retain the pinned strict deployed identity validator as a required first gate."""
            self.assertEqual(admitted, fixture.admitted)
            self.assertEqual(receipt, fixture.deployment)
            self.assertEqual(identity, {"fixture": "strict verifier input"})
            events.append("identity")
        def recover(path, identity):
            """Make the exact recovery boundary observable to later snapshot and validation mocks."""
            events.append("recovery")
            (directory / "checkpoints/latest.transaction.json").unlink()
            return {"recovered": True}
        def terminal(*args):
            """A terminal evidence snapshot must capture the recovered files."""
            self.assertEqual(events, ["identity", "recovery"])
            events.append("snapshot")
            return fixture.run / "terminal.json"
        def validate(*args, **kwargs):
            """Ordinary portable validation still decides whether the recovered checkpoint may continue."""
            self.assertEqual(events, ["identity", "recovery", "snapshot"])
            events.append("validation")
            return {"completed": False, "checkpoint_proof_attempt": 1}
        def module(unit, name):
            """Return only the pinned trainer helper needed for recovery."""
            self.assertEqual(name, "train_optimized")
            return SimpleNamespace(recover_checkpoint=recover)
        with mock.patch.object(deployed, "owned_compute"), \
                mock.patch.object(deployed, "accounting", return_value=({"500": {"state": "TIMEOUT"}}, None)), \
                mock.patch.object(final, "active_owned_jobs", return_value=({}, {})), \
                mock.patch.object(final, "module", side_effect=module), \
                mock.patch.object(deployed, "pre_recovery_identity", side_effect=check_identity), \
                mock.patch.object(deployed, "terminal_receipt", side_effect=terminal), \
                mock.patch.object(deployed, "checkpoint_selector", return_value=1), \
                mock.patch.object(deployed, "validation_command", side_effect=validate):
            result = deployed.refresh(fixture.run, fixture.policy)
        self.assertEqual(result["counts"]["ready"], 1)
        self.assertEqual(events, ["identity", "recovery", "snapshot", "validation"])

    def test_aces_old_status_cannot_recover_other_job_current_placement(self):
        """An ACES current placement for job 501 cannot be recovered under old terminal job 500."""
        fixture, directory = self.aces()
        fixture.write(directory / "attempts/attempt-0001/placement_hardware.json",
                      {"slurm_job_id": "501", "pid": 555, "hostname": "ac049"})
        with mock.patch.object(deployed, "owned_compute"), \
                mock.patch.object(deployed, "accounting", return_value=({"500": {"state": "PREEMPTED"}}, None)), \
                mock.patch.object(final, "active_owned_jobs", return_value=({}, {})), \
                mock.patch.object(deployed, "pre_recovery_identity"), \
                mock.patch.object(final, "module") as load, \
                mock.patch.object(deployed, "terminal_receipt") as snapshot, \
                mock.patch.object(deployed, "validation_command") as validate:
            result = deployed.refresh(fixture.run, fixture.policy)
        self.assertEqual(result["counts"]["held"], 1)
        load.assert_not_called()
        snapshot.assert_not_called()
        validate.assert_not_called()
        self.assertTrue((directory / "checkpoints/latest.transaction.json").exists())

    def test_aces_unmarked_reserved_stage_holds_before_validation_or_snapshot(self):
        """An unmarked stage cannot reach a new deployed validation receipt or worker release."""
        for state in ("running", "ready"):
            fixture, directory = self.aces()
            path = fixture.queue / "status" / (fixture.unit["unit_id"] + ".json")
            final.atomic(path, {**final.read(path), "state": state})
            (directory / "checkpoints/latest.transaction.json").unlink()
            reserved = directory / "checkpoints/latest.previous.metadata.json"
            reserved.write_bytes(b"unrecorded previous generation")
            with self.subTest(state=state), mock.patch.object(deployed, "owned_compute"), \
                    mock.patch.object(deployed, "accounting", return_value=({"500": {"state": "TIMEOUT"}}, None)), \
                    mock.patch.object(deployed, "pre_recovery_identity") as identity, \
                    mock.patch.object(final, "module") as load, \
                    mock.patch.object(deployed, "terminal_receipt") as snapshot, \
                    mock.patch.object(deployed, "validation_command") as validate:
                result = deployed.refresh(fixture.run, fixture.policy)
            self.assertEqual(result["counts"]["held"], 1)
            self.assertIn("unmarked reserved", result["units"][0]["reason"])
            self.assertEqual(reserved.read_bytes(), b"unrecorded previous generation")
            identity.assert_not_called()
            load.assert_not_called()
            snapshot.assert_not_called()
            validate.assert_not_called()
            # The direct submission entry also refuses the same unmarked stage,
            # even if a stale caller has not refreshed its previously ready status.
            final.atomic(path, {**final.read(path), "state": "ready"})
            with mock.patch.object(deployed.socket, "gethostname", return_value="ac049"), \
                    mock.patch.object(deployed, "owned_compute"), mock.patch.object(deployed, "verify_policy"), \
                    mock.patch.object(deployed, "live_monitor"), \
                    mock.patch.object(deployed, "accounting", return_value=({}, {})), \
                    mock.patch.object(deployed, "verify_binding", return_value=(fixture.unit, {}, {})), \
                    mock.patch.object(deployed.subprocess, "run") as submit:
                with self.assertRaisesRegex(ValueError, "unmarked reserved"):
                    deployed.submit(fixture.run, fixture.policy)
            submit.assert_not_called()

    def test_pre_recovery_identity_uses_fresh_pinned_dependencies_and_checks_exact_digest(self):
        """The helper's adjacent dependency wins over a deliberately different controller module."""
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / final.PREFIX
            directory.mkdir(parents=True)
            (directory / "train_deployed.py").write_text('MARKER = "original pinned dependency"\n')
            (directory / "validate_deployed_training_completion.py").write_text(
                "import hashlib, json, types\nimport train_deployed\n"
                "portable_evidence = types.SimpleNamespace()\n"
                "deployment = types.SimpleNamespace(DeploymentEvidence=lambda: None)\n"
                "def checkpoint_identity(admitted, receipt, identity, evidence):\n"
                "    assert train_deployed.MARKER == 'original pinned dependency'\n"
                "    return hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()\n")
            unit = {"python": sys.executable, "helpers": {"source_root": temporary}}
            identity = {"source_commit": "a" * 40, "helpers": unit["helpers"], "configuration": {"precision": "float32 networks/float64 statistics"}}
            with mock.patch.dict(sys.modules, {"train_deployed": SimpleNamespace(MARKER="new controller dependency")}):
                deployed.pre_recovery_identity(unit, {}, {}, identity)
            wrong = subprocess.CompletedProcess([], 0, json.dumps("0" * 64), "")
            with mock.patch.object(deployed.subprocess, "run", return_value=wrong) as call:
                with self.assertRaisesRegex(ValueError, "exact pre-recovery deployed identity"):
                    deployed.pre_recovery_identity(unit, {}, {}, identity)
            self.assertEqual(call.call_args.args[0][1], "-I")
            self.assertEqual(call.call_args.args[0][-1], str(directory))
            self.assertEqual(json.loads(call.call_args.kwargs["input"])["identity"], identity)
            self.assertEqual(call.call_args.kwargs["timeout"], 120)
            changed = {**identity, "helpers": {"source_root": "/different/helper"}}
            with mock.patch.object(deployed.subprocess, "run") as query:
                with self.assertRaisesRegex(ValueError, "exact operational helper identity"):
                    deployed.pre_recovery_identity(unit, {}, {}, changed)
            query.assert_not_called()


if __name__ == "__main__":
    unittest.main()
