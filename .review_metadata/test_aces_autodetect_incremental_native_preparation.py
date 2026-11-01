"""Verify bounded continuation metadata with all SSH and native execution blocked."""

import copy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


HERE = Path(__file__).resolve().parent
definition = importlib.util.spec_from_file_location("aces_continuation_driver", HERE / "continue_aces_autodetect_native_preparation.py")
driver = importlib.util.module_from_spec(definition)
definition.loader.exec_module(driver)


class ContinuationTests(unittest.TestCase):
    """Keep exact reviewed inputs, exclusive publication and raw transport proof separate from actual execution."""

    @classmethod
    def setUpClass(cls):
        """Read the actual thirteen-document input graph without calling any external process."""
        cls.payload = driver.local_payload(HERE / "aces_autodetect_incremental_native_preparation_driver_plan.json")
        cls.plan, cls.data = driver.checked_payload(cls.payload)

    def setUp(self):
        """Create a private metadata-only directory and delete only these synthetic bytes afterward."""
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)

    def test_actual_reviewed_graph_and_forced_existing_master_are_exact(self):
        """The real plan preserves source, software and original inputs before any network call is possible."""
        with patch.object(driver.subprocess, "Popen", side_effect=AssertionError("transport forbidden")):
            payload = driver.local_payload(HERE / "aces_autodetect_incremental_native_preparation_driver_plan.json")
        plan, data = driver.checked_payload(payload)
        self.assertEqual(len(payload["documents"]), 13)
        self.assertEqual(data["continuation"]["source_commit"], driver.TIP)
        self.assertEqual(data["continuation"]["basis_commit"], driver.BASIS)
        self.assertEqual(data["commands"]["ssh_prefix"], driver.SSH)
        self.assertEqual((plan["git_packing_threads"], plan["git_address_space_limit_bytes"]), (2, 2 * 2 ** 30))
        self.assertEqual(sum(len(row["verified_files"]) for row in data["installed_files"]), 13342)
        self.assertEqual(len(driver.expected_roots(data["preserved"])), 8)
        self.assertFalse(any(name in sys.modules for name in driver.FORBIDDEN))

    def test_changed_plan_or_document_fails_before_network_or_output_creation(self):
        """A changed source/private interpreter/mapping input cannot start the transport or create an intent."""
        for name in ("continuation", "reused", "metadata_helper", "verification", "base_snapshot"):
            payload = copy.deepcopy(self.payload)
            payload["documents"][name] += "\n"
            with self.subTest(name=name), patch.object(driver.subprocess, "Popen", side_effect=AssertionError("network forbidden")):
                with self.assertRaisesRegex(ValueError, "reviewed input changed"):
                    driver.local_execute(payload, b"synthetic driver source")
        payload = copy.deepcopy(self.payload)
        payload["plan"] += "\n"
        with self.assertRaisesRegex(ValueError, "driver plan changed"):
            driver.checked_payload(payload)

    def test_foreign_local_or_remote_owner_rejects_before_any_mutation(self):
        """Neither a compute host nor a foreign native owner may use the execution path."""
        with patch.object(driver.socket, "gethostname", return_value="slurm2"), \
                patch.object(driver.subprocess, "Popen", side_effect=AssertionError("network forbidden")):
            with self.assertRaisesRegex(ValueError, "only from portal05"):
                driver.local_execute(self.payload, b"synthetic source")
        with patch.object(driver.os, "getuid", return_value=1), \
                patch.object(driver.subprocess, "Popen", side_effect=AssertionError("native commands forbidden")):
            with self.assertRaisesRegex(ValueError, "exact ACES login owner"):
                driver.remote_main(self.payload)

    def test_partial_intent_is_preserved_and_never_reused(self):
        """A failed owned preparation records its failure and a second call stops before any process."""
        plan = {**self.plan, "local_execution_directory": str(self.directory / "execution")}
        with patch.object(driver, "checked_payload", return_value=(plan, self.data)), \
                patch.object(driver.socket, "gethostname", return_value="portal05"), \
                patch.object(driver.pwd, "getpwuid", return_value=SimpleNamespace(pw_name="sl5nw")), \
                patch.object(driver, "build_bundle", side_effect=ValueError("synthetic bundle failure")), \
                patch.object(driver.subprocess, "Popen", side_effect=AssertionError("no transport")):
            with self.assertRaisesRegex(ValueError, "synthetic bundle failure"):
                driver.local_execute(self.payload, b"synthetic source")
            intent = (Path(plan["local_execution_directory"]) / "intent.json").read_bytes()
            self.assertIn("synthetic bundle failure", (Path(plan["local_execution_directory"]) / "failure.json").read_text())
            with self.assertRaisesRegex(ValueError, "previous local execution"):
                driver.local_execute(self.payload, b"synthetic source")
            self.assertEqual((Path(plan["local_execution_directory"]) / "intent.json").read_bytes(), intent)

    def test_command_failure_and_timeout_keep_raw_bytes_before_rejection(self):
        """Failed metadata commands retain their exact intent, status and received stdout/stderr."""
        for timeout in (False, True):
            audit = self.directory / str(timeout)
            audit.mkdir()
            outcome = subprocess.TimeoutExpired(["synthetic"], 1, output=b"partial stdout\n", stderr=b"partial stderr\n") if timeout \
                else SimpleNamespace(returncode=2, stdout=b"raw stdout\n", stderr=b"raw stderr\n")
            with patch.object(driver.subprocess, "run", side_effect=outcome if timeout else None,
                              return_value=outcome):
                with self.assertRaises(subprocess.TimeoutExpired if timeout else ValueError):
                    driver.invoke(["synthetic"], audit, timeout=1)
            self.assertTrue((audit / "command-0001.intent.json").exists())
            record = json.loads((audit / "command-0001.json").read_bytes())
            self.assertEqual(record["returncode"], None if timeout else 2)
            self.assertEqual((audit / "command-0001.stdout.raw").read_bytes(), b"partial stdout\n" if timeout else b"raw stdout\n")

    def test_bundle_publication_never_overwrites_a_conflict(self):
        """A different existing target and the new complete temporary bytes both survive."""
        temporary, target = self.directory / "incoming", self.directory / "bundle"
        temporary.write_bytes(b"new complete bytes")
        target.write_bytes(b"existing different bytes")
        with self.assertRaises(FileExistsError):
            driver.publish_bundle(temporary, target)
        self.assertEqual(target.read_bytes(), b"existing different bytes")
        self.assertEqual(temporary.read_bytes(), b"new complete bytes")

    def test_bundle_stream_requires_exact_hash_length_and_no_extra_bytes(self):
        """Successful bytes publish once; incomplete, extra or changed bytes stay as partial evidence."""
        content = b"synthetic incremental bytes"
        expected = {"bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()}
        for name, raw in (("complete", content), ("incomplete", content[:-1]), ("extra", content + b"x"), ("changed", b"x" * len(content))):
            target = self.directory / name
            if name == "complete":
                driver.receive_bundle(io.BytesIO(raw), target, expected)
                self.assertEqual(target.read_bytes(), content)
            else:
                with self.subTest(name=name), self.assertRaises(ValueError):
                    driver.receive_bundle(io.BytesIO(raw), target, expected)
                self.assertFalse(target.exists())
                self.assertTrue(target.with_name(target.name + ".incomplete").exists())

    def test_bundle_header_binds_one_reference_and_all_prerequisites(self):
        """A foreign reference, omitted basis or duplicate prerequisite cannot be imported."""
        reference = "refs/transfer/aces_helpers_" + driver.TIP
        valid = b"# v2 git bundle\n-" + driver.BASIS.encode() + b" basis\n" + driver.TIP.encode() + b" " + reference.encode() + b"\n\nPACK"
        path = self.directory / "header.bundle"
        path.write_bytes(valid)
        self.assertEqual(driver.bundle_header(path, driver.TIP, reference, driver.BASIS), [driver.BASIS])
        for changed in (valid.replace(reference.encode(), b"refs/heads/foreign"), valid.replace(driver.BASIS.encode(), b"0" * 40),
                        valid.replace(b" basis\n", b" basis\n-" + driver.BASIS.encode() + b" duplicate\n")):
            path.write_bytes(changed)
            with self.assertRaises(ValueError):
                driver.bundle_header(path, driver.TIP, reference, driver.BASIS)

    def test_real_small_incremental_bundle_has_exact_prerequisite_and_removes_only_its_ref(self):
        """Actual local Git proves the bundle is incremental and the temporary reference is removed."""
        root, audit = self.directory / "repository", self.directory / "audit"
        root.mkdir()
        audit.mkdir()

        def git(*arguments):
            """Run bounded local fixture Git only, with hooks disabled and an explicit test identity."""
            return subprocess.run(["git", "-C", str(root), "-c", "core.hooksPath=/dev/null", *arguments], check=True,
                                  capture_output=True, timeout=30).stdout.decode().strip()

        git("init", "-q")
        (root / "source.py").write_text("original = 1\n")
        git("add", ".")
        git("-c", "user.name=Metadata test", "-c", "user.email=test@example.invalid", "commit", "-qm", "Record basis")
        basis = git("rev-parse", "HEAD")
        (root / "source.py").write_text("original = 1\nnew = 2\n")
        git("add", ".")
        git("-c", "user.name=Metadata test", "-c", "user.email=test@example.invalid", "commit", "-qm", "Record increment")
        tip = git("rev-parse", "HEAD")
        git("update-ref", "refs/remotes/origin/merge-atari-cleanrl", tip)
        continuation = {"source_commit": tip, "basis_commit": basis, "source_tree": git("rev-parse", "HEAD^{tree}"),
                        "new_local_bundle_planned_path": str(self.directory / "new.bundle")}
        record = driver.build_bundle(root, audit, continuation)
        self.assertEqual(record["prerequisites"], [basis])
        self.assertEqual(record["packing_threads"], 2)
        self.assertEqual(git("show-ref", "--verify", "refs/remotes/origin/merge-atari-cleanrl").split()[0], tip)
        self.assertNotIn("refs/transfer/aces_helpers_", git("show-ref"))
        with self.assertRaisesRegex(ValueError, "already exists"):
            driver.build_bundle(root, audit, continuation)

    def test_root_flattening_preserves_dedicated_clone_names_and_rejects_conflicts(self):
        """Human group labels cannot replace the actual pinned physical root path."""
        roots = driver.expected_roots(self.data["preserved"])
        self.assertIn(self.data["reused"]["numerical_source"]["source_root"], roots)
        changed = copy.deepcopy(self.data["preserved"])
        value = copy.deepcopy(next(iter(changed["roots"]["old"].values())))
        value["commit"] = "0" * 40
        changed["roots"]["cloned"]["conflict"] = value
        with self.assertRaisesRegex(ValueError, "conflicting identities"):
            driver.expected_roots(changed)

    def return_rows(self):
        """Create only synthetic path/hash rows for the exact new native packet/data/audit roster."""
        names = driver.PACKET_FILES | {"admission_files/" + Path(row["captured"]["path"]).name
                                      for row in self.data["old_plan"]["admission_archive"]["files"]}
        audit = {"intent.json", "preserved_before.json", "preserved_after.json", "new_helper.json", "native_verification.json", "result.json"}
        rows = []
        for prefix, root, names in (("packet", self.data["continuation"]["planned_native_packet_directory"], names),
                ("data", self.data["continuation"]["planned_native_data_unit"], driver.STANDARD_FILES),
                ("audit", self.plan["remote_execution_directory"], audit)):
            rows += [{"relative_path": prefix + "/" + name, "path": str(Path(root) / name),
                      "sha256": hashlib.sha256(b"synthetic").hexdigest(), "bytes": len(b"synthetic")} for name in sorted(names)]
        return rows

    def test_return_roster_rejects_missing_duplicate_foreign_and_escaped_paths(self):
        """Only complete original-path-bound packet/data artifacts can enter the fresh local receiver."""
        rows = self.return_rows()
        driver.validate_return_roster(rows, self.plan, self.data)
        changes = []
        changes.append(rows[:-1])
        changes.append(rows + [rows[0]])
        changed = copy.deepcopy(rows)
        changed[0]["path"] = "/foreign/packet/file.json"
        changes.append(changed)
        changed = copy.deepcopy(rows)
        changed[0]["relative_path"] = "packet/../../escape"
        changes.append(changed)
        changed = copy.deepcopy(rows)
        changed[0]["bytes"] = True
        changes.append(changed)
        for changed in changes:
            with self.subTest(changed=changed[0]["relative_path"]), self.assertRaises(ValueError):
                driver.validate_return_roster(changed, self.plan, self.data)

    def test_response_prefix_or_driver_hash_cannot_create_received_files(self):
        """Raw errors remain available without being interpreted as a successful native preparation."""
        path = self.directory / "stdout.raw"
        path.write_bytes(b"native command failed before producing a frame\n")
        payload = {**self.payload, "driver_sha256": "a" * 64}
        with self.assertRaisesRegex(ValueError, "unexpected prefix"):
            driver.accept_return(path, self.directory, payload)
        self.assertFalse((self.directory / "received").exists())
        self.assertTrue(path.exists())

    def synthetic_return(self, *, change_bytes=False, grant_credit=False):
        """Create a fully labelled synthetic raw return from actual reviewed configuration metadata, never native observations."""
        encode = lambda value: (json.dumps(value, sort_keys=True) + "\n").encode()
        continuation = self.data["continuation"]
        packet = continuation["planned_native_packet_directory"]
        current = copy.deepcopy(self.data["old_plan"])
        current["helpers"].update(source_root=continuation["new_helper_checkout_root"], source_commit=driver.TIP,
                                  files=continuation["expected_helper_closure"])
        current["qualification_profile"]["name"] = continuation["profile_name"]
        current["qualification_profile"]["physical_gres_mapping_verified"] = False
        current["qualification_profile"]["kernel_device_enforcement_verified"] = False
        current["data_unit"] = {"name": Path(packet).name, "path": continuation["planned_native_data_unit"]}
        current["software_at_prepare"] = {"path": packet + "/software_at_prepare.json",
            "sha256": hashlib.sha256(self.payload["documents"]["software"].encode()).hexdigest()}
        old_packet = Path(self.plan["documents"]["old_plan"]["path"]).parent
        contents = {"packet/admission_files/" + Path(row["captured"]["path"]).name:
            (old_packet / "admission_files" / Path(row["captured"]["path"]).name).read_bytes()
            for row in current["admission_archive"]["files"]}
        for row in current["admission_archive"]["files"]:
            row["captured"]["path"] = packet + "/admission_files/" + Path(row["captured"]["path"]).name
        shape = current["job_shape"]
        command = ["sbatch", "--parsable", "--hold", *["--" + flag + "=" + str(shape[key]) for flag, key in
            (("account", "account"), ("partition", "partition"), ("qos", "qos"), ("nodes", "nodes"), ("ntasks", "ntasks"),
             ("gres", "gres"), ("cpus-per-task", "cpus_per_task"), ("mem", "memory"), ("time", "time"))],
            "--job-name=ACES_H100_full_site_qualification", "--output=" + current["campaign_run_dir"] + "/slurm/jobs/aces_full_site_qualification_%j.log",
            packet + "/job.sbatch"]
        contents.update({"packet/site_validation_plan.json": encode(current), "packet/software_at_prepare.json": self.payload["documents"]["software"].encode(),
            "packet/job.sbatch": b"# synthetic metadata fixture; never execute\n", "packet/command.txt": (shlex.join(command) + "\n").encode(),
            "packet/config_resolved.yaml": encode(current), "packet/experiment_background.md": b"Synthetic zero-credit metadata fixture.\n"})
        manifest = {"artifacts": {name: {"path": packet + "/" + name,
            "sha256": hashlib.sha256(contents["packet/" + name]).hexdigest()}
            for name in driver.PACKET_FILES - {"manifest.yaml"}}, "submission_command": command}
        contents["packet/manifest.yaml"] = encode(manifest)
        contents.update({"data/" + name: contents["packet/" + name] for name in driver.STANDARD_FILES})
        result = {"passed": True, "plan_sha256": driver.PLAN_SHA, "driver_sha256": "a" * 64,
            "source_commit": driver.TIP, "basis_commit": driver.BASIS, "preserved_before_after": True,
            "native_packet": packet, "native_data_unit": continuation["planned_native_data_unit"],
            "workers_permitted": False, "actual_site_success": False, "jobs_submitted": 0,
            "scientific_hypothesis_credit": 0, "comparison_credit": 0, "idea_credit": 1 if grant_credit else 0, "admission_credit": 0}
        preservation = {"roots": driver.expected_roots(self.data["preserved"]), "base_package_count": 77,
            "private_package_count": 7, "active_dependency_count": 128, "base_metadata_and_shared_files_unchanged": True,
            "private_metadata_unchanged": True, "files": {"mapping_files": 72, "asset_files": 4531,
                "installed_locked_files": sum(len(row["verified_files"]) for row in self.data["installed_files"])}}
        audit = {"result.json": result, "intent.json": {"plan_sha256": driver.PLAN_SHA, "driver_sha256": "a" * 64},
            "preserved_before.json": preservation, "preserved_after.json": preservation,
            "new_helper.json": {"source_commit": driver.TIP, "source_tree": continuation["source_tree"], "root": continuation["new_helper_checkout_root"], "helper_count": 18},
            "native_verification.json": {"passed": True, "private_python": self.data["reused"]["private_python"],
                "helper_commit": driver.TIP, "helper_count": 18, "actual_site_success": False, "workers_permitted": False}}
        contents.update({"audit/" + name: encode(value) for name, value in audit.items()})
        rows = []
        for relative, content in sorted(contents.items()):
            prefix, name = relative.split("/", 1)
            root = {"packet": packet, "data": continuation["planned_native_data_unit"], "audit": self.plan["remote_execution_directory"]}[prefix]
            rows.append({"relative_path": relative, "path": root + "/" + name, "bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()})
        header = {"schema_version": 1, "kind": "aces_autodetect_native_preparation_return", "result": result, "files": rows}
        raw = encode(header)
        body = b"".join(contents[row["relative_path"]] for row in rows)
        if change_bytes:
            body = body[:-1] + bytes([body[-1] ^ 1])
        return driver.MAGIC + str(len(raw)).encode() + b"\n" + raw + body, header

    def test_exact_synthetic_raw_return_retains_native_paths_and_initial_data_bytes(self):
        """The receiver copies complete actual-shape raw metadata while making no real native-success claim."""
        raw, expected = self.synthetic_return()
        path = self.directory / "stdout.raw"
        path.write_bytes(raw)
        payload = {**self.payload, "driver_sha256": "a" * 64}
        actual = driver.accept_return(path, self.directory, payload)
        self.assertEqual(actual, expected)
        self.assertEqual(path.read_bytes(), raw)
        for row in actual["files"]:
            self.assertEqual(driver.sha(self.directory / "received" / row["relative_path"]), row["sha256"])
        self.assertFalse(actual["result"]["actual_site_success"])
        self.assertFalse(actual["result"]["workers_permitted"])
        with self.assertRaisesRegex(ValueError, "prior returned-file"):
            driver.accept_return(path, self.directory, payload)

    def test_changed_raw_payload_or_credit_cannot_publish_an_accepted_return(self):
        """Corrupt bytes and coherent nonzero-credit claims are retained as raw failures, never accepted."""
        for name, options in (("changed", {"change_bytes": True}), ("credit", {"grant_credit": True})):
            directory = self.directory / name
            directory.mkdir()
            raw, unused = self.synthetic_return(**options)
            path = directory / "stdout.raw"
            path.write_bytes(raw)
            with self.subTest(name=name), self.assertRaises(ValueError):
                driver.accept_return(path, directory, {**self.payload, "driver_sha256": "a" * 64})
            self.assertEqual(path.read_bytes(), raw)
            self.assertFalse((directory / "accepted_return.json").exists())


if __name__ == "__main__":
    unittest.main()
