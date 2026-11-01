"""Use synthetic device/host records and real local files; grant no GPU qualification."""

import copy
from contextlib import contextmanager
import json
import os
from pathlib import Path
import shutil
import sqlite3
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

SCRIPTS = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(SCRIPTS), str(SCRIPTS / "tests")]
import configured_measurement as shared
import direct_measurement as direct
import measure_selected as selected
import profile_selected as profile
import serval02_card1_binding as core
import test_serval02_card1_binding as binding_tests
from test_configured_measurement import memory_window, raw


class DirectMeasurementTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        binding_tests.DirectBindingTests.setUpClass()
        cls.names = tuple(dict.fromkeys((*direct.helper_names(), "measure_selected.py", "profile_selected.py")))
        for name in cls.names:
            shutil.copyfile(SCRIPTS / name, binding_tests.DirectBindingTests.source / core.PREFIX / name)
        subprocess.run(["git", "-C", str(binding_tests.DirectBindingTests.source), "add", "."], check=True)
        subprocess.run(["git", "-C", str(binding_tests.DirectBindingTests.source), "-c", "user.name=Metadata Test", "-c", "user.email=metadata@example.invalid",
            "commit", "-qm", "Record complete synthetic direct producer closure"], check=True)
        binding_tests.DirectBindingTests.commit = subprocess.check_output(["git", "-C", str(binding_tests.DirectBindingTests.source), "rev-parse", "HEAD"], text=True).strip()

    @classmethod
    def tearDownClass(cls):
        binding_tests.DirectBindingTests.tearDownClass()

    def setUp(self):
        self.f = binding_tests.DirectBindingTests(); self.f.setUp(); self.addCleanup(self.f.doCleanups)
        f = self.f
        f.plan["python"] = sys.executable
        f.plan["helpers"]["files"] = {core.PREFIX + n: core.receipt(SCRIPTS / n)["sha256"] for n in self.names}
        f.write(f.unit / "measurement_plan.json", f.plan)
        f.launch["plan"] = core.receipt(f.unit / "measurement_plan.json")
        f.write(f.unit / "direct_current_launch.json", f.launch)
        f.metadata["current_launch"] = f.launch
        f.binding.update(plan=core.receipt(f.unit / "measurement_plan.json"), metadata_sha256=core.digest(core.encoded(f.metadata)))
        f.write(f.unit / "direct_process_metadata.json", f.metadata)
        f.write(f.unit / "direct_device_binding.json", f.binding)
        for name in self.names:
            path = f.unit / "code" / name; path.parent.mkdir(exist_ok=True)
            shutil.copyfile(SCRIPTS / name, path)

    def direct_window(self):
        f = self.f; value = memory_window(); gpu = f.profile["gpu_uuid"]; pid = f.identity["pid"]
        value.update(kind=shared.DIRECT_OBSERVATION_CAPABILITY, gpu_uuid=gpu, pid=pid,
            direct_binding=core.receipt(f.unit / "direct_device_binding.json"), startup_memory_observed=False)
        boundary = {"custody": f.binding["custody_after"], "current_admission": f.metadata["current_admission"],
            "binding": value["direct_binding"]}
        value.update(direct_before=boundary, direct_after=copy.deepcopy(boundary))
        for sample in value["samples"]:
            sample.update(gpu_uuid=gpu, total_bytes=95830 * 2 ** 20, free_bytes=95318 * 2 ** 20)
            device = sample["device_query"]; device["command"][2] = gpu
            device["stdout"] = f"{gpu}, 95830, 512, 95318\n"; device["sha256"] = shared.sha(device["stdout"].encode())
            old = sample["process_query"]
            converted = shared.process_sample(gpu, pid, lambda cmd: raw(f"{gpu}, {pid}, 512\n".encode()))
            converted.update({key: old[key] for key in ("started_at_seconds", "finished_at_seconds")})
            sample["process_query"] = converted
        return value

    def profile_chain(self):
        f = self.f; f.plan["profile_tool"] = {"path": "/synthetic/nsys"}; f.profile_parent()
        holder = f.identity; nsight_pid = holder["pid"] + 500; waiter_pid = holder["pid"] + 600; capture_pid = holder["pid"] + 700
        def record(pid, parent, birth):
            value = f.process_record(pid, parent, birth, nsight_pid)
            fields = value["stat"].rsplit(")", 1)[1].split(); fields[3] = str(nsight_pid)
            value["stat"] = str(pid) + " (synthetic_direct_process) " + " ".join(fields)
            value["stat_sha256"] = core.digest(value["stat"].encode()); return value
        owner = {"pid": nsight_pid, "process_start_time": "8000", "parent_pid": holder["pid"],
            "parent_start_time": holder["process_start_time"], "command": profile.child_command(f.unit, f.plan, f.fd)}
        f.write(f.unit / "profiler_process.json", owner)
        waiter_record = record(waiter_pid, nsight_pid, "9000")
        waiter = {"schema_version": 1, "kind": "direct_metadata_only_profile_waiter", "process": waiter_record,
            "plan": core.receipt(f.unit / "measurement_plan.json"), "parent_custody": core.receipt(f.unit / "supervisor/direct_process_metadata.json"),
            "profiler_process": core.receipt(f.unit / "profiler_process.json"), "lock": f.metadata["lock_after"], **direct.BOUNDARY}
        f.write(f.unit / "direct_profile_waiter.json", waiter)
        child = record(capture_pid, waiter_pid, "10000")
        child_raw = {"pid": capture_pid, **{key: child[key] for key in ("stat", "stat_sha256", "uid_bytes", "uid_sha256")}}
        launch = {"schema_version": 1, "kind": "direct_actual_capture_child_launch", "command": direct.capture_command(f.plan, f.unit, f.fd),
            "child": child_raw, "pid": capture_pid, "process_start_time": "10000", "parent_pid": waiter_pid,
            "parent_start_time": "9000", "uid": 2618919, "cmdline_sha256": "1" * 64,
            "waiter": core.receipt(f.unit / "direct_profile_waiter.json"), **direct.BOUNDARY}
        f.write(f.unit / "direct_capture_launch.json", launch)
        metadata = {"after": child, "lock_after": f.metadata["lock_after"]}
        wait = {"schema_version": 1, "kind": "direct_actual_capture_child_wait_return", "command": launch["command"],
            "launch": core.receipt(f.unit / "direct_capture_launch.json"), "pid": capture_pid, "waited_pid": capture_pid,
            "process_start_time": "10000", "uid": 2618919, "returncode": 0, "exited_normally": True,
            "wait_semantics": "Popen.wait on this exact spawned child", "waiter_after": waiter_record,
            "lock_after": f.metadata["lock_after"], **direct.BOUNDARY}
        f.write(f.unit / "direct_capture_wait_return.json", wait)
        return metadata, launch, waiter, wait

    def trace_fixture(self):
        metadata, unused_launch, unused_waiter, unused_wait = self.profile_chain(); f = self.f
        trace = f.unit / "trace"; trace.mkdir(); (trace / "capture.nsys-rep").write_bytes(b"synthetic native report")
        path = trace / "capture.sqlite"; pid = metadata["after"]["identity"]["pid"]
        with sqlite3.connect(path) as connection:
            connection.execute("CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL(globalPid INTEGER,start INTEGER,end INTEGER)")
            connection.execute("INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES(?,?,?)", (pid * 2 ** 24, 1, 4))
            connection.execute("CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME(globalTid INTEGER,start INTEGER,end INTEGER)")
            connection.execute("INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES(?,?,?)", (pid * 2 ** 24 + 42, 0, 3))
        def save_analysis():
            f.write(trace / "analysis.json", {"completed": True, "derived_files": [{**core.receipt(path), "bytes": path.stat().st_size}]})
        save_analysis(); return metadata, path, save_analysis

    def test_direct_full_window_is_distinct_and_replays_without_live_queries(self):
        proof = self.direct_window()
        with patch.object(core, "check_live_admission", side_effect=AssertionError("saved replay cannot use live authority")), patch.object(shared, "bounded_query", side_effect=AssertionError("no GPU query")):
            value = direct.validate_window(json.loads(core.encoded(proof)), self.f.unit)
        self.assertEqual(value["sampled_process_peak_bytes"], 512 * 2 ** 20)
        with self.assertRaises(ValueError): shared.validate_window(proof)

    def test_direct_window_rejects_wrong_total_foreign_row_bad_shutdown_and_lost_custody(self):
        for change in (lambda v: v["samples"][0].update(total_bytes=20470 * 2 ** 20),
                lambda v: v["samples"][0]["process_query"].update(other_active_compute_rows=1),
                lambda v: v.update(normal_requested_stop=False), lambda v: v["direct_after"]["custody"]["lock"].update(fd=99999),
                lambda v: v["samples"][0]["device_query"].update(stdout="changed")):
            value = self.direct_window(); change(value)
            with self.subTest(change=change), self.assertRaises(ValueError): direct.validate_window(value, self.f.unit)

    def test_direct_window_keeps_actual_fifteen_percent_free_requirement(self):
        value = self.direct_window(); sample = value["samples"][0]
        sample.update(used_bytes=90000 * 2 ** 20, free_bytes=5830 * 2 ** 20)
        text = f"{self.f.profile['gpu_uuid']}, 95830, 90000, 5830\n"
        sample["device_query"].update(stdout=text, sha256=shared.sha(text.encode()))
        with self.assertRaisesRegex(ValueError, "fifteen percent"): direct.validate_window(value, self.f.unit)

    def test_complete_helpers_are_rechecked_before_observation(self):
        direct.verify_observer_helpers(self.f.plan, self.f.unit)
        (self.f.unit / "code/direct_measurement.py").write_text("changed source\n")
        with self.assertRaises(ValueError): direct.observer_inputs(self.f.binding, self.f.unit / "paired_memory")

    def test_observer_refuses_remembered_binding_and_wrong_directory_before_queries(self):
        for binding, directory in (({**self.f.binding, "pci_address": "0000:00:00.0"}, self.f.unit / "paired_memory"),
                (self.f.binding, self.f.unit / "unapproved_memory")):
            with self.subTest(directory=directory), patch.object(shared, "bounded_query") as query:
                with self.assertRaises(ValueError): direct.MemoryObservation(None, binding, directory)
                query.assert_not_called()

    def test_paired_observer_reuses_unique_windows_and_restores_original(self):
        original = object(); paired = SimpleNamespace(MemoryObservation=original); device = object()
        observer = SimpleNamespace(thread=None)
        with patch.object(direct, "MemoryObservation", return_value=observer) as created:
            with direct.paired_observer(paired, self.f.binding, self.f.unit):
                self.assertIs(paired.MemoryObservation(device), observer)
                self.assertIs(paired.MemoryObservation(device), observer)
                with self.assertRaises(ValueError): paired.MemoryObservation(object())
            created.assert_called_once(); self.assertIs(paired.MemoryObservation, original)

    def test_window_cleanup_preserves_original_numerical_failure(self):
        observer = SimpleNamespace(begin=lambda: None, finish=lambda: (_ for _ in ()).throw(ValueError("sampler failure")))
        with patch.object(direct, "MemoryObservation", return_value=observer):
            with self.assertRaisesRegex(RuntimeError, "original update"):
                with direct.observed_work(None, self.f.binding, self.f.unit / "correctness_memory"):
                    raise RuntimeError("original update")
        self.assertTrue((self.f.unit / "correctness_memory/sampling_failure_after_numerical_failure.json").exists())

    def test_original_selectors_are_not_rewritten_by_child_command(self):
        f = self.f; f.plan["profile_tool"] = {"path": "/synthetic/nsys"}
        ordinary = copy.deepcopy(f.plan); ordinary.pop("direct_binding")
        baseline = profile.child_command(f.unit, ordinary)
        command = profile.child_command(f.unit, f.plan, f.fd)
        self.assertNotIn("--wait=primary", baseline); self.assertIn("--wait=primary", command)
        self.assertIn("--direct-waiter-unit", command); self.assertNotIn("--capture-unit", command)
        self.assertEqual(command[-2:], ["--direct-held-lock-fd", str(f.fd)])
        self.assertEqual(f.plan["gpu_uuid"], f.profile["gpu_uuid"])
        with self.assertRaises(ValueError): profile.child_command(f.unit, f.plan)
        with self.assertRaises(ValueError): profile.child_command(f.unit, ordinary, f.fd)

    def test_private_descriptor_and_waiter_flags_cannot_change_other_modes(self):
        for args in (["--prepared-unit", str(self.f.unit), "--direct-held-lock-fd", "3"],
                ["--direct-waiter-unit", str(self.f.unit)], ["--capture-unit", str(self.f.unit), "--direct-history-inputs", "/changed"]):
            with self.subTest(args=args), self.assertRaises(SystemExit): profile.parse_arguments(args)

    def test_actual_waiter_chain_and_zero_child_wait_are_required(self):
        metadata, launch, waiter, value = self.profile_chain()
        direct.validate_capture_wait(self.f.unit, metadata)
        for change in (lambda v: v.update(returncode=1), lambda v: v.update(returncode=-9, exited_normally=False),
                lambda v: v.update(returncode=True), lambda v: v.update(waited_pid=v["pid"] + 1),
                lambda v: v.update(wait_semantics="Nsight returned zero"), lambda v: v.update(command=["wrong-python"])):
            changed = copy.deepcopy(value); change(changed); self.f.write(self.f.unit / "direct_capture_wait_return.json", changed)
            with self.subTest(change=change), self.assertRaises(ValueError): direct.validate_capture_wait(self.f.unit, metadata)

    def test_wrong_capture_parent_or_dropped_descriptor_refuses_before_live_reads(self):
        metadata, launch, waiter, value = self.profile_chain()
        with patch.object(core, "capture_owner_process") as live:
            with self.assertRaises(ValueError): direct.validate_waiter_chain(self.f.plan, self.f.unit, metadata["after"], self.f.fd + 1, live=True)
            live.assert_not_called()
        launch["parent_start_time"] = "1"; self.f.write(self.f.unit / "direct_capture_launch.json", launch)
        with self.assertRaises(ValueError): direct.validate_capture_wait(self.f.unit, metadata)

    def test_native_trace_positive_events_join_actual_capture_and_exact_bytes(self):
        metadata, path, save = self.trace_fixture()
        receipt = direct.trace_process_proof(self.f.unit, metadata)
        self.assertEqual(receipt, direct.trace_process_proof(self.f.unit, metadata, publish_result=False))
        proof = core.read_receipt(receipt); self.assertEqual((proof["kernels"], proof["cuda_api_calls"]), (1, 1))
        path.write_bytes(path.read_bytes() + b"changed")
        with self.assertRaises(ValueError): direct.trace_process_proof(self.f.unit, metadata, publish_result=False)

    def test_native_trace_other_pid_zero_events_and_unknown_schema_refuse(self):
        metadata, path, save = self.trace_fixture()
        with sqlite3.connect(path) as connection:
            connection.execute("UPDATE CUPTI_ACTIVITY_KIND_KERNEL SET globalPid=?", (55 * 2 ** 24,))
        save()
        with self.assertRaises(ValueError): direct.trace_process_proof(self.f.unit, metadata)
        with sqlite3.connect(path) as connection:
            connection.execute("DELETE FROM CUPTI_ACTIVITY_KIND_KERNEL")
        save()
        with self.assertRaises(ValueError): direct.trace_process_proof(self.f.unit, metadata)
        with sqlite3.connect(path) as connection:
            connection.execute("ALTER TABLE CUPTI_ACTIVITY_KIND_KERNEL RENAME COLUMN globalPid TO unknownIdentity")
        save()
        with self.assertRaises(ValueError): direct.trace_process_proof(self.f.unit, metadata)

    def test_all_unsupported_direct_modes_remain_rejected(self):
        for mode in ("packing-point", "learning-check", "final", "scaling"):
            changed = copy.deepcopy(self.f.plan); changed["mode"] = mode
            with self.subTest(mode=mode), self.assertRaises(ValueError): direct.support(changed)
        changed = copy.deepcopy(self.f.plan); changed["configured_isolation"] = {}
        with self.assertRaises(ValueError): direct.support(changed)
        changed = copy.deepcopy(self.f.plan); changed["device_memory_limit_protocol"] = {}
        with self.assertRaises(ValueError): direct.support(changed)


if __name__ == "__main__":
    unittest.main()
