import ast
from pathlib import Path
p=Path(__file__).parent
s=(p/'accepted_20_09_tests.py').read_text()
def replacefn(name,body):
 global s
 tree=ast.parse(s);cls=next(n for n in tree.body if isinstance(n,ast.ClassDef));n=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name==name)
 lines=s.splitlines(keepends=True);s=''.join(lines[:n.lineno-1])+body.rstrip()+'\n'+''.join(lines[n.end_lineno:])
replacefn('setUp','''    def setUp(self):
        """Prepare only labelled synthetic small source/metadata in a private local Git repository."""
        self.temporary = tempfile.TemporaryDirectory(); self.addCleanup(self.temporary.cleanup)
        self.integration = Path(self.temporary.name) / "synthetic_integration"
        self.run = self.integration / "platform/benchmark_runs/synthetic_campaign"
        literal = "2026-11-05-06-39_native-process-control_jaguar03_python-3.12_gpu-count-1_processor-count-1_host-memory-gib-2_total-seconds-300_work-seconds-240_cleanup-seconds-60_environment-transitions-0_training-updates-0"
        self.packet = self.run / "remote_setup" / literal; self.unit = self.run / "data" / literal
        for directory in (self.packet, self.unit / "code", self.run / "queue", self.run / "slurm/jobs"):
            directory.mkdir(parents=True, exist_ok=True)
        for name in ("launch_diagnostic_once.py", "test_launch_diagnostic_once.py"):
            (self.packet / name).write_bytes((ACTUAL_PACKET / name).read_bytes())
        specification = importlib.util.spec_from_file_location("private_configured_native_launcher", self.packet / "launch_diagnostic_once.py")
        self.module = importlib.util.module_from_spec(specification); specification.loader.exec_module(self.module)
        prefix = self.integration / self.module.PREFIX; prefix.mkdir(parents=True)
        for name in self.module.HELPERS:
            raw = ("# Clearly synthetic source fixture; no tool or native operation. " + name + "\\nVALUE = 1\\n").encode()
            (prefix / name).write_bytes(raw); (self.unit / "code" / name).write_bytes(raw)
        self.git("init", "-q"); source = self.commit()
        hashes = {self.module.PREFIX + name: self.digest(self.unit / "code" / name) for name in self.module.HELPERS}
        self.plan = {"schema_version": 1, "kind": self.module.KIND, "unit_id": literal, "campaign_run_dir": str(self.run),
            "packet": str(self.packet), "python": self.module.PYTHON, "hardware_class": "jaguar03_rtx_a4500", "gpu_uuid": None,
            "maximum_active_seconds": 240, "diagnostic_time_budget": dict(self.module.BUDGET), "diagnostic_boundary": dict(self.module.ZERO),
            "helpers": {"source_commit": source, "files": hashes},
            "profile_tool": {"path": "/explicitly-synthetic/no-tool", "version": "synthetic-only", "sha256": "0" * 64},
            "native_process_control": {"schema_version": 1, "backend": "linux-x86-64-libc-syscall-pidfd-version-1",
                "mode": "observe_configured_target_for_zero_credit_diagnostic", "expected_target_contract": None, "qualification": False,
                "python": self.module.PYTHON, "helper_hashes": {name: hashes[self.module.PREFIX + name]
                    for name in ("linux_pidfd_process_control.py", "native_process_evidence.py")}},
            "configured_native_process_control": {"schema_version": 1, "protocol": "configured-allocation-native-process-control-version-1",
                "hardware_class": "jaguar03_rtx_a4500", "standing_gpu_descriptor_held": False,
                "physical_gres_mapping_verified": False, "kernel_device_enforcement_verified": False, "configured_isolation_trust": True,
                "helper_hashes": {name: hashes[self.module.PREFIX + name] for name in self.module.HELPERS
                    if name != "diagnose_configured_native_process.py"}}}
        for name,value in (("measurement_plan.json",self.plan),("manifest.yaml",self.plan),
                ("config_resolved.yaml",{**self.module.ZERO,"budget":dict(self.module.BUDGET)})):
            self.write(self.unit / name,value)
        import shlex
        producer = [self.module.PYTHON,"-I","-S","-B",str(self.unit / "code/diagnose_configured_native_process.py"),"--prepared-unit",str(self.unit)]
        (self.unit / "command.txt").write_text(shlex.join(producer) + "\\n")
        (self.unit / "experiment_background.md").write_text("Synthetic zero-environment launcher fixture only.\\n")
        self.write(self.unit / "measurement_prepared.json", {"plan_sha256":self.digest(self.unit / "measurement_plan.json")})
        for name in ("manifest.yaml","config_resolved.yaml","command.txt","experiment_background.md"):
            (self.packet / name).write_bytes((self.unit / name).read_bytes())
        self.write(self.packet / "progress.json", {"kind":self.module.KIND,"state":"prepared","scientific_credit":0})
        (self.packet / "job.sbatch").write_bytes(self.module.expected_job_script())
        command=["sbatch","--hold","--parsable","--no-requeue","--partition=gpu","--account=sl5nw_base",
            "--qos=csresnolim","--reservation=sl5nw_168","--nodelist=jaguar03","--nodes=1","--ntasks=1","--cpus-per-task=1","--mem=2G",
            "--gres=gpu:nvidia_rtx_a4500:1","--time=00:05:00",str(self.packet / "job.sbatch")]
        self.write(self.packet / "held_submission_command.json",command)
        checks=[*(self.unit / "code" / name for name in self.module.HELPERS),self.unit / "measurement_plan.json",self.unit / "measurement_prepared.json"]
        (self.packet / "source_checksums.sha256").write_text("".join(self.digest(path) + "  " + str(path) + "\\n" for path in checks))
        self.refreeze_initial()
        for name in ("submitted_jobids.txt","submitted_jobids_uva.txt","submissions.tsv"):(self.run / "slurm" / name).write_text("")
        (self.run / "queue/metadata.lock").write_text("")
        self.module.prepare_binding_once()
        self.write(self.packet / "packet_static_verification.json", {"passed":True,"synthetic_fixture":True,
            "prepared_binding":self.module.reference(self.packet / "prepared_launch_binding.json")})
        frozen = [*self.module.expected_initial_paths(),*(self.packet / name for name in (
            "immutable_artifacts.json","prepared_launch_binding.json","packet_static_verification.json","launch_diagnostic_once.py","test_launch_diagnostic_once.py"))]
        self.write(self.packet / "launcher_source_freeze.json", {"files":{str(path):self.digest(path) for path in frozen}})
        self.write(self.packet / "launcher_independent_review.json", {"kind":"independent_configured_native_diagnostic_prepared_launcher_review",
            "passed":True,"synthetic_fixture":True,"source_freeze":self.module.reference(self.packet / "launcher_source_freeze.json"),
            "prepared_binding":self.module.reference(self.packet / "prepared_launch_binding.json")})
        self.publication=self.commit();self.job="910001";self.calls=[];self.fail_release=False

    def refreeze_initial(self):
        """Freeze private small canonical artifacts; no actual study data is copied."""
        self.write(self.packet / "immutable_artifacts.json", {"files": {str(path):self.file_record(path)
            for path in self.module.expected_initial_paths()}})''')
s=s.replace('"owned_nsight_process_diagnostic_submission"', '"owned_configured_native_process_diagnostic_submission"')
s=s.replace('"source_freeze": self.module.reference(self.packet / "launcher_source_freeze.json")})', '"kind":"independent_configured_native_diagnostic_prepared_launcher_review",\n            "prepared_binding":self.module.reference(self.packet / "prepared_launch_binding.json"),\n            "source_freeze": self.module.reference(self.packet / "launcher_source_freeze.json")})')
s=s.replace('        for index, replacement in enumerate(("UserId=other(1)"', '        self.make_held()\n        for index, replacement in enumerate(("UserId=other(1)"')
s=s.replace('self.module.query_held(self.job, "rejected_private_query_" + str(index))','self.module.query_held(self.job, "rejected_private_query_" + str(index), self.plan)')
replacefn('test_release_receipt_precedes_atomic_sidecar_and_matches_actual_runtime_reader','''    def test_release_receipt_precedes_atomic_sidecar_and_matches_actual_runtime_reader(self):
        """The final durable pair-only receipts satisfy the actual new canonical metadata reader."""
        self.make_held(); self.prepare_fixed_release(); events = []; original = self.module.publish
        def publish(path,value):
            events.append(Path(path).name)
            if Path(path).name == "owned_launch_inputs.json":
                self.assertTrue((self.packet / "release_receipt.json").is_file())
            return original(path,value)
        with mock.patch.object(self.module,"scheduler",side_effect=self.fake_scheduler),mock.patch.object(self.module,"publish",side_effect=publish):
            self.module.release_once(self.publication)
        self.assertLess(events.index("release_receipt.json"),events.index("owned_launch_inputs.json"))
        owned=self.module.document(self.unit / "owned_launch_inputs.json")
        for name in ("plan","submission","release"):self.assertEqual(set(owned[name]),{"path","sha256"})
        source=ast.parse((ACTUAL_PACKET / "schema/diagnose_configured_native_process_approved.py").read_text())
        node=next(item for item in source.body if isinstance(item,ast.FunctionDef) and item.name=="owned_launch")
        identity=SimpleNamespace(read_json=self.module.document,receipt=self.module.reference,require=self.module.require)
        def reference_value(ref,metadata):
            self.assertEqual(ref,self.module.reference(ref["path"]))
            return self.module.document(ref["path"])
        namespace={"Path":Path,"identity":identity,"native":SimpleNamespace(DIAGNOSTIC=self.module.KIND,reference_value=reference_value),
            "os":SimpleNamespace(environ={"SLURM_JOB_ID":self.job,"SLURM_CPUS_PER_TASK":"1","SLURM_MEM_PER_NODE":"2048"})}
        exec(compile(ast.Module(body=[node],type_ignores=[]),"actual_new_owned_launch_metadata_only","exec"),namespace)
        self.assertEqual(namespace["owned_launch"](self.unit,self.plan,SimpleNamespace()),owned)
        with mock.patch.object(self.module,"scheduler",side_effect=self.fake_scheduler):
            with self.assertRaisesRegex(ValueError,"preserve the previous attempt"):self.module.release_once(self.publication)
        self.assertEqual(sum(command[:2]==["scontrol","release"] for command in self.calls),1)''')
extra='''    def test_late_binding_has_only_actual_source_and_plan_and_cannot_be_reused(self):
        binding=self.module.document(self.packet / "prepared_launch_binding.json")
        self.assertEqual(binding["plan"],self.module.reference(self.unit / "measurement_plan.json"))
        self.assertEqual(binding["source_commit"],self.plan["helpers"]["source_commit"])
        self.assertIsNone(binding["actual_job_id"])
        self.assertFalse(binding["actual_release_performed"])
        self.assertEqual(len(binding["canonical_inputs"]),21)
        with mock.patch.object(self.module,"scheduler",side_effect=AssertionError("no native action")):
            with self.assertRaisesRegex(ValueError,"preserve the previous attempt"):self.module.prepare_binding_once()

    def test_missing_binding_and_changed_binding_refuse_without_scheduler(self):
        path=self.packet / "prepared_launch_binding.json";saved=path.read_bytes();path.unlink()
        with mock.patch.object(self.module,"scheduler",side_effect=AssertionError("no native action")):
            with self.assertRaises(FileNotFoundError):self.module.submit_held_once(self.publication)
            path.write_bytes(saved);value=self.module.document(path);value["source_commit"]="f" * 40;self.write(path,value)
            with self.assertRaisesRegex(ValueError,"prepared binding changed"):self.module.submit_held_once(self.publication)
        self.assertFalse((self.packet / "submission_intent.json").exists())

    def test_typed_budget_native_and_wrapper_substitution_refuse(self):
        paths=[self.unit / "measurement_plan.json",self.unit / "manifest.yaml"]
        original=[path.read_bytes() for path in paths]
        for field,value in (("maximum_active_seconds",True),("gpu_uuid","synthetic"),("native_process_control",{})):
            for path in paths:
                plan=dict(self.plan);plan[field]=value;self.write(path,plan)
            self.write(self.unit / "measurement_prepared.json",{"plan_sha256":self.digest(paths[0])})
            (self.packet / "manifest.yaml").write_bytes(paths[1].read_bytes());self.refreeze_initial()
            with self.assertRaises(ValueError):self.module.prepared_records()
            for path,raw in zip(paths,original):path.write_bytes(raw)
        (self.packet / "job.sbatch").write_text("#!/bin/sh\\nexit 0\\n");self.refreeze_initial()
        with self.assertRaisesRegex(ValueError,"Slurm wrapper"):self.module.prepared_records()

    def test_owned_membership_checked_before_query_even_with_decimal_foreign_id(self):
        with mock.patch.object(self.module,"scheduler",side_effect=AssertionError("no foreign scheduler query")):
            with self.assertRaises(FileNotFoundError):self.module.query_held(self.job,"unknown",self.plan)
        self.make_held()
        with mock.patch.object(self.module,"scheduler",side_effect=AssertionError("no foreign scheduler query")):
            with self.assertRaisesRegex(ValueError,"not this exact owned submission"):
                self.module.query_held("910002","foreign",self.plan)

    def test_duplicate_scheduler_fields_and_tres_refuse(self):
        self.make_held()
        for addition in (" JobId=" + self.job," ReqTRES=cpu=1,mem=2G,node=1,gres/gpu=1"):
            with mock.patch.object(self.module,"scheduler",return_value=SimpleNamespace(stdout=(self.held_text().strip()+addition).encode(),returncode=0)):
                with self.assertRaisesRegex(ValueError,"duplicate"):self.module.query_held(self.job,"duplicates",self.plan)
        raw=self.held_text().replace("ReqTRES=cpu=1,mem=2G,node=1,gres/gpu=1","ReqTRES=cpu=1,cpu=1,mem=2G,node=1,gres/gpu=1")
        with mock.patch.object(self.module,"scheduler",return_value=SimpleNamespace(stdout=raw.encode(),returncode=0)):
            with self.assertRaisesRegex(ValueError,"duplicate"):self.module.query_held(self.job,"duplicated_tres",self.plan)

    def test_all_three_ledger_fsyncs_precede_submission_receipt(self):
        events=[];append=self.module.append_member;publish=self.module.publish
        def track_append(path,member):
            result=append(path,member);events.append("fsynced:"+Path(path).name);return result
        def track_publish(path,value):
            if Path(path).name=="submission_receipt.json":
                self.assertEqual(events,["fsynced:submitted_jobids.txt","fsynced:submitted_jobids_uva.txt","fsynced:submissions.tsv"])
                self.assertTrue((self.packet / "submission_ledgers_fsynced.json").is_file())
            return publish(path,value)
        with mock.patch.object(self.module,"append_member",side_effect=track_append),mock.patch.object(self.module,"publish",side_effect=track_publish):self.make_held()

'''
s=s.replace('\n\nif __name__ == "__main__":', '\n'+extra+'\nif __name__ == "__main__":')
(p/'test_launch_diagnostic_once.py').write_text(s)
