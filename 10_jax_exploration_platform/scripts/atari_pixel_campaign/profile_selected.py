"""Profile an explicit accepted pixel baseline without choosing ideas or submitting jobs.

Preparation formats metadata only. Execution uses the staged helpers, a pinned
core, and one recorded Nsight child while its parent holds the card lock. An
original-default diagnostic is never converted into selected-baseline evidence.
"""

import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import importlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import time
import traceback


SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))
import measure_selected as measurement
import diagnose_gpu_execution as diagnostic
import post_comparison_validation as validation
import train_optimized as training

HELPERS = ("profile_selected.py", *measurement.HELPERS)
KIND = "selected_configuration_profile"
OWNED_PROCESS_PROTOCOL = "nsight-owned-descendant-observation-version-1"


def process_birth(pid):
    """Read the birth counter used to distinguish this process from a reused PID."""
    return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[19]


def process_parent(pid):
    """Read the actual parent PID before releasing the recorded numerical child."""
    return int(Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[1])


def profile_updates(configuration, minimum=1):
    """Choose a full execution group and reporting window without partial PPO updates."""
    if type(minimum) is not int or minimum < 1:
        raise ValueError("the minimum profile update count must be a positive integer")
    execution = configuration["execution"]
    multiple = execution["iterations_per_call"]
    if execution["report_readout"] != "none":
        multiple = math.lcm(multiple, execution["report_every_iterations"])
    # Before: group five, minimum one; after: five complete original updates.
    return ((minimum + multiple - 1) // multiple) * multiple


def tool_record(path):
    """Read the actual profiling executable's location, version and bytes without CUDA."""
    requested = str(path)
    located = shutil.which(requested)
    if located is None:
        raise ValueError("the requested Nsight executable is unavailable: " + requested)
    actual = Path(located).resolve(strict=True)
    version = subprocess.check_output([str(actual), "--version"], text=True).strip()
    if not version:
        raise ValueError("Nsight did not report its version")
    return {"path": str(actual), "version": version, "sha256": training.file_sha256(actual)}


def helper_receipt(configured=False, direct=False, common=False, owned_processes=False, *, direct_profile=False, configured_native=False):
    """Bind every staged helper to its exact committed bytes."""
    root = SCRIPTS.parents[2]
    commit = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    return training.source_receipt(root, commit, [str((SCRIPTS / name).relative_to(root)) for name in required_helpers(
        configured, direct, common, owned_processes, direct_profile=direct_profile, configured_native=configured_native)])


def required_helpers(configured=False, direct=False, common=False, owned_processes=False, *, direct_profile=False, configured_native=False):
    """Include the exact optional identity closure while preserving the default profile roster."""
    if direct and (configured or common or owned_processes):
        raise ValueError("direct profile prerequisites use their own historical consumer, not configured/common helper admission")
    if direct_profile and not direct:
        raise ValueError("the new direct profile closure requires the separate direct target")
    if owned_processes and (not configured or direct):
        raise ValueError("owned Nsight observation is supported only by its configured A4500 profile")
    if configured_native:
        if not configured or direct or not owned_processes or direct_profile:
            raise ValueError("native allocation helpers require only the explicit configured owned-process profile")
        native = configured_native_reader(selected=True)
        return tuple(dict.fromkeys((*required_helpers(configured, direct, common, owned_processes), *native.HELPERS)))
    if direct_profile:
        return tuple(sorted(Path(relative).name for relative in direct_profile_reader().helper_roster()))
    optional = ("direct_profile_prerequisites.py", "nsight_process_identity.py", "direct_profile_process.py",
        "direct_capture_release.py", "direct_tool_descendants.py") if direct_profile else (("nsight_process_identity.py",) if owned_processes else ())
    return tuple(dict.fromkeys(("profile_selected.py", *measurement.required_helpers(configured, direct, common, direct_profile), *optional)))


def direct_profile_reader(plan=None):
    """Load the separate exact direct profile adapter, never a configured admission reader."""
    import direct_measurement as support
    reader = support.adjacent("direct_profile_prerequisites")
    if plan is not None:
        reader.contract(plan)
    return reader


def direct_process_reader(plan):
    """Load only the explicit new direct producer/child source protocol."""
    direct_profile_reader(plan)
    import direct_measurement as support
    reader = support.adjacent("direct_profile_process")
    reader.contract(plan, Path(plan["campaign_run_dir"]) / "data" / plan["unit_id"])
    return reader


def owned_processes(plan):
    """Select only the explicit new A4500 process protocol, preserving other profiles."""
    value = plan.get("capture_process_protocol")
    if "direct_binding" in plan:
        if "direct_profile_protocol" in plan:
            direct_profile_reader(plan)
            return False
        if value is not None or "10_jax_exploration_platform/scripts/atari_pixel_campaign/nsight_process_identity.py" in plan.get("helpers", {}).get("files", {}):
            raise ValueError("a historical direct profile cannot borrow the configured process observer")
        return False
    declared = "10_jax_exploration_platform/scripts/atari_pixel_campaign/nsight_process_identity.py" in plan.get("helpers", {}).get("files", {})
    if value is None:
        if declared:
            raise ValueError("the staged process observer lacks its explicit profile protocol")
        return False
    if (value != OWNED_PROCESS_PROTOCOL or "configured_isolation" not in plan or "direct_binding" in plan
            or plan.get("hardware_class") != "jaguar03_rtx_a4500" or not declared):
        raise ValueError("unsupported Nsight process observation protocol")
    return True


def require_reviewed_profile_process_admission():
    """Keep new configured profiles disabled until the real diagnostic receives an independent review."""
    raise ValueError("new configured profiles remain disabled pending the real Nsight process diagnostic and independent review")


def process_reader(plan):
    """Load only the exact staged pure-metadata process helper recorded in this plan."""
    if not owned_processes(plan):
        raise ValueError("the profile has no owned-process observation protocol")
    path = SCRIPTS / "nsight_process_identity.py"
    relative = "10_jax_exploration_platform/scripts/atari_pixel_campaign/" + path.name
    if (path.resolve(strict=True) != path or path.is_symlink()
            or training.file_sha256(path) != plan["helpers"]["files"].get(relative)):
        raise ValueError("the staged Nsight process observation helper changed")
    specification = importlib.util.spec_from_file_location("selected_profile_owned_process_reader", path)
    module = importlib.util.module_from_spec(specification); specification.loader.exec_module(module)
    return module


_CONFIGURED_NATIVE_READERS = {}


def configured_native_reader(plan=None, *, selected=False):
    """Load only an explicitly declared A4500 metadata policy; never import its native provider here."""
    relative = "10_jax_exploration_platform/scripts/atari_pixel_campaign/configured_native_process.py"
    if plan is not None:
        if "direct_binding" in plan:
            if "configured_native_process_control" in plan:
                raise ValueError("configured native control cannot replace direct H100 process admission")
            return None
        selected = any(key in plan for key in ("native_process_control", "configured_native_process_control"))
        declared = relative in plan.get("helpers", {}).get("files", {})
        if not selected:
            if declared:raise ValueError("configured native helpers require their explicit source-bound declaration")
            return None
    if not selected:return None
    path = SCRIPTS / "configured_native_process.py"
    if path.resolve(strict=True) != path or path.is_symlink() or not path.is_file():
        raise ValueError("the explicit configured native policy helper is not its ordinary staged source")
    expected = training.file_sha256(path)
    if plan is not None and plan["helpers"]["files"].get(relative) != expected:
        raise ValueError("the actual configured native policy source changed")
    key = (str(path), expected)
    if key not in _CONFIGURED_NATIVE_READERS:
        spec = importlib.util.spec_from_file_location("selected_configured_native_policy", path)
        module = importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        _CONFIGURED_NATIVE_READERS[key] = module
    module = _CONFIGURED_NATIVE_READERS[key]
    if plan is not None:module.optional_policy(plan)
    return module


def accepted_confirmation_common_reader(plan):
    """Recheck the exact accepted record and its profile-only optional reader declaration."""
    if "direct_binding" in plan:
        direct_profile_reader(plan).validate_metadata(plan)
        return False
    record = plan["inputs"]["accepted_confirmation"]
    confirmation, actual = measurement.read_document(Path(record["path"]))
    if actual != record:
        raise ValueError("the accepted confirmation changed after profile preparation")
    common = "common_correctness_inputs" in confirmation
    declared = plan.get("accepted_confirmation_requires_common_reader", False)
    if (("accepted_confirmation_requires_common_reader" in plan and declared is not True)
            or declared != common):
        raise ValueError("the accepted confirmation and profile helper declaration differ")
    required_helpers("configured_isolation" in plan, "direct_binding" in plan, common)
    relative = "10_jax_exploration_platform/scripts/atari_pixel_campaign/common_confirmation.py"
    if (relative in plan["helpers"]["files"]) != common:
        raise ValueError("the profile omitted or added the accepted confirmation's optional reader")
    return common


@contextmanager
def interruption_cleanup():
    """Turn process termination into an exception so the owned child stops before its lock closes."""
    def interrupted(number, frame):
        """Leave the active capture through its existing owned-child cleanup path."""
        raise InterruptedError("selected-baseline profiling received signal " + str(number))
    previous = {number: signal.getsignal(number) for number in (signal.SIGTERM, signal.SIGINT)}
    for number in previous:
        signal.signal(number, interrupted)
    try:
        yield
    finally:
        for number, handler in previous.items():
            signal.signal(number, handler)


def initial_summaries_complete(registry):
    """Check initial-cohort metadata now; execution later revalidates every original case."""
    entries, counts = registry.get("initial_ideas", []), registry.get("initial_matched_copy_counts", [])
    if len(entries) != 30 or not counts or any(type(count) is not int or count < 1 for count in counts):
        raise ValueError("selected-baseline profiling requires the thirty initial mechanisms at recorded counts")
    if len({entry["name"] for entry in entries}) != 30:
        raise ValueError("the initial mechanisms do not have thirty distinct identities")
    for entry in entries:
        for context in validation.CONTEXTS:
            for copies in counts:
                rows = [row for row in entry.get("results", {}).get(context, []) if row.get("copies") == copies]
                if not any(row.get("implementation_distinct") and row.get("result_path") and (
                        row.get("measurement_valid") and row.get("correctness_passed") or
                        row.get("correctness_attempted") and row.get("failure_stage") == "correctness"
                        and row.get("decision") == "rejected") for row in rows):
                    raise ValueError("an initial mechanism still lacks its matched hardware/copy result")


def prepare(args):
    """Stage immutable profile inputs without loading an array library or touching a GPU."""
    configured = getattr(args, "configured_isolation_profile", None) is not None
    direct = getattr(args, "direct_binding_profile", None) is not None
    if configured:
        require_reviewed_profile_process_admission()
    if direct:
        direct_profile_reader().require_admission()
    native_qualification = getattr(args, "configured_native_process_qualification", None)
    native_review = getattr(args, "configured_native_process_review", None)
    native_selected = native_qualification is not None or native_review is not None
    if native_selected and (not configured or direct or native_qualification is None or native_review is None
            or args.maximum_active_seconds != 240):
        raise ValueError("native A4500 preparation requires both actual qualification/review records and the explicit240second active bound")
    if direct and (configured or args.hardware_class != "h100" or args.gpu_uuid != measurement.CARD1):
        raise ValueError("direct profiling requires only its separately approved card-one profile")
    if configured and (args.gpu_uuid is not None or args.hardware_class != "jaguar03_rtx_a4500"):
        raise ValueError("configured profiling needs an unresolved single A4500 plan")
    if (Path(args.unit_id).name != args.unit_id or args.unit_id in {"", ".", ".."}
            or not configured and (not isinstance(args.gpu_uuid, str) or not args.gpu_uuid.startswith("GPU-")) or args.hardware_class not in validation.CONTEXTS
            or type(args.seed_start) is not int or args.seed_start < 0
            or not math.isfinite(args.maximum_active_seconds) or args.maximum_active_seconds <= 0):
        raise ValueError("profile identity, selected GPU, seed, or active time limit is invalid")
    root, run = args.source_root.resolve(strict=True), args.run_dir.resolve(strict=True)
    core = diagnostic.core_identity(root, args.source_commit)
    paired_source = training.source_receipt(root, args.source_commit, [
        "10_jax_exploration_platform/scripts/atari_pixel_campaign/" + name for name in ("paired_benchmark.py", "ideas.py")])
    configuration, config_input = measurement.read_document(args.configuration)
    configuration = validation.configuration(configuration)
    measurement.validate_complete_fields(root, configuration)
    runtime, runtime_input = measurement.read_document(args.runtime_environment)
    runtime = validation.runtime_environment(runtime)
    assets, asset_input = measurement.read_document(args.asset_record)
    if set(assets) != {"root", "manifest", "sha256"}:
        raise ValueError("profile assets must use the immutable asset-record schema")
    registry, registry_input = measurement.read_document(args.study_registry)
    initial_summaries_complete(registry)
    baseline, baseline_input = measurement.read_document(args.accepted_baseline_specification)
    common = {"source_commit": args.source_commit, "hardware_class": args.hardware_class, "runtime_environment": runtime}
    if any(name in baseline and baseline[name] != value for name, value in common.items()):
        raise ValueError("accepted-baseline source/runtime/context differs from this profile")
    baseline = {**baseline, **common}
    if direct:
        # The actual scientific result is opened only by the owned-compute child.
        review, unused_review_input = measurement.read_document(args.direct_confirmation_review)
        reader = direct_profile_reader()
        unit = reader.confirmation_unit({"campaign_run_dir": str(run)}, review["result"])
        reader.reserved_metadata(review["plan"], unit, "measurement_plan.json")
        confirmation_input = review["result"]
        confirmation_plan, actual_plan_input = measurement.read_document(review["plan"]["path"])
        if actual_plan_input != review["plan"] or confirmation_input["path"] != baseline["confirmation_record"]:
            raise ValueError("the accepted common review changed its frozen plan or baseline path")
        confirmation = {"candidate": confirmation_plan["configuration"], "measurement_mode": "confirmation",
            "decision": review.get("decision"), "direct_binding": confirmation_plan["direct_binding"]}
    else:
        confirmation, confirmation_input = measurement.read_document(baseline["confirmation_record"])
    if (confirmation.get("measurement_mode") != "confirmation" or confirmation.get("decision") != "accepted"
            or validation.configuration_without_copies(confirmation["candidate"]) != validation.configuration_without_copies(configuration)):
        raise ValueError("the profile configuration is not its explicit accepted combined baseline")
    if "configured_isolation" in confirmation and not configured:
        raise ValueError("default profiling cannot consume a configured-isolation measurement")
    if "direct_binding" in confirmation and not direct:
        raise ValueError("default profiling cannot consume an explicit direct-binding measurement")
    common_reader = "common_correctness_inputs" in confirmation
    if common_reader and (not configured or direct):
        raise ValueError("a common-confirmation reader requires only configured A4500 profiling")
    if direct:
        import direct_measurement as support
        profile, unused_profile_input = measurement.read_document(args.direct_binding_profile)
        support.core().require_profile(profile)
        if baseline.get("direct_binding_profile") != profile:
            raise ValueError("the accepted baseline must name the exact separate direct profile")
    elif configured:
        import configured_measurement as support
        explicit_profile, unused_profile_input = measurement.read_document(args.configured_isolation_profile)
        support.core().require_profile(explicit_profile)
        if baseline.get("configured_isolation_profile") != explicit_profile:
            raise ValueError("the accepted baseline specification must explicitly name this configured-isolation profile")
        support.verify_saved_measurement(Path(confirmation_input["path"]).parent, confirmation)
        validation._identity_fields(confirmation, args.hardware_class, args.source_commit, runtime)
    else:
        validation.identity(confirmation, args.hardware_class, args.source_commit, runtime)
    tool, tool_input = measurement.read_document(args.profile_tool_record)
    if (set(tool) != {"path", "version", "sha256"} or not Path(tool["path"]).is_absolute()
            or not isinstance(tool["version"], str) or not tool["version"].strip()
            or len(tool["sha256"]) != 64 or any(character not in "0123456789abcdef" for character in tool["sha256"])):
        raise ValueError("record the target host's actual profiling tool before preparation")
    copies = configuration["learner"]["n_copies"]
    if args.seed_start + copies > 2 ** 32:
        raise ValueError("the profile policy seeds exceed unsigned 32-bit values")
    directory = run / "data" / args.unit_id
    plan = {"schema_version": 1, "kind": KIND, "source_root": str(root), "source_commit": args.source_commit,
        "core": core, "paired_source": paired_source, "helpers": helper_receipt(
            configured, direct, common_reader, configured, direct_profile=direct, configured_native=native_selected),
        "configuration": configuration, "runtime_environment": runtime, "asset_record": assets,
        "hardware_class": args.hardware_class, "gpu_uuid": args.gpu_uuid,
        "campaign_run_dir": str(run), "unit_id": args.unit_id, "compilation_cache_policy": "disabled",
        "copy_seeds": list(range(args.seed_start, args.seed_start + copies)),
        "profile_updates": profile_updates(configuration, args.minimum_updates),
        "minimum_profile_updates": args.minimum_updates, "profile_tool": tool,
        "maximum_active_seconds": args.maximum_active_seconds, "python": str(args.python.absolute()),
        "jaxtari_source_root": str(args.jaxtari_source_root.resolve()) if args.jaxtari_source_root else None,
        "accepted_baseline_specification": baseline, "initial_registry_record": str(directory / "inputs/study_registry.json"),
        "preparation_scope": "source and input metadata only; no measured profile evidence yet",
        "hypothesis_count_contribution": 0, "jobs_submitted_by_producer": False,
        "inputs": {"configuration": config_input, "runtime_environment": runtime_input, "asset_record": asset_input,
                   "study_registry": registry_input, "accepted_baseline_specification": baseline_input,
                   "accepted_confirmation": confirmation_input, "profile_tool_record": tool_input}}
    if common_reader:
        plan["accepted_confirmation_requires_common_reader"] = True
    if configured:
        plan["capture_process_protocol"] = OWNED_PROCESS_PROTOCOL
        import configured_measurement as support
        profile, profile_input = measurement.read_document(args.configured_isolation_profile)
        support.configure_plan(plan, profile, "component_profile")
        plan["inputs"]["configured_isolation_profile"] = profile_input
        support.prepare_qualification_inputs(plan, args.target_qualification_record, args.target_qualification_review)
        if native_selected:
            native = configured_native_reader(selected=True)
            plan["maximum_active_seconds"] = 240
            plan["diagnostic_time_budget"] = dict(native.BUDGET)
            native.configure_plan(plan, native_qualification, native_review, process_reader(plan))
    if direct:
        import direct_measurement as support
        profile, profile_input = measurement.read_document(args.direct_binding_profile)
        reader = direct_profile_reader()
        if "accepted_evidence_snapshot_protocol" in confirmation_plan:
            # The new immutable own-unit pointer and original admission copy
            # must exist before configure_binding() takes their exact reference.
            reader.prepare_history_pointer(args.direct_history_inputs, plan, confirmation_plan)
        reader.configure_binding(plan, profile, args.direct_history_inputs, args.direct_backend_image_record)
        plan["inputs"]["direct_binding_profile"] = profile_input
        plan["inputs"]["direct_history_inputs"] = plan["direct_binding"]["frozen_inputs"]
        reader.configure(plan, baseline, args.direct_confirmation_review,
            args.direct_qualification_record, args.direct_qualification_review)
    accepted_confirmation_common_reader(plan)
    artifacts = {directory / "measurement_plan.json": measurement.json_bytes(plan),
        directory / "manifest.yaml": measurement.json_bytes(plan),
        directory / "config_resolved.yaml": measurement.json_bytes(configuration),
        directory / "inputs/study_registry.json": args.study_registry.read_bytes(),
        directory / "command.txt": (shlex.join(configured_native_reader(plan).producer_command(directory, plan)
            if native_selected else [str(args.python), str(directory / "code/profile_selected.py"),
                "--prepared-unit", str(directory)]) + "\n").encode()}
    for name in required_helpers(configured, direct, common_reader, configured, direct_profile=direct, configured_native=native_selected):
        artifacts[directory / "code" / name] = (SCRIPTS / name).read_bytes()
    artifacts[directory / "experiment_background.md"] = (
        "## Purpose\n\nProfile the explicitly supplied accepted baseline after all thirty initial mechanisms. "
        "This diagnostic chooses no ideas and supplies no optimization-acceptance result.\n\n"
        "## Key hyperparameters\n\nThe full original learner keeps 128 environments per policy, 128 rollout steps, "
        "four epochs, four minibatches and fifty observation-normalization warmup rollouts. "
        f"The capture contains {plan['profile_updates']} complete updates, covering its execution group/report window.\n\n"
        "## Code and config changes\n\nOnly the explicit selected performance settings differ from the original baseline. "
        "Source, assets, runtime, physical GPU and actual library images are verified before the capture. "
        "Compilation is stabilized outside the capture. Raw Nsight data and positive kernel/API summaries are retained.\n\n"
        f"## Git state\n\nNumerical source: `{args.source_commit}`. Helper source: `{plan['helpers']['source_commit']}`. "
        "Exact file hashes are preserved in measurement_plan.json.\n").encode()
    for path, contents in artifacts.items():
        if path.exists() and path.read_bytes() != contents:
            raise ValueError("an existing immutable profile artifact differs: " + str(path))
    for path, contents in artifacts.items():
        measurement.write_once(path, contents)
    measurement.write_once(directory / "measurement_prepared.json", measurement.json_bytes({
        "plan_sha256": hashlib.sha256(measurement.json_bytes(plan)).hexdigest()}))
    return directory


def verify_prepared(directory, plan):
    """Require staged helper bytes, pinned numerical files, and their original output location."""
    if "direct_binding" in plan:
        direct_profile_reader(plan).require_admission()
    if directory != Path(plan["campaign_run_dir"]) / "data" / plan["unit_id"]:
        raise ValueError("profile output differs from its immutable prepared location")
    prepared = json.loads((directory / "measurement_prepared.json").read_text())
    if prepared != {"plan_sha256": training.file_sha256(directory / "measurement_plan.json")}:
        raise ValueError("the immutable profile plan changed")
    if Path(__file__).resolve() != directory / "code/profile_selected.py" or str(Path(sys.executable).absolute()) != plan["python"]:
        raise ValueError("execute the staged profile entrypoint with its exact recorded Python")
    common_reader = accepted_confirmation_common_reader(plan)
    if owned_processes(plan):
        require_reviewed_profile_process_admission()
    for name in required_helpers("configured_isolation" in plan, "direct_binding" in plan, common_reader,
            owned_processes(plan), direct_profile="direct_profile_protocol" in plan,
            configured_native=configured_native_reader(plan) is not None):
        expected = plan["helpers"]["files"]["10_jax_exploration_platform/scripts/atari_pixel_campaign/" + name]
        if training.file_sha256(directory / "code" / name) != expected:
            raise ValueError("a staged profile helper changed: " + name)
    for name, module in (("measure_selected.py", measurement), ("diagnose_gpu_execution.py", diagnostic),
                         ("post_comparison_validation.py", validation), ("train_optimized.py", training)):
        if Path(module.__file__).resolve() != directory / "code" / name:
            raise ValueError("the profile imported a mutable helper: " + name)
    measurement.producer_support(plan)
    if diagnostic.core_identity(Path(plan["source_root"]), plan["source_commit"]) != plan["core"]:
        raise ValueError("the profile core changed after preparation")
    if training.source_receipt(Path(plan["source_root"]), plan["source_commit"],
                               list(plan["paired_source"]["files"])) != plan["paired_source"]:
        raise ValueError("the frozen profile Executor or idea definitions changed")


def validate_measured_baseline(directory, plan):
    """Revalidate original initial cases and the accepted combination on compute, not during preparation."""
    if "direct_binding" in plan:
        return direct_profile_reader(plan).before_compute(directory, plan)
    registry_path = Path(plan["initial_registry_record"])
    if (registry_path != directory / "inputs/study_registry.json"
            or training.file_sha256(registry_path) != plan["inputs"]["study_registry"]["sha256"]):
        raise ValueError("the staged initial study registry differs from its prepared bytes")
    registry = json.loads(registry_path.read_text())
    initial_summaries_complete(registry)
    evidence = validation.Evidence()
    confirmation = plan["inputs"]["accepted_confirmation"]
    if hashlib.sha256(evidence.file(confirmation["path"])).hexdigest() != confirmation["sha256"]:
        raise ValueError("the accepted confirmation changed after profile preparation")
    accepted_confirmation_common_reader(plan)
    proofs = []
    for entry in registry["initial_ideas"]:
        for context in validation.CONTEXTS:
            for copies in registry["initial_matched_copy_counts"]:
                rows = [row for row in entry["results"][context] if row.get("copies") == copies]
                if not rows:
                    raise ValueError("an initial matched result is missing")
                row = rows[-1]
                proofs.append(validation.validate_cohort_case(row["result_path"], entry["name"], context, copies,
                                                              registry["reference_commit"], evidence))
    baseline = validation.validate_combination(plan["accepted_baseline_specification"], evidence)
    if validation.configuration_without_copies(baseline["configuration"]) != validation.configuration_without_copies(plan["configuration"]):
        raise ValueError("the measured accepted baseline differs from the profile configuration")
    diagnostic.atomic_json(directory / "baseline_validation.json", {
        "initial_mechanisms": 30, "initial_matched_copy_counts": registry["initial_matched_copy_counts"],
        "accepted_combination": baseline, "cohort_runtime_provenance": proofs, "evidence": evidence.records()})


def child_command(directory, plan, held_descriptor=None):
    """Record a CUDA-profiler-range capture that cannot silently run without Nsight."""
    if "direct_profile_protocol" in plan:
        if held_descriptor is not None:
            raise ValueError("the actual direct Nsight child must not inherit the producer's standing descriptor")
        return direct_process_reader(plan).tool_command(directory, plan)
    if owned_processes(plan):
        if held_descriptor is not None:
            raise ValueError("the configured process protocol cannot borrow a direct-card descriptor")
        return process_reader(plan).nsight_command(directory, plan)
    command = [plan["profile_tool"]["path"], "profile", "--trace=cuda,cudnn", "--sample=none", "--cpuctxsw=none",
            "--cuda-graph-trace=node", "--capture-range=cudaProfilerApi", "--capture-range-end=stop",
            "--stop-on-exit=true", "--force-overwrite=false", "--output=" + str(directory / "trace/capture"),
            plan["python"], str(directory / "code/profile_selected.py"), "--capture-unit", str(directory)]
    if "direct_binding" in plan:
        if type(held_descriptor) is not int or held_descriptor < 0:
            raise ValueError("direct profiling requires its actual retained descriptor")
        # The metadata-only primary waits on the actual Python capture child.
        command.insert(command.index(plan["python"]), "--wait=primary")
        command[command.index("--capture-unit")] = "--direct-waiter-unit"
        command += ["--direct-held-lock-fd", str(held_descriptor)]
    elif held_descriptor is not None:
        raise ValueError("a descriptor override applies only to the explicit direct profile")
    return command


def stop_own_child(child):
    """Stop only this newly created Nsight process group after an exception or active deadline."""
    if child.poll() is None:
        os.killpg(child.pid, signal.SIGTERM)
        try:
            child.wait(timeout=120)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGKILL)
            child.wait()


def capture_and_summarize(directory, plan, held_descriptor=None):
    """Retain the card lock in the caller while recording/releasing one new profiler child."""
    if "direct_binding" in plan:
        direct_profile_reader(plan).require_admission()
        return capture_direct_owned_processes(directory, plan, held_descriptor)
    if owned_processes(plan):
        if configured_native_reader(plan) is not None:
            raise ValueError("native configured tools require their original whole-tool producer runtime")
        return capture_owned_processes(directory, plan)
    trace = directory / "trace"
    trace.mkdir()
    direct = "direct_binding" in plan
    command = child_command(directory, plan, held_descriptor) if direct else child_command(directory, plan)
    diagnostic.atomic_json(directory / "executed_command.json", command)
    started = time.monotonic()
    with (directory / "capture.log").open("x") as log:
        if direct:
            support = measurement.producer_support(plan)
            parent_metadata = support.core().read_receipt(support.core().receipt(directory / "supervisor/direct_process_metadata.json"))
            support.core().live_custody(parent_metadata, held_descriptor)
            child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True, pass_fds=(held_descriptor,))
        else:
            child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        owner = None
        try:
            # Any receipt failure after process creation must stop this child before its card lock closes.
            owner = {"pid": child.pid, "process_start_time": process_birth(child.pid), "parent_pid": os.getpid(),
                     "parent_start_time": process_birth(os.getpid()), "command": command}
            diagnostic.atomic_json(directory / "profiler_process.json", owner)
            while not (directory / "capture_identity.json").exists():
                if child.poll() is not None:
                    raise RuntimeError("Nsight ended before its capture process recorded its identity")
                if time.monotonic() - started > min(180, plan["maximum_active_seconds"]):
                    raise TimeoutError("the capture process did not record its identity")
                time.sleep(.1)
            identity = json.loads((directory / "capture_identity.json").read_text())
            if (identity["hostname"] != socket.gethostname().split(".")[0]
                    or identity["process_start_time"] != process_birth(identity["pid"])
                    or os.getpgid(identity["pid"]) != child.pid):
                raise ValueError("the recorded capture process is not this live Nsight child")
            diagnostic.atomic_json(directory / "capture_release.json", {**owner, "capture_identity": identity})
            remaining = max(.001, plan["maximum_active_seconds"] - (time.monotonic() - started))
            actual_returncode = child.wait(timeout=remaining)
            if actual_returncode != 0:
                raise RuntimeError("the selected-baseline capture exited unsuccessfully; inspect capture.log")
            if direct:
                support.core().live_custody(parent_metadata, held_descriptor)
                metadata = support.core().read_receipt(support.core().receipt(directory / "direct_process_metadata.json"))
                support.validate_capture_wait(directory, metadata)
                support.core().publish(directory / "direct_profile_terminal.json", {
                    "schema_version": 1, "kind": "direct_retained_descriptor_profile_terminal",
                    "profiler_process": owner, "capture_identity": identity, "command": command,
                    "returncode": actual_returncode, "parent_custody": support.core().receipt(directory / "supervisor/direct_process_metadata.json"),
                    "capture_release": support.core().receipt(directory / "capture_release.json"),
                    "capture_wait": support.core().receipt(directory / "direct_capture_wait_return.json"),
                    "capture_result": support.core().receipt(directory / "capture_result.json"), **support.BOUNDARY})
        except BaseException:
            if direct:
                if owner is None:
                    raw = support.core().capture_owner_process(child.pid)
                    fields = raw["stat"].rsplit(")", 1)[1].split()
                    support.core().validate_owner_process(raw, child.pid, fields[19], os.getpid())
                    owner = {"pid": child.pid, "process_start_time": fields[19], "parent_pid": os.getpid()}
                support.stop_profile_descendants(directory, child, parent_metadata, held_descriptor, owner)
            else:
                stop_own_child(child)
            raise
    if tool_record(plan["profile_tool"]["path"]) != plan["profile_tool"]:
        raise ValueError("the profiling executable changed during capture")
    captured = json.loads((directory / "capture_result.json").read_text())
    analysis = diagnostic.summarize_trace(directory, plan["profile_tool"]["path"])
    for name, column in (("cuda_gpu_kern_sum", "Instances"), ("cuda_api_sum", "Num Calls")):
        summary = analysis["summaries"][name]
        if diagnostic.summary_csv_evidence(summary["path"], column) != {key: summary[key] for key in ("rows", "columns", "sha256")}:
            raise ValueError("a positive profile summary changed before publication")
    result = {**captured, "completed": True, "trace_analysis_record": str(trace / "analysis.json"),
              "trace_report_sha256": training.file_sha256(trace / "capture.nsys-rep"), "profile_tool": plan["profile_tool"]}
    if direct:
        support.core().live_custody(parent_metadata, held_descriptor)
        metadata = support.core().read_receipt(support.core().receipt(directory / "direct_process_metadata.json"))
        result["direct_trace_process_proof"] = support.trace_process_proof(directory, metadata)
        result.update(direct_profile_reader(plan).completed_capture_binding(directory, plan, captured))
    diagnostic.atomic_json(directory / "result.json", result)
    diagnostic.record_progress(directory, "completed", completed=True, profile_updates=result["profile_updates"])


def run_prepared(directory):
    """Keep all CUDA work in one recorded child, publishing evidence only after valid summaries."""
    directory = Path(directory).resolve(strict=True)
    plan = json.loads((directory / "measurement_plan.json").read_text())
    with (directory / "profile.lock").open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        if any((directory / name).exists() for name in ("result.json", "failure.json", "profiler_process.json",
                "capture_identity.json", "capture_release.json", "capture_result.json", "trace", "hardware.json",
                "supervisor_metadata.json", "installed_backend.json", "baseline_validation.json", "asset_receipt.json",
                "profile_process_intent.json", "capture_process_observation.json", "capture_actual_launch.json",
                "capture_actual_wait.json", "profiler_actual_wait.json", "profile_process_failure.json")):
            raise ValueError("retain the previous profile attempt and use a new immutable unit")
        verify_prepared(directory, plan)
        if "jax" in sys.modules or "numpy" in sys.modules or "torch" in sys.modules:
            raise RuntimeError("the profile supervisor must not initialize an array runtime")
        runtime = training.enforce_runtime(plan["runtime_environment"], os.environ)
        for name, value in (("JAXTARI_SKIP_SPRITE_UPDATE", "1"), ("XLA_PYTHON_CLIENT_PREALLOCATE", "false")):
            if os.environ.get(name, value) != value:
                raise ValueError("the profile inherited an unsupported runtime setting: " + name)
            os.environ[name] = value
        try:
            if configured_native_reader(plan) is not None:
                return run_configured_native_prepared(directory, plan, runtime)
            with interruption_cleanup(), measurement.card_admission(directory, plan) as held:
                support = measurement.producer_support(plan)
                direct = "direct_binding" in plan
                if direct:
                    supervisor = directory / "supervisor"; supervisor.mkdir(exist_ok=False)
                    metadata_record = support.core().collect_metadata(plan, supervisor, held.fileno())
                    hardware = {"direct_process_metadata": metadata_record}
                    direct_profile_reader(plan).validate_metadata(plan)
                elif support is not None:
                    supervisor = directory / "supervisor"
                    supervisor.mkdir(exist_ok=False)
                    metadata_record = support.core().collect_metadata(plan, supervisor)
                    hardware = {"configured_isolation_metadata": metadata_record}
                    support.verify_qualification_inputs(plan)
                else:
                    hardware = measurement.preflight(plan)
                if not direct and tool_record(plan["profile_tool"]["path"]) != plan["profile_tool"]:
                    raise ValueError("the compute host's profiling tool differs from the prepared tool record")
                validate_measured_baseline(directory, plan)
                assets = training.verify_assets(plan["asset_record"], os.environ)
                installed = {"jaxtari_source": training.jaxtari_python_receipt(plan.get("jaxtari_source_root")),
                             "gpu_libraries": training.installed_gpu_library_receipts()}
                diagnostic.atomic_json(directory / "supervisor_metadata.json", {"hardware": hardware, "runtime_environment": runtime})
                diagnostic.atomic_json(directory / "asset_receipt.json", assets)
                diagnostic.atomic_json(directory / "installed_backend.json", installed)
                if direct:
                    capture_and_summarize(directory, plan, held.fileno())
                else:
                    capture_and_summarize(directory, plan)
        except BaseException as error:
            diagnostic.atomic_json(directory / "failure.json", {"error": repr(error), "traceback": traceback.format_exc(),
                                    "source_commit": plan["source_commit"], "kind": KIND})
            diagnostic.record_progress(directory, "failed", error=repr(error))
            raise


def capture(directory, direct_held_lock_fd=None):
    """Wait for the recorded parent release before importing the unchanged core/Executor."""
    directory = Path(directory).resolve(strict=True)
    plan = json.loads((directory / "measurement_plan.json").read_text())
    verify_prepared(directory, plan)
    if owned_processes(plan):
        if direct_held_lock_fd is not None:
            raise ValueError("the configured capture cannot borrow a direct-card descriptor")
        return capture_owned_child(directory, plan)
    direct = "direct_binding" in plan
    if direct:
        support = measurement.producer_support(plan); identity_reader = support.core()
        if type(direct_held_lock_fd) is not int or direct_held_lock_fd < 0:
            raise ValueError("the direct capture child lost its exact inherited descriptor")
        parent_metadata = identity_reader.read_receipt(identity_reader.receipt(directory / "supervisor/direct_process_metadata.json"))
        identity_reader.validate_process(identity_reader.current_process())
        identity_reader.validate_lock(identity_reader.lock_metadata(direct_held_lock_fd), parent_metadata["before"]["identity"]["pid"])
        support.validate_waiter_chain(plan, directory, identity_reader.current_process(), direct_held_lock_fd, live=True)
    elif direct_held_lock_fd is not None:
        raise ValueError("an inherited descriptor is supported only by the explicit direct capture child")
    identity = {"hostname": socket.gethostname().split(".")[0], "pid": os.getpid(),
                "process_start_time": process_birth(os.getpid())}
    if direct:
        identity_reader.publish(directory / "capture_identity.json", identity)
    else:
        diagnostic.atomic_json(directory / "capture_identity.json", identity)
    started = time.monotonic()
    while not (directory / "capture_release.json").exists():
        if time.monotonic() - started > 180:
            raise TimeoutError("the parent did not record and release this capture process")
        time.sleep(.1)
    release = json.loads((directory / "capture_release.json").read_text())
    owner = json.loads((directory / "profiler_process.json").read_text())
    if (release != {**owner, "capture_identity": identity}
            or process_birth(release["parent_pid"]) != release["parent_start_time"]
            or process_birth(release["pid"]) != release["process_start_time"]
            or os.getpgrp() != release["pid"] or process_parent(release["pid"]) != release["parent_pid"]):
        raise ValueError("the capture release does not belong to this live recorded parent")
    support = measurement.producer_support(plan)
    if direct:
        metadata_record = support.core().collect_metadata(plan, directory, direct_held_lock_fd,
            support.core().receipt(directory / "supervisor/direct_process_metadata.json"))
        hardware = {"direct_process_metadata": metadata_record}
    else:
        hardware = {"configured_isolation_metadata": support.core().collect_metadata(plan, directory)} if support is not None else measurement.preflight(plan)
    runtime = training.enforce_runtime(plan["runtime_environment"], os.environ)
    hardware["runtime_environment"] = runtime
    installed = json.loads((directory / "installed_backend.json").read_text())
    _capture_core(directory, plan, hardware, installed)
    if direct:
        direct_profile_reader(plan).complete_capture(directory, plan)


def _capture_core(directory, plan, hardware, installed):
    """Warm the exact selected learner, stabilize compilation, then trace one complete window."""
    if "jax" in sys.modules:
        raise RuntimeError("JAX was imported before capture admission")
    root = Path(plan["source_root"]) / "10_jax_exploration_platform"
    sys.path.insert(0, str(root / "src"))
    sys.path.insert(0, str(root / "scripts/atari_pixel_campaign"))
    import jax
    import jaxtari
    from exploration_platform.atari_pixels.config import PixelPPOConfig
    from exploration_platform.atari_pixels.environment import AtariPixelEnvironment
    from exploration_platform.atari_pixels.training import PixelPPORND
    paired = importlib.import_module("paired_benchmark")
    measurement.verify_imported_sources(root, paired)
    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_default_matmul_precision", "highest")
    devices = jax.devices()
    expected = "H100" if plan["hardware_class"] == "h100" else "A4500"
    if len(devices) != 1 or devices[0].platform != "gpu" or expected not in devices[0].device_kind:
        raise ValueError("the profile did not select its recorded single GPU type")
    support = measurement.producer_support(plan)
    direct = "direct_binding" in plan
    binding, numerical_paths = None, None
    if support is not None:
        numerical_paths = support.numerical_module_paths(plan)
        metadata_record = hardware["direct_process_metadata" if direct else "configured_isolation_metadata"]
        binding = support.core().bind_process(plan, metadata_record, devices[0], directory)
        if direct:
            support.post_binding_checks(plan, metadata_record, binding, directory)
            support.capture_backend(directory, binding)
        hardware = {**support.bound_hardware(plan, metadata_record, binding), "runtime_environment": plan["runtime_environment"]}
    with support.observed_work(devices[0], binding, directory / "profile_memory") if support is not None else measurement.nullcontext({}) as observed:
        configuration = plan["configuration"]
        trainer = PixelPPORND(PixelPPOConfig(**configuration["learner"]), AtariPixelEnvironment(**configuration["environment"]))
        executor = paired.Executor(trainer, configuration["execution"])
        execution_started = time.time()
        diagnostic.record_progress(directory, "normalization_warmup", profile_updates=plan["profile_updates"])
        state = trainer.warmup(trainer.initialize(tuple(plan["copy_seeds"])))
        host = paired.host_copy(state)
        del state
        diagnostic.record_progress(directory, "compile_warmup")
        host, warmup = paired.warm_executor(executor, host, plan["profile_updates"])
        loaded = training.actual_gpu_library_receipt(devices[0].client, installed["gpu_libraries"])
        training.validate_gpu_library_receipt(loaded, {"installed_gpu_libraries": installed["gpu_libraries"]})
        state = paired.device_copy(host)
        jax.block_until_ready(state)
        start_iteration = int(host.iteration)
        before = executor.cache_sizes()
        profiler, runtime_path = diagnostic.cuda_profiler()
        diagnostic.record_progress(directory, "capturing_complete_window", profile_updates=plan["profile_updates"])
        if profiler.cudaProfilerStart() != 0:
            raise RuntimeError("CUDA profiler capture did not start")
        try:
            state, metrics = executor.run(state, plan["profile_updates"], initial_iteration=start_iteration)
            jax.block_until_ready((state, metrics))
        finally:
            if profiler.cudaProfilerStop() != 0:
                raise RuntimeError("CUDA profiler capture did not stop")
        end_iteration = int(jax.device_get(state.iteration))
        if end_iteration - start_iteration != plan["profile_updates"] or executor.cache_sizes() != before:
            raise ValueError("the capture changed its complete update count or compiled another signature")
        result = {key: plan[key] for key in ("schema_version", "kind", "source_root", "source_commit", "hardware_class",
            "campaign_run_dir", "unit_id", "compilation_cache_policy", "core", "helpers", "paired_source", "configuration",
            "runtime_environment", "gpu_uuid", "profile_updates", "profile_tool", "copy_seeds")}
        result.update(capture_completed=True, hardware=hardware, profile_start_iteration=start_iteration,
            profile_end_iteration=end_iteration, compiled_signatures_before=list(before), compiled_signatures_after=list(executor.cache_sizes()),
            normalization_warmup_iterations=50, compilation_warmup=warmup, loaded_gpu_libraries=loaded,
            actual_runtime_environment=hardware["runtime_environment"],
            cuda_profiler_runtime=str(runtime_path), execution_identity={"hostname": hardware["hostname"], "pid": os.getpid(),
                "process_start_time": hardware["process_start_time"], "started_at_seconds": execution_started, "finished_at_seconds": time.time()})
        if direct:
            result.update(direct_binding=plan["direct_binding"], resolved_device_binding=binding,
                direct_binding_record=True, numerical_module_paths=numerical_paths,
                direct_qualification_inputs=plan["direct_qualification_inputs"], **support.BOUNDARY)
        elif support is not None:
            result.update(configured_isolation=plan["configured_isolation"], resolved_device_binding=binding,
                configured_isolation_record=True, numerical_module_paths=numerical_paths, target_qualification_inputs=plan["target_qualification_inputs"], **support.BOUNDARY)
    if support is not None:
        result["profile_memory"] = observed
        result["gpu_uuid"] = binding["gpu_uuid"]
    diagnostic.atomic_json(directory / "hardware.json", hardware)
    diagnostic.atomic_json(directory / "capture_result.json", result)


def capture_owned_processes(directory, plan):
    """Release the real waited capture only after observing its owned Nsight ancestry."""
    if configured_native_reader(plan) is not None:
        raise ValueError("native configured captures cannot start outside their original whole-tool scope")
    reader = process_reader(plan)
    trace = directory / "trace"; trace.mkdir()
    command = child_command(directory, plan)
    reader.publish(directory / "executed_command.json", command)
    intent = reader.make_intent(directory, plan, command)
    started, processes, child = time.monotonic(), None, None
    with (directory / "capture.log").open("x") as log:
        child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            owner = {"pid": child.pid, "process_start_time": process_birth(child.pid), "parent_pid": os.getpid(),
                     "parent_start_time": process_birth(os.getpid()), "command": command}
            reader.publish(directory / "profiler_process.json", owner)
            processes = reader.OwnedProcesses(child.pid, owner["process_start_time"], intent["parent"])
            while not all((directory / name).exists() for name in ("capture_identity.json", "capture_actual_launch.json")):
                if child.poll() is not None:
                    raise RuntimeError("Nsight ended before its actual capture launch was recorded")
                if time.monotonic() - started > min(180, plan["maximum_active_seconds"]):
                    raise TimeoutError("the actual capture identity or owned-parent record was absent")
                time.sleep(.1)
            claim = reader.validate_claim(directory, plan, intent, reader.read_json(directory / "capture_identity.json"))
            launch = reader.read_json(directory / "capture_actual_launch.json")
            if (launch["claim"] != reader.receipt(directory / "capture_identity.json")
                    or launch["challenge"] != intent["challenge"] or launch["command"] != intent["capture_command"]
                    or reader.stable(launch["child"]) != reader.stable(claim["process"])
                    or claim["process"]["parent_pid"] != launch["waiter"]["pid"]):
                raise ValueError("the actual Popen capture launch differs from its blocked child")
            observation = processes.observe(claim["pid"], claim["process_start_time"], intent["capture_command"])
            reader.publish(directory / "capture_process_observation.json", observation)
            reader.validate_observation(reader.read_json(directory / "capture_process_observation.json"),
                                        claim, intent["capture_command"], intent["parent"])
            reader.publish(directory / "capture_release.json",
                           {**owner, "capture_identity": claim,
                            "process_observation": reader.receipt(directory / "capture_process_observation.json")})
            remaining = max(.001, plan["maximum_active_seconds"] - (time.monotonic() - started))
            returncode = child.wait(timeout=remaining)
            reader.record_root_wait(directory, intent, owner, returncode)
            if returncode != 0:
                raise RuntimeError("Nsight ended unsuccessfully; retain its actual return and child wait")
            waited = reader.read_json(directory / "capture_actual_wait.json")
            if type(waited["returncode"]) is not int or waited["returncode"] != 0 or waited["interventions"]:
                raise RuntimeError("the actual capture Python did not exit normally without intervention")
        except BaseException as error:
            failure = {"error_type": type(error).__name__, "error": str(error), "traceback": traceback.format_exc()}
            try:
                if processes is not None:
                    failure["cleanup"] = processes.stop(child)
                elif child.poll() is None:
                    # Popen owns this exact unreaped root; no process-group assumption is used.
                    child.terminate()
                    try:
                        failure["popen_returncode"] = child.wait(timeout=120)
                    except subprocess.TimeoutExpired:
                        child.kill(); failure["popen_returncode"] = child.wait()
            except BaseException as cleanup_error:
                failure["cleanup_error"] = repr(cleanup_error)
                reader.publish(directory / "profile_process_failure.json", failure)
                raise RuntimeError("owned capture cleanup failed; retain both errors") from cleanup_error
            reader.publish(directory / "profile_process_failure.json", failure)
            raise
        finally:
            if processes is not None:
                processes.close()
    if tool_record(plan["profile_tool"]["path"]) != plan["profile_tool"]:
        raise ValueError("the profiling executable changed during capture")
    captured = reader.read_json(directory / "capture_result.json")
    analysis = diagnostic.summarize_trace(directory, plan["profile_tool"]["path"])
    for name, column in (("cuda_gpu_kern_sum", "Instances"), ("cuda_api_sum", "Num Calls")):
        summary = analysis["summaries"][name]
        if diagnostic.summary_csv_evidence(summary["path"], column) != {key: summary[key] for key in ("rows", "columns", "sha256")}:
            raise ValueError("the selected-baseline profile has no positive raw CUDA events")
    result = {**captured, "completed": True, "trace_analysis_record": str(directory / "trace/analysis.json"),
              "trace_report_sha256": training.file_sha256(directory / "trace/capture.nsys-rep"),
              "capture_process_protocol": OWNED_PROCESS_PROTOCOL,
              "capture_process_observation": reader.receipt(directory / "capture_process_observation.json"),
              "profiler_actual_wait": reader.receipt(directory / "profiler_actual_wait.json"),
              "capture_actual_wait": reader.receipt(directory / "capture_actual_wait.json")}
    reader.validate_saved(directory, plan, result)
    diagnostic.atomic_json(directory / "result.json", result)
    diagnostic.record_progress(directory, "completed", completed=True, profile_updates=result["profile_updates"])


def run_owned_waiter(directory):
    """Launch and wait on exactly one real Python child without initializing an array runtime."""
    directory = Path(directory).resolve(strict=True)
    plan = json.loads((directory / "measurement_plan.json").read_text())
    verify_prepared(directory, plan)
    if configured_native_reader(plan) is not None:
        return run_configured_native_waiter(directory, plan)
    reader = process_reader(plan)
    if any(name.split(".")[0] in {"jax", "jaxlib", "numpy", "torch"} for name in sys.modules):
        raise RuntimeError("the profile waiter initialized a numerical runtime")
    intent = reader.validate_intent(directory, plan, reader.read_json(directory / "profile_process_intent.json"))
    current = reader.ProcReader().snapshot(os.getpid(), os.getuid(), intent["parent"]["cgroup"])
    if current["command"] != intent["waiter_command"]:
        raise ValueError("the actual metadata waiter command differs from the frozen source")
    parent = reader.parent_handle(intent)
    command, child, launched, interventions = intent["capture_command"], None, None, []
    try:
        child = subprocess.Popen(command)
        started = time.monotonic()
        while not (directory / "capture_identity.json").exists():
            if child.poll() is not None:
                raise RuntimeError("the actual Python child ended before recording its identity")
            if reader.handle_exited(parent) or time.monotonic() - started > 180:
                raise TimeoutError("the supervisor disappeared or the actual capture identity was absent")
            time.sleep(.1)
        claim = reader.validate_claim(directory, plan, intent, reader.read_json(directory / "capture_identity.json"))
        observed = reader.ProcReader().snapshot(child.pid, os.getuid(), current["cgroup"])
        if child.pid != claim["pid"] or observed["parent_pid"] != current["pid"] or reader.stable(observed) != reader.stable(claim["process"]):
            raise ValueError("the actual Popen child differs from its pre-numerical self-record")
        launched = {"schema_version": 1, "challenge": intent["challenge"], "waiter": current, "child": observed,
                    "command": command, "claim": reader.receipt(directory / "capture_identity.json")}
        reader.publish(directory / "capture_actual_launch.json", launched)
        while child.poll() is None:
            if reader.handle_exited(parent):
                interventions.append("the original supervisor exited")
                child.terminate()
                try:
                    child.wait(timeout=120)
                except subprocess.TimeoutExpired:
                    interventions.append("the owned capture required SIGKILL")
                    child.kill(); child.wait()
                break
            time.sleep(.1)
        returncode = child.wait()
        reader.publish(directory / "capture_actual_wait.json", {
            "schema_version": 1, "challenge": intent["challenge"], "command": command,
            "waited_pid": child.pid, "returncode": returncode, "exited_normally": returncode >= 0,
            "interventions": interventions, "launch": reader.receipt(directory / "capture_actual_launch.json"),
            "wait_semantics": "Popen.wait on this exact spawned capture Python"})
        return returncode if returncode >= 0 else 128 - returncode
    except BaseException as error:
        cleanup = None
        if child is not None and child.poll() is None:
            child.terminate()
            try:
                cleanup = child.wait(timeout=120)
            except subprocess.TimeoutExpired:
                child.kill(); cleanup = child.wait()
        reader.publish(directory / "capture_waiter_failure.json", {
            "error": repr(error), "launched": launched, "known_actual_child_pid": child.pid if child else None,
            "actual_cleanup_returncode": cleanup, "challenge": intent["challenge"],
            "numerical_release_file_observed": os.path.lexists(directory / "capture_release.json"),
            "numerical_work_started": None})
        raise
    finally:
        os.close(parent)


def capture_owned_child(directory, plan):
    """Remain blocked until the actual source child and its ancestry have been checked."""
    if configured_native_reader(plan) is not None:
        return capture_configured_native_child(directory, plan)
    reader = process_reader(plan)
    claim = reader.claim_self(directory, plan)
    intent = reader.read_json(directory / "profile_process_intent.json")
    parent = reader.parent_handle(intent)
    try:
        started = time.monotonic()
        while not (directory / "capture_release.json").exists():
            if reader.handle_exited(parent) or time.monotonic() - started > 180:
                raise TimeoutError("the actual supervisor disappeared or did not release this child")
            time.sleep(.1)
        owner = reader.read_json(directory / "profiler_process.json")
        release = reader.read_json(directory / "capture_release.json")
        if release != {**owner, "capture_identity": claim,
                       "process_observation": reader.receipt(directory / "capture_process_observation.json")}:
            raise ValueError("the numerical release differs from this exact observed child")
        reader.validate_live_observation(directory, plan, claim)
        support = measurement.producer_support(plan)
        metadata_record = support.core().collect_metadata(plan, directory)
        hardware = {"configured_isolation_metadata": metadata_record}
        runtime = training.enforce_runtime(plan["runtime_environment"], os.environ)
        hardware["runtime_environment"] = runtime
        installed = reader.read_json(directory / "installed_backend.json")
        _capture_core(directory, plan, hardware, installed)
    finally:
        os.close(parent)


def configured_native_tool_record(directory, plan, scope, deadline, phase):
    """Run every version observation inside the one original owned tool scope."""
    tool = plan["profile_tool"];deadline.remaining("work")
    if training.file_sha256(Path(tool["path"])) != tool["sha256"]:
        raise ValueError("the exact native-profile tool image changed before version execution")
    completed = scope.run([tool["path"], "--version"], capture_output=True, timeout=deadline.timeout(15))
    actual = {"command": [tool["path"], "--version"], "returncode": completed.returncode,
        "stdout": completed.stdout.decode("utf-8", errors="strict"), "stderr": completed.stderr.decode("utf-8", errors="strict")}
    diagnostic.atomic_json(Path(directory) / ("native_tool_version_" + phase + ".json"), actual)
    deadline.remaining("work")
    if completed.returncode != 0 or completed.stderr != b"" or actual["stdout"].strip() != tool["version"]:
        raise ValueError("the scoped native-profile tool version differs from its frozen record")
    if training.file_sha256(Path(tool["path"])) != tool["sha256"]:
        raise ValueError("the exact native-profile tool image changed during version execution")
    return actual


def run_configured_native_prepared(directory, plan, runtime_environment):
    """Retain one explicit provider and whole-tool scope before version through export and close."""
    started = time.monotonic();native = configured_native_reader(plan);metadata = process_reader(plan)
    owner = native.context(directory, plan, metadata)
    if owner["command"] != native.producer_command(directory, plan):
        raise ValueError("execute the exact configured native producer command")
    deadline = native.deadline(directory, plan, metadata, owner=owner, started=started)
    active = None
    try:
        active = native.bootstrap(directory, plan, "producer", metadata, owner, deadline=deadline)
        observer = active.observer
        with interruption_cleanup(), measurement.card_admission(directory, plan):
            # All original source/qualification/asset checks remain before the
            # first tool. Git children cannot enter the closed tool-only domain.
            support = measurement.producer_support(plan)
            supervisor = directory / "supervisor";supervisor.mkdir(exist_ok=False)
            configuration_metadata = support.core().collect_metadata(plan, supervisor)
            support.verify_qualification_inputs(plan)
            validate_measured_baseline(directory, plan)
            assets = training.verify_assets(plan["asset_record"], os.environ)
            installed = {"jaxtari_source": training.jaxtari_python_receipt(plan.get("jaxtari_source_root")),
                "gpu_libraries": training.installed_gpu_library_receipts()}
            diagnostic.atomic_json(directory / "supervisor_metadata.json", {
                "hardware": {"configured_isolation_metadata": configuration_metadata}, "runtime_environment": runtime_environment})
            diagnostic.atomic_json(directory / "asset_receipt.json", assets)
            diagnostic.atomic_json(directory / "installed_backend.json", installed)
            if any(name.split(".")[0] in {"jax", "jaxlib", "numpy", "torch"} for name in sys.modules):
                raise RuntimeError("the configured native metadata producer initialized a numerical runtime")
            deadline.remaining("work")
            with native.tool_scope(directory, plan, observer, owner, deadline=deadline) as scope:
                configured_native_tool_record(directory, plan, scope, deadline, "before")
                result = capture_configured_native_processes(directory, plan, active, scope, deadline)
            active.complete(native.completion_references(plan, "producer"))
            intent = metadata.read_json(directory / "profile_process_intent.json")
            observation = metadata.read_json(directory / "capture_process_observation.json")
            launch = metadata.read_json(directory / "capture_actual_launch.json")
            waited = metadata.read_json(directory / "capture_actual_wait.json")
            result["configured_native_process_proof"] = native.validate_saved(
                directory, plan, metadata, intent, observation, launch, waited, native.saved_deadline(directory, plan, metadata))
            metadata.validate_saved(directory, plan, result)
            diagnostic.atomic_json(directory / "result.json", result)
            diagnostic.record_progress(directory, "completed", completed=True, profile_updates=result["profile_updates"])
            return result
    except BaseException as error:
        if active is not None:active.failure(error)
        raise


def capture_configured_native_processes(directory, plan, active, scope, deadline):
    """Wait on the exact capture root and export only through the already retained producer scope."""
    native = configured_native_reader(plan);observer = native.runtime_observer(plan, active.observer)
    if observer.native_process_runtime is not active or active.role != "producer":
        raise ValueError("the capture must reuse its original unfinished native producer provider")
    if scope.observer is not observer or scope.plan != plan or not scope.active or scope.deadline is not deadline:
        raise ValueError("the native capture cannot replace its established producer scope or original deadline")
    trace = directory / "trace";trace.mkdir()
    command = child_command(directory, plan);observer.publish(directory / "executed_command.json", command)
    intent = observer.make_intent(directory, plan, command)
    processes = None
    with (directory / "capture.log").open("x") as log:
        child = scope.spawn(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        initial = scope.nodes[child.pid]["initial"]
        owner = {"pid": child.pid, "process_start_time": initial["process_start_time"], "parent_pid": os.getpid(),
            "parent_start_time": intent["parent"]["process_start_time"], "command": command}
        observer.publish(directory / "profiler_process.json", owner)
        try:
            while not all((directory / name).exists() for name in ("capture_identity.json", "capture_actual_launch.json")):
                if child.poll() is not None:raise RuntimeError("the actual scoped Nsight root ended before its blocked capture claim")
                deadline.remaining("work");scope.scan();time.sleep(.05)
            # The closed scope already pins the real root through startup.
            # Strict complete-command ancestry starts at the blocked claim,
            # before release, rather than rejecting a transient empty cmdline.
            processes = observer.OwnedProcesses(child.pid, initial["process_start_time"], intent["parent"])
            claim = observer.validate_claim(directory, plan, intent, observer.read_json(directory / "capture_identity.json"))
            launch = observer.read_json(directory / "capture_actual_launch.json")
            if (launch["claim"] != observer.receipt(directory / "capture_identity.json")
                    or launch["challenge"] != intent["challenge"] or launch["command"] != intent["capture_command"]
                    or observer.stable(launch["child"]) != observer.stable(claim["process"])
                    or claim["process"]["parent_pid"] != launch["waiter"]["pid"]):
                raise ValueError("the scoped capture launch differs from the actual source-bound child")
            observation = processes.observe(claim["pid"], claim["process_start_time"], intent["capture_command"])
            observer.publish(directory / "capture_process_observation.json", observation)
            observer.validate_observation(observation, claim, intent["capture_command"], intent["parent"])
            deadline.remaining("work")
            observer.publish(directory / "capture_release.json", {**owner, "capture_identity": claim,
                "process_observation": observer.receipt(directory / "capture_process_observation.json")})
            returned = scope.wait(child, deadline.timeout())
            observer.record_root_wait(directory, intent, owner, returned)
            waited = observer.read_json(directory / "capture_actual_wait.json")
            if returned != 0 or type(waited["returncode"]) is not int or waited["returncode"] != 0 or waited["interventions"]:
                raise RuntimeError("the scoped root or actual capture did not return normally without intervention")
        finally:
            if processes is not None:processes.close()
    configured_native_tool_record(directory, plan, scope, deadline, "after")
    captured = observer.read_json(directory / "capture_result.json")
    analysis = diagnostic.summarize_trace(directory, plan["profile_tool"]["path"], run_command=scope.run)
    deadline.remaining("work")
    if training.file_sha256(Path(plan["profile_tool"]["path"])) != plan["profile_tool"]["sha256"]:
        raise ValueError("the original native-profile tool image changed during final export")
    for name, column in (("cuda_gpu_kern_sum", "Instances"), ("cuda_api_sum", "Num Calls")):
        summary = analysis["summaries"][name]
        if diagnostic.summary_csv_evidence(summary["path"], column) != {key: summary[key] for key in ("rows", "columns", "sha256")}:
            raise ValueError("the scoped native profile lacks unchanged positive raw CUDA events")
    return {**captured, "completed": True, "trace_analysis_record": str(trace / "analysis.json"),
        "trace_report_sha256": training.file_sha256(trace / "capture.nsys-rep"),
        "capture_process_protocol": OWNED_PROCESS_PROTOCOL,
        "capture_process_observation": observer.receipt(directory / "capture_process_observation.json"),
        "profiler_actual_wait": observer.receipt(directory / "profiler_actual_wait.json"),
        "capture_actual_wait": observer.receipt(directory / "capture_actual_wait.json")}


def run_configured_native_waiter(directory, plan):
    """Give this exact metadata waiter its own provider and wait on one positively identified child."""
    native = configured_native_reader(plan);metadata = process_reader(plan)
    if any(name.split(".")[0] in {"jax", "jaxlib", "numpy", "torch"} for name in sys.modules):
        raise RuntimeError("the configured native waiter initialized a numerical runtime")
    owner = native.context(directory, plan, metadata)
    intent = metadata.validate_intent(directory, plan, metadata.read_json(directory / "profile_process_intent.json"))
    if owner["command"] != intent["waiter_command"]:raise ValueError("the native waiter source command changed")
    deadline = native.deadline(directory, plan, metadata)
    active = native.bootstrap(directory, plan, "waiter", metadata, owner, deadline=deadline)
    observer = active.observer;parent = None;child = None;child_handle = None;launched = None
    try:
        parent = observer.parent_handle(intent)
        deadline.remaining("work")
        child = subprocess.Popen(intent["capture_command"], close_fds=True, pass_fds=())
        child_handle = observer.open_process_handle(child.pid)
        while not (directory / "capture_identity.json").exists():
            if child.poll() is not None or observer.handle_exited(parent):
                raise RuntimeError("the native capture or pinned original producer exited before its claim")
            time.sleep(min(.05, deadline.remaining("work")))
        claim = observer.validate_claim(directory, plan, intent, observer.read_json(directory / "capture_identity.json"))
        actual = observer.ProcReader().snapshot(child.pid, os.getuid(), owner["cgroup"])
        if (child.pid != claim["pid"] or actual["parent_pid"] != owner["pid"]
                or observer.stable(actual) != observer.stable(claim["process"]) or observer.handle_exited(child_handle)):
            raise ValueError("the native waiter actual Popen child differs from its pinned source-bound claim")
        launched = {"schema_version": 1, "challenge": intent["challenge"], "waiter": owner, "child": actual,
            "command": intent["capture_command"], "claim": observer.receipt(directory / "capture_identity.json")}
        observer.publish(directory / "capture_actual_launch.json", launched)
        while child.poll() is None:
            if observer.handle_exited(parent):raise RuntimeError("the pinned native producer exited before the capture wait")
            time.sleep(min(.05, deadline.remaining("work")))
        returned = child.wait(timeout=deadline.timeout())
        actual_wait_finished = time.time()
        observer.publish(directory / "capture_actual_wait.json", {"schema_version": 1, "challenge": intent["challenge"],
            "command": intent["capture_command"], "waited_pid": child.pid, "returncode": returned,
            "exited_normally": returned >= 0, "interventions": [], "launch": observer.receipt(directory / "capture_actual_launch.json"),
            "wait_semantics": "Popen.wait on this exact spawned capture Python",
            "actual_wait_finished_at_seconds": actual_wait_finished})
        if returned != 0:raise RuntimeError("the actual native-profile capture returned unsuccessfully")
    except BaseException as error:
        # The producer's closed scope retains all descendants, including an
        # adopted capture. Never guess or signal a process by numeric PID here.
        observer.publish(directory / "capture_waiter_failure.json", {"error": repr(error), "launched": launched,
            "known_actual_child_pid": None if child is None else child.pid,
            "actual_child_wait_completed": child is not None and child.returncode is not None,
            "producer_scope_must_close_remaining_descendants": True, "scientific_credit": 0})
        active.failure(error);raise
    finally:
        if child_handle is not None:observer.close_process_handle(child_handle)
        if parent is not None:observer.close_process_handle(parent)
    active.complete(native.completion_references(plan, "waiter"))
    return returned


def capture_configured_native_child(directory, plan):
    """Keep the numerical child unchanged inside an explicit provider/release/timing envelope."""
    native = configured_native_reader(plan);metadata = process_reader(plan)
    owner = native.context(directory, plan, metadata)
    intent = metadata.validate_intent(directory, plan, metadata.read_json(directory / "profile_process_intent.json"))
    if owner["command"] != intent["capture_command"]:raise ValueError("the native capture source command changed")
    deadline = native.deadline(directory, plan, metadata)
    active = native.bootstrap(directory, plan, "capture", metadata, owner, deadline=deadline)
    observer = active.observer;parent = None
    try:
        claim = observer.claim_self(directory, plan);parent = observer.parent_handle(intent)
        while not os.path.lexists(directory / "capture_release.json"):
            if observer.handle_exited(parent):raise RuntimeError("the pinned native producer disappeared before release")
            time.sleep(min(.05, deadline.remaining("work")))
        profiler = observer.read_json(directory / "profiler_process.json")
        release = observer.read_json(directory / "capture_release.json")
        if release != {**profiler, "capture_identity": claim,
                "process_observation": observer.receipt(directory / "capture_process_observation.json")}:
            raise ValueError("the native release differs from this exact source-bound child")
        observer.validate_live_observation(directory, plan, claim);deadline.remaining("work")
        observer.publish(directory / "capture_release_observed.json", {"release": observer.receipt(directory / "capture_release.json"),
            "capture_identity": observer.receipt(directory / "capture_identity.json"), "observed_at_seconds": time.time(),
            "scientific_credit": 0, "admission_granted": False})
        support = measurement.producer_support(plan)
        configuration_metadata = support.core().collect_metadata(plan, directory)
        hardware = {"configured_isolation_metadata": configuration_metadata}
        hardware["runtime_environment"] = training.enforce_runtime(plan["runtime_environment"], os.environ)
        installed = observer.read_json(directory / "installed_backend.json")
        payload_started = time.time();deadline.remaining("work")
        _capture_core(directory, plan, hardware, installed)
        payload_finished = time.time();deadline.remaining("work")
        observer.publish(directory / "native_capture_payload_timing.json", {"payload_started_at_seconds": payload_started,
            "payload_finished_at_seconds": payload_finished, "diagnostic_deadline": deadline.snapshot(),
            "payload_kind": "original_full_profile_capture", "numerical_timestamps_reinterpreted": False,
            "scientific_credit": 0, "admission_granted": False})
    except BaseException as error:
        active.failure(error);raise
    finally:
        if parent is not None:observer.close_process_handle(parent)
    active.complete(native.completion_references(plan, "capture"))


def capture_direct_owned_processes(directory, plan, held_descriptor):
    """Retain sole producer custody while the new direct observer waits on its actual tool child."""
    if type(held_descriptor) is not int or held_descriptor < 0:
        raise ValueError("the direct tool producer must retain its actual local standing descriptor")
    support = measurement.producer_support(plan)
    parent_metadata = support.core().read_receipt(support.core().receipt(directory / "supervisor/direct_process_metadata.json"))
    support.core().live_custody(parent_metadata, held_descriptor)
    reader = direct_process_reader(plan)
    with reader.tool_scope(directory, plan, held_descriptor) as scope:
        for stage in ("before", "after"):
            actual = scope.run([plan["profile_tool"]["path"], "--version"], capture_output=True, text=True, timeout=15)
            support.core().publish(directory / ("profile_tool_version_" + stage + ".json"), {
                "command": [plan["profile_tool"]["path"], "--version"], "returncode": actual.returncode,
                "stdout": actual.stdout, "stderr": actual.stderr})
            if (actual.returncode != 0 or actual.stderr != "" or actual.stdout.strip() != plan["profile_tool"]["version"]
                    or training.file_sha256(plan["profile_tool"]["path"]) != plan["profile_tool"]["sha256"]):
                raise ValueError("the direct profiling tool changed before or after its actual source-child capture")
            if stage == "before":
                reader.run_tool(directory, plan, held_descriptor, scope=scope)
        analysis = diagnostic.summarize_trace(directory, plan["profile_tool"]["path"], run_command=scope.run)
    support.core().live_custody(parent_metadata, held_descriptor)
    captured = json.loads((directory / "capture_result.json").read_bytes())
    for name, column in (("cuda_gpu_kern_sum", "Instances"), ("cuda_api_sum", "Num Calls")):
        summary = analysis["summaries"][name]
        if diagnostic.summary_csv_evidence(summary["path"], column) != {key: summary[key] for key in ("rows", "columns", "sha256")}:
            raise ValueError("the actual direct profile lacks positive unchanged kernel/API summaries")
    metadata = support.core().read_receipt(support.core().receipt(directory / "direct_process_metadata.json"))
    process_proof = reader.validate_saved(directory, plan)
    terminal = {"schema_version": 1, "kind": "direct_producer_guarded_profile_terminal", "process_proof": process_proof,
        "capture_result": support.core().receipt(directory / "capture_result.json"),
        "parent_custody": support.core().receipt(directory / "supervisor/direct_process_metadata.json"),
        "standing_descriptor_inherited_by_child": False, **support.BOUNDARY}
    support.core().publish(directory / "direct_profile_terminal.json", terminal)
    result = {**captured, "completed": True, "capture_process_protocol": reader.PROTOCOL,
        "trace_analysis_record": str(directory / "trace/analysis.json"),
        "trace_report_sha256": training.file_sha256(directory / "trace/capture.nsys-rep"),
        "direct_trace_process_proof": support.trace_process_proof(directory, metadata),
        "direct_process_wait_proof": process_proof, **direct_profile_reader(plan).completed_capture_binding(directory, plan, captured)}
    support.core().live_custody(parent_metadata, held_descriptor)
    diagnostic.atomic_json(directory / "result.json", result)
    diagnostic.record_progress(directory, "completed", completed=True, profile_updates=result["profile_updates"])


def run_direct_owned_waiter(directory):
    """Wait on the new direct source child without receiving its producer's standing descriptor."""
    directory = Path(directory).resolve(strict=True)
    plan = json.loads((directory / "measurement_plan.json").read_bytes())
    verify_prepared(directory, plan)
    return direct_process_reader(plan).run_waiter(directory, plan)


def capture_direct_owned_child(directory):
    """Require actual source/ancestry/producer custody before unchanged CUDA binding and learner work."""
    directory = Path(directory).resolve(strict=True)
    plan = json.loads((directory / "measurement_plan.json").read_bytes())
    verify_prepared(directory, plan)
    reader = direct_process_reader(plan)
    reader.wait_release(directory, plan)
    support = measurement.producer_support(plan)
    metadata_record = reader.collect_capture_metadata(plan, directory, support.core())
    runtime = training.enforce_runtime(plan["runtime_environment"], os.environ)
    hardware = {"direct_process_metadata": metadata_record, "runtime_environment": runtime}
    installed = json.loads((directory / "installed_backend.json").read_bytes())
    _capture_core(directory, plan, hardware, installed)
    direct_profile_reader(plan).complete_capture(directory, plan)


def parse_arguments(argv=None):
    """Separate metadata/tool inspection, preparation, parent execution, and its private capture child."""
    parser = argparse.ArgumentParser(description=__doc__)
    operation = parser.add_mutually_exclusive_group()
    operation.add_argument("--prepared-unit", type=Path)
    operation.add_argument("--capture-unit", type=Path)
    operation.add_argument("--owned-waiter-unit", type=Path)
    operation.add_argument("--direct-owned-waiter-unit", type=Path, help=argparse.SUPPRESS)
    operation.add_argument("--direct-owned-capture-unit", type=Path, help=argparse.SUPPRESS)
    operation.add_argument("--direct-waiter-unit", type=Path, help=argparse.SUPPRESS)
    operation.add_argument("--inspect-profile-tool", type=Path)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--configured-isolation-profile", type=Path)
    parser.add_argument("--configured-native-process-qualification", type=Path)
    parser.add_argument("--configured-native-process-review", type=Path)
    parser.add_argument("--target-qualification-record", type=Path)
    parser.add_argument("--target-qualification-review", type=Path)
    parser.add_argument("--direct-binding-profile", type=Path)
    parser.add_argument("--direct-history-inputs", type=Path)
    parser.add_argument("--direct-qualification-record", type=Path)
    parser.add_argument("--direct-qualification-review", type=Path)
    parser.add_argument("--direct-backend-image-record", type=Path)
    parser.add_argument("--direct-confirmation-review", type=Path)
    parser.add_argument("--direct-held-lock-fd", type=int, help=argparse.SUPPRESS)
    for name in ("source-root", "configuration", "runtime-environment", "asset-record", "study-registry",
                 "accepted-baseline-specification", "profile-tool-record", "jaxtari-source-root", "run-dir"):
        parser.add_argument("--" + name, type=Path)
    for name in ("source-commit", "unit-id", "gpu-uuid", "hardware-class"):
        parser.add_argument("--" + name)
    parser.add_argument("--seed-start", type=int)
    parser.add_argument("--minimum-updates", type=int)
    parser.add_argument("--maximum-active-seconds", type=float)
    parser.add_argument("--python", type=Path)
    args = parser.parse_args(argv)
    fields = ("source_root", "source_commit", "configuration", "runtime_environment", "asset_record", "study_registry",
              "accepted_baseline_specification", "profile_tool_record", "run_dir", "unit_id", "gpu_uuid", "hardware_class")
    if (args.prepared_unit or args.capture_unit or args.owned_waiter_unit or args.direct_waiter_unit
            or args.direct_owned_waiter_unit or args.direct_owned_capture_unit or args.inspect_profile_tool):
        if args.prepare_only or any(getattr(args, name) is not None for name in (*fields, "jaxtari_source_root", "seed_start",
                                                                              "minimum_updates", "maximum_active_seconds", "python", "configured_isolation_profile", "target_qualification_record", "target_qualification_review",
                                                                              "direct_binding_profile", "direct_history_inputs", "direct_qualification_record", "direct_qualification_review", "direct_backend_image_record", "direct_confirmation_review", "configured_native_process_qualification", "configured_native_process_review")):
            parser.error("execution/tool inspection cannot override a prepared configuration")
        if args.direct_held_lock_fd is not None:
            if (args.capture_unit is None and args.direct_waiter_unit is None) or args.direct_held_lock_fd < 0:
                parser.error("the actual inherited descriptor applies only to a prepared direct capture child")
            plan = json.loads(((args.capture_unit or args.direct_waiter_unit) / "measurement_plan.json").read_text())
            if "direct_binding" not in plan:
                parser.error("the descriptor cannot be supplied to a default or UVA capture")
            measurement.producer_support(plan)
        elif args.direct_waiter_unit is not None:
            parser.error("the direct waiter requires its actual inherited descriptor")
        return args
    if args.direct_held_lock_fd is not None:
        parser.error("the actual inherited descriptor requires a prepared direct capture child")
    if args.configured_isolation_profile is not None:
        if args.gpu_uuid is not None or args.hardware_class != "jaguar03_rtx_a4500" or not args.target_qualification_record or not args.target_qualification_review:
            parser.error("configured profiling needs null physical identity and actual qualification/review")
        fields = tuple(name for name in fields if name != "gpu_uuid")
    elif args.target_qualification_record or args.target_qualification_review:
        parser.error("target qualification inputs require an explicit configured-isolation profile")
    if args.direct_binding_profile is not None:
        if (args.configured_isolation_profile is not None or args.hardware_class != "h100" or args.gpu_uuid != measurement.CARD1
                or any(value is None for value in (args.direct_history_inputs, args.direct_qualification_record,
                    args.direct_qualification_review, args.direct_backend_image_record, args.direct_confirmation_review))):
            parser.error("direct profiling requires its exact card-one profile, history, backend record and actual qualification/common timing reviews")
    elif any(value is not None for value in (args.direct_history_inputs, args.direct_qualification_record,
            args.direct_qualification_review, args.direct_backend_image_record, args.direct_confirmation_review)):
        parser.error("direct profile inputs require the separate direct binding profile")
    if any(getattr(args, name) is None for name in fields):
        parser.error("preparation needs explicit source/config/runtime/assets/initial study/baseline/tool/GPU/output inputs")
    args.seed_start = 1 if args.seed_start is None else args.seed_start
    args.minimum_updates = 1 if args.minimum_updates is None else args.minimum_updates
    args.maximum_active_seconds = 3600 if args.maximum_active_seconds is None else args.maximum_active_seconds
    args.python = args.python or Path("/p/rlprojects/RND/.venvs/atari_pixels_jax/bin/python")
    return args


if __name__ == "__main__":
    arguments = parse_arguments()
    if arguments.inspect_profile_tool:
        print(json.dumps(tool_record(arguments.inspect_profile_tool), indent=2), flush=True)
    elif arguments.prepared_unit:
        run_prepared(arguments.prepared_unit)
    elif arguments.owned_waiter_unit:
        raise SystemExit(run_owned_waiter(arguments.owned_waiter_unit))
    elif arguments.direct_owned_waiter_unit:
        raise SystemExit(run_direct_owned_waiter(arguments.direct_owned_waiter_unit))
    elif arguments.direct_owned_capture_unit:
        capture_direct_owned_child(arguments.direct_owned_capture_unit)
    elif arguments.capture_unit:
        if arguments.direct_held_lock_fd is not None:
            capture(arguments.capture_unit, arguments.direct_held_lock_fd)
        else:
            capture(arguments.capture_unit)
    elif arguments.direct_waiter_unit:
        directory = arguments.direct_waiter_unit.resolve(strict=True)
        plan = json.loads((directory / "measurement_plan.json").read_text())
        verify_prepared(directory, plan)
        raise SystemExit(measurement.producer_support(plan).run_profile_waiter(plan, directory, arguments.direct_held_lock_fd))
    else:
        directory = prepare(arguments)
        print(json.dumps({"prepared": True, "unit_directory": str(directory)}, indent=2), flush=True)
        if not arguments.prepare_only:
            os.execv(str(arguments.python), [str(arguments.python), str(directory / "code/profile_selected.py"),
                                            "--prepared-unit", str(directory)])
