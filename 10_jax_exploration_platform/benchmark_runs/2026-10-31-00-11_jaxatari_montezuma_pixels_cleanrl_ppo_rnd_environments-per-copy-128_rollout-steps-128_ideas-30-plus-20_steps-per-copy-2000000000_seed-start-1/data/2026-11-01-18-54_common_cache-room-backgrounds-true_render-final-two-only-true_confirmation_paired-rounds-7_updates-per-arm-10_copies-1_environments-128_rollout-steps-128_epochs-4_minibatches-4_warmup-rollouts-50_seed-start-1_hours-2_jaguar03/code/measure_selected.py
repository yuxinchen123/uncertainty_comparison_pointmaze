"""Measure explicit combined configurations or one isolated packing point.

This program never selects hypotheses or submits work. --prepare-only stages
immutable metadata/helpers without importing numerical libraries. The recorded
prepared unit can then run on its selected compute GPU via --prepared-unit.
"""

import argparse
from contextlib import contextmanager, nullcontext
import csv
import fcntl
import hashlib
import importlib
import importlib.util
from importlib import metadata
import json
import math
import os
from pathlib import Path
import shlex
import socket
import stat
import statistics
import subprocess
import sys
import threading
import time
import traceback
import xml.etree.ElementTree as ET


SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))
import diagnose_gpu_execution as diagnostic
import post_comparison_validation as validation
import train_optimized as training

HELPERS = ("measure_selected.py", "diagnose_gpu_execution.py", "post_comparison_validation.py", "infrastructure_retry.py", "pregather_async_retry.py",
           "h100_nvl_slurm_queue.py", "h100_nvl_slurm_case.py", "h100_nvl_slurm_evidence.py", "h100_nvl_slurm_site_validation.py",
           "h100_nvl_slurm_qualification.py", "h100_nvl_allocation_observation.py",
           "train_optimized.py", "short_learning.py", "device_memory_limit.py", "validate_training_deployment.py", "portable_evidence.py")
CARD1 = "GPU-fed69954-5a71-dbb2-ee91-544e0eb6b8a3"
MODES = ("correctness", "confirmation", "packing-point", "learning-check")


def requested_protocol(arguments):
    """Keep intermediate measurements unchanged and make the final protocol explicit."""
    # Before: an absent protocol selects seven by ten; after: final confirmation is eleven by twenty.
    final = getattr(arguments, "final_confirmation", False)
    if final and arguments.mode != "confirmation":
        raise ValueError("--final-confirmation applies only to a final paired confirmation")
    rounds = (11 if final else 7) if arguments.rounds is None else arguments.rounds
    minimum = (20 if final else 10) if arguments.minimum_updates_per_round is None else arguments.minimum_updates_per_round
    if (type(rounds) is not int or rounds != (11 if final else 7) or type(minimum) is not int
            or minimum < (20 if final else 10) or final and minimum != 20):
        raise ValueError("use eleven rounds of at least twenty updates for final confirmation; seven rounds otherwise")
    return rounds, minimum, final


def read_document(path):
    """Keep an original JSON path and its exact bytes hash alongside the resolved input."""
    path = Path(path).resolve(strict=True)
    contents = path.read_bytes()
    return json.loads(contents), {"path": str(path), "sha256": hashlib.sha256(contents).hexdigest()}


def installed_backend_receipt(jaxtari_source, gpu_libraries):
    """Record actual installed package versions beside verified Python and GPU library bytes."""
    return {"jaxtari_source": jaxtari_source, "gpu_libraries": gpu_libraries,
            "packages": {name: metadata.version(name) for name in training.PACKAGE_NAMES}}


def write_once(path, contents):
    """Retain identical staged inputs and reject any replacement of an existing unit artifact."""
    if path.exists():
        if path.read_bytes() != contents:
            raise ValueError("an immutable measurement artifact differs: " + str(path))
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(contents)
        stream.flush()
        os.fsync(stream.fileno())


def json_bytes(value):
    """Use JSON, also valid YAML, for complete launch and configuration records."""
    return (json.dumps(value, indent=2, allow_nan=False) + "\n").encode()


def staged_short_learning(directory, plan):
    """Load only the exact staged diagnostic helper, regardless of numerical-root imports or module cache."""
    path = Path(directory) / "code/short_learning.py"
    relative = "10_jax_exploration_platform/scripts/atari_pixel_campaign/short_learning.py"
    expected = plan["helpers"]["files"][relative]
    raw = path.read_bytes()
    if (path.is_symlink() or path.parent.is_symlink() or path.parent.resolve(strict=True).parent != Path(directory).resolve(strict=True)
            or path.resolve(strict=True).parent != path.parent.resolve(strict=True) or hashlib.sha256(raw).hexdigest() != expected):
        raise ValueError("the exact staged short-learning helper changed its location or bytes")
    specification = importlib.util.spec_from_file_location("exact_staged_short_learning_" + expected[:16], path)
    module = importlib.util.module_from_spec(specification)
    # Execute precisely these verified bytes; cached bytecode and numerical-root imports cannot substitute code.
    exec(compile(raw, str(path), "exec"), module.__dict__)
    if Path(module.__file__).resolve() != path.resolve(strict=True):
        raise ValueError("short learning imported a different helper source")
    return module


def completed_study_inputs(arguments, source, runtime, configuration, *, required=None, hardware_class=None):
    """Prepare only reviewed completed-study bytes; native login preparation never recomputes science."""
    proof, review = getattr(arguments, "completed_study_proof", None), getattr(arguments, "completed_study_review", None)
    manifest = getattr(arguments, "completed_study_manifest", None)
    manifest_sha = getattr(arguments, "completed_study_manifest_sha256", None)
    required = (getattr(arguments, "mode", None) == "learning-check" or getattr(arguments, "final_confirmation", False)) if required is None else required
    if not required:
        if any(value is not None for value in (proof, review, manifest, manifest_sha)):
            raise ValueError("completed-study proof inputs apply only to final confirmation or final learning")
        return None, None
    if proof is None or review is None or (manifest is None) != (manifest_sha is None):
        raise ValueError("final preparation requires the completed-study proof, independent review and exact optional mapping hash")
    resolver, mapping = None, None
    if manifest is not None:
        from portable_evidence import ExactPathResolver
        manifest = Path(manifest).resolve(strict=True)
        resolver = ExactPathResolver(manifest, manifest_sha)
        mapping = {"path": str(manifest), "sha256": manifest_sha}
    evidence = validation.Evidence(resolver)
    binding = {name: {"path": evidence.path_identity(path), "sha256": hashlib.sha256(evidence.file(path)).hexdigest()}
               for name, path in (("proof", proof), ("review", review))}
    validation.verify_completed_study(binding, source, runtime, evidence, before_seconds=time.time(),
        hardware_class=hardware_class or arguments.hardware_class, selected_configuration=configuration)
    return binding, mapping


def verify_completed_study_plan(plan):
    """Recheck the exact final-study proof and native mappings immediately before device admission."""
    if "completed_study" not in plan:
        if plan.get("mode") == "learning-check" or "final_confirmation_protocol" in plan or plan.get("scaling_protocol", {}).get("kind") == "fixed_total_copies":
            raise ValueError("final measurement omitted its reviewed completed-fifty-study proof")
        return
    resolver = None
    if plan.get("completed_study_manifest") is not None:
        from portable_evidence import ExactPathResolver
        mapping = plan["completed_study_manifest"]
        resolver = ExactPathResolver(mapping["path"], mapping["sha256"])
    validation.verify_completed_study(plan["completed_study"], plan["source_commit"],
        plan["runtime_environment"], validation.Evidence(resolver), before_seconds=time.time(),
        hardware_class=plan["hardware_class"], selected_configuration=plan["configuration"])


def measurement_window(configurations, minimum=10):
    """Use complete update groups and complete report windows without padding either variant."""
    if type(minimum) is not int or minimum < 10:
        raise ValueError("selected-configuration measurements require at least ten complete updates per round")
    multiple = 1
    for config in configurations:
        options = config["execution"]
        multiple = math.lcm(multiple, options["iterations_per_call"])
        if options["report_readout"] != "none":
            multiple = math.lcm(multiple, options["report_every_iterations"])
    return ((minimum + multiple - 1) // multiple) * multiple


def validate_correctness_protocol(configurations):
    """Reject group pairs the unchanged Executor correctness function cannot compare."""
    count = max(config["execution"]["iterations_per_call"] for config in configurations)
    for config in configurations:
        options = config["execution"]
        if options["report_readout"] != "none":
            count = max(count, options["report_every_iterations"])
    if any(count % config["execution"]["iterations_per_call"] for config in configurations):
        raise ValueError("the frozen correctness protocol cannot compare these execution group sizes; "
                         "use supported groups or a separately reviewed numerical harness")


def helper_receipt(configured=False, direct=False, common=False):
    """Freeze only committed producer/helper bytes, separately from the unchanged numerical core."""
    root = SCRIPTS.parents[2]
    commit = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    paths = [str((SCRIPTS / name).relative_to(root)) for name in required_helpers(configured, direct, common)]
    receipt = training.source_receipt(root, commit, paths)
    return receipt


def required_helpers(configured=False, direct=False, common=False):
    """Keep historical helper closure unchanged and explicitly add the opt-in dependency union."""
    if common and (not configured or direct):
        raise ValueError("common confirmation requires only the configured A4500 helper closure")
    if configured and direct:
        raise ValueError("direct and UVA helper policies cannot be mixed")
    if direct:
        import direct_measurement as support
        return tuple(dict.fromkeys((*HELPERS, *support.helper_names())))
    if not configured:
        return HELPERS
    import configured_measurement as support
    return tuple(dict.fromkeys((*HELPERS, *support.helper_names(), *(("common_confirmation.py",) if common else ()))))


def common_confirmation_support(plan):
    """Load only the hash-bound optional prerequisite helper without numerical imports."""
    if "common_correctness_inputs" not in plan:
        return None
    path = SCRIPTS / "common_confirmation.py"
    expected = plan["helpers"]["files"].get("10_jax_exploration_platform/scripts/atari_pixel_campaign/common_confirmation.py")
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise ValueError("the committed common-confirmation prerequisite helper changed")
    specification = importlib.util.spec_from_file_location("common_confirmation_prerequisite", path)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    module.contract(plan)
    return module


def configured_support(plan):
    """Reject an unsupported consumer before loading the explicit staged identity adapter."""
    if "configured_isolation" not in plan:
        if "target_qualification" in plan or "target_qualification_inputs" in plan:
            raise ValueError("target qualification requires an explicit configured-isolation plan")
        return None
    import configured_measurement as support
    support.core().validate_plan(plan)
    if "target_qualification" in plan:
        if not support.core().exact(plan["target_qualification"], support.QUALIFICATION) or "target_qualification_inputs" in plan:
            raise ValueError("a standalone qualification cannot depend on or alter a prior qualification")
        support.validate_original_qualification(plan)
    elif "target_qualification_inputs" not in plan:
        raise ValueError("ordinary configured measurements require the separate actual qualification inputs")
    return support


def producer_support(plan):
    """Keep the separate direct policy explicit before the existing UVA/default dispatch."""
    if "direct_binding" in plan:
        import direct_measurement as support
        return support.support(plan)
    if "direct_qualification" in plan or "direct_qualification_inputs" in plan:
        raise ValueError("direct qualification requires its explicit direct binding profile")
    return configured_support(plan)


def validate_complete_fields(root, config):
    """Require the pinned source's complete learner/environment fields before constructing arrays."""
    defaults = diagnostic.default_configuration(root)
    if set(config["learner"]) != set(defaults["learner"]) or set(config["environment"]) != set(defaults["environment"]):
        raise ValueError("supply every learner and environment field explicitly from the pinned source")


def prepare(args):
    """Create the unit's full launch records and frozen helper bytes without GPU or numerical imports."""
    if args.mode not in MODES or args.hardware_class not in validation.CONTEXTS:
        raise ValueError("the measurement mode or hardware class is outside this study")
    if getattr(args, 'memory_limit_ownership_profile', None) is not None and not getattr(args, 'record_device_memory_limit', False):
        raise ValueError('an ownership profile requires explicit device-memory instrumentation')
    configured = getattr(args, "configured_isolation_profile", None) is not None
    direct = getattr(args, "direct_binding_profile", None) is not None
    qualification = getattr(args, "target_qualification", False)
    direct_qualification = getattr(args, "direct_target_qualification", False)
    correctness_record = getattr(args, "common_correctness_record", None)
    correctness_review = getattr(args, "common_correctness_review", None)
    common = correctness_record is not None or correctness_review is not None
    if common and (correctness_record is None or correctness_review is None or not configured or direct
            or args.mode != "confirmation" or qualification or direct_qualification or getattr(args, "final_confirmation", False)):
        raise ValueError("common confirmation requires both actual correctness records and only intermediate A4500 confirmation")
    if direct and (configured or qualification or args.hardware_class != "h100" or args.gpu_uuid != CARD1
            or args.mode not in {"correctness", "confirmation"} or getattr(args, "record_device_memory_limit", False)
            or getattr(args, "final_confirmation", False)):
        raise ValueError("direct support is only the approved card-one intermediate or standalone qualification path")
    if direct_qualification and (not direct or args.mode != "correctness"):
        raise ValueError("standalone direct qualification requires explicit direct correctness")
    if qualification and (not configured or args.mode != "correctness"):
        raise ValueError("target qualification requires explicit configured-isolation correctness")
    if configured and (args.gpu_uuid is not None or args.hardware_class != "jaguar03_rtx_a4500"
            or args.mode not in {"correctness", "confirmation"} or getattr(args, "final_confirmation", False)
            or getattr(args, "record_device_memory_limit", False)):
        raise ValueError("configured isolation supports only an unresolved single A4500 intermediate comparison")
    if (not configured and (not isinstance(args.gpu_uuid, str) or not args.gpu_uuid.startswith("GPU-"))) or Path(args.unit_id).name != args.unit_id or args.unit_id in {"", ".", ".."}:
        raise ValueError("record a physical GPU UUID and a plain descriptive unit identifier")
    rounds, minimum, final = requested_protocol(args)
    if not 0 <= args.seed_start < 2 ** 32:
        raise ValueError("use a nonnegative unsigned 32-bit starting seed")
    source_root, run_dir = args.source_root.resolve(strict=True), args.run_dir.resolve(strict=True)
    core = diagnostic.core_identity(source_root, args.source_commit)
    paired_files = ["10_jax_exploration_platform/scripts/atari_pixel_campaign/" + name
                    for name in ("paired_benchmark.py", "ideas.py")]
    paired = training.source_receipt(source_root, args.source_commit, paired_files)
    selected, selected_input = read_document(args.configuration)
    if args.copies is not None:
        if args.mode != "packing-point" or type(args.copies) is not int or args.copies < 1:
            raise ValueError("--copies only changes the independent copy count of a packing point")
        selected = {**selected, "learner": {**selected["learner"], "n_copies": args.copies}}
    selected = validation.configuration(selected)
    validate_complete_fields(source_root, selected)
    reference, reference_input = None, None
    if args.mode != "packing-point":
        if args.reference_configuration is None:
            raise ValueError("a comparison requires an explicit full reference configuration")
        reference, reference_input = read_document(args.reference_configuration)
        reference = validation.configuration(reference)
        validate_complete_fields(source_root, reference)
        if reference["learner"]["n_copies"] != selected["learner"]["n_copies"] or reference == selected and not (qualification or direct_qualification):
            raise ValueError("comparison variants need the same copy count and a distinct implemented setting")
        validate_correctness_protocol([reference, selected])
    elif args.reference_configuration is not None:
        raise ValueError("an isolated packing point must not retain a reference variant")
    runtime, runtime_input = read_document(args.runtime_environment)
    runtime = validation.runtime_environment(runtime)
    completed_study, completed_mapping = completed_study_inputs(args, args.source_commit, runtime, selected)
    assets, asset_input = read_document(args.asset_record)
    if set(assets) != {"root", "manifest", "sha256"}:
        raise ValueError("asset_record requires the immutable asset root, manifest path and manifest hash")
    copies = selected["learner"]["n_copies"]
    if args.seed_start + copies > 2 ** 32:
        raise ValueError("the independent policy seed range exceeds unsigned 32-bit values")
    helpers = helper_receipt(configured, False, True) if common else (helper_receipt(configured, True) if direct else helper_receipt(configured))
    if final or args.mode == "learning-check":
        validation.committed_measurement_sources(helpers, validation.Evidence())
    configurations = [selected] + ([reference] if reference is not None else [])
    window = measurement_window(configurations, minimum)
    learning_updates = measurement_window(configurations, 128) if args.mode == "learning-check" else None
    directory = run_dir / "data" / args.unit_id
    plan = {"schema_version": 1, "kind": "selected_configuration_measurement", "mode": args.mode,
        "source_root": str(source_root), "source_commit": args.source_commit, "core": core,
        "paired_source": paired, "helpers": helpers, "configuration": selected,
        "reference_configuration": reference, "runtime_environment": runtime, "asset_record": assets,
        "jaxtari_source_root": str(args.jaxtari_source_root.resolve()) if args.jaxtari_source_root is not None else None,
        "hardware_class": args.hardware_class, "gpu_uuid": args.gpu_uuid,
        "copy_seeds": list(range(args.seed_start, args.seed_start + copies)),
        "rounds": rounds, "minimum_updates_per_round": minimum, "iterations": window,
        "window_protocol": "complete groups/report windows; requested minimum and actual count recorded",
        "inputs": {"configuration": selected_input, "reference_configuration": reference_input,
                   "runtime_environment": runtime_input, "asset_record": asset_input},
        "campaign_run_dir": str(run_dir), "unit_id": args.unit_id,
        "python": str(args.python.absolute()),
        "compilation_cache_policy": "disabled", "hypothesis_count_contribution": 0,
        "selected_by_producer": False, "jobs_submitted_by_producer": False}
    if configured:
        import configured_measurement as support
        profile, profile_input = read_document(args.configured_isolation_profile)
        support.configure_plan(plan, profile, "selected_measurement")
        plan["inputs"]["configured_isolation_profile"] = profile_input
        if qualification:
            support.validate_original_qualification(plan)
            plan["target_qualification"] = support.QUALIFICATION
            support.qualification_sources(plan, args.cleanrl_source_root, args.qualification_software_result)
        else:
            support.prepare_qualification_inputs(plan, args.target_qualification_record, args.target_qualification_review)
    if common:
        import common_confirmation as prerequisite
        plan["common_correctness_inputs"] = {"result": prerequisite.reference(correctness_record),
            "review": prerequisite.reference(correctness_review)}
        prerequisite.request(plan)
    if direct:
        import direct_measurement as support
        profile, profile_input = read_document(args.direct_binding_profile)
        support.configure_plan(plan, profile, args.direct_history_inputs, "selected_measurement")
        plan["inputs"]["direct_binding_profile"] = profile_input
        plan["inputs"]["direct_history_inputs"] = plan["direct_binding"]["frozen_inputs"]
        if direct_qualification:
            support.validate_original_qualification(plan)
            plan["direct_qualification"] = support.QUALIFICATION
            support.qualification_sources(plan, args.cleanrl_source_root, args.qualification_software_result)
        else:
            support.prepare_qualification_inputs(plan, args.direct_qualification_record, args.direct_qualification_review)
    if getattr(args, 'record_device_memory_limit', False):
        # This diagnostic capability is explicitly separate from ordinary completed packing evidence.
        import device_memory_limit as capacity
        if (args.mode != 'packing-point' or args.memory_limit_allocation_kind not in {'direct', 'slurm'}
                or minimum != 10):
            raise ValueError('instrumented device-memory limits require an explicit isolated packing allocation')
        ledger_source = capacity.file_receipt(args.memory_limit_owned_job_ledger) if args.memory_limit_owned_job_ledger else None
        if ledger_source and ledger_source['path'] != str(run_dir / 'slurm' / ('submitted_jobids_' + args.memory_limit_cluster + '.txt')):
            raise ValueError('the instrumented allocation ledger must be this campaign and cluster exact ledger')
        ledger_bytes = Path(ledger_source['path']).read_bytes() if ledger_source else b''
        if ledger_source and hashlib.sha256(ledger_bytes).hexdigest() != ledger_source['sha256']:
            raise ValueError('the owned job ledger changed during immutable snapshot preparation')
        ledger = {'path': str(directory / 'capacity_owned_job_ledger.txt'), 'sha256': ledger_source['sha256']} if ledger_source else None
        plan['device_memory_limit_protocol'] = capacity.protocol({**capacity.PROTOCOL,
            'allocation_kind': args.memory_limit_allocation_kind, 'cluster': args.memory_limit_cluster,
            'owned_job_ledger': ledger})
        plan['device_memory_limit_ledger_source'] = ledger_source
        if getattr(args, 'memory_limit_ownership_profile', None) is not None:
            if args.memory_limit_ownership_profile != capacity.UVA_CGROUP_V2_OWNERSHIP_PROFILE['kind']:
                raise ValueError('unknown explicit instrumented ownership profile')
            plan['ownership_profile'] = dict(capacity.UVA_CGROUP_V2_OWNERSHIP_PROFILE)
            capacity.selected_ownership_profile(plan)
            capacity.require_ownership_source(plan)
    if final:
        plan["final_confirmation_protocol"] = validation.FINAL_PROTOCOL
    if completed_study is not None:
        plan["completed_study"] = completed_study
        plan["completed_study_manifest"] = completed_mapping
    if final or args.mode == "learning-check":
        source = SCRIPTS / "short_learning.py"
        raw = source.read_bytes()
        relative = "10_jax_exploration_platform/scripts/atari_pixel_campaign/short_learning.py"
        if hashlib.sha256(raw).hexdigest() != helpers["files"][relative]:
            raise ValueError("the prepared short-learning helper changed its exact source bytes")
        contract = importlib.util.module_from_spec(importlib.util.spec_from_file_location("prepared_short_learning", source))
        exec(compile(raw, str(source), "exec"), contract.__dict__)
        if final:
            plan["single_update_latency_protocol"] = {arm: contract.latency_protocol(value)
                for arm, value in (("reference", reference), ("candidate", selected))}
        if args.mode == "learning-check":
            plan["learning_protocol"] = contract.protocol(configurations)
            if plan["learning_protocol"]["updates"] != learning_updates:
                raise ValueError("short learning preparation changed its complete-window budget")
    artifacts = {directory / "measurement_plan.json": json_bytes(plan),
        directory / "manifest.yaml": json_bytes(plan), directory / "config_resolved.yaml": json_bytes({
            "reference": reference, "candidate": selected, "runtime_environment": runtime}),
        directory / "command.txt": (shlex.join([str(args.python), str(directory / "code/measure_selected.py"),
                                              "--prepared-unit", str(directory)]) + "\n").encode()}
    for name in required_helpers(configured, direct, common):
        artifacts[directory / "code" / name] = (SCRIPTS / name).read_bytes()
    if 'device_memory_limit_protocol' in plan:
        artifacts[directory / 'capacity_owned_job_ledger.txt'] = ledger_bytes
    background = ("## Purpose\n\nMeasure an explicitly supplied combined configuration or one isolated policy-copy count. "
        "The producer chooses no hypotheses or configurations and submits no jobs.\n\n"
        "## Key hyperparameters\n\nOriginal CleanRL policy optimization with Random Network Distillation: "
        "128 pixel environments per policy, 128 rollout steps, four epochs, four minibatches, fifty random warmup rollouts. " +
        (f"Independent learning arms each retain {learning_updates} complete original updates and all diagnostic/report rows. "
         if args.mode == "learning-check" else f"{rounds} timing rounds contain {window} complete updates each (requested minimum {minimum}). ") +
        "The correctness-only mode performs the supplied-input and complete-iteration gates without timing rounds. "
        "Neural and optimizer arithmetic uses float32; running statistics use float64; matrix precision is highest.\n\n"
        "## Code and config changes\n\nThe exact frozen numerical core and Executor are imported after source/runtime/assets "
        "validation and card admission. Packing holds only one variant in a fresh process and samples actual device "
        "used/free/total memory as well as process memory. No checkpoint is saved.\n\n"
        f"## Git state\n\nNumerical source: `{args.source_commit}`. Helper source: `{helpers['source_commit']}`. "
        "Original source/helper hashes, full configurations and input receipts are recorded in measurement_plan.json.\n")
    if qualification:
        background += ("\n## Separate target qualification\n\nThis explicit same-original path runs every unchanged learner/environment pytest gate, "
            "all original renderer fixtures, fifty full normalization warmup rollouts and one full reference update. "
            "It requires no accepted common baseline, completed-fifty proof or existing target admission. "
            "Its distinct target_qualification_result.json grants zero scientific, throughput, packing, case or idea credit. "
            "Controlled agent transitions and updates are counted separately from unquantified synthetic/partial test-fixture work; "
            "no whole-job transition total is claimed. Actual independent review remains required before any subsequent use.\n")
    artifacts[directory / "experiment_background.md"] = background.encode()
    for path, contents in artifacts.items():
        if path.exists() and path.read_bytes() != contents:
            raise ValueError("an immutable measurement unit already has different contents: " + str(path))
    for path, contents in artifacts.items():
        write_once(path, contents)
    write_once(directory / "measurement_prepared.json", json_bytes({"plan_sha256": hashlib.sha256(json_bytes(plan)).hexdigest()}))
    return directory


def selected_query(gpu_uuid, fields, *, processes=False):
    """Read only this physical GPU through the specified NVML query."""
    option = "--query-compute-apps=" if processes else "--query-gpu="
    result = subprocess.run(["nvidia-smi", "-i", gpu_uuid, option + fields, "--format=csv,noheader,nounits"],
                            capture_output=True, text=True, check=True, timeout=10)
    return list(csv.reader(result.stdout.splitlines()))


def allocated_gpu_uuid(context, environment):
    """Apply launch_queue's physical-minor proof to one Slurm-visible study GPU."""
    visible = environment.get("CUDA_VISIBLE_DEVICES", "")
    allocations = [environment[name] for name in ("SLURM_JOB_GPUS", "SLURM_STEP_GPUS")
                   if environment.get(name)]
    if not visible or "," in visible or not allocations or any("," in value for value in allocations):
        raise ValueError("one CUDA-visible GPU and one recorded Slurm GPU allocation are required")
    if any(not (value.isdecimal() or value.startswith("GPU-")) for value in allocations):
        raise ValueError("Slurm must identify the single allocated GPU by physical index or UUID")
    result = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid,name", "--format=csv,noheader"],
                            capture_output=True, text=True, check=True, timeout=15)
    rows = list(csv.reader(result.stdout.splitlines()))
    if len(rows) != 1 or len(rows[0]) != 3:
        raise ValueError("nvidia-smi did not resolve exactly the allocated GPU")
    index, gpu, name = [part.strip() for part in rows[0]]
    expected_name = "H100" if context == "h100" else "A4500"
    if not gpu.startswith("GPU-") or expected_name not in name:
        raise ValueError("the visible GPU differs from the measurement hardware class")
    result = subprocess.run(["nvidia-smi", "-q", "-x"], capture_output=True, text=True, check=True, timeout=15)
    devices = ET.fromstring(result.stdout).findall("gpu")
    if (len(devices) != 1 or devices[0].findtext("uuid") != gpu
            or devices[0].findtext("product_name") != name):
        raise ValueError("nvidia-smi's visible row and physical device report disagree")
    minor = devices[0].findtext("minor_number", "")
    if not index.isdecimal() or not minor.isdecimal():
        raise ValueError("the visible GPU must record its numeric index and physical device minor")
    for allocated in allocations:
        if allocated.isdecimal() and int(allocated) != int(minor):
            raise ValueError("Slurm's physical GPU allocation conflicts with the visible device minor")
        if allocated.startswith("GPU-") and gpu != allocated:
            raise ValueError("the visible GPU UUID differs from the Slurm allocation")
    if visible not in {"0", index, minor, gpu}:
        raise ValueError("CUDA visibility conflicts with the physical GPU recorded by Slurm")
    return gpu


def preflight(plan):
    """Verify exact visibility, physical allocation and an unused correctly configured card before JAX."""
    gpu, context = plan["gpu_uuid"], plan["hardware_class"]
    host = socket.gethostname().split(".")[0]
    job = os.environ.get("SLURM_JOB_ID", "")
    original_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not job and host != "serval02":
        raise ValueError("numerical measurements require a compute allocation or an authorized H100 profiling host")
    if job:
        if not job.isdecimal() or int(job) < 1:
            raise ValueError("a measurement requires an actual positive Slurm allocation identifier")
        if allocated_gpu_uuid(context, os.environ) != gpu:
            raise ValueError("the selected physical GPU differs from the Slurm allocation")
        # The physical allocation is proved before changing CUDA's remapped ordinal.
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    elif original_visible != gpu:
        raise ValueError("CUDA_VISIBLE_DEVICES must name exactly the recorded physical GPU UUID")
    rows = selected_query(gpu, "uuid,name,memory.total,driver_version,mig.mode.current,memory.used")
    if len(rows) != 1 or len(rows[0]) != 6:
        raise ValueError("the selected GPU query was ambiguous")
    actual, name, total, driver, mig, used = [part.strip() for part in rows[0]]
    if actual != gpu or ("H100" not in name if context == "h100" else "A4500" not in name or host != "jaguar03"):
        raise ValueError("the selected physical GPU differs from this measurement context")
    if (("H100" in name and mig != "Disabled") or mig not in {"Disabled", "[N/A]"}
            or int(used) >= 1024 or selected_query(gpu, "pid", processes=True)):
        raise ValueError("the selected card is not free in its required GPU mode")
    if "H100" in name:
        confidential = subprocess.check_output(["nvidia-smi", "conf-compute", "-f", "-i", gpu], text=True)
        if "CC status: OFF" not in confidential:
            raise ValueError("the selected H100 must have confidential computing disabled")
    return {"hostname": host, "gpu_uuid": gpu, "gpu_query": f"{gpu}, {name}, {total} MiB, {driver}",
            "pid": os.getpid(), "process_start_time": Path(f"/proc/{os.getpid()}/stat").read_text().rsplit(")", 1)[1].split()[19],
            "slurm_job_id": job, "cuda_visible_devices_original": original_visible,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "")}


@contextmanager
def card_admission(directory, plan):
    """Hold the standing card lock per direct-host unit without creating a waiting CUDA context."""
    host = socket.gethostname().split(".")[0]
    if "direct_binding" in plan:
        support = producer_support(plan); identity = support.core()
        identity.validate_history(plan)
        if host != "serval02" or os.getuid() != 2618919:
            raise ValueError("direct descriptor admission requires the exact approved host and owner")
        pinned = os.open(identity.PROFILE["standing_lock"], os.O_PATH | os.O_NOFOLLOW)
        try:
            observed = os.fstat(pinned); profile = identity.PROFILE
            if (not stat.S_ISREG(observed.st_mode) or observed.st_uid != profile["uid"]
                    or observed.st_dev != profile["lock_device"] or observed.st_ino != profile["lock_inode"]
                    or stat.S_IMODE(observed.st_mode) != profile["lock_mode"]):
                raise ValueError("the existing standing-lock inode or owner changed")
            # This intentional kernel descriptor alias opens only the proved regular inode.
            descriptor = os.open("/proc/self/fd/" + str(pinned), os.O_RDWR)
        finally:
            os.close(pinned)
        with os.fdopen(descriptor, "r+") as stream:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            identity.validate_lock(identity.lock_metadata(descriptor), os.getpid())
            yield stream
        return
    path = None
    if host == "serval05" or any(word in host for word in ("portal", "login")):
        raise ValueError("selected measurements require serval02 card 1 or a Slurm compute host")
    if host == "serval02":
        if plan["gpu_uuid"] != CARD1:
            raise ValueError("serval02 measurements may use only the recorded card-1 UUID")
        path = Path("/tmp/sl5nw_serval02_card1.lock")
    if path is None:
        yield
        return
    diagnostic.record_progress(directory, "waiting_for_standing_lock", gpu_uuid=plan["gpu_uuid"])
    with path.open("a+") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        yield


class MemoryObservation:
    """Sample the selected device's real free/used/total bytes and its compute processes."""

    def __init__(self, device, gpu_uuid):
        """Keep the already initialized device and exact physical UUID."""
        self.device, self.uuid = device, gpu_uuid
        self.stop = threading.Event()
        self.thread, self.samples, self.errors = None, [], []

    def sample(self):
        """Read device-wide memory and per-process allocations at the same observation point."""
        device_rows = selected_query(self.uuid, "uuid,memory.total,memory.used,memory.free")
        if len(device_rows) != 1 or len(device_rows[0]) != 4:
            raise RuntimeError("device memory sampling did not identify one physical GPU")
        gpu, total, used, free = [part.strip() for part in device_rows[0]]
        if gpu != self.uuid:
            raise RuntimeError("device memory sampling returned another GPU UUID")
        processes = []
        for row in selected_query(self.uuid, "pid,gpu_uuid,used_gpu_memory", processes=True):
            if len(row) != 3 or not row[0].strip().isdigit() or not row[2].strip().isdigit() or row[1].strip() != self.uuid:
                raise RuntimeError("process-memory sampling returned an invalid selected-GPU row")
            processes.append({"pid": int(row[0]), "bytes": int(row[2]) * 2 ** 20})
        return {"timestamp_seconds": time.time(), "gpu_uuid": gpu,
                "total_bytes": int(total) * 2 ** 20, "used_bytes": int(used) * 2 ** 20,
                "free_bytes": int(free) * 2 ** 20, "processes": processes}

    def _sample(self):
        """Collect repeated observations without changing or stopping another GPU process."""
        while not self.stop.wait(.2):
            try:
                self.samples.append(self.sample())
            except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
                self.errors.append(repr(error))
                return

    def begin(self):
        """Take the initial memory sample outside the measured training interval."""
        self.samples, self.errors = [self.sample()], []
        self.stop.clear()
        self.thread = threading.Thread(target=self._sample, daemon=True)
        self.thread.start()

    def finish(self):
        """Join sampling and preserve actual device samples plus the allocator's recorded peak."""
        self.stop.set()
        self.thread.join()
        self.samples.append(self.sample())
        if self.errors:
            raise RuntimeError("device memory sampling failed: " + "; ".join(self.errors))
        own = [row["bytes"] for sample in self.samples for row in sample["processes"] if row["pid"] == os.getpid()]
        if not own:
            raise RuntimeError("the selected GPU did not report this process during the measurement")
        allocator = self.device.memory_stats()
        if not allocator or "peak_bytes_in_use" not in allocator:
            raise RuntimeError("the GPU allocator did not expose an actual peak")
        return {"gpu_uuid": self.uuid, "sampled_process_peak_bytes": max(own),
            "process_memory_samples": len(self.samples), "sampling_interval_seconds": .2,
            "allocator": {key: int(value) for key, value in allocator.items()},
            "other_gpu_processes": sorted({row["pid"] for sample in self.samples for row in sample["processes"] if row["pid"] != os.getpid()}),
            "device_memory_samples": [{key: value for key, value in sample.items() if key != "processes"} for sample in self.samples],
            "peak_scope": "one resident variant; sampled actual device memory and lifetime allocator high-water mark"}


def verify_prepared(directory, plan):
    """Check staged helper bytes and unchanged numerical source before any device import."""
    for name in required_helpers("configured_isolation" in plan, "direct_binding" in plan, "common_correctness_inputs" in plan):
        relative = "10_jax_exploration_platform/scripts/atari_pixel_campaign/" + name
        expected = plan["helpers"]["files"][relative]
        if training.file_sha256(directory / "code" / name) != expected:
            raise ValueError("a staged measurement helper changed: " + name)
    for name, module in (("diagnose_gpu_execution.py", diagnostic), ("post_comparison_validation.py", validation),
                         ("train_optimized.py", training)):
        if Path(module.__file__).resolve() != directory / "code" / name:
            raise ValueError("execute the prepared unit in a fresh process with its staged helpers")
    if diagnostic.core_identity(Path(plan["source_root"]), plan["source_commit"]) != plan["core"]:
        raise ValueError("the numerical core changed after measurement preparation")
    if training.source_receipt(Path(plan["source_root"]), plan["source_commit"],
                               list(plan["paired_source"]["files"])) != plan["paired_source"]:
        raise ValueError("the frozen Executor changed after preparation")
    if 'ownership_profile' in plan:
        import device_memory_limit as capacity
        capacity.require_ownership_source(plan)
    producer_support(plan)
    verify_completed_study_plan(plan)
    if "completed_study" in plan:
        validation.committed_measurement_sources(plan["helpers"], validation.Evidence())


def run_prepared(directory):
    """Run one recorded measurement on the compute host, retaining failures without overwriting a prior result."""
    directory = Path(directory).resolve(strict=True)
    plan = json.loads((directory / "measurement_plan.json").read_text())
    ready = json.loads((directory / "measurement_prepared.json").read_text())
    if ready.get("plan_sha256") != hashlib.sha256((directory / "measurement_plan.json").read_bytes()).hexdigest():
        raise ValueError("the prepared measurement plan changed")
    if Path(__file__).resolve() != directory / "code/measure_selected.py":
        raise ValueError("execute the immutable staged producer recorded in command.txt")
    if str(Path(sys.executable).absolute()) != plan["python"]:
        raise ValueError("execute the exact recorded Python environment")
    if directory != Path(plan["campaign_run_dir"]) / "data" / plan["unit_id"]:
        raise ValueError("measurement output directory differs from its original immutable plan")
    if any((directory / name).exists() for name in ('result.json', 'failure.json', 'instrumented_completion.json', 'target_qualification_result.json', 'direct_qualification_result.json')):
        raise ValueError("retain the previous result/failure and use a new measurement unit")
    with (directory / "measurement.lock").open("a+") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        if any((directory / name).exists() for name in ('result.json', 'failure.json', 'instrumented_completion.json', 'target_qualification_result.json', 'direct_qualification_result.json')):
            raise ValueError("retain the previous result/failure and use a new measurement unit")
        verify_prepared(directory, plan)
        if "direct_binding" in plan and any((directory / name).exists() for name in (
                "direct_process_metadata.json", "direct_device_binding.json", "direct_device_binding_attempt.json", "direct_startup_checks.json")):
            raise ValueError("preserve the previous direct attempt and use a new immutable unit before numerical import")
        for name, expected in (("XLA_PYTHON_CLIENT_PREALLOCATE", "false"), ("JAXTARI_SKIP_SPRITE_UPDATE", "1")):
            if os.environ.get(name, expected) != expected:
                raise ValueError("measurement runtime differs from the frozen worker setting: " + name)
            os.environ[name] = expected
        runtime = training.enforce_runtime(plan["runtime_environment"], os.environ)
        try:
            common = common_confirmation_support(plan)
            if common is not None:
                common.before_compute(directory, plan)
            with card_admission(directory, plan) as held:
                support = producer_support(plan)
                direct = "direct_binding" in plan
                if direct:
                    metadata_record = support.core().collect_metadata(plan, directory, held.fileno())
                    hardware = {"direct_process_metadata": metadata_record, "runtime_environment": runtime}
                    if "direct_qualification" not in plan:
                        support.verify_qualification_inputs(plan)
                elif support is not None:
                    isolation_metadata = support.core().collect_metadata(plan, directory)
                    hardware = {"configured_isolation_metadata": isolation_metadata, "runtime_environment": runtime}
                    if "target_qualification" not in plan:
                        support.verify_qualification_inputs(plan)
                else:
                    hardware = {**preflight(plan), "runtime_environment": runtime}
                assets = training.verify_assets(plan["asset_record"], os.environ)
                jaxtari = training.jaxtari_python_receipt(plan.get("jaxtari_source_root"))
                libraries = training.installed_gpu_library_receipts()
                if support is None:
                    training.atomic_json(directory / "hardware.json", hardware)
                training.atomic_json(directory / "asset_receipt.json", assets)
                backend = installed_backend_receipt(jaxtari, libraries)
                training.atomic_json(directory / "installed_backend.json", backend)
                if support is not None and ("target_qualification" in plan or "direct_qualification" in plan):
                    support.verify_qualification_backend(plan, backend)
                if 'device_memory_limit_protocol' in plan:
                    import device_memory_limit as capacity
                    capacity.protocol(plan['device_memory_limit_protocol'])
                    capture = capacity.Capture(directory, plan, hardware, backend, MemoryObservation, write_once, json_bytes)
                    try:
                        _compute(directory, plan, hardware, backend, failure_capture=capture)
                    except BaseException as error:
                        # Preserve the original failure even when the extra diagnostic is unsupported or fails.
                        try:
                            capture.failure(error)
                        except BaseException as diagnostic_error:
                            write_once(directory / 'capacity_capture_failure.json', json_bytes({
                                'error': repr(diagnostic_error), 'traceback': traceback.format_exc(),
                                'original_error': repr(error), 'capacity_credit': 0}))
                        raise
                    finally:
                        capture.close()
                else:
                    _compute(directory, plan, hardware, backend)
        except BaseException as error:
            diagnostic.atomic_json(directory / "failure.json", {"error": repr(error), "exception_type": type(error).__name__,
                "traceback": traceback.format_exc(), "source_commit": plan["source_commit"], "mode": plan["mode"],
                "failure_stage": "correctness" if type(error).__name__ == "CorrectnessMismatch" else "producer_exception"})
            diagnostic.record_progress(directory, "failed", error=repr(error))
            raise


def verify_imported_sources(root, paired):
    """Bind actual imported helpers and JAXtari to the source bytes checked before initialization."""
    for name, module in (("paired_benchmark", paired), ("ideas", sys.modules.get("ideas"))):
        expected = root / "scripts/atari_pixel_campaign" / (name + ".py")
        if module is None or Path(module.__file__).resolve() != expected:
            raise ValueError("the imported numerical helper differs from its recorded source: " + name)
    jaxtari = sys.modules.get("jaxtari")
    expected = Path(metadata.distribution("jaxtari").locate_file("jaxtari/__init__.py")).resolve(strict=True)
    if jaxtari is None or Path(jaxtari.__file__).resolve() != expected:
        raise ValueError("the imported JAXtari package differs from the verified installed Python files")


def _compute(directory, plan, hardware, installed_backend, failure_capture=None):
    """Import the exact core only after card/source/runtime/assets checks, then execute the requested fixed protocol."""
    started_at = time.time()
    if "jax" in sys.modules:
        raise RuntimeError("JAX was imported before selected-measurement admission")
    root = Path(plan["source_root"]) / "10_jax_exploration_platform"
    sys.path.insert(0, str(root / "src"))
    sys.path.insert(0, str(root / "scripts/atari_pixel_campaign"))
    import jax
    import jaxtari
    from exploration_platform.atari_pixels.config import PixelPPOConfig
    from exploration_platform.atari_pixels.environment import AtariPixelEnvironment
    from exploration_platform.atari_pixels.training import PixelPPORND
    paired = importlib.import_module("paired_benchmark")
    verify_imported_sources(root, paired)
    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_default_matmul_precision", "highest")
    devices = jax.devices()
    if len(devices) != 1 or devices[0].platform != "gpu":
        raise ValueError("selected measurements require one visible compute GPU")
    expected = "H100" if plan["hardware_class"] == "h100" else "A4500"
    if expected not in devices[0].device_kind:
        raise ValueError("JAX selected another GPU type")
    support = producer_support(plan)
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
        training.atomic_json(directory / "hardware.json", hardware)
        initial_process = support.process_sample(binding["gpu_uuid"], os.getpid())
        training.atomic_json(directory / "assigned_process_admission.json", initial_process)
        support.validate_process_sample(initial_process, binding["gpu_uuid"], os.getpid())
    if failure_capture is not None:
        failure_capture.bind(devices[0])
    whole_scope = support.observed_work(devices[0], binding, directory / (plan["mode"] + "_memory")) if direct and "direct_qualification" not in plan else nullcontext({})
    with whole_scope as whole_memory:
        selected = plan["configuration"]
        reference = plan["reference_configuration"]
        seeds = tuple(plan["copy_seeds"])
        def progress(value):
            """Publish actual computation stages in the normal per-unit monitoring location."""
            diagnostic.record_progress(directory, value["state"], **{name: item for name, item in value.items() if name != "state"})
        if "target_qualification" in plan or "direct_qualification" in plan:
            with support.observed_work(devices[0], binding, directory / "qualification_memory") as observed:
                with support.paired_observer(paired, binding, directory) if direct else nullcontext():
                    result = support.run_qualification(directory, plan, paired, jax, PixelPPOConfig, PixelPPORND, AtariPixelEnvironment)
            result["qualification_memory"] = observed
        elif plan["mode"] == "learning-check":
            short_learning = staged_short_learning(directory, plan)
            result = short_learning.run(directory, plan, paired, PixelPPOConfig, PixelPPORND, AtariPixelEnvironment, progress, MemoryObservation)
        elif plan["mode"] == "confirmation":
            if "final_confirmation_protocol" in plan:
                short_learning = staged_short_learning(directory, plan)
                if plan.get("single_update_latency_protocol") != {arm: short_learning.latency_protocol(value)
                        for arm, value in (("reference", reference), ("candidate", selected))}:
                    raise ValueError("the predeclared selected reporting and per-update latency contract changed")
            changes = {scope: {name: value for name, value in selected[scope].items() if reference[scope][name] != value}
                       for scope in ("learner", "environment")}
            with support.paired_observer(paired, binding, directory) if support is not None else nullcontext():
                result = paired.run_paired_benchmark(PixelPPOConfig(**reference["learner"]),
                    changes["learner"], changes["environment"], selected["execution"],
                    rounds=plan["rounds"], iterations=plan["iterations"], seed_start=seeds[0], progress=progress,
                    reference_environment=reference["environment"], reference_execution=reference["execution"])
            if "final_confirmation_protocol" in plan:
                short_learning = staged_short_learning(directory, plan)
                latency_memory = MemoryObservation(devices[0], plan["gpu_uuid"])
                latency_memory.begin()
                try:
                    result["synchronized_per_update_latency"] = short_learning.single_update_latencies(
                        paired, selected, reference, seeds, PixelPPOConfig, PixelPPORND, AtariPixelEnvironment)
                finally:
                    observed = latency_memory.finish()
                    training.atomic_json(directory / "single_update_latency_memory.json", observed)
                if observed["other_gpu_processes"]:
                    raise ValueError("another process occupied the GPU during synchronized single-update latency")
                result["synchronized_per_update_latency_memory"] = observed
        else:
            configs = [("candidate", selected)] if plan["mode"] == "packing-point" else [("reference", reference), ("candidate", selected)]
            executors, initial = {}, {}
            for label, config in configs:
                progress({"state": "initializing", "variant": label})
                trainer = PixelPPORND(PixelPPOConfig(**config["learner"]), AtariPixelEnvironment(**config["environment"]))
                executor = paired.Executor(trainer, config["execution"])
                progress({"state": "normalization_warmup", "variant": label})
                state = trainer.warmup(trainer.initialize(seeds))
                initial[label] = paired.host_copy(paired.canonical_training_state(executor, state)) if plan["mode"] == "correctness" else paired.host_copy(state)
                executors[label] = executor
                del state
            if plan["mode"] == "correctness":
                initialized = paired.agreement(initial["reference"], initial["candidate"])
                checks, unused = paired.correctness_checks(executors["reference"], executors["candidate"], initial["reference"])
                del unused
                result = {"correctness_passed": True, "correctness_attempted": True, "implementation_distinct": True,
                          "correctness": {"initialization": initialized, **checks}, "reference": reference, "candidate": selected,
                          "copies": len(seeds), "measurement_valid": False, "decision": "correctness_passed"}
            else:
                executor = executors["candidate"]
                progress({"state": "compile_warmup"})
                host, warmup = paired.warm_executor(executor, initial["candidate"], plan["iterations"])
                training_origin = int(host.iteration)
                memory = MemoryObservation(devices[0], plan["gpu_uuid"])
                measured = []
                for index in range(7):
                    progress({"state": "running", "variant": "candidate", "round": index + 1, "total_rounds": 7})
                    row, host = paired.measure_round(executor, host, plan["iterations"], memory)
                    measured.append(row)
                    diagnostic.atomic_json(directory / "rounds.json", measured)
                if any(row["memory"]["other_gpu_processes"] for row in measured):
                    raise RuntimeError("another process occupied the measured GPU during the isolated packing point")
                result = {"completed": True, "isolated_variant": True, "configuration": selected,
                    "normalization_warmup_iterations": 50, "copies": len(seeds), "copy_seeds": list(seeds),
                    "warmup": warmup, "rounds": measured, "hardware": hardware,
                    "training_start_iteration": training_origin,
                    "median_transitions_per_second": statistics.median(row["transitions_per_second_total"] for row in measured)}
        result.update(source_commit=plan["source_commit"], hardware_class=plan["hardware_class"],
            hardware=hardware,
            installed_backend=installed_backend,
            loaded_gpu_libraries=training.actual_gpu_library_receipt(devices[0].client, installed_backend["gpu_libraries"]),
            runtime_environment=plan["runtime_environment"], measurement_mode=plan["mode"],
            helpers=plan["helpers"], copy_seeds=list(seeds), protocol={"rounds": plan["rounds"],
                "requested_minimum_updates_per_round": plan["minimum_updates_per_round"], "actual_updates_per_round": plan["iterations"]},
            execution_identity={"hostname": hardware["hostname"], "pid": os.getpid(), "process_start_time": hardware["process_start_time"],
                                "started_at_seconds": started_at, "finished_at_seconds": time.time()})
        if direct:
            result.update(direct_binding=plan["direct_binding"], resolved_device_binding=binding,
                direct_binding_record=True, numerical_module_paths=numerical_paths, **support.BOUNDARY)
            if "direct_qualification" not in plan:
                result["direct_qualification_inputs"] = plan["direct_qualification_inputs"]
        elif support is not None:
            result.update(configured_isolation=plan["configured_isolation"], resolved_device_binding=binding,
                configured_isolation_record=True, numerical_module_paths=numerical_paths, memory_note=("Complete own-only active compute rows on the bound assigned UUID, "
                    "actual assigned-device memory samples and separate JAX lifetime allocator peak; "
                    "no claim of continuous kernel enforcement or absence of graphics/system activity"), **support.BOUNDARY)
            if "target_qualification" not in plan:
                result["target_qualification_inputs"] = plan["target_qualification_inputs"]
    if direct and "direct_qualification" not in plan:
        result["direct_whole_work_memory"] = whole_memory
    save_measurement_result(directory, plan, result)


def save_measurement_result(directory, plan, result):
    """Keep a completed instrumented attempt outside ordinary packing and throughput evidence."""
    common = common_confirmation_support(plan)
    if common is not None:
        result.update(common.result_binding(directory, plan))
    instrumented = 'device_memory_limit_protocol' in plan
    if instrumented:
        # The extra lifetime sampler changes the workload; retain raw observations and remeasure normally.
        result = {**result, 'kind': 'instrumented_device_memory_attempt_completion', 'measurement_valid': False,
            'packing_point_credit': 0, 'throughput_credit': 0, 'case_credit': 0, 'idea_credit': 0,
            'formal_packing_evidence': False, 'requires_uninstrumented_remeasurement': True}
    qualification = "target_qualification" in plan
    direct_qualification = "direct_qualification" in plan
    if qualification:
        result.update(kind="uva_configured_isolation_full_original_qualification", measurement_valid=False,
            scientific_credit=0, case_credit=0, idea_credit=0, packing_point_credit=0, throughput_credit=0, site_qualification_credit=0,
            worker_admitted=False, admission_granted=False)
    if direct_qualification:
        import direct_measurement as support
        result.update(kind="direct_serval02_card1_full_original_qualification", measurement_valid=False, **support.BOUNDARY)
    filename = ('direct_qualification_result.json' if direct_qualification else 'target_qualification_result.json'
        if qualification else ('instrumented_completion.json' if instrumented else 'result.json'))
    diagnostic.atomic_json(directory / filename, result)
    diagnostic.record_progress(directory, 'completed_instrumented_attempt_requires_normal_remeasurement' if instrumented else 'completed',
        completed=True, mode=plan['mode'], decision=result.get('decision'), rounds=len(result.get('rounds', [])))


def parse_arguments(argv=None):
    """Expose explicit inputs and a metadata-only preparation path, never hidden algorithm overrides."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-unit", type=Path)
    parser.add_argument('--instrumented-child', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--record-device-memory-limit', action='store_true')
    parser.add_argument('--memory-limit-allocation-kind', choices=('direct', 'slurm'))
    parser.add_argument('--memory-limit-cluster', choices=('uva', 'aces'))
    parser.add_argument('--memory-limit-owned-job-ledger', type=Path)
    parser.add_argument('--memory-limit-ownership-profile', choices=('uva_slurm_cgroup_v2_process_uid_ownership',))
    parser.add_argument("--cleanrl-source-root", type=Path)
    parser.add_argument("--qualification-software-result", type=Path)
    parser.add_argument("--configured-isolation-profile", type=Path)
    parser.add_argument("--target-qualification", action="store_true")
    parser.add_argument("--target-qualification-record", type=Path)
    parser.add_argument("--target-qualification-review", type=Path)
    parser.add_argument("--common-correctness-record", type=Path)
    parser.add_argument("--common-correctness-review", type=Path)
    parser.add_argument("--direct-binding-profile", type=Path)
    parser.add_argument("--direct-history-inputs", type=Path)
    parser.add_argument("--direct-target-qualification", action="store_true")
    parser.add_argument("--direct-qualification-record", type=Path)
    parser.add_argument("--direct-qualification-review", type=Path)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--mode", choices=MODES)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--source-commit")
    parser.add_argument("--configuration", type=Path)
    parser.add_argument("--reference-configuration", type=Path)
    parser.add_argument("--runtime-environment", type=Path)
    parser.add_argument("--asset-record", type=Path)
    parser.add_argument("--jaxtari-source-root", type=Path)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--unit-id")
    parser.add_argument("--gpu-uuid")
    parser.add_argument("--hardware-class", choices=validation.CONTEXTS)
    parser.add_argument("--seed-start", type=int, default=1)
    parser.add_argument("--copies", type=int)
    parser.add_argument("--rounds", type=int)
    parser.add_argument("--minimum-updates-per-round", type=int)
    parser.add_argument("--final-confirmation", action="store_true")
    parser.add_argument("--completed-study-proof", type=Path)
    parser.add_argument("--completed-study-review", type=Path)
    parser.add_argument("--completed-study-manifest", type=Path)
    parser.add_argument("--completed-study-manifest-sha256")
    parser.add_argument("--python", type=Path, default=Path("/p/rlprojects/RND/.venvs/atari_pixels_jax/bin/python"))
    args = parser.parse_args(argv)
    memory_options = ('memory_limit_allocation_kind', 'memory_limit_cluster', 'memory_limit_owned_job_ledger', 'memory_limit_ownership_profile')
    if args.prepared_unit is not None:
        if args.prepare_only or args.target_qualification or args.direct_target_qualification or args.record_device_memory_limit or args.final_confirmation or args.rounds is not None or args.minimum_updates_per_round is not None or any(getattr(args, name) is not None for name in (
                "mode", "source_root", "source_commit", "configuration", "reference_configuration",
                "runtime_environment", "asset_record", "jaxtari_source_root", "run_dir", "unit_id", "gpu_uuid", "hardware_class", "copies",
                "completed_study_proof", "completed_study_review", "completed_study_manifest", "completed_study_manifest_sha256",
                "configured_isolation_profile", "target_qualification_record", "target_qualification_review", "cleanrl_source_root", "qualification_software_result",
                "direct_binding_profile", "direct_history_inputs", "direct_qualification_record", "direct_qualification_review",
                "common_correctness_record", "common_correctness_review", *memory_options)):
            parser.error("--prepared-unit executes its recorded plan and cannot take preparation overrides")
        if args.instrumented_child:
            # The private child entry is meaningful only inside this exact declared prepared protocol.
            import device_memory_limit as capacity
            try:
                plan = capacity.json_document((args.prepared_unit / 'measurement_plan.json').read_bytes())
                if "direct_binding" in plan:
                    raise ValueError("the direct binding profile cannot run an instrumented capacity child")
                capacity.protocol(plan.get('device_memory_limit_protocol'))
                capacity.selected_ownership_profile(plan)
                if plan.get('kind') != 'selected_configuration_measurement' or plan.get('mode') != 'packing-point':
                    raise ValueError('the prepared plan is not an instrumented isolated packing point')
            except (OSError, ValueError, TypeError, KeyError) as error:
                parser.error('--instrumented-child requires a prepared instrumented packing plan: ' + str(error))
        return args
    if args.instrumented_child:
        parser.error('--instrumented-child requires --prepared-unit and its declared instrumented protocol')
    common_options = (args.common_correctness_record, args.common_correctness_review)
    if any(value is not None for value in common_options) and (any(value is None for value in common_options)
            or args.configured_isolation_profile is None or args.direct_binding_profile is not None
            or args.mode != "confirmation" or args.final_confirmation or args.target_qualification):
        parser.error("reviewed common correctness requires only an intermediate configured confirmation")
    if not args.record_device_memory_limit and any(getattr(args, name) is not None for name in memory_options):
        parser.error('memory-limit preparation options require --record-device-memory-limit')
    if (args.cleanrl_source_root is not None or args.qualification_software_result is not None) and not (args.target_qualification or args.direct_target_qualification):
        parser.error("the original CleanRL source applies only to the separate target qualification")
    if args.configured_isolation_profile is None and (args.target_qualification or args.target_qualification_record or args.target_qualification_review):
        parser.error("target qualification options require an explicit configured-isolation profile")
    direct_options = (args.direct_history_inputs, args.direct_qualification_record, args.direct_qualification_review)
    if args.direct_binding_profile is None and (args.direct_target_qualification or any(value is not None for value in direct_options)):
        parser.error("direct options require the separate explicit direct binding profile")
    if args.direct_binding_profile is not None:
        if (args.configured_isolation_profile is not None or args.target_qualification or args.target_qualification_record
                or args.target_qualification_review or args.hardware_class != "h100" or args.gpu_uuid != CARD1
                or args.mode not in {"correctness", "confirmation"} or args.record_device_memory_limit or args.final_confirmation
                or args.direct_history_inputs is None or any(value is not None for value in (
                    args.completed_study_proof, args.completed_study_review, args.completed_study_manifest, args.completed_study_manifest_sha256))):
            parser.error("direct support requires only its exact known-UUID intermediate or standalone qualification protocol")
        if args.direct_target_qualification:
            if (args.mode != "correctness" or args.direct_qualification_record or args.direct_qualification_review
                    or args.cleanrl_source_root is None or args.qualification_software_result is None):
                parser.error("standalone direct qualification cannot depend on a previous direct qualification")
        elif not args.direct_qualification_record or not args.direct_qualification_review:
            parser.error("ordinary direct work requires exact actual qualification and independent review")
    if args.configured_isolation_profile is not None:
        if args.gpu_uuid is not None or args.record_device_memory_limit or args.final_confirmation or args.mode not in {"correctness", "confirmation"}:
            parser.error("configured isolation requires an unresolved intermediate comparison")
        if args.target_qualification:
            if args.mode != "correctness" or args.target_qualification_record or args.target_qualification_review or args.cleanrl_source_root is None or args.qualification_software_result is None:
                parser.error("the standalone qualification cannot depend on a prior target qualification")
        elif not args.target_qualification_record or not args.target_qualification_review:
            parser.error("configured measurements require the actual qualification and its independent review")
    required = ("mode", "source_root", "source_commit", "configuration", "runtime_environment", "asset_record",
                "run_dir", "unit_id", "hardware_class") + (() if args.configured_isolation_profile is not None else ("gpu_uuid",))
    if any(getattr(args, name) is None for name in required):
        parser.error("preparation needs explicit source, configs, runtime, assets, output and selected GPU")
    return args


if __name__ == "__main__":
    arguments = parse_arguments()
    if arguments.prepared_unit is not None:
        plan = json.loads((arguments.prepared_unit / 'measurement_plan.json').read_text())
        if "direct_binding" in plan:
            producer_support(plan)
        if 'device_memory_limit_protocol' in plan and not arguments.instrumented_child:
            import device_memory_limit as capacity
            result = capacity.run_child(arguments.prepared_unit, sys.executable)
            raise SystemExit(result if result >= 0 else 128 - result)
        run_prepared(arguments.prepared_unit)
    else:
        unit = prepare(arguments)
        print(json.dumps({"prepared": True, "unit_directory": str(unit)}, indent=2), flush=True)
        if not arguments.prepare_only:
            os.execv(str(arguments.python), [str(arguments.python), str(unit / "code/measure_selected.py"),
                                            "--prepared-unit", str(unit)])
