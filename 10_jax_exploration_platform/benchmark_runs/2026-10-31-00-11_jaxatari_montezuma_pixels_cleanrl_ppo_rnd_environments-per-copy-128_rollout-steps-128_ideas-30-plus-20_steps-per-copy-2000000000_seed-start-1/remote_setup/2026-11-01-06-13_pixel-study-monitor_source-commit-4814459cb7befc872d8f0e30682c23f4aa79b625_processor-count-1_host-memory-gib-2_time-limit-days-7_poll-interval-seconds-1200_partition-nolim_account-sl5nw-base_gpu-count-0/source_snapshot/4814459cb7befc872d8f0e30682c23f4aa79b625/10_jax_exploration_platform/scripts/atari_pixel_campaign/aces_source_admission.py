"""Read original source admission through exact copied files for ACES only.

Logical admission paths never change. The explicit portable manifest supplies
unchanged bytes and a real detached Git checkout; native software and packages
are deliberately outside this mapping and must be observed on ACES itself.
"""

import hashlib
import importlib.util
from pathlib import Path
import re
import sys


SOURCE_COMMIT = "d28e6735e79788f618acb0ebfde8609db1f776af"
CONTEXTS = ("h100", "jaguar03_rtx_a4500")


def load(name, *, canonical=False):
    """Load a pinned adjacent stdlib helper, rejecting an unrelated canonical resolver."""
    path = Path(__file__).resolve().with_name(name + ".py")
    # Evidence checks this canonical class; never substitute a caller's different checkout.
    key = name if canonical else "aces_source_admission_" + name
    if canonical and key in sys.modules:
        module = sys.modules[key]
        if Path(module.__file__).resolve() != path:
            raise ValueError("the portable resolver was imported from a different helper checkout")
        return module
    specification = importlib.util.spec_from_file_location(key, path)
    module = importlib.util.module_from_spec(specification)
    if canonical:
        sys.modules[key] = module
    specification.loader.exec_module(module)
    return module


def evidence(binding):
    """Require the explicit original campaign and immutable exact-path manifest receipt."""
    portable = load("portable_evidence", canonical=True)
    if (not isinstance(binding, dict) or set(binding) != {"schema_version", "kind", "campaign_run_dir", "evidence_manifest"}
            or type(binding["schema_version"]) is not int or binding["schema_version"] != 1
            or binding["kind"] != "aces_original_source_portable_admission"):
        raise ValueError("portable ACES admission needs its complete version-one binding")
    portable.original_path(binding["campaign_run_dir"])
    record = binding["evidence_manifest"]
    if (not isinstance(record, dict) or set(record) != {"path", "sha256"}
            or not isinstance(record["sha256"], str) or not re.fullmatch(r"[0-9a-f]{64}", record["sha256"])):
        raise ValueError("portable ACES admission needs an exact manifest path and SHA256")
    path = Path(portable.original_path(record["path"]))
    if path != path.resolve(strict=True) or not path.is_file():
        raise ValueError("the ACES evidence manifest must have an ordinary exact physical path")
    resolver = portable.ExactPathResolver(path, record["sha256"])
    return load("post_comparison_validation").Evidence(resolver)


def snapshot(binding):
    """Rebuild the original logical admission and complete numerical identity from mapped bytes."""
    proof = evidence(binding)
    history, retry = load("h100_nvl_slurm_evidence"), load("infrastructure_retry")
    run = Path(binding["campaign_run_dir"])
    sentinel = run / "SOURCE_AND_GATES_READY.json"
    ready = proof.read(sentinel)
    # Accept only the original common baseline; later execution baselines need separate qualification.
    counts = ready.get("matched_copy_counts", [])
    if (ready.get("authorized") is not True or ready.get("source_commit") != SOURCE_COMMIT
            or not isinstance(ready.get("contexts"), list) or len(ready["contexts"]) != 2
            or set(ready["contexts"]) != set(CONTEXTS) or set(ready.get("gate_records", {})) != set(CONTEXTS)
            or not counts or any(type(value) is not int or value < 1 for value in counts)
            or len(set(counts)) != len(counts) or 1 not in counts
            or ready.get("reference_environment", {}) != {} or ready.get("reference_execution", {}) != {}):
        raise ValueError("portable admission did not authorize the complete original common baseline")
    source = ready["source_root"]
    relatives = history.numerical_paths(source, SOURCE_COMMIT, proof)
    logical = retry.git_receipt(source, SOURCE_COMMIT, relatives, proof)
    configuration = history.configuration(logical, proof)
    reference_path = ready["reference_configuration_path"]
    if {**proof.read(reference_path), "n_copies": 1} != configuration["learner"]:
        raise ValueError("portable admission differs from the complete original learner defaults")
    paths = [sentinel, Path(reference_path), *(Path(ready["gate_records"][context]) for context in CONTEXTS)]
    if len(set(map(str, paths))) != 4:
        raise ValueError("portable admission requires four distinct original role paths")
    originals = [history.file_receipt(path, proof) for path in paths]
    checks = {}
    for context in CONTEXTS:
        gate = proof.read(ready["gate_records"][context])
        if (gate.get("passed") is not True or gate.get("source_commit") != SOURCE_COMMIT
                or gate.get("hardware_class") != context or set(gate.get("checks", {})) != {"learner", "renderer"}):
            raise ValueError("an original mapped hardware gate is incomplete or belongs to another context")
        checked = {}
        for kind in ("learner", "renderer"):
            saved = gate["checks"][kind]
            if not isinstance(saved, dict) or set(saved) not in ({"path", "sha256"}, {"path", "sha256", "command", "hardware"}):
                raise ValueError("an original gate needs its exact receipt and optional complete checked metadata")
            raw_receipt = {name: saved[name] for name in ("path", "sha256")}
            record = history.bound(raw_receipt, proof)
            command = record.get("command")
            if (not isinstance(command, list) or not command or not all(isinstance(value, str) for value in command)
                    or not Path(command[0]).is_absolute() or not re.fullmatch(r"python(?:3(?:\.\d+)?)?", Path(command[0]).name)):
                raise ValueError("an original mapped gate must invoke its exact recorded Python directly")
            # Before: original absolute test paths. After: unchanged command identities, explicitly mapped bytes.
            view = {"numerical_source": logical, "campaign_run_dir": str(run), "python": command[0]}
            checked[kind] = history.check_record(raw_receipt, context, kind, view, proof)
            if len(saved) == 4 and saved != checked[kind]:
                raise ValueError("the original checked gate metadata differs from its exact raw command or hardware")
        if context == "h100":
            checks = checked
    admission = {"source_root": source, "source_commit": SOURCE_COMMIT, "files": originals,
        "original_h100_checks": checks, "reference_configuration": originals[1]}
    physical = proof.git_root(source, SOURCE_COMMIT)
    numerical = retry.git_receipt(str(physical), SOURCE_COMMIT, relatives)
    proof.resolver.unchanged_manifest()
    return admission, numerical, configuration


def archive(directory, admission, binding):
    """Copy the four unchanged mapped admission files into the native prepared packet."""
    proof = evidence(binding)
    site = load("h100_nvl_slurm_site_validation")
    rows = []
    for index, original in enumerate(admission["files"]):
        contents = proof.file(original["path"])
        if hashlib.sha256(contents).hexdigest() != original["sha256"]:
            raise ValueError("a mapped original admission file changed before capture")
        target = site.admission_copy_path(directory, index, original)
        load("infrastructure_retry").write_once(target, contents)
        rows.append({"original": original, "captured": site.receipt(target)})
    result = {"schema_version": 1, "kind": "original_source_admission_bytes", "files": rows}
    site.validate_admission_archive(directory, admission, result)
    return result


def check(plan):
    """Reject any changed original admission, mapped source, configuration or operational Git root."""
    if Path(plan["campaign_run_dir"]).name != Path(plan["portable_admission"]["campaign_run_dir"]).name:
        raise ValueError("the portable original and native ACES campaign names differ")
    admission, numerical, configuration = snapshot(plan["portable_admission"])
    if (admission != plan["admission"] or numerical != plan["numerical_source"]
            or configuration != plan["configuration"]):
        raise ValueError("the portable original admission, source or configuration changed")
    return admission
