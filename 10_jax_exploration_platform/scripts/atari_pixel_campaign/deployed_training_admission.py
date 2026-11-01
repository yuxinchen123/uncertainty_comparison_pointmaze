"""Admit an explicit deployed training specification from rebuilt raw deployment evidence.

Logical paths remain part of the original training identity. Physical locations
are returned separately for later runner integration. This does not invoke the
original NVL training-admission path or start training. Real measured-evidence
rebuilding belongs on allocated compute; synthetic metadata tests use no arrays.
"""

import argparse
import ast
import hashlib
import json
from pathlib import Path, PurePosixPath
import subprocess

import portable_evidence
import post_comparison_validation as validation
import train_optimized as training
import validate_training_deployment as deployment


def explicit_path(value):
    """Require a canonical absolute original path without replacing its identity."""
    return portable_evidence.original_path(value)


def full_source_configuration(specification, evidence):
    """Verify actual detached source bytes and every learner/environment field without importing them."""
    root, commit = specification["source_root"], specification["source_commit"]
    physical = deployment.git_root(root, commit, evidence)
    module = Path(root) / "10_jax_exploration_platform/src/exploration_platform/atari_pixels"
    source = {}
    for name in ("config.py", "environment.py"):
        path = module / name
        contents = evidence.file(path)
        relative = str(path.relative_to(root))
        original = subprocess.check_output(["git", "-C", str(physical), "show", commit + ":" + relative])
        if contents != original:
            raise ValueError("operational source differs from its pinned Git blob: " + str(path))
        source[name] = ast.parse(contents)
    config = next(item for item in source["config.py"].body if isinstance(item, ast.ClassDef) and item.name == "PixelPPOConfig")
    fields = {item.target.id for item in config.body if isinstance(item, ast.AnnAssign) and item.value is not None}
    environment = next(item for item in source["environment.py"].body if isinstance(item, ast.ClassDef) and item.name == "AtariPixelEnvironment")
    constructor = next(item for item in environment.body if isinstance(item, ast.FunctionDef) and item.name == "__init__")
    arguments = {item.arg for item in constructor.args.args if item.arg != "self"}
    if set(specification["configuration"]["learner"]) != fields or set(specification["configuration"]["environment"]) != arguments:
        raise ValueError("operational training must explicitly resolve every pinned learner/environment field")
    return str(physical.resolve(strict=True))


def training_assets(record, expected, evidence):
    """Match only immutable manifest metadata; the numerical runner still checks actual asset files."""
    if not isinstance(record, dict) or set(record) != {"root", "manifest", "sha256"}:
        raise ValueError("operational training requires its explicit root/manifest/SHA256 asset record")
    root, manifest_path = explicit_path(record["root"]), explicit_path(record["manifest"])
    if Path(manifest_path).is_relative_to(root):
        raise ValueError("operational asset manifest must remain outside the explicit asset root")
    contents = evidence.file(manifest_path)
    if hashlib.sha256(contents).hexdigest() != deployment.sha256(record["sha256"]):
        raise ValueError("operational asset manifest differs from its required SHA256")
    manifest = portable_evidence.document(contents)
    if (not isinstance(manifest, dict) or set(manifest) != {"schema_version", "source_commit", "files"}
            or type(manifest["schema_version"]) is not int or manifest["schema_version"] != 1
            or manifest["source_commit"] != deployment.JAXTARI_COMMIT
            or not isinstance(manifest["files"], dict) or not manifest["files"]):
        raise ValueError("operational asset manifest lacks the complete pinned-source file map")
    for name, digest in manifest["files"].items():
        if (not isinstance(name, str) or not name or "\0" in name or PurePosixPath(name).is_absolute()
                or ".." in PurePosixPath(name).parts or str(PurePosixPath(name)) != name or name == "."):
            raise ValueError("operational asset names must be canonical relative file paths")
        deployment.sha256(digest)
    identity = {"manifest_sha256": record["sha256"], "source_commit": manifest["source_commit"],
                "file_count": len(manifest["files"]), "tree_digest": validation.canonical_digest(manifest["files"])}
    if identity != expected:
        raise ValueError("operational game assets differ from the globally and target-verified asset proof")
    physical_manifest = evidence.physical_file(manifest_path).resolve(strict=True)
    physical_root = Path(root).resolve(strict=False)
    if physical_manifest.is_relative_to(physical_root):
        raise ValueError("resolved operational manifest lies inside the explicit asset root")
    return identity, str(physical_manifest), str(physical_root)


def validate_deployed_training_admission(path, resolver=None, *, historical_read_only=False):
    """Rebuild the exact deployment receipt and admit its explicit operational source/configuration/seeds."""
    evidence = validation.Evidence(resolver)
    specification = evidence.read(path)
    required = {"schema_version", "unit_id", "source_root", "source_commit", "hardware_class", "configuration", "copy_seeds",
                "runtime_environment", "final_verification", "asset_record", "jaxtari_source_root",
                "deployment_verification", "deployment_verification_sha256"}
    if (not isinstance(specification, dict) or set(specification) != required
            or type(specification["schema_version"]) is not int or specification["schema_version"] != 1
            or specification["hardware_class"] != "h100"):
        raise ValueError("deployed training requires the exact version-one operational H100 specification")
    for key in ("source_root", "final_verification", "jaxtari_source_root", "deployment_verification"):
        explicit_path(specification[key])
    receipt_path = specification["deployment_verification"]
    contents = evidence.file(receipt_path)
    expected_hash = deployment.sha256(specification["deployment_verification_sha256"])
    if hashlib.sha256(contents).hexdigest() != expected_hash:
        raise ValueError("the deployment verification differs from its required SHA256")
    receipt = portable_evidence.document(contents)
    if (not isinstance(receipt, dict) or type(receipt.get("schema_version")) is not int or receipt["schema_version"] != 1
            or receipt.get("kind") != "training_deployment_verification" or "specification_path" not in receipt
            or receipt.get("specification_digest") != validation.canonical_digest(receipt["specification"])):
        raise ValueError("deployment verification needs its exact original specification and explicit specification_path")
    rebuilt = deployment.validate_deployment(receipt["specification"], resolver=resolver,
                                             specification_path=receipt["specification_path"],
                                             historical_read_only=historical_read_only)
    if rebuilt != receipt:
        raise ValueError("deployment raw evidence changed or no longer passes its recorded target/global gates")
    runtime = validation.runtime_environment(specification["runtime_environment"])
    configuration = validation.configuration(specification["configuration"])
    if (runtime != specification["runtime_environment"] or runtime != receipt["runtime_environment"]
            or validation.canonical_digest(configuration) != validation.canonical_digest(receipt["configuration"])
            or validation.canonical_digest(specification["copy_seeds"]) != validation.canonical_digest(receipt["copy_seeds"])
            or any(specification[key] != receipt[key] for key in ("source_root", "source_commit", "hardware_class", "unit_id"))):
        raise ValueError("operational training differs from the exact target source/configuration/seeds/runtime/unit")
    global_proof = receipt["global_final_verification"]
    if (evidence.path_identity(specification["final_verification"]) != global_proof["path"]
            or hashlib.sha256(evidence.file(specification["final_verification"])).hexdigest() != global_proof["sha256"]):
        raise ValueError("operational training names another global final-verification proof")
    physical_source = full_source_configuration(specification, evidence)
    public_root = deployment.git_root(specification["jaxtari_source_root"], deployment.JAXTARI_COMMIT, evidence)
    expected_python = receipt["backend"]["jaxtari_source"]
    observed_python = deployment.jaxtari_identity({**expected_python, "source_root": specification["jaxtari_source_root"]},
                                                 receipt["backend"]["packages"], evidence)
    if observed_python != expected_python:
        raise ValueError("operational JAXtari source differs from the target-verified Python tree")
    assets, manifest, asset_root = training_assets(specification["asset_record"], receipt["assets"], evidence)
    operational_paths = {"source_root": physical_source, "jaxtari_source_root": str(public_root.resolve(strict=True)),
        "training_specification": str(evidence.physical_file(path).resolve(strict=True)),
        "deployment_verification": str(evidence.physical_file(receipt_path).resolve(strict=True)),
        "final_verification": str(evidence.physical_file(specification["final_verification"]).resolve(strict=True)),
        "asset_manifest": manifest, "asset_root": asset_root}
    admitted = {**specification, "configuration": configuration, "runtime_environment": runtime,
        "operational_paths": operational_paths, "evidence_resolution": resolver.receipt() if resolver is not None else None,
        "validation": {"schema_version": 1, "kind": "deployed_training_admission", "source_commit": specification["source_commit"],
            "specification_path": evidence.path_identity(path), "specification_digest": validation.canonical_digest(specification),
            "configuration_digest": validation.canonical_digest(configuration), "runtime_environment": runtime,
            "unit_id": specification["unit_id"], "copy_seeds": specification["copy_seeds"], "selected_copies": configuration["learner"]["n_copies"],
            "deployment_verification": {"path": evidence.path_identity(receipt_path), "sha256": expected_hash},
            "deployment_digest": validation.canonical_digest(receipt), "global_final_verification": global_proof,
            "assets": assets, "jaxtari_source": {"root": specification["jaxtari_source_root"], **expected_python},
            "admission_records": evidence.records(), "evidence": receipt["evidence"]}}
    return training.validate_specification(admitted)


def main():
    """Print an explicitly requested compute-side admission without launching a numerical runtime."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--specification", type=Path, required=True)
    parser.add_argument("--evidence-manifest", type=Path)
    parser.add_argument("--evidence-manifest-sha256")
    args = parser.parse_args()
    if args.evidence_manifest_sha256 and args.evidence_manifest is None:
        parser.error("an expected manifest hash requires --evidence-manifest")
    resolver = portable_evidence.ExactPathResolver(args.evidence_manifest, args.evidence_manifest_sha256) if args.evidence_manifest else None
    print(json.dumps(validate_deployed_training_admission(args.specification, resolver), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
