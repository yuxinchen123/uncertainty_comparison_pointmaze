"""Build and verify explicit transport packages using synthetic bytes and local Git only."""

import copy
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch


SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS))
import prepare_portable_evidence as builder
import portable_evidence as portable


class PortablePackageTests(unittest.TestCase):
    """Test actual byte copying and real detached Git roots without remote or numerical execution."""

    def setUp(self):
        """Create separate originals and an explicit existing package directory."""
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name).resolve()
        self.original = self.directory / "original"
        self.staging = self.directory / "package"
        self.original.mkdir()
        self.staging.mkdir()
        self.input_path = self.directory / "explicit_inputs.json"
        self.inputs = {"schema_version": 1, "kind": "portable_evidence_input_list", "files": [], "git_roots": []}

    def add_file(self, name, contents):
        """List one exact original path and expected digest without rewriting its bytes."""
        path = self.original / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(contents)
        self.inputs["files"].append({"original_path": str(path), "sha256": hashlib.sha256(contents).hexdigest()})
        return path

    def save_inputs(self):
        """Retain deliberate original input-list whitespace as transport evidence."""
        self.input_path.write_text(json.dumps(self.inputs, separators=(",", ":")) + "\n\n")

    def prepare(self):
        """Run the actual local builder; no test replaces its copy or resolver checks."""
        self.save_inputs()
        return builder.prepare(self.input_path, self.staging)

    def git_fixture(self):
        """Pre-stage a real independent checkout whose object store is inside the package."""
        source = self.original / "source"
        source.mkdir()
        (source / "learner.py").write_text("configuration = {'precision': 'float32'}\n")
        subprocess.run(["git", "init", "-q", str(source)], check=True)
        subprocess.run(["git", "-C", str(source), "add", "learner.py"], check=True)
        subprocess.run(["git", "-C", str(source), "-c", "user.name=Synthetic transport test", "-c", "user.email=synthetic@example.invalid",
            "-c", "core.hooksPath=/dev/null", "commit", "-qm", "Create explicit source fixture"], check=True)
        commit = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
        staged = self.staging / "repositories/source"
        subprocess.run(["git", "clone", "-q", "--no-hardlinks", str(source), str(staged)], check=True)
        subprocess.run(["git", "-C", str(staged), "checkout", "-q", "--detach", commit], check=True)
        entry = {"original_root": str(source), "relative_path": "repositories/source", "commit": commit}
        self.inputs["git_roots"].append(entry)
        return source, staged, commit

    def test_package_survives_original_removal_and_retains_raw_json_identities(self):
        """The resolver reads only exact staged bytes after all original files and repositories are removed."""
        source, staged, commit = self.git_fixture()
        original_bytes = ('{"source_root":' + json.dumps(str(source)) + ',"seed":17}\n\n').encode()
        record = self.add_file("raw/result.json", original_bytes)
        duplicate = self.add_file("copied/result.json", original_bytes)
        result = self.prepare()
        manifest_bytes = (self.staging / builder.MANIFEST).read_bytes()
        input_bytes = self.input_path.read_bytes()
        self.assertEqual((self.staging / builder.INPUT_COPY).read_bytes(), input_bytes)
        self.assertEqual(result["mapped_files"], 2)
        self.assertEqual(result["unique_staged_files"], 1)
        resolver = portable.ExactPathResolver(self.staging / builder.MANIFEST, result["manifest_sha256"])
        self.assertEqual(resolver.file_path(record), resolver.file_path(duplicate))
        shutil.rmtree(self.original)
        self.input_path.unlink()
        finalized = builder.verify(self.staging, result["manifest_sha256"], result["transport_receipt_sha256"])
        self.assertTrue(finalized["verified"])
        self.assertEqual(resolver.read_bytes(record), original_bytes)
        self.assertEqual(resolver.git_root(source, commit), staged)
        self.assertEqual((self.staging / builder.MANIFEST).read_bytes(), manifest_bytes)
        self.assertNotIn("jax", sys.modules)
        self.assertNotIn("numpy", sys.modules)

    def test_large_binary_sources_are_streamed_without_read_bytes(self):
        """GPU-image-sized inputs use bounded reads rather than materializing source files."""
        image = self.add_file("libcudnn.so", b"GPU image bytes\0" * (2 ** 18))
        original_read = Path.read_bytes
        def reject_large_read(path):
            """Permit small metadata reads while rejecting whole-image reads at original or staged paths."""
            if path == image or path.parent.name == "files":
                raise AssertionError("large file was read in one allocation")
            return original_read(path)
        with patch.object(Path, "read_bytes", reject_large_read):
            result = self.prepare()
        self.assertTrue(result["verified"])
        self.assertGreater(result["mapped_bytes"], builder.BLOCK_BYTES)

    def test_duplicate_and_noncanonical_original_paths_are_rejected(self):
        """No duplicate identity, path normalization, duplicate JSON key or implicit mapping is accepted."""
        self.add_file("result.json", b"{}")
        original = copy.deepcopy(self.inputs)
        for value in ("relative/result.json", "/tmp/../result.json", "//tmp/result.json", "/tmp/./result.json"):
            self.inputs = copy.deepcopy(original)
            self.inputs["files"][0]["original_path"] = value
            self.save_inputs()
            with self.assertRaisesRegex(ValueError, "canonical absolute"):
                builder.prepare(self.input_path, self.staging)
        self.inputs = copy.deepcopy(original)
        self.inputs["files"].append(dict(self.inputs["files"][0]))
        self.save_inputs()
        with self.assertRaisesRegex(ValueError, "duplicated"):
            builder.prepare(self.input_path, self.staging)
        with self.assertRaisesRegex(ValueError, "duplicate JSON key"):
            builder.input_document(b'{"schema_version":1,"files":[],"files":[],"git_roots":[]}')

    def test_changed_original_or_existing_staged_bytes_are_never_published_as_expected(self):
        """A bad source or previously staged hash object cannot be overwritten into apparent validity."""
        path = self.add_file("original.bin", b"original")
        path.write_bytes(b"changed")
        self.save_inputs()
        with self.assertRaisesRegex(ValueError, "expected SHA256"):
            builder.prepare(self.input_path, self.staging)
        self.assertFalse((self.staging / builder.MANIFEST).exists())
        path.write_bytes(b"original")
        staged = self.staging / "files" / self.inputs["files"][0]["sha256"]
        staged.write_bytes(b"unrelated existing file")
        with self.assertRaisesRegex(ValueError, "different bytes"):
            builder.prepare(self.input_path, self.staging)
        self.assertEqual(staged.read_bytes(), b"unrelated existing file")

    def test_changed_copied_inputs_receipt_or_manifest_are_rejected_on_finalize(self):
        """Remote finalize requires both exact externally retained hashes and the unchanged input-list bytes."""
        self.add_file("record.json", b"{}\n")
        result = self.prepare()
        for relative, message in ((builder.INPUT_COPY, "input list changed"), (builder.MANIFEST, "manifest differs"),
                                  (builder.RECEIPT, "receipt differs")):
            path = self.staging / relative
            original = path.read_bytes()
            path.write_bytes(original + b" ")
            with self.assertRaisesRegex(ValueError, message):
                builder.verify(self.staging, result["manifest_sha256"], result["transport_receipt_sha256"])
            path.write_bytes(original)

    def test_target_symlink_escape_is_rejected_without_writing_outside(self):
        """A pre-existing files-directory symlink cannot redirect any evidence copy."""
        self.add_file("record.json", b"{}")
        outside = self.directory / "outside"
        outside.mkdir()
        (self.staging / "files").symlink_to(outside, target_is_directory=True)
        self.save_inputs()
        with self.assertRaisesRegex(ValueError, "symlink"):
            builder.prepare(self.input_path, self.staging)
        self.assertEqual(list(outside.iterdir()), [])

    def test_git_impostor_dirty_branch_and_wrong_commit_are_rejected(self):
        """A nested archive and changed Git metadata cannot inherit another repository's identity."""
        source, staged, commit = self.git_fixture()
        self.save_inputs()
        (staged / "learner.py").write_text("changed=True\n")
        with self.assertRaisesRegex(ValueError, "clean detached root"):
            builder.prepare(self.input_path, self.staging)
        subprocess.run(["git", "-C", str(staged), "checkout", "--", "learner.py"], check=True)
        self.inputs["git_roots"][0]["commit"] = "0" * 40
        self.save_inputs()
        with self.assertRaisesRegex(ValueError, "clean detached root"):
            builder.prepare(self.input_path, self.staging)
        self.inputs["git_roots"][0]["commit"] = commit
        self.save_inputs()
        subprocess.run(["git", "-C", str(staged), "checkout", "-qb", "named_branch"], check=True)
        with self.assertRaisesRegex(ValueError, "clean detached root"):
            builder.prepare(self.input_path, self.staging)
        subprocess.run(["git", "-C", str(staged), "checkout", "-q", "--detach", commit], check=True)
        nested = staged / "ignored_archive"
        nested.mkdir()
        self.inputs["git_roots"][0]["relative_path"] += "/ignored_archive"
        self.save_inputs()
        with self.assertRaisesRegex(ValueError, "own Git metadata"):
            builder.prepare(self.input_path, self.staging)

    def test_external_git_worktree_metadata_is_rejected(self):
        """A linked checkout cannot depend on its original external repository after transport."""
        source, staged, commit = self.git_fixture()
        shutil.rmtree(staged)
        subprocess.run(["git", "-C", str(source), "worktree", "add", "-q", "--detach", str(staged), commit], check=True)
        self.save_inputs()
        with self.assertRaisesRegex(ValueError, "outside the package"):
            builder.prepare(self.input_path, self.staging)

    def test_external_git_object_alternates_are_rejected(self):
        """A shared clone cannot silently rely on an outside object directory."""
        source, staged, commit = self.git_fixture()
        shutil.rmtree(staged)
        subprocess.run(["git", "clone", "-q", "--shared", str(source), str(staged)], check=True)
        subprocess.run(["git", "-C", str(staged), "checkout", "-q", "--detach", commit], check=True)
        self.save_inputs()
        with self.assertRaisesRegex(ValueError, "external alternates"):
            builder.prepare(self.input_path, self.staging)

    def test_external_object_or_pack_directory_symlinks_are_rejected(self):
        """Accessible outside objects cannot make a package appear independently usable."""
        source, staged, commit = self.git_fixture()
        subprocess.run(["git", "-C", str(staged), "repack", "-adq"], check=True)
        self.save_inputs()
        for relative in ("objects", "objects/pack"):
            with self.subTest(metadata_path=relative):
                target = staged / ".git" / relative
                outside = self.directory / ("outside-" + relative.replace("/", "-"))
                shutil.move(target, outside)
                target.symlink_to(outside, target_is_directory=True)
                try:
                    # Git can still read the pinned commit, which is insufficient proof of portability.
                    actual = subprocess.check_output(["git", "-C", str(staged), "rev-list", "--objects", "--no-object-names", commit], text=True).splitlines()
                    self.assertIn(commit, actual)
                    with self.assertRaisesRegex(ValueError, "metadata cannot contain a symlink"):
                        builder.prepare(self.input_path, self.staging)
                finally:
                    target.unlink()
                    shutil.move(outside, target)

    def test_tracked_source_symlinks_are_separate_from_git_metadata(self):
        """A legitimate tracked relative source alias does not depend on outside Git object storage."""
        source, staged, commit = self.git_fixture()
        (staged / "learner_alias.py").symlink_to("learner.py")
        subprocess.run(["git", "-C", str(staged), "add", "learner_alias.py"], check=True)
        subprocess.run(["git", "-C", str(staged), "-c", "user.name=Synthetic transport test", "-c", "user.email=synthetic@example.invalid",
            "-c", "core.hooksPath=/dev/null", "commit", "-qm", "Add tracked source alias"], check=True)
        updated = subprocess.check_output(["git", "-C", str(staged), "rev-parse", "HEAD"], text=True).strip()
        self.inputs["git_roots"][0]["commit"] = updated
        result = self.prepare()
        self.assertTrue(result["verified"])
        self.assertTrue((staged / "learner_alias.py").is_symlink())

    def test_git_targets_duplicates_and_symlink_aliases_are_rejected(self):
        """Git mappings remain exact and cannot overlap package transport paths or escape by symlink."""
        source, staged, commit = self.git_fixture()
        original = copy.deepcopy(self.inputs)
        for relative in ("../outside", "/absolute", "repositories/./source", "files/source", "transport/source"):
            self.inputs = copy.deepcopy(original)
            self.inputs["git_roots"][0]["relative_path"] = relative
            self.save_inputs()
            with self.assertRaises(ValueError):
                builder.prepare(self.input_path, self.staging)
        self.inputs = copy.deepcopy(original)
        self.inputs["git_roots"].append(dict(self.inputs["git_roots"][0]))
        self.save_inputs()
        with self.assertRaisesRegex(ValueError, "duplicated"):
            builder.prepare(self.input_path, self.staging)
        self.inputs = copy.deepcopy(original)
        (self.staging / "alias").symlink_to(staged, target_is_directory=True)
        self.inputs["git_roots"][0]["relative_path"] = "alias"
        self.save_inputs()
        with self.assertRaisesRegex(ValueError, "symlink"):
            builder.prepare(self.input_path, self.staging)

    def test_preparation_is_idempotent_but_never_replaces_a_different_manifest(self):
        """A second exact preparation keeps all published bytes; a different list requires another package."""
        self.add_file("first.json", b"{}")
        first = self.prepare()
        before = {relative: (self.staging / relative).read_bytes() for relative in (builder.INPUT_COPY, builder.MANIFEST, builder.RECEIPT)}
        second = builder.prepare(self.input_path, self.staging)
        self.assertEqual(first, second)
        self.add_file("second.json", b"[]")
        self.save_inputs()
        with self.assertRaisesRegex(ValueError, "earlier portable transport record differs"):
            builder.prepare(self.input_path, self.staging)
        self.assertEqual({relative: (self.staging / relative).read_bytes() for relative in before}, before)

    def test_alias_loaded_builder_uses_its_adjacent_original_resolver(self):
        """A stale import or different working directory cannot substitute another path resolver."""
        self.add_file("result.json", b'{"retained":"original path"}\n')
        self.save_inputs()
        code = """
import importlib.util,sys,types
from pathlib import Path
wrong=types.ModuleType('portable_evidence')
def reject(*args,**kwargs): raise AssertionError('used an unrelated resolver')
wrong.document=reject
sys.modules['portable_evidence']=wrong
source=Path(sys.argv[1])
assert str(source.parent) not in sys.path
spec=importlib.util.spec_from_file_location('alias_builder',source)
builder=importlib.util.module_from_spec(spec);spec.loader.exec_module(builder)
result=builder.prepare(Path(sys.argv[2]),Path(sys.argv[3]))
assert result['verified']
assert sys.modules['portable_evidence'] is wrong
assert 'jax' not in sys.modules and 'numpy' not in sys.modules
print('adjacent resolver verified')
"""
        result = subprocess.run([sys.executable, "-I", "-c", code, str(SCRIPTS / "prepare_portable_evidence.py"),
            str(self.input_path), str(self.staging)], capture_output=True, text=True, check=True, cwd="/")
        self.assertEqual(result.stdout.strip(), "adjacent resolver verified")


if __name__ == "__main__":
    unittest.main()
