"""Keep metadata scope checks separate from frozen numerical imports."""
import ast
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

SCRIPTS=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(SCRIPTS))
import serval02_card1_binding as core

EXPECTED=("10_jax_exploration_platform/src/exploration_platform/atari_pixels",
          "10_jax_exploration_platform/scripts/atari_pixel_campaign")


class MetadataImportProvenanceTests(unittest.TestCase):
    """Read only a literal original scope and preserve exact public numerical-module ownership."""

    def read_fixture(self,source):
        """Place a synthetic controller beside a synthetic exact metadata-module path."""
        with tempfile.TemporaryDirectory() as temporary:
            directory=Path(temporary);(directory/'controller.py').write_text(source)
            with patch.object(core,'__file__',str(directory/'serval02_card1_binding.py')):
                return core.original_source_scopes()

    def test_exact_original_scope_does_not_execute_controller(self):
        """A dangerous top-level marker is not executed when its literal scope is read."""
        self.assertEqual(self.read_fixture('raise RuntimeError("must not execute")\nSOURCE_SCOPES = '+repr(EXPECTED)+'\n'),EXPECTED)
        self.assertEqual(core.original_source_scopes(),EXPECTED)

    def test_missing_duplicate_dynamic_nested_and_expanded_scopes_refuse(self):
        """No dynamic declaration or second assignment can replace the original two paths."""
        sources=['','SOURCE_SCOPES = list('+repr(EXPECTED)+')','SOURCE_SCOPES = '+repr(EXPECTED)+'\nSOURCE_SCOPES = '+repr(EXPECTED),
            'SOURCE_SCOPES: tuple = '+repr(EXPECTED),'SOURCE_SCOPES = [*'+repr(EXPECTED)+']',
            'if True:\n    SOURCE_SCOPES = '+repr(EXPECTED),'def inner():\n    SOURCE_SCOPES = '+repr(EXPECTED),
            'SOURCE_SCOPES = '+repr((*EXPECTED,'other/source')),'SOURCE_SCOPES = '+repr(tuple(reversed(EXPECTED))),
            'SOURCE_SCOPES = OTHER = '+repr(EXPECTED)]
        for source in sources:
            with self.subTest(source=source),self.assertRaises((ValueError,SyntaxError)):self.read_fixture(source)

    def test_controller_alias_refuses_without_following_target(self):
        """The metadata declaration stays in the exact adjacent ordinary file."""
        with tempfile.TemporaryDirectory() as temporary:
            directory=Path(temporary);other=directory/'elsewhere.py';other.write_text('SOURCE_SCOPES = '+repr(EXPECTED))
            (directory/'controller.py').symlink_to(other)
            with patch.object(core,'__file__',str(directory/'serval02_card1_binding.py')),self.assertRaises(ValueError):core.original_source_scopes()

    def test_existing_public_modules_are_preserved(self):
        """Reading metadata never replaces or deletes an already loaded public helper."""
        ideas,controller=object(),object()
        with patch.dict(sys.modules,{"ideas":ideas,"controller":controller}):
            self.assertEqual(core.original_source_scopes(),EXPECTED)
            self.assertIs(sys.modules["ideas"],ideas)
            self.assertIs(sys.modules["controller"],controller)

    def test_live_check_uses_no_controller_import_or_module_deletion(self):
        """The production correction only changes the metadata constant lookup."""
        tree=ast.parse(Path(core.__file__).read_text());function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='check_live_admission')
        calls=[n for n in ast.walk(function) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name)]
        self.assertIn('original_source_scopes',[n.func.id for n in calls])
        self.assertFalse(any(n.func.id=='adjacent' for n in calls))
        self.assertFalse(any(isinstance(n,ast.Delete) for n in ast.walk(function)))


if __name__=='__main__':unittest.main()
