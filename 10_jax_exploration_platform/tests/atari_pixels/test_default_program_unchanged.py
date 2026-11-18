"""With every new option off, the pixel trainer and environment trace to the program of the reference commit.

Run on a compute node (two subprocesses trace the programs with JAX on the processor). The
reference is the code of commit REFERENCE_COMMIT, the last one before the fast last layer and the
game events were added; that code produced the archived baseline. The comparison is of traced
programs (every operation, shape and type, in order), so it holds on every device; the numbers on
a card are compared with the archive by scripts/atari_pixel_campaign/compare_metric_prefix.py.
"""

import json
import os
from pathlib import Path
import subprocess
import sys

PLATFORM = Path(__file__).resolve().parents[2]
HELPER = Path(__file__).resolve().with_name("program_fingerprints.py")
REFERENCE_COMMIT = "19d501528d1e11161f7ec8f0d71aa0efed4ea1fb"
REFERENCE_PATHS = ("10_jax_exploration_platform/src/exploration_platform/__init__.py",
                   "10_jax_exploration_platform/src/exploration_platform/atari_pixels")


def reference_source(folder):
    """Unpack the reference commit's trainer package into a folder and return its src path."""
    repository = subprocess.run(["git", "-C", str(PLATFORM), "rev-parse", "--show-toplevel"],
                                check=True, capture_output=True, text=True).stdout.strip()
    archive = subprocess.run(["git", "-C", repository, "archive", REFERENCE_COMMIT, "--", *REFERENCE_PATHS],
                             check=True, stdout=subprocess.PIPE).stdout
    subprocess.run(["tar", "-x", "-C", str(folder)], input=archive, check=True)
    return folder / "10_jax_exploration_platform" / "src"


def fingerprints(source):
    """Run the tracing helper against one source tree and return its fingerprints."""
    environment = {**os.environ, "PYTHONPATH": str(source), "JAX_PLATFORMS": "cpu", "PYTHONDONTWRITEBYTECODE": "1"}
    completed = subprocess.run([sys.executable, str(HELPER)], env=environment, capture_output=True, text=True)
    if completed.returncode:
        raise RuntimeError(f"tracing with {source} failed:\n{completed.stderr[-4000:]}")
    return json.loads(completed.stdout.strip().splitlines()[-1])


def test_default_options_trace_to_the_reference_program(tmp_path):
    """Every traced function has the same program text as at the reference commit."""
    expected = fingerprints(reference_source(tmp_path))
    actual = fingerprints(PLATFORM / "src")
    assert set(actual) == set(expected) and len(expected) == 8
    for name, record in expected.items():
        assert record["characters"] > 1000, name
        assert actual[name] == record, f"{name}: the traced program differs from the reference commit"
