"""Make the maze-campaign analysis package importable by the tests, and let a run keep the made-up study."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))


def pytest_addoption(parser):
    """An optional folder on shared storage in which the made-up study's run folder and report are left."""
    parser.addoption('--example-study-dir', default=None,
                     help='write the made-up study of test_maze_report.py here instead of a temporary folder')
