"""Load the grid campaign's general functions, and this folder's own files, under names that cannot clash.

This folder has its own worker.py, submit.py, monitor.py and remote_fleet.py, and so has
`scripts/grid_initialization_campaign/`. The grid campaign's files are therefore loaded as
grid_worker, grid_submit and grid_remote_fleet (`load`), and this folder's files as maze_<name>
(`own`), so one process, or one test session, can hold both sets.

The grid campaign's common.py is imported by its own name: this folder has no common.py, and the
grid campaign's files import it by that name. It holds the atomic JSON records, the queue lock,
the Pacific time display and the read of this run's live job ids. Standard library only.
"""
import importlib.util
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
SCRIPTS = HERE.parent
GRID = SCRIPTS / "grid_initialization_campaign"

# the grid campaign's files import each other by bare name, and experiment_configuration.py sits
# one folder up; both folders go to the END of the path, behind whatever the caller put first
for folder in (GRID, SCRIPTS):
    if str(folder) not in sys.path:
        sys.path.append(str(folder))

import common  # noqa: E402  (scripts/grid_initialization_campaign/common.py)


def load_file(module_name, path):
    """Load one Python file under a given module name, once per process."""
    if module_name not in sys.modules:
        specification = importlib.util.spec_from_file_location(module_name, path)
        module = importlib.util.module_from_spec(specification)
        # registered before it runs, as the import system does, so a failed load leaves no half module
        sys.modules[module_name] = module
        try:
            specification.loader.exec_module(module)
        except BaseException:
            del sys.modules[module_name]
            raise
    return sys.modules[module_name]


def load(name):
    """Load one file of the grid campaign as the module grid_<name>."""
    return load_file("grid_" + name, GRID / f"{name}.py")


def own(name):
    """Load one file of this folder as the module maze_<name>."""
    return load_file("maze_" + name, HERE / f"{name}.py")
